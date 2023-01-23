# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import configparser
import dataclasses
import json
import pathlib
import time
from typing import Dict, List

import torch
import tqdm
import transformers

from utils import bloom


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)


class LambadaDataset(torch.utils.data.Dataset):
    """ LAMBADA dataset class. """

    def __init__(self,
                 path: str | pathlib.Path,
                 tokenizer: transformers.PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        with open(path, 'r') as f:
            inputs, targets = zip(*[
                json.loads(line)["text"] .strip('\n').rsplit(' ', 1)
                for line in f.readlines()])
            # This whitespace preprocessing (additional space to the target)
            # is required.
            targets = [' ' + tgt for tgt in targets]
            self.encodings = self.tokenizer(list(inputs),
                                            targets,
                                            padding=True,
                                            return_token_type_ids=True,
                                            return_tensors='pt')

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return dict(
            input_ids=self.encodings['input_ids'][idx],
            attention_mask=self.encodings['attention_mask'][idx],
            token_type_ids=self.encodings['token_type_ids'][idx]
        )


@dataclasses.dataclass
class Metric:
    acc: float


@dataclasses.dataclass
class RequestAndResult:
    prompt: str
    model_answer: str
    target: str
    input_ids: List[int]
    input_len: int
    output_len: int
    model_params: bloom.BloomParam
    infer_params: bloom.BloomInferParam
    output_ids: List[int]
    metrics: Metric

    def asdict(self):
        return dataclasses.asdict(self)


class Timer:

    def __init__(self):
        self._start_times = {}
        self._total_elapsed_times = {}

    def start(self, tag='__default'):
        self._start_times[tag] = time.time()

    def stop(self, tag='__default'):
        elapsed_time = time.time() - self._start_times[tag]
        if tag not in self._total_elapsed_times:
            self._total_elapsed_times[tag] = 0
        self._total_elapsed_times[tag] += elapsed_time
        return elapsed_time

    def elapsed_time_in_sec(self, tag='__default'):
        if tag not in self._total_elapsed_times:
            return None
        return self._total_elapsed_times[tag]

    def reset(self):
        self._start_times.clear()
        self._total_elapsed_times.clear()


def get_args():
    parser = argparse.ArgumentParser(
        'Evaluation: LAMBADA Task',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    bloom.BloomParam.add_args_group(parser)
    bloom.BloomInferParam.add_args_group(parser)

    group = parser.add_argument_group('LAMBADA Task Parameters')
    group.add_argument(
        '--checkpoint-path', type=str, metavar='DIR', default=None,
        help='A directory of a converted pretrained checkpoint and model config '
             'If None, a model will inference by random weights.')
    group.add_argument(
        '--dataset-path', type=str, metavar='PATH', required=True,
        help="A file path to LAMBADA task dataset.")
    group.add_argument(
        '--output-path', type=str, metavar='PATH', default=None,
        help="Path to sample output file.")
    group.add_argument(
        "--tokenizer-path", type=str, metavar='DIR_OR_PATH', default=None,
        help='A file path of a pretrained tokenizer or a checkpoint directory '
             'of HF pretrained model.')
    group.add_argument(
        '--lib-path', type=str, metavar='PATH', default='./lib/libth_transformer.so',
        help='A FT library path to load `FasterTransformer.ParallelGptOp`')
    group.add_argument(
        '--test-hf', action='store_true',
        help='Run a huggingface model instead of an FT model. The checkpoint '
             'of the huggingface model is assumed to be at --tokenizer-path.')
    group.add_argument(
        '--acc-threshold', type=float, metavar='M', default=None,
        help='The minimum value of the expected accuracy of the LAMBADA '
             'evaluation for a test. If the achieved accuracy is less '
             'than given value, a value error will occurs.')
    group.add_argument(
        '--show-progress', action='store_true',
        help='Show evaluation progress')
    group.add_argument(
        '--inference-data-type', '--data-type', type=str, metavar='TYPE', default=None,
        choices=[None, 'fp32', 'fp16', 'bf16'],
        help='The data type to inference. If None, the data type follows the '
             'checkpoint data type.')
    group.add_argument(
        '--weights-data-type', type=str, metavar='TYPE', default=None,
        choices=[None, 'fp32', 'fp16'],
        help='The data type of FT checkpoint. If None, it will be retrieved '
             'from the config file in the checkpoint directory.')
    group.add_argument(
        '--int8_mode', type=int, default=0, choices=[0, 1],
        help='The level of quantization to perform.'
             ' 0: No quantization. All computation in data_type'
             ' 1: Quantize weights to int8, all compute occurs in fp16/bf16. Not supported when data_type is fp32')
    args = parser.parse_args()

    print('\n=================== Arguments ===================')
    for k, v in vars(args).items():
        print(f' - {k.ljust(25, ".")}: {v}')
    print('=================================================')

    return args


def get_model_and_tokenizer(args: argparse.Namespace):
    tokenizer_path = pathlib.Path(args.tokenizer_path)
    # HF requires left padding for a decoder-only model.
    padding_side = 'left' if args.test_hf else 'right'
    if tokenizer_path.is_dir():
        # Load from the HF's pretrained model directory.
        tokenizer = transformers.BloomTokenizerFast.from_pretrained(
            args.tokenizer_path, padding_side=padding_side)
    else:
        # Directly load from a tokenizer json file.
        tokenizer = transformers.BloomTokenizerFast(
            tokenizer_file=tokenizer_path, padding_side=padding_side)
    # For open-ended generation, the pad token is sometimes replaced by the
    # eos token but the Bloom of HF requires as it is to correctly generate.

    if args.test_hf:
        # Load HF's pretrained model for testing.
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.tokenizer_path).cuda()
        return model, tokenizer

    checkpoint_path = pathlib.Path(args.checkpoint_path)
    config_path = checkpoint_path / 'config.ini'

    if config_path.exists():
        # Read model params from config.
        cfg = configparser.ConfigParser()
        cfg.read(config_path)
        model_name = 'gpt'
        inference_data_type = args.inference_data_type
        if inference_data_type == None:
            inference_data_type = cfg.get(model_name, "weight_data_type")
        model_args = dict(
            head_num=cfg.getint(model_name, 'head_num'),
            size_per_head=cfg.getint(model_name, "size_per_head"),
            layer_num=cfg.getint(model_name, "num_layer"),
            tensor_para_size=cfg.getint(model_name, "tensor_para_size"),
            vocab_size=cfg.getint(model_name, "vocab_size"),
            start_id=cfg.getint(model_name, "start_id"),
            end_id=cfg.getint(model_name, "end_id"),
            weights_data_type=cfg.get(model_name, "weight_data_type"),
            layernorm_eps=cfg.getfloat(model_name, 'layernorm_eps'),
            inference_data_type=inference_data_type)
    else:
        inference_data_type = args.inference_data_type
        if inference_data_type == None:
            inference_data_type = args.weights_data_type
        model_args = dict(head_num=args.num_heads,
                          size_per_head=args.size_per_head,
                          vocab_size=args.vocab_size,
                          start_id=args.start_id or tokenizer.bos_token_id,
                          end_id=args.end_id or tokenizer.eos_token_id,
                          layer_num=args.num_layers,
                          tensor_para_size=args.tensor_para_size,
                          weights_data_type=args.weights_data_type,
                          inference_data_type=inference_data_type)

    # update common parameters
    model_args.update(dict(
        lib_path=args.lib_path,
        pipeline_para_size=args.pipeline_para_size,
        shared_contexts_ratio=args.shared_contexts_ratio,
        int8_mode=args.int8_mode
    ))

    print('[FT][INFO] Load BLOOM model')
    for k, v in model_args.items():
        print(f' - {k.ljust(25, ".")}: {v}')

    # Check sanity and consistency between the model and tokenizer.
    checklist = ['head_num', 'size_per_head', 'vocab_size', 'layer_num',
                 'tensor_para_size', 'tensor_para_size', 'weights_data_type']
    if None in [model_args[k] for k in checklist]:
        none_params = [p for p in checklist if model_args[p] is None]
        print(f'[FT][WARNING] Found None parameters {none_params}. They must '
              f'be provided either by config file or CLI arguments.')
    if model_args['start_id'] != tokenizer.bos_token_id:
        print('[FT][WARNING] Given start_id is not matched with the bos token '
              'id of the pretrained tokenizer.')
    if model_args['end_id'] not in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        print('[FT][WARNING] Given end_id is not matched with neither pad '
              'token id nor eos token id of the pretrained tokenizer.')
    model = bloom.Bloom(**model_args)
    if not model.load(ckpt_path=args.checkpoint_path):
        print('[FT][WARNING] Skip model loading since no checkpoints are found')

    return model, tokenizer


def split_inputs_and_targets(entries: Dict[str, torch.LongTensor],
                             pad_token_id: int,
                             pad_to_left=False):
    input_ids = entries['input_ids']
    attn_mask = entries['attention_mask']
    token_type_ids = entries['token_type_ids']

    # Split inputs and labels by token_type_ids.
    input_token_ids = [
        ids[(mask == 1) & (type_ids == 0)]
        for ids, mask, type_ids in zip(input_ids, attn_mask, token_type_ids)]
    # FT allows int32 tensors.
    input_lengths = torch.tensor(
        [len(input_tokens) for input_tokens in input_token_ids]).int()
    max_length = input_lengths.max()
    input_token_ids = torch.stack([
        torch.nn.functional.pad(
            token_ids,
            pad=[max_length - len(token_ids), 0]
                if pad_to_left else [0, max_length - len(token_ids)],
            mode='constant',
            value=pad_token_id
        ) for token_ids in input_token_ids]).int()
    target_token_ids = [
        ids[(mask == 1) & (type_ids == 1)]
        for ids, mask, type_ids in zip(input_ids, attn_mask, token_type_ids)]
    return input_token_ids, input_lengths, target_token_ids


@torch.no_grad()
def main():
    args = get_args()
    model, tokenizer = get_model_and_tokenizer(args)
    model.eval()

    dataset = LambadaDataset(args.dataset_path, tokenizer=tokenizer)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    num_requests = 0
    num_corrects = 0
    results = {"output": {"lambada": []}, "results": {"lambada": {}}}

    timer = Timer()
    if args.show_progress:
        data_loader = tqdm.tqdm(data_loader)

    for entries in data_loader:
        input_token_ids, input_lengths, target_token_ids = \
            split_inputs_and_targets(entries, tokenizer.pad_token_id, args.test_hf)

        batch_size = input_token_ids.shape[0]
        output_length = max([len(target) for target in target_token_ids])

        params = bloom.BloomInferParam.from_args(args, batch_size)

        if args.test_hf:
            # Outputs (batch_size, seq_length)
            timer.start()
            outputs = model.generate(inputs=input_token_ids.cuda(),
                                     max_new_tokens=output_length,
                                     num_beams=args.beam_width,
                                     temperature=args.temperature,
                                     top_k=args.top_k,
                                     top_p=args.top_p,
                                     repetition_penalty=args.repetition_penalty,
                                     length_penalty=args.len_penalty)
            timer.stop()
            # output_token_ids: input/padding/output
            output_token_ids = outputs[:, input_token_ids.shape[1]:]
            output_token_ids = [
                out[:len(tgt)].cpu()
                for out, tgt in zip(output_token_ids, target_token_ids)]
        else:
            param_dict = params.asdict()
            timer.start()
            outputs = model(start_ids=input_token_ids,
                            start_lengths=input_lengths,
                            output_len=output_length,
                            **param_dict)
            timer.stop()

            if params.return_cum_log_probs or params.return_cum_log_probs > 0:
                outputs = outputs[0]  # output_token_ids.

            # Slice the generated token ids of the 1st beam result.
            # output = input tokens + generated tokens.
            output_token_ids = [
                out[0, length:length+len(tgt)].cpu()
                for out, length, tgt
                in zip(outputs, input_lengths, target_token_ids)]

        output_texts = tokenizer.batch_decode(output_token_ids)
        target_texts = tokenizer.batch_decode(target_token_ids)
        input_texts = tokenizer.batch_decode(input_token_ids)

        # Convert to output objects.
        for i in range(batch_size):
            out = output_token_ids[i]
            tgt = target_token_ids[i].cpu()
            is_correct = (tgt == out).all()
            num_corrects += int(is_correct)
            result = RequestAndResult(
                prompt=input_texts[i],
                model_answer=output_texts[i],
                target=target_texts[i],
                input_ids=input_token_ids[i].tolist(),
                input_len=input_lengths[i].item(),
                output_len=output_length,
                model_params=bloom.BloomParam.from_args(args),
                infer_params=params.slice_args(i),
                output_ids=out,
                metrics=Metric(acc=float(is_correct))
            )
            results['output']['lambada'].append(result.asdict())

        num_requests += batch_size

    accuracy = num_corrects * 100 / num_requests
    # Reference: HF model's LAMBADA Accuracy for bloom-560m ~ 35.36%
    print(f'Accuracy: {accuracy:0.4f}% ({num_corrects}/{num_requests}) '
          f'(elapsed time: {timer.elapsed_time_in_sec():.4f} sec)')
    # Dump prediction json
    results['results']['lambada']['acc'] = accuracy
    if args.output_path:
        output_path = pathlib.Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open(mode='w') as f:
            json.dump(results, f, indent=2, cls=TensorEncoder)

    if args.acc_threshold is not None:
        assert accuracy >= args.acc_threshold, \
            f'TEST FAIL the achieved accuracy ({accuracy:.2f}) is less ' \
            f'than given threshold ({args.acc_threshold:.2f})'
        print('TEST PASS')


if __name__ == "__main__":
    main()
