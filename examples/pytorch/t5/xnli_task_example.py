# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import configparser
import dataclasses
import json
import os
import pathlib
import time

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm


from omegaconf.omegaconf import OmegaConf
from nemo.collections.nlp.data.glue_benchmark.glue_benchmark_dataset import (
    TextToTextGLUEDataset,
    TextToTextXNLIDataset,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.common.metrics.classification_accuracy import ExactStringPerCategoryMatchMetric

from examples.pytorch.t5.utils.ft_encoder import FTT5EncoderWeight, FTT5Encoder
from examples.pytorch.t5.utils.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from examples.pytorch.tokenizer import add_special_tokens_to_tokenizer


def _build_dataset(data_cfg, tokenizer):
    if data_cfg.task_name == 'xnli':
        dataset = TextToTextXNLIDataset(
            data_cfg.file_path,
            task_name=data_cfg.task_name,
            tokenizer=tokenizer,
            max_seq_length=data_cfg.max_seq_length,
            lang_list=data_cfg.eval_languages,
        )
    else:
        dataset = TextToTextGLUEDataset(
            data_cfg.file_path,
            task_name=data_cfg.task_name,
            tokenizer=tokenizer,
            max_seq_length=data_cfg.max_seq_length,
        )
    return dataset


@dataclasses.dataclass
class Metric:
    acc: float


@dataclasses.dataclass
class RequestAndResult:
    model_answer: str
    target: str
    lang: str
    metrics: Metric


def preds_and_labels_to_text(tokenizer, preds, labels):
    preds = preds.cpu().numpy().tolist()
    labels = labels.cpu().numpy().tolist()
    # preds = [pred[0] for pred in preds]

    preds_text, labels_text = [], []
    for _, (pred, label) in enumerate(zip(preds, labels)):
        if tokenizer.eos_id in pred:
            idx = pred.index(tokenizer.eos_id)
            pred = pred[:idx]

        # Legacy sentencepiece detokenization still preserves special tokens which messes up exact string match.
        if hasattr(tokenizer, 'special_token_to_id'):
            pred = [id for id in pred if id not in tokenizer.special_token_to_id.values()]
            label = [id for id in label if id not in tokenizer.special_token_to_id.values()]
        pred = tokenizer.ids_to_text(pred)
        label = tokenizer.ids_to_text(label)
        preds_text.append(pred)
        labels_text.append(label)

    return preds_text, labels_text


def accuracy_score(pred, ref):
    assert len(pred) == len(ref)
    total = len(pred)
    correct = 0
    for p, r in zip(pred, ref):
        if p in r:
            correct += 1
        # else:
        #     print(f"[pred]: {p} [label]: {r}")
    print(f"[total_acc] {correct / total}")
    return correct / total


class InputToken:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class EncoderDecoderConfig:
    def __init__(self, d_model, vocab_size, num_heads, d_kv, d_ff, num_layers, 
                 relative_attention_num_buckets_or_max_pos_seq_len, decoder_start_token_id=0, decoder_end_token_id=1):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.relative_attention_num_buckets = relative_attention_num_buckets_or_max_pos_seq_len
        self.decoder_start_token_id = decoder_start_token_id
        self.decoder_end_token_id = decoder_end_token_id


data_type_mapping = {"fp32": 0, "fp16": 1, "bf16": 2}

def xnli_task(args_dict):
    torch.set_printoptions(precision=6)
    batch_size = args_dict['batch_size']
    beam_size = args_dict['beam_width']
    max_output_len = args_dict['max_output_len']
    beam_search_diversity_rate = args_dict['beam_search_diversity_rate']
    topk = args_dict['sampling_topk']
    topp = args_dict['sampling_topp']
    tensor_para_size = args_dict['tensor_para_size']
    pipeline_para_size = args_dict['pipeline_para_size']

    if args_dict['ckpt_path'] is None:
        raise Exception("Megatron T5 model needs to specify checkpoint path !")

    if dist.is_mpi_available():
        try:
            dist.init_process_group(backend='mpi')
            rank = dist.get_rank()
        except:
            rank = dist.get_rank()
    else:
        rank = 0

    assert dist.get_world_size() == tensor_para_size * pipeline_para_size

    ckpt_path = args_dict['ckpt_path']
    ## read checkpoint config if exists
    ckpt_config = configparser.ConfigParser()

    if args_dict['ckpt_path'] is None:
        raise Exception("Megatron T5 model needs to specify checkpoint path !")

    tokenizer_model_path = os.path.join(ckpt_path, "tokenizer.model")
    ckpt_config_path = os.path.join(ckpt_path, 'config.ini')
    if os.path.isfile(ckpt_config_path):
        ckpt_config.read(ckpt_config_path)
        ## update structure config
        t5_with_bias = ckpt_config.getboolean('structure', 't5_with_bias')
        ## megatron with bias and use absolute position embedding
        ## relative position embedding -> 0, absolute position embedding -> 1
        position_embedding_type = 0 if ckpt_config.get('structure', 'position_embedding_type') == 'relative' else 1
        use_gated_activation = ckpt_config.getboolean('structure', 'use_gated_activation')
        weight_data_type = {"fp16": np.float16, "fp32": np.float32}[ckpt_config.get("encoder", "weight_data_type")]
        activation_type = ckpt_config.get('encoder', 'feed_forward_proj')
        assert ckpt_config.getint("encoder", "tensor_para_size") == tensor_para_size
    else:
        raise Exception("config file does exist with the ckpt !")

    if rank == 0:
        print("\n=============== Argument ===============")
        for key in args_dict:
            print("{}: {}".format(key, args_dict[key]))
        print("========================================")

    lib_path = args_dict['lib_path']

    #xnli
    tokenizer_mt5 = get_nmt_tokenizer(
        library='sentencepiece',
        model_name=None,
        tokenizer_model=tokenizer_model_path,
        vocab_file=None,
        merges_file=None,
        legacy=True,
    )
    add_special_tokens_to_tokenizer(tokenizer_mt5)

    assert tokenizer_mt5.bos_id == ckpt_config.getint("decoder", "decoder_start_token_id")
    assert tokenizer_mt5.eos_id == ckpt_config.getint("decoder", "eos_token_id")

    token_params = {
        tokenizer_mt5.bos_token: tokenizer_mt5.bos_id,
        tokenizer_mt5.eos_token: tokenizer_mt5.eos_id,
        tokenizer_mt5.pad_token: tokenizer_mt5.pad_id,
    }
    print(f"tokenizer special tokens: {token_params}")

    xnli_cfg = OmegaConf.create({
        "file_path": args_dict['data_path'],
        "task_name": "xnli",
        "max_seq_length": 512,
        "eval_languages": ['en', 'es', 'de', 'fr']
    })
    xnli_dataset = _build_dataset(xnli_cfg, tokenizer_mt5)

    data_loader = torch.utils.data.DataLoader(
                xnli_dataset,
                collate_fn=xnli_dataset.collate_fn,
                batch_size=batch_size,
                num_workers=1,
                pin_memory=False,
                drop_last=True)

    q_scaling = 1.0

    encoder_config = EncoderDecoderConfig(ckpt_config.getint('encoder', 'd_model'),
                                          ckpt_config.getint('encoder', 'vocab_size'),
                                          ckpt_config.getint('encoder', 'num_heads'),
                                          ckpt_config.getint('encoder', 'd_kv'),
                                          ckpt_config.getint('encoder', 'd_ff'),
                                          ckpt_config.getint('encoder', 'num_layers'),
                                          ckpt_config.getint('encoder', 'relative_attention_num_buckets_or_max_pos_seq_len')
                                          )
    
    decoder_config = EncoderDecoderConfig(ckpt_config.getint('decoder', 'd_model'),
                                          ckpt_config.getint('decoder', 'vocab_size'),
                                          ckpt_config.getint('decoder', 'num_heads'),
                                          ckpt_config.getint('decoder', 'd_kv'),
                                          ckpt_config.getint('decoder', 'd_ff'),
                                          ckpt_config.getint('decoder', 'num_layers'),
                                          ckpt_config.getint('decoder', 'relative_attention_num_buckets_or_max_pos_seq_len'),
                                          tokenizer_mt5.bos_id,
                                          tokenizer_mt5.eos_id
                                          )

    ## run gemm test
    if os.path.isfile("gemm_config.in") and rank == 0:
        cmd = f"rm gemm_config.in"
        print(f"Run {cmd}")
        os.system(cmd)
    if rank == 0:
        data_type = data_type_mapping[args_dict['data_type']]
        cmd = f"./bin/t5_gemm {batch_size // pipeline_para_size} {beam_size} {128} " \
            f"{encoder_config.d_model} {encoder_config.num_heads} {encoder_config.d_kv} {encoder_config.d_ff} " \
            f"{decoder_config.d_model} {decoder_config.num_heads} {decoder_config.d_kv} {decoder_config.d_ff} " \
            f"{decoder_config.vocab_size} {data_type} {tensor_para_size} 1 > .tmp_gemm.log"
        print(f"Run gemm test: {cmd}")
        os.system(cmd)

    dist.barrier()

    ft_encoder_weight = FTT5EncoderWeight(
        encoder_config,
        tensor_para_size,
        pipeline_para_size,
        t5_with_bias=t5_with_bias,
        use_gated_activation=use_gated_activation,
        position_embedding_type=position_embedding_type,
        weight_data_type=weight_data_type,
    )
    ft_decoding_weight = FTT5DecodingWeight(
        decoder_config,
        tensor_para_size,
        pipeline_para_size,
        t5_with_bias=t5_with_bias,
        use_gated_activation=use_gated_activation,
        position_embedding_type=position_embedding_type,
        weight_data_type=weight_data_type,
    )

    ft_encoder_weight.load_from_bin(args_dict["ckpt_path"])
    ft_decoding_weight.load_from_bin(args_dict["ckpt_path"])

    if args_dict['data_type'] == 'fp16':
        ft_encoder_weight.to_half()
        ft_decoding_weight.to_half()
    elif args_dict['data_type'] == 'fp32':
        ft_encoder_weight.to_single()
        ft_decoding_weight.to_single()
    elif args_dict['data_type'] == 'bf16':
        ft_encoder_weight.to_bfloat16()
        ft_decoding_weight.to_bfloat16()

    remove_padding = True if batch_size > 32 else False
    ft_encoder = FTT5Encoder(ft_encoder_weight.w, lib_path, encoder_config.num_heads,
                            encoder_config.d_kv, encoder_config.d_ff,
                            encoder_config.d_model, remove_padding, encoder_config.num_layers,
                            encoder_config.relative_attention_num_buckets,
                            128, False, q_scaling, tensor_para_size, pipeline_para_size, t5_with_bias, position_embedding_type, activation_type)
    ft_decoding = FTT5Decoding(ft_decoding_weight.w, lib_path,
                            decoder_config.num_heads, decoder_config.d_kv,
                            decoder_config.d_ff, encoder_config.d_model,
                            decoder_config.d_model, decoder_config.num_layers,
                            decoder_config.decoder_start_token_id, decoder_config.decoder_end_token_id,
                            decoder_config.vocab_size,
                            q_scaling,
                            decoder_config.relative_attention_num_buckets, max_distance=128,
                            tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size,
                            t5_with_bias=t5_with_bias, activation_type=activation_type, position_embedding_type=position_embedding_type)

    ft_t5 = FTT5(ft_encoder, ft_decoding)
    
    #metric
    languages = ['de','en','es','fr']
    acc_metric = ExactStringPerCategoryMatchMetric(languages)

    preds_list = []
    labels_list = []
    results_list = []
    start = time.time()
    for idx, batch in tqdm(enumerate(data_loader)):
        input_token = InputToken(batch['text_enc'], batch['enc_mask'])
        ft_decoding_outputs, ft_decoding_seq_lens = ft_t5(input_token,
                                                          None,
                                                          beam_size,
                                                          max_output_len,
                                                          topk,
                                                          topp,
                                                          beam_search_diversity_rate=beam_search_diversity_rate,
                                                          is_return_output_log_probs=args_dict["return_output_log_probs"],
                                                          is_return_cum_log_probs=args_dict["return_cum_log_probs"])
        ft_decoding_outputs = ft_decoding_outputs.squeeze()
        preds, labels = preds_and_labels_to_text(tokenizer_mt5, torch.IntTensor(ft_decoding_outputs), batch['labels'])
        langs = batch['lang']
        for _, (pred, label, lang) in enumerate(zip(preds, labels, langs)):
            _ = acc_metric(pred, label, lang)
        labels_list += labels
        preds_list += preds

        results_list.extend([
            RequestAndResult(
                model_answer=pred,
                target=label,
                lang=lang,
                metrics=Metric(acc=(pred == label))
            )
            for lang, pred, label in zip(langs, preds, labels)
        ])

    end = time.time()

    lang_accuracy = acc_metric.compute()

    if rank == 0:

        print(f"\n[Elapsed Time]: {end - start} seconds")

        # each language
        for lang in languages:
            print(f'[{lang}_acc]', lang_accuracy[lang].item())

        # total accuracy
        accuracy = accuracy_score(preds_list, labels_list)
        output_path = args_dict.get("output_path")
        if output_path is not None and rank == 0:
            output_path = pathlib.Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as output_file:
                results = {
                    "results": {
                        "xnli": {
                            "acc": accuracy
                        }
                    },
                    "output": {
                        "xnli": [
                            dataclasses.asdict(r) for r in results_list
                        ]
                    }
                }
                json.dump(results, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-beam', '--beam_width', type=int, default=4, metavar='NUMBER',
                        help='beam width (default: 4)')
    parser.add_argument('-s', '--max_output_len', type=int, default=10, metavar='NUMBER',
                        help='max output length (default: 10)')
    parser.add_argument('-diversity_rate', '--beam_search_diversity_rate', type=float, default=0.0, metavar='NUMBER',
                        help='deviersity rate of beam search. default is 0. When diversity rate = 0, it is equivalent to the naive beams earch.')
    parser.add_argument('-topk', '--sampling_topk', type=int, default=1, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.0, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type (default: fp32)', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('-lib_path', '--lib_path', type=str, default="/workspace/FasterTransformer/build/lib/libth_t5.so", metavar='STRING',
                        help='the path of FasterTransformer pytorch t5 op library.')
    parser.add_argument('-data_path', '--data_path', type=str, required=True, help="the xnli task data path")
    parser.add_argument('-tensor_para_size', '--tensor_para_size', type=int, default=1, metavar='NUMBER',
                    help='size of tensor parallelism (default: 1)')
    parser.add_argument('-pipeline_para_size', '--pipeline_para_size', type=int, default=1, metavar='NUMBER',
                        help='size of pipeline parallelism (default: 1)')
    # assume checkpoint config is also in the same path
    parser.add_argument('--ckpt_path', type=str, help='path to the checkpoint file.')
    parser.add_argument('--output_path', help='path to results file with calculated metrics.')
    parser.add_argument('--return_output_log_probs', action='store_true',
                        help='Return the log probability of generated tokens.')
    parser.add_argument('--return_cum_log_probs', action='store_true',
                        help='Return the cumulative log probability of generated tokens.')
    args = parser.parse_args()

    xnli_task(vars(args))