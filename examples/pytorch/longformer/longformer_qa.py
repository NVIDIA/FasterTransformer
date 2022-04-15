import argparse
import sys
import os
import json
import time

import torch
from transformers import LongformerTokenizer, LongformerForQuestionAnswering

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from examples.pytorch.longformer.model import from_hf_longformer_weight_to_ft, FTLongformerEncoder

def parse_from_config(model_dir):
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    layer_num = config['num_hidden_layers']
    hidden_size = config['hidden_size']
    head_num = config['num_attention_heads']
    size_per_head = hidden_size // head_num
    intermediate_size = config['intermediate_size']
    # assume all local attn window are same size. TODO: Improve later
    local_attn_window_size = config['attention_window'][0]
    attn_scaler = 1.0 / (size_per_head ** 0.5)
    return (layer_num, hidden_size, head_num, size_per_head,
            intermediate_size, local_attn_window_size, attn_scaler)


def build_ft_longformer(hf_model_dir, layer_num, head_num, size_per_head,
                        intermediate_size, local_attn_window_size,
                        max_global_token_num, batch_size, seq_len,
                        attn_scaler, ft_longformer_lib, fp16):
    weights_file = os.path.join(hf_model_dir, 'pytorch_model.bin')
    ft_encoder = FTLongformerEncoder(weights_file, layer_num, head_num, size_per_head,
                                     intermediate_size, local_attn_window_size,
                                     max_global_token_num, batch_size, seq_len,
                                     attn_scaler, ft_longformer_lib, fp16)
    ft_longformer = build_hf_longformer(hf_model_dir)
    if fp16:
        ft_longformer = ft_longformer.half()
    ft_longformer.cuda()
    ft_longformer.eval()
    ft_encoder.set_hf_plugin_mode(True)
    ft_longformer.longformer.encoder = ft_encoder
    return ft_longformer


def build_hf_longformer(model_dir):
    hf_longformer = LongformerForQuestionAnswering.from_pretrained(model_dir)
    hf_longformer.cuda()
    hf_longformer.eval()
    return hf_longformer


def prepare_input(question, passage_text, seq_len, batch_size, model_dir, fp16):
    tokenizer = LongformerTokenizer.from_pretrained(model_dir)
    encoding = tokenizer(question, passage_text, return_token_type_ids=True)
    qa_sep_index = 0
    for token_id in encoding['input_ids']:
        if token_id == tokenizer.sep_token_id:
            break
        qa_sep_index += 1

    actual_seq_len = len(encoding['input_ids'])
    input_ids = torch.ones((seq_len, ), dtype=torch.int32)  # hf use 1 as padding
    input_ids[:actual_seq_len] = torch.tensor(encoding['input_ids'], dtype=torch.int32)

    local_attn_mask = torch.zeros((seq_len, ), dtype=torch.float32)
    local_attn_mask[:actual_seq_len] = torch.tensor(encoding['attention_mask'], dtype=torch.float32)

    global_attn_mask = torch.zeros_like(local_attn_mask, dtype=torch.float32)
    # mark all question's token as global attention
    global_attn_mask[:qa_sep_index] = 1.0

    # make a batch
    input_ids_b = torch.stack([input_ids for _ in range(batch_size)], axis=0).contiguous().cuda()
    local_attn_mask_b = torch.stack([local_attn_mask for _ in range(batch_size)], axis=0).contiguous()
    global_attn_mask_b = torch.stack([global_attn_mask for _ in range(batch_size)], axis=0).contiguous()

    if fp16:
        local_attn_mask_b = local_attn_mask_b.half()
        global_attn_mask_b = global_attn_mask_b.half()

    local_attn_mask_b = local_attn_mask_b.cuda()
    global_attn_mask_b = global_attn_mask_b.cuda()

    return input_ids_b, local_attn_mask_b, global_attn_mask_b, input_ids, actual_seq_len


def decode_output(outputs, model_dir, input_ids, actual_seq_len):
    tokenizer = LongformerTokenizer.from_pretrained(model_dir)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist()[:actual_seq_len])
    # need to
    start_logits = start_logits[0, :actual_seq_len]
    end_logits = end_logits[0, :actual_seq_len]
    answer_tokens = all_tokens[torch.argmax(start_logits):torch.argmax(end_logits) + 1]
    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))  # remove space prepending space token
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-dir', required=True,
                        help='Path to huggingface model dir where model file and config file is stored')
    parser.add_argument('-l', '--ft-longformer-lib', type=str, default=os.path.join(project_root, 'build', 'lib', 'libth_longformer.so'),
                        help='Path to fastertransformer longformer pytorch op lib')
    parser.add_argument('--fp16', action='store_true', help="Use FP16")
    parser.add_argument('-p', '--passage', type=str, nargs='*', help='Text for paragraph/passage for LongformerBERT QA',
                        default=None)
    parser.add_argument('-pf', '--passage-file', type=str, help='File containing input passage',
                        default=None)
    parser.add_argument('-q', '--question', required=True, type=str, nargs='*', help='Text for query/question for LongformerBERT QA',
                        default='')
    parser.add_argument('-s', '--sequence-length',
                        help='The sequence length to use. Defaults to 1024',
                        default=1024, type=int)
    parser.add_argument('-b', '--batch-size',
                        help='Batch size to use. Note, it just copy the single question and passage token to form a batch, just for performance test.',
                        default=1, type=int)
    parser.add_argument("-g", "--max-global-attention-num", default=128,
                        help="Max global attention token num from start of the sequence to the end.", type=int)
    parser.add_argument('-r', '--repeat-test-num',
                        help='If specified, will run inference serveral rounds, to test average performace.',
                        type=int,
                        default=None)
    args, _ = parser.parse_known_args()
    print("======== Arguments ========")
    print(args)

    with open(os.path.join(args.model_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    # prepare question and passage
    question = ' '.join(args.question)
    if bool(args.passage) == bool(args.passage_file):
        raise RuntimeError("You must specify only one of --passage or --passage-file.")

    if args.passage:
        passage_text = ' '.join(args.passage)
    else:
        with open(args.passage_file, 'r', encoding="utf-8") as f:
            passage_text = f.read()

    # prepare model config and weights
    model_dir = args.model_dir
    ft_longformer_lib = args.ft_longformer_lib
    seq_len = args.sequence_length
    batch_size = args.batch_size
    repeat_num = args.repeat_test_num if args.repeat_test_num else 0
    max_global_token_num = args.max_global_attention_num

    (layer_num, hidden_size, head_num, size_per_head,
     intermediate_size, local_attn_window_size, attn_scaler) = parse_from_config(model_dir)

    # huggeingFace longformer
    hf_longformer = build_hf_longformer(model_dir)
    if args.fp16:
        hf_longformer = hf_longformer.half()
    # fastertransformer longformer
    ft_longformer = build_ft_longformer(model_dir, layer_num, head_num, size_per_head,
                                        intermediate_size, local_attn_window_size,
                                        max_global_token_num, batch_size, seq_len,
                                        attn_scaler, ft_longformer_lib, args.fp16)
    # prepare input
    input_ids_b, local_attn_mask_b, global_attn_mask_b, input_ids, actual_seq_len = prepare_input(
        question, passage_text, seq_len, batch_size, model_dir, args.fp16)

    # 1. Compare the performance between HF and FT, using dummy input
    dummy_local_attn_mask_b = torch.ones_like(local_attn_mask_b)
    extended_mask_b = (global_attn_mask_b + dummy_local_attn_mask_b) * 10000. - 10000.
    dummy_embedding_out = torch.rand(batch_size, seq_len, hidden_size, dtype=torch.float32)
    if args.fp16:
        dummy_embedding_out = dummy_embedding_out.half()
    dummy_embedding_out = dummy_embedding_out.cuda()
    hf_encoder = hf_longformer.longformer.encoder
    ft_encoder = ft_longformer.longformer.encoder

    with torch.no_grad():
        # HuggingFace warmup
        for i in range(10):
            output = hf_encoder(dummy_embedding_out, attention_mask=extended_mask_b, head_mask=None,
                                output_attentions=None, output_hidden_states=None, return_dict=True)

        start = time.time()
        for i in range(repeat_num):
            output = hf_encoder(dummy_embedding_out, attention_mask=extended_mask_b, head_mask=None,
                                output_attentions=None, output_hidden_states=None, return_dict=True)
        stop = time.time()
        print("HuggingFace Longformer encoder average latency {:.3f} second ({} iterations)".format((stop - start) / repeat_num, repeat_num))

    ft_longformer.longformer.encoder.set_hf_plugin_mode(False)
    with torch.no_grad():
        # FT warmup
        for i in range(10):
            output = ft_encoder.forward(dummy_embedding_out, dummy_local_attn_mask_b, global_attn_mask_b)

        start = time.time()
        for i in range(repeat_num):
            output = ft_encoder.forward(dummy_embedding_out, dummy_local_attn_mask_b, global_attn_mask_b)
        stop = time.time()
        print("FasterTransformer Longformer encoder average latency {:.3f} second ({} iterations)".format((stop - start) / repeat_num, repeat_num))

    # 2. Verify the correctness
    ft_longformer.longformer.encoder.set_hf_plugin_mode(True)
    with torch.no_grad():
        outputs = ft_longformer(input_ids_b,
                                attention_mask=local_attn_mask_b,
                                global_attention_mask=global_attn_mask_b)
        ft_answer = decode_output(outputs, model_dir, input_ids, actual_seq_len)

        outputs = hf_longformer(input_ids_b,
                                attention_mask=local_attn_mask_b,
                                global_attention_mask=global_attn_mask_b)
        hf_answer = decode_output(outputs, model_dir, input_ids, actual_seq_len)
        print("HuggingFace Answer: " + hf_answer)
        print("FasterTransformer Answer: " + ft_answer)

if __name__ == '__main__':
    main()
