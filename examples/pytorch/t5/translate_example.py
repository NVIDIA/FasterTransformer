# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
import sys
import math
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(dir_path + "/../../../3rdparty/transformers/src/")

from transformers import PreTrainedTokenizerFast
from transformers import T5ForConditionalGeneration, T5Tokenizer # transformers-4.10.0-py3
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")

from examples.pytorch.t5.utils.ft_encoder import FTT5EncoderWeight, FTT5Encoder
from examples.pytorch.t5.utils.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from examples.pytorch.decoding.utils.recover_bpe import recover_bpe

def bleu_score(pred, ref):
    from sacrebleu import corpus_bleu
    bleu = corpus_bleu(pred, [ref], force=True)
    print("       bleu score: {:6.2f}".format(bleu.score))
    print("       bleu counts: {}".format(bleu.counts))
    print("       bleu totals: {}".format(bleu.totals))
    print("       bleu precisions: {}".format(bleu.precisions))
    print("       bleu sys_len: {}; ref_len: {}".format(bleu.sys_len, bleu.ref_len))
    return bleu

class TranslationResult(object):
    def __init__(self, name, frame_work):
        self.name = name
        self.frame_work = frame_work # FT or HF
        self.file_name = name + ".txt"

        self.token_list = []
        self.batch_ids_list = []
        self.batch_seq_len_list = []
        self.batch_num = 0
        self.execution_time = 0.0  # seconds
        self.sentence_num = 0
        self.token_num = 0
        self.bleu_score = None
            
def translate(args_dict):
    torch.set_printoptions(precision=6)
    batch_size = args_dict['batch_size']
    beam_size = args_dict['beam_width']
    max_seq_len = args_dict['max_seq_len']
    source_file = args_dict["source"]
    tgt_file = args_dict["target"]
    time_args = args_dict["test_time"]
    beam_search_diversity_rate = args_dict['beam_search_diversity_rate']
    topk = args_dict['sampling_topk']
    topp = args_dict['sampling_topp']
    tensor_para_size = args_dict['tensor_para_size']
    pipeline_para_size = args_dict['pipeline_para_size']
    max_ite = args_dict['max_iteration']
    ## huggingface without bias and use relative position embedding
    ## relative position embedding -> 0, absolute position embedding -> 1
    t5_with_bias = 0
    position_embedding_type = 0
    ## only huggingface model path supported
    model_path = args_dict['model_path'] if args_dict['model_path'] != None else args_dict['model']
    ckpt_path = args_dict['ckpt_path']
    model_type = args_dict['model_type']
    ## read checkpoint config if exists
    ckpt_config = configparser.ConfigParser()
    if (model_type == "Megatron"):
        ckpt_config_path = os.path.join(ckpt_path, 'config.ini')
        if os.path.isfile(ckpt_config_path):
            ckpt_config.read(ckpt_config_path)
            ## update structure config
            t5_with_bias = ckpt_config.getint('structure', 't5_with_bias')
            position_embedding_type = ckpt_config.getint('structure', 'position_embedding_type')
        else:
            raise Exception("config file does exist with the ckpt !")

    if model_type == "Megatron" and args_dict['ckpt_path'] == None:
        raise Exception("Megatron T5 model needs to specify checkpoint path !")

    print("\n=============== Argument ===============")
    for key in args_dict:
        print("{}: {}".format(key, args_dict[key]))
    print("========================================")

    lib_path = args_dict['lib_path']

    t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    if dist.is_mpi_available():
        try:
            dist.init_process_group(backend='mpi')
            rank = dist.get_rank()
        except:
            rank = dist.get_rank()
    else:
        rank = 0
    
    if time_args.find("0") != -1 or time_args.find("2") != -1:
        t5_model = t5_model.to(rank)
        if args_dict['data_type'] == 'fp16':
            t5_model = t5_model.half()
    ## TODO: modidy Megatron T5 Converter
    ## TODO: add megatron t5 tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

    encoder_config = t5_model.encoder.config
    decoder_config = t5_model.decoder.config
    q_scaling = 1.0 / (math.sqrt(encoder_config.d_kv))
    if (model_type == "Megatron"):
        ## update configs when using Megatron model structure
        q_scaling = 1.0

        encoder_ckpt_config = ckpt_config['encoder']
        decoder_ckpt_config = ckpt_config['decoder']
        encoder_config.d_model = ckpt_config.getint('encoder', 'd_model')
        encoder_config.vocab_size = ckpt_config.getint('encoder', 'vocab_size')
        encoder_config.num_heads = ckpt_config.getint('encoder', 'num_heads')
        encoder_config.d_kv = ckpt_config.getint('encoder', 'd_kv')
        encoder_config.d_ff = ckpt_config.getint('encoder', 'd_ff')
        encoder_config.num_layers = ckpt_config.getint('encoder', 'num_layers')
        encoder_config.relative_attention_num_buckets = ckpt_config.getint('encoder', 'relative_attention_num_buckets_or_max_pos_seq_len')

        decoder_config.d_model = ckpt_config.getint('decoder', 'd_model')
        decoder_config.vocab_size = ckpt_config.getint('decoder', 'vocab_size')
        decoder_config.num_heads = ckpt_config.getint('decoder', 'num_heads')
        decoder_config.d_kv = ckpt_config.getint('decoder', 'd_kv')
        decoder_config.d_ff = ckpt_config.getint('decoder', 'd_ff')
        decoder_config.num_layers = ckpt_config.getint('decoder', 'num_layers')
        decoder_config.relative_attention_num_buckets = ckpt_config.getint('decoder', 'relative_attention_num_buckets_or_max_pos_seq_len')
        decoder_config.decoder_start_token_id = 30522 # Only for megatron t5 model
        decoder_config.eos_token_id = 30523 # Only for megatron t5 model

    print(f"{model_type} encoder_config: {encoder_config}")
    print(f"{model_type} decoder_config: {decoder_config}")

    if os.path.isfile("gemm_config.in") and rank == 0:
        cmd = f"rm gemm_config.in"
        print(f"Run {cmd}")
        os.system(cmd)
    translation_result_list = []
    if time_args.find("0") != -1:
        translation_result_list.append(TranslationResult("hf-beamsearch-warmup", "HF"))
        translation_result_list.append(TranslationResult("hf-beamsearch", "HF"))
    if time_args.find("1") != -1:
        translation_result_list.append(TranslationResult("ft-beamsearch-warmup", "FT"))
        translation_result_list.append(TranslationResult("ft-beamsearch", "FT"))
        if rank == 0:
            is_fp16 = 1 if args_dict['data_type'] == 'fp16' else 0
            cmd = f"./bin/t5_gemm {batch_size // pipeline_para_size} {beam_size} {128} " \
                f"{encoder_config.d_model} {encoder_config.num_heads} {encoder_config.d_kv} {encoder_config.d_ff} " \
                f"{decoder_config.d_model} {decoder_config.num_heads} {decoder_config.d_kv} {decoder_config.d_ff} " \
                f"{decoder_config.vocab_size} {is_fp16} {tensor_para_size} 1 > .tmp_gemm.log"
            print(f"Run gemm test: {cmd}")
            os.system(cmd)
    if time_args.find("2") != -1:
        translation_result_list.append(TranslationResult("hf-sampling-warmup", "HF"))
        translation_result_list.append(TranslationResult("hf-sampling", "HF"))
    if time_args.find("3") != -1:
        translation_result_list.append(TranslationResult("ft-sampling-warmup", "FT"))
        translation_result_list.append(TranslationResult("ft-sampling", "FT"))
        if rank == 0:
            is_fp16 = 1 if args_dict['data_type'] == 'fp16' else 0
            cmd = f"./bin/t5_gemm {batch_size // pipeline_para_size} {1} {128} " \
                f"{encoder_config.d_model} {encoder_config.num_heads} {encoder_config.d_kv} {encoder_config.d_ff} " \
                f"{decoder_config.d_model} {decoder_config.num_heads} {decoder_config.d_kv} {decoder_config.d_ff} " \
                f"{decoder_config.vocab_size} {is_fp16} {tensor_para_size} 1 1 > .tmp_gemm.log"
            print(f"Run gemm test: {cmd}")
            os.system(cmd)

    if time_args.find("1") != -1 or time_args.find("3") != -1:
        ft_encoder_weight = FTT5EncoderWeight(encoder_config, tensor_para_size, pipeline_para_size, t5_with_bias, position_embedding_type)
        ft_decoding_weight = FTT5DecodingWeight(decoder_config, tensor_para_size, pipeline_para_size, t5_with_bias, position_embedding_type)

        if args_dict["ckpt_path"] is not None:
            ft_encoder_weight.load_from_bin(args_dict["ckpt_path"])
            ft_decoding_weight.load_from_bin(args_dict["ckpt_path"])
        else:
            ft_encoder_weight.load_from_model(t5_model)
            ft_decoding_weight.load_from_model(t5_model)
        
        if args_dict['data_type'] == 'fp16':
            t5_model = t5_model.half()
            ft_encoder_weight.to_half()
            ft_decoding_weight.to_half()

        remove_padding = True if batch_size > 32 else False
        ft_encoder = FTT5Encoder(ft_encoder_weight.w, lib_path, encoder_config.num_heads,
                                encoder_config.d_kv, encoder_config.d_ff,
                                encoder_config.d_model, remove_padding, encoder_config.num_layers,
                                encoder_config.relative_attention_num_buckets,
                                128, False, q_scaling, tensor_para_size, pipeline_para_size, t5_with_bias, position_embedding_type)
        ft_decoding = FTT5Decoding(ft_decoding_weight.w, lib_path,
                                decoder_config.num_heads, decoder_config.d_kv,
                                decoder_config.d_ff, encoder_config.d_model,
                                decoder_config.d_model, decoder_config.num_layers,
                                decoder_config.decoder_start_token_id, decoder_config.eos_token_id,
                                decoder_config.vocab_size,
                                q_scaling,
                                decoder_config.relative_attention_num_buckets, max_distance=128,
                                tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size,
                                t5_with_bias=t5_with_bias, position_embedding_type = position_embedding_type)

        ft_t5 = FTT5(ft_encoder, ft_decoding)

    with open(source_file, 'r') as f:
        src_text = recover_bpe(f.readlines())
        src_text = ["translate English to German: " + line.strip() for line in src_text]

    with open(tgt_file, 'r') as f:
        tgt_text = recover_bpe(f.readlines())

    for i in range(len(translation_result_list)):
        sys.stdout.flush()
        prev = 0
        start_time = datetime.now()
        while prev < len(src_text):
            input_texts = src_text[prev:prev+batch_size]
            prev += batch_size
            input_token = tokenizer(input_texts, return_tensors='pt', padding=True)
            
            if translation_result_list[i].frame_work == "HF":
                if translation_result_list[i].name.find("beamsearch") != -1:
                    hf_outputs = t5_model.generate(input_token.input_ids.to("cuda"), 
                                                   max_length=max_seq_len,
                                                   early_stopping=True,
                                                   num_beams=beam_size)
                elif translation_result_list[i].name.find("sampling") != -1:
                    hf_outputs = t5_model.generate(input_token.input_ids.to("cuda"),
                                                   max_length=max_seq_len,
                                                   early_stopping=True,
                                                   do_sample=True,
                                                   top_k=topk if topk > 0 else None,
                                                   top_p=topp if topp > 0.0 else None)
                translation_result_list[i].batch_ids_list.append(hf_outputs)
                translation_result_list[i].batch_seq_len_list.append(np.ones(len(input_texts)) * max_seq_len)
            elif translation_result_list[i].frame_work == "FT":
                tmp_beam_size = beam_size
                if translation_result_list[i].name.find("sampling") != -1:
                    tmp_beam_size = 1
                ft_decoding_outputs, ft_decoding_seq_lens = ft_t5(input_token,
                                                                  tmp_beam_size,
                                                                  max_seq_len,
                                                                  topk,
                                                                  topp,
                                                                  beam_search_diversity_rate=beam_search_diversity_rate,
                                                                  is_return_output_log_probs=args_dict["return_output_log_probs"],
                                                                  is_return_cum_log_probs=args_dict["return_cum_log_probs"])
                translation_result_list[i].batch_ids_list.append(ft_decoding_outputs)
                translation_result_list[i].batch_seq_len_list.append(ft_decoding_seq_lens)
            
            translation_result_list[i].sentence_num += len(input_token)
            translation_result_list[i].batch_num += 1
            if translation_result_list[i].name.find("warmup") != -1 and \
                (translation_result_list[i].batch_num > 10 or translation_result_list[i].sentence_num > 300):
                break
            if translation_result_list[i].batch_num >= max_ite:
                break
    
        stop_time = datetime.now()
        translation_result_list[i].execution_time = (stop_time - start_time).total_seconds()
        if translation_result_list[i].name.find("warmup") != -1:
            continue
        
        for batch_token, batch_seq_len in zip(translation_result_list[i].batch_ids_list, translation_result_list[i].batch_seq_len_list):
            for j in range(len(batch_token)):
                if translation_result_list[i].frame_work == "HF":
                    translation_result_list[i].token_list.append(fast_tokenizer.decode(batch_token[j][1:], skip_special_tokens=True))
                    translation_result_list[i].token_num += sum(batch_token[j][1:] != 0)
                elif translation_result_list[i].frame_work == "FT":
                    translation_result_list[i].token_list.append(fast_tokenizer.decode(batch_token[j][0][:batch_seq_len[j][0]], skip_special_tokens=True))
                    translation_result_list[i].token_num += batch_seq_len[j][0]

        if rank == 0:
            translation_result_list[i].bleu_score = bleu_score(translation_result_list[i].token_list, tgt_text[:len(translation_result_list[i].token_list)])
            with open(translation_result_list[i].name + ".txt", 'w') as f:
                for line in translation_result_list[i].token_list:
                    f.write(line)
    
    if rank == 0:
        for t in translation_result_list:
            if t.name.find("warmup") != -1: 
                continue
            print(f"[INFO] {t.name} translates {t.batch_num} batches taking {t.execution_time:.2f} sec to translate "
                f"{t.token_num} tokens, BLEU score: {t.bleu_score.score:.2f}, {(t.token_num / t.execution_time):.0f} tokens/sec."
                f" ({t.bleu_score.sys_len} words, {(t.bleu_score.sys_len / t.execution_time):.0f} words/sec)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-beam', '--beam_width', type=int, default=4, metavar='NUMBER',
                        help='beam width (default: 4)')
    parser.add_argument('-s', '--max_seq_len', type=int, default=200, metavar='NUMBER',
                        help='max sequence length (default: 200)')
    parser.add_argument("--source", default="../examples/pytorch/decoding/utils/translation/test.en",
                        help="Path to the source file.")
    parser.add_argument("--target", default="../examples/pytorch/decoding/utils/translation/test.de",
                        help="Path to the target file.")
    parser.add_argument('-time', '--test_time', type=str, default='', metavar='STRING',
                        help='''
                            Test the time of which one (default: '' (not test anyone) ); 
                            '': not test anyone 
                            '0': test hf_beamsearch  
                            '1': test ft_beamsearch 
                            '2': test hf_sampling 
                            '3': test ft_sampling 
                            'e.g., if you want to test tf_beamsearch and ft_sampling, 
                            then you need to use -time '03' ''')
    parser.add_argument('-diversity_rate', '--beam_search_diversity_rate', type=float, default=0.0, metavar='NUMBER',
                        help='deviersity rate of beam search. default is 0. When diversity rate = 0, it is equivalent to the naive beams earch.')
    parser.add_argument('-topk', '--sampling_topk', type=int, default=1, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.0, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type (default: fp32)', choices=['fp32', 'fp16'])
    parser.add_argument('-lib_path', '--lib_path', type=str, default="lib/libth_t5.so", metavar='STRING',
                        help='the path of FasterTransformer pytorch t5 op library.')
    parser.add_argument('-model_path', '--model_path', type=str, default=None, metavar='STRING',
                        help='T5 model path.')
    parser.add_argument('-model', '--model', type=str, default="t5-small", metavar='STRING',
                        help='T5 model size. Only used when --model_path=None', choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"])
    parser.add_argument('-tensor_para_size', '--tensor_para_size', type=int, default=1, metavar='NUMBER',
                        help='size of tensor parallelism (default: 1)')
    parser.add_argument('-pipeline_para_size', '--pipeline_para_size', type=int, default=1, metavar='NUMBER',
                        help='size of pipeline parallelism (default: 1)')
    # assume checkpoint config is also in the same path
    parser.add_argument('--ckpt_path', type=str, help='path to the checkpoint file.')
    parser.add_argument('-max_ite', '--max_iteration', type=int, default=100000, metavar='NUMBER',
                        help='Maximum iteraiton for translation, default is 100000 (as large as possible to run all test set).')
    parser.add_argument('--model_type', type=str, default="Huggingface", choices=["Huggingface", "Megatron"],
                        help='Megatron T5 uses bias and supports both absulte and relative positional embedding;'
                        'Huggingface T4 adopts the paper\'s implementation and has no bias')
    parser.add_argument('--return_output_log_probs', action='store_true',
                        help='Return the log probability of generated tokens.')
    parser.add_argument('--return_cum_log_probs', action='store_true',
                        help='Return the cumulative log probability of generated tokens.')
    args = parser.parse_args()

    translate(vars(args))