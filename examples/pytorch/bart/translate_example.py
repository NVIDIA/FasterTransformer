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
import os
import sys
import math
import logging
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist

from transformers import BartForConditionalGeneration, BartTokenizer 
from transformers import MBartForConditionalGeneration, MBartTokenizer 

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.bart.utils.ft_encoder import FTBartEncoderWeight, FTBartEncoder
from examples.pytorch.bart.utils.ft_decoding import FTBartDecodingWeight, FTBartDecoding, FTBart
from examples.pytorch.decoding.utils.recover_bpe import recover_bpe

LOGGER = logging.getLogger(__name__)

gemm_data_type_mapping = {"fp32":0, "fp16":1, "bf16":2}

def bleu_score(pred, ref):
    from sacrebleu import corpus_bleu
    bleu = corpus_bleu(pred, [ref], force=True)
    LOGGER.info("       bleu score: {:6.2f}".format(bleu.score))
    LOGGER.info("       bleu counts: {}".format(bleu.counts))
    LOGGER.info("       bleu totals: {}".format(bleu.totals))
    LOGGER.info("       bleu precisions: {}".format(bleu.precisions))
    LOGGER.info("       bleu sys_len: {}; ref_len: {}".format(bleu.sys_len, bleu.ref_len))
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
    # repetition_penalty = args_dict["repetition_penalty"]
    # temperature = args_dict["temperature"]
    # len_penalty = args_dict["len_penalty"]
    model_path = args_dict['model_path'] if args_dict['model_path'] != None else args_dict['model']
    lib_path = args_dict['lib_path']

    if dist.is_mpi_available():
        try:
            dist.init_process_group(backend='mpi')
            rank = dist.get_rank()
        except:
            rank = dist.get_rank()
    else:
        rank = 0
    
    if rank == 0:
        LOGGER.info("\n=============== Argument ===============")
        for key in args_dict:
            LOGGER.info("{}: {}".format(key, args_dict[key]))
        LOGGER.info("========================================")

    if 'mbart' not in model_path:
        hf_bart_model = BartForConditionalGeneration.from_pretrained(model_path)
        tokenizer = BartTokenizer.from_pretrained(model_path)
        layernorm_type = "post_layernorm"
    else:
        hf_bart_model = MBartForConditionalGeneration.from_pretrained(model_path)
        tokenizer = MBartTokenizer.from_pretrained(model_path)
        layernorm_type = "pre_layernorm"
    is_mbart = hf_bart_model.config.add_final_layer_norm
    hf_bart_model = hf_bart_model.eval().to('cuda')
    try:
        fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    except:
        fast_tokenizer = tokenizer

    config = hf_bart_model.config
    activation_type = config.activation_function
    bart_with_bias = True
    use_gated_activation = False
    position_embedding_type = 1 # absolute positional embedding
    weight_data_type = np.float32
    encoder_head_size = config.d_model // config.encoder_attention_heads
    decoder_head_size = config.d_model // config.decoder_attention_heads

    if time_args.find("0") != -1 or time_args.find("2") != -1:
        hf_bart_model = hf_bart_model.to(rank)
        if args_dict['inference_data_type'] == 'fp16':
            hf_bart_model = hf_bart_model.half()
        elif args_dict['inference_data_type'] == 'bf16':
            hf_bart_model = hf_bart_model ## bfloat inference not supported yet

    if rank == 0:
        LOGGER.debug(f"config: {config}")

    if os.path.isfile("gemm_config.in") and rank == 0:
        cmd = f"rm gemm_config.in"
        LOGGER.info(f"Run {cmd}")
        os.system(cmd)

    translation_result_list = []
    if time_args.find("0") != -1:
        translation_result_list.append(TranslationResult("hf-beamsearch-warmup", "HF"))
        translation_result_list.append(TranslationResult("hf-beamsearch", "HF"))

    if time_args.find("1") != -1:
        translation_result_list.append(TranslationResult("ft-beamsearch-warmup", "FT"))
        translation_result_list.append(TranslationResult("ft-beamsearch", "FT"))
        if rank == 0 and not args_dict["skip_gemm_test"]:
            inference_data_type = gemm_data_type_mapping[args_dict['inference_data_type']]
            cmd = f"./bin/t5_gemm {math.ceil(batch_size / pipeline_para_size)} {beam_size} {max_seq_len} " \
                f"{config.d_model} {config.encoder_attention_heads} {encoder_head_size} {config.encoder_ffn_dim} " \
                f"{config.d_model} {config.decoder_attention_heads} {decoder_head_size} {config.decoder_ffn_dim} " \
                f"{config.vocab_size} {inference_data_type} {tensor_para_size} 0 > .tmp_gemm.log"
            LOGGER.info(f"Run gemm test: {cmd}")
            os.system(cmd)

    if time_args.find("2") != -1:
        translation_result_list.append(TranslationResult("hf-sampling-warmup", "HF"))
        translation_result_list.append(TranslationResult("hf-sampling", "HF"))
    if time_args.find("3") != -1:
        translation_result_list.append(TranslationResult("ft-sampling-warmup", "FT"))
        translation_result_list.append(TranslationResult("ft-sampling", "FT"))
        if rank == 0 and not args_dict["skip_gemm_test"]:
            inference_data_type = gemm_data_type_mapping[args_dict['inference_data_type']]
            cmd = f"./bin/t5_gemm {math.ceil(batch_size / pipeline_para_size)} {1} {max_seq_len} " \
                f"{config.d_model} {config.encoder_attention_heads} {encoder_head_size} {config.encoder_ffn_dim} " \
                f"{config.d_model} {config.decoder_attention_heads} {decoder_head_size} {config.decoder_ffn_dim} " \
                f"{config.vocab_size} {inference_data_type} {tensor_para_size} 1 > .tmp_gemm.log"
            LOGGER.info(f"Run gemm test: {cmd}")
            os.system(cmd)

    if time_args.find("1") != -1 or time_args.find("3") != -1:
        remove_padding = True if batch_size > 32 else False
        ft_encoder_weight = FTBartEncoderWeight(
            config,
            tensor_para_size,
            pipeline_para_size,
            bart_with_bias=bart_with_bias,
            mbart=is_mbart,
            use_gated_activation=use_gated_activation,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type,
        )
        ft_encoder_weight.load_from_model(hf_bart_model.float())

        ft_decoding_weight = FTBartDecodingWeight(
            config,
            tensor_para_size,
            pipeline_para_size,
            bart_with_bias=bart_with_bias,
            mbart=is_mbart,
            use_gated_activation=use_gated_activation,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type,
        )
        ft_decoding_weight.load_from_model(hf_bart_model.float())
        if args_dict['inference_data_type'] == "fp16":
            ft_encoder_weight.to_half()
            ft_decoding_weight.to_half()
        elif args_dict['inference_data_type'] == "bf16":
            ft_encoder_weight.to_bfloat16()
            ft_decoding_weight.to_bfloat16()
        
        ft_encoder = FTBartEncoder(ft_encoder_weight.w, lib_path, config.encoder_attention_heads,
                                   encoder_head_size, config.encoder_ffn_dim,
                                   config.d_model, remove_padding, config.encoder_layers, 
                                   tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size, 
                                   bart_with_bias=bart_with_bias, mbart=is_mbart,
                                   position_embedding_type=position_embedding_type, 
                                   activation_type=activation_type, layernorm_type=layernorm_type)

        ft_decoding = FTBartDecoding(ft_decoding_weight.w, lib_path,
                                     config.decoder_attention_heads, decoder_head_size,
                                     config.decoder_ffn_dim, config.d_model,
                                     config.d_model, config.decoder_layers,
                                     config.decoder_start_token_id, config.eos_token_id, config.vocab_size,
                                     tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size, 
                                     bart_with_bias=bart_with_bias, mbart=is_mbart,
                                     position_embedding_type=position_embedding_type, 
                                     activation_type=activation_type, layernorm_type=layernorm_type)

        ft_bart = FTBart(ft_encoder, ft_decoding)

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
                    hf_outputs = hf_bart_model.generate(input_token.input_ids.to("cuda"), 
                                                        max_length=max_seq_len,
                                                        early_stopping=True,
                                                        num_beams=beam_size)
                elif translation_result_list[i].name.find("sampling") != -1:
                    hf_outputs = hf_bart_model.generate(input_token.input_ids.to("cuda"),
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
                return_dict = ft_bart(input_token['input_ids'],
                                      input_token['attention_mask'],
                                      inputs_embeds=None,
                                      beam_size=tmp_beam_size,
                                      max_seq_len=max_seq_len,
                                      top_k=topk,
                                      top_p=topp,
                                      beam_search_diversity_rate=beam_search_diversity_rate,
                                      is_return_output_log_probs=args_dict["return_output_log_probs"],
                                      is_return_cum_log_probs=args_dict["return_cum_log_probs"],)
                ft_output_ids = return_dict['output_ids']
                ft_sequence_length = return_dict['sequence_lengths']
                translation_result_list[i].batch_ids_list.append(ft_output_ids)
                translation_result_list[i].batch_seq_len_list.append(ft_sequence_length)
            
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
                    translation_result_list[i].token_num += sum(batch_token[j][:] != 0)
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
            LOGGER.info(f"{t.name} translates {t.batch_num} batches taking {t.execution_time:.2f} sec to translate "
                f"{t.token_num} tokens, BLEU score: {t.bleu_score.score:.2f}, {(t.token_num / t.execution_time):.0f} tokens/sec."
                f" ({t.bleu_score.sys_len} words, {(t.bleu_score.sys_len / t.execution_time):.0f} words/sec)")
            
            if t.name == "ft-beamsearch" and args_dict["ft_beamsearch_BLEU_threshold"] != None:
                print(t.bleu_score.score, args_dict["ft_beamsearch_BLEU_threshold"])
                assert t.bleu_score.score >= args_dict["ft_beamsearch_BLEU_threshold"], f"[ERROR] {t.name} test fail !"
                LOGGER.info(f"{t.name} PASS !")

            if t.name == "ft-sampling" and args_dict["ft_sampling_BLEU_threshold"] != None:
                print(t.bleu_score.score, args_dict["ft_sampling_BLEU_threshold"])
                assert t.bleu_score.score >= args_dict["ft_sampling_BLEU_threshold"], f"[ERROR] {t.name} test fail !"
                LOGGER.info(f"{t.name} PASS !")

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
                        help='deviersity rate of beam search. default is 0. When diversity rate = 0, it is equivalent to the naive beam search.')
    # parser.add_argument('-repeat_penalty', '--repetition_penalty', type=float, default=1.0, metavar='NUMBER',
    #                     help='Repetition penalty for generating tokens. Default is 1.0.')
    # parser.add_argument('-temperature', '--temperature', type=float, default=1.0, metavar='NUMBER',
    #                     help='Temperature penalty for generating tokens. Default is 1.0.')
    # parser.add_argument('-len_penalty', '--len_penalty', type=float, default=0.0, metavar='NUMBER',
    #                     help='Length penalty for generating tokens. Default is 0.0.')
    parser.add_argument('-topk', '--sampling_topk', type=int, default=1, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.0, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')
    parser.add_argument('-d', '--inference_data_type', type=str, default="fp32", metavar='STRING',
                        help='data type for inference (default: fp32)', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('-lib_path', '--lib_path', type=str, default="lib/libth_bart.so", metavar='STRING',
                        help='the path of FasterTransformer pytorch bart op library.')
    parser.add_argument('-model_path', '--model_path', type=str, default=None, metavar='STRING',
                        help='T5 model path.')
    parser.add_argument('-model', '--model', type=str, default="bart-base", metavar='STRING',
                        help='T5 model size. Only used when --model_path=None')
    parser.add_argument('-tensor_para_size', '--tensor_para_size', type=int, default=1, metavar='NUMBER',
                        help='size of tensor parallelism (default: 1)')
    parser.add_argument('-pipeline_para_size', '--pipeline_para_size', type=int, default=1, metavar='NUMBER',
                        help='size of pipeline parallelism (default: 1)')
    parser.add_argument('-max_ite', '--max_iteration', type=int, default=100000, metavar='NUMBER',
                        help='Maximum iteraiton for translation, default is 100000 (as large as possible to run all test set).')
    parser.add_argument('--return_output_log_probs', action='store_true',
                        help='Return the log probability of generated tokens.')
    parser.add_argument('--return_cum_log_probs', action='store_true',
                        help='Return the cumulative log probability of generated tokens.')
    parser.add_argument('--ft_beamsearch_BLEU_threshold', type=float,
                        help='Threshold of FT beam search BLEU score')
    parser.add_argument('--ft_sampling_BLEU_threshold', type=float,
                        help='Threshold of FT beam search BLEU score')
    parser.add_argument("--verbose", action="store_true", help="Provide verbose messages")
    parser.add_argument('--skip_gemm_test', action='store_true')
    args = parser.parse_args()
    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=log_format)
    translate(vars(args))