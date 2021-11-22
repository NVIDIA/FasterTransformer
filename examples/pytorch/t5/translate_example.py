# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from datetime import datetime
import numpy as np
import torch
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
    
    print("\n=============== Argument ===============")
    for key in args_dict:
        print("{}: {}".format(key, args_dict[key]))
    print("========================================")

    lib_path = args_dict['lib_path']

    t5_model = T5ForConditionalGeneration.from_pretrained(args_dict['model'])
    
    if time_args.find("0") != -1 or time_args.find("2") != -1:
        t5_model = t5_model.to("cuda")
        if args_dict['data_type'] == 'fp16':
            t5_model = t5_model.half()
    tokenizer = T5Tokenizer.from_pretrained(args_dict['model'])
    fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(args_dict['model'])

    if time_args.find("1") != -1 or time_args.find("3") != -1:
        encoder_config = t5_model.encoder.config
        decoder_config = t5_model.decoder.config

        ft_encoder_weight = FTT5EncoderWeight(encoder_config, tensor_para_size, pipeline_para_size)
        ft_decoding_weight = FTT5DecodingWeight(decoder_config, tensor_para_size, pipeline_para_size)
    
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

        # Set remove padding = False since we don't support remove padding under pipeline parallel now
        remove_padding = True
        ft_encoder = FTT5Encoder(ft_encoder_weight.w, lib_path, encoder_config.num_heads,
                                encoder_config.d_kv, encoder_config.d_ff,
                                encoder_config.d_model, remove_padding, encoder_config.num_layers,
                                encoder_config.relative_attention_num_buckets,
                                128, False, 1.0, tensor_para_size, pipeline_para_size)

        ft_decoding = FTT5Decoding(ft_decoding_weight.w, lib_path,
                                decoder_config.num_heads, decoder_config.d_kv,
                                decoder_config.d_ff, encoder_config.d_model,
                                decoder_config.d_model, decoder_config.num_layers,
                                decoder_config.decoder_start_token_id, decoder_config.eos_token_id,
                                decoder_config.vocab_size,
                                decoder_config.relative_attention_num_buckets, max_distance=128,
                                beam_search_diversity_rate=beam_search_diversity_rate, top_k=topk, top_p=topp,
                                temperature=1.0, len_penalty=1.0, repetition_penalty=1.0,
                                tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size)

        ft_t5 = FTT5(ft_encoder, ft_decoding)

    with open(source_file, 'r') as f:
        src_text = recover_bpe(f.readlines())
        src_text = ["translate English to German: " + line.strip() for line in src_text]

    with open(tgt_file, 'r') as f:
        tgt_text = recover_bpe(f.readlines())

    translation_result_list = []
    if time_args.find("0") != -1:
        translation_result_list.append(TranslationResult("hf-beamsearch-warmup", "HF"))
        translation_result_list.append(TranslationResult("hf-beamsearch", "HF"))
    if time_args.find("1") != -1:
        translation_result_list.append(TranslationResult("ft-beamsearch-warmup", "FT"))
        translation_result_list.append(TranslationResult("ft-beamsearch", "FT"))
    if time_args.find("2") != -1:
        translation_result_list.append(TranslationResult("hf-sampling-warmup", "HF"))
        translation_result_list.append(TranslationResult("hf-sampling", "HF"))
    if time_args.find("3") != -1:
        translation_result_list.append(TranslationResult("ft-sampling-warmup", "FT"))
        translation_result_list.append(TranslationResult("ft-sampling", "FT"))

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
                ft_decoding_outputs, ft_decoding_seq_lens = ft_t5(input_token, tmp_beam_size, max_seq_len)
                translation_result_list[i].batch_ids_list.append(ft_decoding_outputs)
                translation_result_list[i].batch_seq_len_list.append(ft_decoding_seq_lens)
            
            translation_result_list[i].sentence_num += len(input_token)
            translation_result_list[i].batch_num += 1
            if translation_result_list[i].name.find("warmup") != -1 and \
                (translation_result_list[i].batch_num > 10 or translation_result_list[i].sentence_num > 300):
                break

        stop_time = datetime.now()
        translation_result_list[i].execution_time = (stop_time - start_time).total_seconds()
        
        for batch_token, batch_seq_len in zip(translation_result_list[i].batch_ids_list, translation_result_list[i].batch_seq_len_list):
            for j in range(len(batch_token)):
                if translation_result_list[i].frame_work == "HF":
                    translation_result_list[i].token_list.append(fast_tokenizer.decode(batch_token[j][1:], skip_special_tokens=True))
                elif translation_result_list[i].frame_work == "FT":
                    translation_result_list[i].token_list.append(fast_tokenizer.decode(batch_token[j][0][:batch_seq_len[j][0]], skip_special_tokens=True))

        translation_result_list[i].bleu_score = bleu_score(translation_result_list[i].token_list, tgt_text[:len(translation_result_list[i].token_list)])
        with open(translation_result_list[i].name + ".txt", 'w') as f:
            for line in translation_result_list[i].token_list:
                f.write(line)
    
    for t in translation_result_list:
        if t.name.find("warmup") != -1: 
            continue
        print("[INFO] {} translates {} batches taking {:.2f} sec to translate {} tokens, BLEU score: {:.2f}, {:.0f} tokens/sec.".format(
                t.name, t.batch_num, t.execution_time, t.bleu_score.sys_len, t.bleu_score.score, t.bleu_score.sys_len / t.execution_time))

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
    parser.add_argument('-model', '--model', type=str, default="t5-small", metavar='STRING',
                        help='T5 model size.', choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"])
    parser.add_argument('-tensor_para_size', '--tensor_para_size', type=int, default=1, metavar='NUMBER',
                        help='size of tensor parallelism (default: 1)')
    parser.add_argument('-pipeline_para_size', '--pipeline_para_size', type=int, default=1, metavar='NUMBER',
                        help='size of pipeline parallelism (default: 1)')
    parser.add_argument('--ckpt_path', type=str, help='path to the checkpoint file.')
    args = parser.parse_args()

    translate(vars(args))