"""
Modified From https://github.com/OpenNMT/OpenNMT-tf/blob/r1/examples/library/minimal_transformer_training.py

MIT License

Copyright (c) 2017-present The OpenNMT Authors.

This example demonstrates how to train a standard Transformer model using
OpenNMT-tf as a library in about 200 lines of code. While relatively short,
this example contains some advanced concepts such as dataset bucketing and
prefetching, token-based batching, gradients accumulation, beam search, etc.
Currently, the beam search part is not easily customizable. This is expected to
be improved for TensorFlow 2.0 which is eager first.

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

"""

# Use opennmt-tf-1.25.1

import argparse
import copy
from datetime import datetime
import numpy as np
import os
import sys

import tensorflow as tf
import opennmt as onmt

from opennmt import constants
from opennmt.utils import misc

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.tensorflow.decoding.utils.ft_decoding import ft_decoding
from examples.tensorflow.decoding.utils.bleu_score import bleu_score
from examples.tensorflow.decoder.utils.decoding import tf_sampling_decoding
from examples.tensorflow.decoder.utils.decoding import tf_beamsearch_decoding
from examples.tensorflow.decoder.utils.common import DecodingArgumentNew
from examples.tensorflow.decoder.utils.common import TransformerArgument
from examples.tensorflow.decoder.utils.common import DecodingSamplingArgument
from examples.tensorflow.decoder.utils.common import DecodingBeamsearchArgument
from examples.tensorflow.encoder.utils.encoder import ft_encoder_opennmt
from examples.tensorflow.encoder.utils.encoder import tf_encoder_opennmt

NUM_HEADS = 8
NUM_LAYERS = 6
HIDDEN_UNITS = 512
SIZE_PER_HEAD = 64
FFN_INNER_DIM = 2048

encoder = onmt.encoders.SelfAttentionEncoder(
    num_layers=NUM_LAYERS,
    num_units=HIDDEN_UNITS,
    num_heads=NUM_HEADS,
    ffn_inner_dim=FFN_INNER_DIM,
    dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1)
decoder = onmt.decoders.SelfAttentionDecoder(
    num_layers=NUM_LAYERS,
    num_units=HIDDEN_UNITS,
    num_heads=NUM_HEADS,
    ffn_inner_dim=FFN_INNER_DIM,
    dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1)

def translate(args_dict):

    batch_size = args_dict['batch_size']
    beam_size = args_dict['beam_width']
    max_seq_len = args_dict['max_seq_len']
    model_dir = args_dict["model_dir"]
    source_file = args_dict["source"]
    tgt_file = args_dict["target"]
    time_args = args_dict["test_time"]
    beam_search_diversity_rate = args_dict['beam_search_diversity_rate']
    sampling_topk = args_dict['sampling_topk']
    sampling_topp = args_dict['sampling_topp']
    tf_datatype = tf.float32
    max_ite = args_dict['max_iteration']
    if args_dict['data_type'] == "fp16":
        tf_datatype = tf.float16
    
    print("\n=============== Argument ===============")
    for key in args_dict:
        print("{}: {}".format(key, args_dict[key]))
    print("========================================")

    # Define the "base" Transformer model.
    source_inputter = onmt.inputters.WordEmbedder("source_vocabulary", embedding_size=512, dtype=tf_datatype)
    target_inputter = onmt.inputters.WordEmbedder("target_vocabulary", embedding_size=512, dtype=tf_datatype)

    inputter = onmt.inputters.ExampleInputter(source_inputter, target_inputter)
    inputter.initialize({
        "source_vocabulary": args_dict["source_vocabulary"],
        "target_vocabulary": args_dict["target_vocabulary"]
    })

    mode = tf.estimator.ModeKeys.PREDICT

    np.random.seed(1)
    tf.set_random_seed(1)

    # Create the inference dataset.
    dataset = inputter.make_inference_dataset(source_file, batch_size)
    iterator = dataset.make_initializable_iterator()
    source = iterator.get_next()

    encoder_args = TransformerArgument(beam_width=1,
                                       head_num=NUM_HEADS,
                                       size_per_head=SIZE_PER_HEAD,
                                       inter_size=NUM_HEADS*SIZE_PER_HEAD*4,
                                       num_layer=NUM_LAYERS,
                                       dtype=tf_datatype,
                                       remove_padding=True,
                                       allow_gemm_test=False)

    # Encode the source.
    with tf.variable_scope("transformer/encoder"):
        source_embedding = source_inputter.make_inputs(source)
        source_embedding = tf.cast(source_embedding, tf_datatype)

        # Using onmt fp16 for encoder.encode leads to significant accuracy drop
        # So, we rewrite the encoder
        # memory, _, _ = encoder.encode(source_embedding, source["length"], mode=mode)
        memory = tf_encoder_opennmt(source_embedding, encoder_args, source["length"])

        encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        encoder_variables_dict = {}
        for v in encoder_vars:
            encoder_variables_dict[v.name] = tf.cast(v, tf_datatype)
        ft_encoder_result = ft_encoder_opennmt(inputs=source_embedding,
                                               encoder_args=encoder_args,
                                               encoder_vars_dict=encoder_variables_dict,
                                               sequence_length=source["length"])

    # Generate the target.
    with tf.variable_scope("transformer/decoder", reuse=tf.AUTO_REUSE):
        target_inputter.build()
        batch_size = tf.shape(memory)[0]
        start_tokens = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
        end_token = constants.END_OF_SENTENCE_ID
        target_embedding = tf.cast(target_inputter.embedding, tf_datatype)
        target_ids, _, target_length, _ = decoder.dynamic_decode_and_search(
            target_embedding,
            start_tokens,
            end_token,
            vocab_size=target_inputter.vocabulary_size,
            beam_width=beam_size,
            memory=memory,
            memory_sequence_length=source["length"],
            maximum_iterations=max_seq_len)
        target_vocab_rev = target_inputter.vocabulary_lookup_reverse()
        target_tokens = target_vocab_rev.lookup(tf.cast(target_ids, tf.int64))

        decoder_args = TransformerArgument(beam_width=beam_size,
                                           head_num=NUM_HEADS,
                                           size_per_head=SIZE_PER_HEAD,
                                           inter_size=NUM_HEADS*SIZE_PER_HEAD*4,
                                           num_layer=NUM_LAYERS,
                                           dtype=tf_datatype,
                                           kernel_init_range=0.00,
                                           bias_init_range=0.00)

        decoder_args_2 = copy.deepcopy(decoder_args)  # for beam search
        decoder_args_2.__dict__ = copy.deepcopy(decoder_args.__dict__)
        decoder_args_2.beam_width = 1  # for sampling

        ft_decoder_beamsearch_args = DecodingBeamsearchArgument(target_inputter.vocabulary_size,
                                                                constants.START_OF_SENTENCE_ID,
                                                                constants.END_OF_SENTENCE_ID,
                                                                max_seq_len,
                                                                decoder_args,
                                                                beam_search_diversity_rate)

        ft_decoder_sampling_args = DecodingSamplingArgument(target_inputter.vocabulary_size,
                                                            constants.START_OF_SENTENCE_ID,
                                                            constants.END_OF_SENTENCE_ID,
                                                            max_seq_len,
                                                            decoder_args_2,
                                                            sampling_topk,
                                                            sampling_topp)

        decoding_beamsearch_args = DecodingArgumentNew(target_inputter.vocabulary_size,
                                                       constants.START_OF_SENTENCE_ID,
                                                       constants.END_OF_SENTENCE_ID,
                                                       max_seq_len,
                                                       beam_search_diversity_rate,
                                                       0,
                                                       0.0,
                                                       decoder_args)

        decoding_sampling_args = DecodingArgumentNew(target_inputter.vocabulary_size,
                                                     constants.START_OF_SENTENCE_ID,
                                                     constants.END_OF_SENTENCE_ID,
                                                     max_seq_len,
                                                     0.0,
                                                     sampling_topk,
                                                     sampling_topp,
                                                     decoder_args_2)

        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        ft_target_ids, ft_target_length, _, _, _ = ft_decoding(ft_encoder_result,
                                                               source["length"],
                                                               target_embedding,
                                                               all_vars,
                                                               decoding_beamsearch_args)
        ft_target_tokens = target_vocab_rev.lookup(tf.cast(ft_target_ids, tf.int64))

        ft_sampling_target_ids, ft_sampling_target_length, _, _, _ = ft_decoding(ft_encoder_result,
                                                                                 source["length"],
                                                                                 target_embedding,
                                                                                 all_vars,
                                                                                 decoding_sampling_args)
        ft_sampling_target_tokens = target_vocab_rev.lookup(tf.cast(ft_sampling_target_ids, tf.int64))

    # ### TF Sampling Decoding ###
    tf_sampling_target_ids, tf_sampling_target_length = tf_sampling_decoding(memory,
                                                                             source["length"],
                                                                             target_embedding,
                                                                             ft_decoder_sampling_args,
                                                                             decoder_type=0)

    # tf_sampling_target_tokens: [batch_size, seq_len]
    tf_sampling_target_tokens = target_vocab_rev.lookup(tf.cast(tf_sampling_target_ids, tf.int64))
    # ### end of TF BeamSearch Decoding ###

    ### OP BeamSearch Decoder ###
    ft_decoder_beamsearch_target_ids, ft_decoder_beamsearch_target_length, _, _, _ = tf_beamsearch_decoding(memory,
                                                                                                            source["length"],
                                                                                                            target_embedding,
                                                                                                            ft_decoder_beamsearch_args,
                                                                                                            decoder_type=1)

    # ft_decoder_beamsearch_target_tokens: [batch_size, beam_width, seq_len]
    ft_decoder_beamsearch_target_tokens = target_vocab_rev.lookup(tf.cast(ft_decoder_beamsearch_target_ids, tf.int64))
    ### end of OP BeamSearch Decoder ###

    ### OP Sampling Decoder ###
    ft_decoder_sampling_target_ids, ft_decoder_sampling_target_length = tf_sampling_decoding(memory,
                                                                                             source["length"],
                                                                                             target_embedding,
                                                                                             ft_decoder_sampling_args,
                                                                                             decoder_type=1)

    ft_decoder_sampling_target_tokens = target_vocab_rev.lookup(tf.cast(ft_decoder_sampling_target_ids, tf.int64))
    ### end of OP BeamSearch Decoder ###

    class TranslationResult(object):
        def __init__(self, token_op, length_op, name):
            self.token_op = token_op
            self.length_op = length_op
            self.name = name
            self.file_name = name + ".txt"

            self.token_list = []
            self.length_list = []
            self.batch_num = 0
            self.execution_time = 0.0  # seconds
            self.sentence_num = 0
            self.bleu_score = None

    translation_result_list = []

    if time_args != "":
        translation_result_list.append(TranslationResult(
            tf_sampling_target_tokens, tf_sampling_target_length, "tf-decoding-sampling-for-warmup"))
    if time_args.find("0") != -1:
        translation_result_list.append(TranslationResult(
            target_tokens, target_length, "tf-decoding-beamsearch"))
    if time_args.find("1") != -1:
        translation_result_list.append(TranslationResult(
            ft_decoder_beamsearch_target_tokens, ft_decoder_beamsearch_target_length, "ft-decoder-beamsearch"))
    if time_args.find("2") != -1:
        translation_result_list.append(TranslationResult(
            ft_target_tokens, ft_target_length, "ft-decoding-beamsearch"))
    if time_args.find("3") != -1:
        translation_result_list.append(TranslationResult(
            tf_sampling_target_tokens, tf_sampling_target_length, "tf-decoding-sampling"))
    if time_args.find("4") != -1:
        translation_result_list.append(TranslationResult(
            ft_decoder_sampling_target_tokens, ft_decoder_sampling_target_length, "ft-decoder-sampling"))
    if time_args.find("5") != -1:
        translation_result_list.append(TranslationResult(
            ft_sampling_target_tokens, ft_sampling_target_length, "ft-decoding-sampling"))

    # Iterates on the dataset.
    float_checkpoint_path = tf.train.latest_checkpoint(model_dir)
    half_checkpoint_path = tf.train.latest_checkpoint(model_dir + "_fp16")

    float_var_list = []
    half_var_list = []
    for var in tf.global_variables():
        if var.dtype.base_dtype == tf.float32:
            float_var_list.append(var)
        elif var.dtype.base_dtype == tf.float16:
            half_var_list.append(var)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    for i in range(len(translation_result_list)):
        with tf.Session(config=config) as sess:
            if(len(float_var_list) > 0):
                float_saver = tf.train.Saver(float_var_list)
                float_saver.restore(sess, float_checkpoint_path)
            if(len(half_var_list) > 0):
                half_saver = tf.train.Saver(half_var_list)
                half_saver.restore(sess, half_checkpoint_path)

            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)

            t1 = datetime.now()
            while True:
                try:
                    batch_tokens, batch_length = sess.run([translation_result_list[i].token_op,
                                                           translation_result_list[i].length_op])
                    for tokens, length in zip(batch_tokens, batch_length):
                        # misc.print_bytes(b" ".join(tokens[0][:length[0] - 1]))
                        if translation_result_list[i].name.find("beamsearch") != -1:
                            translation_result_list[i].token_list.append(
                                b" ".join(tokens[0][:length[0] - 1]).decode("UTF-8"))
                        else:
                            translation_result_list[i].token_list.append(b" ".join(tokens[:length - 1]).decode("UTF-8"))

                    translation_result_list[i].batch_num += 1

                    if translation_result_list[i].name == "tf-decoding-sampling-for-warmup" and translation_result_list[i].batch_num > 20:
                        break
                    if translation_result_list[i].batch_num >= max_ite: 
                        break
                except tf.errors.OutOfRangeError:
                    break
            t2 = datetime.now()
            time_sum = (t2 - t1).total_seconds()
            translation_result_list[i].execution_time = time_sum

            with open(translation_result_list[i].file_name, "w") as file_b:
                for s in translation_result_list[i].token_list:
                    file_b.write(s)
                    file_b.write("\n")

            ref_file_path = "./.ref_file.txt"
            os.system("head -n %d %s > %s" % (len(translation_result_list[i].token_list), tgt_file, ref_file_path))
            translation_result_list[i].bleu_score = bleu_score(translation_result_list[i].file_name, ref_file_path)
            os.system("rm {}".format(ref_file_path))

    for t in translation_result_list:
        if t.name == "tf-decoding-sampling-for-warmup":
            continue
        print("[INFO] {} translates {} batches taking {:.2f} sec to translate {} tokens, BLEU score: {:.2f}, {:.0f} tokens/sec.".format(
            t.name, t.batch_num, t.execution_time, t.bleu_score.sys_len, t.bleu_score.score, t.bleu_score.sys_len / t.execution_time))

    return translation_result_list

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-beam', '--beam_width', type=int, default=4, metavar='NUMBER',
                        help='beam width (default: 4)')
    parser.add_argument('-s', '--max_seq_len', type=int, default=200, metavar='NUMBER',
                        help='max sequence length (default: 200)')
    parser.add_argument("--source", default="../examples/tensorflow/decoding/utils/translation/test.en",
                        help="Path to the source file.")
    parser.add_argument("--target", default="../examples/tensorflow/decoding/utils/translation/test.de",
                        help="Path to the target file.")
    parser.add_argument("--source_vocabulary", default="../examples/tensorflow/decoding/utils/translation/wmtende.vocab",
                        help="Path to the source vocabulary.")
    parser.add_argument("--target_vocabulary", default="../examples/tensorflow/decoding/utils/translation/wmtende.vocab",
                        help="Path to the target vocabulary.")
    parser.add_argument("--model_dir", default="../translation/ckpt",
                        help="Directory where checkpoint are written.")
    parser.add_argument('-time', '--test_time', type=str, default='', metavar='STRING',
                        help='''
                            Test the time of which one (default: '' (not test anyone) ); 
                            '': not test anyone 
                            '0': test tf_decoding_beamsearch  
                            '1': test op_decoder_beamsearch 
                            '2': test op_decoding_beamsearch 
                            '3': test tf_decoding_sampling 
                            '4': test op_decoder_sampling 
                            '5': test op_decoding_sampling 
                            'e.g., if you want to test op_decoder_beamsearch and op_decoding_sampling, 
                            then you need to use -time '15' ''')
    parser.add_argument('-diversity_rate', '--beam_search_diversity_rate', type=float, default=0.0, metavar='NUMBER',
                        help='deviersity rate of beam search. default is 0. When diversity rate = 0, it is equivalent to the naive beams earch.')
    parser.add_argument('-topk', '--sampling_topk', type=int, default=1, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.0, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type (default: fp32)', choices=['fp32', 'fp16'])
    parser.add_argument('-max_ite', '--max_iteration', type=int, default=100000, metavar='NUMBER',
                        help='Maximum iteraiton for translation, default is 100000 (as large as possible to run all test set).')
    args = parser.parse_args()
    translate(vars(args))

# example script
# python ../examples/tensorflow/decoding/translate_example.py --source ../examples/tensorflow/decoding/utils/translation/test.en --target ../examples/tensorflow/decoding/utils/translation/test.de --source_vocabulary ../examples/tensorflow/decoding/utils/translation/wmtende.vocab --target_vocabulary ../examples/tensorflow/decoding/utils/translation/wmtende.vocab --model_dir ../translation/ckpt/ -time 02


if __name__ == "__main__":
    main()
