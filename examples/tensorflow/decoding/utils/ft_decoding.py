# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import tensorflow as tf
import os
import pickle 
import sys
from examples.tensorflow.decoder.utils.position import SinusoidalPositionEncoder

def finalize(beam_width, parent_ids, sequence_lengths, outputs, end_id, max_seq_len=None):
    maximum_lengths = tf.reduce_max(tf.reshape(
        sequence_lengths, [-1, beam_width]), axis=-1)
    
    if max_seq_len != None:
        array_shape = [max_seq_len, -1, beam_width]
    else:
        array_shape = [tf.reduce_max(maximum_lengths), -1, beam_width]
        
    step_ids = tf.reshape(outputs, array_shape)
    parent_ids = tf.reshape(parent_ids, array_shape)

    ids = tf.contrib.seq2seq.gather_tree(
        step_ids, parent_ids, maximum_lengths, end_id)

    ids = tf.transpose(ids, perm=[1, 2, 0])
    lengths = tf.not_equal(ids, end_id)
    lengths = tf.cast(lengths, tf.int32)
    lengths = tf.reduce_sum(lengths, axis=-1)
    return ids, lengths

def ft_decoding(memory_tensor,
                        memory_sequence_length,
                        embedding_table,
                        decoding_vars,
                        decoding_args,
                        using_model_var=True,
                        checkpoint_filename=None):
    '''
    Run the decoding with beam search by TensorFlow.
    
    Args:
        memory_tensor: A tf.tensor with shape [batch_size * beam_width, max(memory_sequence_length), encoder_hidden_dimension]. 
                       The results of encoder transformer layer. The rank must be 3. 
                       Note that it must be extended by beam_width times.
        memory_sequence_length: A tf.Tensor with shape [batch_size * beam_width], type tf.int. 
                                The lenght of each sentence of results of encoder. 
                                Note that it must be extended by beam_width times.
        embedding_table: A tf.Tensor with shape [vocab_size, hidden_dimension]. 
                         The embedding table of embedding lookup for each step.
        decoder_vars: A list of tf.Tensor. The variables for decoding. A list of model variables of TensorFlow model. 
        decoder_args: The arguments for decoding. The details are in the class "DecodingBeamsearchArgument" of common.py
        using_model_var: A bool value. Using the model variables of TensorFlow or not. 
                         The details are described in 'preprocess_decoder_var' function in the following.
        checkpoint_filename: A string. The checkpoint file name of storing the values of model.
                             The details are described in 'preprocess_decoder_var' function in the following.
    Outputs:
        finalized_output_ids: A tf.Tensor with shape [batch_size, beam_width, max(sequence_lengths)], with tf.int type. 
                                 Finalized output_ids by beam search algorithm and parent_ids.
        finalized_sequence_lengths: A tf.Tensor with shape [batch_size * beam_width], with int type.
                                       Finalized sequence_lengths by beam search algorithm and parent_ids.
        output_ids: A tf.Tensor with shape [batch_size, beam_width, max(sequence_lengths)], with tf.int type. 
                       The results of decoding. It contains the id of token of vocabulary.
        parent_ids: A tf.Tensor with shape [batch_size, beam_width, max(sequence_lengths)], with tf.int type.
                       The beam index of output ids for each step. 
        sequence_lengths: A tf.Tensor with shape [batch_size * beam_width], with int type.
    '''

    decoder_args = decoding_args.decoder_args
    decoding_op_module = tf.load_op_library(os.path.join('./lib/libtf_decoding.so'))
    
    extended_memory = tf.contrib.seq2seq.tile_batch(
        memory_tensor, multiplier=decoder_args.beam_width)
    extended_memory_sequence_length = tf.contrib.seq2seq.tile_batch(
        memory_sequence_length, multiplier=decoder_args.beam_width)
    
    position_encoder = SinusoidalPositionEncoder()
    position_encoding_table = position_encoder._create_position_encoding_table(
        decoding_args.max_seq_len, decoder_args.head_num * decoder_args.size_per_head, decoder_args.dtype)
    # shape of position_encoding_table: [max_seq_len, hidden_dim]

    cross_key_kernel_list = []
    cross_value_kernel_list = []
    cross_key_bias_list = []
    cross_value_bias_list = []
    
    var_dict = {}
    for v in decoding_vars:
        var_dict[v.name] = v
    
    for l in range(decoder_args.num_layer):
        layer_prefix_name = "transformer/decoder/layer_%d/" % l
        cross_key_kernel, cross_value_kernel = tf.split(var_dict[layer_prefix_name + 'multi_head/conv1d_1/kernel:0'], 2, axis=-1)
        cross_key_bias, cross_value_bias = tf.split(var_dict[layer_prefix_name + 'multi_head/conv1d_1/bias:0'], 2, axis=-1)
        
        cross_key_kernel_list.append(cross_key_kernel)
        cross_value_kernel_list.append(cross_value_kernel)
        cross_key_bias_list.append(cross_key_bias)
        cross_value_bias_list.append(cross_value_bias)

    output_ids, parent_ids, sequence_lengths = decoding_op_module.decoding(
        extended_memory, # 1
        extended_memory_sequence_length, # 2
        [var_dict["transformer/decoder/layer_%d/masked_multi_head/LayerNorm/beta:0" % l] for l in range(decoder_args.num_layer)], # 7
        [var_dict["transformer/decoder/layer_%d/masked_multi_head/LayerNorm/gamma:0" % l] for l in range(decoder_args.num_layer)], # 8
        [var_dict["transformer/decoder/layer_%d/masked_multi_head/conv1d/kernel:0" % l] for l in range(decoder_args.num_layer)], # 9
        [var_dict["transformer/decoder/layer_%d/masked_multi_head/conv1d/bias:0" % l] for l in range(decoder_args.num_layer)], # 10
        [var_dict["transformer/decoder/layer_%d/masked_multi_head/conv1d_1/kernel:0" % l] for l in range(decoder_args.num_layer)], # 11
        [var_dict["transformer/decoder/layer_%d/masked_multi_head/conv1d_1/bias:0" % l] for l in range(decoder_args.num_layer)],  # 12
        [var_dict["transformer/decoder/layer_%d/multi_head/LayerNorm/beta:0" % l] for l in range(decoder_args.num_layer)], # 13
        [var_dict["transformer/decoder/layer_%d/multi_head/LayerNorm/gamma:0" % l] for l in range(decoder_args.num_layer)], # 14
        [var_dict["transformer/decoder/layer_%d/multi_head/conv1d/kernel:0" % l] for l in range(decoder_args.num_layer)], # 15
        [var_dict["transformer/decoder/layer_%d/multi_head/conv1d/bias:0" % l] for l in range(decoder_args.num_layer)], # 16
        cross_key_kernel_list, # 17
        cross_key_bias_list, # 18
        cross_value_kernel_list, # 19
        cross_value_bias_list, # 20
        [var_dict["transformer/decoder/layer_%d/multi_head/conv1d_2/kernel:0" % l] for l in range(decoder_args.num_layer)], # 21
        [var_dict["transformer/decoder/layer_%d/multi_head/conv1d_2/bias:0" % l] for l in range(decoder_args.num_layer)], # 22
        [var_dict["transformer/decoder/layer_%d/ffn/LayerNorm/beta:0" % l] for l in range(decoder_args.num_layer)], # 23
        [var_dict["transformer/decoder/layer_%d/ffn/LayerNorm/gamma:0" % l] for l in range(decoder_args.num_layer)], # 24
        [var_dict["transformer/decoder/layer_%d/ffn/conv1d/kernel:0" % l] for l in range(decoder_args.num_layer)], # 25
        [var_dict["transformer/decoder/layer_%d/ffn/conv1d/bias:0" % l] for l in range(decoder_args.num_layer)], # 26
        [var_dict["transformer/decoder/layer_%d/ffn/conv1d_1/kernel:0" % l] for l in range(decoder_args.num_layer)], # 27
        [var_dict["transformer/decoder/layer_%d/ffn/conv1d_1/bias:0" % l] for l in range(decoder_args.num_layer)], # 28
        var_dict['transformer/decoder/LayerNorm/beta:0'], # 28
        var_dict['transformer/decoder/LayerNorm/gamma:0'], # 29
        position_encoding_table, # 33 
        embedding_table, # 30
        var_dict['transformer/decoder/dense/kernel:0'], # 31
        var_dict['transformer/decoder/dense/bias:0'], # 32
        max_seq_len=decoding_args.max_seq_len,
        beam_width=decoder_args.beam_width,
        head_num=decoder_args.head_num, 
        size_per_head=decoder_args.size_per_head,
        inter_size=decoder_args.inter_size,
        num_layer=decoder_args.num_layer,
        start_id=decoding_args.start_id, 
        end_id=decoding_args.end_id,
        beam_search_diversity_rate=decoding_args.beam_search_diversity_rate,
        top_k=decoding_args.top_k,
        top_p=decoding_args.top_p,
        temperature=1.0,
        len_penalty=1.0,
        repetition_penalty=1.0)

    if decoder_args.beam_width > 1:
        output_ids = tf.transpose(output_ids, [1, 2, 0])
        # TODO(bhsueh) Remove useless outputs
        return output_ids, sequence_lengths, None, None, None
    else:
        output_ids = tf.transpose(output_ids, [1, 0])
        
        return output_ids, sequence_lengths, None, None, None