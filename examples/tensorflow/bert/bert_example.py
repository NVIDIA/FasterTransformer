# Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

'''
This is a sample code to demonstrate how to use the TensorFlow custom op with 
FasterTransformer library in encoder.

This sample code builds a BERT transformer model by TensorFlow and TensorFlow 
custom op. Then compare the maximum difference of them to verify the correctness
of FasterTransformer. 

Users are also able to use this sample code to test the average forward time of 
TensorFlow and FasterTransformer. 
'''

import argparse
import copy
import numpy as np
import tensorflow as tf
import threading
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.tensorflow.bert.utils.common import TransformerArgument
from examples.tensorflow.bert.utils.common import time_test
from examples.tensorflow.bert.utils.common import cross_check
from examples.tensorflow.bert.utils.bert import tf_bert
from examples.tensorflow.bert.utils.bert import ft_bert
from examples.tensorflow.bert.utils.bert import build_sequence_mask

def bert_example(args_dict):
    print("\n=============== Argument ===============")
    for key in args_dict:
        print("{}: {}".format(key, args_dict[key]))
    print("========================================")

    np.random.seed(1)
    tf.set_random_seed(1)

    batch_size = args_dict['batch_size']
    num_layer = args_dict['num_layer']
    max_seq_len = args_dict['max_seq_len']
    avg_seq_len = args_dict['avg_seq_len']
    head_num = args_dict['head_number']
    size_per_head = args_dict['size_per_head']
    inter_size = args_dict['inter_size']
    if inter_size == 0:
        inter_size = head_num * size_per_head * 4
    tf_datatype = tf.float32
    np_datatype = np.float32
    atol_threshold = 3e-5
    int8_mode = args_dict['int8_mode']
    if args_dict['data_type'] == "fp16":
        tf_datatype = tf.float16
        np_datatype = np.float16
        atol_threshold = 3e-2

    hidden_dim = head_num * size_per_head

    sequence_length = np.random.randint(1, max_seq_len + 1, size=batch_size)
    if avg_seq_len != -1:
        # This means we use "remove_padding" and set other average sequence length
        sequence_length = np.ones(batch_size) * avg_seq_len
    else:
        sequence_length = np.ones(batch_size) * (max_seq_len / 2)
    sequence_length = sequence_length.astype(np.int32)

    from_data = np.random.randn(batch_size, max_seq_len, hidden_dim)
    from_tensor = tf.convert_to_tensor(from_data, dtype=tf_datatype)
    
    attention_mask = build_sequence_mask(sequence_length, num_heads=head_num, maximum_length=max_seq_len, dtype=tf_datatype)
    
    encoder_args = TransformerArgument(beam_width=1,
                                       head_num=head_num,
                                       size_per_head=size_per_head,
                                       inter_size=inter_size,
                                       num_layer=num_layer,
                                       dtype=tf_datatype,
                                       int8_mode=int8_mode,
                                       remove_padding=False)

    eff_encoder_args = copy.deepcopy(encoder_args)
    eff_encoder_args.remove_padding = True

    tf_encoder_result = tf_bert(input_tensor=from_tensor,
                                   encoder_args=encoder_args,
                                   attention_mask=attention_mask)

    encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    encoder_variables_dict = {}
    for v in encoder_vars:
        encoder_variables_dict[v.name] = v
    
    op_encoder_result = ft_bert(inputs=from_tensor,
                                encoder_args=encoder_args,
                                encoder_vars_dict=encoder_variables_dict,
                                sequence_length=sequence_length)

    eff_encoder_result = ft_bert(inputs=from_tensor,
                                encoder_args=eff_encoder_args,
                                encoder_vars_dict=encoder_variables_dict,
                                sequence_length=sequence_length)

    '''
    Because FasterTransformer skip some computation for the padding parts, 
    if we do not mask these parts, the cross check result would be wrong. 
    '''
    # Prevent nan since we will skip to write the data to some position, and these positions may be dirty.
    eff_encoder_result = tf.where(tf.is_nan(eff_encoder_result), tf.zeros_like(eff_encoder_result), eff_encoder_result)
    
    tf_encoder_result = tf_encoder_result * tf.expand_dims(tf.sequence_mask(sequence_length, maxlen=max_seq_len, dtype=tf_datatype), axis=-1)
    op_encoder_result = op_encoder_result * tf.expand_dims(tf.sequence_mask(sequence_length, maxlen=max_seq_len, dtype=tf_datatype), axis=-1)
    eff_encoder_result = eff_encoder_result * tf.expand_dims(tf.sequence_mask(sequence_length, maxlen=max_seq_len, dtype=tf_datatype), axis=-1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for idx, name in enumerate(encoder_variables_dict):
            print((str(idx) + " " + str(name) + " " +
                   str(encoder_variables_dict[name].shape)) + " " + str(encoder_variables_dict[name].dtype))
        
        print("#################################")
        tf_encoder_result_val = sess.run(tf_encoder_result)
        op_encoder_result_val = sess.run(op_encoder_result)
        eff_encoder_result_val = sess.run(eff_encoder_result)

        cross_check("Encoder TF v.s. FT with tensor input", tf_encoder_result_val, op_encoder_result_val, atol_threshold)
        cross_check("Encoder TF v.s. EFF-FT with tensor input", tf_encoder_result_val, eff_encoder_result_val, atol_threshold)
        
        op_diff = abs(tf_encoder_result_val.reshape([-1]) - op_encoder_result_val.reshape([-1]))
        eff_diff = abs(tf_encoder_result_val.reshape([-1]) - eff_encoder_result_val.reshape([-1]))
        max_diff = max(op_diff.max(), eff_diff.max())
        max_diff = op_diff.max()

        ite = 50
        def _cond(from_tensor):
            return tf.constant(True)
            
        def _ft_body(from_tensor):
            op_encoder_result = ft_bert(inputs=from_tensor,
                                            encoder_args=encoder_args,
                                            encoder_vars_dict=encoder_variables_dict,
                                            sequence_length=sequence_length)
            return op_encoder_result

        def _eff_body(from_tensor):
            eff_encoder_result = ft_bert(inputs=from_tensor,
                                            encoder_args=eff_encoder_args,
                                            encoder_vars_dict=encoder_variables_dict,
                                            sequence_length=sequence_length)
            return eff_encoder_result

        def _tf_body(from_tensor):
            tf_encoder_result = tf_bert(input_tensor=from_tensor,
                                            encoder_args=encoder_args,
                                            attention_mask=attention_mask)
            return tf_encoder_result

        tf_while_tensor = tf.while_loop(_cond,
                                        _tf_body,
                                        loop_vars=[from_tensor],
                                        back_prop=False,
                                        maximum_iterations=ite)

        ft_while_tensor = tf.while_loop(_cond,
                                        _ft_body,
                                        loop_vars=[from_tensor],
                                        back_prop=False,
                                        maximum_iterations=ite)

        eff_while_tensor = tf.while_loop(_cond,
                                         _eff_body,
                                         loop_vars=[from_tensor],
                                         back_prop=False,
                                         maximum_iterations=ite)

        if args_dict['test_time'] == 1:

            # Using while loop to run 'ite' times to ignore the overheads of memory copy and model preprocess.
            # We use these times as the profiling results.
            tf_while_time = time_test(sess, tf_while_tensor, 1) / ite # while_loop has run ite times
            # time.sleep(60)
            ft_while_time = time_test(sess, ft_while_tensor, 1) / ite # while_loop has run ite times
            # time.sleep(60)
            eff_while_time = time_test(sess, eff_while_tensor, 1) / ite # while_loop has run ite times
            # time.sleep(60)
            ft_type = args_dict['data_type'].upper()
            
            print("[INFO] batch_size {} max_seq_len {} precision {} {} layer TF-while-time     {:6.2f} ms ( {} iterations)".format(batch_size, max_seq_len, args_dict['data_type'].upper(), num_layer, tf_while_time, ite))
            print("[INFO] batch_size {} max_seq_len {} precision {} {} layer FT-OP-while-time  {:6.2f} ms ( {} iterations)".format(batch_size, max_seq_len, ft_type, num_layer, ft_while_time, ite))
            print("[INFO] batch_size {} max_seq_len {} precision {} {} layer EFF-OP-while-time {:6.2f} ms ( {} iterations)".format(batch_size, max_seq_len, ft_type, num_layer, eff_while_time, ite))


        if args_dict['thread_num'] > 1:
            # Multi-threading demonstration
            thread_list = []
            thread_num = args_dict['thread_num']
            def run():
                ft_while_time = time_test(sess, ft_while_tensor, 1) / ite # while_loop has run ite times
                print("[INFO] batch_size {} max_seq_len {} {} layer FT-OP-while-time {:6.2f} ms with {} threads".format(batch_size,
                    max_seq_len, num_layer, ft_while_time, thread_num))

            for i in range(thread_num):
                thread_list.append(threading.Thread(target=run, name="RunFT"))
            for t in thread_list:
                t.start()
            for t in thread_list:
                t.join()

    sys.stdout.flush()
    return max_diff

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', '--batch_size', type=int, default=4, metavar='NUMBER',
                        help='batch size (default: 4)')
    parser.add_argument('-l', '--num_layer', type=int, default=12, metavar='NUMBER',
                        help='number of layers (default: 12)')
    parser.add_argument('-s', '--max_seq_len', type=int, default=32, metavar='NUMBER',
                        help='max sequence length (default: 32)')
    parser.add_argument('-n', '--head_number', type=int, default=12, metavar='NUMBER',
                        help='head number (default: 12)')
    parser.add_argument('-size', '--size_per_head', type=int, default=64, metavar='NUMBER',
                        help='size per head (default: 64)')
    parser.add_argument('-inter_size', '--inter_size', type=int, default=0, metavar='NUMBER',
                        help='inter_size (default: 0)')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type (default: fp32)', choices=['fp32', 'fp16'])
    parser.add_argument('-time', '--test_time', type=int, default=0, metavar='BOOL',
                        help='test the time or not. (default: False (0)), True is 1.',
                        choices=[0, 1])
    parser.add_argument('-int8', '--int8_mode', type=int, default=0, metavar='NUMBER',
                        help='int8 mode. (default: 0)',
                        choices=[0, 1, 2, 3])
    parser.add_argument('-avg_seq', '--avg_seq_len', type=int, default=-1, metavar='NUMBER',
                        help='average sequence length (default: -1)')
    parser.add_argument('-thread_num', '--thread_num', type=int, default=1, metavar='int',
                        help='Testing multithread if thread_num > 1.')

    args = parser.parse_args()
    bert_example(vars(args))