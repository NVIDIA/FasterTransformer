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

from __future__ import print_function
import unittest
import os
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
import tensorflow as tf
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../..")
from examples.tensorflow.encoder.encoder_example import encoder_example

class TestEncoder(unittest.TestCase):
    
    common_args_dict = {'batch_size' : 4,
                        'num_layer' : 12,
                        'max_seq_len': 32,
                        'head_number': 12,
                        'size_per_head': 64,
                        'inter_size': 12 * 64 * 4,
                        'allow_gemm_test': 'False',
                        'test_time': 0,
                        'data_type': 'fp32',
                        'remove_padding': 'False',
                        'avg_seq_len': -1,
                        'thread_num': 1,
                        'int8_mode': 0
                        }
    threshold = {'fp32': 3e-5, 'fp16': 4e-2 }

    def test_batch_fp32(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp32'
        
        for batch in [1, 8, 64, 128]:
            args_dict['batch_size'] = batch
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat .tmp.gemm.log".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_batch_fp16(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp16'
        
        for batch in [1, 8, 64, 128]:
            args_dict['batch_size'] = batch
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat .tmp.gemm.log".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_size_fp32(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp32'
        args_dict['head_number'] = 8
        
        for size in [32, 40, 64, 120, 128]:
            args_dict['size_per_head'] = size
            args_dict['inter_size'] = 8 * size * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat .tmp.gemm.log".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_size_fp16(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp16'
        args_dict['head_number'] = 8
        
        for size in [32, 40, 64, 120, 128]:
            args_dict['size_per_head'] = size
            args_dict['inter_size'] = 8 * size * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat .tmp.gemm.log".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
        
    def test_head_fp32(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp32'
        args_dict['size_per_head'] = 64
        
        for h in [8, 12, 17, 24, 29, 32]:
            args_dict['head_number'] = h
            args_dict['inter_size'] = h * 64 * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat .tmp.gemm.log".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_head_fp32(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp16'
        args_dict['size_per_head'] = 64
        
        for h in [8, 12, 17, 24, 29, 32]:
            args_dict['head_number'] = h
            args_dict['inter_size'] = h * 64 * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat .tmp.gemm.log".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_hidden_fp32(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp32'
        
        for p in [tuple([12, 64]), tuple([16, 64]), tuple([4, 32]), tuple([8, 96])]:
            args_dict['head_number'] = p[0]
            args_dict['size_per_head'] = p[1]
            args_dict['inter_size'] = p[0] * p[1] * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat .tmp.gemm.log".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_hidden_fp16(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp16'
        
        for p in [tuple([12, 64]), tuple([16, 64]), tuple([4, 32]), tuple([8, 96])]:
            args_dict['head_number'] = p[0]
            args_dict['size_per_head'] = p[1]
            args_dict['inter_size'] = p[0] * p[1] * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat .tmp.gemm.log".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_seqlen_fp32(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp32'
        
        for seqlen in [32, 130, 511, 1024, 1536]:
            args_dict['max_seq_len'] = seqlen
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat .tmp.gemm.log".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_seqlen_fp16(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp16'
        
        for seqlen in [32, 130, 511, 1024, 1536]:
            args_dict['max_seq_len'] = seqlen
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat .tmp.gemm.log".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

if __name__ == "__main__":
    unittest.main()
