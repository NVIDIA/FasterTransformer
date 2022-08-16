# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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
from examples.tensorflow.bert.bert_example import bert_example
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
    threshold = {'fp32': 3e-5, 'fp16': 4e-2, 'bf16': 5e-2 }
    test_level = 1

    def test_batch_fp32(self):
        if self.test_level >= 3:
            print(f"[INFO] test level {self.test_level}, run unit test test_batch_fp32 (level {3})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_batch_fp32 (level {3})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp32'
        
        for batch in [1, 8, 64, 128]:
            args_dict['batch_size'] = batch
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_batch_fp16(self):
        if self.test_level >= 2:
            print(f"[INFO] test level {self.test_level}, run unit test test_batch_fp16 (level {2})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_batch_fp16 (level {2})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp16'
        
        for batch in [1, 8, 64, 128]:
            args_dict['batch_size'] = batch
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_batch_bf16(self):
        if self.test_level >= 2:
            print(f"[INFO] test level {self.test_level}, run unit test test_batch_bf16 (level {2})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_batch_bf16 (level {2})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'bf16'
        
        for batch in [1, 8, 64, 128]:
            args_dict['batch_size'] = batch
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} 2 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head']))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_size_fp32(self):
        if self.test_level >= 3:
            print(f"[INFO] test level {self.test_level}, run unit test test_size_fp32 (level {3})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_size_fp32 (level {3})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp32'
        args_dict['head_number'] = 8
        
        for size in [32, 40, 64, 120, 128]:
            args_dict['size_per_head'] = size
            args_dict['inter_size'] = args_dict['head_number'] * size * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_size_fp16(self):
        if self.test_level >= 2:
            print(f"[INFO] test level {self.test_level}, run unit test test_size_fp16 (level {2})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_size_fp16 (level {2})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp16'
        args_dict['head_number'] = 12
        
        for size in [32, 40, 64, 120, 128]:
            args_dict['size_per_head'] = size
            args_dict['inter_size'] = args_dict['head_number'] * size * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
        
    def test_size_bf16(self):
        if self.test_level >= 2:
            print(f"[INFO] test level {self.test_level}, run unit test test_size_bf16 (level {2})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_size_bf16 (level {2})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'bf16'
        args_dict['head_number'] = 12
        
        for size in [32, 40, 64, 120, 128]:
            args_dict['size_per_head'] = size
            args_dict['inter_size'] = args_dict['head_number'] * size * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} 2 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head']))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_head_fp32(self):
        if self.test_level >= 3:
            print(f"[INFO] test level {self.test_level}, run unit test test_head_fp32 (level {3})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_head_fp32 (level {3})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp32'
        args_dict['size_per_head'] = 64
        
        for h in [8, 12, 17, 24, 29, 32]:
            args_dict['head_number'] = h
            args_dict['inter_size'] = h * args_dict['size_per_head'] * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_head_fp16(self):
        if self.test_level >= 2:
            print(f"[INFO] test level {self.test_level}, run unit test test_head_fp16 (level {2})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_head_fp16 (level {2})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp16'
        args_dict['size_per_head'] = 64
        
        for h in [8, 12, 17, 24, 29, 32]:
            args_dict['head_number'] = h
            args_dict['inter_size'] = h * args_dict['size_per_head'] * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_head_bf16(self):
        if self.test_level >= 2:
            print(f"[INFO] test level {self.test_level}, run unit test test_head_bf16 (level {2})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_head_bf16 (level {2})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'bf16'
        args_dict['size_per_head'] = 64
        
        for h in [8, 12, 17, 24, 29, 32]:
            args_dict['head_number'] = h
            args_dict['inter_size'] = h * args_dict['size_per_head'] * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} 2 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head']))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_hidden_fp32(self):
        if self.test_level >= 3:
            print(f"[INFO] test level {self.test_level}, run unit test test_hidden_fp32 (level {3})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_hidden_fp32 (level {3})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp32'
        
        for p in [tuple([12, 64]), tuple([16, 64]), tuple([4, 32]), tuple([8, 96]), tuple([12, 120])]:
            args_dict['head_number'] = p[0]
            args_dict['size_per_head'] = p[1]
            args_dict['inter_size'] = p[0] * p[1] * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_hidden_fp16(self):
        if self.test_level >= 1:
            print(f"[INFO] test level {self.test_level}, run unit test test_hidden_fp16 (level {1})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_hidden_fp16 (level {1})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp16'
        
        for p in [tuple([12, 64]), tuple([16, 64]), tuple([4, 32]), tuple([8, 96]), tuple([12, 120])]:
            args_dict['head_number'] = p[0]
            args_dict['size_per_head'] = p[1]
            args_dict['inter_size'] = p[0] * p[1] * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_hidden_bf16(self):
        if self.test_level >= 1:
            print(f"[INFO] test level {self.test_level}, run unit test test_hidden_bf16 (level {1})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_hidden_bf16 (level {1})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'bf16'
        
        for p in [tuple([12, 64]), tuple([16, 64]), tuple([4, 32]), tuple([8, 96]), tuple([12, 120])]:
            args_dict['head_number'] = p[0]
            args_dict['size_per_head'] = p[1]
            args_dict['inter_size'] = p[0] * p[1] * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} 2 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head']))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_seqlen_fp32(self):
        if self.test_level >= 3:
            print(f"[INFO] test level {self.test_level}, run unit test test_seqlen_fp32 (level {3})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_seqlen_fp32 (level {3})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp32'
        
        for seqlen in [32, 130, 511, 1024, 1536]:
            args_dict['max_seq_len'] = seqlen
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_seqlen_fp16(self):
        if self.test_level >= 1:
            print(f"[INFO] test level {self.test_level}, run unit test test_seqlen_fp16 (level {1})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_seqlen_fp16 (level {1})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp16'
        
        for seqlen in [32, 130, 511, 1024, 1536]:
            args_dict['max_seq_len'] = seqlen
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_seqlen_bf16(self):
        if self.test_level >= 1:
            print(f"[INFO] test level {self.test_level}, run unit test test_seqlen_bf16 (level {1})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_seqlen_bf16 (level {1})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'bf16'
        
        for seqlen in [32, 130, 511, 1024, 1536]:
            args_dict['max_seq_len'] = seqlen
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} 2 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head']))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_large_model_fp32(self):
        if self.test_level >= 3:
            print(f"[INFO] test level {self.test_level}, run unit test test_large_model_fp32 (level {3})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_large_model_fp32 (level {3})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp32'
        args_dict['num_layer'] = 4
        
        for p in [tuple([32, 64]), tuple([64, 64]), tuple([32, 128])]:
            args_dict['head_number'] = p[0]
            args_dict['size_per_head'] = p[1]
            args_dict['inter_size'] = p[0] * p[1] * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])
    
    def test_large_model_fp16(self):
        if self.test_level >= 2:
            print(f"[INFO] test level {self.test_level}, run unit test test_large_model_fp16 (level {2})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_large_model_fp16 (level {2})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp16'
        args_dict['num_layer'] = 4
        threshold = 0.08 # Use larger threshold for larger model, need to check it makes sense or not

        for p in [tuple([32, 64]), tuple([64, 64]), tuple([32, 128])]:
            args_dict['head_number'] = p[0]
            args_dict['size_per_head'] = p[1]
            args_dict['inter_size'] = p[0] * p[1] * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head'],
                                                            args_dict['data_type'] == 'fp16'))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < threshold)
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < threshold)

    def test_large_model_bf16(self):
        if self.test_level >= 2:
            print(f"[INFO] test level {self.test_level}, run unit test test_large_model_bf16 (level {2})")
        else:
            print(f"[INFO] test level {self.test_level}, skip unit test test_large_model_bf16 (level {2})")
            return
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'bf16'
        args_dict['num_layer'] = 4
        threshold = 0.08 # Use larger threshold for larger model, need to check it makes sense or not

        for p in [tuple([32, 64]), tuple([64, 64]), tuple([32, 128])]:
            args_dict['head_number'] = p[0]
            args_dict['size_per_head'] = p[1]
            args_dict['inter_size'] = p[0] * p[1] * 4
            tf.reset_default_graph()
            os.system("./bin/bert_gemm {} {} {} {} 2 0 > .tmp.gemm.log && cat gemm_config.in".format(args_dict['batch_size'], args_dict['max_seq_len'],
                                                            args_dict['head_number'], args_dict['size_per_head']))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < threshold)
            max_diff = encoder_example(args_dict)
            self.assertTrue(max_diff < threshold)

if __name__ == "__main__":
    test_level = 1
    if len(sys.argv) > 1:
        test_level = sys.argv.pop()
    TestEncoder.test_level = int(test_level)
    unittest.main()
    