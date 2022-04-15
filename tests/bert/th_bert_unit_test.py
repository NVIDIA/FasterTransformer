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
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../..")
from examples.pytorch.bert.bert_example import bert_example

class TestEncoder(unittest.TestCase):
    
    common_args_dict = {'batch_size' : 4,
                        'layer_num' : 12,
                        'seq_len': 32,
                        'head_num': 12,
                        'head_size': 64,
                        'inter_size': 12 * 64 * 4,
                        'allow_gemm_test': False,
                        'sparse': False,
                        'time': False,
                        'fp16': False,
                        'remove_padding': False,
                        'avg_seq_len': -1,
                        'thread_num': 1,
                        'ths_path': 'lib/libth_bert.so',
                        'weight_path': None,
                        'int8_mode': 0
                        }
    threshold = {False: 3e-5, True: 4e-2 }

    def test_batch_fp32(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        
        for batch in [1, 8, 64, 128]:
            args_dict['batch_size'] = batch
            
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat .tmp.gemm.log".format(args_dict['batch_size'], args_dict['seq_len'],
                                                            args_dict['head_num'], args_dict['head_size'],
                                                            args_dict['fp16'] == True))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['fp16']])

    def test_batch_fp16(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['fp16'] = True
        
        for batch in [1, 8, 64, 128]:
            args_dict['batch_size'] = batch
            
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat .tmp.gemm.log".format(args_dict['batch_size'], args_dict['seq_len'],
                                                            args_dict['head_num'], args_dict['head_size'],
                                                            args_dict['fp16'] == True))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['fp16']])

    def test_hidden_fp32(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        
        for p in [tuple([12, 64]), tuple([16, 64]), tuple([4, 32]), tuple([8, 96])]:
            args_dict['head_num'] = p[0]
            args_dict['head_size'] = p[1]
            args_dict['inter_size'] = p[0] * p[1] * 4
            
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat .tmp.gemm.log".format(args_dict['batch_size'], args_dict['seq_len'],
                                                            args_dict['head_num'], args_dict['head_size'],
                                                            args_dict['fp16'] == True))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['fp16']])

    def test_hidden_fp16(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['fp16'] = True
        
        for p in [tuple([12, 64]), tuple([16, 64]), tuple([4, 32]), tuple([8, 96])]:
            args_dict['head_num'] = p[0]
            args_dict['head_size'] = p[1]
            args_dict['inter_size'] = p[0] * p[1] * 4
            
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat .tmp.gemm.log".format(args_dict['batch_size'], args_dict['seq_len'],
                                                            args_dict['head_num'], args_dict['head_size'],
                                                            args_dict['fp16'] == True))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['fp16']])

    def test_seqlen_fp32(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        
        for seqlen in [32, 130, 511, 1024, 1536]:
            args_dict['seq_len'] = seqlen
            if seqlen == 1536:
                args_dict['layer_num'] = 6
            
            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat .tmp.gemm.log".format(args_dict['batch_size'], args_dict['seq_len'],
                                                            args_dict['head_num'], args_dict['head_size'],
                                                            args_dict['fp16'] == True))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['fp16']])

    def test_seqlen_fp16(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['fp16'] = True
        
        for seqlen in [32, 130, 511, 1024, 1536]:
            args_dict['seq_len'] = seqlen
            if seqlen == 1536:
                args_dict['layer_num'] = 6

            os.system("./bin/bert_gemm {} {} {} {} {} 0 > .tmp.gemm.log && cat .tmp.gemm.log".format(args_dict['batch_size'], args_dict['seq_len'],
                                                            args_dict['head_num'], args_dict['head_size'],
                                                            args_dict['fp16'] == True))
            max_diff = bert_example(args_dict)
            self.assertTrue(max_diff < self.threshold[args_dict['fp16']])

if __name__ == "__main__":
    unittest.main()
