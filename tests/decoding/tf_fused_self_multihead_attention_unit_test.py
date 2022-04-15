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

from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import sys
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

USE_CACHE_BATCH_MAJOR_ATTENTION = True

class TestFusedQKVMutiheadAttention(unittest.TestCase):

    def test_attn_batch_fp32(self):
        for b in [1, 4, 32, 128]:
            tf.reset_default_graph()
            self.run_attn(b, 128, 12, 64, tf.float32)

    def test_attn_batch_fp16(self):
        for b in [1, 4, 32, 128]:
            tf.reset_default_graph()
            self.run_attn(b, 128, 12, 64, tf.float16)
    
    def test_attn_seq_fp32(self):
        for seq in [64, 96, 128, 384]:
            tf.reset_default_graph()
            self.run_attn(4, seq, 12, 64, tf.float32)

    def test_attn_seq_fp16(self):
        for seq in [64, 96, 128, 384]:
            tf.reset_default_graph()
            self.run_attn(4, seq, 12, 64, tf.float16)
    
    def test_attn_head_fp32(self):
        for head in [8, 12, 16]:
            tf.reset_default_graph()
            self.run_attn(4, 128, head, 64, tf.float32)

    def test_attn_head_fp16(self):
        for head in [8, 12, 16]:
            tf.reset_default_graph()
            self.run_attn(4, 128, head, 64, tf.float16)

    def test_attn_size_fp32(self):
        for size in [32, 64, 96, 128, 160, 192, 224, 256]:
            tf.reset_default_graph()
            self.run_attn(4, 128, 12, size, tf.float32)

    def test_attn_size_fp16(self):
        for size in [32, 64, 96, 128, 160, 192, 224, 256]:
            tf.reset_default_graph()
            self.run_attn(4, 128, 12, size, tf.float16)

    def run_attn(self, batch_size, seq_len, head_num, size_per_head, data_type):
        threshold = 3e-5
        if data_type == tf.float16:
            threshold = 4e-3
        # Inputs: qkv_buf and k/v cache
        # Do: update k/v cahce, and compute attention (Q*K, QK*V)
        # Output: attention result, new k/v cache
        # Notes: Only used for decoder, so seqlen of q is always 1.
        
        np.random.seed(1)
        tf.set_random_seed(1)

        qkv_buf = tf.random.normal([batch_size, 3, head_num, size_per_head], dtype=data_type)
        qkv_bias = tf.random.normal([3, head_num, size_per_head], dtype=data_type)
        k_cache = tf.random.normal([batch_size, head_num, seq_len - 1, size_per_head], dtype=data_type)
        v_cache = tf.random.normal([batch_size, head_num, seq_len - 1, size_per_head], dtype=data_type)

        q, k, v = tf.split(qkv_buf + qkv_bias, 3, axis=1)
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        keys = tf.concat([k_cache, k], axis=2)
        values = tf.concat([v_cache, v], axis=2)

        tf_k_cache = keys
        tf_v_cache = values

        q *= (size_per_head)**-0.5
        dot = tf.matmul(q, keys, transpose_b=True)

        attn = tf.cast(tf.nn.softmax(tf.cast(dot, data_type)), dot.dtype)
        context = tf.matmul(attn, values)
        tf_attn_result = tf.transpose(context, [0, 2, 1, 3])

        fused_multihead_attention_op = tf.load_op_library(os.path.join('./lib/libtf_fused_self_attention.so'))
        # if USE_CACHE_BATCH_MAJOR_ATTENTION == True
        # The layout of the cache buffer for the keys is [batch_size, head_num, size_per_head/x, seq_len, x] 
        # where x == 8 for FP16 and x == 4 for FP32 where the fastest moving dimension (contiguous data) 
        # is the rightmost one. The values for x are chosen to create chunks of 16 bytes.
        # The layout of the cache buffer for the values is [batch_size, head_num, seq_len, size_per_head].
        if USE_CACHE_BATCH_MAJOR_ATTENTION == True:
            x = 8 if data_type == tf.float16 else 4
            assert size_per_head % x == 0
            ft_k_cache = tf.concat([k_cache, tf.zeros_like(k)], axis=2)
            ft_k_cache_shape = np.array([batch_size, head_num, seq_len, size_per_head / x, x], dtype=np.int32)
            ft_k_cache = tf.reshape(ft_k_cache, ft_k_cache_shape)
            ft_k_cache = tf.transpose(ft_k_cache, [0, 1, 3, 2, 4])
            ft_v_cache = tf.concat([v_cache, tf.zeros_like(v)], axis=2)
        else :
            ft_k_cache = tf.concat([k_cache, tf.zeros_like(k)], axis=2) # [batch_size, head_num, seq_len + 1, size_per_head]
            ft_k_cache = tf.transpose(ft_k_cache, [2, 0, 1, 3]) # [seq_len + 1, batch_size, head_num, size_per_head]
            ft_v_cache = tf.concat([v_cache, tf.zeros_like(v)], axis=2)
            ft_v_cache = tf.transpose(ft_v_cache, [2, 0, 1, 3])


        ft_attn_result, ft_k_cache, ft_v_cache = fused_multihead_attention_op.fused_qkv_multi_head_attention(qkv_buf, 
                                                                                                             qkv_bias,
                                                                                                             ft_k_cache,
                                                                                                             ft_v_cache,
                                                                                                             batch_size,
                                                                                                             seq_len,
                                                                                                             head_num,
                                                                                                             size_per_head)

        if USE_CACHE_BATCH_MAJOR_ATTENTION == True:
            ft_k_cache = tf.transpose(ft_k_cache, [0, 1, 3, 2, 4])
            ft_k_cache_shape = np.array([batch_size, head_num, seq_len, size_per_head], dtype=np.int32)
            ft_k_cache = tf.reshape(ft_k_cache, ft_k_cache_shape)
        else:
            ft_k_cache = tf.transpose(ft_k_cache, [1, 2, 0, 3]) # [batch_size, head_num, seq_len + 1, size_per_head]
            ft_v_cache = tf.transpose(ft_v_cache, [1, 2, 0, 3])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:
            print(batch_size, seq_len, head_num, size_per_head)

            sess.run(tf.global_variables_initializer())

            tf_attn_result_val, ft_attn_result_val, k_cache_diff_val, v_cache_diff_val = sess.run([tf_attn_result, 
                                                                          ft_attn_result,
                                                                          tf_k_cache - ft_k_cache,
                                                                          tf_v_cache - ft_v_cache])
            
            attn_diff_val = tf_attn_result_val - ft_attn_result_val
            attn_max_diff = abs(attn_diff_val).max()
            attn_max_diff_id = abs(attn_diff_val).argmax()
            print("attn_max_diff_id = ", attn_max_diff_id)
            k_cache_max_diff = abs(k_cache_diff_val).max()
            v_cache_max_diff = abs(v_cache_diff_val).max()

            print("tf_attn_result_val at max diff = ", tf_attn_result_val.flatten()[attn_max_diff_id])
            print("ft_attn_result_val at max diff = ", ft_attn_result_val.flatten()[attn_max_diff_id])
            print("threshold = ", threshold)
            print(attn_max_diff)
            print(k_cache_max_diff)
            print(v_cache_max_diff)
            sys.stdout.flush()

            assert(attn_max_diff < threshold)
            assert(k_cache_max_diff < threshold)
            assert(v_cache_max_diff < threshold)

if __name__ == "__main__":
    unittest.main()