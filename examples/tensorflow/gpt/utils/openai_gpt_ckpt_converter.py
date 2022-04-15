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

import numpy as np
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../..")
import argparse
import tensorflow.compat.v1  as tf
tf.disable_v2_behavior()

"""
This file converts the model of TensorFlow checkpoint to numpy array,
and store the numpy into <.bin> file. We also little modify the name.
The followings are examples:

For example, the original name of variable of the 3rd transformer layers are:
    model.h<layer>.ln_1.b
    model.h<layer>.ln_1.g
    model.h<layer>.attn.c_attn.b
    model.h<layer>.attn.c_attn.w
    model.h<layer>.attn.c_proj.b
    model.h<layer>.attn.c_proj.w
    model.h<layer>.ln_2.b
    model.h<layer>.ln_2.g
    model.h<layer>.mlp.c_fc.b
    model.h<layer>.mlp.c_fc.w
    model.h<layer>.mlp.c_proj.b
    model.h<layer>.mlp.c_proj.w
and we convert them to 
    model.layers.3.input_layernorm.weight
    model.layers.3.input_layernorm.bias
    model.layers.3.attention.query_key_value.weight
    model.layers.3.attention.query_key_value.bias
    model.layers.3.attention.dense.weight
    model.layers.3.attention.dense.bias
    model.layers.3.post_attention_layernorm.weight
    model.layers.3.post_attention_layernorm.bias
    model.layers.3.mlp.dense_h_to_4h.weight
    model.layers.3.mlp.dense_h_to_4h.bias
    model.layers.3.mlp.dense_4h_to_h.weight
    model.layers.3.mlp.dense_4h_to_h.bias

For other variables:
    model.wpe
    model.wte
    model.ln_f.b
    model.ln_f.g
we convert them to
    model.wpe (no change)
    model.wte (no change)
    model.final_layernorm.weight
    model.final_layernorm.bias

Note that we convert the "gamma" and "beta" of layernorm to "weight" and
"bias".

This converter would skip the variables about training. For example,
the weights come from Adam optimizers.

For multi-gpu weights, we need to split the following weights:
    1. attn/c_attn/w: we need to reshape to [hidden_dim, 3, hidden_dim], split
       at last axis, and then reshape to [hidden_dim, 3 * hidden_dim / gpu_num].
       Namely, we split by W = [W_1, W_2, ...]
       If we do not fuse QKV, we will convert from [h, 3 * h] to [3, h, h]
    2. attn/c_attn/b: it is similar to attn/c_attn/w
    3. attn/c_proj/w: we need to split at axis 1. Namely, we split by W = [ [W_1], [W_2] ]
    4. mlp/c_fc/w: we need to split at axis 0. Namely, we split by W = [W1, W2]
    5. mlp/c_fc/b: it is similar to mlp/c_fc/w
    6. mlp/c_proj/w: we need to split at axis 1. Namely, we split by W = [ [W_1], [W_2] ]
        
    Note that we do not need to split followings variables:
        attn/c_proj/b 
        mlp/c_proj/b
        ln_1/g, ln_1/b
        ln_2/g, ln_2/b
        wte, wpe
"""

# def convert_to_bin(args):
# split the ckpt from 1 to n
def split_and_convert(args):
    if args.fused_qkv == 1:
        saved_dir = args.saved_dir + "/%d-gpu/" % args.gpu_num
    else:
        saved_dir = args.saved_dir + "/unfusedQKV-%d-gpu/" % args.gpu_num
        
    if(os.path.exists(saved_dir) == False):
        os.makedirs(saved_dir)

    ckpt_name = args.in_file
    gpu_num = args.gpu_num

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(ckpt_name + ".meta")
        saver.restore(sess, (ckpt_name))
        all_variables = tf.trainable_variables()
        ckpt = {}

        all_val = sess.run(all_variables)
        for var, val in zip(all_variables, all_val):
            if var.name.find("Adam") == -1:
                print(var.name, var.shape)
                val = np.squeeze(val)
                # spilt the kernel for multi-gpu inference
                saved_name = var.name.replace("model/h", "model.layers.").replace("/", ".")
                if saved_name.find(".w:0") != -1:
                    saved_name = saved_name.replace(".w:0", ".weight")
                elif saved_name.find(".b:0") != -1:
                    saved_name = saved_name.replace(".b:0", ".bias")
                elif saved_name.find(".g:0") != -1:
                    saved_name = saved_name.replace(".g:0", ".weight")
                elif saved_name.find(".wpe:0") != -1:
                    saved_name = saved_name.replace(".wpe:0", ".wpe")
                elif saved_name.find(".wte:0") != -1:
                    saved_name = saved_name.replace(".wte:0", ".wte")

                if saved_name.find("ln_1") != -1:
                   saved_name = saved_name.replace("ln_1", "input_layernorm")
                elif saved_name.find("attn.c_attn") != -1:
                   saved_name = saved_name.replace("attn.c_attn", "attention.query_key_value")
                elif saved_name.find("attn.c_proj") != -1:
                   saved_name = saved_name.replace("attn.c_proj", "attention.dense")
                elif saved_name.find("ln_2") != -1:
                   saved_name = saved_name.replace("ln_2", "post_attention_layernorm")
                elif saved_name.find("mlp.c_fc") != -1:
                   saved_name = saved_name.replace("mlp.c_fc", "mlp.dense_h_to_4h")
                elif saved_name.find("mlp.c_proj") != -1:
                   saved_name = saved_name.replace("mlp.c_proj", "mlp.dense_4h_to_h")
                elif saved_name.find("ln_f") != -1:
                   saved_name = saved_name.replace("ln_f", "final_layernorm")

                if var.name.find("attn/c_attn") != -1:
                    val = val.reshape([-1, 3, (int)(val.shape[-1] / 3)])
                    if args.fused_qkv == 0:
                        val = val.transpose([1, 0, 2])

                    split_vals = np.split(val, gpu_num, axis=-1)
                    for i in range(gpu_num):
                        saved_path = saved_dir + saved_name + ".%d.bin" % i
                        split_vals[i].astype(np.float32).tofile(saved_path)

                elif var.name.find("attn/c_proj/w") != -1:
                    split_vals = np.split(val, gpu_num, axis=0)
                    for i in range(gpu_num):
                        saved_path = saved_dir + saved_name + ".%d.bin" % i
                        split_vals[i].astype(np.float32).tofile(saved_path)
                elif var.name.find("mlp/c_fc") != -1:
                    split_vals = np.split(val, gpu_num, axis=-1)
                    for i in range(gpu_num):
                        saved_path = saved_dir + saved_name + ".%d.bin" % i
                        split_vals[i].astype(np.float32).tofile(saved_path)
                elif var.name.find("mlp/c_proj/w") != -1:
                    split_vals = np.split(val, gpu_num, axis=0)
                    for i in range(gpu_num):
                        saved_path = saved_dir + saved_name + ".%d.bin" % i
                        split_vals[i].astype(np.float32).tofile(saved_path)
                else:
                    saved_path = saved_dir + saved_name + ".bin"
                    val.astype(np.float32).tofile(saved_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-saved_dir', '-o', type=str, help='file name of output file', required=True)
    parser.add_argument('-in_file', '-i', type=str, help='file name of input checkpoint file', required=True)
    parser.add_argument('-gpu_num', '-g', type=int, default=1, help='How many gpus for inference')
    parser.add_argument('-fused_qkv', '-fused_qkv', type=int, default=1, help='Fuse the qkv weights or not. Default is true (1)', choices=[0, 1])

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    split_and_convert(args)