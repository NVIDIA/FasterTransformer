from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

# usage example
# python ./runData.py -i $data_file -o $output_file -j $json_file -m $ckpt_file -b $batch_size -l $seq_len -f $use_float16 -n $index


import getopt
import modeling
import numpy as np
from tensorflow.python.client import timeline
import tensorflow as tf
from datetime import datetime
import json
import sys
import absl.logging as _logging    # pylint: disable=unused-import


from absl import flags
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def getInput(input_file, batch_size, i):
    data = np.load(input_file)
    arr_input_ids = data["input_ids:0"]
    arr_input_mask = data["input_mask:0"]
    arr_segment_ids = data["segment_ids:0"]

    input_ids = np.transpose(arr_input_ids[i * batch_size:(i + 1) * batch_size, :])
    input_mask = np.transpose(arr_input_mask[i * batch_size:(i + 1) * batch_size, :])
    segment_ids = np.transpose(arr_segment_ids[i * batch_size:(i + 1) * batch_size, :])

    print("Get input batch {}, {}, {}".format(input_ids.dtype, input_mask.dtype, segment_ids.dtype))

    data.close()
    return input_ids, input_mask, segment_ids


def getJson(json_file):
    json_f = open(json_file)
    data = json.load(json_f)
    n_token = data["n_token"]
    untie_r = data["untie_r"]
    ff_activation = data["ff_activation"]
    d_inner = data["d_inner"]
    d_head = data["d_head"]
    n_head = data["n_head"]
    d_model = data["d_model"]
    n_head = data["n_head"]
    n_layer = data["n_layer"]
    json_f.close()

    return n_token, untie_r, ff_activation, d_inner, d_head, n_head, d_model, n_head, n_layer


def runTest(json_file, seq_len, batch_size, input_ids, input_mask, segment_ids, use_float16):
    # Acquire network settings
    n_token, untie_r, ff_activation, d_inner, d_head, n_head, d_model, n_head, n_layer = getJson(json_file)

    # Set Running parameters
    attn_type = "bi"  # attn_type="uni"
    bi_data = False
    dropout = 0.1
    dropatt = 0.1
    is_training = False
    reuse = False
    use_tpu = False
    mem_len = None
    reuse_len = None

    initializer = tf.initializers.random_normal(
        stddev=0.02,
        seed=None)

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        output, arr_output = modeling.transformer_xl(
            input_ids, n_token, n_layer, d_model, n_head,
            d_head, d_inner, dropout, dropatt, attn_type,
            bi_data, initializer, is_training, mem_len, untie_r=untie_r,
            ff_activation=ff_activation, input_mask=input_mask, seg_id=segment_ids, use_float16=use_float16,
            use_tpu=use_tpu, reuse_len=reuse_len)
    return output, arr_output, n_layer


def usage():
    print(" -i input_file")
    print(" -o output_file")
    print(" -j json_file")
    print(" -m model_file")
    print(" -l max_seq_length")
    print(" -b batch_size")
    print(" -o output_file")
    print(" -f use_float16")
    print(" -n index of the inputdata batch")
    print(" -h output help info")
    print("Example: python runData.py -i ./data.npz -o output.npz -j xlnet_cased_L-12_H-768_A-12/xlnet_config.json \
            -m xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt -b 8 -l 128 -n 12")


if __name__ == "__main__":
    # Init perameter
    seq_len = 128
    batch_size = 8
    use_float16 = False
    input_file = "./data.npz"
    index = 0

    json_file = "../../../Data/xlnet_cased_L-12_H-768_A-12/xlnet_config.json"
    model_file = "../../../Data/xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt"
    output_file = "./output.npz"

    # Set perameter
    opts, args = getopt.getopt(sys.argv[1:], "hi:j:m:b:l:f:o:n:")
    for op, value in opts:
        if op == "-i":
            input_file = value
        elif op == "-o":
            output_file = value
        elif op == "-j":
            json_file = value
        elif op == "-m":
            model_file = value
        elif op == "-b":
            batch_size = int(value)
        elif op == "-l":
            seq_len = int(value)
        elif op == "-f":
            use_float16 = bool(int(value))
        elif op == "-n":
            index = int(value)
        elif op == "-h":
            usage()
            sys.exit()

    print("USE FLOAT 16: ", str(use_float16))

    # Get Input Value
    input_ids, input_mask, segment_ids = getInput(input_file, batch_size, index)

    tf.reset_default_graph()

    # Set input
    t_input_ids = tf.placeholder(tf.int32, shape=[seq_len, batch_size])
    t_input_mask = tf.placeholder(tf.float32, shape=[seq_len, batch_size])
    t_segment_ids = tf.placeholder(tf.int32, shape=[seq_len, batch_size])

    # Define Output
    output, arr_output, n_layer = runTest(json_file, seq_len, batch_size, t_input_ids, t_input_mask, t_segment_ids,
                                          use_float16)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, model_file)
        # Run
        arr_output = sess.run(arr_output, feed_dict={t_input_ids: input_ids,
                              t_input_mask: input_mask, t_segment_ids: segment_ids})

    # Save result
    data = {}
    i = 0

    data["attn_mask"] = arr_output[0].transpose((2, 0, 1, 3))
    data["output_h"] = arr_output[1].transpose((1, 0, 2))
    data["seg_mat"] = arr_output[2].transpose((2, 0, 1, 3))
    data["pos_emb"] = arr_output[3].transpose((1, 0, 2))

    arr_output = arr_output[4:]
    for r in arr_output:
        print(r.shape)

    CACHE_NUM = 8
    for i in range(n_layer):
        data["layer_{}_q_head_h".format(i)] = arr_output[i * CACHE_NUM].transpose((1, 0, 2, 3))
        data["layer_{}_k_head_h".format((i))] = arr_output[i * CACHE_NUM + 1].transpose((1, 0, 2, 3))
        data["layer_{}_v_head_h".format((i))] = arr_output[i * CACHE_NUM + 2].transpose((1, 0, 2, 3))
        data["layer_{}_k_head_r".format((i))] = arr_output[i * CACHE_NUM + 3].transpose((1, 0, 2, 3))
        data["layer_{}_attn_vec".format((i))] = arr_output[i * CACHE_NUM + 4].transpose((1, 0, 2, 3))
        data["layer_{}_attn_output".format((i))] = arr_output[i * CACHE_NUM + 5].transpose((1, 0, 2))
        data["layer_{}_layer_1".format((i))] = arr_output[i * CACHE_NUM + 6].transpose((1, 0, 2))
        data["layer_{}".format((i))] = arr_output[i * CACHE_NUM + 7].transpose((1, 0, 2))

    np.savez(output_file, **data)
