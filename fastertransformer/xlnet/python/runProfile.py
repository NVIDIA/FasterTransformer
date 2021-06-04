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




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import absl.logging as _logging    # pylint: disable=unused-import

import tensorflow as tf

import sys
import getopt 

import json
import absl.logging as _logging    # pylint: disable=unused-import
import modeling
import numpy as np
from datetime import datetime

from tensorflow.python.client import timeline

def getTensor(shape, dtype):
    t= np.random.randn(shape[0], shape[1], shape[2])
    p= tf.convert_to_tensor(t, dtype=float)
    return p

def getTensor4(shape, dtype):
    t= np.random.randn(shape[0], shape[1], shape[2], shape[3])
    p= tf.convert_to_tensor(t, dtype=float)
    return p

def getTensor2(shape, dtype):
    t= np.random.randn(shape[0], shape[1])
    p= tf.convert_to_tensor(t, dtype=float)
    return p


def getTimeline(run_metadata, output_h):
        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
            sess.run(tf.global_variables_initializer())

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            sess.run(output_h)
            Model_variables = tf.GraphKeys.GLOBAL_VARIABLES

def getJson(json_file):
    json_f=open(json_file)
    data = json.load(json_f)
    n_token=data["n_token"]
    untie_r=data["untie_r"]
    ff_activation=data["ff_activation"]
    d_inner=data["d_inner"]
    d_head=data["d_head"]
    n_head=data["n_head"]
    d_model=data["d_model"]
    n_head=data["n_head"]
    n_layer=data["n_layer"]
    json_f.close()

    return n_token,untie_r,ff_activation,d_inner,d_head,n_head, d_model, n_head, n_layer



def runtest(qlen, bsz, warm_up_ite, profile_ite):
    plen=qlen*2
    i=0

    n_token,untie_r,ff_activation,d_inner,d_head,n_head, d_model, n_head, n_layer=getJson(json_file)

    mems = [None] * n_layer


    dropout=0.1
    dropatt=0.1
    is_training=False
    reuse=False

    with tf.variable_scope('layer_{}'.format(i)):

        output_h=getTensor(shape=(qlen, bsz, 768), dtype=tf.float32)
        pos_emb=getTensor(shape=(plen, bsz, 768), dtype=tf.float32)

        r_w_bias=tf.Variable(tf.random_normal([12, 12, 64],dtype=tf.float32))
        r_r_bias=tf.Variable(tf.random_normal([12, 12, 64],dtype=tf.float32))

        #r_w_bias=tf.Variable(tf.random_normal([12, 64],dtype=tf.float32))
        #r_r_bias=tf.Variable(tf.random_normal([12, 64],dtype=tf.float32))


        seg_mat=getTensor4(shape=(qlen, qlen, bsz, 2), dtype=tf.float32)
        r_s_bias_i=getTensor2(shape=(12, 64), dtype=tf.float32)
        seg_embed_i=getTensor(shape=(2, 12, 64), dtype=tf.float32)

        non_tgt_mask=getTensor4(shape=(qlen, qlen, bsz, 1), dtype=tf.float32)

        initializer = tf.initializers.random_normal(
                    stddev=0.02,
                    seed=None)

        output_h, _ = modeling.rel_multihead_attn(
                h=output_h,
                r=pos_emb,
                r_w_bias=r_w_bias[i],
                r_r_bias=r_r_bias[i],
                seg_mat=seg_mat,
                r_s_bias=r_s_bias_i,
                seg_embed=seg_embed_i,
                attn_mask=non_tgt_mask,
                mems=mems[i],
                d_model=d_model,
                n_head=n_head,
                d_head=d_head,
                dropout=dropout,
                dropatt=dropatt,
                is_training=is_training,
                kernel_initializer=initializer,
                reuse=reuse)

        output_h,_ = modeling.positionwise_ffn(
                 inp=output_h,
                 d_model=d_model,
                 d_inner=d_inner,
                 dropout=dropout,
                 kernel_initializer=initializer,
                 activation_type='gelu',
                 is_training=is_training,
                 reuse=reuse)
        

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        time_sum = 0
        for i in range(warm_up_ite):
            sess.run(output_h, options=run_options, run_metadata=run_metadata)

        time_sum = 0
        for i in range(profile_ite):
            a = datetime.now()
            sess.run(output_h, options=run_options, run_metadata=run_metadata)
            b = datetime.now()
            time_sum += (b - a).total_seconds()
        
        time=time_sum * 1000 / profile_ite
        record='RUN_TIME: batch_size= '+str(bsz)+'  seq_len= '+str(qlen)+' run_time= '+str(time)+ ' MS'

        print(record)
        
        return record


def usage():
    print(" -b batch_size, default 8")
    print(" -s seq_len, default 128")
    print(" -w warm_up_ite (Run the attention layer for warm_up_ite times first), default 5")
    print(" -t profile_ite (Run the attention layer for profile_ite times to get the performance), default 10")
    print(" -j json_file (The json_file of XLNET), default ../../../Data/xlnet_cased_L-12_H-768_A-12/xlnet_config.json")
    print("Example: python runProfile.py -b 8 -s 128 -w 50 -t 100 -j ../../data/xlnet_cased_L-12_H-768_A-12/xlnet_config.json")




if __name__ == "__main__":
    seq_len=128
    batch_size=8
    warm_up_ite=100
    profile_ite=200
    output_file="./xla.log"
    json_file="../data/xlnet_cased_L-12_H-768_A-12/xlnet_config.json"

    opts, args = getopt.getopt(sys.argv[1:], "b:s:w:t:j:h") 
    for op, value in opts:
        if op == "-b":
            batch_size =int(value)
        elif op == "-s":
            seq_len = int(value)
        elif op == "-w":
            warm_up_ite=int(value)
        elif op == "-t":
            profile_ite=int(value)
        elif op == "-j":
            json_file = value
        elif op == "-h":
            usage()
            sys.exit()


    record=runtest(seq_len, batch_size, warm_up_ite, profile_ite)

