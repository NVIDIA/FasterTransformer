#!/usr/bin/env python3

# Modified MIT License

# Software Copyright (c) 2019 OpenAI

# We don’t claim ownership of the content you create with GPT-2, so it is yours to do with as you please.
# We only ask that you use GPT-2 responsibly and clearly indicate your content was created using GPT-2.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# The above copyright notice and this permission notice need not be included
# with content created by the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

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

import fire
import json
import os
import numpy as np
import sys
import tensorflow as tf

from tensorflow.contrib.training import HParams

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
import examples.tensorflow.gpt.utils.gpt_token_encoder as encoder
from examples.tensorflow.decoder.utils.common import TransformerArgument
from examples.tensorflow.decoder.utils.common import DecodingArgumentNew
from examples.tensorflow.decoder.utils.common import time_test

def sample_model(
    vocab_file="../models/gpt2-vocab.json",
    bpe_file="../models/gpt2-merges.txt",
    model_name='124M',
    nsamples=1,
    batch_size=2,
    length=32,
    temperature=1,
    top_k=4,
    top_p=0.0,
    models_dir='../models/openai_gpt_model',
    data_type='fp32',
    beam_width=1
):
    """Run the sample_model.

    :model_name=124M : String, which model to use
    :nsamples=0 : Number of samples to return, if 0, continues to
     generate samples indefinately.
    :batch_size=1 : Number of batches (only affects speed/memory).
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=4 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    np.random.seed(1)
    tf.set_random_seed(1)

    if data_type == 'fp32':
        tf_data_type = tf.float32
    elif data_type == 'fp16':
        tf_data_type = tf.float16
    else:
        assert(False)
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    vocab_file=os.path.join(models_dir, model_name, 'encoder.json')
    bpe_file=os.path.join(models_dir, model_name, 'vocab.bpe')
    enc = encoder.get_encoder(vocab_file, bpe_file)
    hparams = HParams(n_vocab=0,
                      n_ctx=1024,
                      n_embd=768,
                      n_head=12,
                      n_layer=12)
    
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        saver = tf.train.import_meta_graph("{}/{}/model.ckpt.meta".format(models_dir, model_name))

        # lengths = np.random.randint(low=1, high=4, size=batch_size)
        lengths = np.ones([batch_size], dtype=np.int32) * 8 # TODO support various input lengths
        # lengths = np.zeros([batch_size], dtype=np.int32) # unconditional case
        max_start_length = lengths.max()

        start_ids = np.ones([batch_size, max_start_length]) * enc.encoder['<|endoftext|>']
        # for i in range(batch_size):
        #     start_ids[i][0:lengths[i]] = 198
        # User can put some real start ids here, we use '\n' (198) here.

        sess.run(tf.global_variables_initializer())
        print("[INFO] restore the model {}/{}".format(models_dir, model_name))
        saver.restore(sess, ("{}/{}/model.ckpt".format(models_dir, model_name)))
        
        decoder_args = TransformerArgument(beam_width=beam_width,
                                           head_num=hparams.n_head,
                                           size_per_head=hparams.n_embd // hparams.n_head,
                                           inter_size=hparams.n_embd * 4,
                                           num_layer=hparams.n_layer,
                                           dtype=tf_data_type,
                                           kernel_init_range=0.00,
                                           bias_init_range=0.00)

        decoding_args = DecodingArgumentNew(hparams.n_vocab,
                                            enc.encoder['<|endoftext|>'],
                                            enc.encoder['<|endoftext|>'],
                                            length,
                                            0.0,
                                            top_k,
                                            top_p,
                                            decoder_args)
        
        ckpt_dict = {}
        for var in tf.trainable_variables():
            ckpt_dict[var.name] = var
        
        op_output, sequence_length = ft_gpt_op(ckpt_dict,
                                               decoding_args,
                                               batch_size,
                                               start_ids,
                                               lengths)

        generated = 0
        
        while nsamples == 0 or generated < nsamples:
            op_out, seq_len = sess.run([op_output, sequence_length])
            for i in range(batch_size):
                generated += 1

                if beam_width > 1:
                    for j in range(beam_width):
                        print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                        print(enc.decode(op_out[i][j][:seq_len[i][j]]))
                else:
                        print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                        print(enc.decode(op_out[i][:seq_len[i]]))

def finalize(input_ids, beam_width, parent_ids, sequence_lengths, outputs, end_id, max_seq_len=None):
    maximum_lengths = tf.reduce_max(tf.reshape(
        sequence_lengths, [-1, beam_width]), axis=-1)
    
    if max_seq_len != None:
        array_shape = [max_seq_len, -1, beam_width]
    else:
        array_shape = [tf.reduce_max(maximum_lengths), -1, beam_width]
        
    step_ids = tf.reshape(outputs, array_shape)
    parent_ids = tf.reshape(parent_ids, array_shape)

    # ids = tf.contrib.seq2seq.gather_tree(
    #     step_ids, parent_ids, maximum_lengths, end_id)
    
    # Since we use end_id to padding, we cannot use end_id in the gather_tree
    ids = tf.contrib.seq2seq.gather_tree(
        step_ids, parent_ids, maximum_lengths, -1)

    ids = tf.transpose(ids, perm=[1, 2, 0])
    lengths = tf.not_equal(ids, end_id)
    lengths = tf.cast(lengths, tf.int32)
    
    max_input_length = tf.shape(input_ids)[-1]
    input_ids = tf.reshape(input_ids, [-1, beam_width, max_input_length])
    padding_lengths = tf.cast(tf.equal(input_ids, end_id), tf.int32)
    padding_lengths = tf.reduce_sum(padding_lengths, axis=-1)
    lengths = tf.reduce_sum(lengths, axis=-1)
    lengths = lengths + padding_lengths
    return ids, lengths

def ft_gpt_op(var_dict,
              decoding_args,
              batch_size,
              input_ids,
              input_lengths):
    """Run the decoding with sampling by FasterTransformer.

    Args:
        decoder_vars: A list of tf.Tensor. The variables for decoding. A list of model variables of TensorFlow model.
        decoder_args: The arguments for decoding. The details are in the class "DecodingArgumentNew" of common.py
    Outputs:
        output_ids: A tf.Tensor with shape [batch_size, max(sequence_lengths)], with int type.
                    The results of decoding. It contains the id of token of vocabulary.
        sequence_lengths: A tf.Tensor with shape [batch_size], with int type.
    """

    decoder_args = decoding_args.decoder_args

    gpt_op_module = tf.load_op_library(os.path.join('./lib/libtf_gpt.so'))
    data_type = decoder_args.dtype

    output_ids, parent_ids, sequence_length, cum_log_probs = gpt_op_module.gpt(
        input_ids, # 0
        input_lengths, # 1
        [tf.cast(var_dict["model/h%d/ln_1/b:0" % l], data_type) for l in range(decoder_args.num_layer)], # 2
        [tf.cast(var_dict["model/h%d/ln_1/g:0" % l], data_type) for l in range(decoder_args.num_layer)], # 3
        [tf.cast(var_dict["model/h%d/attn/c_attn/w:0" % l], data_type) for l in range(decoder_args.num_layer)], # 4
        [tf.cast(var_dict["model/h%d/attn/c_attn/b:0" % l], data_type) for l in range(decoder_args.num_layer)], # 5
        [tf.cast(var_dict["model/h%d/attn/c_proj/w:0" % l], data_type) for l in range(decoder_args.num_layer)], # 6
        [tf.cast(var_dict["model/h%d/attn/c_proj/b:0" % l], data_type) for l in range(decoder_args.num_layer)],  # 7
        [tf.cast(var_dict["model/h%d/ln_2/b:0" % l], data_type) for l in range(decoder_args.num_layer)], # 8
        [tf.cast(var_dict["model/h%d/ln_2/g:0" % l], data_type) for l in range(decoder_args.num_layer)], # 9
        [tf.cast(var_dict["model/h%d/mlp/c_fc/w:0" % l], data_type) for l in range(decoder_args.num_layer)], # 10
        [tf.cast(var_dict["model/h%d/mlp/c_fc/b:0" % l], data_type)for l in range(decoder_args.num_layer)], # 11
        [tf.cast(var_dict["model/h%d/mlp/c_proj/w:0" % l], data_type) for l in range(decoder_args.num_layer)], # 12
        [tf.cast(var_dict["model/h%d/mlp/c_proj/b:0" % l], data_type) for l in range(decoder_args.num_layer)], # 13
        tf.cast(var_dict['model/ln_f/b:0'], data_type), # 14
        tf.cast(var_dict['model/ln_f/g:0'], data_type), # 15
        tf.cast(var_dict['model/wpe:0'], data_type), # 16
        tf.cast(var_dict['model/wte:0'], data_type), # 17
        tf.cast(var_dict['model/wte:0'], data_type), # 18
        max_batch_size=batch_size,
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
        repetition_penalty=1.0,
        output_log_probs=True,
        request_output_length=decoding_args.max_seq_len - input_lengths.max())

    return output_ids, sequence_length

if __name__ == '__main__':
    fire.Fire(sample_model)

