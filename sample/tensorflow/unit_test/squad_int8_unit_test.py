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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf
import json
sys.path.append("./tensorflow/tensorflow_bert")
from squad_evaluate_v1_1 import evaluate
flags = tf.flags
FLAGS = flags.FLAGS

def type_convert(model_file, output_dir):
    if not os.path.exists(output_dir):
        os.system("mkdir "+output_dir)
    os.system("python tensorflow/tensorflow_bert/ckpt_type_convert.py --init_checkpoint={} --fp16_checkpoint={}".format(model_file, output_dir+"/model.ckpt"))

def ckpt_quantization(model_file, output_dir, int8_mode):
    if not os.path.exists(output_dir):
        os.system("mkdir "+output_dir)
    os.system("python tensorflow/tensorflow_bert/ckpt_quantization.py \
               --init_checkpoint={} \
               --quantized_checkpoint={} \
               --int8_mode={}".format(model_file, output_dir+"/model.ckpt", int8_mode))

def run_int8_squad(vocab_file, bert_config_file, init_checkpoint, train_file, predict_file, max_seq_length, output_dir, int8_mode, remove_padding, allow_gemm_test):
    if not os.path.exists(output_dir):
        os.system("mkdir "+output_dir)
    os.system("python tensorflow/tensorflow_bert/run_squad_wrap.py \
               --floatx=float16 \
               --predict_batch_size=8 \
               --vocab_file={} \
               --bert_config_file={} \
               --init_checkpoint={} \
               --train_file={} \
               --do_predict=True \
               --predict_file={} \
               --max_seq_length={} \
               --output_dir={} \
               --int8_mode={} \
               --remove_padding={} \
               --allow_gemm_test={}".format(vocab_file, bert_config_file, init_checkpoint, train_file, predict_file, max_seq_length, output_dir, int8_mode, remove_padding, allow_gemm_test))

def run_evaluate(file_path, truth_dataset):
    with open(file_path) as f, open(truth_dataset) as b:
        f_json = json.load(f)
        b_json = json.load(b)

        dataset = b_json['data']
        score = evaluate(dataset, f_json)
        return score

def checkF1Score(score, test_mark, f1_score_file):
    score_map = {}
    with open(f1_score_file, "r") as fin:
        for line in fin.readlines():
            parts = line.strip().split(" ")
            score_map[parts[0]] = float(parts[1])
    if test_mark in score_map:
        if abs(score - float(score_map[test_mark])) < 0.01:
            print("[INFO] TEST PASS : {} vs {}".format(score, score_map[test_mark]))
        else:
            print("[ERROR] TEST FAILED : {} vs {}".format(score, score_map[test_mark]))
    else:
        print("[ERROR] Target f1 score of {} is not in score file".format(test_mark))

if __name__ == "__main__":
    flags.DEFINE_string("vocab_file", "squad_model/vocab.txt", "vocab file")
    flags.DEFINE_string("bert_config_file", "squad_model/bert_config.json", "bert config file")
    flags.DEFINE_string("output_dir", "squad_int8_unittest_output", "dir of output")
    flags.DEFINE_string("squad_data_dir", "squad_data", "dir of squad data")
    flags.DEFINE_string("model_file", "", "the initial checkpoint")
    flags.DEFINE_string("f1_score_file", "tensorflow/unit_test/int8_f1_score", "the file contains the f1 score under different params.")
    flags.mark_flag_as_required("model_file")
    flags.DEFINE_bool("remove_padding", False, "Remove padding or Not")
    flags.DEFINE_integer("int8_mode", 0, "whether use int8 or not; and how to use int8")
    flags.DEFINE_bool("allow_gemm_test", False, "whether allow gemm test inside FT.")
    flags.DEFINE_integer("seq_len", 384, "The sequence length.")

    if not os.path.exists(flags.FLAGS.output_dir):
        os.system("mkdir "+flags.FLAGS.output_dir)
    if flags.FLAGS.model_file.endswith(".index"):
        flags.FLAGS.model_file = flags.FLAGS.model_file[:-6]

    print("[INFO] test_squad_int8")
    type_convert(flags.FLAGS.model_file, flags.FLAGS.output_dir+"/fp16");

    ckpt_quantization(flags.FLAGS.output_dir+"/fp16/model.ckpt", flags.FLAGS.output_dir+"/fp16_quantized", flags.FLAGS.int8_mode)
   
    run_int8_squad(flags.FLAGS.vocab_file, flags.FLAGS.bert_config_file, flags.FLAGS.output_dir+"/fp16_quantized/model.ckpt", flags.FLAGS.squad_data_dir+"/train-v1.1.json", flags.FLAGS.squad_data_dir+"/dev-v1.1.json", flags.FLAGS.seq_len, flags.FLAGS.output_dir, flags.FLAGS.int8_mode, flags.FLAGS.remove_padding, flags.FLAGS.allow_gemm_test)
    
    score = run_evaluate(flags.FLAGS.output_dir+"/predictions.json", flags.FLAGS.squad_data_dir+"/dev-v1.1.json")
    
    test_mark = "{}_{}".format(flags.FLAGS.int8_mode, flags.FLAGS.seq_len)
    checkF1Score(score["f1"], test_mark, flags.FLAGS.f1_score_file)

    
