# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
FasterTransformer library.

This sample code builds a DeBERTa transformer model by TensorFlow and FasterTransformer's TensorFlow  custom op. Then compare the results on random inputs to verify the correctness of FasterTransformer implementation.
Note that DeBERTa FasterTransformer implementation does not include pooling layer or downstream task heads. Therefore the comparison was made on the raw hidden states from the DeBERTa encoder model.

Users are also able to use this sample code to test the average forward time of 
TensorFlow and FasterTransformer. 
'''

import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from transformers import DebertaV2Tokenizer, TFDebertaV2ForSequenceClassification

dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = dir_path + "/../../.."
sys.path.append(ROOT_DIR)
from examples.tensorflow.deberta.utils.ft_deberta import FTDebertaWeights, FTDebertaModel, FTHFDebertaModel
from examples.tensorflow.bert.utils.common import cross_check

def main(args):
    model_name = args['model']
    batch_size = args['batch_size']

    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

    # Model setup - Huggingface TensorFlow
    model_tf = TFDebertaV2ForSequenceClassification.from_pretrained(model_name)

    # Model setup - FasterTransformer
    lib_path = os.path.join(ROOT_DIR, './build/lib/libtf_deberta.so')

    ft_deberta_weight = FTDebertaWeights(model_tf.config, tensor_para_size=1, pipeline_para_size=1)
    ft_deberta_weight.load_from_model(model_tf)
    ft_deberta = FTDebertaModel(lib_path, ft_deberta_weight)

    # Random input
    random_sentences = tokenizer.batch_decode([np.random.randint(1, model_tf.config.vocab_size, size=np.random.randint(
        1, model_tf.config.max_position_embeddings)) for _ in range(batch_size)])
    inputs = tokenizer(random_sentences, padding=True, return_tensors="tf")

    # Inference and simple timing
    measurement_iters = 10
    tf_latencies = []
    ft_latencies = []

    # TF E2E
    for _ in range(measurement_iters):
        start_time = time.time()
        output_tf = model_tf(**inputs)
        end_time = time.time()
        tf_latencies.append(end_time - start_time)
    tf_p50 = np.percentile(tf_latencies, 50)
    tf_p99 = np.percentile(tf_latencies, 99)

    logits_tf = output_tf.logits
    # print("TF results: ", logits_tf)
    # predicted_class_id = int(tf.math.argmax(logits_tf, axis=-1)[0])
    # print(model.config.id2label[predicted_class_id])

    # FT E2E
    # trick to wrap FT inside HF by replacing TF layer, see ft_deberta.py
    model_tf.deberta = FTHFDebertaModel(ft_deberta, remove_padding=True)
    # w/ padding removal by default i.e., Effective Transformer

    for _ in range(measurement_iters):
        start_time = time.time()
        output_ft = model_tf(**inputs)
        end_time = time.time()
        ft_latencies.append(end_time - start_time)
    ft_p50 = np.percentile(ft_latencies, 50)
    ft_p99 = np.percentile(ft_latencies, 99)

    logits_ft = output_ft.logits
    # print("FT results: ", logits_ft)

    print(f"TF p50: {tf_p50*1000:.2f} ms, p99: {tf_p99*1000:.2f} ms ")
    print(f"FT p50: {ft_p50*1000:.2f} ms, p99: {ft_p99*1000:.2f} ms ")

    # Correctness check
    atol_threshold = 3e-3
    cross_check("TF v.s. FT", logits_tf, logits_ft, atol_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-model', '--model', type=str, default="microsoft/deberta-v3-base", metavar='STRING',
    help='DeBERTa-V3 model variants. Note DeBERTa-V2 and -V1 variants are both slightly different from V3, thus not supported in the current example yet')
    # not tested for the moment and not supported
    parser.add_argument('-tensor_para_size', '--tensor_para_size', type=int, default=1, metavar='NUMBER',
                        help='size of tensor parallelism (default: 1). This feature hasn\'t been tested.')
    parser.add_argument('-pipeline_para_size', '--pipeline_para_size', type=int, default=1, metavar='NUMBER',
                        help='size of pipeline parallelism (default: 1). This feature hasn\'t been tested.')
    args = parser.parse_args()

    main(vars(args))