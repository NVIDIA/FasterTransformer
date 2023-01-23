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

import numpy as np
import tensorflow as tf
import math
import os
from collections import namedtuple


class FTDebertaWeights(object):
    def __init__(
        self,
        config,
        tensor_para_size=1,
        pipeline_para_size=1
    ):
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size / config.num_attention_heads
        self.max_relative_positions = config.max_position_embeddings
        self.relative_position_buckets = config.position_buckets
        self.inter_size = config.intermediate_size
        self.num_layer = config.num_hidden_layers

        self.config = config
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_rank = 0  # no mpi for the moment
        self.pipeline_para_size = pipeline_para_size
        self.activation_type = config.hidden_act
        self.weights = None
        self.q_scaling = np.sqrt(3)

        assert tensor_para_size == 1, "This op only supports TP = 1 now."
        assert pipeline_para_size == 1, "This op only supports PP = 1 now."

    def load_from_model(self, model):
        """
        Routine to load DeBERTa weights from a HuggingFace model. This assumes the latest DeBERT-v2 architecture and DeBERTa-v3 weights.
        """

        start_layer = self.pipeline_para_rank * self.num_layer // self.pipeline_para_size
        end_layer = (self.pipeline_para_rank + 1) * self.num_layer // self.pipeline_para_size

        weight_data_type = {'float32': tf.float32, 'float16': tf.float16}[model.dtype]

        variables_dict = {}
        for var in model.variables:
            variables_dict[var.name] = var.numpy()

        var_prefix_model = model.name + '/deberta/'  # model-level weights
        var_prefix_layer = model.name + '/deberta/encoder/'  # layer-level weights

        # model-level weight loading
        word_embedding_table = variables_dict.get(var_prefix_model + "embeddings/word_embeddings/weight:0")
        word_embedding_layernorm_gamma = variables_dict.get(var_prefix_model + "embeddings/LayerNorm/gamma:0")
        word_embedding_layernorm_beta = variables_dict.get(var_prefix_model + "embeddings/LayerNorm/beta:0")
        relative_embedding_table = variables_dict.get(var_prefix_model + "encoder/rel_embeddings.weight:0")
        relative_embedding_layernorm_gamma = variables_dict.get(var_prefix_model + "encoder/LayerNorm/gamma:0")
        relative_embedding_layernorm_beta = variables_dict.get(var_prefix_model + "encoder/LayerNorm/beta:0")

        # layer-level weight loading
        attn_q_kernel = [variables_dict.get(
            var_prefix_layer + f"layer_._{i}/attention/self/query_proj/kernel:0") for i in range(start_layer, end_layer)]
        attn_q_bias = [variables_dict.get(
            var_prefix_layer + f"layer_._{i}/attention/self/query_proj/bias:0") for i in range(start_layer, end_layer)]

        attn_k_kernel = [variables_dict.get(
            var_prefix_layer + f"layer_._{i}/attention/self/key_proj/kernel:0") for i in range(start_layer, end_layer)]
        attn_k_bias = [variables_dict.get(
            var_prefix_layer + f"layer_._{i}/attention/self/key_proj/bias:0") for i in range(start_layer, end_layer)]

        attn_v_kernel = [variables_dict.get(
            var_prefix_layer + f"layer_._{i}/attention/self/value_proj/kernel:0") for i in range(start_layer, end_layer)]
        attn_v_bias = [variables_dict.get(
            var_prefix_layer + f"layer_._{i}/attention/self/value_proj/bias:0") for i in range(start_layer, end_layer)]

        attn_output_kernel = [variables_dict.get(
            var_prefix_layer + f"layer_._{i}/attention/output/dense/kernel:0") for i in range(start_layer, end_layer)]
        attn_output_bias = [variables_dict.get(
            var_prefix_layer + f"layer_._{i}/attention/output/dense/bias:0") for i in range(start_layer, end_layer)]

        attn_output_layernorm_gamma = [variables_dict.get(
            var_prefix_layer + f"layer_._{i}/attention/output/LayerNorm/gamma:0") for i in range(start_layer, end_layer)]
        attn_output_layernorm_beta = [variables_dict.get(
            var_prefix_layer + f"layer_._{i}/attention/output/LayerNorm/beta:0") for i in range(start_layer, end_layer)]

        inter_kernel = [variables_dict.get(
            var_prefix_layer + f"layer_._{i}/intermediate/dense/kernel:0") for i in range(start_layer, end_layer)]
        inter_bias = [variables_dict.get(
            var_prefix_layer + f"layer_._{i}/intermediate/dense/bias:0") for i in range(start_layer, end_layer)]

        output_kernel = [variables_dict.get(
            var_prefix_layer + f"layer_._{i}/output/dense/kernel:0") for i in range(start_layer, end_layer)]
        output_bias = [variables_dict.get(
            var_prefix_layer + f"layer_._{i}/output/dense/bias:0") for i in range(start_layer, end_layer)]

        output_layernorm_gamma = [variables_dict.get(
            var_prefix_layer + f"layer_._{i}/output/LayerNorm/gamma:0") for i in range(start_layer, end_layer)]
        output_layernorm_beta = [variables_dict.get(
            var_prefix_layer + f"layer_._{i}/output/LayerNorm/beta:0") for i in range(start_layer, end_layer)]

        # pack the arguments into a tuple that mirrors the TF custom OP input
        weights = [
            word_embedding_table,
            word_embedding_layernorm_gamma,
            word_embedding_layernorm_beta,
            relative_embedding_table,
            relative_embedding_layernorm_gamma,
            relative_embedding_layernorm_beta,
            attn_q_kernel,
            attn_q_bias,
            attn_k_kernel,
            attn_k_bias,
            attn_v_kernel,
            attn_v_bias,
            attn_output_kernel,
            attn_output_bias,
            attn_output_layernorm_gamma,
            attn_output_layernorm_beta,
            inter_kernel,
            inter_bias,
            output_kernel,
            output_bias,
            output_layernorm_gamma,
            output_layernorm_beta
        ]

        # clean up if there is None. Note - we cannot use np.array([0]) as TF won't accept empty tensors
        for i in range(0, len(weights)):
            if weights[i] is None:
                weights[i] = tf.constant([0], dtype=weight_data_type)
            elif type(weights[i]) is list:
                weights[i] = [tf.constant([0], dtype=weight_data_type) if w is None else tf.convert_to_tensor(
                    w, dtype=weight_data_type) for w in weights[i]]
            else:
                weights[i] = tf.convert_to_tensor(weights[i], dtype=weight_data_type)

        self.weights = tuple(weights)


class FTDebertaModel():
    def __init__(self, lib_path, params):
        self.transformer_op_module = tf.load_op_library(lib_path)
        self.params = params

    def __call__(self, input_ids, seq_len, remove_padding=True):
        return self.forward(input_ids, seq_len, remove_padding=remove_padding)

    def forward(self, input_ids, seq_len, remove_padding=True):
        outputs = self.transformer_op_module.deberta(input_ids,
                                                     seq_len,
                                                     *self.params.weights,
                                                     head_num=self.params.num_heads,
                                                     size_per_head=self.params.head_size,
                                                     max_relative_positions=self.params.max_relative_positions,
                                                     relative_position_buckets=self.params.relative_position_buckets,
                                                     inter_size=self.params.inter_size,
                                                     num_layer=self.params.num_layer,
                                                     remove_padding=remove_padding,
                                                     q_scaling=self.params.q_scaling)
        return outputs


class FTHFDebertaModel():
    def __init__(self, ft_model, remove_padding=True):
        self.model = ft_model
        self.remove_padding = remove_padding

    def __call__(self, input_ids, attention_mask, **kwargs):
        seq_len = tf.reduce_sum(attention_mask, axis=1)
        outputs = self.model.forward(input_ids, seq_len, remove_padding=self.remove_padding)

        # to match HF structure
        FTOutput = namedtuple("FTOutput", ["output", "hidden_states", "attentions"])
        o = FTOutput(outputs, None, None)
        return o
