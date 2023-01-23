# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

class FTT5EncoderParams(object):

    def __init__(
        self,
        config,
        tensor_para_size=1,
        pipeline_para_size=1,
        *,
        t5_with_bias=False,
        position_embedding_type=None
    ):
        self.num_heads = config.num_heads
        self.head_size = config.d_kv
        self.inter_size = config.d_ff
        self.d_model = config.d_model
        self.num_layer = config.num_layers
        self.num_bucket = config.relative_attention_num_buckets if hasattr(config, 'relative_attention_num_buckets') else 32
        self.max_distance = config.relative_attention_max_distance if hasattr(config, 'relative_attention_max_distance') else 128
        self.config = config
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_rank = 0 # no mpi for the moment
        self.pipeline_para_size = pipeline_para_size
        self.t5_with_bias = t5_with_bias
        self.activation_type = config.feed_forward_proj
        self.weights = None
        self.q_scaling = 1.0 / (math.sqrt(config.d_kv))
        # relative position embedding -> 0, absolute position embedding -> 1
        assert tensor_para_size == 1, "This op only supports TP = 1 now."
        assert pipeline_para_size == 1, "This op only supports PP = 1 now."
        self.position_embedding_type = position_embedding_type or 0

    def load_from_model(self, model):
        """
        Routine to load T5 encoder weights from a HuggingFace model. This assumes the regular T5 (NOT v1.1) architecture.
    
        Notes:
        - Note that FasterTransformer currently doesn't support gated GELU.
        - The relative attention bias is transposed with respect to the HF model.
        """

        start_layer = self.pipeline_para_rank * self.num_layer // self.pipeline_para_size
        end_layer = (self.pipeline_para_rank + 1) * self.num_layer // self.pipeline_para_size

        weight_data_type = {'float32': tf.float32, 'float16': tf.float16}[model.dtype]

        variables_dict = {}
        for var in model.variables:
            variables_dict[var.name] = var.numpy()
        
        var_prefix = model.name + '/encoder/'

        # fill the datastructures holding the weights:

        # layer_._0/layer_norm
        attr_output_layernorm_beta  = [variables_dict.get(var_prefix + f"block_._{i}/layer_._0/layer_norm/bias:0") for i in range(start_layer, end_layer)]
        attr_output_layernorm_gamma = [variables_dict.get(var_prefix + f"block_._{i}/layer_._0/layer_norm/weight:0") for i in range(start_layer, end_layer)]
        
        # layer_._0/SelfAttention/q
        attr_q_kernel               = [variables_dict.get(var_prefix + f"block_._{i}/layer_._0/SelfAttention/q/kernel:0") for i in range(start_layer, end_layer)]
        attr_q_bias                 = [variables_dict.get(var_prefix + f"block_._{i}/layer_._0/SelfAttention/q/bias:0") for i in range(start_layer, end_layer)]
        
        # layer_._0/SelfAttention/k
        attr_k_kernel               = [variables_dict.get(var_prefix + f"block_._{i}/layer_._0/SelfAttention/k/kernel:0") for i in range(start_layer, end_layer)]
        attr_k_bias                 = [variables_dict.get(var_prefix + f"block_._{i}/layer_._0/SelfAttention/k/bias:0") for i in range(start_layer, end_layer)]

        # layer_._0/SelfAttention/v
        attr_v_kernel               = [variables_dict.get(var_prefix + f"block_._{i}/layer_._0/SelfAttention/v/kernel:0") for i in range(start_layer, end_layer)]
        attr_v_bias                 = [variables_dict.get(var_prefix + f"block_._{i}/layer_._0/SelfAttention/v/bias:0") for i in range(start_layer, end_layer)]

        # layer_._0/SelfAttention/o
        attr_output_kernel          = [variables_dict.get(var_prefix + f"block_._{i}/layer_._0/SelfAttention/o/kernel:0") for i in range(start_layer, end_layer)]
        attr_output_bias            = [variables_dict.get(var_prefix + f"block_._{i}/layer_._0/SelfAttention/o/bias:0") for i in range(start_layer, end_layer)]

        # layer_._1/layer_norm
        ffn_output_layernorm_beta   = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/layer_norm/bias:0") for i in range(start_layer, end_layer)]
        ffn_output_layernorm_gamma  = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/layer_norm/weight:0") for i in range(start_layer, end_layer)]

        if self.config.feed_forward_proj == "relu" or self.config.feed_forward_proj == "gelu":
            # format of t5-small
            # layer_._1/DenseReluDense/wi
            ffn_inter_kernel            = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/DenseReluDense/wi/kernel:0") for i in range(start_layer, end_layer)]
            ffn_inter_bias              = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/DenseReluDense/wi/bias:0") for i in range(start_layer, end_layer)]

            ffn_inter2_kernel           = [tf.constant([0], dtype=weight_data_type) for i in range(start_layer, end_layer)]
            ffn_inter2_bias             = [tf.constant([0], dtype=weight_data_type) for i in range(start_layer, end_layer)]
        elif self.config.feed_forward_proj == "gated-relu" or self.config.feed_forward_proj == "gated-gelu":
            # format of google/t5-v1_1-small
            # layer_._1/DenseReluDense/wi_0
            ffn_inter_kernel            = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/DenseReluDense/wi_0/kernel:0") for i in range(start_layer, end_layer)]
            ffn_inter_bias              = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/DenseReluDense/wi_0/bias:0") for i in range(start_layer, end_layer)]
            
            # layer_._1/DenseReluDense/wi_1
            # only applies to gated models
            ffn_inter2_kernel           = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/DenseReluDense/wi_1/kernel:0") for i in range(start_layer, end_layer)]
            ffn_inter2_bias             = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/DenseReluDense/wi_1/bias:0") for i in range(start_layer, end_layer)]
        else:
            assert False, f"FT does not support activation type {self.config.feed_forward_proj}"

        # layer_._1/DenseReluDense/wo
        ffn_output_kernel           = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/DenseReluDense/wo/kernel:0") for i in range(start_layer, end_layer)]
        ffn_output_bias             = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/DenseReluDense/wo/bias:0") for i in range(start_layer, end_layer)]
        
        # final_layer_norm/weight:0
        output_layernorm_beta        = variables_dict.get(var_prefix + f"final_layer_norm/bias:0")
        output_layernorm_gamma       = variables_dict.get(var_prefix + f"final_layer_norm/weight:0")

        # other weights
        output_absolute_or_relative_position_embedding = np.transpose(variables_dict.get(var_prefix + f"block_._{0}/layer_._0/SelfAttention/relative_attention_bias/embeddings:0"))
        output_embedding_table = model.get_input_embeddings().weight

        # pack the arguments into a tuple that mirrors the TF custom OP input
        weights = [
            attr_output_layernorm_beta,
            attr_output_layernorm_gamma,
            attr_q_kernel,
            attr_q_bias,
            attr_k_kernel,
            attr_k_bias,
            attr_v_kernel,
            attr_v_bias,
            attr_output_kernel,
            attr_output_bias,
            ffn_output_layernorm_beta,
            ffn_output_layernorm_gamma,
            ffn_inter_kernel,
            ffn_inter_bias,
            ffn_inter2_kernel,
            ffn_inter2_bias,
            ffn_output_kernel,
            ffn_output_bias,
            output_layernorm_beta,
            output_layernorm_gamma,
            output_absolute_or_relative_position_embedding,
            output_embedding_table
        ]

        # clean up if there is None. Note - we cannot use np.array([0]) as TF won't accept empty tensors
        for i in range(0, len(weights)):
            if weights[i] is None:
                weights[i] = tf.constant([0], dtype=weight_data_type)
            elif type(weights[i]) is list:
                weights[i] = [tf.constant([0], dtype=weight_data_type) if w is None else tf.convert_to_tensor(w, dtype=weight_data_type) for w in weights[i]]
            else:
                weights[i] = tf.convert_to_tensor(weights[i], dtype=weight_data_type)
                
        self.weights = tuple(weights)

# wrapper function
def ftt5_encoder(inputs, seq_len, encoder_params):
    transformer_op_module = tf.load_op_library(os.path.join('./lib/libtf_t5.so'))

    outputs = transformer_op_module.t5_encoder(inputs,
                                               seq_len,
                                               *encoder_params.weights,
                                               head_num = encoder_params.num_heads,
                                               head_size = encoder_params.head_size, # encoder_config.d_kv
                                               inter_size = encoder_params.inter_size, # encoder_config.d_ff,
                                               num_layer = encoder_params.num_layer,
                                               d_model = encoder_params.d_model,
                                               num_bucket = encoder_params.num_bucket,
                                               max_distance = encoder_params.max_distance,
                                               remove_padding = True,
                                               t5_with_bias = encoder_params.t5_with_bias,
                                               activation_type = encoder_params.activation_type,
                                               q_scaling = encoder_params.q_scaling,
                                               position_embedding_type=encoder_params.position_embedding_type)

    return outputs
