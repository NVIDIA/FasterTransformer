# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
import math

transformer_op_module = tf.load_op_library(os.path.join('./lib/libtf_t5.so'))

from utils.ft_encoder import FTT5EncoderParams, ftt5_encoder

class FTT5DecodingParams(object):

    def __init__(
        self,
        config,
        tensor_para_size=1,
        pipeline_para_size=1,
        *,
        t5_with_bias=False,
        position_embedding_type=None,
        tie_word_embeddings=None
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
        self.start_id = config.decoder_start_token_id
        self.end_id = config.eos_token_id
        self.vocab_size = config.vocab_size
        assert tensor_para_size == 1, "This op only supports TP = 1 now."
        assert pipeline_para_size == 1, "This op only supports PP = 1 now."
        self.position_embedding_type = position_embedding_type or 0
        self.tie_word_embeddings = tie_word_embeddings or False

    def load_from_model(self, model):
        """
        Routine to load T5 decoding weights from a HuggingFace model. This assumes the regular T5 (NOT v1.1) architecture.
    
        Notes:
        - Note that FasterTransformer currently doesn't support gated GELU.
        - The relative attention bias is transposed with respect to the HF model.
        """

        # for the moment obsolete. everything runs on single GPU
        start_layer = self.pipeline_para_rank * self.num_layer // self.pipeline_para_size
        end_layer = (self.pipeline_para_rank + 1) * self.num_layer // self.pipeline_para_size

        weight_data_type = {'float32': tf.float32, 'float16': tf.float16}[model.dtype]

        variables_dict = {}
        for var in model.variables:
            variables_dict[var.name] = var.numpy()
        
        var_prefix = model.name + '/decoder/'

        # fill the datastructures holding the weights:
        # layer_._0/layer_norm
        pre_layernorm_beta          = [variables_dict.get(var_prefix + f"block_._{i}/layer_._0/layer_norm/bias:0") for i in range(start_layer, end_layer)]
        pre_layernorm_gamma         = [variables_dict.get(var_prefix + f"block_._{i}/layer_._0/layer_norm/weight:0") for i in range(start_layer, end_layer)]
        
        # layer_._0/SelfAttention/q
        self_qkv_kernel             = [tf.stack([
                                        variables_dict.get(var_prefix + f"block_._{i}/layer_._0/SelfAttention/q/kernel:0"),
                                        variables_dict.get(var_prefix + f"block_._{i}/layer_._0/SelfAttention/k/kernel:0"),
                                        variables_dict.get(var_prefix + f"block_._{i}/layer_._0/SelfAttention/v/kernel:0")
                                      ], -2) for i in range(start_layer, end_layer)]
        self_qkv_bias               = [tf.stack([
                                        variables_dict.get(var_prefix + f"block_._{i}/layer_._0/SelfAttention/q/bias:0") or tf.constant([0], dtype=weight_data_type),
                                        variables_dict.get(var_prefix + f"block_._{i}/layer_._0/SelfAttention/k/bias:0") or tf.constant([0], dtype=weight_data_type),
                                        variables_dict.get(var_prefix + f"block_._{i}/layer_._0/SelfAttention/v/bias:0") or tf.constant([0], dtype=weight_data_type)
                                      ], -2) for i in range(start_layer, end_layer)]        

        # layer_._0/SelfAttention/o
        self_output_kernel          = [variables_dict.get(var_prefix + f"block_._{i}/layer_._0/SelfAttention/o/kernel:0") for i in range(start_layer, end_layer)]
        self_output_bias            = [variables_dict.get(var_prefix + f"block_._{i}/layer_._0/SelfAttention/o/bias:0") for i in range(start_layer, end_layer)]

        # layer_._1/layer_norm
        self_layernorm_beta         = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/layer_norm/bias:0") for i in range(start_layer, end_layer)]
        self_layernorm_gamma        = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/layer_norm/weight:0") for i in range(start_layer, end_layer)]

        # layer_._1/EncDecAttention/q
        cross_q_kernel              = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/EncDecAttention/q/kernel:0") for i in range(start_layer, end_layer)]
        cross_q_bias                = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/EncDecAttention/q/bias:0") for i in range(start_layer, end_layer)]
        
        # layer_._1/EncDecAttention/k
        cross_k_kernel              = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/EncDecAttention/k/kernel:0") for i in range(start_layer, end_layer)]
        cross_k_bias                = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/EncDecAttention/k/bias:0") for i in range(start_layer, end_layer)]

        # layer_._1/EncDecAttention/v
        cross_v_kernel              = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/EncDecAttention/v/kernel:0") for i in range(start_layer, end_layer)]
        cross_v_bias                = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/EncDecAttention/v/bias:0") for i in range(start_layer, end_layer)]

        # layer_._1/EncDecAttention/o
        cross_output_kernel         = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/EncDecAttention/o/kernel:0") for i in range(start_layer, end_layer)]
        cross_output_bias           = [variables_dict.get(var_prefix + f"block_._{i}/layer_._1/EncDecAttention/o/bias:0") for i in range(start_layer, end_layer)]

        # layer_._2/layer_norm
        cross_layernorm_beta        = [variables_dict.get(var_prefix + f"block_._{i}/layer_._2/layer_norm/bias:0") for i in range(start_layer, end_layer)]
        cross_layernorm_gamma       = [variables_dict.get(var_prefix + f"block_._{i}/layer_._2/layer_norm/weight:0") for i in range(start_layer, end_layer)]

        if self.config.feed_forward_proj == "relu" or self.config.feed_forward_proj == "gelu":
            # format of t5-small
            # layer_._2/DenseReluDense/wi
            ffn_inter_kernel            = [variables_dict.get(var_prefix + f"block_._{i}/layer_._2/DenseReluDense/wi/kernel:0") for i in range(start_layer, end_layer)]
            ffn_inter_bias              = [variables_dict.get(var_prefix + f"block_._{i}/layer_._2/DenseReluDense/wi/bias:0") for i in range(start_layer, end_layer)]

            ffn_inter2_kernel           = [tf.constant([0], dtype=weight_data_type) for i in range(start_layer, end_layer)]
            ffn_inter2_bias             = [tf.constant([0], dtype=weight_data_type) for i in range(start_layer, end_layer)]
        elif self.config.feed_forward_proj == "gated-relu" or self.config.feed_forward_proj == "gated-gelu":
            # format of google/t5-v1_1-small
            # layer_._2/DenseReluDense/wi_0
            ffn_inter_kernel            = [variables_dict.get(var_prefix + f"block_._{i}/layer_._2/DenseReluDense/wi_0/kernel:0") for i in range(start_layer, end_layer)]
            ffn_inter_bias              = [variables_dict.get(var_prefix + f"block_._{i}/layer_._2/DenseReluDense/wi_0/bias:0") for i in range(start_layer, end_layer)]
            
            # layer_._2/DenseReluDense/wi_1
            # only applies to gated models
            ffn_inter2_kernel           = [variables_dict.get(var_prefix + f"block_._{i}/layer_._2/DenseReluDense/wi_1/kernel:0") for i in range(start_layer, end_layer)]
            ffn_inter2_bias             = [variables_dict.get(var_prefix + f"block_._{i}/layer_._2/DenseReluDense/wi_1/bias:0") for i in range(start_layer, end_layer)]
        else:
            assert False, f"FT does not support activation type {self.config.feed_forward_proj}"

        # layer_._2/DenseReluDense/wo
        ffn_output_kernel           = [variables_dict.get(var_prefix + f"block_._{i}/layer_._2/DenseReluDense/wo/kernel:0") for i in range(start_layer, end_layer)]
        ffn_output_bias             = [variables_dict.get(var_prefix + f"block_._{i}/layer_._2/DenseReluDense/wo/bias:0") for i in range(start_layer, end_layer)]
        
        # final_layer_norm/weight:0
        output_layernorm_beta       = variables_dict.get(var_prefix + f"final_layer_norm/bias:0")
        output_layernorm_gamma      = variables_dict.get(var_prefix + f"final_layer_norm/weight:0")

        # other weights
        pre_encoder_embedding_table = model.get_input_embeddings().weight
        if variables_dict.get(f"tft5_for_conditional_generation/lm_head/kernel:0") is not None:
            # format of google/t5-v1_1-small
            # In t5 v1_1, pre_encoder_embedding_table and post_decoder_embedding_kernel are different
            post_decoder_embedding_kernel = variables_dict.get(f"tft5_for_conditional_generation/lm_head/kernel:0").transpose()
            post_decoder_embedding_bias = variables_dict.get(f"tft5_for_conditional_generation/lm_head//bias:0" or tf.constant([0], dtype=weight_data_type))
        else:
            # format of t5-small
            post_decoder_embedding_kernel = variables_dict.get(f"shared/shared/weight:0")
            post_decoder_embedding_bias = variables_dict.get(f"shared/shared/bias:0" or tf.constant([0], dtype=weight_data_type))
        output_absolute_or_relative_position_embedding = np.transpose(variables_dict.get(var_prefix + f"block_._{0}/layer_._0/SelfAttention/relative_attention_bias/embeddings:0"))

        # # pack the arguments into a tuple that mirrors the TF custom OP input
        weights = [
            pre_layernorm_beta,
            pre_layernorm_gamma,
            self_qkv_kernel,
            self_qkv_bias,
            self_output_kernel,
            self_output_bias,
            self_layernorm_beta,
            self_layernorm_gamma,
            cross_q_kernel,
            cross_q_bias,
            cross_k_kernel,
            cross_k_bias,
            cross_v_kernel,
            cross_v_bias,
            cross_output_kernel,
            cross_output_bias,
            cross_layernorm_beta,
            cross_layernorm_gamma,
            ffn_inter_kernel,
            ffn_inter_bias,
            ffn_inter2_kernel,
            ffn_inter2_bias,
            ffn_output_kernel,
            ffn_output_bias,
            output_layernorm_beta,
            output_layernorm_gamma,
            pre_encoder_embedding_table,
            post_decoder_embedding_kernel,
            post_decoder_embedding_bias,
            output_absolute_or_relative_position_embedding
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

def ftt5_decoding(mem_hidden_states, mem_seq_len, decoding_params, max_seq_len, beam_width, top_k = 1, top_p = 0.0,
                    beam_search_diversity_rate = 0.0, temperature = 1.0, len_penalty = 0.0, repetition_penalty = 1.0,
                    random_seed = 0, return_cum_log_probs = False, return_output_log_probs = False):

    outputs = transformer_op_module.t5_decoding(mem_hidden_states,
                                                mem_seq_len,
                                                *decoding_params.weights,
                                                max_seq_len = max_seq_len,
                                                beam_width = beam_width,
                                                head_num = decoding_params.num_heads,
                                                head_size = decoding_params.head_size,
                                                inter_size = decoding_params.inter_size,
                                                num_layer = decoding_params.num_layer,
                                                d_model = decoding_params.d_model,
                                                num_bucket = decoding_params.num_bucket,
                                                max_distance = decoding_params.max_distance,
                                                start_id = decoding_params.start_id,
                                                end_id = decoding_params.end_id,
                                                beam_search_diversity_rate = beam_search_diversity_rate,
                                                top_k = top_k,
                                                top_p = top_p,
                                                temperature = temperature,
                                                len_penalty = len_penalty,
                                                repetition_penalty = repetition_penalty,
                                                return_cum_log_probs = return_cum_log_probs,
                                                return_output_log_probs = return_output_log_probs,
                                                t5_with_bias = decoding_params.t5_with_bias,
                                                activation_type = decoding_params.activation_type,
                                                q_scaling = decoding_params.q_scaling,
                                                position_embedding_type = decoding_params.position_embedding_type,
                                                random_seed = random_seed,
                                                tie_word_embeddings = decoding_params.tie_word_embeddings)
        
    return outputs

class FTT5Model():
    def __init__(self, encoder_params, decoding_params):
        self.encoder_params = encoder_params
        self.decoding_params = decoding_params

    def compute(self, input_tokens, beam_width, max_seq_len, top_k=1, top_p = 0.0, beam_search_diversity_rate = 0.0,
                temperature = 1.0, len_penalty = 0.0, repetition_penalty = 1.0, random_seed=0):

        input_ids = tf.cast(input_tokens.input_ids, tf.int32) # maybe convert to int32
        
        mem_seq_len = 0
        if hasattr(input_tokens, "attention_mask"):
            mem_seq_len = np.sum(input_tokens.attention_mask, axis=1)
        else:
            mem_seq_len = input_tokens.seq_len
        mem_seq_len = tf.cast(mem_seq_len, tf.int32)

        encoder_outputs = ftt5_encoder(input_ids, mem_seq_len, self.encoder_params)
        ft_decoding_output_ids, ft_decoding_seq_lens, ft_output_log_probs, ft_cum_log_probs = ftt5_decoding(encoder_outputs,
                                                                                                            mem_seq_len,
                                                                                                            self.decoding_params,
                                                                                                            max_seq_len,
                                                                                                            beam_width,
                                                                                                            top_k,
                                                                                                            top_p,
                                                                                                            beam_search_diversity_rate,
                                                                                                            temperature,
                                                                                                            len_penalty, 
                                                                                                            repetition_penalty,
                                                                                                            random_seed=random_seed)

        ft_decoding_output_ids = tf.reshape(ft_decoding_output_ids, [-1, beam_width, max_seq_len])
        ft_decoding_seq_lens = tf.reshape(ft_decoding_seq_lens, [-1, beam_width])

        return ft_decoding_output_ids.numpy(), ft_decoding_seq_lens.numpy()