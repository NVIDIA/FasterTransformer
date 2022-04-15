/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/models/longformer/LongformerEncoder.h"
#include "src/fastertransformer/th_op/th_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

class FasterTransformerLongformerEncoder: public th::jit::CustomClassHolder {
private:
    size_t layer_num_;
    size_t in_dim_;
    size_t head_num_;
    size_t size_per_head_;
    size_t intermediate_size_;
    size_t local_attn_window_size_;
    size_t max_global_token_num_;
    size_t max_batch_size_;
    size_t max_seq_len_;
    float attn_scaler_;
    size_t hidden_units_;

    cublasLtHandle_t _cublasltHandle;
    ft::cublasAlgoMap* cublas_algo_map_;
    std::mutex* cublas_wrapper_mutex_;

public:
    FasterTransformerLongformerEncoder(int64_t layer_num,
                                       int64_t in_dim,
                                       int64_t head_num,
                                       int64_t size_per_head,
                                       int64_t intermediate_size,
                                       int64_t local_attn_window_size,
                                       int64_t max_global_token_num,
                                       int64_t max_batch_size,
                                       int64_t max_seq_len,
                                       double attn_scaler);

    ~FasterTransformerLongformerEncoder();

    th::Tensor forward(th::Tensor input,
                       th::Tensor local_attn_mask,
                       th::Tensor global_attn_mask,
                       th::Tensor th_weights,
                       int64_t device_id = 0);

    template<typename T>
    void setWeight(int layer_num,
                   int in_dim,
                   int hidden_units,
                   int intermediate_size,
                   th::Tensor th_weights,
                   std::vector<ft::LongformerLayerWeight<T>>* weights)
    {
        weights->clear();
        weights->resize(layer_num);
        auto weights_ptr = get_ptr<T>(th_weights);
        int offside = 0;
        for (int i = 0; i < layer_num; i++) {
            // q k v kg vg weights and bias should be continous, required by the ft longformer encoder.
            weights->at(i).query_weights.kernel = weights_ptr + offside;  // q
            offside += (i == 0 ? in_dim : hidden_units) * hidden_units;
            weights->at(i).key_weights.kernel = weights_ptr + offside;  // k
            offside += (i == 0 ? in_dim : hidden_units) * hidden_units;
            weights->at(i).value_weights.kernel = weights_ptr + offside;  // v
            offside += (i == 0 ? in_dim : hidden_units) * hidden_units;
            weights->at(i).global_key_weights.kernel = weights_ptr + offside;  // kg
            offside += (i == 0 ? in_dim : hidden_units) * hidden_units;
            weights->at(i).global_value_weights.kernel = weights_ptr + offside;  // vg
            offside += (i == 0 ? in_dim : hidden_units) * hidden_units;

            weights->at(i).global_query_weights.kernel = weights_ptr + offside;  // qg
            offside += (i == 0 ? in_dim : hidden_units) * hidden_units;

            weights->at(i).query_weights.bias = weights_ptr + offside;  // q
            offside += hidden_units;
            weights->at(i).key_weights.bias = weights_ptr + offside;  // k
            offside += hidden_units;
            weights->at(i).value_weights.bias = weights_ptr + offside;  // v
            offside += hidden_units;
            weights->at(i).global_key_weights.bias = weights_ptr + offside;  // kg
            offside += hidden_units;
            weights->at(i).global_value_weights.bias = weights_ptr + offside;  // vg
            offside += hidden_units;

            weights->at(i).global_query_weights.bias = weights_ptr + offside;  // qg
            offside += hidden_units;

            weights->at(i).attention_output_weights.kernel = weights_ptr + offside;  // attn output
            offside += hidden_units * hidden_units;
            weights->at(i).attention_output_weights.bias = weights_ptr + offside;  // attn output
            offside += hidden_units;

            weights->at(i).attention_output_layernorm_weights.gamma = weights_ptr + offside;  // attn output layernorm
            offside += hidden_units;
            weights->at(i).attention_output_layernorm_weights.beta = weights_ptr + offside;  // attn output layernorm
            offside += hidden_units;

            weights->at(i).ffn_weights.intermediate_weight.kernel = weights_ptr + offside;  // inter
            offside += hidden_units * intermediate_size;
            weights->at(i).ffn_weights.intermediate_weight.bias = weights_ptr + offside;  // inter
            offside += intermediate_size;

            weights->at(i).ffn_weights.output_weight.kernel = weights_ptr + offside;  // output
            offside += hidden_units * intermediate_size;
            weights->at(i).ffn_weights.output_weight.bias = weights_ptr + offside;  // output
            offside += hidden_units;

            weights->at(i).output_layernorm_weights.gamma = weights_ptr + offside;  // output layernorm
            offside += hidden_units;
            weights->at(i).output_layernorm_weights.beta = weights_ptr + offside;  // output layernorm
            offside += hidden_units;
        }
    }
};

}  // namespace torch_ext