/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/t5/T5Encoder.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include "src/fastertransformer/utils/prompt_learning.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

class IFT5Encoder {
public:
    virtual ~IFT5Encoder() {}
    virtual void forward(size_t                   batch_size,
                         size_t                   seq_len,
                         th::optional<th::Tensor> input,
                         th::Tensor&              sequence_lengths,
                         th::optional<th::Tensor> inputs_embeds,
                         th::Tensor&              output,
                         bool                     removing_padding) = 0;
};

template<typename T>
class FTT5Encoder: public IFT5Encoder {
public:
    FTT5Encoder(int64_t                        head_num,
                int64_t                        head_size,
                int64_t                        inter_size,
                int64_t                        d_model,
                int64_t                        layer_num,
                int64_t                        num_bucket,
                int64_t                        expert_num,
                int64_t                        max_distance,
                bool                           sparse,
                float                          q_scaling,
                int64_t                        moe_k,
                int64_t                        tensor_para_size,
                int64_t                        pipeline_para_size,
                bool                           t5_with_bias,
                ft::PositionEmbeddingType      position_embedding_type,
                ft::ActivationType             activation_type,
                int64_t                        adapter_inter_size,
                ft::LayerNormType              adapter_layer_norm_type,
                const std::vector<int64_t>     moe_layer_index,
                const std::vector<th::Tensor>& w);

    ~FTT5Encoder() override
    {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        ft::ftNcclParamDestroy(tensor_para_);
        ft::ftNcclParamDestroy(pipeline_para_);
        cublasLtDestroy(_cublasltHandle);
#ifdef SPARSITY_ENABLED
        if (_sparse) {
            cusparseLtDestroy(&_cusparseLtHandle);
        }
#endif
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void forward(size_t                   batch_size,
                 size_t                   seq_len,
                 th::optional<th::Tensor> input_ids,
                 th::Tensor&              sequence_lengths,
                 th::optional<th::Tensor> inputs_embeds,
                 th::Tensor&              output,
                 bool                     removing_padding) override;

private:
    const int64_t             _head_num;
    const int64_t             _head_size;
    const int64_t             _inter_size;
    const int64_t             _d_model;
    const int64_t             _layer_num;
    const int64_t             _num_bucket;
    const int64_t             _expert_num;
    const int64_t             _max_distance;
    const int64_t             _moe_k;
    std::vector<th::Tensor>   _weights;
    bool                      _t5_with_bias;
    std::vector<int64_t>      _moe_layer_index;
    ft::PositionEmbeddingType _position_embedding_type;
    ft::ActivationType        _activation_type;
    int64_t                   _adapter_inter_size;
    ft::LayerNormType         _adapter_layer_norm_type;
    bool                      _sparse;
    const float               _q_scaling;
    int                       sm_;
    cublasLtHandle_t          _cublasltHandle;
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t _cusparseLtHandle;
#endif
    std::mutex*            cublas_wrapper_mutex_;
    ft::cublasAlgoMap*     cublas_algo_map_;
    ft::T5EncoderWeight<T> t5_encoder_weights;

    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;
};

class FasterTransformerT5Encoder: public th::jit::CustomClassHolder {
public:
    FasterTransformerT5Encoder(th::Tensor           attr_output_layernorm_gamma,              // 0
                               th::Tensor           q_kernel,                                 // 1
                               th::Tensor           k_kernel,                                 // 2
                               th::Tensor           v_kernel,                                 // 3
                               th::Tensor           attr_output_kernel,                       // 4
                               th::Tensor           output_layernorm_gamma,                   // 5
                               th::Tensor           inter_kernel,                             // 6
                               th::Tensor           inter_kernel2,                            // 7
                               th::Tensor           output_kernel,                            // 8
                               th::Tensor           post_transformer_layernorm_gamma,         // 9
                               th::Tensor           absolute_or_relative_position_embedding,  // 10
                               th::Tensor           embedding_table,                          // 11
                               th::Tensor           attr_output_layernorm_beta,               // 12
                               th::Tensor           q_bias,                                   // 13
                               th::Tensor           k_bias,                                   // 14
                               th::Tensor           v_bias,                                   // 15
                               th::Tensor           attr_output_bias,                         // 16
                               th::Tensor           output_layernorm_beta,                    // 17
                               th::Tensor           inter_bias,                               // 18
                               th::Tensor           inter_bias2,                              // 19
                               th::Tensor           output_bias,                              // 20
                               th::Tensor           post_transformer_layernorm_beta,          // 21
                               th::Tensor           moe_gate,                                 // 22
                               th::Tensor           after_attn_adapter_weight_in,             // 23
                               th::Tensor           after_attn_adapter_weight_out,            // 24
                               th::Tensor           after_attn_adapter_layernorm_gamma,       // 25
                               th::Tensor           after_attn_adapter_layernorm_beta,        // 26
                               th::Tensor           after_ffn_adapter_weight_in,              // 27
                               th::Tensor           after_ffn_adapter_weight_out,             // 28
                               th::Tensor           after_ffn_adapter_layernorm_gamma,        // 29
                               th::Tensor           after_ffn_adapter_layernorm_beta,         // 30
                               std::vector<int64_t> moe_layer_index,
                               int64_t              head_num,
                               int64_t              head_size,
                               int64_t              inter_size,
                               int64_t              d_model,
                               bool                 remove_padding,
                               int64_t              layer_num,
                               int64_t              num_bucket,
                               int64_t              expert_num,
                               int64_t              max_distance,
                               bool                 sparse,
                               double               q_scaling,
                               int64_t              tensor_para_size,
                               int64_t              pipeline_para_size,
                               bool                 t5_with_bias,
                               int64_t              position_embedding_type,
                               int64_t              moe_k,
                               std::string          activation_type,
                               int64_t              adapter_inter_size,
                               std::string          adapter_norm_position);

    ~FasterTransformerT5Encoder();

    th::Tensor
    forward(th::optional<th::Tensor> input, th::Tensor sequence_lengths, th::optional<th::Tensor> input_embeds);

    std::vector<th::Tensor> get_pickle_info() const;

private:
    const at::ScalarType    _st;
    bool                    _remove_padding;
    int64_t                 d_model_;
    IFT5Encoder*            ft_t5_encoder;
    std::vector<th::Tensor> weights;
};

}  // namespace torch_ext
