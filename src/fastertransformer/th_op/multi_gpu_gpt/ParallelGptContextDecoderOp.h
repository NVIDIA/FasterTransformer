/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptContextDecoder.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

class IFtGptContextDecoder {
public:
    virtual ~IFtGptContextDecoder() {}
    virtual void forward(th::Tensor&              decoder_output,
                         th::Tensor&              key_cache,
                         th::Tensor&              value_cache,
                         th::Tensor&              last_token_hidden_states,
                         th::Tensor&              input_embeds,
                         th::Tensor&              attention_mask,
                         th::Tensor&              input_lengths,
                         th::optional<th::Tensor> compact_idx,
                         th::optional<th::Tensor> batch_to_compact_idx,
                         th::optional<th::Tensor> linear_bias_slopes) = 0;
};

template<typename T>
class FtGptContextDecoder: public IFtGptContextDecoder {
public:
    FtGptContextDecoder(const size_t                  num_heads,
                        const size_t                  size_per_head,
                        const size_t                  inter_size,
                        const size_t                  num_layers,
                        const ft::gptVariantParams    gpt_variant_params,
                        const int                     tensor_para_size,
                        const int                     pipeline_para_size,
                        const int                     int8_mode,
                        const std::vector<th::Tensor> weights,
                        const std::vector<th::Tensor> int8_weights,
                        const std::vector<th::Tensor> int8_scales,
                        const bool                    remove_padding);

    ~FtGptContextDecoder() override;

    void forward(th::Tensor&              decoder_output,
                 th::Tensor&              key_cache,
                 th::Tensor&              value_cache,
                 th::Tensor&              last_token_hidden_states,
                 th::Tensor&              input_embeds,
                 th::Tensor&              attention_mask,
                 th::Tensor&              input_lengths,
                 th::optional<th::Tensor> compact_idx,
                 th::optional<th::Tensor> batch_to_compact_idx,
                 th::optional<th::Tensor> linear_bias_slopes) override;

private:
    const size_t num_heads_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t num_layers_;

    const ft::gptVariantParams gpt_variant_params_;

    const int int8_mode_;

    std::vector<th::Tensor> weights_;
    std::vector<th::Tensor> int8_weights_;
    std::vector<th::Tensor> int8_scales_;

    const bool remove_padding_;

    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;

    cublasLtHandle_t      cublaslt_handle_;
    std::mutex*           cublas_wrapper_mutex_;
    ft::cublasAlgoMap*    cublas_algo_map_;
    struct cudaDeviceProp prop_;

    std::vector<ft::ParallelGptDecoderLayerWeight<T>*> gpt_layer_weights_;
};

class ParallelGptContextDecoderOp: public th::jit::CustomClassHolder {
public:
    ParallelGptContextDecoderOp(const int64_t                 num_heads,
                                const int64_t                 size_per_head,
                                const int64_t                 inter_size,
                                const int64_t                 num_layers,
                                const int64_t                 tensor_para_size,
                                const int64_t                 pipeline_para_size,
                                const double                  layernorm_eps,
                                const std::string             layernorm_type,
                                const std::string             activation_type,
                                const bool                    has_adapters,
                                const int64_t                 adapter_inter_size,
                                const int64_t                 int8_mode,
                                const std::vector<th::Tensor> weights,
                                const std::vector<th::Tensor> int8_weights,
                                const std::vector<th::Tensor> scale,
                                const bool                    remove_padding);

    ~ParallelGptContextDecoderOp();

    std::vector<th::Tensor> forward(th::Tensor               input_embeds,
                                    th::Tensor               attention_mask,
                                    th::Tensor               input_lengths,
                                    th::optional<int64_t>    memory_length_opt,
                                    th::optional<th::Tensor> compact_idx_opt,
                                    th::optional<th::Tensor> batch_to_compact_idx_opt,
                                    th::optional<th::Tensor> linear_bias_slopes_opt);

private:
    size_t num_heads_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layers_;
    size_t tensor_para_size_;
    size_t pipeline_para_size_;

    at::ScalarType          scalar_type_;
    IFtGptContextDecoder*   gpt_context_decoder_;
    std::vector<th::Tensor> weights;

    // The chunk size (16 / sizeof(T)) for key cache in fmha.
    size_t chunk_size_;
};

}  // namespace torch_ext
