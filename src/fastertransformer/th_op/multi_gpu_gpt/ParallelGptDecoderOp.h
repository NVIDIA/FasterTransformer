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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoder.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

class IFtGptDecoder {
public:
    virtual ~IFtGptDecoder() {}

    virtual void forward(const int64_t            max_input_length,
                         const int64_t            step,
                         const int64_t            ite,
                         th::Tensor&              input_embeds,
                         th::Tensor&              input_lengths,
                         th::Tensor&              finished,
                         th::Tensor&              total_padding_tokens,
                         th::Tensor&              masked_tokens,
                         th::Tensor&              decoder_output,
                         th::Tensor&              key_cache,
                         th::Tensor&              value_cache,
                         th::optional<th::Tensor> cache_indirection,
                         th::optional<th::Tensor> linear_bias_slopes) = 0;
};

template<typename T>
class FtGptDecoder: public IFtGptDecoder {

public:
    FtGptDecoder(const size_t                  num_heads,
                 const size_t                  size_per_head,
                 const size_t                  inter_size,
                 const size_t                  num_layers,
                 const ft::gptVariantParams    gpt_variant_params,
                 const int                     tensor_para_size,
                 const int                     pipeline_para_size,
                 const int                     int8_mode,
                 const std::vector<th::Tensor> weights,
                 const std::vector<th::Tensor> int8_weights,
                 const std::vector<th::Tensor> scale);

    ~FtGptDecoder() override;

    void forward(const int64_t            max_input_length,
                 const int64_t            step,
                 const int64_t            ite,
                 th::Tensor&              input_embeds,
                 th::Tensor&              input_lengths,
                 th::Tensor&              finished,
                 th::Tensor&              total_padding_tokens,
                 th::Tensor&              masked_tokens,
                 th::Tensor&              decoder_output,
                 th::Tensor&              key_cache,
                 th::Tensor&              value_cache,
                 th::optional<th::Tensor> cache_indirection,
                 th::optional<th::Tensor> linear_bias_slopes) override;

private:
    const size_t num_heads_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t num_layers_;

    const ft::gptVariantParams gpt_variant_params_;

    const int int8_mode_;

    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;

    std::vector<th::Tensor> weights_;
    std::vector<th::Tensor> int8_weights_;
    std::vector<th::Tensor> int8_scales_;

    cublasLtHandle_t      cublaslt_handle_;
    std::mutex*           cublas_wrapper_mutex_;
    ft::cublasAlgoMap*    cublas_algo_map_;
    struct cudaDeviceProp prop_;

    std::vector<ft::ParallelGptDecoderLayerWeight<T>*> gpt_layer_weights_;
};

class ParallelGptDecoderOp: public th::jit::CustomClassHolder {
public:
    ParallelGptDecoderOp(const int64_t                 num_heads,
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
                         const std::vector<th::Tensor> int8_scales);

    ~ParallelGptDecoderOp();

    std::vector<th::Tensor> forward(const int64_t            max_input_length,
                                    const int64_t            step,
                                    const int64_t            ite,
                                    th::Tensor               input_embeds,
                                    th::Tensor               input_lengths,
                                    th::Tensor               finished,
                                    th::Tensor               total_padding_tokens,
                                    th::Tensor               masked_tokens,
                                    th::Tensor               key_cache,
                                    th::Tensor               value_cache,
                                    th::optional<th::Tensor> cache_indirection_opt,
                                    th::optional<th::Tensor> linear_bias_slopes_opt);

private:
    at::ScalarType          scalar_type_;
    IFtGptDecoder*          gpt_decoder_;
    std::vector<th::Tensor> weights_;
};

}  // namespace torch_ext
