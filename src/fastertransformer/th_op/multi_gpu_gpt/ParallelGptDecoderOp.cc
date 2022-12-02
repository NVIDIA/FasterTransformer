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

#include "src/fastertransformer/th_op/multi_gpu_gpt/ParallelGptDecoderOp.h"

namespace th = torch;
namespace ft = fastertransformer;

namespace torch_ext {

template<typename T>
FtGptDecoder<T>::FtGptDecoder(const size_t                  num_heads,
                              const size_t                  size_per_head,
                              const size_t                  inter_size,
                              const size_t                  num_layers,
                              const ft::gptVariantParams    gpt_variant_params,
                              const int                     tensor_para_size,
                              const int                     pipeline_para_size,
                              const int                     int8_mode,
                              const std::vector<th::Tensor> weights,
                              const std::vector<th::Tensor> int8_weights,
                              const std::vector<th::Tensor> int8_scales):
    num_heads_(num_heads),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layers_(num_layers),
    gpt_variant_params_(gpt_variant_params),
    int8_mode_(int8_mode),
    weights_(weights),
    int8_weights_(int8_weights),
    int8_scales_(int8_scales)
{
    ft::check_cuda_error(cublasLtCreate(&cublaslt_handle_));
    cublas_algo_map_      = new ft::cublasAlgoMap(GEMM_CONFIG);
    cublas_wrapper_mutex_ = new std::mutex();

    ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size, pipeline_para_size);

    size_t local_num_layers = (num_layers_ + pipeline_para_.world_size_ - 1) / pipeline_para_.world_size_;
    size_t first_layer      = local_num_layers * pipeline_para_.rank_;

    gpt_layer_weights_.reserve(num_layers_);

    for (size_t i = 0; i < num_layers_; i++) {
        gpt_layer_weights_.push_back(new ft::ParallelGptDecoderLayerWeight<T>(int8_mode_));

        if (i / local_num_layers != pipeline_para_.rank_) {
            // Layer i is not in the current pipeline parallel group.
            continue;
        }

        // clang-format off
        size_t local_layer_index = i - first_layer;
        gpt_layer_weights_[i]->pre_layernorm_weights.gamma                           = get_ptr<T>(weights_[local_layer_index + 0  * local_num_layers]);
        gpt_layer_weights_[i]->pre_layernorm_weights.beta                            = get_ptr<T>(weights_[local_layer_index + 1  * local_num_layers]);
        gpt_layer_weights_[i]->self_attention_weights.query_weight.kernel            = get_ptr<T>(weights_[local_layer_index + 2  * local_num_layers]);
        gpt_layer_weights_[i]->self_attention_weights.query_weight.bias              = get_ptr<T>(weights_[local_layer_index + 3  * local_num_layers]);
        gpt_layer_weights_[i]->self_attention_weights.attention_output_weight.kernel = get_ptr<T>(weights_[local_layer_index + 4  * local_num_layers]);
        gpt_layer_weights_[i]->self_attention_weights.attention_output_weight.bias   = get_ptr<T>(weights_[local_layer_index + 5  * local_num_layers]);
        gpt_layer_weights_[i]->self_attn_layernorm_weights.gamma                     = get_ptr<T>(weights_[local_layer_index + 6  * local_num_layers]);
        gpt_layer_weights_[i]->self_attn_layernorm_weights.beta                      = get_ptr<T>(weights_[local_layer_index + 7  * local_num_layers]);
        gpt_layer_weights_[i]->ffn_weights.intermediate_weight.kernel                = get_ptr<T>(weights_[local_layer_index + 8  * local_num_layers]);
        gpt_layer_weights_[i]->ffn_weights.intermediate_weight.bias                  = get_ptr<T>(weights_[local_layer_index + 9  * local_num_layers]);
        gpt_layer_weights_[i]->ffn_weights.output_weight.kernel                      = get_ptr<T>(weights_[local_layer_index + 10 * local_num_layers]);
        gpt_layer_weights_[i]->ffn_weights.output_weight.bias                        = get_ptr<T>(weights_[local_layer_index + 11 * local_num_layers]);

        if (int8_mode_ != 0) {
            gpt_layer_weights_[i]->self_attention_weights.query_weight.int8_kernel            = get_ptr<int8_t>(int8_weights_[local_layer_index + 0 * local_num_layers]);
            gpt_layer_weights_[i]->self_attention_weights.attention_output_weight.int8_kernel = get_ptr<int8_t>(int8_weights_[local_layer_index + 1 * local_num_layers]);
            gpt_layer_weights_[i]->ffn_weights.intermediate_weight.int8_kernel                = get_ptr<int8_t>(int8_weights_[local_layer_index + 2 * local_num_layers]);
            gpt_layer_weights_[i]->ffn_weights.output_weight.int8_kernel                      = get_ptr<int8_t>(int8_weights_[local_layer_index + 3 * local_num_layers]);

            if (int8_mode_ == 1) {
                gpt_layer_weights_[i]->self_attention_weights.query_weight.weight_only_quant_scale            = get_ptr<T>(int8_scales_[local_layer_index + 0 * local_num_layers]);
                gpt_layer_weights_[i]->self_attention_weights.attention_output_weight.weight_only_quant_scale = get_ptr<T>(int8_scales_[local_layer_index + 1 * local_num_layers]);
                gpt_layer_weights_[i]->ffn_weights.intermediate_weight.weight_only_quant_scale                = get_ptr<T>(int8_scales_[local_layer_index + 2 * local_num_layers]);
                gpt_layer_weights_[i]->ffn_weights.output_weight.weight_only_quant_scale                      = get_ptr<T>(int8_scales_[local_layer_index + 3 * local_num_layers]);
            }
            else {
                gpt_layer_weights_[i]->self_attention_weights.query_weight.scale            = get_ptr<float>(int8_scales_[local_layer_index + 0 * local_num_layers]);
                gpt_layer_weights_[i]->self_attention_weights.attention_output_weight.scale = get_ptr<float>(int8_scales_[local_layer_index + 1 * local_num_layers]);
                gpt_layer_weights_[i]->ffn_weights.intermediate_weight.scale                = get_ptr<float>(int8_scales_[local_layer_index + 2 * local_num_layers]);
                gpt_layer_weights_[i]->ffn_weights.output_weight.scale                      = get_ptr<float>(int8_scales_[local_layer_index + 3 * local_num_layers]);
            }
        }

        if (gpt_variant_params_.has_adapters) {
            gpt_layer_weights_[i]->after_attention_adapter_weights.intermediate_weight.kernel = get_ptr<T>(weights_[local_layer_index + 12 * local_num_layers]);
            gpt_layer_weights_[i]->after_attention_adapter_weights.intermediate_weight.bias   = get_ptr<T>(weights_[local_layer_index + 13 * local_num_layers]);
            gpt_layer_weights_[i]->after_attention_adapter_weights.output_weight.kernel       = get_ptr<T>(weights_[local_layer_index + 14 * local_num_layers]);
            gpt_layer_weights_[i]->after_attention_adapter_weights.output_weight.bias         = get_ptr<T>(weights_[local_layer_index + 15 * local_num_layers]);
            gpt_layer_weights_[i]->after_ffn_adapter_weights.intermediate_weight.kernel       = get_ptr<T>(weights_[local_layer_index + 16 * local_num_layers]);
            gpt_layer_weights_[i]->after_ffn_adapter_weights.intermediate_weight.bias         = get_ptr<T>(weights_[local_layer_index + 17 * local_num_layers]);
            gpt_layer_weights_[i]->after_ffn_adapter_weights.output_weight.kernel             = get_ptr<T>(weights_[local_layer_index + 18 * local_num_layers]);
            gpt_layer_weights_[i]->after_ffn_adapter_weights.output_weight.bias               = get_ptr<T>(weights_[local_layer_index + 19 * local_num_layers]);

            if (int8_mode_ != 0) {
                gpt_layer_weights_[i]->after_attention_adapter_weights.intermediate_weight.int8_kernel = get_ptr<int8_t>(int8_weights_[local_layer_index + 4 * local_num_layers]);
                gpt_layer_weights_[i]->after_attention_adapter_weights.output_weight.int8_kernel       = get_ptr<int8_t>(int8_weights_[local_layer_index + 5 * local_num_layers]);
                gpt_layer_weights_[i]->after_ffn_adapter_weights.intermediate_weight.int8_kernel       = get_ptr<int8_t>(int8_weights_[local_layer_index + 6 * local_num_layers]);
                gpt_layer_weights_[i]->after_ffn_adapter_weights.output_weight.int8_kernel             = get_ptr<int8_t>(int8_weights_[local_layer_index + 7 * local_num_layers]);

                if (int8_mode_ == 1) {
                    gpt_layer_weights_[i]->after_attention_adapter_weights.intermediate_weight.weight_only_quant_scale = get_ptr<T>(int8_scales_[local_layer_index + 4 * local_num_layers]);
                    gpt_layer_weights_[i]->after_attention_adapter_weights.output_weight.weight_only_quant_scale       = get_ptr<T>(int8_scales_[local_layer_index + 5 * local_num_layers]);
                    gpt_layer_weights_[i]->after_ffn_adapter_weights.intermediate_weight.weight_only_quant_scale       = get_ptr<T>(int8_scales_[local_layer_index + 6 * local_num_layers]);
                    gpt_layer_weights_[i]->after_ffn_adapter_weights.output_weight.weight_only_quant_scale             = get_ptr<T>(int8_scales_[local_layer_index + 7 * local_num_layers]);
                }
                else {
                    gpt_layer_weights_[i]->after_attention_adapter_weights.intermediate_weight.scale = get_ptr<float>(int8_scales_[local_layer_index + 4 * local_num_layers]);
                    gpt_layer_weights_[i]->after_attention_adapter_weights.output_weight.scale       = get_ptr<float>(int8_scales_[local_layer_index + 5 * local_num_layers]);
                    gpt_layer_weights_[i]->after_ffn_adapter_weights.intermediate_weight.scale       = get_ptr<float>(int8_scales_[local_layer_index + 6 * local_num_layers]);
                    gpt_layer_weights_[i]->after_ffn_adapter_weights.output_weight.scale             = get_ptr<float>(int8_scales_[local_layer_index + 7 * local_num_layers]);
                }
            }
        }
        // clang-format on
    }
}

template<typename T>
FtGptDecoder<T>::~FtGptDecoder()
{
    ft::ftNcclParamDestroy(tensor_para_);
    ft::ftNcclParamDestroy(pipeline_para_);
    cublasLtDestroy(cublaslt_handle_);
    delete cublas_algo_map_;
    delete cublas_wrapper_mutex_;
}

template<typename T>
void FtGptDecoder<T>::forward(const int64_t            max_input_length,
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
                              th::optional<th::Tensor> cache_indirection_opt,
                              th::optional<th::Tensor> linear_bias_slopes_opt)
{
    // Input Arguments:
    //     input_embeds: [local_batch_size * beam_width, hidden_units], T
    //     input_lengths: [local_batch_size * beam_width], int
    //     finished: [local_batch_size * beam_width], bool
    //     total_padding_tokens: [local_batch_size * beam_width], int, optional
    //     masked_tokens [local_batch_size * beam_width, memory_length]
    //     decoder_output: [local_batch_size * beam_width, max_input_length, hidden_units]
    //     key_cache: [num_layers, batch_size * beam_width, local_num_heads, size_per_head / x, memory_length, x]
    //         x = 16 / sizeof(T)
    //     value_cache: [num_layers, batch_size * beam_width, local_num_heads, memory_length, hidden_units]
    //     cache_indirection [local_batch_size, beam_width, memory_length], int, optional.
    //     linear_bias_slopes_opt: [num_heads], optional

    auto stream        = at::cuda::getCurrentCUDAStream().stream();
    auto cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);

    ft::Allocator<ft::AllocatorType::TH> allocator;
    allocator.setStream(stream);

    ft::cublasMMWrapper cublas_wrapper(
        cublas_handle, cublaslt_handle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, &allocator);

    if (std::is_same<T, half>::value) {
        cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper.setBF16GemmConfig();
    }
#endif
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    const int  _max_input_length = static_cast<int>(max_input_length);
    const int  _step             = static_cast<int>(step);
    const uint _ite              = static_cast<uint>(ite);

    ft::ParallelGptDecoder<T> gpt_decoder(0,
                                          num_heads_,
                                          size_per_head_,
                                          inter_size_,
                                          num_layers_,
                                          gpt_variant_params_.layernorm_eps,
                                          gpt_variant_params_,
                                          tensor_para_,
                                          pipeline_para_,
                                          stream,
                                          &cublas_wrapper,
                                          &allocator,
                                          false,
                                          false,
                                          int8_mode_,
                                          nullptr,
                                          0);

    std::unordered_map<std::string, ft::Tensor> input_tensors{
        {"decoder_input", convert_tensor<T>(input_embeds)},
        {"finished", convert_tensor<bool>(finished)},
        {"input_lengths", convert_tensor<int>(input_lengths)},
        {"total_padding_tokens", convert_tensor<int>(total_padding_tokens)},
        {"max_input_length", ft::Tensor(ft::MEMORY_CPU, ft::TYPE_INT32, {1}, &_max_input_length)},
        {"step", ft::Tensor(ft::MEMORY_CPU, ft::TYPE_INT32, {1}, &_step)},
        {"ite", ft::Tensor(ft::MEMORY_CPU, ft::TYPE_INT32, {1}, &_ite)},
        {"masked_tokens", convert_tensor<bool>(masked_tokens)}};
    if (cache_indirection_opt.has_value()) {
        ft::FT_CHECK_WITH_INFO(
            cache_indirection_opt.value().dim() == 3,
            ft::fmtstr("cache_indirection assumes to be of shape (batch_size, beam_width, memory_length), "
                       "but got %s",
                       ft::vec2str(convert_shape(cache_indirection_opt.value())).c_str()));
        input_tensors.insert({"cache_indirection", convert_tensor<int>(cache_indirection_opt.value())});
    }
    if (linear_bias_slopes_opt.has_value()) {
        input_tensors.insert({"linear_bias_slopes", convert_tensor<T>(linear_bias_slopes_opt.value())});
    }

    std::unordered_map<std::string, ft::Tensor> output_tensors{{"decoder_output", convert_tensor<T>(decoder_output)},
                                                               {"key_cache", convert_tensor<T>(key_cache)},
                                                               {"value_cache", convert_tensor<T>(value_cache)}};

    gpt_decoder.forward(&output_tensors, &input_tensors, &gpt_layer_weights_);
}

ParallelGptDecoderOp::ParallelGptDecoderOp(const int64_t                 num_heads,
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
                                           const std::vector<th::Tensor> scale):
    scalar_type_(weights[0].scalar_type())
{
    for (auto weight : weights) {
        CHECK_INPUT(weight, scalar_type_);
    }

    if (int8_mode == 1) {
        TORCH_CHECK(scalar_type_ != torch::kFloat32, "Int8 weight only quant does not work for FP32.");
        for (auto int8_weight : int8_weights) {
            CHECK_INPUT(int8_weight, torch::kInt8);
        }

        for (auto weight_scale : scale) {
            CHECK_INPUT(weight_scale, scalar_type_);
        }
    }

    ft::gptVariantParams gpt_variant_params;
    gpt_variant_params.layernorm_eps      = static_cast<float>(layernorm_eps);
    gpt_variant_params.layernorm_type     = ft::getLayerNormType(layernorm_type);
    gpt_variant_params.activation_type    = ft::getActivationType(activation_type);
    gpt_variant_params.has_adapters       = has_adapters;
    gpt_variant_params.adapter_inter_size = static_cast<size_t>(adapter_inter_size);

#define CREATE_INSTANCE(T_)                                                                                            \
    gpt_decoder_ = new FtGptDecoder<T_>(static_cast<size_t>(num_heads),                                                \
                                        static_cast<size_t>(size_per_head),                                            \
                                        static_cast<size_t>(inter_size),                                               \
                                        static_cast<size_t>(num_layers),                                               \
                                        gpt_variant_params,                                                            \
                                        tensor_para_size,                                                              \
                                        pipeline_para_size,                                                            \
                                        int8_mode,                                                                     \
                                        weights,                                                                       \
                                        int8_weights,                                                                  \
                                        scale)

    switch (scalar_type_) {
        case at::ScalarType::Float:
            CREATE_INSTANCE(float);
            break;
        case at::ScalarType::Half:
            CREATE_INSTANCE(half);
            break;
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16:
            CREATE_INSTANCE(__nv_bfloat16);
            break;
#endif
        default:
            throw std::runtime_error("Wrong tensor type.");
    }
#undef CREATE_INSTANCE
}

ParallelGptDecoderOp::~ParallelGptDecoderOp()
{
    delete gpt_decoder_;
}

std::vector<th::Tensor> ParallelGptDecoderOp::forward(const int64_t            max_input_length,
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
                                                      th::optional<th::Tensor> linear_bias_slopes_opt)
{
    CHECK_INPUT(input_embeds, scalar_type_);
    CHECK_INPUT(finished, torch::kBool);
    CHECK_INPUT(input_lengths, torch::kInt32);
    CHECK_INPUT(total_padding_tokens, torch::kInt32);
    CHECK_INPUT(masked_tokens, torch::kBool);
    CHECK_INPUT(key_cache, scalar_type_);
    CHECK_INPUT(value_cache, scalar_type_);
    if (cache_indirection_opt.has_value()) {
        CHECK_INPUT(cache_indirection_opt.value(), torch::kInt32);
    }
    if (linear_bias_slopes_opt.has_value()) {
        CHECK_INPUT(linear_bias_slopes_opt.value(), scalar_type_);
    }

    th::Tensor decoder_output = torch::empty_like(input_embeds);

    gpt_decoder_->forward(max_input_length,
                          step,
                          ite,
                          input_embeds,
                          input_lengths,
                          finished,
                          total_padding_tokens,
                          masked_tokens,
                          decoder_output,
                          key_cache,
                          value_cache,
                          cache_indirection_opt,
                          linear_bias_slopes_opt);
    return std::vector<th::Tensor>{decoder_output};
}

}  // namespace torch_ext

static auto fasterTransformerGptDecoderTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::ParallelGptDecoderOp>("FasterTransformerParallelGptDecoderOp")
#else
    torch::jit::class_<torch_ext::ParallelGptDecoderOp>("FasterTransformer", "ParallelGptDecoderOp")
#endif
        .def(torch::jit::init<int64_t,                     // num_heads
                              int64_t,                     // size_per_head
                              int64_t,                     // inter_size
                              int64_t,                     // num_layers
                              int64_t,                     // tensor_para_size
                              int64_t,                     // pipeline_para_size
                              double,                      // layernorm_eps
                              std::string,                 // layernorm_type
                              std::string,                 // activation_type
                              bool,                        // has_adapters
                              int64_t,                     // adapter_inter_size
                              int64_t,                     // int8_mode
                              std::vector<th::Tensor>,     // weights
                              std::vector<th::Tensor>,     // int8_weights
                              std::vector<th::Tensor>>())  // scale
        .def("forward", &torch_ext::ParallelGptDecoderOp::forward);
