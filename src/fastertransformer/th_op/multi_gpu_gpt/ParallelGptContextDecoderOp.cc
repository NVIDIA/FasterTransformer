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

#include "src/fastertransformer/th_op/multi_gpu_gpt/ParallelGptContextDecoderOp.h"

namespace th = torch;
namespace ft = fastertransformer;

namespace torch_ext {

template<typename T>
FtGptContextDecoder<T>::FtGptContextDecoder(const size_t                  num_heads,
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
                                            const bool                    remove_padding):
    num_heads_(num_heads),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layers_(num_layers),
    gpt_variant_params_(gpt_variant_params),
    int8_mode_(int8_mode),
    weights_(weights),
    int8_weights_(int8_weights),
    int8_scales_(int8_scales),
    remove_padding_(remove_padding)
{
    ft::check_cuda_error(cublasLtCreate(&cublaslt_handle_));
    cublas_algo_map_      = new ft::cublasAlgoMap(GEMM_CONFIG);
    cublas_wrapper_mutex_ = new std::mutex();

    ft::ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size, pipeline_para_size);

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
FtGptContextDecoder<T>::~FtGptContextDecoder()
{
    ft::ftNcclParamDestroy(tensor_para_);
    ft::ftNcclParamDestroy(pipeline_para_);
    cublasLtDestroy(cublaslt_handle_);
    delete cublas_algo_map_;
    delete cublas_wrapper_mutex_;
}

template<typename T>
void FtGptContextDecoder<T>::forward(th::Tensor&              decoder_output,
                                     th::Tensor&              key_cache,
                                     th::Tensor&              value_cache,
                                     th::Tensor&              last_token_hidden_states,
                                     th::Tensor&              input_embeds,
                                     th::Tensor&              attention_mask,
                                     th::Tensor&              input_lengths,
                                     th::optional<th::Tensor> compact_idx,
                                     th::optional<th::Tensor> batch_to_compact_idx,
                                     th::optional<th::Tensor> linear_bias_slopes)
{
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

    ft::AttentionType attention_type = ft::getAttentionType<T>(size_per_head_,
                                                               ft::getSMVersion(),
                                                               remove_padding_,  // remove_padding
                                                               0,                // gpt supports any-seq-length fmha
                                                               true,             // is_fuse
                                                               false,            // with_relative_position_bias
                                                               true);            // causal_mask

    ft::ParallelGptContextDecoder<T> gpt_context_decoder(0,
                                                         0,
                                                         num_heads_,
                                                         size_per_head_,
                                                         inter_size_,
                                                         num_layers_,
                                                         0,   // expert_num
                                                         0,   // moe_k
                                                         {},  // moe_layer_index
                                                         gpt_variant_params_.layernorm_eps,
                                                         gpt_variant_params_,
                                                         tensor_para_,
                                                         pipeline_para_,
                                                         stream,
                                                         &cublas_wrapper,
                                                         &allocator,
                                                         false,
                                                         true,
                                                         attention_type,
                                                         false,
                                                         int8_mode_,
                                                         nullptr,
                                                         0);

    ft::TensorMap input_tensors({{"decoder_input", convert_tensor<T>(input_embeds)},
                                 {"attention_mask", convert_tensor<T>(attention_mask)},
                                 {"input_lengths", convert_tensor<int>(input_lengths)}});

    if (compact_idx.has_value() || batch_to_compact_idx.has_value()) {
        FT_CHECK_WITH_INFO(
            compact_idx.has_value() && batch_to_compact_idx.has_value(),
            "Please provide both compact_idx and batch_to_compact_idx to enable shared context feature.");
        input_tensors.insert("compact_idx", convert_tensor<int>(compact_idx.value()));
        input_tensors.insert("batch_to_compact_idx", convert_tensor<int>(batch_to_compact_idx.value()));
    }
    if (linear_bias_slopes.has_value()) {
        input_tensors.insert("linear_bias_slopes", convert_tensor<T>(linear_bias_slopes.value()));
    }

    ft::TensorMap output_tensors({{"decoder_output", convert_tensor<T>(decoder_output)},
                                  {"key_cache", convert_tensor<T>(key_cache)},
                                  {"value_cache", convert_tensor<T>(value_cache)},
                                  {"last_token_hidden_units", convert_tensor<T>(last_token_hidden_states)}});

    gpt_context_decoder.forward(&output_tensors, &input_tensors, &gpt_layer_weights_);
}

ParallelGptContextDecoderOp::ParallelGptContextDecoderOp(const int64_t                 num_heads,
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
                                                         const bool                    remove_padding):
    num_heads_(num_heads),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layers_(num_layers),
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    scalar_type_(weights[0].scalar_type())
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
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
    gpt_context_decoder_ = new FtGptContextDecoder<T_>(static_cast<size_t>(num_heads),                                 \
                                                       static_cast<size_t>(size_per_head),                             \
                                                       static_cast<size_t>(inter_size),                                \
                                                       static_cast<size_t>(num_layers),                                \
                                                       gpt_variant_params,                                             \
                                                       tensor_para_size,                                               \
                                                       pipeline_para_size,                                             \
                                                       int8_mode,                                                      \
                                                       weights,                                                        \
                                                       int8_weights,                                                   \
                                                       scale,                                                          \
                                                       remove_padding);                                                \
    chunk_size_          = 16 / sizeof(T_)

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

ParallelGptContextDecoderOp::~ParallelGptContextDecoderOp()
{
    delete gpt_context_decoder_;
}

std::vector<th::Tensor> ParallelGptContextDecoderOp::forward(th::Tensor               input_embeds,
                                                             th::Tensor               attention_mask,
                                                             th::Tensor               input_lengths,
                                                             th::optional<int64_t>    memory_length_opt,
                                                             th::optional<th::Tensor> compact_idx_opt,
                                                             th::optional<th::Tensor> batch_to_compact_idx_opt,
                                                             th::optional<th::Tensor> linear_bias_slopes_opt)
{
    // Input Arguments:
    //     input_embeds: [batch_size * beam_width, max_input_length, hidden_units], T
    //     attention_mask: [batch_size * beam_width, 1, max_input_length, max_input_length], T
    //     input_lengths: [batch_size * beam_width], int
    //     memory_length_opt: scalar, optional
    //     compact_idx_opt: [compact_batchxbeam], int, optional
    //     batch_to_compact_idx_opt: [batch_size * beam_width], int, optional
    //     linear_bias_slopes_opt: [num_heads], optional
    // Output Arguments:
    //     decoder_output: [batch_size * beam_width, max_input_length, hidden_units]
    //     key_cache: [num_layers, batch_size * beam_width, local_num_heads, size_per_head / x, memory_length, x]
    //         x = 16 / sizeof(T), memory_length = max_input_length or max_input_length + gen_length
    //     value_cache: [num_layers, batch_size * beam_width, local_num_heads, memory_length, hidden_units]
    //         memory_length = max_input_length or max_input_length + gen_length
    //     last_token_hidden_states: [batch_size * beam_width, hidden_units]

    CHECK_INPUT(input_embeds, scalar_type_);
    FT_CHECK_WITH_INFO(
        input_embeds.dim() == 3,
        ft::fmtstr("input_embeds is of shape (batch_size * beam_width, max_input_length, hidden_size), "
                   "but got dim=%d shape=%s",
                   (int)input_embeds.dim(),
                   ft::vec2str(convert_shape(input_embeds)).c_str())
            .c_str());
    CHECK_INPUT(attention_mask, scalar_type_);
    CHECK_INPUT(input_lengths, torch::kInt32);
    if (compact_idx_opt.has_value()) {
        CHECK_INPUT(compact_idx_opt.value(), torch::kInt32);
    }
    if (batch_to_compact_idx_opt.has_value()) {
        CHECK_INPUT(batch_to_compact_idx_opt.value(), torch::kInt32);
    }

    int batch_size       = input_embeds.size(0);
    int max_input_length = input_embeds.size(1);
    int hidden_units     = input_embeds.size(2);

    th::Tensor decoder_output = torch::empty_like(input_embeds);
    th::Tensor last_token_hidden_states =
        torch::empty({(int64_t)batch_size, (int64_t)hidden_units},
                     torch::dtype(scalar_type_).device(torch::kCUDA).requires_grad(false));

    int mem_length = memory_length_opt.has_value() ? memory_length_opt.value() : max_input_length;

    th::Tensor key_cache = torch::zeros({static_cast<long int>(num_layers_ / pipeline_para_size_),
                                         static_cast<long int>(batch_size),
                                         static_cast<long int>(num_heads_ / tensor_para_size_),
                                         static_cast<long int>(size_per_head_ / chunk_size_),
                                         static_cast<long int>(mem_length),
                                         static_cast<long int>(chunk_size_)},
                                        torch::dtype(scalar_type_).device(torch::kCUDA).requires_grad(false));

    th::Tensor value_cache = torch::zeros({static_cast<long int>(num_layers_ / pipeline_para_size_),
                                           static_cast<long int>(batch_size),
                                           static_cast<long int>(num_heads_ / tensor_para_size_),
                                           static_cast<long int>(mem_length),
                                           static_cast<long int>(size_per_head_)},
                                          torch::dtype(scalar_type_).device(torch::kCUDA).requires_grad(false));

    gpt_context_decoder_->forward(decoder_output,
                                  key_cache,
                                  value_cache,
                                  last_token_hidden_states,
                                  input_embeds,
                                  attention_mask,
                                  input_lengths,
                                  compact_idx_opt,
                                  batch_to_compact_idx_opt,
                                  linear_bias_slopes_opt);

    return std::vector<th::Tensor>{decoder_output, key_cache, value_cache, last_token_hidden_states};
}

}  // namespace torch_ext

static auto fasterTransformerGptContextDecoderTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::ParallelGptContextDecoderOp>("FasterTransformerParallelGptContextDecoderOp")
#else
    torch::jit::class_<torch_ext::ParallelGptContextDecoderOp>("FasterTransformer", "ParallelGptContextDecoderOp")
#endif
        .def(torch::jit::init<int64_t,                  // num_heads
                              int64_t,                  // size_per_head
                              int64_t,                  // inter_size
                              int64_t,                  // num_layers
                              int64_t,                  // tensor_para_size
                              int64_t,                  // pipeline_para_size
                              double,                   // layernorm_eps
                              std::string,              // layernorm_type
                              std::string,              // activation_type
                              bool,                     // has_adapter
                              int64_t,                  // adapter_inter_size
                              int64_t,                  // int8_mode
                              std::vector<th::Tensor>,  // weights
                              std::vector<th::Tensor>,  // int8_weights
                              std::vector<th::Tensor>,  // scale
                              bool>())                  // remove_padding
        .def("forward", &torch_ext::ParallelGptContextDecoderOp::forward);
