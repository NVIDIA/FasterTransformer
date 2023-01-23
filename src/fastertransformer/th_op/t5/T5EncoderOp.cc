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

#include "src/fastertransformer/th_op/t5/T5EncoderOp.h"

namespace torch_ext {

template<typename T>
FTT5Encoder<T>::FTT5Encoder(int64_t                        head_num,
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
                            const std::vector<th::Tensor>& w):
    _head_num(head_num),
    _head_size(head_size),
    _inter_size(inter_size),
    _d_model(d_model),
    _layer_num(layer_num),
    _num_bucket(num_bucket),
    _expert_num(expert_num),
    _max_distance(max_distance),
    _sparse(sparse),
    _q_scaling(q_scaling),
    _moe_k(moe_k),
    _t5_with_bias(t5_with_bias),
    _position_embedding_type(position_embedding_type),
    _activation_type(activation_type),
    _adapter_inter_size{adapter_inter_size},
    _adapter_layer_norm_type{adapter_layer_norm_type},
    _moe_layer_index(moe_layer_index),
    _weights(w)
{
    bool use_gated_activation = isGatedActivation(_activation_type);
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    ft::ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size, pipeline_para_size);

#ifndef SPARSITY_ENABLED
    if (sparse) {
        std::cout << "[WARNING] Sparsity support is not enabled. Will use dense GEMM instead.\n" << std::flush;
    }
#endif
    int hidden_dim = _head_num * _head_size;
    ft::check_cuda_error(cublasLtCreate(&_cublasltHandle));
    sm_ = ft::getSMVersion();
#ifdef SPARSITY_ENABLED
    if (sparse) {
        CHECK_CUSPARSE(cusparseLtInit(&_cusparseLtHandle));
    }
#endif
    std::string sp_config_fname = sparse ? "spgemm_config.in" : "";
    cublas_algo_map_            = new ft::cublasAlgoMap("gemm_config.in", sp_config_fname);
    cublas_wrapper_mutex_       = new std::mutex();

    t5_encoder_weights.resizeLayer(_layer_num);
    t5_encoder_weights.setT5StructureDiff(t5_with_bias, use_gated_activation, position_embedding_type);
    int dense_weight_index     = 0;  // the inter and out kernel has the same index
    int moe_dense_weight_index = 0;  // the moe inter and out kernel has the same index

    for (int i = 0; i < _layer_num; i++) {
        int local_num_layer = (int)(ceil(_layer_num * 1.0f / pipeline_para_.world_size_));
        if (!(i < _layer_num && (i >= local_num_layer * pipeline_para_.rank_)
              && (i < local_num_layer * (pipeline_para_.rank_ + 1)))) {
            continue;
        }

        auto const  layer_index           = i - local_num_layer * pipeline_para_.rank_;
        auto* const encoder_layer_weights = t5_encoder_weights.t5_encoder_layer_weights[i];
        auto const  world_size            = tensor_para_.world_size_;

        encoder_layer_weights->attn_layernorm_weights_.gamma = get_ptr<T>(_weights[0]) + _d_model * layer_index;
        encoder_layer_weights->attention_weights_.query_weight.kernel =
            get_ptr<T>(_weights[1]) + _d_model * hidden_dim / world_size * layer_index;
        encoder_layer_weights->attention_weights_.key_weight.kernel =
            get_ptr<T>(_weights[2]) + _d_model * hidden_dim / world_size * layer_index;
        encoder_layer_weights->attention_weights_.value_weight.kernel =
            get_ptr<T>(_weights[3]) + _d_model * hidden_dim / world_size * layer_index;
        encoder_layer_weights->attention_weights_.attention_output_weight.kernel =
            get_ptr<T>(_weights[4]) + hidden_dim / world_size * _d_model * layer_index;
        encoder_layer_weights->ffn_layernorm_weights_.gamma = get_ptr<T>(_weights[5]) + _d_model * layer_index;
        encoder_layer_weights->ffn_weights_.intermediate_weight.kernel =
            get_ptr<T>(_weights[6]) + _d_model * _inter_size / world_size * dense_weight_index
            + _expert_num * _d_model * _inter_size / world_size * moe_dense_weight_index;
        if (use_gated_activation) {
            encoder_layer_weights->ffn_weights_.intermediate_weight2.kernel =
                get_ptr<T>(_weights[7]) + _d_model * _inter_size / world_size * layer_index;
        }
        encoder_layer_weights->ffn_weights_.output_weight.kernel =
            get_ptr<T>(_weights[8]) + _inter_size / world_size * _d_model * dense_weight_index
            + _expert_num * _d_model * _inter_size / world_size * moe_dense_weight_index;

        if (_t5_with_bias) {
            encoder_layer_weights->attn_layernorm_weights_.beta = get_ptr<T>(_weights[12]) + _d_model * layer_index;
            encoder_layer_weights->attention_weights_.query_weight.bias =
                get_ptr<T>(_weights[13]) + hidden_dim / world_size * layer_index;
            encoder_layer_weights->attention_weights_.key_weight.bias =
                get_ptr<T>(_weights[14]) + hidden_dim / world_size * layer_index;
            encoder_layer_weights->attention_weights_.value_weight.bias =
                get_ptr<T>(_weights[15]) + hidden_dim / world_size * layer_index;
            encoder_layer_weights->attention_weights_.attention_output_weight.bias =
                get_ptr<T>(_weights[16]) + _d_model * layer_index;
            encoder_layer_weights->ffn_layernorm_weights_.beta = get_ptr<T>(_weights[17]) + _d_model * layer_index;
            encoder_layer_weights->ffn_weights_.intermediate_weight.bias =
                get_ptr<T>(_weights[18]) + _inter_size / world_size * dense_weight_index
                + _expert_num * _inter_size / world_size * moe_dense_weight_index;

            if (use_gated_activation) {
                encoder_layer_weights->ffn_weights_.intermediate_weight2.bias =
                    get_ptr<T>(_weights[19]) + _inter_size / world_size * layer_index;
            }

            encoder_layer_weights->ffn_weights_.output_weight.bias = get_ptr<T>(_weights[20])
                                                                     + _d_model * dense_weight_index
                                                                     + _expert_num * _d_model * moe_dense_weight_index;
        }

        if (std::find(moe_layer_index.begin(), moe_layer_index.end(), i) == moe_layer_index.end()) {
            dense_weight_index += 1;
        }

        if (std::find(moe_layer_index.begin(), moe_layer_index.end(), i) != moe_layer_index.end()) {
            encoder_layer_weights->ffn_weights_.gating_weight.kernel =
                get_ptr<T>(_weights[22]) + _d_model * _expert_num * moe_dense_weight_index;
            moe_dense_weight_index += 1;
        }

        if (_adapter_inter_size > 0) {
            auto& adapter_weights = encoder_layer_weights->adapter_weights_;
            adapter_weights.setAdapterInterSize(_adapter_inter_size);
            auto& attn_adapter_weights = adapter_weights.after_attention_adapter_weights_;
            attn_adapter_weights.input_weight().kernel =
                get_ptr<T>(_weights[23]) + layer_index * d_model * _adapter_inter_size / world_size;
            attn_adapter_weights.output_weight().kernel =
                get_ptr<T>(_weights[24]) + layer_index * d_model * _adapter_inter_size / world_size;
            attn_adapter_weights.layer_norm_weight.gamma = get_ptr<T>(_weights[25]) + layer_index * d_model;
            attn_adapter_weights.layer_norm_weight.beta  = get_ptr<T>(_weights[26]) + layer_index * d_model;
            auto& ffn_adapter_weights                    = adapter_weights.after_ffn_adapter_weights_;
            ffn_adapter_weights.input_weight().kernel =
                get_ptr<T>(_weights[27]) + layer_index * d_model * _adapter_inter_size / world_size;
            ffn_adapter_weights.output_weight().kernel =
                get_ptr<T>(_weights[28]) + layer_index * d_model * _adapter_inter_size / world_size;
            ffn_adapter_weights.layer_norm_weight.gamma = get_ptr<T>(_weights[29]) + layer_index * d_model;
            ffn_adapter_weights.layer_norm_weight.beta  = get_ptr<T>(_weights[30]) + layer_index * d_model;
        }
    }
    t5_encoder_weights.post_transformer_layernorm_weights.gamma = get_ptr<T>(_weights[9]);
    t5_encoder_weights.absolute_or_relative_position_embedding  = get_ptr<T>(_weights[10]);
    t5_encoder_weights.embedding_table                          = get_ptr<T>(_weights[11]);
    if (_t5_with_bias) {
        t5_encoder_weights.post_transformer_layernorm_weights.beta = get_ptr<T>(_weights[21]);
    }

#ifdef SPARSITY_ENABLED
    if (sparse) {
        auto           stream        = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t _cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(_cublasHandle, stream);
        ft::cublasMMWrapper cublas_wrapper = ft::cublasMMWrapper(_cublasHandle,
                                                                 _cublasltHandle,
                                                                 _cusparseLtHandle,
                                                                 stream,
                                                                 cublas_algo_map_,
                                                                 cublas_wrapper_mutex_,
                                                                 nullptr);
        for (int i = 0; i < _layer_num; ++i) {
            t5_encoder_weights.t5_encoder_layer_weights[i]->compress_weights(cublas_wrapper, hidden_dim);
        }
    }
#endif
}

template<typename T>
void FTT5Encoder<T>::forward(size_t                   batch_size,
                             size_t                   seq_len,
                             th::optional<th::Tensor> input_ids,
                             th::Tensor&              sequence_lengths,
                             th::optional<th::Tensor> inputs_embeds,
                             th::Tensor&              output,
                             bool                     removing_padding)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    auto           stream        = at::cuda::getCurrentCUDAStream().stream();
    cublasHandle_t _cublasHandle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(_cublasHandle, stream);
    ft::Allocator<ft::AllocatorType::TH>* allocator = new ft::Allocator<ft::AllocatorType::TH>();
    ft::cublasMMWrapper*                  cublas_wrapper =
#ifdef SPARSITY_ENABLED
        new ft::cublasMMWrapper(_cublasHandle,
                                _cublasltHandle,
                                _cusparseLtHandle,
                                stream,
                                cublas_algo_map_,
                                cublas_wrapper_mutex_,
                                allocator);
#else
        new ft::cublasMMWrapper(
            _cublasHandle, _cublasltHandle, stream, cublas_algo_map_, cublas_wrapper_mutex_, allocator);
#endif

    if (std::is_same<T, half>::value) {
        cublas_wrapper->setFP16GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper->setBF16GemmConfig();
    }
#endif
    else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }

    ft::AttentionType attention_type = ft::getAttentionType<T>(_head_size, sm_, removing_padding, seq_len, false, true);

    ft::T5Encoder<T>* t5_encoder =
        new ft::T5Encoder<T>(batch_size,
                             seq_len,
                             _head_num,
                             _head_size,
                             _inter_size,
                             _d_model,
                             _layer_num,
                             _num_bucket,
                             _expert_num,
                             _max_distance,
                             _moe_k,
                             sm_,
                             _q_scaling,
                             _moe_layer_index,
                             stream,
                             cublas_wrapper,
                             allocator,
                             false,
                             attention_type,
                             _sparse,
                             _activation_type,
                             ft::LayerNormType::pre_layernorm,
                             tensor_para_,
                             pipeline_para_,
                             0,
                             ft::PromptLearningType::no_prompt,
                             nullptr,
                             0,
                             ft::LinearAdapterConfig{_adapter_inter_size, _adapter_layer_norm_type});

    ft::TensorMap input_tensors({{"sequence_length", convert_tensor<int>(sequence_lengths)}});

    if (inputs_embeds.has_value()) {
        if (std::is_same<T, float>::value) {
            TORCH_CHECK(inputs_embeds.value().dtype() == torch::kFloat32, "inputs_embeds dtype should be float32");
        }
        else if (std::is_same<T, half>::value) {
            TORCH_CHECK(inputs_embeds.value().dtype() == torch::kFloat16, "inputs_embeds dtype should be float16");
        }
        input_tensors.insert("inputs_embeds", convert_tensor<T>(inputs_embeds.value()));
    }
    else {
        // already check that input_ids and input_embeds cannot be empty at the same time
        input_tensors.insert("input_ids", convert_tensor<int>(input_ids.value()));
    }

    ft::TensorMap output_tensors({{"output_hidden_state", convert_tensor<T>(output)}});

    try {
        t5_encoder->forward(&output_tensors, &input_tensors, &t5_encoder_weights);
    }
    catch (std::runtime_error& error) {
        std::cout << error.what();
        exit(-1);
    }
    catch (...) {
        std::cout << "Runtime error";
        exit(-1);
    }
    delete t5_encoder;
    delete cublas_wrapper;
    delete allocator;
}

template class FTT5Encoder<float>;
template class FTT5Encoder<half>;
#ifdef ENABLE_BF16
template class FTT5Encoder<__nv_bfloat16>;
#endif

FasterTransformerT5Encoder::FasterTransformerT5Encoder(th::Tensor           attr_output_layernorm_gamma,
                                                       th::Tensor           q_kernel,
                                                       th::Tensor           k_kernel,
                                                       th::Tensor           v_kernel,
                                                       th::Tensor           attr_output_kernel,
                                                       th::Tensor           output_layernorm_gamma,
                                                       th::Tensor           inter_kernel,
                                                       th::Tensor           inter_kernel2,
                                                       th::Tensor           output_kernel,
                                                       th::Tensor           post_transformer_layernorm_gamma,
                                                       th::Tensor           absolute_or_relative_position_embedding,
                                                       th::Tensor           embedding_table,
                                                       th::Tensor           attr_output_layernorm_beta,
                                                       th::Tensor           q_bias,
                                                       th::Tensor           k_bias,
                                                       th::Tensor           v_bias,
                                                       th::Tensor           attr_output_bias,
                                                       th::Tensor           output_layernorm_beta,
                                                       th::Tensor           inter_bias,
                                                       th::Tensor           inter_bias2,
                                                       th::Tensor           output_bias,
                                                       th::Tensor           post_transformer_layernorm_beta,
                                                       th::Tensor           moe_gate,
                                                       th::Tensor           after_attn_adapter_weight_in,
                                                       th::Tensor           after_attn_adapter_weight_out,
                                                       th::Tensor           after_attn_adapter_layernorm_gamma,
                                                       th::Tensor           after_attn_adapter_layernorm_beta,
                                                       th::Tensor           after_ffn_adapter_weight_in,
                                                       th::Tensor           after_ffn_adapter_weight_out,
                                                       th::Tensor           after_ffn_adapter_layernorm_gamma,
                                                       th::Tensor           after_ffn_adapter_layernorm_beta,
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
                                                       std::string          adapter_norm_position):
    d_model_(d_model),
    _st(q_kernel.scalar_type()),
    _remove_padding(remove_padding),
    weights{attr_output_layernorm_gamma,
            q_kernel,
            k_kernel,
            v_kernel,
            attr_output_kernel,
            output_layernorm_gamma,
            inter_kernel,
            inter_kernel2,
            output_kernel,
            post_transformer_layernorm_gamma,
            absolute_or_relative_position_embedding,
            embedding_table,
            attr_output_layernorm_beta,
            q_bias,
            k_bias,
            v_bias,
            attr_output_bias,
            output_layernorm_beta,
            inter_bias,
            inter_bias2,
            output_bias,
            post_transformer_layernorm_beta,
            moe_gate,
            after_attn_adapter_weight_in,
            after_attn_adapter_weight_out,
            after_attn_adapter_layernorm_gamma,
            after_attn_adapter_layernorm_beta,
            after_ffn_adapter_weight_in,
            after_ffn_adapter_weight_out,
            after_ffn_adapter_layernorm_gamma,
            after_ffn_adapter_layernorm_beta}
{
    CHECK_INPUT(q_kernel, _st);                                 // d_model, hidden_dim
    CHECK_INPUT(k_kernel, _st);                                 // d_model, hidden_dim
    CHECK_INPUT(v_kernel, _st);                                 // d_model, hidden_dim
    CHECK_INPUT(attr_output_kernel, _st);                       // hidden_dim, d_model
    CHECK_INPUT(attr_output_layernorm_gamma, _st);              // d_model
    CHECK_INPUT(inter_kernel, _st);                             // d_model, inter_size
    CHECK_INPUT(inter_kernel2, _st);                            // d_model, inter_size
    CHECK_INPUT(output_kernel, _st);                            // inter_size, d_model
    CHECK_INPUT(output_layernorm_gamma, _st);                   // d_model
    CHECK_INPUT(post_transformer_layernorm_gamma, _st);         // d_model
    CHECK_INPUT(absolute_or_relative_position_embedding, _st);  // head_num, num_bucket or max_seq_len, d_model
    CHECK_INPUT(embedding_table, _st);                          // vocab_size, d_model
    if (t5_with_bias) {
        CHECK_INPUT(q_bias, _st);                           // hidden_dim
        CHECK_INPUT(k_bias, _st);                           // hidden_dim
        CHECK_INPUT(v_bias, _st);                           // hidden_dim
        CHECK_INPUT(attr_output_bias, _st);                 // d_model
        CHECK_INPUT(attr_output_layernorm_beta, _st);       // d_model
        CHECK_INPUT(inter_bias, _st);                       // inter_size
        CHECK_INPUT(inter_bias2, _st);                      // inter_size
        CHECK_INPUT(output_bias, _st);                      // d_model
        CHECK_INPUT(output_layernorm_beta, _st);            // d_model
        CHECK_INPUT(post_transformer_layernorm_beta, _st);  // d_model
    }
    if (expert_num != 0) {
        CHECK_INPUT(moe_gate, _st);  // hidden_dim, num_experts
    }

    auto const adapter_layer_norm_type =
        ft::LinearAdapterConfig::toLayerNormType(adapter_norm_position.empty() ? "pre" : adapter_norm_position);

    switch (_st) {
        case at::ScalarType::Float:
            ft_t5_encoder = new FTT5Encoder<float>(head_num,
                                                   head_size,
                                                   inter_size,
                                                   d_model,
                                                   layer_num,
                                                   num_bucket,
                                                   expert_num,
                                                   max_distance,
                                                   sparse,
                                                   q_scaling,
                                                   moe_k,
                                                   tensor_para_size,
                                                   pipeline_para_size,
                                                   t5_with_bias,
                                                   ft::PositionEmbeddingType(position_embedding_type),
                                                   ft::getActivationType(activation_type),
                                                   adapter_inter_size,
                                                   adapter_layer_norm_type,
                                                   moe_layer_index,
                                                   weights);
            break;
        case at::ScalarType::Half:
            ft_t5_encoder = new FTT5Encoder<half>(head_num,
                                                  head_size,
                                                  inter_size,
                                                  d_model,
                                                  layer_num,
                                                  num_bucket,
                                                  expert_num,
                                                  max_distance,
                                                  sparse,
                                                  q_scaling,
                                                  moe_k,
                                                  tensor_para_size,
                                                  pipeline_para_size,
                                                  t5_with_bias,
                                                  ft::PositionEmbeddingType(position_embedding_type),
                                                  ft::getActivationType(activation_type),
                                                  adapter_inter_size,
                                                  adapter_layer_norm_type,
                                                  moe_layer_index,
                                                  weights);
            break;
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16:
            ft_t5_encoder = new FTT5Encoder<__nv_bfloat16>(head_num,
                                                           head_size,
                                                           inter_size,
                                                           d_model,
                                                           layer_num,
                                                           num_bucket,
                                                           expert_num,
                                                           max_distance,
                                                           sparse,
                                                           q_scaling,
                                                           moe_k,
                                                           tensor_para_size,
                                                           pipeline_para_size,
                                                           t5_with_bias,
                                                           ft::PositionEmbeddingType(position_embedding_type),
                                                           ft::getActivationType(activation_type),
                                                           adapter_inter_size,
                                                           adapter_layer_norm_type,
                                                           moe_layer_index,
                                                           weights);
            break;
#endif
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
}

FasterTransformerT5Encoder::~FasterTransformerT5Encoder()
{
    delete ft_t5_encoder;
}

th::Tensor FasterTransformerT5Encoder::forward(th::optional<th::Tensor> input_ids,
                                               th::Tensor               sequence_lengths,
                                               th::optional<th::Tensor> inputs_embeds)
{
    if (input_ids.has_value()) {
        CHECK_CONTIGUOUS(input_ids.value());
        TORCH_CHECK(input_ids.value().dtype() == torch::kInt32, "input_ids dtype should be int32");
    }

    CHECK_CONTIGUOUS(sequence_lengths);
    TORCH_CHECK(sequence_lengths.dtype() == torch::kInt32, "sequence_lengths dtype should be int32");

    if (inputs_embeds.has_value()) {
        CHECK_CONTIGUOUS(inputs_embeds.value());
        TORCH_CHECK(inputs_embeds.value().dtype() == torch::kFloat32
                        || inputs_embeds.value().dtype() == torch::kFloat16,
                    "inputs_embeds dtype should be float32 or float16");
    }

    TORCH_CHECK(input_ids.has_value() || inputs_embeds.has_value(),
                "input_ids and inputs_embeds should not be empty at the same time.");

    size_t  batch_size = inputs_embeds.has_value() ? inputs_embeds.value().size(0) : input_ids.value().size(0);
    size_t  seq_len    = inputs_embeds.has_value() ? inputs_embeds.value().size(1) : input_ids.value().size(1);
    int64_t d_model    = d_model_;

    auto output = torch::empty({(long int)batch_size, (long int)seq_len, (long int)d_model},
                               torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    ft_t5_encoder->forward(batch_size, seq_len, input_ids, sequence_lengths, inputs_embeds, output, _remove_padding);
    return output;
}

std::vector<th::Tensor> FasterTransformerT5Encoder::get_pickle_info() const
{
    std::vector<th::Tensor> tmp(weights);
    return tmp;
}

}  // namespace torch_ext

static auto fasterTransformerT5EncoderTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::FasterTransformerT5Encoder>("FasterTransformerT5Encoder")
#else
    torch::jit::class_<torch_ext::FasterTransformerT5Encoder>("FasterTransformer", "T5Encoder")
#endif
        .def(torch::jit::init<th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              std::vector<int64_t>,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              bool,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              bool,
                              double,
                              int64_t,
                              int64_t,
                              bool,
                              int64_t,
                              int64_t,
                              std::string,
                              int64_t,
                              std::string>())
        .def("forward", &torch_ext::FasterTransformerT5Encoder::forward);
