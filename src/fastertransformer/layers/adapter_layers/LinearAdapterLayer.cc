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

#include "LinearAdapterLayer.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/layers/TensorParallelSiluFfnLayer.h"
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

LayerNormType LinearAdapterConfig::toLayerNormType(const std::string& layer_norm_type)
{
    if (layer_norm_type == "pre") {
        return LayerNormType::pre_layernorm;
    }
    else if (layer_norm_type == "post") {
        return LayerNormType::post_layernorm;
    }
    else if (layer_norm_type == "invalid") {
        return LayerNormType::InvalidType;
    }
    else {
        FT_THROW("Layernorm Type: " + layer_norm_type + " not supported!");
    }
}

std::string LinearAdapterConfig::toString(LayerNormType layer_norm_type)
{
    if (layer_norm_type == LayerNormType::pre_layernorm) {
        return "pre";
    }
    else if (layer_norm_type == LayerNormType::post_layernorm) {
        return "post";
    }
    else if (layer_norm_type == LayerNormType::InvalidType) {
        return "invalid";
    }
    else {
        FT_THROW("Layernorm Type: " + std::to_string(static_cast<int>(layer_norm_type)) + " not supported!");
    }
}
std::string LinearAdapterConfig::toString() const
{
    return std::string("[inter_size: ") + std::to_string(inter_size_)
           + ", layer_norm_type: " + toString(layer_norm_type_) + "]";
}

template<typename T>
LinearAdapterLayer<T>::LinearAdapterLayer(LinearAdapterConfig const&          config,
                                          size_t                              max_batch_size,
                                          size_t                              max_seq_len,
                                          size_t                              hidden_size,
                                          NcclParam const&                    tensor_para,
                                          cudaStream_t                        stream,
                                          cublasMMWrapper*                    cublas_wrapper,
                                          IAllocator*                         allocator,
                                          bool                                is_free_buffer_after_forward,
                                          bool                                is_sparse,
                                          std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                                          int                                 enable_custom_all_reduce,
                                          float                               layer_norm_eps):
    BaseLayer{stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, is_sparse},
    ffn_layer_{new TensorParallelSiluFfnLayer<T>(max_batch_size,
                                                 max_seq_len,
                                                 1,
                                                 hidden_size,
                                                 0,
                                                 config.interSize(),
                                                 tensor_para,
                                                 stream,
                                                 cublas_wrapper,
                                                 allocator,
                                                 true,
                                                 is_free_buffer_after_forward,
                                                 is_sparse,
                                                 false,
                                                 custom_all_reduce_comm,
                                                 enable_custom_all_reduce,
                                                 0)},
    layer_norm_type_{config.layerNormType()},
    layer_norm_eps_{layer_norm_eps},
    max_token_size_{max_batch_size * max_seq_len},
    hidden_size_{hidden_size}
{
    FT_LOG_DEBUG("Constructing LinearAdapterLayer with config: %s", config.toString().c_str());
    allocateBuffer();
}

template<typename T>
LinearAdapterLayer<T>::~LinearAdapterLayer()
{
    freeBuffer();
}

template<typename T>
void LinearAdapterLayer<T>::allocateBuffer()
{
    adapter_buf_ = (T*)allocator_->reMalloc(adapter_buf_, sizeof(T) * max_token_size_ * hidden_size_, false);
}

template<typename T>
void LinearAdapterLayer<T>::allocateBuffer(size_t token_size)
{
    adapter_buf_ = (T*)allocator_->reMalloc(adapter_buf_, sizeof(T) * token_size * hidden_size_, false);
}

template<typename T>
void LinearAdapterLayer<T>::freeBuffer()
{
    if (adapter_buf_) {
        allocator_->free(reinterpret_cast<void**>(&adapter_buf_));
    }
}

template<typename T>
void LinearAdapterLayer<T>::forward(Tensor* output, const Tensor* input, const LinearAdapterWeight<T>* adapter_weight)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    auto* const output_data = output->getPtr<T>();
    auto* const input_data  = input->getPtr<T>();
    auto* const gamma       = adapter_weight->layer_norm_weight.gamma;
    auto* const beta        = adapter_weight->layer_norm_weight.beta;
    auto const  m           = input->shape[0];
    auto const  n           = input->shape[1];
    FT_CHECK(n == hidden_size_);
    FT_CHECK(output->shape[0] == m);
    FT_CHECK(output->shape[1] == n);
    allocateBuffer(m);

    Tensor    adapter_tensor{MemoryType::MEMORY_GPU, getTensorType<T>(), {m, n}, adapter_buf_};
    TensorMap output_map{{{"ffn_output", adapter_tensor}}};

    if (layer_norm_type_ == LayerNormType::pre_layernorm) {
        invokeGeneralLayerNorm(
            adapter_buf_, input_data, gamma, beta, layer_norm_eps_, m, n, nullptr, 0, ffn_layer_->getStream());
        TensorMap input_map{{{"ffn_input", adapter_tensor}}};
        ffn_layer_->forward(&output_map, &input_map, &adapter_weight->ffn_weight);
    }
    else if (layer_norm_type_ == LayerNormType::post_layernorm) {
        Tensor    input_tensor{MemoryType::MEMORY_GPU, getTensorType<T>(), {m, n}, input_data};
        TensorMap input_map{{{"ffn_input", input_tensor}}};
        ffn_layer_->forward(&output_map, &input_map, &adapter_weight->ffn_weight);
        invokeGeneralLayerNorm(
            adapter_buf_, adapter_buf_, gamma, beta, layer_norm_eps_, m, n, nullptr, 0, ffn_layer_->getStream());
    }
    else {
        FT_CHECK_WITH_INFO(false, "Unsupported layer norm type");
    }

    invokeT5AddResidual(output_data, adapter_buf_, m, n, ffn_layer_->getStream());

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class LinearAdapterLayer<float>;
template class LinearAdapterLayer<half>;
#ifdef ENABLE_BF16
template class LinearAdapterLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer