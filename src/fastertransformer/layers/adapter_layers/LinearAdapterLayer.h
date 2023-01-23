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

#pragma once

#include <memory>
#include <string>

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/layers/adapter_layers/LinearAdapterWeight.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace fastertransformer {

class LinearAdapterConfig {
public:
    LinearAdapterConfig(): inter_size_{0}, layer_norm_type_{LayerNormType::InvalidType} {}

    LinearAdapterConfig(std::size_t inter_size, LayerNormType layer_norm_type):
        inter_size_{inter_size}, layer_norm_type_{layer_norm_type}
    {
    }

    LinearAdapterConfig(const LinearAdapterConfig& other)            = default;
    LinearAdapterConfig& operator=(const LinearAdapterConfig& other) = default;

    std::size_t interSize() const
    {
        return inter_size_;
    }

    void interSize(std::size_t inter_size)
    {
        inter_size_ = inter_size;
    }

    LayerNormType layerNormType() const
    {
        return layer_norm_type_;
    }

    void layerNormType(LayerNormType layer_norm_type)
    {
        layer_norm_type_ = layer_norm_type;
    }

    void layerNormType(std::string const& layer_norm_type)
    {
        layer_norm_type_ = toLayerNormType(layer_norm_type);
    }

    bool enabled() const
    {
        return inter_size_ > 0;
    }

    std::string toString() const;

    static std::string toString(LayerNormType layer_norm_type);

    static LayerNormType toLayerNormType(std::string const& layer_norm_type);

private:
    std::size_t   inter_size_;
    LayerNormType layer_norm_type_;
};

template<typename T>
class LinearAdapterLayer: public BaseLayer {
public:
    LinearAdapterLayer(LinearAdapterConfig const&          config,
                       std::size_t                         max_batch_size,
                       std::size_t                         max_seq_len,
                       std::size_t                         hidden_size,
                       NcclParam const&                    tensor_para,
                       cudaStream_t                        stream,
                       cublasMMWrapper*                    cublas_wrapper,
                       IAllocator*                         allocator,
                       bool                                is_free_buffer_after_forward,
                       bool                                is_sparse,
                       std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm   = nullptr,
                       int                                 enable_custom_all_reduce = 0,
                       float                               layer_norm_eps           = 1e-6f);
    ~LinearAdapterLayer() override;

    LinearAdapterLayer(LinearAdapterLayer const&)            = delete;
    LinearAdapterLayer& operator=(LinearAdapterLayer const&) = delete;

    virtual void forward(Tensor* output, Tensor const* input, const LinearAdapterWeight<T>* adapter_weight);

    void setStream(cudaStream_t stream) override
    {
        BaseLayer::setStream(stream);
        ffn_layer_->setStream(stream);
    }

protected:
    void allocateBuffer() override;
    void allocateBuffer(std::size_t token_size);
    void freeBuffer() override;

    T* adapter_buf_ = nullptr;

private:
    std::unique_ptr<FfnLayer<T>> ffn_layer_;
    LayerNormType const          layer_norm_type_;
    float const                  layer_norm_eps_;
    std::size_t const            max_token_size_;
    std::size_t const            hidden_size_;
};

}  // namespace fastertransformer
