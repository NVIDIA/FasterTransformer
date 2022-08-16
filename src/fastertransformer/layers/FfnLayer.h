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

#pragma once

#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/matrix_vector_multiplication.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <vector>

namespace fastertransformer {

enum class ActivationType {
    Gelu,
    Relu,
    Silu,
    GeGLU,
    ReGLU,
    SiGLU,
    InvalidType
};

inline ActivationType getActivationType(std::string activation_type_str)
{
    if (activation_type_str == "Gelu" || activation_type_str == "gelu") {
        return ActivationType::Gelu;
    }
    else if (activation_type_str == "Relu" || activation_type_str == "relu") {
        return ActivationType::Relu;
    }
    else if (activation_type_str == "Silu" || activation_type_str == "silu") {
        return ActivationType::Silu;
    }
    else if (activation_type_str == "GeGLU" || activation_type_str == "geglu" || activation_type_str == "gated-gelu") {
        return ActivationType::GeGLU;
    }
    else if (activation_type_str == "ReGLU" || activation_type_str == "reglu" || activation_type_str == "gated-relu") {
        return ActivationType::ReGLU;
    }
    else if (activation_type_str == "SiGLU" || activation_type_str == "gated-silu") {
        return ActivationType::SiGLU;
    }
    else {
        FT_CHECK_WITH_INFO(false, "Activation Type: " + activation_type_str + " not supported !");
    }
    return ActivationType::InvalidType;
}

inline bool isGatedActivation(ActivationType activaiton_type)
{
    return activaiton_type == ActivationType::GeGLU || activaiton_type == ActivationType::ReGLU
           || activaiton_type == ActivationType::SiGLU;
}

template<typename T>
class FfnLayer: public BaseLayer {
private:
    // buffer handling
    size_t max_token_num_ = 0;

    // meta data
    size_t head_num_;
    size_t size_per_head_;

    // int8_mode_ == 1 for weight quantized only gemm for GPT
    int int8_mode_ = 0;

    // calculated data
    size_t hidden_units_;

    // gated activation
    bool use_gated_activation_;

    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidTokenNum(size_t token_num);
    void allocateBuffer(size_t token_num);

protected:
    T* inter_buf_   = nullptr;
    T* inter_buf_2_ = nullptr;  // for gated activation
    // the inter size for runtime ffn layer
    size_t inter_size_;
    /* used to allocater memory buffers
       different ffn layers (inter_size) will
       reuse the same ffn layer with the max inter size.
       max_inter_size will be passed as inter_size when initializing the ffn layer
    */
    size_t       max_inter_size_;
    virtual void invokeAddBiasActivation(const int m, const T* bias)                       = 0;
    virtual void invokeAddBiasGatedActivation(const int m, const T* bias1, const T* bias2) = 0;

public:
    FfnLayer(size_t           max_batch_size,
             size_t           max_seq_len,
             size_t           head_num,
             size_t           size_per_head,
             size_t           inter_size,
             cudaStream_t     stream,
             cublasMMWrapper* cublas_wrapper,
             IAllocator*      allocator,
             bool             is_free_buffer_after_forward,
             bool             sparse               = false,
             int              int8_mode            = 0,
             bool             use_gated_activation = false);

    FfnLayer(FfnLayer<T> const& ffn_layer);

    virtual ~FfnLayer();

    void resetInterSize(size_t runtime_inter_size)
    {
        inter_size_ = runtime_inter_size;
    }

    virtual void forward(std::vector<fastertransformer::Tensor>*       output_tensors,
                         const std::vector<fastertransformer::Tensor>* input_tensors,
                         const FfnWeight<T>*                           ffn_weights);
};

template<typename T>
class GeluFfnLayer: public FfnLayer<T> {
public:
    GeluFfnLayer(size_t           max_batch_size,
                 size_t           max_seq_len,
                 size_t           head_num,
                 size_t           size_per_head,
                 size_t           inter_size,
                 cudaStream_t     stream,
                 cublasMMWrapper* cublas_wrapper,
                 IAllocator*      allocator,
                 bool             is_free_buffer_after_forward,
                 bool             sparse               = false,
                 int              int8_mode            = 0,
                 bool             use_gated_activation = false);

    GeluFfnLayer(GeluFfnLayer<T> const& ffn_layer);

    virtual ~GeluFfnLayer() = default;

protected:
    using FfnLayer<T>::stream_;

private:
    using FfnLayer<T>::inter_buf_;
    using FfnLayer<T>::inter_buf_2_;
    using FfnLayer<T>::inter_size_;
    void invokeAddBiasActivation(const int m, const T* bias) override;
    void invokeAddBiasGatedActivation(const int m, const T* bias1, const T* bias2) override;
};

template<typename T>
class ReluFfnLayer: public FfnLayer<T> {
public:
    ReluFfnLayer(size_t           max_batch_size,
                 size_t           max_seq_len,
                 size_t           head_num,
                 size_t           size_per_head,
                 size_t           inter_size,
                 cudaStream_t     stream,
                 cublasMMWrapper* cublas_wrapper,
                 IAllocator*      allocator,
                 bool             is_free_buffer_after_forward,
                 bool             sparse               = false,
                 bool             use_gated_activation = false);

    ReluFfnLayer(ReluFfnLayer<T> const& ffn_layer);

    virtual ~ReluFfnLayer() = default;

protected:
    using FfnLayer<T>::stream_;

private:
    using FfnLayer<T>::inter_buf_;
    using FfnLayer<T>::inter_buf_2_;
    using FfnLayer<T>::inter_size_;
    void invokeAddBiasActivation(const int m, const T* bias) override;
    void invokeAddBiasGatedActivation(const int m, const T* bias1, const T* bias2) override;
};

template<typename T>
class SiluFfnLayer: public FfnLayer<T> {
public:
    SiluFfnLayer(size_t           max_batch_size,
                 size_t           max_seq_len,
                 size_t           head_num,
                 size_t           size_per_head,
                 size_t           inter_size,
                 cudaStream_t     stream,
                 cublasMMWrapper* cublas_wrapper,
                 IAllocator*      allocator,
                 bool             is_free_buffer_after_forward,
                 bool             sparse               = false,
                 bool             use_gated_activation = false);

    SiluFfnLayer(SiluFfnLayer<T> const& ffn_layer);

    virtual ~SiluFfnLayer() = default;

protected:
    using FfnLayer<T>::stream_;

private:
    using FfnLayer<T>::inter_buf_;
    using FfnLayer<T>::inter_buf_2_;
    using FfnLayer<T>::inter_size_;
    void invokeAddBiasActivation(const int m, const T* bias) override;
    void invokeAddBiasGatedActivation(const int m, const T* bias1, const T* bias2) override;
};

}  // namespace fastertransformer
