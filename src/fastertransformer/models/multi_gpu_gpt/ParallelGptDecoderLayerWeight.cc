/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const int hidden_units,
                                                                const int inter_size,
                                                                const int tensor_para_size,
                                                                const int tensor_para_rank,
                                                                const int int8_mode):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    int8_mode_(int8_mode)
{
    mallocWeights();
    setWeightPtr();
    if (int8_mode_ != 0) {
        transposeCalibrateQuantizeWeight();
    }
}

template<typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const int int8_mode): int8_mode_(int8_mode)
{
}

template<typename T>
ParallelGptDecoderLayerWeight<T>::~ParallelGptDecoderLayerWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < 12; i++) {
            deviceFree(weights_ptr[i]);
        }

        pre_layernorm_weights.beta = nullptr;
        pre_layernorm_weights.gamma = nullptr;
        self_attention_weights.query_weight.kernel = nullptr;
        self_attention_weights.query_weight.bias = nullptr;
        self_attention_weights.attention_output_weight.kernel = nullptr;
        self_attention_weights.attention_output_weight.bias = nullptr;
        self_attn_layernorm_weights.beta = nullptr;
        self_attn_layernorm_weights.gamma = nullptr;

        ffn_weights.intermediate_weight.kernel = nullptr;
        ffn_weights.intermediate_weight.bias = nullptr;
        ffn_weights.output_weight.kernel = nullptr;
        ffn_weights.output_weight.bias = nullptr;

        if (int8_mode_ != 0) {
            for (int i = 0; i < 4; i++) {
                deviceFree(int8_weights_ptr[i]);
                deviceFree(scale_ptr[i]);
            }
            self_attention_weights.query_weight.int8_kernel = nullptr;
            self_attention_weights.query_weight.scale = nullptr;
            self_attention_weights.attention_output_weight.int8_kernel = nullptr;
            self_attention_weights.attention_output_weight.scale = nullptr;
            ffn_weights.intermediate_weight.int8_kernel = nullptr;
            ffn_weights.intermediate_weight.scale = nullptr;
            ffn_weights.output_weight.int8_kernel = nullptr;
            ffn_weights.output_weight.scale = nullptr;
        }

        is_maintain_buffer = false;
    }
}

template<typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const ParallelGptDecoderLayerWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    int8_mode_(other.int8_mode_)
{
    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], hidden_units_);

    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);

    if (int8_mode_ != 0) {
        cudaD2Dcpy(
            int8_weights_ptr[0], other.int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[1], other.int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(int8_weights_ptr[2], other.int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[3], other.int8_weights_ptr[3], inter_size_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(scale_ptr[0], other.scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[1], other.scale_ptr[1], hidden_units_);
        cudaD2Dcpy(scale_ptr[2], other.scale_ptr[2], inter_size_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[3], other.scale_ptr[3], hidden_units_);
    }

    setWeightPtr();
}

template<typename T>
ParallelGptDecoderLayerWeight<T>&
ParallelGptDecoderLayerWeight<T>::operator=(const ParallelGptDecoderLayerWeight& other)
{
    hidden_units_ = other.hidden_units_;
    inter_size_ = other.inter_size_;
    tensor_para_size_ = other.tensor_para_size_;
    tensor_para_rank_ = other.tensor_para_rank_;
    int8_mode_ = other.int8_mode_;

    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], hidden_units_);

    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);

    if (int8_mode_ != 0) {
        cudaD2Dcpy(
            int8_weights_ptr[0], other.int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[1], other.int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(int8_weights_ptr[2], other.int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[3], other.int8_weights_ptr[3], inter_size_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(scale_ptr[0], other.scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[1], other.scale_ptr[1], hidden_units_);
        cudaD2Dcpy(scale_ptr[2], other.scale_ptr[2], inter_size_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[3], other.scale_ptr[3], hidden_units_);
    }

    setWeightPtr();
    return *this;
}

template<typename T>
void ParallelGptDecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T>(weights_ptr[0], {(int)hidden_units_}, dir_path + ".input_layernorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[1], {(int)hidden_units_}, dir_path + ".input_layernorm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[2],
                         {(int)hidden_units_, (int)(3 * hidden_units_ / tensor_para_size_)},
                         dir_path + ".attention.query_key_value.weight." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[3],
                         {3, (int)(hidden_units_ / tensor_para_size_)},
                         dir_path + ".attention.query_key_value.bias." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[4],
                         {(int)(hidden_units_ / tensor_para_size_), (int)hidden_units_},
                         dir_path + ".attention.dense.weight." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[5], {(int)hidden_units_}, dir_path + ".attention.dense.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[6], {(int)hidden_units_}, dir_path + ".post_attention_layernorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[7], {(int)hidden_units_}, dir_path + ".post_attention_layernorm.weight.bin", model_file_type);

    loadWeightFromBin<T>(weights_ptr[8],
                         {(int)hidden_units_, (int)(inter_size_ / tensor_para_size_)},
                         dir_path + ".mlp.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[9],
                         {(int)(inter_size_ / tensor_para_size_)},
                         dir_path + ".mlp.dense_h_to_4h.bias." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[10],
                         {(int)(inter_size_ / tensor_para_size_), (int)hidden_units_},
                         dir_path + ".mlp.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[11], {(int)hidden_units_}, dir_path + ".mlp.dense_4h_to_h.bias.bin", model_file_type);

    if (int8_mode_ != 0) {
        transposeCalibrateQuantizeWeight();
    }
}

template<typename T>
void ParallelGptDecoderLayerWeight<T>::setWeightPtr()
{
    pre_layernorm_weights.beta = weights_ptr[0];
    pre_layernorm_weights.gamma = weights_ptr[1];
    self_attention_weights.query_weight.kernel = weights_ptr[2];
    self_attention_weights.query_weight.bias = weights_ptr[3];
    self_attention_weights.attention_output_weight.kernel = weights_ptr[4];
    self_attention_weights.attention_output_weight.bias = weights_ptr[5];
    self_attn_layernorm_weights.beta = weights_ptr[6];
    self_attn_layernorm_weights.gamma = weights_ptr[7];

    ffn_weights.intermediate_weight.kernel = weights_ptr[8];
    ffn_weights.intermediate_weight.bias = weights_ptr[9];
    ffn_weights.output_weight.kernel = weights_ptr[10];
    ffn_weights.output_weight.bias = weights_ptr[11];

    if (int8_mode_ != 0) {
        self_attention_weights.query_weight.int8_kernel = int8_weights_ptr[0];
        self_attention_weights.query_weight.scale = scale_ptr[0];
        self_attention_weights.attention_output_weight.int8_kernel = int8_weights_ptr[1];
        self_attention_weights.attention_output_weight.scale = scale_ptr[1];
        ffn_weights.intermediate_weight.int8_kernel = int8_weights_ptr[2];
        ffn_weights.intermediate_weight.scale = scale_ptr[2];
        ffn_weights.output_weight.int8_kernel = int8_weights_ptr[3];
        ffn_weights.output_weight.scale = scale_ptr[3];
    }

    is_maintain_buffer = true;
}

template<typename T>
void ParallelGptDecoderLayerWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[0], hidden_units_);
    deviceMalloc(&weights_ptr[1], hidden_units_);
    deviceMalloc(&weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[5], hidden_units_);
    deviceMalloc(&weights_ptr[6], hidden_units_);
    deviceMalloc(&weights_ptr[7], hidden_units_);

    deviceMalloc(&weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[9], inter_size_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[11], hidden_units_);

    if (int8_mode_ != 0) {
        deviceMalloc(&int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        deviceMalloc(&int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        deviceMalloc(&int8_weights_ptr[3], inter_size_ / tensor_para_size_ * hidden_units_);

        deviceMalloc(&scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&scale_ptr[1], hidden_units_);
        deviceMalloc(&scale_ptr[2], inter_size_ / tensor_para_size_);
        deviceMalloc(&scale_ptr[3], hidden_units_);
    }
}

#ifdef SPARSITY_ENABLED
template<typename T>
void ParallelGptDecoderLayerWeight<T>::compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim)
{
    hidden_units_ = hidden_dim;
    inter_size_ = 4 * hidden_units_;

    const size_t num_sparse_weights = 4;
    size_t shapes[num_sparse_weights][2] = {{hidden_units_, 3 * hidden_units_ / tensor_para_size_},
                                            {hidden_units_ / tensor_para_size_, hidden_units_},
                                            {hidden_units_, inter_size_ / tensor_para_size_},
                                            {inter_size_ / tensor_para_size_, hidden_units_}};

    const T* dense_weights[num_sparse_weights] = {self_attention_weights.query_weight.kernel,
                                                  self_attention_weights.attention_output_weight.kernel,
                                                  ffn_weights.intermediate_weight.kernel,
                                                  ffn_weights.output_weight.kernel};

    for (size_t i = 0; i < num_sparse_weights; ++i) {
        int m = shapes[i][1];
        int k = shapes[i][0];
        size_t compressed_size = cublas_wrapper.getSparseMatrixSize(m, k);
        deviceMalloc(&sp_weights_ptr[i], static_cast<int>(compressed_size), false);
        cublas_wrapper.compressMatrix(dense_weights[i], sp_weights_ptr[i], m, k);
    }

    self_attention_weights.query_weight.sp_kernel = sp_weights_ptr[0];
    self_attention_weights.attention_output_weight.sp_kernel = sp_weights_ptr[1];
    ffn_weights.intermediate_weight.sp_kernel = sp_weights_ptr[2];
    ffn_weights.output_weight.sp_kernel = sp_weights_ptr[3];
    is_maintain_sp_buffer = true;
}
#endif

template<typename T>
void ParallelGptDecoderLayerWeight<T>::transposeCalibrateQuantizeWeight()
{
    invokeLdnCalibrateWeightPerChannel(
        scale_ptr[0], weights_ptr[2], hidden_units_, 3 * hidden_units_ / tensor_para_size_, stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(int8_weights_ptr[0],
                                               scale_ptr[0],
                                               weights_ptr[2],
                                               hidden_units_,
                                               3 * hidden_units_ / tensor_para_size_,
                                               stream_);

    invokeLdnCalibrateWeightPerChannel(
        scale_ptr[1], weights_ptr[4], hidden_units_ / tensor_para_size_, hidden_units_, stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(
        int8_weights_ptr[1], scale_ptr[1], weights_ptr[4], hidden_units_ / tensor_para_size_, hidden_units_, stream_);

    invokeLdnCalibrateWeightPerChannel(
        scale_ptr[2], weights_ptr[8], hidden_units_, inter_size_ / tensor_para_size_, stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(
        int8_weights_ptr[2], scale_ptr[2], weights_ptr[8], hidden_units_, inter_size_ / tensor_para_size_, stream_);

    invokeLdnCalibrateWeightPerChannel(
        scale_ptr[3], weights_ptr[10], inter_size_ / tensor_para_size_, hidden_units_, stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(
        int8_weights_ptr[3], scale_ptr[3], weights_ptr[10], inter_size_ / tensor_para_size_, hidden_units_, stream_);
}

template struct ParallelGptDecoderLayerWeight<float>;
template struct ParallelGptDecoderLayerWeight<half>;
#ifdef ENABLE_BF16
template struct ParallelGptDecoderLayerWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
