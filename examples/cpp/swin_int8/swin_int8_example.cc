/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "examples/cpp/swin/functions.h"
#include "src/fastertransformer/models/swin_int8/SwinINT8.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "stdio.h"
#include "stdlib.h"
#include <cublasLt.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <sys/time.h>

using namespace fastertransformer;
using namespace std;

template<typename T>
void test(int version, int int8_mode, int model_type, int window_size, int img_size, int batch)
{
    cudnnHandle_t    cudnn_handle;
    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t     stream;
    cudaStreamCreate(&stream);
    checkCUDNN(cudnnCreate(&cudnn_handle));
    checkCUDNN(cudnnSetStream(cudnn_handle, stream));
    check_cuda_error(cublasCreate(&cublas_handle));
    check_cuda_error(cublasLtCreate(&cublaslt_handle));
    check_cuda_error(cublasSetStream(cublas_handle, stream));

    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap(IGEMM_CONFIG);

    std::mutex* cublas_wrapper_mutex = new std::mutex();

    int sm = getSMVersion();

    bool _use_ORDER_COL32_2R_4R4 = false;
#if (CUDART_VERSION >= 11000)
    if (sm >= 80) {
        _use_ORDER_COL32_2R_4R4 = true;
    }
#endif

    cublasINT8MMWrapper* cublas_wrapper = new cublasINT8MMWrapper(
        cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, _use_ORDER_COL32_2R_4R4);
    if (std::is_same<T, half>::value) {
        cublas_wrapper->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }

    int embed_dim;
    int shift_size = window_size / 2;
    int depths[4], num_heads[4];

    if (model_type == 0) {  // tiny
        embed_dim    = 96;
        depths[0]    = 2;
        depths[1]    = 2;
        depths[2]    = 6;
        depths[3]    = 2;
        num_heads[0] = 3;
        num_heads[1] = 6;
        num_heads[2] = 12;
        num_heads[3] = 24;
    }
    else if (model_type == 1) {  // small
        embed_dim    = 96;
        depths[0]    = 2;
        depths[1]    = 2;
        depths[2]    = 18;
        depths[3]    = 2;
        num_heads[0] = 3;
        num_heads[1] = 6;
        num_heads[2] = 12;
        num_heads[3] = 24;
    }
    else if (model_type == 2) {  // base
        embed_dim    = 128;
        depths[0]    = 2;
        depths[1]    = 2;
        depths[2]    = 18;
        depths[3]    = 2;
        num_heads[0] = 4;
        num_heads[1] = 8;
        num_heads[2] = 16;
        num_heads[3] = 32;
    }
    else if (model_type == 3) {  // large
        embed_dim    = 192;
        depths[0]    = 2;
        depths[1]    = 2;
        depths[2]    = 18;
        depths[3]    = 2;
        num_heads[0] = 6;
        num_heads[1] = 12;
        num_heads[2] = 24;
        num_heads[3] = 48;
    }
    else if (model_type == 4) {  // huge
        embed_dim    = 352;
        depths[0]    = 2;
        depths[1]    = 2;
        depths[2]    = 18;
        depths[3]    = 2;
        num_heads[0] = 11;
        num_heads[1] = 22;
        num_heads[2] = 44;
        num_heads[3] = 88;
    }
    else if (model_type == 5) {  // gaint
        embed_dim    = 512;
        depths[0]    = 2;
        depths[1]    = 2;
        depths[2]    = 42;
        depths[3]    = 2;
        num_heads[0] = 16;
        num_heads[1] = 32;
        num_heads[2] = 64;
        num_heads[3] = 128;
    }
    else {
        FT_LOG_ERROR("Unsupported model type!");
        exit(-1);
    }
    int   in_chans   = 3;
    bool  ape        = false;
    bool  patch_norm = true;
    float mlp_ratio  = 4.0f;
    bool  qkv_bias   = true;
    float qk_scale   = 1.0f;
    int   layer_num  = 4;
    int   patch_size = 4;

    int output_dim = int(pow(2, layer_num - 1)) * embed_dim;
    int weight_num = getWeightNum(layer_num, depths, version);
    // calculate the size of each weight
    std::vector<size_t> weight_size;
    std::vector<T*>     weight;
    generateWeightSize(weight_size,
                       layer_num,
                       embed_dim,
                       mlp_ratio,
                       window_size,
                       img_size,
                       patch_size,
                       in_chans,
                       depths,
                       num_heads,
                       version);
    for (int i = 0; i < weight_size.size(); i++) {
        T* weight_ptr;
        deviceMalloc(&weight_ptr, weight_size[i], false);
        weight.push_back(weight_ptr);
    }

    SwinTransformerINT8Weight<T> params;
    int                          weight_idx = 0;
    // We should pre-process the weights
    // as we do in examples/pytorch/swin/SwinTransformerWeightTransposeQKVWeight.py
    // in the following sample, we assume all the weights are well pre-processed
    for (int l = 0; l < layer_num; l++) {
        SwinTransformerINT8BasicLayerWeight<T> bl;
        for (int di = 0; di < depths[l]; di++) {
            SwinTransformerINT8BlockWeight<T> p;
            // qkv weight should be transposed from [3*head*size, k] into [k, 3*head*size]
            p.attention_weights.query_weight.kernel = weight[weight_idx++];
            // for Version = 1 ; qkv bias should be in shape [3*head*size]
            // for Version = 2 ; we concat q_bias, zero_k_bias, v_bias into [3*head*size]
            p.attention_weights.query_weight.bias              = weight[weight_idx++];
            p.attention_weights.attention_output_weight.kernel = weight[weight_idx++];
            p.attention_weights.attention_output_weight.bias   = weight[weight_idx++];
            p.ffn_weights.intermediate_weight.kernel           = weight[weight_idx++];
            p.ffn_weights.intermediate_weight.bias             = weight[weight_idx++];
            p.ffn_weights.output_weight.kernel                 = weight[weight_idx++];
            p.ffn_weights.output_weight.bias                   = weight[weight_idx++];
            p.attn_layernorm_weights.gamma                     = weight[weight_idx++];
            p.attn_layernorm_weights.beta                      = weight[weight_idx++];
            p.ffn_layernorm_weights.gamma                      = weight[weight_idx++];
            p.ffn_layernorm_weights.beta                       = weight[weight_idx++];
            // Please use invokeGenRelativePosBias(for version == 1) or invokeGenRelativePosBiasV2(for version == 2) to
            // get attention_relative_pos_bias from attention_relative_pos_bias_table;
            // Notice : for some model, like (img_size, window_size) = (224, 16), the window_size_in_use of last layer
            // may changes
            p.attention_relative_pos_bias = weight[weight_idx++];

            // For cases we can use trt fused mha kernels, we should use invokeTransformMask() to transform the
            // relative_position_bias; For cases we can not use trt fused mha kernels, we should set
            // trt_relative_position_bias = nullptr, to save device memory
            p.trt_relative_position_bias = weight[weight_idx++];

            // if version = 2, attention_logit_scale should be processed as
            // torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
            p.attention_logit_scale = (version == 1) ? nullptr : weight[weight_idx++];
            p.scalelist.size_       = ACTIVATION_AMAX_NUM + 5 + INT8O_GEMM_NUM + TRT_AMAX_NUM;
            p.scalelist.p2_offset_  = ACTIVATION_AMAX_NUM;
            p.scalelist.p3_offset_  = ACTIVATION_AMAX_NUM + 5;
            p.scalelist.p4_offset_  = ACTIVATION_AMAX_NUM + 5 + INT8O_GEMM_NUM;
            float* d_scale_list;
            deviceMalloc(&(d_scale_list), p.scalelist.size_, false);
            p.scalelist.d_scale_list_ = d_scale_list;
            p.scalelist.h_scale_list_ = (float*)malloc(p.scalelist.size_ * sizeof(float));
            // cudaMemcpy(p.scalelist.h_scale_list_, p.scalelist.d_scale_list_, 96, cudaMemcpyDeviceToHost);
            bl.block_weight_list.push_back(p);
        }
        bl.merge_layernorm_weights.gamma = weight[weight_idx++];
        bl.merge_layernorm_weights.beta  = weight[weight_idx++];
        bl.merge_linear_weights.kernel   = weight[weight_idx++];
        bl.attn_mask                     = weight[weight_idx++];

        // For cases we can use trt fused mha kernels, we should use invokeTransformMask() to transform the attn_mask;
        // For cases we can not use trt fused mha kernels, we should set trt_attn_mask = nullptr, to save device memory
        bl.trt_attn_mask = weight[weight_idx++];

        params.basic_layer_weight_list.push_back(bl);
    }
    params.patchEmbed_linear_weights.kernel = weight[weight_idx++];
    params.patchEmbed_linear_weights.bias   = weight[weight_idx++];
    params.patchEmbed_norm_weights.gamma    = weight[weight_idx++];
    params.patchEmbed_norm_weights.beta     = weight[weight_idx++];
    params.norm_weights.gamma               = weight[weight_idx++];
    params.norm_weights.beta                = weight[weight_idx++];

    T *input_d, *output_d;
    deviceMalloc(&input_d, batch * img_size * img_size * in_chans, false);
    deviceMalloc(&output_d, batch * output_dim, false);

    fastertransformer::Allocator<AllocatorType::CUDA> allocator(0);

    int                    max_batch = batch;
    SwinTransformerINT8<T> sw(int8_mode,
                              max_batch,
                              img_size,
                              patch_size,
                              in_chans,
                              embed_dim,
                              window_size,
                              depths,
                              num_heads,
                              ape,
                              patch_norm,
                              layer_num,
                              mlp_ratio,
                              cudnn_handle,
                              stream,
                              cublas_wrapper,
                              &allocator,
                              false,
                              qkv_bias,
                              qk_scale,
                              version);

    int       sm_ptr[1] = {sm};
    TensorMap input_tensors{
        {"input_query",
         Tensor{MEMORY_GPU,
                getTensorType<T>(),
                std::vector<size_t>{(size_t)batch, (size_t)in_chans, (size_t)img_size, (size_t)img_size},
                input_d}},
        {"additional_params", Tensor{MEMORY_CPU, TYPE_INT8, std::vector<size_t>{1}, sm_ptr}}};

    TensorMap output_tensors{
        {"hidden_features",
         Tensor{MEMORY_GPU, getTensorType<T>(), std::vector<size_t>{(size_t)batch, (size_t)output_dim}, output_d}}};

    // warmup
    for (int i = 0; i < 10; i++) {
        sw.forward(&output_tensors, &input_tensors, params);
    }

    int       ite = 100;
    CudaTimer cuda_timer(stream);
    cuda_timer.start();
    for (int i = 0; i < ite; i++) {
        sw.forward(&output_tensors, &input_tensors, params);
    }
    float total_time = cuda_timer.stop();

    FT_LOG_INFO("batch_size %ld model_type:%d "
                "FT-CPP-time %.2f ms (%d iterations) ",
                batch,
                model_type,
                total_time / ite,
                ite);

    delete cublas_algo_map;
    delete cublas_wrapper_mutex;

    // free data
    for (int i = 0; i < weight.size(); i++) {
        check_cuda_error(cudaFree(weight[i]));
    }
    check_cuda_error(cudaFree(output_d));
    check_cuda_error(cudaFree(input_d));
    check_cuda_error(cublasDestroy(cublas_handle));
    checkCUDNN(cudnnDestroy(cudnn_handle));

    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
}

int main(int argc, char* argv[])
{
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    if (argc != 7) {
        FT_LOG_ERROR(
            "swin_int8_example version(1/2) int8_mode(1/2) model_type(0-3) window_size(7/8/12/16/24) img_size(192/224/256/384) batch_size");
        FT_LOG_INFO("model_type:");
        FT_LOG_INFO("0: tiny");
        FT_LOG_INFO("1: small");
        FT_LOG_INFO("2: base");
        FT_LOG_INFO("3: large");
        FT_LOG_INFO("e.g., ./bin/swin_int8_example 1 1 0 7 224 32");
        return 0;
    }
    FT_LOG_INFO("Device %s", prop.name);

    int version     = atoi(argv[1]);
    int int8_mode   = atoi(argv[2]);
    int model_type  = atoi(argv[3]);
    int window_size = atoi(argv[4]);
    int img_size    = atoi(argv[5]);
    int batch       = atoi(argv[6]);

    if (version != 1 && version != 2) {
        FT_LOG_ERROR("version is not supported");
        return -1;
    }

    test<half>(version, int8_mode, model_type, window_size, img_size, batch);
}
