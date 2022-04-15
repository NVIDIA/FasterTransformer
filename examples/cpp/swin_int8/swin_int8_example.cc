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
void test(int model_type, int batch)
{
    cudnnHandle_t cudnn_handle;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;
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

    bool is_tiny = true;

    int embed_dim = is_tiny ? 96 : 192;
    int window_size = is_tiny ? 7 : 12;
    int int8_mode = is_tiny ? 2 : 4;
    int img_size = is_tiny ? 224 : 384;
    int shift_size = window_size / 2;
    int depths[4], num_heads[4];
    if (is_tiny) {
        depths[0] = 2;
        depths[1] = 2;
        depths[2] = 6;
        depths[3] = 2;
        num_heads[0] = 3;
        num_heads[1] = 6;
        num_heads[2] = 12;
        num_heads[3] = 24;
    }
    else {
        depths[0] = 2;
        depths[1] = 2;
        depths[2] = 18;
        depths[3] = 2;
        num_heads[0] = 6;
        num_heads[1] = 12;
        num_heads[2] = 24;
        num_heads[3] = 48;
    }

    if (model_type == 0) {
        int8_mode = 1;
        embed_dim = 96;
        window_size = 7;
        img_size = 224;
        shift_size = window_size / 2;
        depths[0] = 2;
        depths[1] = 2;
        depths[2] = 6;
        depths[3] = 2;
        num_heads[0] = 3;
        num_heads[1] = 6;
        num_heads[2] = 12;
        num_heads[3] = 24;
    }
    else if (model_type == 1) {
        int8_mode = 1;
        embed_dim = 96;
        window_size = 7;
        img_size = 224;
        shift_size = window_size / 2;
        depths[0] = 2;
        depths[1] = 2;
        depths[2] = 18;
        depths[3] = 2;
        num_heads[0] = 3;
        num_heads[1] = 6;
        num_heads[2] = 12;
        num_heads[3] = 24;
    }
    else if (model_type == 2) {
        int8_mode = 1;
        embed_dim = 128;
        window_size = 7;
        img_size = 224;
        shift_size = window_size / 2;
        depths[0] = 2;
        depths[1] = 2;
        depths[2] = 18;
        depths[3] = 2;
        num_heads[0] = 4;
        num_heads[1] = 8;
        num_heads[2] = 16;
        num_heads[3] = 32;
    }
    else if (model_type == 3) {
        int8_mode = 1;
        embed_dim = 128;
        window_size = 12;
        img_size = 384;
        shift_size = window_size / 2;
        depths[0] = 2;
        depths[1] = 2;
        depths[2] = 18;
        depths[3] = 2;
        num_heads[0] = 4;
        num_heads[1] = 8;
        num_heads[2] = 16;
        num_heads[3] = 32;
    }
    else if (model_type == 4) {
        int8_mode = 1;
        embed_dim = 192;
        window_size = 7;
        img_size = 224;
        shift_size = window_size / 2;
        depths[0] = 2;
        depths[1] = 2;
        depths[2] = 18;
        depths[3] = 2;
        num_heads[0] = 6;
        num_heads[1] = 12;
        num_heads[2] = 24;
        num_heads[3] = 48;
    }
    else if (model_type == 5) {
        int8_mode = 1;
        embed_dim = 192;
        window_size = 12;
        img_size = 384;
        shift_size = window_size / 2;
        depths[0] = 2;
        depths[1] = 2;
        depths[2] = 18;
        depths[3] = 2;
        num_heads[0] = 6;
        num_heads[1] = 12;
        num_heads[2] = 24;
        num_heads[3] = 48;
    }
    int in_chans = 3;
    bool ape = false;
    bool patch_norm = true;
    float mlp_ratio = 4.0f;
    bool qkv_bias = true;
    float qk_scale = 1.0f;
    int layer_num = 4;
    int patch_size = 4;

    int output_dim = int(pow(2, layer_num - 1)) * embed_dim;
    int weight_num = getWeightNum(layer_num, depths);
    // calculate the size of each weight
    std::vector<size_t> weight_size;
    std::vector<T*> weight;
    generateWeightSize(
        weight_size, layer_num, embed_dim, mlp_ratio, window_size, img_size, patch_size, in_chans, depths, num_heads);
    for (int i = 0; i < weight_size.size(); i++) {
        T* weight_ptr;
        deviceMalloc(&weight_ptr, weight_size[i], false);
        weight.push_back(weight_ptr);
    }

    SwinTransformerINT8Weight<T> params;
    int weight_idx = 0;
    int hidden_dim = embed_dim;
    for (int l = 0; l < layer_num; l++) {
        SwinTransformerINT8BasicLayerWeight<T> bl;
        for (int di = 0; di < depths[l]; di++) {
            SwinTransformerINT8BlockWeight<T> p;
            p.attention_weights.query_weight.kernel = weight[weight_idx++];
            p.attention_weights.query_weight.bias = weight[weight_idx++];
            p.attention_weights.attention_output_weight.kernel = weight[weight_idx++];
            p.attention_weights.attention_output_weight.bias = weight[weight_idx++];
            p.ffn_weights.intermediate_weight.kernel = weight[weight_idx++];
            p.ffn_weights.intermediate_weight.bias = weight[weight_idx++];
            p.ffn_weights.output_weight.kernel = weight[weight_idx++];
            p.ffn_weights.output_weight.bias = weight[weight_idx++];
            p.attn_layernorm_weights.gamma = weight[weight_idx++];
            p.attn_layernorm_weights.beta = weight[weight_idx++];
            p.ffn_layernorm_weights.gamma = weight[weight_idx++];
            p.ffn_layernorm_weights.beta = weight[weight_idx++];
            p.scalelist.size_ = ACTIVATION_AMAX_NUM + 5 + INT8O_GEMM_NUM + TRT_AMAX_NUM;
            p.scalelist.p2_offset_ = ACTIVATION_AMAX_NUM;
            p.scalelist.p3_offset_ = ACTIVATION_AMAX_NUM + 5;
            p.scalelist.p4_offset_ = ACTIVATION_AMAX_NUM + 5 + INT8O_GEMM_NUM;
            float* d_scale_list;
            deviceMalloc(&(d_scale_list), p.scalelist.size_, false);
            p.scalelist.d_scale_list_ = d_scale_list;
            p.scalelist.h_scale_list_ = (float*)malloc(p.scalelist.size_ * sizeof(float));
            // cudaMemcpy(p.scalelist.h_scale_list_, p.scalelist.d_scale_list_, 96, cudaMemcpyDeviceToHost);
            // Please use invokeGenRelativePosBias to get attention_relative_pos_bias from
            // attention_relative_pos_bias_table;
            p.attention_relative_pos_bias = weight[weight_idx++];
            bl.block_weight_list.push_back(p);
        }
        bl.merge_layernorm_weights.gamma = weight[weight_idx++];
        bl.merge_layernorm_weights.beta = weight[weight_idx++];
        bl.merge_linear_weights.kernel = weight[weight_idx++];
        bl.attn_mask = weight[weight_idx++];
        params.basic_layer_weight_list.push_back(bl);
        hidden_dim *= 2;
    }
    params.patchEmbed_linear_weights.kernel = weight[weight_idx++];
    params.patchEmbed_linear_weights.bias = weight[weight_idx++];
    params.patchEmbed_norm_weights.gamma = weight[weight_idx++];
    params.patchEmbed_norm_weights.beta = weight[weight_idx++];
    params.norm_weights.gamma = weight[weight_idx++];
    params.norm_weights.beta = weight[weight_idx++];

    T *input_d, *output_d;
    deviceMalloc(&input_d, batch * img_size * img_size * in_chans, false);
    deviceMalloc(&output_d, batch * output_dim, false);

    fastertransformer::Allocator<AllocatorType::CUDA> allocator(0);
    int max_batch = batch;
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
                              qk_scale);
    int sm_ptr[1] = {sm};
    std::vector<Tensor> input_tensors =
        std::vector<Tensor>{Tensor{MEMORY_GPU,
                                   getTensorType<T>(),
                                   std::vector<size_t>{(size_t)batch, (size_t)in_chans, (size_t)img_size * img_size},
                                   input_d},
                            Tensor{MEMORY_CPU, TYPE_INT8, std::vector<size_t>{1}, sm_ptr}};

    std::vector<Tensor> output_tensors =
        std::vector<Tensor>{Tensor{MEMORY_GPU,
                                   getTensorType<T>(),
                                   std::vector<size_t>{(size_t)batch, (size_t)img_size * img_size, (size_t)in_chans},
                                   output_d}};

    // warmup
    for (int i = 0; i < 10; i++) {
        sw.forward(&output_tensors, &input_tensors, params);
    }

    int ite = 100;
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
    if (argc != 3) {
        printf("[ERROR] swin_int8_example model_type(0-5) batch_size\n");
        printf("model_type:\n");
        printf("0: tiny\t7x7\n");
        printf("1: small\t7x7\n");
        printf("2: base\t7x7\n");
        printf("3: base\t12x12\n");
        printf("4: large\t7x7\n");
        printf("5: large\t12x12\n");
        printf("e.g., ./bin/swin_int8_example 0 32\n");
        return 0;
    }
    printf("Device %s\n", prop.name);

    int model_type = atoi(argv[1]);
    int batch = atoi(argv[2]);

    test<half>(model_type, batch);
}
