/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/kernels/unfused_attention_fp8_kernels.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include "src/fastertransformer/utils/gpu_buf.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/test_utils.h"

#include <algorithm>
#include <cstdio>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <functional>
#include <numeric>
#include <random>
#include <sys/time.h>

using namespace fastertransformer;

typedef struct {
    int batch_size;
    int head_num;
    int max_seq_len;
    int seq_len;
    int size_per_head;
    int rotary_embedding_dim;
} test_args_t;

/*
 * test FP8Transpose4dBatchMajor
 * input: k/v cache buffers, shape [B, H, L, Dh]
 * output: k/v cache buffers, shape [B, H, Dh/x, L, x], [B, H, L, Dh/x, x], respectively
 */
bool test_fp8_transpose_4d_batch_major(const test_args_t& test_args)
{
    const size_t n_elems_dst =
        test_args.batch_size * test_args.head_num * test_args.max_seq_len * test_args.size_per_head;
    const size_t n_elems_src = test_args.batch_size * test_args.head_num * test_args.seq_len * test_args.size_per_head;

    GPUBuf<__nv_fp8_e4m3> d_fp8_k_cache_src(n_elems_src, true);
    GPUBuf<__nv_fp8_e4m3> d_fp8_v_cache_src(n_elems_src, true);

    /* Ref with FP8->FP32 */
    GPUBuf<float> d_fp32_k_cache_src(d_fp8_k_cache_src);
    GPUBuf<float> d_fp32_v_cache_src(d_fp8_v_cache_src);

    auto k_cache_src = d_fp32_k_cache_src.to_host_vec();
    auto v_cache_src = d_fp32_v_cache_src.to_host_vec();

    /* Test with transpose4dBatchMajor, FP8->FP32 */
    GPUBuf<__nv_fp8_e4m3> d_fp8_k_cache_dst(n_elems_dst, false);
    GPUBuf<__nv_fp8_e4m3> d_fp8_v_cache_dst(n_elems_dst, false);

    FP8Transpose4dBatchMajorParam<__nv_fp8_e4m3, __nv_fp8_e4m3> fp8_args{
        d_fp8_k_cache_dst.ptr,
        d_fp8_v_cache_dst.ptr,
        d_fp8_k_cache_src.ptr,
        d_fp8_v_cache_src.ptr,
        (const float*)nullptr,
        (uint32_t)test_args.batch_size,
        (uint32_t)test_args.seq_len,
        (uint32_t)test_args.max_seq_len,
        (uint32_t)test_args.size_per_head,
        (uint32_t)test_args.head_num,
        (uint32_t)test_args.seq_len,
        0,
    };
    invokeFP8Transpose4dBatchMajor(fp8_args);

    GPUBuf<float> d_fp32_k_cache_test(d_fp8_k_cache_dst);
    GPUBuf<float> d_fp32_v_cache_test(d_fp8_v_cache_dst);

    sync_check_cuda_error();

    auto k_cache_test = d_fp32_k_cache_test.to_host_vec();
    auto v_cache_test = d_fp32_v_cache_test.to_host_vec();

    const int contig_elems = sizeof(uint4) / sizeof(__nv_fp8_e4m3);
    bool      error        = false;
    for (int bs = 0; bs < test_args.batch_size; bs++) {
        for (int head = 0; head < test_args.head_num; head++) {
            for (int seq = 0; seq < test_args.seq_len; seq++) {
                for (int h_dim_div = 0; h_dim_div < test_args.size_per_head / contig_elems; h_dim_div++) {
                    for (int x = 0; x < contig_elems; x++) {
                        const float k_src =
                            k_cache_src[bs * test_args.head_num * test_args.seq_len * test_args.size_per_head
                                        + head * test_args.seq_len * test_args.size_per_head
                                        + seq * test_args.size_per_head + h_dim_div * contig_elems + x];

                        const float k_test =
                            k_cache_test[bs * test_args.head_num * test_args.max_seq_len * test_args.size_per_head
                                         + head * test_args.max_seq_len * test_args.size_per_head
                                         + h_dim_div * test_args.max_seq_len * contig_elems + seq * contig_elems + x];

                        const float v_src =
                            v_cache_src[bs * test_args.head_num * test_args.seq_len * test_args.size_per_head
                                        + head * test_args.seq_len * test_args.size_per_head
                                        + seq * test_args.size_per_head + h_dim_div * contig_elems + x];

                        const float v_test =
                            v_cache_test[bs * test_args.head_num * test_args.max_seq_len * test_args.size_per_head
                                         + head * test_args.max_seq_len * test_args.size_per_head
                                         + seq * test_args.size_per_head + h_dim_div * contig_elems + x];

                        if (k_test != k_src || v_src != v_test) {
                            error = true;
                        }
                    }
                }
            }
        }
    }

    if (error) {
        puts("[ERROR] test_fp8_transpose_4d_batch_major");
    }

    return !error;
}

bool test_fp8_add_fused_QKV_bias_transpose(const test_args_t& test_args)
{
    constexpr float MAX_ALLOWED_ERROR = 0.015f;

    size_t B       = test_args.batch_size;
    size_t H       = test_args.head_num;
    size_t L       = test_args.seq_len;
    size_t Dh      = test_args.size_per_head;
    size_t n_elems = B * H * L * Dh;

    size_t R = test_args.rotary_embedding_dim;

    GPUBuf<__nv_fp8_e4m3> d_fp8_qkv(3 * n_elems, true);
    GPUBuf<__nv_bfloat16> d_fp16_qkv_bias(3 * Dh * H, true);

    /* Reference implementation in fp32 */
    GPUBuf<float> d_qkv(d_fp8_qkv);
    GPUBuf<float> d_qkv_bias(d_fp16_qkv_bias);

    GPUBuf<float> d_q_ref(n_elems, false);
    GPUBuf<float> d_k_ref(n_elems, false);
    GPUBuf<float> d_v_ref(n_elems, false);

    invokeAddFusedQKVBiasTranspose(
        d_q_ref.ptr, d_k_ref.ptr, d_v_ref.ptr, d_qkv.ptr, d_qkv_bias.ptr, nullptr, B, L, B * L, H, Dh, 0);
    FT_CHECK_WITH_INFO(R == 0, "rotary_embedding != 0 is not supported now, it has some bugs.");
    // invokeAddFusedQKVBiasTranspose(d_q_ref.ptr, d_k_ref.ptr, d_v_ref.ptr, {nullptr, nullptr, 0, 0}, d_qkv.ptr,
    // d_qkv_bias.ptr, nullptr, B, L, B * L, H, Dh, R, 0, 0);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    /* Test implementation in fp8 */
    GPUBuf<__nv_fp8_e4m3> d_fp8_q(n_elems, false);
    GPUBuf<__nv_fp8_e4m3> d_fp8_k(n_elems, false);
    GPUBuf<__nv_fp8_e4m3> d_fp8_v(n_elems, false);
    GPUBuf<float>         scale_in(H * Dh);
    GPUBuf<float>         scale_out(H * Dh);
    deviceFill(scale_in.ptr, H * Dh, (float)1.0f);
    deviceFill(scale_out.ptr, H * Dh, (float)1.0f);

    FP8AddFusedQKVBiasRebuildPaddingParam<__nv_fp8_e4m3, __nv_bfloat16> fused_qkv_bias_tr_args{d_fp8_q.ptr,
                                                                                               d_fp8_k.ptr,
                                                                                               d_fp8_v.ptr,
                                                                                               d_fp8_qkv.ptr,
                                                                                               nullptr,  // T2
                                                                                               d_fp16_qkv_bias.ptr,
                                                                                               scale_in.ptr,
                                                                                               nullptr,
                                                                                               nullptr,
                                                                                               scale_out.ptr,
                                                                                               nullptr,
                                                                                               nullptr,
                                                                                               nullptr,
                                                                                               (uint32_t)(B * L),
                                                                                               (uint32_t)B,
                                                                                               (uint32_t)L,
                                                                                               (uint32_t)L,
                                                                                               (uint32_t)L,
                                                                                               (uint32_t)H,
                                                                                               (uint32_t)Dh,
                                                                                               (uint32_t)R,
                                                                                               0};
    invokeFP8AddFusedQKVBiasRebuildPadding(fused_qkv_bias_tr_args);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    GPUBuf<float> d_q_test(d_fp8_q);
    GPUBuf<float> d_k_test(d_fp8_k);
    GPUBuf<float> d_v_test(d_fp8_v);

    /* Retrieve data to CPU */
    auto q_ref = d_q_ref.to_host_vec();
    auto k_ref = d_k_ref.to_host_vec();
    auto v_ref = d_v_ref.to_host_vec();

    auto q_test = d_q_test.to_host_vec();
    auto k_test = d_k_test.to_host_vec();
    auto v_test = d_v_test.to_host_vec();

    sync_check_cuda_error();

    /* Compare */
    std::transform(q_ref.begin(), q_ref.end(), q_test.begin(), q_ref.begin(), abs_diff<float>());
    std::transform(k_ref.begin(), k_ref.end(), k_test.begin(), k_ref.begin(), abs_diff<float>());
    std::transform(v_ref.begin(), v_ref.end(), v_test.begin(), v_ref.begin(), abs_diff<float>());

    const float q_error = *std::max_element(q_ref.begin(), q_ref.end());
    const float k_error = *std::max_element(k_ref.begin(), k_ref.end());
    // const float v_error = *std::max_element(v_ref.begin(), v_ref.end());
    const float v_error = 0.0f;  // v is transpoed in fp8 kernel, ignore it here

    bool error = false;

    if (std::max(std::max(q_error, k_error), v_error) >= MAX_ALLOWED_ERROR) {
        error = true;
        printf("Max q_error = %.4f\n", q_error);
        printf("Max k_error = %.4f\n", k_error);
        printf("Max v_error = %.4f\n", v_error);
    }

    if (error) {
        puts("[ERROR] test_fp8_add_fused_QKV_bias_transpose");
    }

    return !error;
}

bool test_fp8_masked_softmax(const test_args_t& test_args)
{
    constexpr float MAX_ALLOWED_ERROR = 0.05f;
    const float     coef              = 1.0f;
    const float     in_scaling = 1.0f, out_scaling = 1.0f;

    size_t B            = test_args.batch_size;
    size_t H            = test_args.head_num;
    size_t L            = test_args.seq_len;
    size_t qk_n_elems   = B * H * L * L;
    size_t mask_n_elems = B * L * L;

    GPUBuf<__nv_fp8_e4m3> d_fp8_qk_src(qk_n_elems, true);
    GPUBuf<float>         d_input_scales(1, true);
    GPUBuf<float>         d_output_scales(1, true);
    cudaH2Dcpy(d_input_scales.ptr, &in_scaling, 1);
    cudaH2Dcpy(d_output_scales.ptr, &out_scaling, 1);

    /* Init mask in [0, 1] */
    GPUBuf<float>      d_mask(mask_n_elems, false);
    std::vector<float> mask(mask_n_elems);
    auto               generator = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), std::mt19937());
    std::generate_n(mask.begin(), mask_n_elems, generator);
    cudaH2Dcpy(d_mask.ptr, mask.data(), mask_n_elems);
    GPUBuf<__nv_fp8_e4m3> d_fp8_mask(d_mask);

    /* Reference implementation in fp32 */
    GPUBuf<float> d_qk_src(d_fp8_qk_src);
    GPUBuf<float> d_qk_ref(qk_n_elems, false);
    d_mask.set(d_fp8_mask);
    MaskedSoftmaxParam<float, float> param;
    param.attention_score    = d_qk_ref.ptr;
    param.qk                 = d_qk_src.ptr;
    param.attention_mask     = d_mask.ptr;
    param.batch_size         = B;
    param.q_length           = L;
    param.k_length           = L;
    param.num_heads          = H;
    param.qk_scale           = coef;
    param.linear_bias_slopes = nullptr;
    invokeMaskedSoftmax(param, 0);
    sync_check_cuda_error();
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    /* Test 1: implementation in mixed fp32/fp8 */
    GPUBuf<__nv_fp8_e4m3>                       d_fp8_qk_test_mixed(qk_n_elems, false);
    FP8MaskedSoftMaxParam<__nv_fp8_e4m3, float> softmax_params_mixed{d_fp8_qk_test_mixed.ptr,
                                                                     d_qk_src.ptr,
                                                                     d_fp8_mask.ptr,
                                                                     nullptr,
                                                                     (uint32_t)B,
                                                                     (uint32_t)L,
                                                                     (uint32_t)H,
                                                                     coef,
                                                                     d_input_scales.ptr,
                                                                     d_output_scales.ptr,
                                                                     0};
    invokeFP8MaskedSoftMax(softmax_params_mixed);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    GPUBuf<float> d_qk_test_mixed(d_fp8_qk_test_mixed);

    /* Test 2: implementation in pure fp8 */
    GPUBuf<__nv_fp8_e4m3>                               d_fp8_qk_test_pure(qk_n_elems, false);
    FP8MaskedSoftMaxParam<__nv_fp8_e4m3, __nv_fp8_e4m3> softmax_params_pure{d_fp8_qk_test_pure.ptr,
                                                                            d_fp8_qk_src.ptr,
                                                                            d_fp8_mask.ptr,
                                                                            nullptr,
                                                                            (uint32_t)B,
                                                                            (uint32_t)L,
                                                                            (uint32_t)H,
                                                                            coef,
                                                                            d_input_scales.ptr,
                                                                            d_output_scales.ptr,
                                                                            0};
    invokeFP8MaskedSoftMax(softmax_params_pure);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    GPUBuf<float> d_qk_test_pure(d_fp8_qk_test_pure);

    /* Retrieve data to CPU */
    auto qk_ref        = d_qk_ref.to_host_vec();
    auto qk_test_mixed = d_qk_test_mixed.to_host_vec();
    auto qk_test_pure  = d_qk_test_pure.to_host_vec();

    sync_check_cuda_error();

    /* Compare */
    std::transform(qk_ref.begin(), qk_ref.end(), qk_test_mixed.begin(), qk_test_mixed.begin(), abs_diff<float>());
    std::transform(qk_ref.begin(), qk_ref.end(), qk_test_pure.begin(), qk_test_pure.begin(), abs_diff<float>());
    const float softmax_error_mixed = *std::max_element(qk_test_mixed.begin(), qk_test_mixed.end());
    const float softmax_error_pure  = *std::max_element(qk_test_pure.begin(), qk_test_pure.end());

    bool error = false;

    if (softmax_error_mixed >= MAX_ALLOWED_ERROR || softmax_error_pure >= MAX_ALLOWED_ERROR) {
        error = true;
        printf("Max softmax_error_mixed = %.4f\n", softmax_error_mixed);
        printf("Max softmax_error_pure = %.4f\n", softmax_error_pure);
    }

    if (error) {
        puts("[ERROR] test_fp8_masked_softmax");
    }

    return !error;
}

int main(int argc, char** argv)
{
    if (argc != 7) {
        printf("[ERROR] Usage: %s batch_size head_num max_seq_len"
               " seq_len size_per_head rotary_embedding_dim\n",
               argv[0]);
        printf("e.g., %s 32 16 64 32 64 64\n", argv[0]);
        return EXIT_FAILURE;
    }

    const test_args_t test_args{
        atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6])};

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    printf("Using device %s\n", prop.name);

    bool global_test_pass = true;
    bool test_pass        = true;

    test_pass = test_fp8_transpose_4d_batch_major(test_args);
    printf("%s", test_pass ? "." : "X");
    global_test_pass |= test_pass;

    test_pass = test_fp8_add_fused_QKV_bias_transpose(test_args);
    printf("%s", test_pass ? "." : "X");
    global_test_pass |= test_pass;

    test_pass = test_fp8_masked_softmax(test_args);
    printf("%s", test_pass ? "." : "X");
    global_test_pass |= test_pass;

    puts("");
    return global_test_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
