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

#include "src/fastertransformer/kernels/layernorm_fp8_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/utils/gpu_buf.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/test_utils.h"
#include <algorithm>
#include <cstdio>
#include <cuda_profiler_api.h>
#include <sys/time.h>

using namespace fastertransformer;

template<typename T>
int checkNonZero(T* A, int size)
{
    T* h_A = (T*)malloc(sizeof(T) * size);
    cudaMemcpy(h_A, A, sizeof(T) * size, cudaMemcpyDeviceToHost);
    int noneZeroNum = 0;
    for (int ii = 0; ii < size; ii++) {
        if (fabs(float(h_A[ii]) - 0.0f) > 0.0001f) {
            noneZeroNum += 1;
        }
    }
    free(h_A);
    return noneZeroNum;
}

template<typename TA, typename TB>
void checkMat(TA* A, TB* B, int size, char* mark)
{
    float max_diff = -10000.0f;
    float max_diff_a, max_diff_b;
    TA*   matA       = (TA*)malloc(sizeof(TA) * size);
    TB*   matB       = (TB*)malloc(sizeof(TB) * size);
    int   not_passed = 0;
    cudaMemcpy(matA, A, sizeof(TA) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(matB, B, sizeof(TB) * size, cudaMemcpyDeviceToHost);
    float A_nonZero_ratio = float(checkNonZero(A, size)) / float(size);
    float B_nonZero_ratio = float(checkNonZero(B, size)) / float(size);
    if (A_nonZero_ratio < 0.1 || B_nonZero_ratio < 0.1) {
        printf("[%s] nonZero ratio [%f] [%f]\n", mark, A_nonZero_ratio, B_nonZero_ratio);
    }
    printf("[INFO] A  B  diff rel_diff\n");
    for (int jjj = 0; jjj < size; jjj++) {
        float diff = fabs(float(matA[jjj]) - float(matB[jjj]));
        if (diff > max_diff) {
            max_diff   = diff;
            max_diff_a = float(matA[jjj]);
            max_diff_b = float(matB[jjj]);
        }
        if (fabs(float(matA[jjj]) - float(matB[jjj])) > 0.001) {
            not_passed += 1;
            if (not_passed < 1000)
                printf("%d %f %f %f (%f\%)\n",
                       jjj,
                       float(matA[jjj]),
                       float(matB[jjj]),
                       float(matA[jjj]) - float(matB[jjj]),
                       (float(matA[jjj]) - float(matB[jjj])) / (float(matA[jjj] + 1e-6f)));
        }
    }
    printf("[%s] max diff : %f ; a : %f ; b : %f\n", mark, max_diff, max_diff_a, max_diff_b);
    if (not_passed != 0)
        printf("[%s] different elements : %d \n", mark, not_passed);
    else
        printf("[%s] check pass!\n", mark);
    free(matA);
    free(matB);
}

template<typename T, int quantize_mode>
void layernorm_fp8_test(const int m, const int n);

bool test_fp8_general_add_bias_residual_pre_layernorm(int m, int n)
{
    const size_t n_elems = m * n;
    bool         error   = false;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    GPUBuf<__nv_fp8_e4m3> d_fp8_in(n_elems, true);
    GPUBuf<__nv_fp8_e4m3> d_fp8_residual(n_elems, true);
    GPUBuf<__nv_fp8_e4m3> d_fp8_bias(n, true);
    GPUBuf<__nv_fp8_e4m3> d_fp8_gamma(n, true);
    GPUBuf<__nv_fp8_e4m3> d_fp8_beta(n, true);
    GPUBuf<__nv_fp8_e4m3> d_fp8_out(n_elems, false);

    /* Reference implementation in FP32 */
    GPUBuf<float> d_fp32_in(d_fp8_in);
    GPUBuf<float> d_fp32_residual(d_fp8_residual);
    GPUBuf<float> d_fp32_bias(d_fp8_bias);
    GPUBuf<float> d_fp32_gamma(d_fp8_gamma);
    GPUBuf<float> d_fp32_beta(d_fp8_beta);
    GPUBuf<float> d_fp32_out(n_elems, false);

    invokeGeneralAddBiasResidualPreLayerNorm(d_fp32_out.ptr,
                                             d_fp32_out.ptr,
                                             d_fp32_in.ptr,
                                             d_fp32_residual.ptr,
                                             d_fp32_gamma.ptr,
                                             d_fp32_beta.ptr,
                                             d_fp32_bias.ptr,
                                             1e-6f,
                                             m,
                                             n,
                                             (float*)nullptr,
                                             (float*)nullptr,
                                             (float*)nullptr,
                                             (float*)nullptr,
                                             0,
                                             stream,
                                             0);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    auto residual_ref = d_fp32_residual.to_host_vec();
    auto out_ref      = d_fp32_out.to_host_vec();
    sync_check_cuda_error();

    /* Test implementation in FP32 */
    GPUBuf<float>                                            d_fp32_residual_fp32_fp32(d_fp8_residual);
    float                                                    quant = 0.0f, dequant = 0.0f;
    GeneralFP8AddBiasResidualPreLayerNormParam<float, float> kernel_args_fp32_fp32{d_fp32_out.ptr,
                                                                                   d_fp32_in.ptr,
                                                                                   d_fp32_residual_fp32_fp32.ptr,
                                                                                   d_fp32_bias.ptr,
                                                                                   d_fp32_gamma.ptr,
                                                                                   d_fp32_beta.ptr,
                                                                                   nullptr,  // &quant,
                                                                                   nullptr,  // &dequant,
                                                                                   m,
                                                                                   n,
                                                                                   0,
                                                                                   false};
    invokeGeneralFP8AddBiasResidualPreLayerNorm(kernel_args_fp32_fp32);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    auto residual_test_fp32_fp32 = d_fp32_residual_fp32_fp32.to_host_vec();
    auto out_test_fp32_fp32      = d_fp32_out.to_host_vec();

    /* Test implementation in FP32/FP8 */
    GPUBuf<float>                                                    d_fp32_residual_fp32_fp8(d_fp8_residual);
    GeneralFP8AddBiasResidualPreLayerNormParam<__nv_fp8_e4m3, float> kernel_args_fp32_fp8{d_fp8_out.ptr,
                                                                                          d_fp32_in.ptr,
                                                                                          d_fp32_residual_fp32_fp8.ptr,
                                                                                          d_fp32_bias.ptr,
                                                                                          d_fp32_gamma.ptr,
                                                                                          d_fp32_beta.ptr,
                                                                                          nullptr,  // &quant,
                                                                                          nullptr,  // &dequant,
                                                                                          m,
                                                                                          n,
                                                                                          0,
                                                                                          false};
    invokeGeneralFP8AddBiasResidualPreLayerNorm(kernel_args_fp32_fp8);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    auto residual_test_fp32_fp8 = d_fp32_residual_fp32_fp8.to_host_vec();
    auto out_test_fp32_fp8      = (GPUBuf<float>(d_fp8_out)).to_host_vec();

    /* Test implementation in FP16/FP16 */
    GPUBuf<half>                                           d_fp16_residual_fp16_fp16(d_fp8_residual);
    GPUBuf<half>                                           d_fp16_out(d_fp8_out);
    GeneralFP8AddBiasResidualPreLayerNormParam<half, half> kernel_args_fp16_fp16{d_fp16_out.ptr,
                                                                                 (GPUBuf<half>(d_fp8_in)).ptr,
                                                                                 d_fp16_residual_fp16_fp16.ptr,
                                                                                 (GPUBuf<half>(d_fp8_bias)).ptr,
                                                                                 (GPUBuf<half>(d_fp8_gamma)).ptr,
                                                                                 (GPUBuf<half>(d_fp8_beta)).ptr,
                                                                                 nullptr,  // &quant,
                                                                                 nullptr,  // &dequant,
                                                                                 m,
                                                                                 n,
                                                                                 0,
                                                                                 false};
    invokeGeneralFP8AddBiasResidualPreLayerNorm(kernel_args_fp16_fp16);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    auto residual_test_fp16_fp16 = d_fp16_residual_fp16_fp16.to_host_vec();
    auto out_test_fp16_fp16      = (GPUBuf<float>(d_fp16_out)).to_host_vec();

    /* Comparisons */
    sync_check_cuda_error();

    /* FP32_FP32 */
    std::transform(
        out_ref.begin(), out_ref.end(), out_test_fp32_fp32.begin(), out_test_fp32_fp32.begin(), abs_diff<float>());
    std::transform(residual_ref.begin(),
                   residual_ref.end(),
                   residual_test_fp32_fp32.begin(),
                   residual_test_fp32_fp32.begin(),
                   abs_diff<float>());

    const float error_out_fp32_fp32 = *std::max_element(out_test_fp32_fp32.begin(), out_test_fp32_fp32.end());
    const float error_res_fp32_fp32 = *std::max_element(residual_test_fp32_fp32.begin(), residual_test_fp32_fp32.end());

    /* FP32_FP8 */
    std::transform(
        out_ref.begin(), out_ref.end(), out_test_fp16_fp16.begin(), out_test_fp16_fp16.begin(), abs_diff<float>());
    std::transform(residual_ref.begin(),
                   residual_ref.end(),
                   residual_test_fp16_fp16.begin(),
                   residual_test_fp16_fp16.begin(),
                   abs_diff<float>());

    const float error_out_fp16_fp16 = *std::max_element(out_test_fp16_fp16.begin(), out_test_fp16_fp16.end());
    const float error_res_fp16_fp16 = *std::max_element(residual_test_fp16_fp16.begin(), residual_test_fp16_fp16.end());

    /* FP32_FP8 */
    std::transform(
        out_ref.begin(), out_ref.end(), out_test_fp32_fp8.begin(), out_test_fp32_fp8.begin(), abs_diff<float>());
    std::transform(residual_ref.begin(),
                   residual_ref.end(),
                   residual_test_fp32_fp8.begin(),
                   residual_test_fp32_fp8.begin(),
                   abs_diff<float>());

    const float error_out_fp32_fp8 = *std::max_element(out_test_fp32_fp8.begin(), out_test_fp32_fp8.end());
    const float error_res_fp32_fp8 = *std::max_element(residual_test_fp32_fp8.begin(), residual_test_fp32_fp8.end());

    printf("[FP32/FP32] max error out %f\n", error_out_fp32_fp32);
    printf("[FP32/FP32] max error res %f\n", error_res_fp32_fp32);

    printf("[FP16/FP16] max error out %f\n", error_out_fp16_fp16);
    printf("[FP16/FP16] max error res %f\n", error_res_fp16_fp16);

    printf("[FP32/FP8] max error out %f\n", error_out_fp32_fp8);
    printf("[FP32/FP8] max error res %f\n", error_res_fp32_fp8);

    /* Profiling */
    printf("[FP32/FP32] ");
    TIMEIT(true,
           10,
           stream,
           invokeGeneralAddBiasResidualPreLayerNorm,
           d_fp32_out.ptr,
           d_fp32_out.ptr,
           d_fp32_in.ptr,
           d_fp32_residual.ptr,
           d_fp32_gamma.ptr,
           d_fp32_beta.ptr,
           d_fp32_bias.ptr,
           1e-6f,
           m,
           n,
           (float*)nullptr,
           (float*)nullptr,
           (float*)nullptr,
           (float*)nullptr,
           0,
           stream,
           0);

    kernel_args_fp32_fp32.stream = stream;
    printf("[FP32/FP32] ");
    TIMEIT(true, 10, stream, invokeGeneralFP8AddBiasResidualPreLayerNorm, kernel_args_fp32_fp32);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    kernel_args_fp32_fp8.stream = stream;
    printf("[FP32/FP8] ");
    TIMEIT(true, 10, stream, invokeGeneralFP8AddBiasResidualPreLayerNorm, kernel_args_fp32_fp8);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    kernel_args_fp16_fp16.stream = stream;
    printf("[FP16/FP16] ");
    TIMEIT(true, 10, stream, invokeGeneralFP8AddBiasResidualPreLayerNorm, kernel_args_fp16_fp16);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    return !error;
}

int main(int argc, char** argv)
{
    if (argc != 5) {
        printf("[ERROR] layernorm_fp8_test max_m max_n is_fp16 quantize_mode\n");
        printf("e.g., ./bin/layernorm_fp8_test 1 1024 1 0\n");
        return 0;
    }

    int       max_m         = atoi(argv[1]);
    int       max_n         = atoi(argv[2]);
    int       is_fp16       = atoi(argv[3]);
    const int quantize_mode = atoi(argv[4]);

    if (is_fp16) {
        if (quantize_mode == 0) {
            layernorm_fp8_test<half, 0>(max_m, max_n);
        }
        else if (quantize_mode == 1) {
            layernorm_fp8_test<half, 1>(max_m, max_n);
        }
    }
    else {
        if (quantize_mode == 0) {
            layernorm_fp8_test<float, 0>(max_m, max_n);
        }
        else if (quantize_mode == 1) {
            layernorm_fp8_test<float, 1>(max_m, max_n);
        }
    }

    bool global_test_pass = true;
    bool test_pass        = true;

    test_pass = test_fp8_general_add_bias_residual_pre_layernorm(max_m, max_n);
    printf("%s", test_pass ? "." : "X");
    global_test_pass |= test_pass;

    puts("");
    return global_test_pass ? EXIT_SUCCESS : EXIT_FAILURE;

    // Use to compare the performance
    // for (int m = 1; m <= max_m; m *= 2) {
    //     for (int n = 128; n <= max_n; n *= 2) {
    //         if (is_fp16) {
    //             layernorm_fp8_test<half>(m, n);
    //         }
    //         else {
    //             layernorm_fp8_test<float>(m, n);
    //         }
    //     }
    // }
    // return 0;
}

template<typename T, int quantize_mode>
void layernorm_fp8_test(const int m, const int n)
{
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    printf("Device %s\n", prop.name);

    // baseline buffer
    T* input_baseline;
    T* output_baseline;
    T* gamma;
    T* beta;
    deviceMalloc(&input_baseline, m * n);
    deviceMalloc(&output_baseline, m * n);
    deviceMalloc(&gamma, n);
    deviceMalloc(&beta, n);

    // fp8 buffer
    // __nv_fp8_e4m3* fp8_input;
    __nv_bfloat16* bf16_input;
    __nv_fp8_e4m3* fp8_output;
    float*         fp8_input_deq_ptr;
    float*         fp8_output_qua_ptr;
    T*             fp8_fp32_output;
    // cudaMalloc(&fp8_input, sizeof(__nv_fp8_e4m3) * m * n);
    cudaMalloc(&bf16_input, sizeof(__nv_bfloat16) * m * n);
    cudaMalloc(&fp8_output, sizeof(__nv_fp8_e4m3) * m * n);
    deviceMalloc(&fp8_input_deq_ptr, n);
    deviceMalloc(&fp8_output_qua_ptr, n);
    deviceMalloc(&fp8_fp32_output, m * n);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    const int ite = 5000;

    invokeGeneralLayerNorm<T>(output_baseline, input_baseline, gamma, beta, 1e-6, m, n, (float*)nullptr, 0, stream);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    T* h_input_baseline  = new T[m * n];
    T* h_output_baseline = new T[m * n];
    cudaD2Hcpy(h_input_baseline, input_baseline, m * n);
    cudaD2Hcpy(h_output_baseline, output_baseline, m * n);

    __nv_bfloat16* h_bf16_input = new __nv_bfloat16[m * n];
    for (int i = 0; i < m * n; i++) {
        h_bf16_input[i] = (__nv_bfloat16)(h_input_baseline[i]);
    }
    cudaH2Dcpy(bf16_input, h_bf16_input, m * n);
    delete[] h_bf16_input;

    // 0: per channel quantization
    // 1: per tensor quantization

#define USING_FP8

#ifdef USING_FP8
    if (quantize_mode == 0) {
        float* h_input_amax  = new float[n];
        float* h_output_amax = new float[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0) {
                    h_input_amax[j]  = h_input_baseline[i * n + j];
                    h_output_amax[j] = h_output_baseline[i * n + j];
                }
                else {
                    h_input_amax[j]  = std::max(h_input_amax[j], (float)h_input_baseline[i * n + j]);
                    h_output_amax[j] = std::max(h_output_amax[j], (float)h_output_baseline[i * n + j]);
                }
            }
        }
        cudaH2Dcpy(fp8_input_deq_ptr, h_input_amax, n);
        cudaH2Dcpy(fp8_output_qua_ptr, h_output_amax, n);
        delete h_input_amax;
        delete h_output_amax;
    }
    else {
        float h_input_amax  = *std::max_element(h_input_baseline, h_input_baseline + m * n);
        float h_output_amax = *std::max_element(h_output_baseline, h_output_baseline + m * n);
        printf("[INFO] h_input_amax: %f \n", h_input_amax);
        printf("[INFO] h_output_amax: %f \n", h_output_amax);
        cudaH2Dcpy(fp8_input_deq_ptr, &h_input_amax, 1);
        cudaH2Dcpy(fp8_output_qua_ptr, &h_output_amax, 1);
    }

    // invokeQuatizeVectorE4M3<T, quantize_mode>(fp8_input, fp8_input_deq_ptr, input_baseline, m * n, n, stream);
    // FP8LayerNormParam<__nv_fp8_e4m3, __nv_bfloat16> param{
    //     fp8_output, bf16_input, gamma, beta, fp8_input_deq_ptr, fp8_output_qua_ptr, m, n, stream, true};
    // invokeFP8LayerNorm<__nv_fp8_e4m3, __nv_bfloat16, quantize_mode>(param);
    // invokeDequatizeVectorE4M3<T, quantize_mode>(fp8_fp32_output, fp8_output_qua_ptr, fp8_output, m * n, n, stream);
#else
    float h_input_amax  = 1.0f;
    float h_output_amax = 1.0f;
    cudaH2Dcpy(fp8_input_deq_ptr, &h_input_amax, 1);
    cudaH2Dcpy(fp8_output_qua_ptr, &h_output_amax, 1);
    FP8LayerNormParam<T, T> param{
        fp8_fp32_output, input_baseline, gamma, beta, fp8_input_deq_ptr, fp8_output_qua_ptr, m, n, stream, false};
    invokeFP8LayerNorm<T, T, quantize_mode>(param);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif

    cudaDeviceSynchronize();
    checkMat(fp8_fp32_output, output_baseline, m * n, "fp8_fp32_output v.s. output_baseline");

    // // warmup
    for (int i = 0; i < 1000; i++) {
        invokeGeneralLayerNorm<T>(output_baseline, input_baseline, gamma, beta, 1e-6, m, n, (float*)nullptr, 0, stream);
    }
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    struct timeval start, end;
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);
    for (int i = 0; i < ite; i++) {
        invokeGeneralLayerNorm<T>(output_baseline, input_baseline, gamma, beta, 1e-6, m, n, (float*)nullptr, 0, stream);
    }
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    gettimeofday(&end, NULL);
    float baseline_time = ((end.tv_sec - start.tv_sec) * 1000000. + (end.tv_usec - start.tv_usec) * 1.) / ite;

    struct timeval start_2, end_2;
    cudaDeviceSynchronize();
    gettimeofday(&start_2, NULL);
    for (int i = 0; i < ite; i++) {
#ifdef USING_FP8
        // invokeFP8LayerNorm<__nv_fp8_e4m3, T, quantize_mode>(param);
#else
        // invokeFP8LayerNorm<T, T, quantize_mode>(param);
#endif
    }
    cudaDeviceSynchronize();
    gettimeofday(&end_2, NULL);
    float fp8_time = ((end_2.tv_sec - start_2.tv_sec) * 1000000. + (end_2.tv_usec - start_2.tv_usec) * 1.) / ite;

    printf("[INFO] baseline time: %f us\n", baseline_time);
    printf("[INFO] fp8 time: %f us\n", fp8_time);
    printf("[INFO] m %d, n %d, speedup: %f \n", m, n, baseline_time / fp8_time);
    return;
}
