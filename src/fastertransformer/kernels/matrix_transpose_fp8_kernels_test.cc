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

#include "src/fastertransformer/kernels/matrix_transpose_kernels.h"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/test_utils.h"
#include <algorithm>
#include <cstdio>
#include <cuda_profiler_api.h>
#include <functional>
#include <numeric>
#include <sys/time.h>

using namespace fastertransformer;

int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("[ERROR] Usage: bin/matrix_transpose_fp8_kernels_test m n\n");
        printf("e.g., bin/matrix_transpose_fp8_kernels_test 2048 512\n");
        return 0;
    }

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    printf("Using device %s\n", prop.name);

    int dim_m = atoi(argv[1]);
    int dim_n = atoi(argv[2]);

    float* fp32_input;
    float* fp32_output_ref;
    float* fp32_output_test;

    __nv_fp8_e4m3* fp8_input;
    __nv_fp8_e4m3* fp8_output;

    // Allocate random fp8 input
    deviceMalloc(&fp8_input, dim_m * dim_n, true);
    deviceMalloc(&fp8_output, dim_m * dim_n, false);

    deviceMalloc(&fp32_input, dim_n * dim_m, false);
    deviceMalloc(&fp32_output_ref, dim_n * dim_m, false);
    deviceMalloc(&fp32_output_test, dim_n * dim_m, false);

    /********** Correctness verification **********/

    // Reference
    invokeCudaD2Dcpyfp82Float(fp32_input, fp8_input, dim_m * dim_n, 0);
    invokeMatrixTranspose(fp32_output_ref, fp32_input, dim_m, dim_n, 0);

    // Test
    invokeMatrixTranspose(fp8_output, fp8_input, dim_m, dim_n, 0);
    invokeCudaD2Dcpyfp82Float(fp32_output_test, fp8_output, dim_m * dim_n, 0);
    sync_check_cuda_error();

    std::vector<float> d_fp32_test(dim_m * dim_n);
    std::vector<float> d_fp32_ref(dim_m * dim_n);
    cudaD2Hcpy(d_fp32_ref.data(), fp32_output_ref, dim_m * dim_n);
    cudaD2Hcpy(d_fp32_test.data(), fp32_output_test, dim_m * dim_n);

    std::transform(d_fp32_ref.begin(), d_fp32_ref.end(), d_fp32_test.begin(), d_fp32_ref.begin(), abs_diff<float>());
    float global_error = std::accumulate(d_fp32_ref.begin(), d_fp32_ref.end(), 0.0f);

    printf("global_error = %.2f\n", global_error);

    /********** Performance verification **********/
    const int    loop_n = 1000;
    float        elapsed;
    cudaStream_t stream;
    cudaEvent_t  start, end;
    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, stream);
    for (int i = 0; i < loop_n; i++) {
        invokeMatrixTranspose(fp32_output_ref, fp32_input, dim_m, dim_n, stream);
    }
    cudaEventRecord(end, stream);

    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, start, end);

    printf("Timing reference: %.2fms\n", elapsed);

    cudaEventRecord(start, stream);
    for (int i = 0; i < loop_n; i++) {
        invokeCudaD2Dcpyfp82Float(fp32_output_test, fp8_output, dim_m * dim_n, stream);
    }
    cudaEventRecord(end, stream);

    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, start, end);
    printf("Timing test: %.2fms\n", elapsed);

    cudaFree(fp8_input);
    cudaFree(fp8_output);
    cudaFree(fp32_input);
    cudaFree(fp32_output_ref);
    cudaFree(fp32_output_test);

    return EXIT_SUCCESS;
}
