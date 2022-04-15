/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/kernels/quantize_weight.h"
#include "src/fastertransformer/th_op/th_utils.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

#ifdef SPARSITY_ENABLED
namespace {
void compressInt8Matrix(void* output, const void* input, const int m, const int k, cudaStream_t stream)
{
    cusparseLtHandle_t _cusparseLtHandle;
    CHECK_CUSPARSE(cusparseLtInit(&_cusparseLtHandle));
    cusparseLtMatDescriptor_t matA;
    unsigned alignment = 16;
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
        &_cusparseLtHandle, &matA, m, k, k, alignment, CUDA_R_8I, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT))
    CHECK_CUSPARSE(cusparseLtSpMMACompress2(
        &_cusparseLtHandle, &matA, true, CUSPARSE_OPERATION_NON_TRANSPOSE, input, output, stream))
    cusparseLtDestroy(&_cusparseLtHandle);
}
}  // namespace
#endif

namespace torch_ext {
using torch::Tensor;

Tensor weight_quantize(Tensor weight, Tensor quant_max, bool sparse)
{
    bool use_ORDER_COL32_2R_4R4 = false;
#if (CUDART_VERSION >= 11000)
    int device{-1};
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    if (props.major * 10 + props.minor >= 80) {
        use_ORDER_COL32_2R_4R4 = true;
    }
#endif

    CHECK_TH_CUDA(weight);
    CHECK_TH_CUDA(weight);
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight dtype should be float32");
    TORCH_CHECK(weight.numel() != 0, "weight should not be empty tensor");
    TORCH_CHECK(weight.dim() == 2, "Invalid rank. The rank of weight should be 2");

    int k = weight.size(0);
    int n = weight.size(1);

    CHECK_TH_CUDA(quant_max);
    CHECK_TH_CUDA(quant_max);
    TORCH_CHECK(quant_max.dtype() == torch::kFloat32, "quant_max dtype should be float32");
    TORCH_CHECK(quant_max.numel() == n, "quant_max wrong shape");

    const float* weight_ = get_ptr<float>(weight);
    const float* quant_max_ = get_ptr<float>(quant_max);

    auto output = torch::empty({k * n}, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false));
    int8_t* transform_out = get_ptr<int8_t>(output);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

#ifdef SPARSITY_ENABLED
    if (sparse) {
        int format = 0;
        auto tmp = torch::empty({k * n}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
        int8_t* tmp_out = get_ptr<int8_t>(tmp);
        fastertransformer::invokeQuantizeWeight(tmp_out, weight_, quant_max_, n, k, format, stream);
        compressInt8Matrix(transform_out, tmp_out, n, k, stream);
        return output;
    }
#endif
    int format = use_ORDER_COL32_2R_4R4 ? 1 : 2;
    fastertransformer::invokeQuantizeWeight(transform_out, weight_, quant_max_, n, k, format, stream);
    return output;
}

}  // namespace torch_ext

static auto weight_quantize =
    torch::RegisterOperators("fastertransformer::weight_quantize", &torch_ext::weight_quantize);
