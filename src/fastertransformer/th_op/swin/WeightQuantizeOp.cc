/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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

namespace torch_ext {
using torch::Tensor;
using namespace fastertransformer;

Tensor swin_weight_quantize(Tensor weight, Tensor quant_max)
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
    TORCH_CHECK(weight.dtype() == torch::kFloat16, "weight dtype should be float16");
    TORCH_CHECK(weight.numel() != 0, "weight should not be empty tensor");
    TORCH_CHECK(weight.dim() == 2, "Invalid rank. The rank of weight should be 2");

    int k = weight.size(0);
    int n = weight.size(1);

    CHECK_TH_CUDA(quant_max);
    TORCH_CHECK(quant_max.dtype() == torch::kFloat32, "quant_max dtype should be float32");
    TORCH_CHECK(quant_max.numel() == 1, "quant_max wrong shape");

    const half* weight_ = get_ptr<half>(weight);
    const float* quant_max_ = get_ptr<float>(quant_max);

    auto output = torch::empty({k * n}, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false));
    int8_t* transform_out = get_ptr<int8_t>(output);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    int format = use_ORDER_COL32_2R_4R4 ? 1 : 2;
    const int scale_is_vector = 0;
    invokeQuantizeWeight(transform_out, weight_, quant_max_, n, k, format, stream, scale_is_vector);
    return output;
}

}  // namespace torch_ext

static auto weight_quantize =
    torch::RegisterOperators("fastertransformer::swin_weight_quantize", &torch_ext::swin_weight_quantize);
