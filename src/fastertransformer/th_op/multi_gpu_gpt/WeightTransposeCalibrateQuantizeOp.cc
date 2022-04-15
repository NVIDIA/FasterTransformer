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

#include "src/fastertransformer/th_op/multi_gpu_gpt/WeightTransposeCalibrateQuantizeOp.h"
#include "src/fastertransformer/utils/convert_data_type.h"

template<typename T>
void ldnCalibrateWeightPerChannel(float* scale_out, const T* weight, const int k, const int n)
{
    for (int n_i = 0; n_i < n; n_i++) {
        float amax = 0.0f;
        for (int k_i = 0; k_i < k; k_i++) {
            float val = fabs(weight[k_i * n + n_i]);
            if (amax < val) {
                amax = val;
            }
        }
        scale_out[n_i] = amax / 127.0f;
    }
}

template<typename T>
void ldnTransposeQuantizeWeightPerChannel(int8_t* output, const float* scale, const T* weight, const int k, const int n)
{
    for (int n_i = 0; n_i < n; n_i++) {
        float scale_val = scale[n_i];
        for (int k_i = 0; k_i < k; k_i++) {
            float val = weight[k_i * n + n_i];
            output[n_i * k + k_i] = float_to_int8_rn_host(val / scale_val);
        }
    }
}

namespace torch_ext {
using torch::Tensor;

std::vector<Tensor> weight_transpose_calibrate_quantize(Tensor weight)
{

    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight dtype should be float32");
    TORCH_CHECK(weight.numel() != 0, "weight should not be empty tensor");
    TORCH_CHECK(weight.dim() == 2, "Invalid rank. The rank of weight should be 2");

    int k = weight.size(0);
    int n = weight.size(1);

    const float* weight_ = get_ptr<float>(weight);

    if (weight.device() == torch::kCUDA) {
        auto int8_weight = torch::empty({k * n}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
        int8_t* int8_weight_out = get_ptr<int8_t>(int8_weight);
        auto scale = torch::empty({n}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
        float* scale_out = get_ptr<float>(scale);

        auto stream = at::cuda::getCurrentCUDAStream().stream();

        fastertransformer::invokeLdnCalibrateWeightPerChannel(scale_out, weight_, k, n, stream);
        fastertransformer::invokeLdnTransposeQuantizeWeightPerChannel(
            int8_weight_out, scale_out, weight_, k, n, stream);

        return std::vector<Tensor>{int8_weight, scale};
    }
    else {
        auto int8_weight = torch::empty({k * n}, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));
        int8_t* int8_weight_out = get_ptr<int8_t>(int8_weight);
        auto scale = torch::empty({n}, torch::dtype(torch::kFloat32).device(torch::kCPU).requires_grad(false));
        float* scale_out = get_ptr<float>(scale);

        ldnCalibrateWeightPerChannel(scale_out, weight_, k, n);
        ldnTransposeQuantizeWeightPerChannel(int8_weight_out, scale_out, weight_, k, n);

        return std::vector<Tensor>{int8_weight, scale};
    }
}

}  // namespace torch_ext
