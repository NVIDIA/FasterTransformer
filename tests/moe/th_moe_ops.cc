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

#include <iostream>
#include <vector>

#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>

#include "src/fastertransformer/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"
#include "src/fastertransformer/kernels/moe_kernels.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"

#include "cutlass/numeric_types.h"

using torch::Tensor;

namespace torch_ext {

std::vector<Tensor> gating_softmax_torch(Tensor input, int64_t k)
{
    const int rank = input.dim();
    int       m    = 1;
    for (int i = 0; i < rank - 1; ++i) {
        m *= input.size(i);
    }

    const int            num_experts = input.size(rank - 1);
    const at::ScalarType _st         = input.scalar_type();
    auto                 stream      = at::cuda::getCurrentCUDAStream().stream();
    CHECK_INPUT(input, _st);

    std::vector<long int> val_shape(rank);
    std::vector<long int> idx_shape(rank);
    for (int i = 0; i < rank - 1; ++i) {
        val_shape[i] = input.size(i);
        idx_shape[i] = input.size(i);
    }

    val_shape[rank - 1] = k;
    idx_shape[rank - 1] = k;

    auto max_values = torch::zeros(val_shape, torch::dtype(input.dtype()).device(torch::kCUDA).requires_grad(false));
    auto indices    = torch::zeros(idx_shape, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    auto rows       = torch::zeros(idx_shape, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    auto softmax_out =
        torch::zeros({m, num_experts}, torch::dtype(input.dtype()).device(torch::kCUDA).requires_grad(false));
    int* idx_ptr  = get_ptr<int>(indices);
    int* rows_ptr = get_ptr<int>(rows);

    switch (_st) {
        case at::ScalarType::Float: {
            const float* input_ptr = get_ptr<float>(input);
            float*       max_vals  = get_ptr<float>(max_values);
            float*       softmax_out_ptr = get_ptr<float>(softmax_out);
            fastertransformer::topk_gating_softmax_kernelLauncher<float>(
                input_ptr, nullptr, max_vals, softmax_out_ptr, idx_ptr, rows_ptr, m, num_experts, k, stream);

            return {max_values, indices, rows};
        }
        case at::ScalarType::Half: {
            const half* input_ptr = get_ptr<half>(input);
            half*       max_vals  = get_ptr<half>(max_values);
            half*       softmax_out_ptr = get_ptr<half>(softmax_out);
            fastertransformer::topk_gating_softmax_kernelLauncher<half>(
                input_ptr, nullptr, max_vals, softmax_out_ptr, idx_ptr, rows_ptr, m, num_experts, k, stream);
            return {max_values, indices, rows};
        }
#ifdef ENABLE_BF16
        case torch::kBFloat16: {
            const __nv_bfloat16* input_ptr = get_ptr<__nv_bfloat16>(input);
            __nv_bfloat16*       max_vals  = get_ptr<__nv_bfloat16>(max_values);
            __nv_bfloat16*       softmax_out_ptr = get_ptr<__nv_bfloat16>(softmax_out);
            fastertransformer::topk_gating_softmax_kernelLauncher<__nv_bfloat16>(
                input_ptr, nullptr, max_vals, softmax_out_ptr, idx_ptr, rows_ptr, m, num_experts, k, stream);
            return {max_values, indices, rows};
        }
#endif
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
}

template<typename T, typename WeightType>
Tensor grouped_gemm_bias_helper(Tensor                            activations,
                                Tensor                            weights,
                                Tensor                            weight_scales,
                                Tensor                            biases,
                                Tensor                            rows_per_expert,
                                fastertransformer::ActivationType activation_type)
{
    const at::ScalarType _st             = activations.scalar_type();
    auto                 stream          = at::cuda::getCurrentCUDAStream().stream();
    const int            num_rows        = activations.size(0);
    const int64_t        gemm_k          = activations.size(1);
    const bool           is_packed_int4s = weights.size(-1) == weight_scales.size(-1) / 2;
    const int64_t        gemm_n          = is_packed_int4s ? 2 * weights.size(-1) : weights.size(-1);
    const int64_t        experts         = weights.size(0);

    auto res = torch::zeros({num_rows, gemm_n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

    auto total_rows_before_expert =
        torch::zeros({experts}, torch::dtype(torch::kInt64).device(torch::kCUDA).requires_grad(false));
    int64_t* total_rows_before_expert_ptr = get_ptr<int64_t>(total_rows_before_expert);

    int* rows_per_expert_ptr = get_ptr<int>(rows_per_expert);

    std::vector<int> rows_per_expert_h(experts);
    cudaError_t      result =
        cudaMemcpy(rows_per_expert_h.data(), rows_per_expert_ptr, sizeof(int) * experts, cudaMemcpyDeviceToHost);
    TORCH_CHECK(result == cudaSuccess, "First memcpy failed");

    std::vector<int64_t> total_rows_before_expert_h(experts);
    for (int expert = 0; expert < experts; ++expert) {
        const int64_t last_row_for_prev_expert = expert == 0 ? 0 : total_rows_before_expert_h[expert - 1];
        total_rows_before_expert_h[expert]     = last_row_for_prev_expert + rows_per_expert_h[expert];
    }
    result = cudaMemcpy(total_rows_before_expert_ptr,
                        total_rows_before_expert_h.data(),
                        sizeof(int64_t) * experts,
                        cudaMemcpyHostToDevice);
    TORCH_CHECK(result == cudaSuccess, "Second memcpy failed");

    T* act_ptr          = get_ptr<T>(activations);
    T* bias_ptr         = get_ptr<T>(biases);
    T* res_ptr          = get_ptr<T>(res);
    T* weight_scale_ptr = get_ptr<T>(weight_scales);

    fastertransformer::MoeGemmRunner<T, WeightType> moe_gemm_runner;
    WeightType*                                     wt_ptr = get_ptr<WeightType>(weights);

    moe_gemm_runner.moe_gemm_bias_act(act_ptr,
                                      wt_ptr,
                                      weight_scale_ptr,
                                      bias_ptr,
                                      res_ptr,
                                      total_rows_before_expert_ptr,
                                      num_rows,
                                      gemm_n,
                                      gemm_k,
                                      experts,
                                      activation_type,
                                      stream);

    return res;
}

Tensor grouped_gemm_bias(Tensor      activations,
                         Tensor      weights,
                         Tensor      weight_scales,
                         Tensor      biases,
                         Tensor      rows_per_expert,
                         std::string activation_type_str)
{

    const at::ScalarType _st = activations.scalar_type();
    CHECK_INPUT(activations, _st);
    CHECK_INPUT(biases, _st);
    CHECK_INPUT(weight_scales, _st);
    CHECK_INPUT(rows_per_expert, torch::kInt32);

    const bool is_packed_int4s = weights.size(-1) == weight_scales.size(-1) / 2;

    fastertransformer::ActivationType activation_type = fastertransformer::ActivationType::InvalidType;
    if (activation_type_str == "identity") {
        activation_type = fastertransformer::ActivationType::Identity;
    }
    else {
        activation_type = fastertransformer::getActivationType(activation_type_str);
    }

    switch (_st) {
        case at::ScalarType::Float: {
            if (weights.scalar_type() == _st) {
                CHECK_INPUT(weights, torch::kFloat32);
                return grouped_gemm_bias_helper<float, float>(
                    activations, weights, weight_scales, biases, rows_per_expert, activation_type);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(weights.scalar_type()));
                TORCH_CHECK(false, err_msg);
            }
            break;
        }
        case at::ScalarType::Half: {
            if (weights.scalar_type() == _st) {
                CHECK_INPUT(weights, torch::kFloat16);
                return grouped_gemm_bias_helper<half, half>(
                    activations, weights, weight_scales, biases, rows_per_expert, activation_type);
            }
            else if (weights.scalar_type() == torch::kInt8 && !is_packed_int4s) {
                CHECK_INPUT(weights, torch::kInt8);
                return grouped_gemm_bias_helper<half, uint8_t>(
                    activations, weights, weight_scales, biases, rows_per_expert, activation_type);
            }
            else if (weights.scalar_type() == torch::kInt8 && is_packed_int4s) {
                CHECK_INPUT(weights, torch::kInt8);
                return grouped_gemm_bias_helper<half, cutlass::uint4b_t>(
                    activations, weights, weight_scales, biases, rows_per_expert, activation_type);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(weights.scalar_type()));
                TORCH_CHECK(false, err_msg);
            }
            break;
        }
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16: {
            if (weights.scalar_type() == _st) {
                CHECK_INPUT(weights, torch::kBFloat16);
                return grouped_gemm_bias_helper<__nv_bfloat16, __nv_bfloat16>(
                    activations, weights, weight_scales, biases, rows_per_expert, activation_type);
            }
            else if (weights.scalar_type() == torch::kInt8 && !is_packed_int4s) {
                CHECK_INPUT(weights, torch::kInt8);
                return grouped_gemm_bias_helper<__nv_bfloat16, uint8_t>(
                    activations, weights, weight_scales, biases, rows_per_expert, activation_type);
            }
            else if (weights.scalar_type() == torch::kInt8 && is_packed_int4s) {
                CHECK_INPUT(weights, torch::kInt8);
                return grouped_gemm_bias_helper<__nv_bfloat16, cutlass::uint4b_t>(
                    activations, weights, weight_scales, biases, rows_per_expert, activation_type);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(weights.scalar_type()));
                TORCH_CHECK(false, err_msg);
            }
            break;
        }
#endif
        default:
            TORCH_CHECK(false, "Incompatible tensor type for grouped gemm bias");
    }
}

template<typename T, typename WeightType>
Tensor run_moe_fc_helper(Tensor                            input_activations,
                         Tensor                            gating_output,
                         Tensor                            fc1_expert_weights,
                         Tensor                            fc1_scales,
                         Tensor                            fc1_expert_biases,
                         fastertransformer::ActivationType fc1_activation_type,
                         Tensor                            fc2_expert_weights,
                         Tensor                            fc2_scales,
                         Tensor                            fc2_expert_biases,
                         Tensor                            skip_layer,
                         Tensor                            finished,
                         const int                         active_rows,
                         const int                         k)
{

    const int num_rows    = input_activations.size(0);
    const int hidden_size = input_activations.size(1);
    const int inter_size  = fc2_expert_weights.size(1);
    const int num_experts = gating_output.size(-1);
    auto      stream      = at::cuda::getCurrentCUDAStream().stream();

    T* input_act_ptr     = get_ptr<T>(input_activations);
    T* gating_output_ptr = get_ptr<T>(gating_output);

    WeightType*           fc1_expert_weights_ptr = get_ptr<WeightType>(fc1_expert_weights);
    static constexpr bool is_fp16_or_fp32 =
        std::is_same<WeightType, float>::value || std::is_same<WeightType, half>::value;
#ifdef ENABLE_BF16
    static constexpr bool ignore_scales = is_fp16_or_fp32 || std::is_same<WeightType, __nv_bfloat16>::value;
#else
    static constexpr bool ignore_scales = is_fp16_or_fp32;
#endif

    T* fc1_scales_ptr        = ignore_scales ? nullptr : get_ptr<T>(fc1_scales);
    T* fc1_expert_biases_ptr = get_ptr<T>(fc1_expert_biases);

    WeightType* fc2_expert_weights_ptr = get_ptr<WeightType>(fc2_expert_weights);
    T*          fc2_scales_ptr         = ignore_scales ? nullptr : get_ptr<T>(fc2_scales);
    T*          fc2_expert_biases_ptr  = get_ptr<T>(fc2_expert_biases);

    T*    skip_layer_ptr = get_ptr<T>(skip_layer);
    bool* finished_ptr   = get_ptr<bool>(finished);

    fastertransformer::CutlassMoeFCRunner<T, WeightType> moe_runner;
    long int bytes        = moe_runner.getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, k);
    auto workspace_tensor = torch::empty({bytes}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
    char* workspace_ptr   = get_ptr<char>(workspace_tensor);

    const at::ScalarType _st = input_activations.scalar_type();
    auto                 fc2_output =
        torch::empty({k * num_rows, hidden_size}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    T* fc2_output_ptr = get_ptr<T>(fc2_output);

    auto expert_scales     = torch::empty({num_rows, k}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    T*   expert_scales_ptr = get_ptr<T>(expert_scales);

    auto expanded_source_row_to_expanded_dest_row =
        torch::empty({num_rows, k}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    int* expanded_source_row_to_expanded_dest_row_ptr = get_ptr<int>(expanded_source_row_to_expanded_dest_row);

    auto expert_for_source_row =
        torch::empty({num_rows, k}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    int* expert_for_source_row_ptr = get_ptr<int>(expert_for_source_row);

    moe_runner.run_moe_fc(input_act_ptr,
                          gating_output_ptr,
                          fc1_expert_weights_ptr,
                          fc1_scales_ptr,
                          fc1_expert_biases_ptr,
                          fc1_activation_type,
                          fc2_expert_weights_ptr,
                          fc2_scales_ptr,
                          num_rows,
                          hidden_size,
                          inter_size,
                          num_experts,
                          k,
                          workspace_ptr,
                          fc2_output_ptr,
                          finished_ptr,
                          active_rows,
                          expert_scales_ptr,
                          expanded_source_row_to_expanded_dest_row_ptr,
                          expert_for_source_row_ptr,
                          stream);

    auto output_tensor =
        torch::empty({num_rows, hidden_size}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    T* output_tensor_ptr = get_ptr<T>(output_tensor);

    fastertransformer::finalize_moe_routing_kernelLauncher(fc2_output_ptr,
                                                           output_tensor_ptr,
                                                           skip_layer_ptr,
                                                           fc2_expert_biases_ptr,
                                                           expert_scales_ptr,
                                                           expanded_source_row_to_expanded_dest_row_ptr,
                                                           expert_for_source_row_ptr,
                                                           num_rows,
                                                           hidden_size,
                                                           k,
                                                           stream);
    return output_tensor;
}

Tensor run_moe_fc(Tensor      input_activations,
                  Tensor      gating_output,
                  Tensor      fc1_expert_weights,
                  Tensor      fc1_scales,
                  Tensor      fc1_expert_biases,
                  std::string fc1_activation_type_str,
                  Tensor      fc2_expert_weights,
                  Tensor      fc2_scales,
                  Tensor      fc2_expert_biases,
                  Tensor      skip_layer,
                  Tensor      finished,
                  int64_t     active_rows,
                  int64_t     k)
{

    const at::ScalarType _st = input_activations.scalar_type();

    const int num_rows    = input_activations.size(0);
    const int hidden_size = input_activations.size(1);
    const int inter_size  = fc2_expert_weights.size(1);
    const int num_experts = gating_output.size(-1);

    // We signal int4 by having the last weight dim be half the size of the scales. This is because int4 elements are
    // packed into a single byte.
    torch::ScalarType quant_type = fc2_expert_weights.scalar_type();
    TORCH_CHECK(fc2_expert_weights.scalar_type() == fc1_expert_weights.scalar_type(),
                "FC1 and FC2 must be quantized to the same type");
    if (fc1_scales.dim() > 0 && fc1_expert_weights.size(-1) == fc1_scales.size(-1) / 2) {
        TORCH_CHECK(fc2_expert_weights.size(-1) == fc2_scales.size(-1) / 2, "FC1 and FC2 must be both be int4.");
        quant_type = at::ScalarType::QUInt4x2;
    }

    CHECK_INPUT(input_activations, _st);
    TORCH_CHECK(input_activations.dim() == 2, "Invalid rank for activations");

    CHECK_INPUT(gating_output, _st);
    TORCH_CHECK(gating_output.dim() == 2, "Invalid rank for gating output");
    TORCH_CHECK(gating_output.size(0) == num_rows, "gating output and activations must have same number of rows");

    CHECK_TH_CUDA(fc1_expert_weights);
    CHECK_CONTIGUOUS(fc1_expert_weights);
    TORCH_CHECK(fc1_expert_weights.dim() == 3, "Invalid rank for fc1 weights");
    TORCH_CHECK(fc1_expert_weights.size(0) == num_experts, "Experts mismatch between gate outputs and fc1 weights");
    TORCH_CHECK(fc1_expert_weights.size(1) == hidden_size,
                "Activation last dim must equal size of dim 1 for fc1 weight");

    const int fc1_num_cols =
        quant_type == at::ScalarType::QUInt4x2 ? 2 * fc1_expert_weights.size(-1) : fc1_expert_weights.size(-1);
    if (_st != torch::kFloat32 && _st != torch::kFloat16) {
        CHECK_INPUT(fc1_scales, _st);
        TORCH_CHECK(fc1_scales.dim() == 2, "Invalid rank for fc1 scales");
        TORCH_CHECK(fc1_scales.size(0) == num_experts, "Experts mismatch between gate outputs and fc1 scales");
        TORCH_CHECK(fc1_scales.size(-1) == fc1_num_cols, "Mismatch between fc1 weights and scale shapes");
        TORCH_CHECK(fc1_scales.size(-1) == fc1_expert_biases.size(-1), "Mismatch between fc1 scale and bias shapes");
    }

    CHECK_INPUT(fc1_expert_biases, _st);
    TORCH_CHECK(fc1_expert_biases.dim() == 2, "Invalid rank for fc1 biases");
    TORCH_CHECK(fc1_expert_biases.size(0) == gating_output.size(-1),
                "Experts mismatch between gate outputs and fc1 biases");

    CHECK_TH_CUDA(fc2_expert_weights);
    CHECK_CONTIGUOUS(fc2_expert_weights);
    TORCH_CHECK(fc2_expert_weights.dim() == 3, "Invalid rank for fc2 weights");
    TORCH_CHECK(fc2_expert_weights.size(0) == gating_output.size(-1),
                "Experts mismatch between gate outputs and fc2 weights");
    TORCH_CHECK(fc2_expert_weights.size(1) == fc1_num_cols, "fc1 weight last dim must equal dim 1 of fc2 weights");

    if (_st != torch::kFloat32 && _st != torch::kFloat16) {
        CHECK_INPUT(fc2_scales, _st);
        TORCH_CHECK(fc2_scales.dim() == 2, "Invalid rank for fc2 scales");
        TORCH_CHECK(fc2_scales.size(0) == gating_output.size(-1),
                    "Experts mismatch between gate outputs and fc2 scales");
        const int fc2_num_cols =
            quant_type == at::ScalarType::QUInt4x2 ? 2 * fc2_expert_weights.size(-1) : fc2_expert_weights.size(-1);
        TORCH_CHECK(fc2_scales.size(-1) == fc2_num_cols, "Mismatch between fc2 weights and scale shapes");
        TORCH_CHECK(fc2_scales.size(-1) == fc2_expert_biases.size(-1), "Mismatch between fc2 scale and bias shapes");
    }

    CHECK_INPUT(fc2_expert_biases, _st);
    TORCH_CHECK(fc2_expert_biases.dim() == 2, "Invalid rank for fc2 biases");
    TORCH_CHECK(fc2_expert_biases.size(0) == num_experts, "Experts mismatch between gate outputs and fc2 biases");

    CHECK_INPUT(skip_layer, _st);
    TORCH_CHECK(skip_layer.sizes() == input_activations.sizes(), "Invalid rank for skip connection");

    CHECK_INPUT(finished, torch::kBool);
    TORCH_CHECK(finished.dim() == 1, "Invalid rank for finished tensor");
    TORCH_CHECK(finished.size(0) == input_activations.size(0),
                "Finished and activations must have same number of rows");

    Tensor output_tensor;

    fastertransformer::ActivationType fc1_activation_type = fastertransformer::ActivationType::InvalidType;
    if (fc1_activation_type_str == "identity") {
        fc1_activation_type = fastertransformer::ActivationType::Identity;
    }
    else {
        fc1_activation_type = fastertransformer::getActivationType(fc1_activation_type_str);
    }

    switch (_st) {
        case at::ScalarType::Float: {

            if (quant_type == _st) {
                output_tensor = run_moe_fc_helper<float, float>(input_activations,
                                                                gating_output,
                                                                fc1_expert_weights,
                                                                fc1_scales,
                                                                fc1_expert_biases,
                                                                fc1_activation_type,
                                                                fc2_expert_weights,
                                                                fc2_scales,
                                                                fc2_expert_biases,
                                                                skip_layer,
                                                                finished,
                                                                active_rows,
                                                                k);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
        case at::ScalarType::Half: {

            if (quant_type == _st) {
                output_tensor = run_moe_fc_helper<half, half>(input_activations,
                                                              gating_output,
                                                              fc1_expert_weights,
                                                              fc1_scales,
                                                              fc1_expert_biases,
                                                              fc1_activation_type,
                                                              fc2_expert_weights,
                                                              fc2_scales,
                                                              fc2_expert_biases,
                                                              skip_layer,
                                                              finished,
                                                              active_rows,
                                                              k);
            }
            else if (quant_type == torch::kInt8) {
                output_tensor = run_moe_fc_helper<half, uint8_t>(input_activations,
                                                                 gating_output,
                                                                 fc1_expert_weights,
                                                                 fc1_scales,
                                                                 fc1_expert_biases,
                                                                 fc1_activation_type,
                                                                 fc2_expert_weights,
                                                                 fc2_scales,
                                                                 fc2_expert_biases,
                                                                 skip_layer,
                                                                 finished,
                                                                 active_rows,
                                                                 k);
            }
            else if (quant_type == at::ScalarType::QUInt4x2) {
                output_tensor = run_moe_fc_helper<half, cutlass::uint4b_t>(input_activations,
                                                                           gating_output,
                                                                           fc1_expert_weights,
                                                                           fc1_scales,
                                                                           fc1_expert_biases,
                                                                           fc1_activation_type,
                                                                           fc2_expert_weights,
                                                                           fc2_scales,
                                                                           fc2_expert_biases,
                                                                           skip_layer,
                                                                           finished,
                                                                           active_rows,
                                                                           k);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16: {
            if (quant_type == _st) {
                output_tensor = run_moe_fc_helper<__nv_bfloat16, __nv_bfloat16>(input_activations,
                                                                                gating_output,
                                                                                fc1_expert_weights,
                                                                                fc1_scales,
                                                                                fc1_expert_biases,
                                                                                fc1_activation_type,
                                                                                fc2_expert_weights,
                                                                                fc2_scales,
                                                                                fc2_expert_biases,
                                                                                skip_layer,
                                                                                finished,
                                                                                active_rows,
                                                                                k);
            }
            else if (quant_type == torch::kInt8) {
                output_tensor = run_moe_fc_helper<__nv_bfloat16, uint8_t>(input_activations,
                                                                          gating_output,
                                                                          fc1_expert_weights,
                                                                          fc1_scales,
                                                                          fc1_expert_biases,
                                                                          fc1_activation_type,
                                                                          fc2_expert_weights,
                                                                          fc2_scales,
                                                                          fc2_expert_biases,
                                                                          skip_layer,
                                                                          finished,
                                                                          active_rows,
                                                                          k);
            }
            else if (quant_type == at::ScalarType::QUInt4x2) {
                output_tensor = run_moe_fc_helper<__nv_bfloat16, cutlass::uint4b_t>(input_activations,
                                                                                    gating_output,
                                                                                    fc1_expert_weights,
                                                                                    fc1_scales,
                                                                                    fc1_expert_biases,
                                                                                    fc1_activation_type,
                                                                                    fc2_expert_weights,
                                                                                    fc2_scales,
                                                                                    fc2_expert_biases,
                                                                                    skip_layer,
                                                                                    finished,
                                                                                    active_rows,
                                                                                    k);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
#endif
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
    return output_tensor;
}

TORCH_LIBRARY(moe_unit_ops, m)
{
    m.def("gating_softmax", gating_softmax_torch);
    m.def("grouped_gemm_bias", grouped_gemm_bias);
    m.def("run_moe_fc", run_moe_fc);
}
}  // namespace torch_ext