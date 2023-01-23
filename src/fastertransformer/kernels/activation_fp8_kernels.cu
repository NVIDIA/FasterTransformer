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

#include "src/fastertransformer/kernels/activation_fp8_kernels.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_fallbacks.cuh"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

__inline__ __device__ float tanh_opt(float x)
{
    return tanh(x);
#if (__CUDA_ARCH__ >= 750)
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
#else
    const float exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#endif
}

template<typename T>
__inline__ __device__ T gelu(T x)
{
    float f_x = (float)(x);
    float cdf = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (f_x + 0.044715f * f_x * f_x * f_x))));
    return (T)(f_x * cdf);
}

template<>
__inline__ __device__ __nv_bfloat162 gelu(__nv_bfloat162 val)
{
    __nv_bfloat162 val_pow3 = __hmul2(val, __hmul2(val, val));
    float2         tmp_pow  = cuda_cast<float2>(val_pow3);
    float2         tmp      = cuda_cast<float2>(val);

    tmp.x = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
    tmp.y = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
    return __hmul2(val, __float22bfloat162_rn(tmp));
}

template<typename T1, typename T2>
__global__ void FP8AddBiasGelu(FP8ActivationParam<T1, T2> param)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < param.m * param.n; id += blockDim.x * gridDim.x) {
        T2 val = (T2)((float)(param.out[id]) * __ldg(param.input_scale));
        // TODO(bhsueh) check do we need it
        // if (param.input_scale_2 != nullptr) {
        //     val = (float)val * __ldg(param.input_scale_2 + (id % param.n)) / __ldg(param.input_scale_2_min);
        // }
        if (param.bias != nullptr) {
            T2 reg_bias = __ldg(&param.bias[id % param.n]);
            val         = val + reg_bias;
        }
        param.out[id] = (T1)((float)gelu(val) * __ldg(param.output_scale));
    }
}

// template<>
// __global__ void FP8AddBiasGelu(FP8ActivationParam<__nv_fp8_e4m3, __nv_bfloat16> param)
// {
//     float input_scale = __ldg(param.input_scale);
//     float output_scale = __ldg(param.output_scale);
//     __nv_fp8x4_e4m3* out_ptr = (__nv_fp8x4_e4m3*)param.out;
//     __nv_bfloat162* bias_ptr = (__nv_bfloat162*)param.bias;

//     __nv_bfloat162 input_scale_2 = __floats2bfloat162_rn(input_scale, input_scale);
//     __nv_bfloat162 output_scale_2 = __floats2bfloat162_rn(output_scale, output_scale);

//     for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < param.m * (param.n / 4); id += blockDim.x * gridDim.x)
//     {
//         __nv_bfloat162 val_1;
//         __nv_bfloat162 val_2;

//         fp8x4_e4m3_to_bfloat2(&val_1, &val_2, &out_ptr[id]);

//         __nv_bfloat162 bias_val_1 = bias_ptr[(id * 2) % (param.n / 2)];
//         __nv_bfloat162 bias_val_2 = bias_ptr[(id * 2 + 1) % (param.n / 2)];

//         val_1 = hmul2(val_1, input_scale_2);
//         val_2 = hmul2(val_2, input_scale_2);
//         val_1 = gelu(hadd2(val_1, bias_val_1));
//         val_2 = gelu(hadd2(val_2, bias_val_2));
//         val_1 = hmul2(val_1, output_scale_2);
//         val_2 = hmul2(val_2, output_scale_2);

//         out_ptr[id] = __nv_fp8x4_e4m3(val_1, val_2);
//     }
// }

// NOTE: input is bfloat (pack8 to have 128 bit input, and 64 bit output)
template<>
__global__ void FP8AddBiasGelu(FP8ActivationParam<__nv_fp8_e4m3, __nv_bfloat16> param)
{
    float         input_scale     = __ldg(param.input_scale);
    float         output_scale    = __ldg(param.output_scale);
    constexpr int Packed_Elements = 8;
    using packed_out              = __nv_fp8_8_e4m3;
    using packed_bias             = __nv_bfloat168;
    using packed_in               = __nv_bfloat168;
    packed_out*  out_ptr          = (packed_out*)param.out;
    packed_in*   in_ptr           = (packed_in*)param.in;
    packed_bias* bias_ptr         = (packed_bias*)param.bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < param.m * (param.n / Packed_Elements);
         id += blockDim.x * gridDim.x) {
        packed_in   val  = in_ptr[id];
        packed_bias bias = bias_ptr[id % (param.n / Packed_Elements)];
        packed_out  val_output;
#pragma unroll
        for (int i = 0; i < Packed_Elements; i++) {
            val_output.array[i] = __nv_fp8_e4m3(gelu((float)val.array[i] + (float)bias.array[i]) * output_scale);
        }
        // f_val.x = gelu(f_val.x + f_bias.x) * output_scale;
        // f_val.y = gelu(f_val.y + f_bias.x) * output_scale;
        // val = hmul2(gelu(hadd2(val, bias)), output_scale_2);
        // val = hmul2(gelu(hadd2(val, bias)), output_scale_2);

        out_ptr[id] = val_output;
    }
}

template<typename T1, typename T2>
void invokeFP8AddBiasGelu(FP8ActivationParam<T1, T2> param)
{
    FT_CHECK(param.n % 8 == 0);
    // NOTE: input data type is bfloat now
    const int data_type_factor = 8;  // pack 8 elements for bfloat16 input
    const int total_packs      = (param.n * param.m) / data_type_factor;
    dim3      block(256);
    int       num_packs_per_thread = 1;
    while ((total_packs + (block.x * num_packs_per_thread) - 1) / (block.x * num_packs_per_thread) > 512) {
        num_packs_per_thread *= 2;
    }
    dim3 grid((total_packs + (block.x * num_packs_per_thread) - 1) / (block.x * num_packs_per_thread));
    FP8AddBiasGelu<<<grid, block, 0, param.stream>>>(param);
}

template void
invokeFP8AddBiasGelu<__nv_fp8_e4m3, __nv_bfloat16>(FP8ActivationParam<__nv_fp8_e4m3, __nv_bfloat16> param);

template<typename T1, typename T2>
__global__ void FP8AddBiasRelu(FP8ActivationParam<T1, T2> param)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < param.m * param.n; id += blockDim.x * gridDim.x) {
        T2 val = (T2)((float)(param.out[id]) * __ldg(param.input_scale));
        // TODO(bhsueh) check do we need it
        // if (param.input_scale_2 != nullptr) {
        //     val = (float)val * __ldg(param.input_scale_2 + (id % param.n)) / __ldg(param.input_scale_2_min);
        // }
        if (param.bias != nullptr) {
            T2 reg_bias = __ldg(&param.bias[id % param.n]);
            val         = val + reg_bias;
        }
        val           = (T2)(((float)val) * __ldg(param.output_scale));
        param.out[id] = (T1)((val > (T2)0.0f) ? val : (T2)0.0f);
    }
}

// template<>
// __global__ void FP8AddBiasRelu(FP8ActivationParam<__nv_fp8_e4m3, __nv_bfloat16> param)
// {
//     float input_scale = __ldg(param.input_scale);
//     float output_scale = __ldg(param.output_scale);
//     __nv_fp8x4_e4m3* out_ptr = (__nv_fp8x4_e4m3*)param.out;
//     __nv_bfloat162* bias_ptr = (__nv_bfloat162*)param.bias;

//     __nv_bfloat162 input_scale_2 = __floats2bfloat162_rn(input_scale, input_scale);
//     __nv_bfloat162 output_scale_2 = __floats2bfloat162_rn(output_scale, output_scale);

//     for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < param.m * (param.n / 4); id += blockDim.x * gridDim.x)
//     {
//         __nv_bfloat162 val_1;
//         __nv_bfloat162 val_2;

//         fp8x4_e4m3_to_bfloat2(&val_1, &val_2, &out_ptr[id]);

//         __nv_bfloat162 bias_val_1 = bias_ptr[(id * 2) % (param.n / 2)];
//         __nv_bfloat162 bias_val_2 = bias_ptr[(id * 2 + 1) % (param.n / 2)];

//         val_1 = hmul2(val_1, input_scale_2);
//         val_2 = hmul2(val_2, input_scale_2);
//         val_1 = hadd2(val_1, bias_val_1);
//         val_2 = hadd2(val_2, bias_val_2);
//         val_1.x = val_1.x > (__nv_bfloat16)0.0f ? val_1.x : (__nv_bfloat16)0.0f;
//         val_1.y = val_1.y > (__nv_bfloat16)0.0f ? val_1.y : (__nv_bfloat16)0.0f;
//         val_2.x = val_2.x > (__nv_bfloat16)0.0f ? val_2.x : (__nv_bfloat16)0.0f;
//         val_2.y = val_2.y > (__nv_bfloat16)0.0f ? val_2.y : (__nv_bfloat16)0.0f;
//         val_1 = hmul2(val_1, output_scale_2);
//         val_2 = hmul2(val_2, output_scale_2);

//         out_ptr[id] = __nv_fp8x4_e4m3(val_1, val_2);
//     }
// }

// NOTE: input is bfloat
template<>
__global__ void FP8AddBiasRelu(FP8ActivationParam<__nv_fp8_e4m3, __nv_bfloat16> param)
{
    float            input_scale  = __ldg(param.input_scale);
    float            output_scale = __ldg(param.output_scale);
    __nv_fp8x2_e4m3* out_ptr      = (__nv_fp8x2_e4m3*)param.out;
    __nv_bfloat162*  in_ptr       = (__nv_bfloat162*)param.in;
    __nv_bfloat162*  bias_ptr     = (__nv_bfloat162*)param.bias;

    __nv_bfloat162 input_scale_2  = __floats2bfloat162_rn(input_scale, input_scale);
    __nv_bfloat162 output_scale_2 = __floats2bfloat162_rn(output_scale, output_scale);

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < param.m * (param.n / 2); id += blockDim.x * gridDim.x) {
        __nv_bfloat162 val  = in_ptr[id];
        __nv_bfloat162 bias = bias_ptr[id % (param.n / 2)];
        val                 = hadd2(val, bias);
        val.x               = val.x > (__nv_bfloat16)0.0f ? val.x : (__nv_bfloat16)0.0f;
        val.y               = val.y > (__nv_bfloat16)0.0f ? val.y : (__nv_bfloat16)0.0f;
        val                 = hmul2(val, output_scale_2);

        out_ptr[id] = __nv_fp8x2_e4m3(val);
    }
}

template<typename T1, typename T2>
void invokeFP8AddBiasRelu(FP8ActivationParam<T1, T2> param)
{
    FT_CHECK(param.n % 2 == 0);
    // NOTE: input data type is bfloat now
    const int data_type_factor = 2;  // 1 for fp32, 2 for fp16/bf16, 4 for fp86
    dim3      block, grid;
    if (param.n / 4 / data_type_factor <= 1024) {
        block.x = param.n / 4 / data_type_factor;
        grid.x  = param.m;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(param.m * param.n / 1024.);
    }
    FP8AddBiasRelu<<<grid, block, 0, param.stream>>>(param);
}

template void
invokeFP8AddBiasRelu<__nv_fp8_e4m3, __nv_bfloat16>(FP8ActivationParam<__nv_fp8_e4m3, __nv_bfloat16> param);

// template<typename T1, typename T2>
// void invokeFP8AddBias(FP8ActivationParam<T1, T2> param) {
//     // TODO FP8
// }

// template void
// invokeFP8AddBias<__nv_fp8_e4m3, half>(FP8ActivationParam<__nv_fp8_e4m3, half> param);

}  // namespace fastertransformer
