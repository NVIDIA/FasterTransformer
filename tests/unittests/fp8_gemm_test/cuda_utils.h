#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#pragma once

#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include <cstdint>
#include <cstdlib>
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <iostream>

template<typename T_OUT, typename T_IN>
void cudaCast(T_OUT* out, T_IN const* const in, const uint32_t size);

#ifdef ENABLE_FP8
template<typename T, int QUANTIZE_MODE>
void invokeQuatizeVectorE4M3(__nv_fp8_e4m3* output,
                             float const* input_qua_amax_ptr,
                             T const* input,
                             uint32_t size,
                             uint32_t n,
                             cudaStream_t stream);
#endif

// template<typename T, int QUANTIZE_MODE>
// void invokeDequatizeVectorE4M3(
//     T* output, float const* qua_amax_ptr, __nv_fp8_e4m3 const* input, uint32_t size, uint32_t n, cudaStream_t
//     stream);

template<typename T_OUT, typename T_IN, int QUANTIZE_MODE>
void invokeDequatizeVectorE4M3(
    T_OUT* output, float const* qua_amax_ptr, T_IN const* input, uint32_t size, uint32_t n, cudaStream_t stream);

#endif  // CUDA_UTILS_H