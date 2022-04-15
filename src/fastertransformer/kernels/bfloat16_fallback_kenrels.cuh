/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include <cuda_fp16.h>

namespace fastertransformer {

#ifdef ENABLE_BF16
inline __device__ float2 bf1622float2(const __nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float2 f_val;
    f_val.x = __low2float(val); 
    f_val.y = __high2float(val);
    return f_val;
#else
    return __bfloat1622float2(val);
#endif
}

inline __device__ __nv_bfloat162 bf162bf162(const __nv_bfloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    __nv_bfloat162 val2;
    val2.x = val;
    val2.y = val;
    return val2;
#else
    return __bfloat162bfloat162(val);
#endif
}

inline __device__ __nv_bfloat162 bf16hadd2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fxl + fyl, fxh + fyh);
#else
    return __hadd2(x, y);
#endif
}

inline __device__ __nv_bfloat16 bf16hadd(const __nv_bfloat16 x, const __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16( __bfloat162float(x) + __bfloat162float(y) );
#else
    return __hadd(x, y);
#endif
}

inline __device__ __nv_bfloat162 bf16hsub2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fxl - fyl, fxh - fyh);
#else
    return __hsub2(x, y);
#endif
}

inline __device__ __nv_bfloat16 bf16hsub(const __nv_bfloat16 x, const __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16( __bfloat162float(x) - __bfloat162float(y) );
#else
    return __hsub(x, y);
#endif
}

inline __device__ __nv_bfloat162 bf16hmul2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fxl * fyl, fxh * fyh);
#else
    return __hmul2(x, y);
#endif
}

inline __device__ __nv_bfloat16 bf16hmul(const __nv_bfloat16 x, const __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16( __bfloat162float(x) * __bfloat162float(y) );
#else 
    return __hmul(x, y);
#endif
}

inline __device__ __nv_bfloat162 bf16hfma2(const __nv_bfloat162 x, const __nv_bfloat162 y, const __nv_bfloat162 z) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh, fyl, fyh, fzl, fzh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    fzl = __low2float(z);
    fzh = __high2float(z);
    return __floats2bfloat162_rn(fxl * fyl + fzl, fxh * fyh + fzh);
#else
    return __hfma2(x, y, z);
#endif
}

inline __device__ __nv_bfloat162 bf16exp2(const __nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh;
    fxl = __low2float(x);
    fxh = __high2float(x);;
    return __floats2bfloat162_rn(expf(fxl), expf(fxh));
#else
    return h2exp(x);
#endif
}
#endif // ENABLE_BF16

template<typename T>
inline __device__ T ldg(const T* val) {
    return __ldg(val);
}

#ifdef ENABLE_BF16
template<>
inline __device__ __nv_bfloat162 ldg(const __nv_bfloat162* val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return val[0];
#else
    return __ldg(val);
#endif
}

template<>
inline __device__ __nv_bfloat16 ldg(const __nv_bfloat16* val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return val[0];
#else
    return __ldg(val);
#endif
}
#endif // ENABLE_BF16

// Get type2 from type or vice versa (applied to half and bfloat16)
template<typename T>
struct TypeConverter {using Type = half2;}; // keep for generality

template<>
struct TypeConverter<half2> {using Type = half;};

template<>
struct TypeConverter<half> {using Type = half2;};

#ifdef ENABLE_BF16
template<>
struct TypeConverter<__nv_bfloat162> {using Type = __nv_bfloat16;};

template<>
struct TypeConverter<__nv_bfloat16> {using Type = __nv_bfloat162;};
#endif // ENABLE_BF16

// Convert float to type2 (applied to half2 and bfloat162)
template<typename T>
inline __device__ T float2type2(float a);

template<>
inline __device__ half2 float2type2(float a) {
    return __float2half2_rn(a);
}

#ifdef ENABLE_BF16
template<>
inline __device__ __nv_bfloat162 float2type2(float a) {
    return __float2bfloat162_rn(a);
}
#endif // ENABLE_BF16

// Convert float to type (applied to half and bfloat16)
template<typename T>
inline __device__ T float2type(float a);

template<>
inline __device__ half float2type(float a) {
    return __float2half_rn(a);
}

#ifdef ENABLE_BF16
template<>
inline __device__ __nv_bfloat16 float2type(float a) {
    return __float2bfloat16_rn(a);
}
#endif // ENABLE_BF16

// Convert type to type2 (applied to half and bfloat16)
template<typename T_IN, typename T_OUT>
inline __device__ T_OUT type2type2(T_IN a);

template<>
inline __device__ half2 type2type2(half a) {
    return __half2half2(a);
}

#ifdef ENABLE_BF16
template<>
inline __device__ __nv_bfloat162 type2type2(__nv_bfloat16 a) {
    return bf162bf162(a);
}
#endif // ENABLE_BF16

// Defined math operations (bfloat16 fallback to fp32 when it is not supported)
template<typename T>
inline __device__ T hadd2(T a, T b) {
    return __hadd2(a, b);
}

#ifdef ENABLE_BF16
template<>
inline __device__ __nv_bfloat162 hadd2(__nv_bfloat162 a, __nv_bfloat162 b) {
    return bf16hadd2(a, b);
}
#endif // ENABLE_BF16

template<typename T>
inline __device__ T hsub2(T a, T b) {
    return __hsub2(a, b);
}

#ifdef ENABLE_BF16
template<>
inline __device__ __nv_bfloat162 hsub2(__nv_bfloat162 a, __nv_bfloat162 b) {
    return bf16hsub2(a, b);
}
#endif // ENABLE_BF16

template<typename T>
inline __device__ T hmul2(T a, T b) {
    return __hmul2(a, b);
}

#ifdef ENABLE_BF16
template<>
inline __device__ __nv_bfloat162 hmul2(__nv_bfloat162 a, __nv_bfloat162 b) {
    return bf16hmul2(a, b);
}
#endif // ENABLE_BF16

template<typename T>
inline __device__ T hexp2(T a) {
    return h2exp(a);
}

#ifdef ENABLE_BF16
template<>
inline __device__ __nv_bfloat162 hexp2(__nv_bfloat162 a) {
    return bf16exp2(a);
}
#endif // ENABLE_BF16

}  // namespace fastertransformer