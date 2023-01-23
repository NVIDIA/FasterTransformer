/*
 * Copyright 2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef __CUDA_FP8_H__
#define __CUDA_FP8_H__

/* Set up function decorations */
#if defined(__CUDACC__)
#define __CUDA_FP8_DECL__ static __device__ __inline__
#define __CUDA_HOSTDEVICE_FP8__ __host__ __device__
#define __CUDA_HOSTDEVICE_FP8_DECL__ static __host__ __device__ __inline__
#else /* !defined(__CUDACC__) */
#if defined(__GNUC__)
#define __CUDA_HOSTDEVICE_FP8_DECL__ static __attribute__((unused))
#else
#define __CUDA_HOSTDEVICE_FP8_DECL__ static
#endif /* defined(__GNUC__) */
#define __CUDA_HOSTDEVICE_FP8__
#endif /* defined(__CUDACC_) */

#if !defined(_MSC_VER) && __cplusplus >= 201103L
#define __CPP_VERSION_AT_LEAST_11_FP8
#elif _MSC_FULL_VER >= 190024210 && _MSVC_LANG >= 201103L
#define __CPP_VERSION_AT_LEAST_11_FP8
#endif

#if (defined __CPP_VERSION_AT_LEAST_11_FP8)
/* bring fixed width integer types */
#include <cstdint>
#else
#include <stdint.h>
#endif

/* bring in __half_raw data type */
#include "cuda_fp16.h"
/* bring in __nv_bfloat16_raw data type */
#include "cuda_bf16.h"
/* bring in float2, double4, etc vector types */
#include "vector_types.h"

/**
 * \defgroup CUDA_MATH_INTRINSIC_FP8 FP8 Intrinsics
 * This section describes fp8 intrinsic functions.
 * To use these functions, include the header file \p cuda_fp8.h in your
 * program.
 */

/**
 * \defgroup CUDA_MATH_FP8_MISC FP8 Conversion and Data Movement
 * \ingroup CUDA_MATH_INTRINSIC_FP8
 * To use these functions, include the header file \p cuda_fp8.h in your
 * program.
 */

/**
 * \ingroup CUDA_MATH_FP8_MISC
 * \brief __nv_fp8_storage_t 8-bit unsigned integer
 * type abstraction used to for \p fp8 floating-point
 * numbers storage.
 */
typedef uint8_t __nv_fp8_storage_t;

/**
 * \ingroup CUDA_MATH_FP8_MISC
 * \brief __nv_fp8x2_storage_t 16-bit unsigned integer
 * type abstraction used to for storage of pairs of
 * \p fp8 floating-point numbers.
 */
typedef uint16_t __nv_fp8x2_storage_t;

/**
 * \ingroup CUDA_MATH_FP8_MISC
 * \brief __nv_fp8x4_storage_t 32-bit unsigned integer
 * type abstraction used to for storage of tetrads of
 * \p fp8 floating-point numbers.
 */
typedef uint32_t __nv_fp8x4_storage_t;

/**
 * \ingroup CUDA_MATH_FP8_MISC
 * \brief __nv_saturation_t enumerates the modes applicable when
 * performing a narrowing conversion to \p fp8 destination types.
 *
 * \details
 * - __NV_NOSAT - means no saturation to finite is performed when
 * rounding values outside the range of destination type.
 * NOTE: for fp8 type of e4m3 kind, the results that are larger than
 * the maximum representable finite number become NaN.
 * - __NV_SATFINITE - means inputs larger than the maximum representable
 * finite number MAXNORM of the target format round to the MAXNORM of
 * the same sign.
 */
typedef enum __nv_saturation_t {
    __NV_NOSAT,
    __NV_SATFINITE,
} __nv_saturation_t;

/**
 * \ingroup CUDA_MATH_FP8_MISC
 * \brief __nv_fp8_interpretation_t enumerates the possible
 * interpretations of the 8-bit values when referring to them as
 * \p fp8 types.
 *
 * \details
 *  - __NV_E4M3 stands for \p fp8 numbers of \p e4m3 kind.
 *  - __NV_E5M2 stands for \p fp8 numbers of \p e5m2 kind.
 */
typedef enum __nv_fp8_interpretation_t {
    __NV_E4M3,
    __NV_E5M2,
} __nv_fp8_interpretation_t;

/* Forward-declaration of C-style APIs */

/**
 * \ingroup CUDA_MATH_FP8_MISC
 * \brief Converts input \p double precision \p x to \p fp8 type of the
 * requested kind using round-to-nearest-even rounding and requested saturation
 * mode.
 *
 * \details Converts input \p x to \p fp8 type of the kind specified by
 * \p fp8_interpretation parameter,
 * using round-to-nearest-even rounding and
 * saturation mode specified by \p saturate parameter.
 *
 * \returns
 * - The \p __nv_fp8_storage_t value holds the result of conversion.
 */
__CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t
__nv_cvt_double_to_fp8(const double x, const __nv_saturation_t saturate,
                       const __nv_fp8_interpretation_t fp8_interpretation);
__CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8x2_storage_t
__nv_cvt_double2_to_fp8x2(const double2 x, const __nv_saturation_t saturate,
                          const __nv_fp8_interpretation_t fp8_interpretation);

/**
 * \ingroup CUDA_MATH_FP8_MISC
 * \brief Converts input \p single precision \p x to \p fp8 type of the
 * requested kind using round-to-nearest-even rounding and requested saturation
 * mode.
 *
 * \details Converts input \p x to \p fp8 type of the kind specified by
 * \p fp8_interpretation parameter,
 * using round-to-nearest-even rounding and
 * saturation mode specified by \p saturate parameter.
 *
 * \returns
 * - The \p __nv_fp8_storage_t value holds the result of conversion.
 */
__CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t
__nv_cvt_float_to_fp8(const float x, const __nv_saturation_t saturate,
                      const __nv_fp8_interpretation_t fp8_interpretation);
__CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8x2_storage_t
__nv_cvt_float2_to_fp8x2(const float2 x, const __nv_saturation_t saturate,
                         const __nv_fp8_interpretation_t fp8_interpretation);

/**
 * \ingroup CUDA_MATH_FP8_MISC
 * \brief Converts input \p half precision \p x to \p fp8 type of the requested
 * kind using round-to-nearest-even rounding and requested saturation mode.
 *
 * \details Converts input \p x to \p fp8 type of the kind specified by
 * \p fp8_interpretation parameter,
 * using round-to-nearest-even rounding and
 * saturation mode specified by \p saturate parameter.
 *
 * \returns
 * - The \p __nv_fp8_storage_t value holds the result of conversion.
 */
__CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t
__nv_cvt_halfraw_to_fp8(const __half_raw x, const __nv_saturation_t saturate,
                        const __nv_fp8_interpretation_t fp8_interpretation);
__CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8x2_storage_t __nv_cvt_halfraw2_to_fp8x2(
    const __half2_raw x, const __nv_saturation_t saturate,
    const __nv_fp8_interpretation_t fp8_interpretation);

/**
 * \ingroup CUDA_MATH_FP8_MISC
 * \brief Converts input \p nv_bfloat16 precision \p x to \p fp8 type of the
 * requested kind using round-to-nearest-even rounding and requested saturation
 * mode.
 *
 * \details Converts input \p x to \p fp8 type of the kind specified by
 * \p fp8_interpretation parameter,
 * using round-to-nearest-even rounding and
 * saturation mode specified by \p saturate parameter.
 *
 * \returns
 * - The \p __nv_fp8_storage_t value holds the result of conversion.
 */
__CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t __nv_cvt_bfloat16raw_to_fp8(
    const __nv_bfloat16_raw x, const __nv_saturation_t saturate,
    const __nv_fp8_interpretation_t fp8_interpretation);
__CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8x2_storage_t
__nv_cvt_bfloat16raw2_to_fp8x2(
    const __nv_bfloat162_raw x, const __nv_saturation_t saturate,
    const __nv_fp8_interpretation_t fp8_interpretation);

/**
 * \ingroup CUDA_MATH_FP8_MISC
 * \brief Converts input \p fp8 \p x of the specified kind
 * to \p half precision
 *
 * \details Converts input \p x of \p fp8 type of the kind specified by
 * \p fp8_interpretation parameter
 * to \p half precision.
 *
 * \returns
 * - The \p __half_raw value holds the result of conversion.
 */
__CUDA_HOSTDEVICE_FP8_DECL__ __half_raw
__nv_cvt_fp8_to_halfraw(const __nv_fp8_storage_t x,
                        const __nv_fp8_interpretation_t fp8_interpretation);
__CUDA_HOSTDEVICE_FP8_DECL__ __half2_raw
__nv_cvt_fp8x2_to_halfraw2(const __nv_fp8x2_storage_t x,
                           const __nv_fp8_interpretation_t fp8_interpretation);

#if defined(__cplusplus)

#define __CUDA_FP8_TYPES_EXIST__

/* Forward-declaration of structures defined in "cuda_fp8.hpp" */

/**
 * \brief __nv_fp8_e5m2 datatype
 *
 * \details This structure implements the datatype for storing
 * \p fp8 floating-point numbers of \p e5m2 kind:
 * with 1 sign, 5 exponent, 1 implicit and 2 explicit mantissa bits.
 *
 * The structure implements converting constructors and operators.
 */
struct __nv_fp8_e5m2;

/**
 * \brief __nv_fp8x2_e5m2 datatype
 *
 * \details This structure implements the datatype for storage
 * and operations on the vector of two \p fp8 values of \p e5m2 kind.
 */
struct __nv_fp8x2_e5m2;

/**
 * \brief __nv_fp8x4_e5m2 datatype
 *
 * \details This structure implements the datatype for storage
 * and operations on the vector of four \p fp8 values of \p e5m2 kind.
 */
struct __nv_fp8x4_e5m2;

/**
 * \brief __nv_fp8_e4m3 datatype
 *
 * \details This structure implements the datatype for storing
 * \p fp8 floating-point numbers of \p e4m3 kind:
 * with 1 sign, 4 exponent, 1 implicit and 3 explicit mantissa bits.
 * The encoding doesn't support Infinity.
 * NaNs are limited to 0x7F and 0xFF values.
 *
 * The structure implements converting constructors and operators.
 */
struct __nv_fp8_e4m3;

/**
 * \brief __nv_fp8x2_e4m3 datatype
 *
 * \details This structure implements the datatype for storage
 * and operations on the vector of two \p fp8 values of \p e4m3 kind.
 */
struct __nv_fp8x2_e4m3;

/**
 * \brief __nv_fp8x4_e4m3 datatype
 *
 * \details This structure implements the datatype for storage
 * and operations on the vector of four \p fp8 values of \p e4m3 kind.
 */
struct __nv_fp8x4_e4m3;

#endif /* defined(__cplusplus) */

#include "cuda_fp8.hpp"

#undef __CUDA_FP8_DECL__
#undef __CUDA_HOSTDEVICE_FP8__
#undef __CUDA_HOSTDEVICE_FP8_DECL__

#if defined(__CPP_VERSION_AT_LEAST_11_FP8)
#undef __CPP_VERSION_AT_LEAST_11_FP8
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP8) */

#endif /* end of include guard: __CUDA_FP8_H__ */
