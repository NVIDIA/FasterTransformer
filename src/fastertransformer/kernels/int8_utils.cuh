/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

static inline __device__ int8_t float_to_int8_rn(float x)
{
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;"
               : "=r"(dst)
               : "f"(x));
  return reinterpret_cast<const int8_t &>(dst);
}
