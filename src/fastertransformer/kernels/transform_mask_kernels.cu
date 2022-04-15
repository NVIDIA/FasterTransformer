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

#include "reduce_kernel_utils.cuh"
#include "transform_mask_kernels.h"

namespace fastertransformer {

/*******************  invokeTransformMask  ***********************/

// transform mask [B, S, S](half) into [B, S2*S2/64, 64](half), S2 is the actural  seqlen used in fmha row-major
// in one MMA (16*16 elements calculated by a warp), each thread calculates 8 elements
// the offsets of elements calculated by each thread are : for n, +0 +1 +8 +9; for m, +0 +8 (M_XMMAS*N_XMMAS times)
// in transformed_mask, the masks of one warp are stored in 4 continuous rows ([4, 64]), with two elements of one thread
// stored in 2 continuous halfs. one cta calculates warps_m*warps_n mma == 16*warps_m*16*warps_n elements grid(B,
// S2*S2/64) block(32)
__global__ void transform_mask_kernel(half2* tranformed_mask,
                                      const half2* mask,
                                      const uint32_t warps_m,
                                      const uint32_t warps_n,
                                      const uint32_t B,
                                      const uint32_t S,
                                      const uint32_t S2)
{
    const int bi = blockIdx.x;
    const int r = blockIdx.y;

    const int N_per_XMMAS = warps_n << 4;
    const int M_per_XMMAS = warps_m << 4;
    const int N_XMMAS = (S2 + N_per_XMMAS - 1) / (N_per_XMMAS);
    const int warps_in_XMMAS = warps_m * warps_n;
    const half2* mask_b = mask + ((bi * S * S) >> 1);
    half2* tranformed_mask_b = tranformed_mask + (bi * gridDim.y << 5);  //((bi * gridDim.y << 6) >> 1);

    half2 tmp = {half(-30000.0f), half(-30000.0f)};

    int c = threadIdx.x * 2;
    int elt_offset = c % 2;
    int warp_id = r / 4;
    int elt_in_thread = (r % 4) * 2 + elt_offset;
    int noffset_in_warp = (((elt_in_thread & 3) >> 1) << 3) + (elt_in_thread & 1);
    int moffset_in_warp = ((elt_in_thread >> 2) & 1) << 3;

    int XMMAS_mi = warp_id / (N_XMMAS * warps_in_XMMAS);
    int XMMAS_ni = warp_id % (N_XMMAS * warps_in_XMMAS) / warps_in_XMMAS;
    int warp_id_in_XMMAS = warp_id - (XMMAS_mi * N_XMMAS + XMMAS_ni) * warps_in_XMMAS;
    int warp_mi = warp_id_in_XMMAS % warps_m;
    int warp_ni = warp_id_in_XMMAS / warps_m;
    int noffset = XMMAS_ni * N_per_XMMAS + (warp_ni << 4) + noffset_in_warp;
    int moffset = XMMAS_mi * M_per_XMMAS + (warp_mi << 4) + moffset_in_warp;

    int mi = moffset + (c >> 3);
    int ni = noffset + (((c >> 1) & 3) << 1);

    if (mi < S && ni < S) {
        tmp = __ldg(mask_b + ((mi * S + ni) >> 1));
    }

    tranformed_mask_b[(r << 5) + threadIdx.x] = tmp;
}

// transform mask [B, S, S](half) into [B, S2*S2/64, 64](half), S2 is the actural  seqlen used in fmha row-major
// in one MMA (16*16 elements calculated by a warp), each thread calculates 8 elements
// the offsets of elements calculated by each thread are : for n, +0 +1 +8 +9; for m, +0 +8 (M_XMMAS*N_XMMAS times)
// in transformed_mask, the masks of one warp are stored in 4 continuous rows ([4, 64]), with two elements of one thread
// stored in 2 continuous halfs. one cta calculates warps_m*warps_n mma == 16*warps_m*16*warps_n elements grid(B,
// S2*S2/64) block(32)
__global__ void transform_mask_kernel(half* tranformed_mask,
                                      const half* mask,
                                      const uint32_t warps_m,
                                      const uint32_t warps_n,
                                      const uint32_t B,
                                      const uint32_t S,
                                      const uint32_t S2)
{
    const int bi = blockIdx.x;
    const int r = blockIdx.y;

    const int N_per_XMMAS = warps_n << 4;
    const int M_per_XMMAS = warps_m << 4;
    const int N_XMMAS = (S2 + N_per_XMMAS - 1) / (N_per_XMMAS);
    const int warps_in_XMMAS = warps_m * warps_n;
    half2* tranformed_mask_b = (half2*)(tranformed_mask + (bi * gridDim.y << 6));

    half2 tmp = {half(-30000.0f), half(-30000.0f)};

    int c = threadIdx.x * 2;
    int elt_offset = c % 2;
    int warp_id = r / 4;
    int elt_in_thread = (r % 4) * 2 + elt_offset;
    int noffset_in_warp = (((elt_in_thread & 3) >> 1) << 3) + (elt_in_thread & 1);
    int moffset_in_warp = ((elt_in_thread >> 2) & 1) << 3;

    int XMMAS_mi = warp_id / (N_XMMAS * warps_in_XMMAS);
    int XMMAS_ni = warp_id % (N_XMMAS * warps_in_XMMAS) / warps_in_XMMAS;
    int warp_id_in_XMMAS = warp_id - (XMMAS_mi * N_XMMAS + XMMAS_ni) * warps_in_XMMAS;
    int warp_mi = warp_id_in_XMMAS % warps_m;
    int warp_ni = warp_id_in_XMMAS / warps_m;
    int noffset = XMMAS_ni * N_per_XMMAS + (warp_ni << 4) + noffset_in_warp;
    int moffset = XMMAS_mi * M_per_XMMAS + (warp_mi << 4) + moffset_in_warp;

    int mi = moffset + (c >> 3);
    int ni = noffset + (((c >> 1) & 3) << 1);

    if (mi < S) {
        mask += bi * S * S;
        int idx = mi * S + ni;
        if (ni < S) {
            tmp.x = __ldg(mask + idx);
        }
        if (ni + 1 < S) {
            tmp.y = __ldg(mask + idx + 1);
        }
    }

    tranformed_mask_b[(r << 5) + threadIdx.x] = tmp;
}

void invokeTransformMask(
    half* tranformed_mask, const half* mask, const uint32_t B, const uint32_t S, cudaStream_t stream)
{
    uint32_t S2;
    uint32_t warps_m = 2, warps_n = 2;
    if (S <= 64) {
        S2 = 64;
    }
    else if (S <= 128) {
        S2 = 128;
    }
    else if (S <= 256) {
        S2 = 256;
        warps_m = 1;
        warps_n = 4;
    }
    else if (S <= 384) {
        S2 = 384;
        warps_m = 1;
        warps_n = 8;
    }
    else {
        printf("[ERROR][invokeTransformMask]unsupport seq_len %d\n", S);
        exit(-1);
    }
    assert(S2 * S2 % 64 == 0);
    dim3 grid(B, S2 * S2 / 64);
    dim3 block(32);
    if (S % 2 == 0) {
        transform_mask_kernel<<<grid, block, 0, stream>>>(
            (half2*)tranformed_mask, (const half2*)mask, warps_m, warps_n, B, S, S2);
    }
    else {
        transform_mask_kernel<<<grid, block, 0, stream>>>(tranformed_mask, mask, warps_m, warps_n, B, S, S2);
    }
}

}  // namespace fastertransformer
