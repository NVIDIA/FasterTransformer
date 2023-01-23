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

#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include "src/fastertransformer/utils/gpu_buf.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/test_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

using namespace fastertransformer;

typedef struct {
    int batch_size;
    int head_num;
    int max_seq_len;
    int size_per_head;
    int rotary_dimension;
} test_args_t;

template<typename T>
void set_params_struct(Masked_multihead_attention_params<T>& params,
                       T*                                    out,
                       const T*                              q,
                       const T*                              q_bias,
                       const T*                              k,
                       const T*                              k_bias,
                       const T*                              v,
                       const T*                              v_bias,
                       T*                                    k_cache,
                       T*                                    v_cache,
                       const int*                            cache_indir,
                       int                                   stride,
                       int                                   batch_size,
                       int                                   beam_width,
                       int                                   seq_length,
                       int                                   num_heads,
                       int                                   hidden_size_per_head,
                       int                                   rotary_embedding_dim,
                       int                                   timestep,
                       float                                 inv_sqrt_dh,
                       const int*                            input_lengths,
                       int                                   max_input_len,
                       const T*                              relative_attention_bias,
                       int                                   relative_attention_bias_stride)
{
    params.out                            = out;
    params.q                              = q;
    params.q_bias                         = q_bias;
    params.k                              = k;
    params.k_bias                         = k_bias;
    params.v                              = v;
    params.v_bias                         = v_bias;
    params.k_cache                        = k_cache;
    params.v_cache                        = v_cache;
    params.cache_indir                    = cache_indir;
    params.stride                         = stride;
    params.batch_size                     = batch_size;
    params.beam_width                     = beam_width;
    params.memory_max_len                 = seq_length;
    params.num_heads                      = num_heads;
    params.hidden_size_per_head           = hidden_size_per_head;
    params.rotary_embedding_dim           = rotary_embedding_dim;
    params.timestep                       = timestep;
    params.inv_sqrt_dh                    = inv_sqrt_dh;
    params.prefix_prompt_lengths          = input_lengths;
    params.max_input_length               = max_input_len;
    params.relative_attention_bias        = relative_attention_bias;
    params.relative_attention_bias_stride = relative_attention_bias_stride;

    params.finished                 = nullptr;
    params.memory_length_per_sample = nullptr;
    params.length_per_sample        = nullptr;
}

template<typename T>
GPUBuf<T> reshape_key_cache(const GPUBuf<T>& key_cache, int BS, int H, int Dh, int L, int x_orig, int x_targ)
{
    auto           h_key_cache = key_cache.to_host_vec();
    std::vector<T> h_key_cache_r(h_key_cache.size());

    for (int b = 0; b < BS; b++) {
        for (int h = 0; h < H; h++) {
            for (int d = 0; d < Dh; d++) {
                for (int l = 0; l < L; l++) {

                    int in_d = d / x_orig, out_d = d / x_targ;
                    int in_x = d % x_orig, out_x = d % x_targ;

                    int in_offset  = (((b * H + h) * (Dh / x_orig) + in_d) * L + l) * x_orig + in_x;
                    int out_offset = (((b * H + h) * (Dh / x_targ) + out_d) * L + l) * x_targ + out_x;

                    h_key_cache_r[out_offset] = h_key_cache[in_offset];
                }
            }
        }
    }
    GPUBuf<T> key_cache_T(key_cache.size);
    key_cache_T.set(h_key_cache_r.data());

    return key_cache_T;
}

template<typename T>
struct string_rep_t {
    static const std::string value;
};
template<>
const std::string string_rep_t<half>::value{"FP16"};
template<>
const std::string string_rep_t<__nv_bfloat16>::value{"BF16"};
template<>
const std::string string_rep_t<__nv_fp8_e4m3>::value{"FP8"};

template<typename T>
struct mha_type_t {
    using Type = T;
};
template<>
struct mha_type_t<half> {
    using Type = uint16_t;
};

template<typename T>
bool test_masked_multihead_attention(const test_args_t& test_args)
{
    using Tmha                    = typename mha_type_t<T>::Type;
    const float max_allowed_error = 0.05f;

    int BS = test_args.batch_size;
    int H  = test_args.head_num;
    int L  = test_args.max_seq_len;
    int Dh = test_args.size_per_head;
    int R  = test_args.rotary_dimension;

    GPUBuf<T> q_T(BS * Dh * H), q_bias_T(Dh * H);
    GPUBuf<T> k_T(BS * Dh * H), k_bias_T(Dh * H);
    GPUBuf<T> v_T(BS * Dh * H), v_bias_T(Dh * H);
    GPUBuf<T> kcache_T(BS * Dh * H * L);  // read as [BS, H, Dh/x, L, x]
    GPUBuf<T> vcache_T(BS * Dh * H * L);
    GPUBuf<T> out_T(BS * Dh * H);

    GPUBuf<int> seq_lengths(BS);
    seq_lengths.set((std::vector<int>(BS, L / 4)).data());

    GPUBuf<float> q_fp32(q_T), q_bias_fp32(q_bias_T);
    GPUBuf<float> k_fp32(k_T), k_bias_fp32(k_bias_T);
    GPUBuf<float> v_fp32(v_T), v_bias_fp32(v_bias_T);
    GPUBuf<float> kcache_fp32(reshape_key_cache(kcache_T, BS, H, Dh, L, 16 / sizeof(T), 16 / sizeof(float)));
    GPUBuf<float> vcache_fp32(vcache_T);
    GPUBuf<float> out_fp32(BS * Dh * H);

    Masked_multihead_attention_params<float> params_fp32;
    set_params_struct(params_fp32,
                      out_fp32.ptr,
                      q_fp32.ptr,
                      q_bias_fp32.ptr,
                      k_fp32.ptr,
                      k_bias_fp32.ptr,
                      v_fp32.ptr,
                      v_bias_fp32.ptr,
                      kcache_fp32.ptr,
                      vcache_fp32.ptr,
                      nullptr,
                      0,
                      BS,
                      1,
                      L,
                      H,
                      Dh,
                      R,
                      L - 1,
                      1.0 / sqrtf(Dh),
                      seq_lengths.ptr,
                      L / 4,
                      (const float*)nullptr,
                      0);
    masked_multihead_attention(params_fp32, 0);

    auto mha_ref = out_fp32.to_host_vec();

    Masked_multihead_attention_params<Tmha> params_T;
    set_params_struct(params_T,
                      (Tmha*)out_T.ptr,
                      (Tmha*)q_T.ptr,
                      (Tmha*)q_bias_T.ptr,
                      (Tmha*)k_T.ptr,
                      (Tmha*)k_bias_T.ptr,
                      (Tmha*)v_T.ptr,
                      (Tmha*)v_bias_T.ptr,
                      (Tmha*)kcache_T.ptr,
                      (Tmha*)vcache_T.ptr,
                      nullptr,
                      0,
                      BS,
                      1,
                      L,
                      H,
                      Dh,
                      R,
                      L - 1,
                      1.0 / sqrtf(Dh),
                      seq_lengths.ptr,
                      L / 4,
                      (const Tmha*)nullptr,
                      0);
    masked_multihead_attention(params_T, 0);

    auto mha_T_test = GPUBuf<float>(out_T).to_host_vec();

    /* for (int bs = 0; bs < BS; bs++) { */
    /*     for (int h = 0; h < H; h++) { */
    /*         for (int d = 0; d < Dh; d++) { */
    /*             float ref = mha_ref[bs * H * Dh + h * Dh + d]; */
    /*             float test = mha_T_test[bs * H * Dh + h * Dh + d]; */
    /*             const float diff = abs_diff<float>()(ref, test); */
    /*             const float rel_diff = rel_abs_diff<float>()(ref, test); */

    /*             printf("[%d, %d, %d] Error %.2e (=%.2f%%)\n", bs, h, d, diff, rel_diff * 100); */
    /*         } */
    /*     } */
    /* } */

    std::transform(mha_ref.begin(), mha_ref.end(), mha_T_test.begin(), mha_T_test.begin(), abs_diff<float>());
    const float T_error = *std::max_element(mha_T_test.begin(), mha_T_test.end());

    bool error = false;
    if (T_error > max_allowed_error) {
        error = true;
        printf("Max abs diff = %.2f\n", T_error);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    printf("[FP32] ");
    TIMEIT(true, 10, stream, masked_multihead_attention, params_fp32, stream);
    printf("[%s] ", string_rep_t<T>::value.c_str());
    TIMEIT(true, 10, stream, masked_multihead_attention, params_T, stream);

    return !error;
}

int main(int argc, char** argv)
{
    if (argc != 6) {
        printf("[ERROR] Usage: %s batch_size head_num max_seq_len"
               "size_per_head rotary_dim\n",
               argv[0]);
        printf("e.g., %s 32 16 40 256 32\n", argv[0]);
        return EXIT_FAILURE;
    }

    const test_args_t test_args{atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5])};

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    printf("Using device %s\n", prop.name);

    bool global_test_pass = true;
    bool test_pass        = true;

    test_pass = test_masked_multihead_attention<half>(test_args);
    printf("%s", test_pass ? "." : "X");
    global_test_pass |= test_pass;

#ifdef ENABLE_BF16
    test_pass = test_masked_multihead_attention<__nv_bfloat16>(test_args);
    printf("%s", test_pass ? "." : "X");
    global_test_pass |= test_pass;
#endif

#ifdef ENABLE_FP8
    test_pass = test_masked_multihead_attention<__nv_fp8_e4m3>(test_args);
    printf("%s", test_pass ? "." : "X");
    global_test_pass |= test_pass;
#endif

    puts("");
    return global_test_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
