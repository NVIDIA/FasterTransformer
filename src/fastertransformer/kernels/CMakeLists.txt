# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

add_subdirectory(cutlass_kernels)

add_library(image_shift_partition_kernels image_shift_partition_kernels.cu)
set_property(TARGET image_shift_partition_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET image_shift_partition_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(image_merge_kernels image_merge_kernels.cu)
set_property(TARGET image_merge_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET image_merge_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(normalize_kernels normalize_kernels.cu)
set_property(TARGET normalize_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET normalize_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(layernorm_kernels STATIC layernorm_kernels.cu)
set_property(TARGET layernorm_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET layernorm_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(layernorm_int8_kernels STATIC layernorm_int8_kernels.cu)
set_property(TARGET layernorm_int8_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET layernorm_int8_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

if(ENABLE_FP8)
add_library(layernorm_fp8_kernels STATIC layernorm_fp8_kernels.cu)
set_property(TARGET layernorm_fp8_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET layernorm_fp8_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
endif()

if(ENABLE_FP8)
add_executable(layernorm_fp8_kernels_test layernorm_fp8_kernels_test.cc)
target_link_libraries(layernorm_fp8_kernels_test PUBLIC layernorm_fp8_kernels layernorm_kernels memory_utils)
endif()

add_library(ban_bad_words STATIC ban_bad_words.cu)
set_property(TARGET ban_bad_words PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ban_bad_words PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(stop_criteria STATIC stop_criteria_kernels.cu)
set_property(TARGET stop_criteria PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET stop_criteria PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(activation_kernels STATIC activation_kernels.cu)
set_property(TARGET activation_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET activation_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(activation_int8_kernels STATIC activation_int8_kernels.cu)
set_property(TARGET activation_int8_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET activation_int8_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

if(ENABLE_FP8)
add_library(activation_fp8_kernels STATIC activation_fp8_kernels.cu)
set_property(TARGET activation_fp8_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET activation_fp8_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
endif()

add_library(layout_transformer_int8_kernels STATIC layout_transformer_int8_kernels.cu)
set_property(TARGET layout_transformer_int8_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET layout_transformer_int8_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(quantization_int8_kernels STATIC quantization_int8_kernels.cu)
set_property(TARGET quantization_int8_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET quantization_int8_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(quantize_weight STATIC quantize_weight.cu)
set_property(TARGET quantize_weight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET quantize_weight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(calibrate_quantize_weight_kernels STATIC calibrate_quantize_weight_kernels.cu)
set_property(TARGET calibrate_quantize_weight_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET calibrate_quantize_weight_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(gen_relative_pos_bias STATIC gen_relative_pos_bias.cu)
set_property(TARGET gen_relative_pos_bias PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET gen_relative_pos_bias PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(gen_relative_pos_bias PUBLIC activation_kernels)

add_library(transform_mask_kernels STATIC transform_mask_kernels.cu)
set_property(TARGET transform_mask_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET transform_mask_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(reverse_roll_kernels STATIC reverse_roll_kernels.cu)
set_property(TARGET reverse_roll_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET reverse_roll_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(dequantize_kernels STATIC dequantize_kernels.cu)
set_property(TARGET dequantize_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET dequantize_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(softmax_int8_kernels STATIC softmax_int8_kernels.cu)
set_property(TARGET softmax_int8_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET softmax_int8_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(logprob_kernels STATIC logprob_kernels.cu)
set_property(TARGET logprob_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET logprob_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(transpose_int8_kernels STATIC transpose_int8_kernels.cu)
set_property(TARGET transpose_int8_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET transpose_int8_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(matrix_transpose_kernels STATIC matrix_transpose_kernels.cu)
set_property(TARGET matrix_transpose_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET matrix_transpose_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

if(ENABLE_FP8)
add_executable(matrix_transpose_fp8_kernels_test matrix_transpose_fp8_kernels_test.cc)
target_link_libraries(matrix_transpose_fp8_kernels_test PUBLIC matrix_transpose_kernels memory_utils)
endif()

add_library(unfused_attention_int8_kernels STATIC unfused_attention_int8_kernels.cu)
set_property(TARGET unfused_attention_int8_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET unfused_attention_int8_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(unfused_attention_kernels STATIC unfused_attention_kernels.cu)
set_property(TARGET unfused_attention_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET unfused_attention_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

if(ENABLE_FP8)
add_library(unfused_attention_fp8_kernels STATIC unfused_attention_fp8_kernels.cu)
set_property(TARGET unfused_attention_fp8_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET unfused_attention_fp8_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
endif()

add_library(disentangled_attention_kernels STATIC disentangled_attention_kernels.cu)
set_property(TARGET disentangled_attention_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET disentangled_attention_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

if(ENABLE_FP8)
add_executable(unfused_attention_fp8_kernels_test unfused_attention_fp8_kernels_test.cc)
target_link_libraries(unfused_attention_fp8_kernels_test PUBLIC unfused_attention_fp8_kernels unfused_attention_kernels memory_utils)
endif()

add_library(bert_preprocess_kernels STATIC bert_preprocess_kernels.cu)
set_property(TARGET bert_preprocess_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET bert_preprocess_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(xlnet_preprocess_kernels STATIC xlnet_preprocess_kernels.cu)
set_property(TARGET xlnet_preprocess_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET xlnet_preprocess_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(xlnet_attention_kernels STATIC xlnet_attention_kernels.cu)
set_property(TARGET xlnet_attention_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET xlnet_attention_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)


set(decoder_masked_multihead_attention_files
    decoder_masked_multihead_attention.cu
)
file(GLOB decoder_masked_multihead_attention_files ${decoder_masked_multihead_attention_files} ./decoder_masked_multihead_attention/*.cu)
add_library(decoder_masked_multihead_attention STATIC ${decoder_masked_multihead_attention_files})
set_property(TARGET decoder_masked_multihead_attention PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET decoder_masked_multihead_attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

if(ENABLE_FP8)
add_executable(decoder_masked_multihead_attention_fp8_test decoder_masked_multihead_attention_fp8_test.cc)
target_link_libraries(decoder_masked_multihead_attention_fp8_test PUBLIC decoder_masked_multihead_attention memory_utils)
endif()

add_library(add_residual_kernels STATIC add_residual_kernels.cu)
set_property(TARGET add_residual_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET add_residual_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(longformer_kernels STATIC longformer_kernels.cu)
set_property(TARGET longformer_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET longformer_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(add_bias_transpose_kernels STATIC add_bias_transpose_kernels.cu)
set_property(TARGET add_bias_transpose_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET add_bias_transpose_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(online_softmax_beamsearch_kernels STATIC online_softmax_beamsearch_kernels.cu)
set_property(TARGET online_softmax_beamsearch_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET online_softmax_beamsearch_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(decoding_kernels STATIC decoding_kernels.cu)
set_property(TARGET decoding_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET decoding_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(gpt_kernels STATIC gpt_kernels.cu)
set_property(TARGET gpt_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET gpt_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(beam_search_penalty_kernels STATIC beam_search_penalty_kernels.cu)
set_property(TARGET beam_search_penalty_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET beam_search_penalty_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(beam_search_penalty_kernels PRIVATE cuda_utils)

add_library(beam_search_topk_kernels STATIC beam_search_topk_kernels.cu)
set_property(TARGET beam_search_topk_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET beam_search_topk_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(sampling_topk_kernels STATIC sampling_topk_kernels.cu)
set_property(TARGET sampling_topk_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET sampling_topk_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(sampling_topp_kernels STATIC sampling_topp_kernels.cu)
set_property(TARGET sampling_topp_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET sampling_topp_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(sampling_penalty_kernels STATIC sampling_penalty_kernels.cu)
set_property(TARGET sampling_penalty_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET sampling_penalty_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(matrix_vector_multiplication STATIC matrix_vector_multiplication.cu)
set_property(TARGET matrix_vector_multiplication PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET matrix_vector_multiplication PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(custom_ar_kernels STATIC custom_ar_kernels.cu)
set_property(TARGET custom_ar_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET custom_ar_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(vit_kernels STATIC vit_kernels.cu)
set_property(TARGET vit_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET vit_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(moe_kernels STATIC moe_kernels.cu)
set_property(TARGET moe_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET moe_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(moe_kernels PRIVATE moe_gemm_kernels)
