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

namespace fastertransformer {

void invokeMappingRemovePaddingData(const int batch_size,
                                    const int seq_len,
                                    const int valid_word_num,
                                    int* mapping,
                                    const int* sequence_id_offset,
                                    cudaStream_t stream);

template<typename T>
void invokeAddQKBiasTransform(int8_t* q_buf,
                              int8_t* k_buf,
                              const int32_t* Q,
                              const T* bias_Q,
                              const int32_t* K,
                              const T* bias_K,
                              const int batch_size,
                              const int seq_len,
                              const int head_num,
                              const int size_per_head,
                              const float* q_weight_amax,
                              const float* q_input_deQFactor_div127_ptr,
                              const float* k_weight_amax,
                              const float* k_input_deQFactor_div127_ptr,
                              const float* q_output_scale_ptr,
                              const float* k_output_scale_ptr,
                              bool use_ORDER_COL32_2R_4R4,
                              cudaStream_t stream);

template<typename T>
void invokeAddQKBiasTransform(int8_t* q_buf,
                              int8_t* k_buf,
                              const int8_t* Q,
                              const T* bias_Q,
                              const int8_t* K,
                              const T* bias_K,
                              const int batch_size,
                              const int seq_len,
                              const int head_num,
                              const int size_per_head,
                              const float* q_input_deQFactor_ptr,
                              const float* k_input_deQFactor_ptr,
                              const float* q_output_scale_ptr,
                              const float* k_output_scale_ptr,
                              bool use_ORDER_COL32_2R_4R4,
                              cudaStream_t stream);

template<typename T>
void invokeAddQKBiasTransformRow(int8_t* q_buf,
                                 int8_t* k_buf,
                                 const int8_t* Q,
                                 const T* bias_Q,
                                 const int8_t* K,
                                 const T* bias_K,
                                 const int batch_size,
                                 const int seq_len,
                                 const int head_num,
                                 const int size_per_head,
                                 const float* q_input_deQFactor_ptr,
                                 const float* k_input_deQFactor_ptr,
                                 const float* q_output_scale_ptr,
                                 const float* k_output_scale_ptr,
                                 bool use_ORDER_COL32_2R_4R4,
                                 cudaStream_t stream);

template<typename T>
void invokeAddQKBiasTransformRebuildPadding(int8_t* q_buf,
                                            int8_t* k_buf,
                                            const int32_t* Q,
                                            const T* bias_Q,
                                            const int32_t* K,
                                            const T* bias_K,
                                            const int* sequence_id_offset,
                                            const int valid_word_num,
                                            const int batch_size,
                                            const int seq_len,
                                            const int head_num,
                                            const int size_per_head,
                                            const float* q_weight_amax,
                                            const float* q_input_deQFactor_div127_ptr,
                                            const float* k_weight_amax,
                                            const float* k_input_deQFactor_div127_ptr,
                                            const float* q_output_scale_ptr,
                                            const float* k_output_scale_ptr,
                                            bool use_ORDER_COL32_2R_4R4,
                                            cudaStream_t stream);

template<typename T>
void invokeAddQKBiasTransformRebuildPadding(int8_t* q_buf,
                                            int8_t* k_buf,
                                            const int8_t* Q,
                                            const T* bias_Q,
                                            const int8_t* K,
                                            const T* bias_K,
                                            const int* sequence_id_offset,
                                            const int valid_word_num,
                                            const int batch_size,
                                            const int seq_len,
                                            const int head_num,
                                            const int size_per_head,
                                            const float* q_deQFactor_ptr,
                                            const float* k_deQFactor_ptr,
                                            const float* q_output_scale_ptr,
                                            const float* k_output_scale_ptr,
                                            bool use_ORDER_COL32_2R_4R4,
                                            cudaStream_t stream);

template<typename T>
void invokeAddQKBiasTransformRebuildPaddingRow(int8_t* q_buf,
                                               int8_t* k_buf,
                                               const int8_t* Q,
                                               const T* bias_Q,
                                               const int8_t* K,
                                               const T* bias_K,
                                               const int* sequence_id_offset,
                                               const int valid_word_num,
                                               const int batch_size,
                                               const int seq_len,
                                               const int head_num,
                                               const int size_per_head,
                                               const float* q_deQFactor_ptr,
                                               const float* k_deQFactor_ptr,
                                               const float* q_output_scale_ptr,
                                               const float* k_output_scale_ptr,
                                               bool use_ORDER_COL32_2R_4R4,
                                               cudaStream_t stream);

template<typename T>
void invokeAddVBiasTransform(int8_t* v_buf,
                             const int32_t* V,
                             const T* V_bias,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head,
                             const float* weight_amax,
                             const float* input_deQFactor_div127_ptr,
                             const float* out_scale_ptr,
                             bool use_ORDER_COL32_2R_4R4,
                             cudaStream_t stream);

template<typename T>
void invokeAddVBiasTransform(int8_t* v_buf,
                             const int8_t* V,
                             const T* V_bias,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head,
                             const float* input_deQFactor_ptr,
                             const float* out_scale_ptr,
                             bool use_ORDER_COL32_2R_4R4,
                             cudaStream_t stream);

template<typename T>
void invokeAddVBiasTransformRow(int8_t* v_buf,
                                const int8_t* V,
                                const T* V_bias,
                                const int batch_size,
                                const int seq_len,
                                const int head_num,
                                const int size_per_head,
                                const float* input_deQFactor_ptr,
                                const float* out_scale_ptr,
                                bool use_ORDER_COL32_2R_4R4,
                                cudaStream_t stream);

template<typename T>
void invokeAddVBiasTransformRebuildPadding(int8_t* v_buf,
                                           const int32_t* V,
                                           const T* V_bias,
                                           const int* sequence_id_map,
                                           const int valid_word_num,
                                           const int batch_size,
                                           const int seq_len,
                                           const int head_num,
                                           const int size_per_head,
                                           const float* weight_amax,
                                           const float* input_deQFactor_div127_ptr,
                                           const float* out_scale_ptr,
                                           bool use_ORDER_COL32_2R_4R4,
                                           cudaStream_t stream);

template<typename T>
void invokeAddVBiasTransformRebuildPadding(int8_t* v_buf,
                                           const int8_t* V,
                                           const T* V_bias,
                                           const int* sequence_id_map,
                                           const int valid_word_num,
                                           const int batch_size,
                                           const int seq_len,
                                           const int head_num,
                                           const int size_per_head,
                                           const float* deQFactor_ptr,
                                           const float* out_scale_ptr,
                                           bool use_ORDER_COL32_2R_4R4,
                                           cudaStream_t stream);

template<typename T>
void invokeAddVBiasTransformRebuildPaddingRow(int8_t* v_buf,
                                              const int8_t* V,
                                              const T* V_bias,
                                              const int* sequence_id_map,
                                              const int valid_word_num,
                                              const int batch_size,
                                              const int seq_len,
                                              const int head_num,
                                              const int size_per_head,
                                              const float* deQFactor_ptr,
                                              const float* out_scale_ptr,
                                              bool use_ORDER_COL32_2R_4R4,
                                              cudaStream_t stream);

}  // namespace fastertransformer
