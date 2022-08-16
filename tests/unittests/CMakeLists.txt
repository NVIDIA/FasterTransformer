# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

add_executable(test_gemm test_gemm.cu)
target_link_libraries(test_gemm PUBLIC -lcublas -lcudart -lcurand gemm cublasMMWrapper tensor)

add_executable(test_gpt_kernels test_gpt_kernels.cu)
target_link_libraries(test_gpt_kernels PUBLIC
                      gpt_kernels memory_utils)

add_executable(test_sampling test_sampling.cu)
target_link_libraries(test_sampling PUBLIC
                      -lcublas -lcublasLt -lcudart
                      cublasMMWrapper memory_utils
                      DynamicDecodeLayer TopKSamplingLayer TopPSamplingLayer TopKTopPSamplingLayer tensor)

add_executable(test_logprob_kernels test_logprob_kernels.cu)
target_link_libraries(test_logprob_kernels PUBLIC
                    -lcublas -lcublasLt -lcudart
                    logprob_kernels memory_utils)

add_executable(test_penalty_kernels test_penalty_kernels.cu)
target_link_libraries(test_penalty_kernels PUBLIC
                      -lcublas -lcublasLt -lcudart
                      sampling_penalty_kernels beam_search_penalty_kernels memory_utils)

add_executable(test_sampling_kernels test_sampling_kernels.cu)
target_link_libraries(test_sampling_kernels PUBLIC -lcudart
                      sampling_topk_kernels sampling_topp_kernels memory_utils tensor)
add_executable(test_tensor test_tensor.cu)
target_link_libraries(test_tensor PUBLIC tensor)

add_executable(test_activation test_activation.cu)
target_link_libraries(test_activation PUBLIC
                    -lcublas -lcublasLt -lcudart
                    activation_kernels memory_utils)

add_executable(test_context_decoder_layer test_context_decoder_layer.cu)
target_link_libraries(test_context_decoder_layer PUBLIC
                      ParallelGpt -lcublas -lcublasLt -lcudart
                      memory_utils tensor)
