# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

add_executable(gpt_fp8_example gpt_fp8_example.cc)
target_link_libraries(gpt_fp8_example PUBLIC -lcublas -lcublasLt -lcudart GptFP8 ParallelGptWeight nvtx_utils nccl_utils mpi_utils)

add_executable(gpt_fp8_triton_example gpt_fp8_triton_example.cc)
target_link_libraries(gpt_fp8_triton_example PUBLIC -lcublas -lcublasLt -lcudart
                            ParallelGptFP8TritonBackend TransformerTritonBackend memory_utils
                            custom_ar_comm gpt_example_utils -lpthread)
