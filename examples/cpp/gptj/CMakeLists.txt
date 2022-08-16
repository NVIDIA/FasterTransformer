# Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

add_executable(gptj_example gptj_example.cc)
target_link_libraries(gptj_example PUBLIC -lcublas -lcublasLt -lcudart
                      GptJ nvtx_utils gpt_example_utils word_list mpi_utils nccl_utils)

add_executable(gptj_triton_example gptj_triton_example.cc)
target_link_libraries(gptj_triton_example PUBLIC -lcublas -lcublasLt -lcudart -lpthread
                      GptJTritonBackend TransformerTritonBackend custom_ar_comm
                      gpt_example_utils word_list mpi_utils nccl_utils)
