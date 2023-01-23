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

add_executable(bert_example bert_example.cc)
target_link_libraries(bert_example PUBLIC -lcublas -lcublasLt -lcudart nvtx_utils Bert)
if (SPARSITY_SUPPORT)
target_link_libraries(bert_example PUBLIC -lcusparse -lcusparseLt)
endif()

add_executable(bert_triton_example bert_triton_example.cc)
target_link_libraries(bert_triton_example PUBLIC -lcublas -lcublasLt -lcudart -lpthread
                      BertTritonBackend TransformerTritonBackend nvtx_utils mpi_utils nccl_utils)
