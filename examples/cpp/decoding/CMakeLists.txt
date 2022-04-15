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

add_executable(decoding_example decoding_example.cc)
target_link_libraries(decoding_example PUBLIC -lcublasLt Decoding nvtx_utils)

add_executable(layernorm_test layernorm_test.cc)
target_link_libraries(layernorm_test PUBLIC -lcublasLt layernorm_kernels memory_utils)
