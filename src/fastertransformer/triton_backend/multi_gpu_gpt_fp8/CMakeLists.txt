# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

set(parallel_gpt_fp8_triton_backend_files
    ParallelGptFP8TritonModel.cc
    ParallelGptFP8TritonModelInstance.cc
)

add_library(ParallelGptFP8TritonBackend STATIC ${parallel_gpt_fp8_triton_backend_files})
set_property(TARGET ParallelGptFP8TritonBackend PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(ParallelGptFP8TritonBackend PRIVATE TransformerTritonBackend GptFP8)
target_compile_features(ParallelGptFP8TritonBackend PRIVATE cxx_std_14)