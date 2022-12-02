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

cmake_minimum_required(VERSION 3.8)

set(t5_encoder_triton_backend_files
    T5EncoderTritonModel.cc
    T5EncoderTritonModelInstance.cc
)

add_library(T5EncoderTritonBackend SHARED ${t5_encoder_triton_backend_files})
target_link_libraries(T5EncoderTritonBackend PRIVATE TransformerTritonBackend T5Encoder -lcublasLt)
target_compile_features(T5EncoderTritonBackend PRIVATE cxx_std_14)
