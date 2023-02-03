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

add_executable(wenet_encoder_example wenet_encoder_example.cc)
target_link_libraries(wenet_encoder_example PUBLIC -lcublas -lcublasLt -lcudart "${CUDNN_LIBRARY_PATH}" WenetEncoder)
if (SPARSITY_SUPPORT)
target_link_libraries(wenet_encoder_example PUBLIC -lcusparse -lcusparseLt)
endif()

add_executable(wenet_decoder_example wenet_decoder_example.cc)
target_link_libraries(wenet_decoder_example PUBLIC -lcublas -lcublasLt -lcudart "${CUDNN_LIBRARY_PATH}" WenetDecoder)
if(SPARSITY_SUPPORT)
    target_link_libraries(wenet_decoder_example PUBLIC -lcublas -lcusparse -lcusparseLt)
endif()
