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

add_definitions(-DGOOGLE_CUDA=1)

if(EXISTS ${TF_PATH}libtensorflow_framework.so)
    set(tf_link
        -ltensorflow_framework
    )
elseif(EXISTS ${TF_PATH}libtensorflow_framework.so.1)
    set(tf_link
        -l:libtensorflow_framework.so.1
    )
elseif(EXISTS ${TF_PATH}libtensorflow_framework.so.2)
    set(tf_link
        -l:libtensorflow_framework.so.2
    )
endif()

add_subdirectory(bert)
add_subdirectory(encoder)
add_subdirectory(decoder)
add_subdirectory(decoding)
add_subdirectory(gpt)