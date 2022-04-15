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

add_subdirectory(bert)
add_subdirectory(bert_int8)
add_subdirectory(decoder)
add_subdirectory(longformer)
add_subdirectory(decoding)
add_subdirectory(xlnet)

add_subdirectory(t5)
add_subdirectory(gptj)
add_subdirectory(multi_gpu_gpt)
add_subdirectory(swin)
add_subdirectory(swin_int8)
add_subdirectory(vit)
add_subdirectory(vit_int8)