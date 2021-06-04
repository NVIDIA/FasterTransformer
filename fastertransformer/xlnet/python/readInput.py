# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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




import numpy as np


if __name__=="__main__":
    data = np.load('./data.npz')
    batch_size = 8
    max_seq_length=128

    arr_input_ids=data["input_ids:0"];
    arr_input_mask=data["input_mask:0"];
    arr_segment_ids=data["segment_ids:0"];
    arr_label_ids=data["label_ids:0"];

    print("Length {} {} {} {} ".format(arr_input_ids.shape, arr_input_mask.shape, arr_segment_ids.shape,
            arr_label_ids.shape))
    for i in range(arr_input_ids.size/128):
        if i % 500 == 0:
            input_ids=arr_input_ids[i,:]
            print("Writing example {} in shape {} with {} ".format(i,input_ids.shape, input_ids))


