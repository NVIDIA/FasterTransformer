#
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
#

import numpy as np

np.random.seed(97)

data = {}

fpList = [32,16]
bsList = [1,8,32,128]
slList = [32,128,384]

for bs in bsList:
    for sl in slList:       
        for fp in fpList:
            name = '-fp'+str(fp)+'-bs'+str(bs)+'-sl'+str(sl)
            data['encoder'+name]    = np.random.randint(0,32128,[bs,sl]).astype(np.int32)
            data['decoding'+name]   = np.random.rand(bs,sl,512).astype([np.float32,np.float16][int(fp==16)])*2-1
            data['seqLen'+name]     = np.full([bs],sl,dtype=np.int32)

np.savez("T5PluginTestIO.npz",**data)

#for k in data.keys():
#    print(k,data[k].shape,data[k].dtype,data[k].reshape(-1)[:10])
print("create T5 test data finish!")

