/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_fp8.h>
#include "conv1x1_interface.hpp"
#include <unordered_map>
#include <memory>

namespace fastertransformer {

void invokeSwizzleQgmmaWeights(int C, int K, uint8_t* h_B, uint8_t* d_B, uint16_t* h_bias, uint16_t* d_bias);

class qgmma1x1Launcher
{
public:
    struct qgmmaDimsKey {
        bool relu;
        bool gelu;
    
        // constructor
        qgmmaDimsKey(bool relu, bool gelu)
        {
            this->relu = relu;
            this->gelu = gelu;
        }
    
        // `operator==` is required to compare keys in case of a hash collision
        bool operator==(const qgmmaDimsKey &p) const {
            return relu == p.relu && gelu == p.gelu;
        }
    };

    // The specialized hash function for `unordered_map` keys
    struct hashQgmmaDimsKey
    {
        std::size_t operator() (const qgmmaDimsKey &node) const
        {
            std::size_t h1 = std::hash<bool>()(node.relu);
            std::size_t h2 = std::hash<bool>()(node.gelu);
    
            return h1 ^ h2;
        }
    };

    qgmma1x1Launcher() {}

    template<bool RELU, bool GELU>
    void invokeQgmma1x1(__nv_fp8_e4m3* res,
                        int m,
                        int n,
                        int k,
                        const __nv_fp8_e4m3* input,
                        const __nv_fp8_e4m3* kernel,
                        const __nv_bfloat16* bias,
                        const float input_scale,
                        const float kernel_scale,
                        const float output_scale,
                        void* workspace,
                        cudaStream_t stream);

    template<bool RELU, bool GELU>
    void getWorkSpaceSize(int n,
                          size_t& workspace_size);

private:
    std::unordered_map<qgmmaDimsKey, std::unique_ptr<Conv1x1Interface>, hashQgmmaDimsKey> launcher_map_;
};

}  // namespace fastertransformer