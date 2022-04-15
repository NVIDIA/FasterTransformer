/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "cuda_runtime.h"
#include "nccl.h"
#include <stdio.h>

#if defined(NCCL_VERSION_CODE) && (NCCL_VERSION_CODE >= 21003)
#define ENABLE_BF16_NCCL
#endif

namespace fastertransformer {

#define NCCLCHECK(cmd)                                                                                                 \
    do {                                                                                                               \
        ncclResult_t r = cmd;                                                                                          \
        if (r != ncclSuccess) {                                                                                        \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

struct NcclParam {
    int rank_{0};
    int world_size_{1};
    ncclComm_t nccl_comm_ = nullptr;

    NcclParam(): rank_(0), world_size_(1), nccl_comm_(nullptr){};
    NcclParam(int rank, int world_size, ncclComm_t comm): rank_(rank), world_size_(world_size), nccl_comm_(comm){};
    NcclParam(NcclParam const& param):
        rank_(param.rank_), world_size_(param.world_size_), nccl_comm_(param.nccl_comm_){};

    // int layers_per_group{0};
    // bool is_valid(int i)
    // {
    //     if (i >= layers_per_group * rank && i < layers_per_group * (rank + 1))
    //         return true;
    //     else
    //         return false;
    // }
    // int local_batch_size{-1};
};

// New APIs
template<typename T>
void ftNcclAllReduceSum(const T* send_buf, T* recv_buf, const int data_size, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
void ftNcclAllGather(
    const T* send_buf, T* recv_buf, const int data_size, const int rank, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
void ftNcclBroadCast(T* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
void ftNcclRecv(T* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
void ftNcclSend(const T* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
ncclDataType_t getNcclDataType();

size_t getLocalBatchSize(const size_t batch_size, const size_t seq_len, const size_t pipeline_para_size);

}  // namespace fastertransformer