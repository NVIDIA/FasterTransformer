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

#include "src/fastertransformer/utils/nccl_utils.h"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

template<typename T>
void ftNcclAllReduceSum(const T* send_buf, T* recv_buf, const int data_size, ncclComm_t comm, cudaStream_t stream)
{
    ncclDataType_t nccl_data_type;
    if (std::is_same<T, float>::value) {
        nccl_data_type = ncclFloat;
    }
    else if (std::is_same<T, half>::value) {
        nccl_data_type = ncclHalf;
    }
    else if (std::is_same<T, int>::value) {
        nccl_data_type = ncclInt;
    }
    else {
        printf("[ERROR] reduce sum only support float, half and int. \n");
        exit(-1);
    }
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclAllReduce((const void*)send_buf, (void*)recv_buf, data_size, nccl_data_type, ncclSum, comm, stream));
    NCCLCHECK(ncclGroupEnd());
}

template<typename T>
void ftNcclAllGather(
    const T* send_buf, T* recv_buf, const int data_size, const int rank, ncclComm_t comm, cudaStream_t stream)
{
    ncclDataType_t nccl_data_type;
    if (std::is_same<T, float>::value) {
        nccl_data_type = ncclFloat;
    }
    else if (std::is_same<T, half>::value) {
        nccl_data_type = ncclHalf;
    }
    else if (std::is_same<T, int>::value) {
        nccl_data_type = ncclInt;
    }
    else {
        printf("[ERROR] all2all gather only support float, half and int. \n");
        exit(-1);
    }
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclAllGather(send_buf + rank * data_size, recv_buf, data_size, nccl_data_type, comm, stream));
    NCCLCHECK(ncclGroupEnd());
}

template<typename T>
void ftNcclSend(const T* send_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream)
{
    ncclDataType_t nccl_data_type;
    if (std::is_same<T, float>::value) {
        nccl_data_type = ncclFloat;
    }
    else if (std::is_same<T, half>::value) {
        nccl_data_type = ncclHalf;
    }
    else if (std::is_same<T, int>::value) {
        nccl_data_type = ncclInt;
    }
    else if (std::is_same<T, bool>::value) {
        nccl_data_type = ncclInt8;
    }
    else {
        printf("[ERROR] ftNcclSend only support float, half, int and bool. \n");
        exit(-1);
    }
    NCCLCHECK(ncclSend(send_buf, data_size, nccl_data_type, peer, comm, stream));
    // cudaDeviceSynchronize();
}

template void
ftNcclSend(const float* send_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);
template void
ftNcclSend(const half* send_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);
template void
ftNcclSend(const int* send_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);
template void
ftNcclSend(const bool* send_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);

template<typename T>
void ftNcclRecv(T* recv_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream)
{
    ncclDataType_t nccl_data_type;
    if (std::is_same<T, float>::value) {
        nccl_data_type = ncclFloat;
    }
    else if (std::is_same<T, half>::value) {
        nccl_data_type = ncclHalf;
    }
    else if (std::is_same<T, int>::value) {
        nccl_data_type = ncclInt;
    }
    else if (std::is_same<T, bool>::value) {
        nccl_data_type = ncclInt8;
    }
    else {
        printf("[ERROR] ncclRecv only support float, half, int and bool. \n");
        exit(-1);
    }
    NCCLCHECK(ncclRecv(recv_buf, data_size, nccl_data_type, peer, comm, stream));
    // cudaDeviceSynchronize();
}

template void ftNcclRecv(float* recv_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);
template void ftNcclRecv(half* recv_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);
template void ftNcclRecv(int* recv_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);
template void ftNcclRecv(bool* recv_buf, const int data_size, const int peer, ncclComm_t comm, cudaStream_t stream);

template<typename T>
void ftNcclBroadCast(T* buff, const int data_size, const int root, ncclComm_t comm, cudaStream_t stream)
{
    ncclDataType_t nccl_data_type;
    if (std::is_same<T, bool>::value) {
        nccl_data_type = ncclInt8;
    }
    else if (std::is_same<T, int>::value) {
        nccl_data_type = ncclInt;
    }
    else {
        printf("[ERROR] ftNcclBroadCast only support bool and int. \n");
        exit(-1);
    }
    NCCLCHECK(ncclBcast(buff, data_size, nccl_data_type, root, comm, stream));
    // cudaDeviceSynchronize();
}

template void ftNcclBroadCast(bool* buff, const int data_size, const int root, ncclComm_t comm, cudaStream_t stream);
template void ftNcclBroadCast(int* buff, const int data_size, const int root, ncclComm_t comm, cudaStream_t stream);

template void
ftNcclAllReduceSum(const float* send_buf, float* recv_buf, const int data_size, ncclComm_t comm, cudaStream_t stream);

template void
ftNcclAllReduceSum(const half* send_buf, half* recv_buf, const int data_size, ncclComm_t comm, cudaStream_t stream);

template void ftNcclAllGather(
    const float* send_buf, float* recv_buf, const int data_size, const int rank, ncclComm_t comm, cudaStream_t stream);

template void ftNcclAllGather(
    const half* send_buf, half* recv_buf, const int data_size, const int rank, ncclComm_t comm, cudaStream_t stream);

size_t getLocalBatchSize(const size_t batch_size, const size_t seq_len, const size_t pipeline_para_size)
{
    size_t local_batch_size = batch_size;
    if (local_batch_size % pipeline_para_size == 0) {
        local_batch_size /= pipeline_para_size;
    }
    while (local_batch_size * seq_len > 2048 && local_batch_size % 2 == 0) {
        local_batch_size /= 2;
    }
    return local_batch_size;
}

}  // namespace fastertransformer
