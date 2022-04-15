/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/gptj/GptJ.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include <cuda_fp16.h>

namespace ft = fastertransformer;

template<typename T>
struct GptJTritonModel: public AbstractTransformerModel {
    GptJTritonModel(size_t max_seq_len,
                    size_t head_num,
                    size_t size_per_head,
                    size_t inter_size,
                    size_t num_layer,
                    size_t vocab_size,
                    size_t rotary_embedding_dim,
                    int start_id,
                    int end_id,
                    size_t tensor_para_size,
                    size_t pipeline_para_size,
                    int enable_custom_all_reduce,
                    std::string model_name,
                    std::string model_dir);

    ~GptJTritonModel() = default;

    virtual std::unique_ptr<AbstractTransformerModelInstance>
    createModelInstance(int deviceId,
                        int rank,
                        cudaStream_t stream,
                        std::pair<std::vector<ncclComm_t>, std::vector<ncclComm_t>> nccl_comms,
                        std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm = nullptr) override;

    virtual void createCustomComms(std::vector<std::shared_ptr<ft::AbstractCustomComm>>* custom_all_reduce_comms,
                                   int world_size) override;

    virtual std::pair<std::vector<ncclComm_t>, std::vector<ncclComm_t>>
    createNcclComms(std::vector<ncclUniqueId> nccl_ids,
                    const int node_id,
                    bool multi_instances = false,
                    int instance_id = 0) override;

    virtual std::vector<ncclUniqueId> createNcclIds(const uint32_t world_size, bool multi_instances = false) override;

    virtual std::string toString() override;
    virtual int getTensorParaSize() override;
    virtual int getPipelineParaSize() override;

private:
    const size_t max_seq_len_;
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t num_layer_;
    const size_t vocab_size_;
    const size_t rotary_embedding_dim_;
    const int start_id_;
    const int end_id_;
    const size_t tensor_para_size_;
    const size_t pipeline_para_size_;

    bool is_fp16_;
    int enable_custom_all_reduce_ = 0;

    std::string model_name_;
    std::string model_dir_;
};