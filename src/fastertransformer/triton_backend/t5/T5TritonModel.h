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

#include "3rdparty/INIReader.h"
#include "src/fastertransformer/models/t5/T5Decoding.h"
#include "src/fastertransformer/models/t5/T5Encoder.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include <cuda_fp16.h>

namespace ft = fastertransformer;

template<typename T>
struct T5TritonModel: public AbstractTransformerModel {
    T5TritonModel(INIReader reader, std::string model_dir);

    T5TritonModel(size_t tensor_para_size,
                  size_t pipeline_para_size,
                  int enable_custom_all_reduce,
                  std::string model_dir,
                  int int8_mode);

    ~T5TritonModel() = default;

    virtual std::unique_ptr<AbstractTransformerModelInstance>
    createModelInstance(int deviceId,
                        int rank,
                        cudaStream_t stream,
                        std::pair<std::vector<ncclComm_t>, std::vector<ncclComm_t>> nccl_comms,
                        std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm = nullptr);

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
    // encoder
    size_t encoder_head_num_;
    size_t encoder_size_per_head_;
    size_t encoder_d_model_;
    size_t encoder_inter_size_;
    size_t encoder_num_layer_;
    size_t encoder_vocab_size_;
    size_t encoder_num_bucket_or_max_pos_seq_len_;

    // decoding
    size_t decoding_head_num_;
    size_t decoding_size_per_head_;
    size_t decoding_d_model_;
    size_t decoding_inter_size_;
    size_t decoding_num_layer_;
    size_t decoding_vocab_size_;
    size_t decoding_num_bucket_or_max_pos_seq_len_;

    float q_scaling_;

    size_t max_distance_;
    int start_id_;
    int end_id_;

    size_t tensor_para_size_;
    size_t pipeline_para_size_;

    // t5 structure difference
    bool t5_with_bias_;
    ft::PositionEmbeddingType position_embedding_type_;

    bool is_fp16_;
    int int8_mode_;

    int enable_custom_all_reduce_ = 0;

    std::string model_name_;
    std::string model_dir_;
};