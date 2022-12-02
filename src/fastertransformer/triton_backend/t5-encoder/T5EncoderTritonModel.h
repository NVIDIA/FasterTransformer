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

#pragma once

#include "3rdparty/INIReader.h"
#include "src/fastertransformer/models/t5/T5Encoder.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include <cuda_fp16.h>

namespace ft = fastertransformer;

template<typename T>
struct T5EncoderTritonModel: public AbstractTransformerModel {
    T5EncoderTritonModel(INIReader reader, std::string model_dir);

    T5EncoderTritonModel(size_t      tensor_para_size,
                         size_t      pipeline_para_size,
                         int         enable_custom_all_reduce,
                         std::string model_dir,
                         int         int8_mode);

    ~T5EncoderTritonModel() = default;

    virtual std::unique_ptr<AbstractTransformerModelInstance>
    createModelInstance(int                                                               deviceId,
                        int                                                               rank,
                        cudaStream_t                                                      stream,
                        std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
                        std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm = nullptr);

    virtual void createSharedWeights(int deviceId, int rank) override;

    virtual void createCustomComms(std::vector<std::shared_ptr<ft::AbstractCustomComm>>* custom_all_reduce_comms,
                                   int                                                   world_size) override;

    virtual std::string toString() override;
    virtual int         getTensorParaSize() override;
    virtual int         getPipelineParaSize() override;

private:
    // encoder
    size_t encoder_head_num_;
    size_t encoder_size_per_head_;
    size_t encoder_d_model_;
    size_t encoder_inter_size_;
    size_t encoder_num_layer_;
    size_t encoder_vocab_size_;
    size_t encoder_num_bucket_or_max_pos_seq_len_;

    // prompt for encoder
    size_t                                     num_tasks_                  = 0;
    int                                        prompt_learning_start_id_   = 0;
    ft::PromptLearningType                     prompt_learning_type_       = ft::PromptLearningType::no_prompt;
    std::map<std::string, std::pair<int, int>> prompt_learning_table_pair_ = {};

    float  q_scaling_;
    size_t ia3_num_tasks_;

    size_t max_distance_;

    size_t tensor_para_size_;
    size_t pipeline_para_size_;

    // shared weights for each device
    std::vector<std::shared_ptr<ft::T5EncoderWeight<T>>> encoder_shared_weights_;

    // t5 structure difference
    bool                      t5_with_bias_;
    bool                      use_gated_activation_;
    ft::PositionEmbeddingType position_embedding_type_;
    ft::ActivationType        activation_type_;

    bool is_fp16_;
    int  int8_mode_;

    int enable_custom_all_reduce_ = 0;

    std::string model_name_;
    std::string model_dir_;
};
