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

#include "src/fastertransformer/models/gptneox/GptNeoX.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include <cuda_fp16.h>

namespace ft = fastertransformer;

template<typename T>
struct GptNeoXTritonModel: public AbstractTransformerModel {
    GptNeoXTritonModel(size_t      tensor_para_size,
                       size_t      pipeline_para_size,
                       int         enable_custom_all_reduce,
                       std::string model_dir);

    ~GptNeoXTritonModel() = default;

    virtual std::unique_ptr<AbstractTransformerModelInstance>
    createModelInstance(int                                                               deviceId,
                        int                                                               rank,
                        cudaStream_t                                                      stream,
                        std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
                        std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm = nullptr) override;

    virtual void createSharedWeights(int deviceId, int rank) override;

    virtual void createCustomComms(std::vector<std::shared_ptr<ft::AbstractCustomComm>>* custom_all_reduce_comms,
                                   int                                                   world_size) override;

    virtual std::string toString() override;
    virtual int         getTensorParaSize() override;
    virtual int         getPipelineParaSize() override;

private:
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t vocab_size_;
    size_t rotary_embedding_dim_;
    int    start_id_;
    int    end_id_;
    size_t tensor_para_size_;
    size_t pipeline_para_size_;

    // shared weights for each device
    std::vector<std::shared_ptr<ft::GptNeoXWeight<T>>> shared_weights_;

    // residual type
    bool use_gptj_residual_ = true;

    // number of tasks (for prefix-prompt, p/prompt-tuning)
    size_t                                     num_tasks_                  = 0;
    int                                        prompt_learning_start_id_   = 0;
    ft::PromptLearningType                     prompt_learning_type_       = ft::PromptLearningType::no_prompt;
    std::map<std::string, std::pair<int, int>> prompt_learning_table_pair_ = {};

    bool is_fp16_;
    int  enable_custom_all_reduce_ = 0;

    std::string model_name_;
    std::string model_dir_;
};
