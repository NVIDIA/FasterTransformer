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

#include "src/fastertransformer/models/bert/Bert.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"

namespace ft = fastertransformer;

template<typename T>
struct BertTritonModel: public AbstractTransformerModel {
    BertTritonModel(size_t      tensor_para_size,
                    size_t      pipeline_para_size,
                    bool        enable_custom_all_reduce,
                    std::string model_dir,
                    int         int8_mode,
                    bool        is_sparse,
                    bool        is_remove_padding);

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
    size_t tensor_para_size_;
    size_t pipeline_para_size_;

    float              q_scaling_;
    bool               is_remove_padding_;
    bool               is_sparse_;
    ft::ActivationType activation_type_;
    ft::LayerNormType  layernorm_type_;

    std::string                                     model_name_;
    std::string                                     model_dir_;
    int                                             int8_mode_                = 0;
    bool                                            enable_custom_all_reduce_ = 0;
    bool                                            is_sparse                 = false;
    std::vector<std::shared_ptr<ft::BertWeight<T>>> shared_weights_;
};
