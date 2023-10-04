/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "src/fastertransformer/models/bart/BartDecoding.h"
#include "src/fastertransformer/models/bart/BartEncoder.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include <cuda_fp16.h>

namespace ft = fastertransformer;

template<typename T>
struct BartTritonModel: public AbstractTransformerModel {
    BartTritonModel(INIReader reader, std::string model_dir);

    BartTritonModel(size_t      tensor_para_size,
                  size_t      pipeline_para_size,
                  int         enable_custom_all_reduce,
                  std::string model_dir,
                  int         int8_mode);

    ~BartTritonModel() = default;

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
    size_t                  encoder_head_num_;
    size_t                  encoder_size_per_head_;
    size_t                  encoder_d_model_;
    size_t                  encoder_inter_size_;
    size_t                  encoder_num_layer_;
    size_t                  encoder_vocab_size_;
    size_t                  encoder_max_pos_seq_len_;

    // decoding
    size_t                  decoding_head_num_;
    size_t                  decoding_size_per_head_;
    size_t                  decoding_d_model_;
    size_t                  decoding_inter_size_;
    size_t                  decoding_num_layer_;
    size_t                  decoding_vocab_size_;
    size_t                  decoding_max_pos_seq_len_;

    float  q_scaling_ = 1.f;

    size_t max_distance_;
    int    start_id_;
    int    end_id_;

    bool tie_word_embeddings_ = false;

    size_t tensor_para_size_;
    size_t pipeline_para_size_;

    // shared weights for each device
    std::vector<std::shared_ptr<ft::BartEncoderWeight<T>>>  encoder_shared_weights_;
    std::vector<std::shared_ptr<ft::BartDecodingWeight<T>>> decoding_shared_weights_;

    // bart structure difference
    bool                      bart_with_bias_ = true;
    bool                      mbart_para_ = false;
    bool                      use_gated_activation_ = false;
    ft::PositionEmbeddingType position_embedding_type_ = ft::PositionEmbeddingType::absolute;
    ft::ActivationType        activation_type_;
    ft::LayerNormType         layernorm_type_ = ft::LayerNormType::post_layernorm;

    bool is_fp16_;
    int  int8_mode_;

    int enable_custom_all_reduce_ = 0;

    std::string model_name_;
    std::string model_dir_;
};
