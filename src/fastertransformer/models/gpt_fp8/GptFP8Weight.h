/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/kernels/layernorm_fp8_kernels.h"
#include "src/fastertransformer/models/gpt_fp8/GptFP8DecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T1, typename T2>
struct GptFP8Weight {

    GptFP8Weight() = default;
    GptFP8Weight(const int hidden_units,
                 const int inter_size,
                 const int vocab_size,
                 const int num_layer,
                 const int max_seq_len,
                 const int tensor_para_size,
                 const int tensor_para_rank,
                 const int layer_para_size,
                 const int layer_para_rank);
    ~GptFP8Weight();
    GptFP8Weight(const GptFP8Weight& other);
    GptFP8Weight& operator=(const GptFP8Weight& other);
    void          loadModel(std::string dir_path);
    void          resizeLayer(const int num_layer);
#ifdef SPARSITY_ENABLED
    void compress_weights(cublasMMWrapper& cublas_wrapper);
#endif
    void transposeWeight();

    std::vector<GptFP8DecoderLayerWeight<T1, T2>*> decoder_layer_weights;
    const T2*                                      position_encoding_table     = nullptr;
    const T2*                                      pre_decoder_embedding_table = nullptr;
    LayerNormWeight<T2>                            post_decoder_layernorm;
    DenseWeight<T2, T2>                            post_decoder_embedding;

private:
    void setWeightPtr();
    void mallocWeights();
    bool isValidLayerParallelId(int l);

    int                                   hidden_units_;
    int                                   inter_size_;
    int                                   vocab_size_;
    int                                   num_layer_;
    int                                   max_seq_len_;
    int                                   tensor_para_size_;
    int                                   tensor_para_rank_;
    int                                   layer_para_size_;
    int                                   layer_para_rank_;
    bool                                  is_maintain_buffer = false;
    std::vector<std::pair<uint32_t, T2*>> table_ptr;
    std::vector<std::pair<uint32_t, T2*>> vec_ptr;
};

}  // namespace fastertransformer
