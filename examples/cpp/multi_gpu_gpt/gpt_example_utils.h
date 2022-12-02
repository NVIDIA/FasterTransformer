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

#include <cuda.h>
#include <map>
#include <string>
#include <vector>

#include "3rdparty/INIReader.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace fastertransformer {

struct model_config_t {
    std::string model_name;
    std::string model_dir;
    bool        sparse;
    int         int8_mode;
    int         attention_type;

    int tensor_para_size;
    int pipeline_para_size;

    size_t head_num;
    size_t size_per_head;
    size_t vocab_size;
    size_t decoder_layers;
    size_t hidden_units;
    size_t inter_size;
    size_t max_seq_len;

    int start_id;
    int end_id;

    int                prompt_learning_start_id;
    PromptLearningType prompt_learning_type;
    int                prompt_learning_num_tasks;

    gptVariantParams gpt_variants;
};

struct request_config_t {
    size_t   request_batch_size;
    size_t   request_output_len;
    bool     is_return_log_probs;
    bool     is_return_context_cum_log_probs;
    bool     is_return_context_embeddings;
    bool     remove_padding;
    uint32_t memory_len;

    size_t beam_width;
    int    top_k;
    float  top_p;
    float  temperature;
    float  repetition_penalty;

    float len_penalty;
    float beam_search_diversity_rate;
    float shared_contexts_ratio;

    int start_id;
    int end_id;
};

model_config_t   read_model_config(const INIReader& reader);
request_config_t read_request_config(const INIReader& reader);

template<typename T>
static inline void safe_free(T*& ptr)
{
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

std::map<std::string, std::pair<int, int>> init_prompt_tuning_map(const INIReader&      reader,
                                                                  const model_config_t& config);

Allocator<AllocatorType::CUDA> init_cuda_ctx(cudaStream_t& stream, cudaDeviceProp& prop, int rank);

std::pair<int, int> init_multiprocessing(const model_config_t& model_config);
void                init_nccl(const model_config_t& model_config, NcclParam& tensor_para, NcclParam& pipeline_para);

cublasMMWrapper init_cublas_ctx(DataType                        data_type,
                                cudaStream_t&                   stream,
                                Allocator<AllocatorType::CUDA>& allocator,
                                cublasHandle_t&                 cublas_handle,
                                std::mutex&                     cublas_wrapper_mutex,
                                cublasLtHandle_t&               cublaslt_handle,
                                cublasAlgoMap&                  cublas_algo_map
#ifdef SPARSITY_ENABLED
                                ,
                                cusparseLtHandle_t& cusparselt_handle
#endif
);

int read_start_ids(size_t            batch_size,
                   std::vector<int>* v_start_lengths,
                   std::vector<int>* v_start_ids,
                   size_t&           max_input_len,
                   const int         end_id,
                   const int         beam_width,
                   std::string       file_name);

void populate_request(std::unordered_map<std::string, Tensor>& input_tensors,
                      std::unordered_map<std::string, Tensor>& output_tensors,
                      int*&                                    d_input_ids,
                      int*&                                    d_input_lengths,
                      std::vector<int>&                        v_start_lengths,
                      std::vector<int>&                        output_seq_len,
                      std::vector<int>&                        p_prompt_tuning_task_name_ids,
                      int*&                                    d_output_ids,
                      int*&                                    d_sequence_lengths,
                      float*&                                  output_log_probs,
                      float*&                                  d_cum_log_probs,
                      float*&                                  output_context_embeddings,
                      const uint64_t&                          random_seed,
                      const std::string&                       csv_path,
                      const model_config_t&                    model_config,
                      const request_config_t&                  request_config);

void write_output_tensors(std::unordered_map<std::string, Tensor>& output_tensors);

}  // namespace fastertransformer
