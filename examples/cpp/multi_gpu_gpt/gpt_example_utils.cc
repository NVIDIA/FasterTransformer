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
#include "examples/cpp/multi_gpu_gpt/gpt_example_utils.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <fstream>
#include <sstream>

namespace fastertransformer {

int read_start_ids(size_t            batch_size,
                   std::vector<int>* v_start_lengths,
                   std::vector<int>* v_start_ids,
                   size_t&           max_input_len,
                   const int         end_id,
                   const int         beam_width,
                   std::string       file_name)
{
    std::vector<std::vector<int>> tmp_start_ids;
    std::vector<int>              tmp_start_lengths;

    std::ifstream start_id_file(file_name, std::ios::in);
    int           line_num = 0;
    if (start_id_file.is_open()) {
        std::string line;
        while (std::getline(start_id_file, line)) {
            std::stringstream lineStream(line);
            std::string       vals;
            int               i1 = 0;
            std::vector<int>  tmp_vec;
            while (std::getline(lineStream, vals, ',')) {
                tmp_vec.push_back(std::stoi(vals));
                i1++;
            }
            tmp_start_ids.push_back(tmp_vec);
            tmp_start_lengths.push_back(i1);
            line_num++;
        }
        if (batch_size == 0) {
            batch_size = line_num;
        }
    }
    else {
        printf("[WARNING] Cannot open the file '%s'. \n", file_name.c_str());
        max_input_len = 0;
        return 0;
    }

    max_input_len = tmp_start_lengths.data()[0];
    for (uint i = 1; i < (uint)tmp_start_lengths.size(); i++) {
        max_input_len = max_input_len > tmp_start_lengths.data()[i] ? max_input_len : tmp_start_lengths.data()[i];
    }

    while ((int)tmp_start_lengths.size() < batch_size) {
        std::vector<int> padding_ids;
        for (int i = 0; i < max_input_len; i++) {
            padding_ids.push_back(end_id);
        }
        tmp_start_ids.push_back(padding_ids);
        tmp_start_lengths.push_back(max_input_len);
    }

    // Add padding
    for (int i = 0; i < (int)tmp_start_ids.size(); i++) {
        for (int j = (int)tmp_start_ids[i].size(); j < max_input_len; j++) {
            tmp_start_ids[i].push_back(end_id);
        }
    }

    for (int i = 0; i < (int)tmp_start_ids.size(); i++) {
        for (int b = 0; b < beam_width; b++) {
            for (int j = 0; j < (int)tmp_start_ids[i].size(); j++) {
                v_start_ids->push_back(tmp_start_ids[i][j]);
            }
            v_start_lengths->push_back(tmp_start_lengths[i]);
        }
    }
    return batch_size;
}

model_config_t read_model_config(const INIReader& reader)
{
    model_config_t config;

    config.model_name = reader.Get("ft_instance_hyperparameter", "model_name");
    config.model_dir  = std::string(reader.Get("ft_instance_hyperparameter", "model_dir"));
    config.sparse     = static_cast<bool>(reader.GetInteger("ft_instance_hyperparameter", "sparse"));
    config.int8_mode  = reader.GetInteger("ft_instance_hyperparameter", "int8_mode");

    config.tensor_para_size   = reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size");
    config.pipeline_para_size = reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size");

    config.head_num       = reader.GetInteger(config.model_name, "head_num");
    config.size_per_head  = reader.GetInteger(config.model_name, "size_per_head");
    config.vocab_size     = reader.GetInteger(config.model_name, "vocab_size");
    config.decoder_layers = reader.GetInteger(config.model_name, "decoder_layers");
    config.hidden_units   = config.head_num * config.size_per_head;
    config.inter_size     = reader.GetInteger(config.model_name, "inter_size", 4 * config.hidden_units);

    FT_CHECK(config.head_num % config.tensor_para_size == 0);
    FT_CHECK(config.decoder_layers % config.pipeline_para_size == 0);

    // GPT Variants parameters: e.g. OPT or BLOOM.
    const std::string model_variant = std::string(reader.Get(config.model_name, "model_variant", "gpt"));

    if (model_variant == "opt-pre") {
        config.gpt_variants.layernorm_eps              = 1e-5f;
        config.gpt_variants.layernorm_type             = LayerNormType::pre_layernorm;
        config.gpt_variants.activation_type            = ActivationType::Relu;
        config.gpt_variants.has_post_decoder_layernorm = true;
    }
    else if (model_variant == "opt-post") {
        config.gpt_variants.layernorm_eps              = 1e-5f;
        config.gpt_variants.layernorm_type             = LayerNormType::post_layernorm;
        config.gpt_variants.activation_type            = ActivationType::Relu;
        config.gpt_variants.has_post_decoder_layernorm = false;
    }
    else if (model_variant == "bloom-pre") {
        config.gpt_variants.layernorm_eps              = 1e-5f;
        config.gpt_variants.layernorm_type             = LayerNormType::pre_layernorm;
        config.gpt_variants.activation_type            = ActivationType::Gelu;
        config.gpt_variants.has_positional_encoding    = false;
        config.gpt_variants.has_pre_decoder_layernorm  = true;
        config.gpt_variants.has_post_decoder_layernorm = true;
        config.gpt_variants.use_attention_linear_bias  = true;
    }
    else if (model_variant == "bloom-post") {
        config.gpt_variants.layernorm_eps              = 1e-5f;
        config.gpt_variants.layernorm_type             = LayerNormType::post_layernorm;
        config.gpt_variants.activation_type            = ActivationType::Gelu;
        config.gpt_variants.has_positional_encoding    = false;
        config.gpt_variants.has_pre_decoder_layernorm  = true;
        config.gpt_variants.has_post_decoder_layernorm = true;
        config.gpt_variants.use_attention_linear_bias  = true;
    }
    config.gpt_variants.has_adapters = reader.GetBoolean(config.model_name, "has_adapters", false);
    config.gpt_variants.adapter_inter_size =
        reader.GetInteger(config.model_name, "adapter_inter_size", config.inter_size);
    config.gpt_variants.layernorm_eps =
        reader.GetFloat(config.model_name, "layernorm_eps", config.gpt_variants.layernorm_eps);
    config.max_seq_len = config.gpt_variants.has_positional_encoding ?
                             (size_t)reader.GetInteger("ft_instance_hyperparameter", "max_seq_len") :
                             FT_SEQ_LEN_MAX;

    config.start_id = reader.GetInteger(config.model_name, "start_id");
    config.end_id   = reader.GetInteger(config.model_name, "end_id");

    // Prompt Learning Configurations
    config.prompt_learning_start_id =
        reader.GetInteger(config.model_name, "prompt_learning_start_id", config.end_id + 1);
    config.prompt_learning_type =
        static_cast<PromptLearningType>(reader.GetInteger(config.model_name, "prompt_learning_type", 0));
    config.prompt_learning_num_tasks = reader.GetInteger(config.model_name, "num_tasks", 0);

    return config;
}

request_config_t read_request_config(const INIReader& reader)
{
    request_config_t config;

    config.beam_width  = reader.GetInteger("ft_instance_hyperparameter", "beam_width");
    config.top_k       = reader.GetInteger("ft_instance_hyperparameter", "top_k");
    config.top_p       = reader.GetFloat("ft_instance_hyperparameter", "top_p");
    config.temperature = reader.GetFloat("ft_instance_hyperparameter", "temperature");
    config.min_length  = reader.GetInteger("ft_instance_hyperparameter", "min_length", 0);

    config.repetition_penalty = reader.GetFloat("ft_instance_hyperparameter", "repetition_penalty", 1.0f);
    config.presence_penalty   = reader.GetFloat("ft_instance_hyperparameter", "presence_penalty", 0.0f);
    FT_CHECK_WITH_INFO(
        config.repetition_penalty == 1.0f || config.presence_penalty == 0.0f,
        fmtstr("Found ambiguous parameters repetition_penalty (%f) and presence_penalty (%f) "
               "which are mutually exclusive. Please remove one of repetition_penalty or presence_penalty "
               "or set to a default value.",
               config.repetition_penalty,
               config.presence_penalty));

    config.request_batch_size = reader.GetInteger("request", "request_batch_size");
    // The length of tokens we hope this model to generate
    config.request_output_len  = reader.GetInteger("request", "request_output_len");
    config.is_return_log_probs = reader.GetBoolean("request", "return_log_probs", false);
    // Whether to include input contexts in computing the cumulative log probabilities.
    config.is_return_context_cum_log_probs = reader.GetBoolean("request", "context_log_probs", false);
    config.is_return_context_embeddings    = reader.GetBoolean("request", "context_embeddings", false);
    if (config.is_return_log_probs && !config.is_return_context_cum_log_probs) {
        FT_LOG_WARNING("context_log_probs will be ignored since return_log_probs is disabled.");
    }
    config.remove_padding = reader.GetBoolean("request", "remove_padding", false);
    config.memory_len     = reader.GetInteger("request", "memory_len", 0);

    config.beam_search_diversity_rate = reader.GetFloat("ft_instance_hyperparameter", "beam_search_diversity_rate");
    config.len_penalty                = reader.GetFloat("ft_instance_hyperparameter", "len_penalty");
    config.shared_contexts_ratio      = reader.GetFloat("ft_instance_hyperparameter", "shared_contexts_ratio", 1.0f);

    return config;
}

std::map<std::string, std::pair<int, int>> init_prompt_tuning_map(const INIReader& reader, const model_config_t& config)
{
    // NOTE：specify task names, take name id, prompt length in order to load those prompt learning tables.
    // for example:
    // std::map<std::string, std::pair<int, int>> p_prompt_tuning_table_pair_{{"sentiment", {0, 10}},
    //                                                                        {"intent_and_slot", {1, 10}},
    //                                                                        {"squad", {2, 16}}};
    std::map<std::string, std::pair<int, int>> p_prompt_tuning_map;
    // NOTE: get prompt table pairs from configuration files
    for (int task_name_id = 0; task_name_id < config.prompt_learning_num_tasks; task_name_id++) {
        std::string config_task_name = config.model_name + "_task_" + std::to_string(task_name_id);
        std::string task_name        = reader.Get(config_task_name, "task_name");
        const int   prompt_length    = reader.GetInteger(config_task_name, "prompt_length", 0);
        p_prompt_tuning_map.insert({task_name, {task_name_id, prompt_length}});
    }

    return p_prompt_tuning_map;
}

std::pair<int, int> init_multiprocessing(const model_config_t& model_config)
{
    // Prepare the parallelism parameters
    int rank       = mpi::getCommWorldRank();
    int world_size = mpi::getCommWorldSize();
    if (rank == 0) {
        printf("Total ranks: %d.\n", world_size);
    }

    if (model_config.tensor_para_size * model_config.pipeline_para_size != world_size) {
        printf("[ERROR] tensor_para_size * pipeline_para_size should equal to world_size \n");
        exit(-1);
    }

    const int layers_per_group = model_config.decoder_layers / model_config.pipeline_para_size;
    if (layers_per_group * model_config.pipeline_para_size != (int)model_config.decoder_layers) {
        printf("[ERROR] layers_per_group (%d) * pipeline_para_size (%d) should equal to decoder_layers (%ld) \n",
               layers_per_group,
               model_config.pipeline_para_size,
               model_config.decoder_layers);
        exit(-1);
    }

    return {rank, world_size};
}

Allocator<AllocatorType::CUDA> init_cuda_ctx(cudaStream_t& stream, cudaDeviceProp& prop, int rank)
{
    int device, device_count;
    check_cuda_error(cudaGetDeviceCount(&device_count));
    check_cuda_error(cudaSetDevice(rank % device_count));
    check_cuda_error(cudaGetDevice(&device));
    printf("P%d is running with %d GPU.\n", rank, device);

    check_cuda_error(cudaGetDeviceProperties(&prop, device));
    printf("Device %s\n", prop.name);
    ft_nvtx::setDeviceDomain(getDevice());

    return Allocator<AllocatorType::CUDA>(getDevice());
}

void init_nccl(const model_config_t& model_config, NcclParam& tensor_para, NcclParam& pipeline_para)
{
    // assume gpu_num = k * n,
    // tensor parallelism group size is n
    // pipeline parallelism group size is k
    ftNcclInitialize(tensor_para, pipeline_para, model_config.tensor_para_size, model_config.pipeline_para_size);
}
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
)
{
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);
#ifdef SPARSITY_ENABLED
    CHECK_CUSPARSE(cusparseLtInit(&cusparselt_handle));
    cublas_algo_map = cublasAlgoMap(GEMM_CONFIG, SPGEMM_CONFIG);
#else
    cublas_algo_map = cublasAlgoMap(GEMM_CONFIG);
#endif

#ifdef SPARSITY_ENABLED
    cublasMMWrapper cublas_wrapper = cublasMMWrapper(
        cublas_handle, cublaslt_handle, cusparselt_handle, stream, &cublas_algo_map, &cublas_wrapper_mutex, &allocator);
#else
    cublasMMWrapper cublas_wrapper =
        cublasMMWrapper(cublas_handle, cublaslt_handle, stream, &cublas_algo_map, &cublas_wrapper_mutex, &allocator);
#endif

    if (data_type == DataType::TYPE_FP16) {
        cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }
#ifdef ENABLE_BF16
    else if (data_type == DataType::TYPE_BF16) {
        cublas_wrapper.setBF16GemmConfig();
    }
#endif
    else if (data_type == DataType::TYPE_FP32) {
        cublas_wrapper.setFP32GemmConfig();
    }

    return cublas_wrapper;
}

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
                      const request_config_t&                  request_config)
{
    // Read ids of request from file.
    size_t     max_input_len      = 0;
    auto       request_batch_size = request_config.request_batch_size;
    const auto beam_width         = request_config.beam_width;
    const auto request_output_len = request_config.request_output_len;

    std::vector<int> v_start_ids;
    request_batch_size = read_start_ids(
        request_batch_size, &v_start_lengths, &v_start_ids, max_input_len, model_config.end_id, 1, csv_path);

    if (max_input_len > 0) {
        // conditional case.
        deviceMalloc(&d_input_ids, request_batch_size * max_input_len, false);
        deviceMalloc(&d_input_lengths, request_batch_size, false);

        cudaH2Dcpy(d_input_ids, v_start_ids.data(), request_batch_size * max_input_len);
        cudaH2Dcpy(d_input_lengths, v_start_lengths.data(), request_batch_size);
    }
    const size_t total_output_len = max_input_len + request_output_len;

    output_seq_len = std::vector<int>(request_batch_size, total_output_len);

    input_tensors.insert({"input_ids", {MEMORY_GPU, TYPE_INT32, {request_batch_size, max_input_len}, d_input_ids}});
    input_tensors.insert({"input_lengths", {MEMORY_GPU, TYPE_INT32, {request_batch_size}, d_input_lengths}});
    input_tensors.insert({"input_lengths_h", {MEMORY_CPU, TYPE_INT32, {request_batch_size}, v_start_lengths.data()}});
    input_tensors.insert({"output_seq_len", {MEMORY_CPU, TYPE_UINT32, {request_batch_size}, output_seq_len.data()}});
    input_tensors.insert({"temperature", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.temperature}});
    input_tensors.insert({"len_penalty", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.len_penalty}});
    if (request_config.repetition_penalty != 1.0f) {
        input_tensors.insert({"repetition_penalty", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.repetition_penalty}});
    }
    if (request_config.presence_penalty != 0.0f) {
        input_tensors.insert({"presence_penalty", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.presence_penalty}});
    }
    input_tensors.insert({"min_length", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.min_length}});
    input_tensors.insert({"random_seed", {MEMORY_CPU, TYPE_UINT64, {1}, &random_seed}});

    if (request_config.top_k == 0 && request_config.top_p == 0.0f) {
        FT_CHECK(beam_width > 1);
        input_tensors.insert(
            {"beam_search_diversity_rate", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.beam_search_diversity_rate}});
    }
    else if (request_config.top_p != 0.0f) {
        input_tensors.insert({"runtime_top_p", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.top_p}});
    }
    else if (request_config.top_k != 0) {
        input_tensors.insert({"runtime_top_k", {MEMORY_CPU, TYPE_UINT32, {1}, &request_config.top_k}});
    }

    // NOTE: task_name_ids for each sequence in one batch
    // Each sequence can have different prompt learning task ids
    p_prompt_tuning_task_name_ids = std::vector<int>(request_batch_size, 0);
    if (model_config.prompt_learning_num_tasks > 0) {
        input_tensors.insert({"prompt_learning_task_name_ids",
                              {MEMORY_CPU, TYPE_INT32, {request_batch_size}, p_prompt_tuning_task_name_ids.data()}});
    }

    if (request_config.memory_len > 0) {
        input_tensors.insert({"memory_len", {MEMORY_CPU, TYPE_UINT32, {1}, &request_config.memory_len}});
    }
    if (request_config.is_return_log_probs) {
        input_tensors.insert({"is_return_context_cum_log_probs",
                              {MEMORY_CPU, TYPE_BOOL, {1}, &request_config.is_return_context_cum_log_probs}});
    }
    if (request_config.is_return_context_embeddings) {
        input_tensors.insert({"is_return_context_embeddings",
                              {MEMORY_CPU, TYPE_BOOL, {1}, &request_config.is_return_context_embeddings}});
    }

    deviceMalloc(&d_output_ids, request_batch_size * request_config.beam_width * total_output_len, false);
    deviceMalloc(&d_sequence_lengths, request_batch_size * request_config.beam_width, false);

    output_tensors.insert(
        {"output_ids", {MEMORY_GPU, TYPE_INT32, {request_batch_size, beam_width, total_output_len}, d_output_ids}});
    output_tensors.insert(
        {"sequence_length", {MEMORY_GPU, TYPE_INT32, {request_batch_size, beam_width}, d_sequence_lengths}});

    if (request_config.is_return_log_probs) {
        deviceMalloc(&output_log_probs, request_batch_size * beam_width * request_output_len);
        deviceMalloc(&d_cum_log_probs, request_batch_size * beam_width);
        output_tensors.insert(
            {"output_log_probs",
             {MEMORY_GPU, TYPE_FP32, {request_batch_size, beam_width, request_output_len}, output_log_probs}});
        output_tensors.insert(
            {"cum_log_probs", {MEMORY_GPU, TYPE_FP32, {request_batch_size, beam_width}, d_cum_log_probs}});
    }

    if (request_config.is_return_context_embeddings) {
        deviceMalloc(&output_context_embeddings, request_batch_size * beam_width * model_config.hidden_units);
        output_tensors.insert({"context_embeddings",
                               {MEMORY_GPU,
                                TYPE_FP32,
                                {request_batch_size, beam_width, model_config.hidden_units},
                                output_context_embeddings}});
    }
}

void write_output_tensors(std::unordered_map<std::string, Tensor>& output_tensors)
{
    std::string file_name = "out";
    auto        out_file  = std::ofstream(file_name, std::ios::out);
    if (!out_file.is_open()) {
        printf("[WARNING] Cannot write results into output file %s\n", file_name.c_str());
    }
    else {
        size_t           out_count        = output_tensors.at("output_ids").size();
        const auto       total_output_len = output_tensors.at("output_ids").shape[2];
        std::vector<int> h_buf(out_count);
        cudaD2Hcpy(h_buf.data(), output_tensors.at("output_ids").getPtr<int>(), out_count);

        std::cout << "Writing " << out_count << " elements\n";
        int zeroCount = 0;
        for (size_t i = 0; i < out_count; i++) {
            if (h_buf[i] == 0) {
                zeroCount++;
            }
            out_file << h_buf[i] << " ";
            if ((i + 1) % (total_output_len) == 0) {
                out_file << std::endl;
            }
            if (i < 10) {
                printf("%5d ", h_buf[i]);
            }
            if ((i + 1) % (total_output_len) == 0 && i < 10) {
                std::cout << std::endl;
            }
        }
        std::cout << std::endl << "zeroCount = " << zeroCount << std::endl;
    }
    out_file.close();

    if (output_tensors.count("cum_log_probs")) {
        std::string   logprob_fname = "logprob.out";
        std::ofstream logprob_file  = std::ofstream("logprob.out", std::ios::out);
        if (!logprob_file.is_open()) {
            printf("[WARNING] Cannot write results into output file %s\n", logprob_fname.c_str());
        }
        else {
            size_t cum_log_probs_size = output_tensors.at("cum_log_probs").size();
            printf("[INFO] Writing %ld elements (log probs)\n", cum_log_probs_size);
            std::vector<float> h_buf(cum_log_probs_size);
            cudaD2Hcpy(h_buf.data(), output_tensors.at("cum_log_probs").getPtr<float>(), cum_log_probs_size);
            for (size_t i = 0; i < cum_log_probs_size; i++) {
                logprob_file << h_buf[i] << std::endl;
                if (i < 10) {
                    printf(" %10.6f\n", h_buf[i]);
                }
            }
        }
        logprob_file.close();
    }

    if (output_tensors.count("embeddings")) {
        output_tensors.at("context_embeddings").saveNpy("ctx_emb.npy");
    }
}

}  // namespace fastertransformer
