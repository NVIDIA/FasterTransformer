/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/triton_backend/gptj/GptJTritonModel.h"
#include "3rdparty/INIReader.h"
#include "src/fastertransformer/triton_backend/gptj/GptJTritonModelInstance.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/utils/allocator.h"

namespace ft = fastertransformer;

std::shared_ptr<AbstractTransformerModel> AbstractTransformerModel::createGptJModel(std::string inifile)
{
    INIReader reader = INIReader(inifile);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << inifile << "'\n";
        return nullptr;
    }

    const std::string model_name       = reader.Get("ft_instance_hyperparameter", "model_name");
    const std::string data_type        = reader.Get("ft_instance_hyperparameter", "data_type");
    int               tensor_para_size = reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size");
    std::string       model_dir        = reader.Get("ft_instance_hyperparameter", "model_dir");
    model_dir                          = model_dir + "/" + std::to_string(tensor_para_size) + "-gpu/";

    // Prompt Learning Configurations
    int end_id                   = reader.GetInteger(model_name, "end_id");
    int prompt_learning_start_id = reader.GetInteger(model_name, "prompt_learning_start_id", end_id + 1);
    ft::PromptLearningType prompt_learning_type =
        static_cast<ft::PromptLearningType>(reader.GetInteger(model_name, "prompt_learning_type", 0));

    std::map<std::string, std::pair<int, int>> prompt_learning_table_pair;

    // NOTE: get prompt from configuration files
    const int num_tasks = reader.GetInteger(model_name, "num_tasks", 0);
    for (int task_name_id = 0; task_name_id < num_tasks; task_name_id++) {
        std::string config_task_name = model_name + "_task_" + std::to_string(task_name_id);
        std::string task_name        = reader.Get(config_task_name, "task_name");
        const int   prompt_length    = reader.GetInteger(config_task_name, "prompt_length", 0);
        prompt_learning_table_pair.insert({task_name, {task_name_id, prompt_length}});
    }

    if (data_type == "fp16") {
        return std::make_shared<GptJTritonModel<half>>(
            reader.GetInteger("ft_instance_hyperparameter", "max_seq_len"),
            reader.GetInteger(model_name, "head_num"),
            reader.GetInteger(model_name, "size_per_head"),
            reader.GetInteger(model_name, "inter_size"),
            reader.GetInteger(model_name, "decoder_layers"),
            reader.GetInteger(model_name, "vocab_size"),
            reader.GetInteger(model_name, "rotary_embedding"),
            reader.GetInteger(model_name, "start_id"),
            end_id,
            prompt_learning_start_id,
            prompt_learning_type,
            prompt_learning_table_pair,
            reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "enable_custom_all_reduce", 0),
            model_name,
            model_dir);
    }
    else if (data_type == "fp32") {
        return std::make_shared<GptJTritonModel<float>>(
            reader.GetInteger("ft_instance_hyperparameter", "max_seq_len"),
            reader.GetInteger(model_name, "head_num"),
            reader.GetInteger(model_name, "size_per_head"),
            reader.GetInteger(model_name, "inter_size"),
            reader.GetInteger(model_name, "decoder_layers"),
            reader.GetInteger(model_name, "vocab_size"),
            reader.GetInteger(model_name, "rotary_embedding"),
            reader.GetInteger(model_name, "start_id"),
            end_id,
            prompt_learning_start_id,
            prompt_learning_type,
            prompt_learning_table_pair,
            reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "enable_custom_all_reduce", 0),
            model_name,
            model_dir);
    }
#ifdef ENABLE_BF16
    else if (data_type == "bf16") {
        return std::make_shared<GptJTritonModel<__nv_bfloat16>>(
            reader.GetInteger("ft_instance_hyperparameter", "max_seq_len"),
            reader.GetInteger(model_name, "head_num"),
            reader.GetInteger(model_name, "size_per_head"),
            reader.GetInteger(model_name, "inter_size"),
            reader.GetInteger(model_name, "decoder_layers"),
            reader.GetInteger(model_name, "vocab_size"),
            reader.GetInteger(model_name, "rotary_embedding"),
            reader.GetInteger(model_name, "start_id"),
            end_id,
            prompt_learning_start_id,
            prompt_learning_type,
            prompt_learning_table_pair,
            reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "enable_custom_all_reduce", 0),
            model_name,
            model_dir);
    }
#endif
    else {
        FT_LOG_ERROR("Unsupported data type " + data_type);
        exit(-1);
    }
}

template<typename T>
GptJTritonModel<T>::GptJTritonModel(size_t      tensor_para_size,
                                    size_t      pipeline_para_size,
                                    int         enable_custom_all_reduce,
                                    std::string model_dir):
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    shared_weights_(std::vector<std::shared_ptr<ft::GptJWeight<T>>>(ft::getDeviceCount())),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    model_dir_(model_dir)
{
    INIReader reader = INIReader(model_dir + "/config.ini");
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << model_dir << "/config.ini"
                  << "'\n";
        ft::FT_CHECK(false);
    }

    /* GPTJ Model Config Example
    [gptj]
    model_name=gpt-j-6B
    head_num=16
    size_per_head=256
    inter_size=16384
    num_layer=28
    rotary_embedding_dim=64
    vocab_size=50400
    start_id=50256
    end_id=50256
    prompt_learning_start_id=50257 ; only required by p/prompt tuning
    prompt_learning_type=3
    num_tasks=2

    [task_0]
    task_name=no_prompt
    prompt_length=0

    [task_1]
    task_name=len1_seed100
    prompt_length=1
    */
    model_name_           = reader.Get("gptj", "model_name");
    head_num_             = reader.GetInteger("gptj", "head_num");
    size_per_head_        = reader.GetInteger("gptj", "size_per_head");
    inter_size_           = reader.GetInteger("gptj", "inter_size");
    num_layer_            = reader.GetInteger("gptj", "num_layer");
    vocab_size_           = reader.GetInteger("gptj", "vocab_size");
    rotary_embedding_dim_ = reader.GetInteger("gptj", "rotary_embedding");
    start_id_             = reader.GetInteger("gptj", "start_id");
    end_id_               = reader.GetInteger("gptj", "end_id");

    num_tasks_ = reader.GetInteger("gptj", "num_tasks", 0);

    prompt_learning_start_id_ = reader.GetInteger("gptj", "prompt_learning_start_id", end_id_ + 1);
    prompt_learning_type_ = static_cast<ft::PromptLearningType>(reader.GetInteger("gptj", "prompt_learning_type", 0));

    for (int task_name_id = 0; task_name_id < num_tasks_; task_name_id++) {
        std::string config_task_name = "task_" + std::to_string(task_name_id);
        std::string task_name        = reader.Get(config_task_name, "task_name");
        const int   prompt_length    = reader.GetInteger(config_task_name, "prompt_length", 0);
        prompt_learning_table_pair_.insert({task_name, {task_name_id, prompt_length}});
    }
}

template<typename T>
GptJTritonModel<T>::GptJTritonModel(size_t                                     max_seq_len,
                                    size_t                                     head_num,
                                    size_t                                     size_per_head,
                                    size_t                                     inter_size,
                                    size_t                                     num_layer,
                                    size_t                                     vocab_size,
                                    size_t                                     rotary_embedding_dim,
                                    int                                        start_id,
                                    int                                        end_id,
                                    int                                        prompt_learning_start_id,
                                    ft::PromptLearningType                     prompt_learning_type,
                                    std::map<std::string, std::pair<int, int>> prompt_learning_table_pair,
                                    size_t                                     tensor_para_size,
                                    size_t                                     pipeline_para_size,
                                    int                                        enable_custom_all_reduce,
                                    std::string                                model_name,
                                    std::string                                model_dir):
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    rotary_embedding_dim_(rotary_embedding_dim),
    start_id_(start_id),
    end_id_(end_id),
    prompt_learning_start_id_(prompt_learning_start_id),
    prompt_learning_type_(prompt_learning_type),
    prompt_learning_table_pair_(prompt_learning_table_pair),
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    shared_weights_(std::vector<std::shared_ptr<ft::GptJWeight<T>>>(ft::getDeviceCount())),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    model_name_(model_name),
    model_dir_(model_dir)
{
}

template<typename T>
std::unique_ptr<AbstractTransformerModelInstance>
GptJTritonModel<T>::createModelInstance(int                                                               device_id,
                                        int                                                               rank,
                                        cudaStream_t                                                      stream,
                                        std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
                                        std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    const int comms_rank = device_id % (tensor_para_size_ * pipeline_para_size_);

    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator(
        new ft::Allocator<ft::AllocatorType::CUDA>(device_id));

    allocator->setStream(stream);

    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;

    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);

    std::unique_ptr<ft::cublasAlgoMap>   cublas_algo_map(new ft::cublasAlgoMap("gemm_config.in"));
    std::unique_ptr<std::mutex>          cublas_wrapper_mutex(new std::mutex());
    std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper(new ft::cublasMMWrapper(
        cublas_handle, cublaslt_handle, stream, cublas_algo_map.get(), cublas_wrapper_mutex.get(), allocator.get()));

    std::unique_ptr<cudaDeviceProp> cuda_device_prop_ptr(new cudaDeviceProp);
    ft::check_cuda_error(cudaGetDeviceProperties(cuda_device_prop_ptr.get(), device_id));

    if (std::is_same<T, half>::value) {
        cublas_wrapper->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper->setBF16GemmConfig();
    }
#endif
    else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }

    ft::NcclParam tensor_para   = nccl_params.first[comms_rank];
    ft::NcclParam pipeline_para = nccl_params.second[comms_rank];

    auto gpt =
        std::make_unique<ft::GptJ<T>>(ft::GptJ<T>(0,  // max_batch_size, FT will adjust the buffer automatically.
                                                  0,  // max_seq_len, FT will adjust the buffer automatically.
                                                  0,  // max_input_len, FT will adjust the buffer automatically.
                                                  0,
                                                  head_num_,
                                                  size_per_head_,
                                                  inter_size_,
                                                  num_layer_,
                                                  vocab_size_,
                                                  rotary_embedding_dim_,
                                                  start_id_,
                                                  end_id_,
                                                  prompt_learning_start_id_,  // p/prompt tuning virtual token start id
                                                  prompt_learning_type_,
                                                  0.0f,  // beam_search_diversity_rate_,
                                                  0,     // top_k_,
                                                  0.0f,  // top_p_,
                                                  0,     // random seed, note that all gpus should use same seed
                                                  0.0f,  // temperature_,
                                                  0.0f,  // len_penalty_,
                                                  0.0f,  // repetition_penalty_,
                                                  tensor_para,
                                                  pipeline_para,
                                                  stream,
                                                  cublas_wrapper.get(),
                                                  allocator.get(),
                                                  false,
                                                  cuda_device_prop_ptr.get(),
                                                  custom_all_reduce_comm,
                                                  enable_custom_all_reduce_));

    return std::unique_ptr<GptJTritonModelInstance<T>>(new GptJTritonModelInstance<T>(std::move(gpt),
                                                                                      shared_weights_[device_id],
                                                                                      std::move(allocator),
                                                                                      std::move(cublas_algo_map),
                                                                                      std::move(cublas_wrapper_mutex),
                                                                                      std::move(cublas_wrapper),
                                                                                      std::move(cuda_device_prop_ptr)));
}

template<typename T>
void GptJTritonModel<T>::createSharedWeights(int device_id, int rank)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    const int tensor_para_rank   = rank % tensor_para_size_;
    const int pipeline_para_rank = rank / tensor_para_size_;
    shared_weights_[device_id]   = std::make_shared<ft::GptJWeight<T>>(head_num_ * size_per_head_,
                                                                     inter_size_,
                                                                     vocab_size_,
                                                                     num_layer_,
                                                                     max_seq_len_,  // not needed
                                                                     tensor_para_size_,
                                                                     tensor_para_rank,
                                                                     pipeline_para_size_,
                                                                     pipeline_para_rank,
                                                                     prompt_learning_type_,
                                                                     prompt_learning_table_pair_);
    shared_weights_[device_id]->loadModel(model_dir_);
    return;
}

template<typename T>
std::string GptJTritonModel<T>::toString()
{
    std::stringstream ss;
    ss << "Model: "
       << "\nhead_num: " << head_num_ << "\nsize_per_head: " << size_per_head_ << "\ninter_size: " << inter_size_
       << "\nnum_layer: " << num_layer_ << "\nvocab_size: " << vocab_size_ << "\nstart_id: " << start_id_
       << "\nend_id: " << end_id_ << "\ntensor_para_size: " << tensor_para_size_
       << "\npipeline_para_size: " << pipeline_para_size_ << "\nenable_custom_all_reduce: " << enable_custom_all_reduce_
       << "\nmodel_name: " << model_name_ << "\nmodel_dir: " << model_dir_ << std::endl;
    return ss.str();
}

template<typename T>
void GptJTritonModel<T>::createCustomComms(
    std::vector<std::shared_ptr<ft::AbstractCustomComm>>* custom_all_reduce_comms, int world_size)
{
    using commDataType = typename ft::CustomARCommTypeConverter<T>::Type;
    ft::initCustomAllReduceComm<commDataType>(custom_all_reduce_comms, enable_custom_all_reduce_, world_size);
}

template<typename T>
int GptJTritonModel<T>::getTensorParaSize()
{
    return tensor_para_size_;
}

template<typename T>
int GptJTritonModel<T>::getPipelineParaSize()
{
    return pipeline_para_size_;
}

template struct GptJTritonModel<float>;
template struct GptJTritonModel<half>;
#ifdef ENABLE_BF16
template struct GptJTritonModel<__nv_bfloat16>;
#endif
