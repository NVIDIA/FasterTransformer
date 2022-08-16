/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "3rdparty/INIReader.h"

#include "src/fastertransformer/triton_backend/multi_gpu_gpt/ParallelGptTritonModel.h"
#include "src/fastertransformer/triton_backend/multi_gpu_gpt/ParallelGptTritonModelInstance.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/utils/allocator.h"

namespace ft = fastertransformer;

std::shared_ptr<AbstractTransformerModel> AbstractTransformerModel::createGptModel(std::string inifile)
{
    INIReader reader = INIReader(inifile);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << inifile << "'\n";
        return nullptr;
    }

    const std::string model_name = reader.Get("ft_instance_hyperparameter", "model_name");
    const std::string data_type  = reader.Get("ft_instance_hyperparameter", "data_type");

    // gpt variant parameters
    ft::gptVariantParams gpt_variant_params{};
    std::string          model_variant = reader.Get(model_name, "model_variant", "gpt");

    if (model_variant == "opt-pre") {
        gpt_variant_params.layernorm_eps              = 1e-5f;
        gpt_variant_params.layernorm_type             = ft::LayerNormType::pre_layernorm;
        gpt_variant_params.activation_type            = ft::ActivationType::Relu;
        gpt_variant_params.has_post_decoder_layernorm = false;
    }
    else if (model_variant == "opt-post") {
        gpt_variant_params.layernorm_eps              = 1e-5f;
        gpt_variant_params.layernorm_type             = ft::LayerNormType::post_layernorm;
        gpt_variant_params.activation_type            = ft::ActivationType::Relu;
        gpt_variant_params.has_post_decoder_layernorm = false;
    }

    gpt_variant_params.has_adapters = reader.GetBoolean(model_name, "has_adapters", false);

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
        return std::make_shared<ParallelGptTritonModel<half>>(
            reader.GetInteger("ft_instance_hyperparameter", "max_seq_len"),
            reader.GetInteger(model_name, "head_num"),
            reader.GetInteger(model_name, "size_per_head"),
            reader.GetInteger(model_name, "inter_size"),
            reader.GetInteger(model_name, "decoder_layers"),
            reader.GetInteger(model_name, "vocab_size"),
            reader.GetInteger(model_name, "start_id"),
            end_id,
            prompt_learning_start_id,
            prompt_learning_type,
            prompt_learning_table_pair,
            gpt_variant_params,
            reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size"),
            reader.Get("ft_instance_hyperparameter", "model_name"),
            reader.Get("ft_instance_hyperparameter", "model_dir"),
            reader.GetInteger("ft_instance_hyperparameter", "int8_mode"),
            reader.GetInteger("ft_instance_hyperparameter", "enable_custom_all_reduce", 0));
    }
#ifdef ENABLE_BF16
    else if (data_type == "bf16") {
        return std::make_shared<ParallelGptTritonModel<__nv_bfloat16>>(
            reader.GetInteger("ft_instance_hyperparameter", "max_seq_len"),
            reader.GetInteger(model_name, "head_num"),
            reader.GetInteger(model_name, "size_per_head"),
            reader.GetInteger(model_name, "inter_size"),
            reader.GetInteger(model_name, "decoder_layers"),
            reader.GetInteger(model_name, "vocab_size"),
            reader.GetInteger(model_name, "start_id"),
            end_id,
            prompt_learning_start_id,
            prompt_learning_type,
            prompt_learning_table_pair,
            gpt_variant_params,
            reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size"),
            reader.Get("ft_instance_hyperparameter", "model_name"),
            reader.Get("ft_instance_hyperparameter", "model_dir"),
            reader.GetInteger("ft_instance_hyperparameter", "int8_mode"),
            reader.GetInteger("ft_instance_hyperparameter", "enable_custom_all_reduce", 0));
    }
#endif
    else if (data_type == "fp32") {
        return std::make_shared<ParallelGptTritonModel<float>>(
            reader.GetInteger("ft_instance_hyperparameter", "max_seq_len"),
            reader.GetInteger(model_name, "head_num"),
            reader.GetInteger(model_name, "size_per_head"),
            reader.GetInteger(model_name, "inter_size"),
            reader.GetInteger(model_name, "decoder_layers"),
            reader.GetInteger(model_name, "vocab_size"),
            reader.GetInteger(model_name, "start_id"),
            end_id,
            prompt_learning_start_id,
            prompt_learning_type,
            prompt_learning_table_pair,
            gpt_variant_params,
            reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size"),
            reader.Get("ft_instance_hyperparameter", "model_name"),
            reader.Get("ft_instance_hyperparameter", "model_dir"),
            reader.GetInteger("ft_instance_hyperparameter", "int8_mode"),
            reader.GetInteger("ft_instance_hyperparameter", "enable_custom_all_reduce", 0));
    }
    else {
        FT_LOG_ERROR("Unsupported data type " + data_type);
        exit(-1);
    }
}

template<typename T>
ParallelGptTritonModel<T>::ParallelGptTritonModel(size_t      tensor_para_size,
                                                  size_t      pipeline_para_size,
                                                  int         enable_custom_all_reduce,
                                                  std::string model_dir,
                                                  int         int8_mode):
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    shared_weights_(std::vector<std::shared_ptr<ft::ParallelGptWeight<T>>>(ft::getDeviceCount())),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    model_dir_(model_dir),
    int8_mode_(int8_mode)
{
    INIReader reader = INIReader(model_dir + "/config.ini");
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << model_dir << "/config.ini"
                  << "'\n";
        ft::FT_CHECK(false);
    }

    /* GPT Configuration File Example
    [gpt]
    model_name=gpt
    max_pos_seq_len=2048 ;for position embedding tables
    head_num=12
    size_per_head=64
    inter_size=3072
    num_layer=12
    layernorm_eps=1e-6 # optional for the default gpt
    layernorm_type=pre_layernorm # optional for the default gpt
    activation_type=Gelu # optional for the default gpt
    has_post_decoder_layernorm=1 # optional for the default gpt
    vocab_size=50257
    start_id=50256
    end_id=50256
    prompt_learning_start_id=50257
    prompt_learning_type=3
    num_tasks=3

    [task_0]
    task_name=sentiment
    prompt_length=10

    [task_1]
    task_name=intent_and_slot
    prompt_length=10

    [task_2]
    task_name=squad
    prompt_length=16
    */

    model_name_    = reader.Get("gpt", "model_name");
    max_seq_len_   = reader.GetInteger("gpt", "max_pos_seq_len");
    head_num_      = reader.GetInteger("gpt", "head_num");
    size_per_head_ = reader.GetInteger("gpt", "size_per_head");
    inter_size_    = reader.GetInteger("gpt", "inter_size");
    num_layer_     = reader.GetInteger("gpt", "num_layer");
    vocab_size_    = reader.GetInteger("gpt", "vocab_size");
    /* Meta Opt Examples
    layernorm_eps=1e-5
    layernorm_type=pre_layernorm
    activation_type=Relu
    has_post_decoder_layernorm=0
    */
    gpt_variant_params_.layernorm_eps   = reader.GetFloat("gpt", "layernorm_eps", 1e-6f);
    gpt_variant_params_.layernorm_type  = ft::getLayerNormType(reader.Get("gpt", "layernorm_type", "pre_layernorm"));
    gpt_variant_params_.activation_type = ft::getActivationType(reader.Get("gpt", "activation_type", "Gelu"));
    gpt_variant_params_.has_post_decoder_layernorm = reader.GetBoolean("gpt", "has_post_decoder_layernorm", "1");
    /* Megatron GPT Adapter Examples
    has_adapters=True
    adapter_inter_size=1024
    */
    gpt_variant_params_.has_adapters       = reader.GetBoolean("gpt", "has_adapters", false);
    gpt_variant_params_.adapter_inter_size = reader.GetInteger("gpt", "adapter_inter_size", inter_size_);
    start_id_                              = reader.GetInteger("gpt", "start_id");
    end_id_                                = reader.GetInteger("gpt", "end_id");

    num_tasks_                = reader.GetInteger("gpt", "num_tasks", 0);
    prompt_learning_start_id_ = reader.GetInteger("gpt", "prompt_learning_start_id", end_id_ + 1);
    prompt_learning_type_ = static_cast<ft::PromptLearningType>(reader.GetInteger("gpt", "prompt_learning_type", 0));

    for (int task_name_id = 0; task_name_id < num_tasks_; task_name_id++) {
        std::string config_task_name = "task_" + std::to_string(task_name_id);
        std::string task_name        = reader.Get(config_task_name, "task_name");
        const int   prompt_length    = reader.GetInteger(config_task_name, "prompt_length", 0);
        prompt_learning_table_pair_.insert({task_name, {task_name_id, prompt_length}});
    }
}

template<typename T>
ParallelGptTritonModel<T>::ParallelGptTritonModel(size_t                                     max_seq_len,
                                                  size_t                                     head_num,
                                                  size_t                                     size_per_head,
                                                  size_t                                     inter_size,
                                                  size_t                                     num_layer,
                                                  size_t                                     vocab_size,
                                                  int                                        start_id,
                                                  int                                        end_id,
                                                  int                                        prompt_learning_start_id,
                                                  ft::PromptLearningType                     prompt_learning_type,
                                                  std::map<std::string, std::pair<int, int>> prompt_learning_table_pair,
                                                  ft::gptVariantParams                       gpt_variant_params,
                                                  size_t                                     tensor_para_size,
                                                  size_t                                     pipeline_para_size,
                                                  std::string                                model_name,
                                                  std::string                                model_dir,
                                                  int                                        int8_mode,
                                                  int                                        enable_custom_all_reduce):
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    start_id_(start_id),
    end_id_(end_id),
    prompt_learning_start_id_(prompt_learning_start_id),
    prompt_learning_type_(prompt_learning_type),
    prompt_learning_table_pair_(prompt_learning_table_pair),
    gpt_variant_params_(gpt_variant_params),
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    shared_weights_(std::vector<std::shared_ptr<ft::ParallelGptWeight<T>>>(ft::getDeviceCount())),
    model_name_(model_name),
    model_dir_(model_dir),
    int8_mode_(int8_mode),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
}

template<typename T>
std::unique_ptr<AbstractTransformerModelInstance> ParallelGptTritonModel<T>::createModelInstance(
    int                                                               device_id,
    int                                                               rank,
    cudaStream_t                                                      stream,
    std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
    std::shared_ptr<ft::AbstractCustomComm>                           custom_all_reduce_comm)
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
        std::make_unique<ft::ParallelGpt<T>>(0,  // max_batch_size, FT will adjust the buffer automatically.
                                             0,  // max_seq_len, FT will adjust the buffer automatically.
                                             0,  // max_input_len, FT will adjust the buffer automatically.
                                             0,
                                             head_num_,
                                             size_per_head_,
                                             inter_size_,
                                             num_layer_,
                                             vocab_size_,
                                             start_id_,
                                             end_id_,
                                             prompt_learning_start_id_,  // p/prompt tuning virtual token start id
                                             prompt_learning_type_,
                                             gpt_variant_params_,
                                             0.0f,  // beam_search_diversity_rate_,
                                             1,     // top_k_,
                                             0.0f,  // top_p_,
                                             0,     // random seed, note that all gpus should use same seed
                                             1.0f,  // temperature_,
                                             0.0f,  // len_penalty_,
                                             1.0f,  // repetition_penalty_,
                                             tensor_para,
                                             pipeline_para,
                                             stream,
                                             cublas_wrapper.get(),
                                             allocator.get(),
                                             false,
                                             cuda_device_prop_ptr.get(),
                                             false,
                                             int8_mode_,
                                             custom_all_reduce_comm,
                                             enable_custom_all_reduce_);

    return std::unique_ptr<ParallelGptTritonModelInstance<T>>(
        new ParallelGptTritonModelInstance<T>(std::move(gpt),
                                              shared_weights_[device_id],
                                              std::move(allocator),
                                              std::move(cublas_algo_map),
                                              std::move(cublas_wrapper_mutex),
                                              std::move(cublas_wrapper),
                                              std::move(cuda_device_prop_ptr)));
}

template<typename T>
void ParallelGptTritonModel<T>::createSharedWeights(int device_id, int rank)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    const int tensor_para_rank   = rank % tensor_para_size_;
    const int pipeline_para_rank = rank / tensor_para_size_;
    shared_weights_[device_id]   = std::make_shared<ft::ParallelGptWeight<T>>(head_num_ * size_per_head_,
                                                                            inter_size_,
                                                                            vocab_size_,
                                                                            num_layer_,
                                                                            max_seq_len_,
                                                                            tensor_para_size_,
                                                                            tensor_para_rank,
                                                                            pipeline_para_size_,
                                                                            pipeline_para_rank,
                                                                            int8_mode_,
                                                                            prompt_learning_type_,
                                                                            prompt_learning_table_pair_,
                                                                            gpt_variant_params_);
    shared_weights_[device_id]->loadModel(model_dir_);
    return;
}

template<typename T>
std::string ParallelGptTritonModel<T>::toString()
{
    std::stringstream ss;
    ss << "Model: "
       << "\nmax_seq_len: " << max_seq_len_ << "\nhead_num: " << head_num_ << "\nsize_per_head: " << size_per_head_
       << "\ninter_size: " << inter_size_ << "\nnum_layer: " << num_layer_ << "\nvocab_size: " << vocab_size_
       << "\nlayernorm_eps" << gpt_variant_params_.layernorm_eps << "\nlayernorm_type"
       << static_cast<int>(gpt_variant_params_.layernorm_type) << "\nactivation_type"
       << static_cast<int>(gpt_variant_params_.activation_type) << "\nhas_post_decoder_layernorm"
       << gpt_variant_params_.has_post_decoder_layernorm << "\nstart_id: " << start_id_ << "\nend_id: " << end_id_
       << "\ntensor_para_size: " << tensor_para_size_ << "\npipeline_para_size: " << pipeline_para_size_
       << "\nint8_mode: " << int8_mode_ << "\nenable_custom_all_reduce: " << enable_custom_all_reduce_
       << "\nmodel_name: " << model_name_ << "\nmodel_dir: " << model_dir_ << std::endl;
    return ss.str();
}

template<typename T>
void ParallelGptTritonModel<T>::createCustomComms(
    std::vector<std::shared_ptr<ft::AbstractCustomComm>>* custom_all_reduce_comms, int world_size)
{
    using commDataType = typename ft::CustomARCommTypeConverter<T>::Type;
    ft::initCustomAllReduceComm<commDataType>(custom_all_reduce_comms, enable_custom_all_reduce_, world_size);
}

template<typename T>
int ParallelGptTritonModel<T>::getTensorParaSize()
{
    return tensor_para_size_;
}

template<typename T>
int ParallelGptTritonModel<T>::getPipelineParaSize()
{
    return pipeline_para_size_;
}

template struct ParallelGptTritonModel<float>;
template struct ParallelGptTritonModel<half>;
#ifdef ENABLE_BF16
template struct ParallelGptTritonModel<__nv_bfloat16>;
#endif
