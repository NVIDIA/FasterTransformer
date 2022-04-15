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
    const std::string data_type = reader.Get("ft_instance_hyperparameter", "data_type");
    if (data_type == "fp16") {
        return std::make_shared<ParallelGptTritonModel<half>>(
            reader.GetInteger("ft_instance_hyperparameter", "max_seq_len"),
            reader.GetInteger(model_name, "head_num"),
            reader.GetInteger(model_name, "size_per_head"),
            reader.GetInteger(model_name, "inter_size"),
            reader.GetInteger(model_name, "decoder_layers"),
            reader.GetInteger(model_name, "vocab_size"),
            reader.GetInteger(model_name, "start_id"),
            reader.GetInteger(model_name, "end_id"),
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
            reader.GetInteger(model_name, "end_id"),
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
            reader.GetInteger(model_name, "end_id"),
            reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size"),
            reader.Get("ft_instance_hyperparameter", "model_name"),
            reader.Get("ft_instance_hyperparameter", "model_dir"),
            reader.GetInteger("ft_instance_hyperparameter", "int8_mode"),
            reader.GetInteger("ft_instance_hyperparameter", "enable_custom_all_reduce", 0));
    }
    else {
        FT_LOG_ERROR("Unspported data type " + data_type);
        exit(-1);
    }
}

template<typename T>
ParallelGptTritonModel<T>::ParallelGptTritonModel(size_t max_seq_len,
                                                  size_t head_num,
                                                  size_t size_per_head,
                                                  size_t inter_size,
                                                  size_t num_layer,
                                                  size_t vocab_size,
                                                  int start_id,
                                                  int end_id,
                                                  size_t tensor_para_size,
                                                  size_t pipeline_para_size,
                                                  std::string model_name,
                                                  std::string model_dir,
                                                  int int8_mode,
                                                  int enable_custom_all_reduce):
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    start_id_(start_id),
    end_id_(end_id),
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    model_name_(model_name),
    model_dir_(model_dir),
    int8_mode_(int8_mode),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
}

template<typename T>
std::unique_ptr<AbstractTransformerModelInstance>
ParallelGptTritonModel<T>::createModelInstance(int device_id,
                                               int rank,
                                               cudaStream_t stream,
                                               std::pair<std::vector<ncclComm_t>, std::vector<ncclComm_t>> nccl_comms,
                                               std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    const int tensor_para_rank = rank % tensor_para_size_;
    const int pipeline_para_rank = rank / tensor_para_size_;

    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator(
        new ft::Allocator<ft::AllocatorType::CUDA>(device_id));

    allocator->setStream(stream);

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;

    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);

    std::unique_ptr<ft::cublasAlgoMap> cublas_algo_map(new ft::cublasAlgoMap("gemm_config.in"));
    std::unique_ptr<std::mutex> cublas_wrapper_mutex(new std::mutex());
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

    ft::NcclParam tensor_para(tensor_para_rank, tensor_para_size_, nccl_comms.first[device_id]);
    ft::NcclParam pipeline_para(pipeline_para_rank, pipeline_para_size_, nccl_comms.second[device_id]);

    auto gpt = std::make_unique<ft::ParallelGpt<T>>(
        ft::ParallelGpt<T>(0,  // max_batch_size, FT will adjust the buffer automatically.
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
                           0.0f,  // beam_search_diversity_rate_,
                           1,     // top_k_,
                           0.0f,  // top_p_,
                           0,     // random seed, note that all gpus should use same seed
                           1.0f,  // temperature_,
                           1.0f,  // len_penalty_,
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
                           enable_custom_all_reduce_));

    auto weight = std::unique_ptr<ft::ParallelGptWeight<T>>(new ft::ParallelGptWeight<T>(head_num_ * size_per_head_,
                                                                                         inter_size_,
                                                                                         vocab_size_,
                                                                                         num_layer_,
                                                                                         max_seq_len_,
                                                                                         tensor_para_size_,
                                                                                         tensor_para_rank,
                                                                                         pipeline_para_size_,
                                                                                         pipeline_para_rank,
                                                                                         int8_mode_));

    weight->loadModel(model_dir_);
    return std::unique_ptr<ParallelGptTritonModelInstance<T>>(
        new ParallelGptTritonModelInstance<T>(std::move(gpt),
                                              std::move(weight),
                                              std::move(allocator),
                                              std::move(cublas_algo_map),
                                              std::move(cublas_wrapper_mutex),
                                              std::move(cublas_wrapper),
                                              std::move(cuda_device_prop_ptr)));
}

template<typename T>
std::string ParallelGptTritonModel<T>::toString()
{
    std::stringstream ss;
    ss << "Model: "
       << "\nmax_seq_len: " << max_seq_len_ << "\nhead_num: " << head_num_ << "\nsize_per_head: " << size_per_head_
       << "\ninter_size: " << inter_size_ << "\nnum_layer: " << num_layer_ << "\nvocab_size: " << vocab_size_
       << "\nstart_id: " << start_id_ << "\nend_id: " << end_id_ << "\ntensor_para_size: " << tensor_para_size_
       << "\npipeline_para_size: " << pipeline_para_size_ << "\nint8_mode: " << int8_mode_
       << "\nenable_custom_all_reduce: " << enable_custom_all_reduce_ << "\nmodel_name: " << model_name_
       << "\nmodel_dir: " << model_dir_ << std::endl;
    return ss.str();
}

template<typename T>
std::vector<ncclUniqueId> ParallelGptTritonModel<T>::createNcclIds(const uint32_t world_size, bool multi_instances)
{
    std::vector<ncclUniqueId> nccl_ids(tensor_para_size_ + pipeline_para_size_);
    if (multi_instances) {
        if (tensor_para_size_ * pipeline_para_size_ != 1) {
            printf(
                "[ERROR] Multiple Instances currently only support tensor_para_size_ and  pipeline_para_size_ both 1\n");
            ft::FT_CHECK(tensor_para_size_ == 1 && pipeline_para_size_ == 1);
        }
        nccl_ids.resize(2);
    }
    else {
        if (world_size != tensor_para_size_ * pipeline_para_size_) {
            ft::FT_CHECK_WITH_INFO(world_size == tensor_para_size_ * pipeline_para_size_,
                                   "world_size == tensor_para_size_ * pipeline_para_size_ ("
                                       + std::to_string(world_size) + " != " + std::to_string(tensor_para_size_) + "*"
                                       + std::to_string(pipeline_para_size_) + ")");
        }
    }

    for (uint32_t i = 0; i < nccl_ids.size(); i++) {
        NCCLCHECK(ncclGetUniqueId(&nccl_ids[i]));
    }
    return nccl_ids;
}

template<typename T>
std::pair<std::vector<ncclComm_t>, std::vector<ncclComm_t>> ParallelGptTritonModel<T>::createNcclComms(
    std::vector<ncclUniqueId> nccl_ids, const int node_id, bool multi_instances, int instance_id)
{
    const int gpu_count = ft::getDeviceCount();
    std::vector<ncclComm_t> tensor_para_comms(gpu_count);
    std::vector<ncclComm_t> pipeline_para_comms(gpu_count);
    if (multi_instances) {
        ncclUniqueId tensor_para_nccl_uid = nccl_ids[0];
        ncclUniqueId pipeline_para_nccl_uid = nccl_ids[1];
        size_t tensor_para_rank = 0;
        size_t pipeline_para_rank = 0;

        ft::check_cuda_error(cudaSetDevice(instance_id));
        NCCLCHECK(ncclCommInitRank(
            &tensor_para_comms[instance_id], tensor_para_size_, tensor_para_nccl_uid, tensor_para_rank));
        NCCLCHECK(ncclCommInitRank(
            &pipeline_para_comms[instance_id], pipeline_para_size_, pipeline_para_nccl_uid, pipeline_para_rank));
    }
    else {
        NCCLCHECK(ncclGroupStart());
        for (int gid = 0; gid < gpu_count; gid++) {
            int rank = node_id * gpu_count + gid;
            size_t tensor_para_rank = rank % tensor_para_size_;
            size_t pipeline_para_rank = rank / tensor_para_size_;
            ncclUniqueId tensor_para_nccl_uid = nccl_ids[pipeline_para_rank];
            ncclUniqueId pipeline_para_nccl_uid = nccl_ids[pipeline_para_size_ + tensor_para_rank];

            ft::check_cuda_error(cudaSetDevice(gid));
            NCCLCHECK(
                ncclCommInitRank(&tensor_para_comms[gid], tensor_para_size_, tensor_para_nccl_uid, tensor_para_rank));
            NCCLCHECK(ncclCommInitRank(
                &pipeline_para_comms[gid], pipeline_para_size_, pipeline_para_nccl_uid, pipeline_para_rank));
        }
        NCCLCHECK(ncclGroupEnd());
    }
    return std::pair<std::vector<ncclComm_t>, std::vector<ncclComm_t>>(tensor_para_comms, pipeline_para_comms);
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