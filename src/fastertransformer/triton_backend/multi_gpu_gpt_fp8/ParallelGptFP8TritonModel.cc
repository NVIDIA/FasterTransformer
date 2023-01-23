/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/triton_backend/multi_gpu_gpt_fp8/ParallelGptFP8TritonModel.h"
#include "src/fastertransformer/triton_backend/multi_gpu_gpt_fp8/ParallelGptFP8TritonModelInstance.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/utils/allocator.h"

namespace ft = fastertransformer;

std::shared_ptr<AbstractTransformerModel> AbstractTransformerModel::createGptFP8Model(std::string inifile)
{
    INIReader reader = INIReader(inifile);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << inifile << "'\n";
        return nullptr;
    }

    const std::string model_name = reader.Get("ft_instance_hyperparameter", "model_name");

    return std::make_shared<ParallelGptFP8TritonModel<__nv_fp8_e4m3, __nv_bfloat16>>(
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
        reader.GetInteger("ft_instance_hyperparameter", "fp8_mode", 0),
        reader.GetInteger("ft_instance_hyperparameter", "enable_custom_all_reduce", 0));
}

template<typename T1, typename T2>
ParallelGptFP8TritonModel<T1, T2>::ParallelGptFP8TritonModel(size_t      tensor_para_size,
                                                             size_t      pipeline_para_size,
                                                             int         enable_custom_all_reduce,
                                                             std::string model_dir,
                                                             int         fp8_mode):
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    shared_weights_(std::vector<std::shared_ptr<ft::GptFP8Weight<T1, T2>>>(ft::getDeviceCount())),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    model_dir_(model_dir),
    fp8_mode_(fp8_mode)
{
    INIReader reader = INIReader(model_dir + "/config.ini");
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << model_dir << "/config.ini"
                  << "'\n";
        ft::FT_CHECK(false);
    }

    model_name_    = reader.Get("gpt", "model_name");
    max_seq_len_   = reader.GetInteger("gpt", "max_pos_seq_len");
    head_num_      = reader.GetInteger("gpt", "head_num");
    size_per_head_ = reader.GetInteger("gpt", "size_per_head");
    inter_size_    = reader.GetInteger("gpt", "inter_size");
    num_layer_     = reader.GetInteger("gpt", "num_layer");
    vocab_size_    = reader.GetInteger("gpt", "vocab_size");
    start_id_      = reader.GetInteger("gpt", "start_id");
    end_id_        = reader.GetInteger("gpt", "end_id");
}

template<typename T1, typename T2>
ParallelGptFP8TritonModel<T1, T2>::ParallelGptFP8TritonModel(size_t      max_seq_len,
                                                             size_t      head_num,
                                                             size_t      size_per_head,
                                                             size_t      inter_size,
                                                             size_t      num_layer,
                                                             size_t      vocab_size,
                                                             int         start_id,
                                                             int         end_id,
                                                             size_t      tensor_para_size,
                                                             size_t      pipeline_para_size,
                                                             std::string model_name,
                                                             std::string model_dir,
                                                             int         fp8_mode,
                                                             int         enable_custom_all_reduce):
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
    shared_weights_(std::vector<std::shared_ptr<ft::GptFP8Weight<T1, T2>>>(ft::getDeviceCount())),
    model_name_(model_name),
    model_dir_(model_dir),
    fp8_mode_(fp8_mode),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
}

template<typename T1, typename T2>
std::unique_ptr<AbstractTransformerModelInstance> ParallelGptFP8TritonModel<T1, T2>::createModelInstance(
    int                                                               device_id,
    int                                                               rank,
    cudaStream_t                                                      stream,
    std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
    std::shared_ptr<ft::AbstractCustomComm>                           custom_all_reduce_comm)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    const int comms_rank         = device_id % (tensor_para_size_ * pipeline_para_size_);
    const int tensor_para_rank   = rank % tensor_para_size_;
    const int pipeline_para_rank = rank / tensor_para_size_;

    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator(
        new ft::Allocator<ft::AllocatorType::CUDA>(device_id));

    allocator->setStream(stream);

    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;

    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);

    std::unique_ptr<ft::cublasAlgoMap>      cublas_algo_map(new ft::cublasAlgoMap("gemm_config.in"));
    std::unique_ptr<std::mutex>             cublas_wrapper_mutex(new std::mutex());
    std::unique_ptr<ft::cublasFP8MMWrapper> cublas_wrapper(new ft::cublasFP8MMWrapper(
        cublas_handle, cublaslt_handle, stream, cublas_algo_map.get(), cublas_wrapper_mutex.get(), allocator.get()));

    std::unique_ptr<cudaDeviceProp> cuda_device_prop_ptr(new cudaDeviceProp);
    ft::check_cuda_error(cudaGetDeviceProperties(cuda_device_prop_ptr.get(), device_id));

    cublas_wrapper->setGemmConfig(CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F);

    ft::NcclParam tensor_para   = nccl_params.first[comms_rank];
    ft::NcclParam pipeline_para = nccl_params.second[comms_rank];

    auto gpt = std::make_unique<ft::GptFP8<T1, T2>>(0,
                                                    head_num_,
                                                    size_per_head_,
                                                    inter_size_,
                                                    num_layer_,
                                                    vocab_size_,
                                                    start_id_,
                                                    end_id_,
                                                    tensor_para,
                                                    pipeline_para,
                                                    stream,
                                                    cublas_wrapper.get(),
                                                    allocator.get(),
                                                    false,
                                                    cuda_device_prop_ptr.get(),
                                                    false);

    return std::unique_ptr<ParallelGptFP8TritonModelInstance<T1, T2>>(
        new ParallelGptFP8TritonModelInstance<T1, T2>(std::move(gpt),
                                                      shared_weights_[device_id],
                                                      std::move(allocator),
                                                      std::move(cublas_algo_map),
                                                      std::move(cublas_wrapper_mutex),
                                                      std::move(cublas_wrapper),
                                                      std::move(cuda_device_prop_ptr)));
}

template<typename T1, typename T2>
void ParallelGptFP8TritonModel<T1, T2>::createSharedWeights(int device_id, int rank)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    const int tensor_para_rank   = rank % tensor_para_size_;
    const int pipeline_para_rank = rank / tensor_para_size_;
    shared_weights_[device_id]   = std::make_shared<ft::GptFP8Weight<T1, T2>>(head_num_ * size_per_head_,
                                                                            inter_size_,
                                                                            vocab_size_,
                                                                            num_layer_,
                                                                            max_seq_len_,
                                                                            tensor_para_size_,
                                                                            tensor_para_rank,
                                                                            pipeline_para_size_,
                                                                            pipeline_para_rank);
    shared_weights_[device_id]->loadModel(model_dir_);
    shared_weights_[device_id]->transposeWeight();
    return;
}

template<typename T1, typename T2>
std::string ParallelGptFP8TritonModel<T1, T2>::toString()
{
    std::stringstream ss;
    ss << "Model: "
       << "\nmax_seq_len: " << max_seq_len_ << "\nhead_num: " << head_num_ << "\nsize_per_head: " << size_per_head_
       << "\ninter_size: " << inter_size_ << "\nnum_layer: " << num_layer_ << "\nvocab_size: " << vocab_size_
       << "\nstart_id: " << start_id_ << "\nend_id: " << end_id_ << "\ntensor_para_size: " << tensor_para_size_
       << "\npipeline_para_size: " << pipeline_para_size_ << "\fp8_mode: " << fp8_mode_
       << "\nenable_custom_all_reduce: " << enable_custom_all_reduce_ << "\nmodel_name: " << model_name_
       << "\nmodel_dir: " << model_dir_ << std::endl;
    return ss.str();
}

template<typename T1, typename T2>
void ParallelGptFP8TritonModel<T1, T2>::createCustomComms(
    std::vector<std::shared_ptr<ft::AbstractCustomComm>>* custom_all_reduce_comms, int world_size)
{
    using commDataType = typename ft::CustomARCommTypeConverter<T2>::Type;
    ft::initCustomAllReduceComm<commDataType>(custom_all_reduce_comms, enable_custom_all_reduce_, world_size);
}

template<typename T1, typename T2>
int ParallelGptFP8TritonModel<T1, T2>::getTensorParaSize()
{
    return tensor_para_size_;
}

template<typename T1, typename T2>
int ParallelGptFP8TritonModel<T1, T2>::getPipelineParaSize()
{
    return pipeline_para_size_;
}

#ifdef ENABLE_BF16
template struct ParallelGptFP8TritonModel<__nv_fp8_e4m3, __nv_bfloat16>;
#endif
