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

#include "3rdparty/INIReader.h"

#include "src/fastertransformer/triton_backend/deberta/DebertaTritonModel.h"
#include "src/fastertransformer/triton_backend/deberta/DebertaTritonModelInstance.h"

namespace ft = fastertransformer;

template<typename T>
DebertaTritonModel<T>::DebertaTritonModel(size_t      tensor_para_size,
                                          size_t      pipeline_para_size,
                                          bool        enable_custom_all_reduce,
                                          std::string model_dir,
                                          bool        is_sparse,
                                          bool        is_remove_padding):
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    shared_weights_(std::vector<std::shared_ptr<ft::DebertaWeight<T>>>(ft::getDeviceCount())),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    model_dir_(model_dir),
    is_sparse_(is_sparse),
    is_remove_padding_(is_remove_padding)
{
    FT_CHECK_WITH_INFO(is_sparse == false, "still not support sparse in deberta backend");

    INIReader reader = INIReader(model_dir + "/config.ini");
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << model_dir << "/config.ini"
                  << "'\n";
        ft::FT_CHECK(false);
    }

    /* Deberta base Configuration File Example
    [deberta]
    model_name = deberta
    hidden_size = 1024
    num_layer = 24
    head_num = 16
    size_per_head = 64
    activation_type = gelu
    inter_size = 4096
    vocab_size = 128100
    max_relative_positions = 512
    relative_position_buckets = 256
    weight_data_type = fp32
    */

    model_name_                 = reader.Get("deberta", "model_name");
    head_num_                   = reader.GetInteger("deberta", "head_num");
    size_per_head_              = reader.GetInteger("deberta", "size_per_head");
    inter_size_                 = reader.GetInteger("deberta", "inter_size");
    vocab_size_                 = reader.GetInteger("deberta", "vocab_size");
    num_layer_                  = reader.GetInteger("deberta", "num_layer");
    max_relative_positions_     = reader.GetInteger("deberta", "max_relative_positions");
    relative_position_buckets_  = reader.GetInteger("deberta", "relative_position_buckets");
    layernorm_type_             = ft::getLayerNormType("post_layernorm");
    activation_type_            = ft::getActivationType(reader.Get("deberta", "activation_type", "Gelu"));
    q_scaling_                  = reader.GetFloat("deberta", "q_scaling", sqrtf(3.0f));
}

template<typename T>
std::unique_ptr<AbstractTransformerModelInstance>
DebertaTritonModel<T>::createModelInstance(int                                                            device_id,
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

    auto deberta =
        std::make_unique<ft::Deberta<T>>(ft::Deberta<T>(0,  // max_batch_size, FT will adjust the buffer automatically.
                                                  0,  //  max_seq_len, FT will adjust the buffer automatically.
                                                  head_num_,
                                                  size_per_head_,
                                                  max_relative_positions_,
                                                  relative_position_buckets_,
                                                  inter_size_,
                                                  num_layer_,
                                                  q_scaling_,
                                                  stream,
                                                  cublas_wrapper.get(),
                                                  allocator.get(),
                                                  false,
                                                  is_sparse_,
                                                  activation_type_,
                                                  layernorm_type_,
                                                  tensor_para,
                                                  pipeline_para,
                                                  custom_all_reduce_comm,
                                                  enable_custom_all_reduce_));

#ifdef SPARSITY_ENABLED
    if (is_sparse_) {
        for (int i = 0; i < num_layer_; ++i) {
            shared_weights_[device_id]->deberta_layer_weights[i].compress_weights(*(cublas_wrapper.get()),
                                                                               head_num_ * size_per_head_);
        }
    }
#endif

    return std::unique_ptr<DebertaTritonModelInstance<T>>(new DebertaTritonModelInstance<T>(std::move(deberta),
                                                                                      shared_weights_[device_id],
                                                                                      std::move(allocator),
                                                                                      std::move(cublas_algo_map),
                                                                                      std::move(cublas_wrapper_mutex),
                                                                                      std::move(cublas_wrapper),
                                                                                      std::move(cuda_device_prop_ptr)));
}

template<typename T>
void DebertaTritonModel<T>::createSharedWeights(int device_id, int rank)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    const int tensor_para_rank   = rank % tensor_para_size_;
    const int pipeline_para_rank = rank / tensor_para_size_;
    shared_weights_[device_id]   = std::make_shared<ft::DebertaWeight<T>>(head_num_ * size_per_head_,
                                                                     inter_size_,
                                                                     max_relative_positions_,
                                                                     relative_position_buckets_,
                                                                     vocab_size_,
                                                                     num_layer_,
                                                                     tensor_para_size_,
                                                                     tensor_para_rank,
                                                                     pipeline_para_size_,
                                                                     pipeline_para_rank);

    shared_weights_[device_id]->loadModel(model_dir_);
    return;
}

template<typename T>
std::string DebertaTritonModel<T>::toString()
{
    std::stringstream ss;
    ss << "Model: " << model_name_ << "\nmodel_dir: " << model_dir_ << "\nhead_num: " << head_num_
       << "\nsize_per_head: " << size_per_head_ << "\ninter_size: " << inter_size_ << "\nnum_layer: " << num_layer_
       << "\ntensor_para_size: " << tensor_para_size_ << "\npipeline_para_size: " << pipeline_para_size_
       << "\nmax_relative_positions: " << max_relative_positions_ << "\nrelative_position_buckets: " << relative_position_buckets_
       << "\nq_scaling: " << q_scaling_ << "\nis_remove_padding: " << is_remove_padding_
       << "\nis_sparse: " << is_sparse_ << "\nactivation_type: " << static_cast<int>(activation_type_)
       << "\nlayernorm_type: " << static_cast<int>(layernorm_type_) << "\nvocab_size: " << vocab_size_
       << "\nenable_custom_all_reduce:" << enable_custom_all_reduce_ << std::endl;

    return ss.str();
}

template<typename T>
void DebertaTritonModel<T>::createCustomComms(
    std::vector<std::shared_ptr<ft::AbstractCustomComm>>* custom_all_reduce_comms, int world_size)
{
    using commDataType = typename ft::CustomARCommTypeConverter<T>::Type;
    ft::initCustomAllReduceComm<commDataType>(custom_all_reduce_comms, enable_custom_all_reduce_, world_size);
}

template<typename T>
int DebertaTritonModel<T>::getTensorParaSize()
{
    return tensor_para_size_;
}

template<typename T>
int DebertaTritonModel<T>::getPipelineParaSize()
{
    return pipeline_para_size_;
}

template struct DebertaTritonModel<float>;
template struct DebertaTritonModel<half>;
#ifdef ENABLE_BF16
template struct DebertaTritonModel<__nv_bfloat16>;
#endif
