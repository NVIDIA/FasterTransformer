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

#include "3rdparty/INIReader.h"

#include "src/fastertransformer/triton_backend/bert/BertTritonModel.h"
#include "src/fastertransformer/triton_backend/bert/BertTritonModelInstance.h"

namespace ft = fastertransformer;

template<typename T>
BertTritonModel<T>::BertTritonModel(size_t      tensor_para_size,
                                    size_t      pipeline_para_size,
                                    bool        enable_custom_all_reduce,
                                    std::string model_dir,
                                    int         int8_mode,
                                    bool        is_sparse,
                                    bool        is_remove_padding):
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    shared_weights_(std::vector<std::shared_ptr<ft::BertWeight<T>>>(ft::getDeviceCount())),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    model_dir_(model_dir),
    int8_mode_(int8_mode),
    is_sparse_(is_sparse),
    is_remove_padding_(is_remove_padding)
{
    ft::FT_CHECK_WITH_INFO(int8_mode_ == 0, "still not support int8 in bert backend");
    ft::FT_CHECK_WITH_INFO(is_sparse == false, "still not support sparse in bert backend");

    INIReader reader = INIReader(model_dir + "/config.ini");
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << model_dir << "/config.ini"
                  << "'\n";
        ft::FT_CHECK(false);
    }

    /* Bert base Configuration File Example
    [bert]
    model_name = bert
    position_embedding_type = absolute
    hidden_size = 768
    num_hidden_layers = 12
    head_num = 12
    size_per_head = 64
    activation_type = gelu
    inter_size = 3072
    max_position_embeddings = 512
    layer_norm_eps = 1e-12
    weight_data_type = fp32
    */

    model_name_      = reader.Get("bert", "model_name");
    head_num_        = reader.GetInteger("bert", "head_num");
    size_per_head_   = reader.GetInteger("bert", "size_per_head");
    inter_size_      = reader.GetInteger("bert", "inter_size");
    num_layer_       = reader.GetInteger("bert", "num_layer");
    layernorm_type_  = ft::getLayerNormType("post_layernorm");
    activation_type_ = ft::getActivationType(reader.Get("bert", "activation_type", "Gelu"));
    q_scaling_       = reader.GetFloat("bert", "q_scaling", 1.0f);
}

template<typename T>
std::unique_ptr<AbstractTransformerModelInstance>
BertTritonModel<T>::createModelInstance(int                                                               device_id,
                                        int                                                               rank,
                                        cudaStream_t                                                      stream,
                                        std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
                                        std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm)
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

    const int         max_seq_len = 384;
    ft::AttentionType attention_type =
        ft::getAttentionType<T>(size_per_head_, ft::getSMVersion(), is_remove_padding_, max_seq_len);

    auto bert =
        std::make_unique<ft::Bert<T>>(ft::Bert<T>(0,  // max_batch_size, FT will adjust the buffer automatically.
                                                  0,  //  max_seq_len, FT will adjust the buffer automatically.
                                                  head_num_,
                                                  size_per_head_,
                                                  inter_size_,
                                                  num_layer_,
                                                  ft::getSMVersion(),
                                                  q_scaling_,
                                                  stream,
                                                  cublas_wrapper.get(),
                                                  allocator.get(),
                                                  false,
                                                  attention_type,
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
            shared_weights_[device_id]->bert_layer_weights[i].compress_weights(*(cublas_wrapper.get()),
                                                                               head_num_ * size_per_head_);
        }
    }
#endif

    return std::unique_ptr<BertTritonModelInstance<T>>(new BertTritonModelInstance<T>(std::move(bert),
                                                                                      shared_weights_[device_id],
                                                                                      std::move(allocator),
                                                                                      std::move(cublas_algo_map),
                                                                                      std::move(cublas_wrapper_mutex),
                                                                                      std::move(cublas_wrapper),
                                                                                      std::move(cuda_device_prop_ptr)));
}

template<typename T>
void BertTritonModel<T>::createSharedWeights(int device_id, int rank)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    const int tensor_para_rank   = rank % tensor_para_size_;
    const int pipeline_para_rank = rank / tensor_para_size_;
    shared_weights_[device_id]   = std::make_shared<ft::BertWeight<T>>(head_num_ * size_per_head_,
                                                                     inter_size_,
                                                                     num_layer_,
                                                                     tensor_para_size_,
                                                                     tensor_para_rank,
                                                                     pipeline_para_size_,
                                                                     pipeline_para_rank);

    shared_weights_[device_id]->loadModel(model_dir_);
    return;
}

template<typename T>
std::string BertTritonModel<T>::toString()
{
    std::stringstream ss;
    ss << "Model: " << model_name_ << "\nmodel_dir: " << model_dir_ << "\nhead_num: " << head_num_
       << "\nsize_per_head: " << size_per_head_ << "\ninter_size: " << inter_size_ << "\nnum_layer: " << num_layer_
       << "\ntensor_para_size: " << tensor_para_size_ << "\npipeline_para_size: " << pipeline_para_size_
       << "\nq_scaling: " << q_scaling_ << "\nis_remove_padding: " << is_remove_padding_
       << "\nis_sparse: " << is_sparse_ << "\nactivation_type: " << static_cast<int>(activation_type_)
       << "\nlayernorm_type: " << static_cast<int>(layernorm_type_) << "\nint8_mode:" << int8_mode_
       << "\nenable_custom_all_reduce:" << enable_custom_all_reduce_ << "\nis_sparse: " << is_sparse << std::endl;

    return ss.str();
}

template<typename T>
void BertTritonModel<T>::createCustomComms(
    std::vector<std::shared_ptr<ft::AbstractCustomComm>>* custom_all_reduce_comms, int world_size)
{
    using commDataType = typename ft::CustomARCommTypeConverter<T>::Type;
    ft::initCustomAllReduceComm<commDataType>(custom_all_reduce_comms, enable_custom_all_reduce_, world_size);
}

template<typename T>
int BertTritonModel<T>::getTensorParaSize()
{
    return tensor_para_size_;
}

template<typename T>
int BertTritonModel<T>::getPipelineParaSize()
{
    return pipeline_para_size_;
}

template struct BertTritonModel<float>;
template struct BertTritonModel<half>;
#ifdef ENABLE_BF16
template struct BertTritonModel<__nv_bfloat16>;
#endif
