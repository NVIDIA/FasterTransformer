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

#include "src/fastertransformer/triton_backend/t5/T5TritonModel.h"
#include "src/fastertransformer/triton_backend/t5/T5TritonModelInstance.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/utils/allocator.h"

namespace ft = fastertransformer;

std::shared_ptr<AbstractTransformerModel> AbstractTransformerModel::createT5Model(std::string model_dir)
{
    INIReader reader = INIReader(model_dir + "/config.ini");
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << model_dir << "/config.ini"
                  << "'\n";
        return nullptr;
    }

    const std::string data_type = reader.Get("ft_instance_hyperparameter", "data_type");
    if (data_type == "fp16") {
        return std::make_shared<T5TritonModel<half>>(reader, model_dir);
    }
#ifdef ENABLE_BF16
    else if (data_type == "bf16") {
        return std::make_shared<T5TritonModel<__nv_bfloat16>>(reader, model_dir);
    }
#endif
    else if (data_type == "fp32") {
        return std::make_shared<T5TritonModel<float>>(reader, model_dir);
    }
    else {
        FT_LOG_ERROR("Unsupported data type " + data_type);
        exit(-1);
    }
}

template<typename T>
T5TritonModel<T>::T5TritonModel(INIReader reader, std::string model_dir): model_dir_(model_dir)
{
    // encoder
    encoder_head_num_      = reader.GetInteger("encoder", "num_heads");
    encoder_size_per_head_ = reader.GetInteger("encoder", "d_kv");
    encoder_d_model_       = reader.GetInteger("encoder", "d_model");
    encoder_inter_size_    = reader.GetInteger("encoder", "d_ff");
    encoder_num_layer_     = reader.GetInteger("encoder", "num_layers");
    encoder_vocab_size_    = reader.GetInteger("encoder", "vocab_size");
    encoder_num_bucket_or_max_pos_seq_len_ =
        reader.GetInteger("encoder", "relative_attention_num_buckets_or_max_pos_seq_len");

    // decoding
    decoding_head_num_      = reader.GetInteger("decoder", "num_heads");
    decoding_size_per_head_ = reader.GetInteger("decoder", "d_kv");
    decoding_d_model_       = reader.GetInteger("decoder", "d_model");
    decoding_inter_size_    = reader.GetInteger("decoder", "d_ff");
    decoding_num_layer_     = reader.GetInteger("decoder", "num_layers");
    decoding_vocab_size_    = reader.GetInteger("decoder", "vocab_size");
    decoding_num_bucket_or_max_pos_seq_len_ =
        reader.GetInteger("decoder", "relative_attention_num_buckets_or_max_pos_seq_len");
    start_id_                 = reader.GetInteger("decoder", "decoder_start_token_id");
    end_id_                   = reader.GetInteger("decoder", "eos_token_id");
    tensor_para_size_         = reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size");
    pipeline_para_size_       = reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size");
    enable_custom_all_reduce_ = reader.GetInteger("ft_instance_hyperparameter", "enable_custom_all_reduce", 0);
    t5_with_bias_             = reader.GetBoolean("structure", "t5_with_bias", false);
    use_gated_activation_     = reader.GetBoolean("structure", "use_gated_activation", false);
    position_embedding_type_ =
        ft::PositionEmbeddingType(reader.Get("structure", "position_embedding_type", "relative") == "relative" ? 0 : 1);
    q_scaling_    = t5_with_bias_ ? 1.0f : (1.0f / (sqrt(encoder_size_per_head_) * 1.0f));
    max_distance_ = 128;  // use default value of huggingface here
}

template<typename T>
T5TritonModel<T>::T5TritonModel(size_t      tensor_para_size,
                                size_t      pipeline_para_size,
                                int         enable_custom_all_reduce,
                                std::string model_dir,
                                int         int8_mode):
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    encoder_shared_weights_(std::vector<std::shared_ptr<ft::T5EncoderWeight<T>>>(ft::getDeviceCount())),
    decoding_shared_weights_(std::vector<std::shared_ptr<ft::T5DecodingWeight<T>>>(ft::getDeviceCount())),
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

    ft::FT_CHECK(int8_mode_ == 0);

    model_name_ = reader.Get("encoder", "_name_or_path");
    // encoder
    encoder_head_num_      = reader.GetInteger("encoder", "num_heads");
    encoder_size_per_head_ = reader.GetInteger("encoder", "d_kv");
    encoder_d_model_       = reader.GetInteger("encoder", "d_model");
    encoder_inter_size_    = reader.GetInteger("encoder", "d_ff");
    encoder_num_layer_     = reader.GetInteger("encoder", "num_layers");
    encoder_vocab_size_    = reader.GetInteger("encoder", "vocab_size");
    encoder_num_bucket_or_max_pos_seq_len_ =
        reader.GetInteger("encoder", "relative_attention_num_buckets_or_max_pos_seq_len");

    // decoding
    decoding_head_num_      = reader.GetInteger("decoder", "num_heads");
    decoding_size_per_head_ = reader.GetInteger("decoder", "d_kv");
    decoding_d_model_       = reader.GetInteger("decoder", "d_model");
    decoding_inter_size_    = reader.GetInteger("decoder", "d_ff");
    decoding_num_layer_     = reader.GetInteger("decoder", "num_layers");
    decoding_vocab_size_    = reader.GetInteger("decoder", "vocab_size");
    decoding_num_bucket_or_max_pos_seq_len_ =
        reader.GetInteger("decoder", "relative_attention_num_buckets_or_max_pos_seq_len");
    start_id_            = reader.GetInteger("decoder", "decoder_start_token_id");
    end_id_              = reader.GetInteger("decoder", "eos_token_id");
    tie_word_embeddings_ = reader.GetBoolean("decoder", "tie_word_embeddings", true);

    t5_with_bias_         = reader.GetBoolean("structure", "t5_with_bias", false);
    use_gated_activation_ = reader.GetBoolean("structure", "use_gated_activation", false);
    activation_type_      = ft::getActivationType(reader.Get("encoder", "feed_forward_proj"));
    position_embedding_type_ =
        ft::PositionEmbeddingType(reader.Get("structure", "position_embedding_type", "relative") == "relative" ? 0 : 1);
    q_scaling_ = t5_with_bias_ ? 1.0f : (1.0f / (sqrt(encoder_size_per_head_) * 1.0f));

    max_distance_ = 128;  // use default value of huggingface here
}

template<typename T>
std::unique_ptr<AbstractTransformerModelInstance>
T5TritonModel<T>::createModelInstance(int                                                               device_id,
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

    const int sm_ = ft::getSMVersion();

    // TODO(bhsueh) not support fused mha
    ft::AttentionType attention_type =
        ft::getAttentionType<T>(encoder_size_per_head_, sm_, true, encoder_num_bucket_or_max_pos_seq_len_, false);

    ft::NcclParam tensor_para_   = nccl_params.first[comms_rank];
    ft::NcclParam pipeline_para_ = nccl_params.second[comms_rank];

    auto encoder = std::make_unique<ft::T5Encoder<T>>(ft::T5Encoder<T>(0,
                                                                       0,
                                                                       encoder_head_num_,
                                                                       encoder_size_per_head_,
                                                                       encoder_inter_size_,
                                                                       encoder_d_model_,
                                                                       encoder_num_layer_,
                                                                       encoder_num_bucket_or_max_pos_seq_len_,
                                                                       max_distance_,
                                                                       sm_,
                                                                       q_scaling_,
                                                                       stream,
                                                                       cublas_wrapper.get(),
                                                                       allocator.get(),
                                                                       false,
                                                                       attention_type,
                                                                       false,
                                                                       activation_type_,
                                                                       ft::LayerNormType::pre_layernorm,
                                                                       tensor_para_,
                                                                       pipeline_para_,
                                                                       custom_all_reduce_comm,
                                                                       enable_custom_all_reduce_));

    auto decoding = std::make_unique<ft::T5Decoding<T>>(ft::T5Decoding<T>(0,
                                                                          0,
                                                                          0,
                                                                          0,
                                                                          decoding_head_num_,
                                                                          decoding_size_per_head_,
                                                                          decoding_inter_size_,
                                                                          decoding_d_model_,
                                                                          decoding_num_layer_,
                                                                          decoding_vocab_size_,
                                                                          decoding_num_bucket_or_max_pos_seq_len_,
                                                                          max_distance_,
                                                                          q_scaling_,
                                                                          start_id_,
                                                                          end_id_,
                                                                          0.0f,  // beam_search_diversity_rate_,
                                                                          1,     // top_k_,
                                                                          0.0f,  // top_p_,
                                                                          1.0f,  // temperature_,
                                                                          0.0f,  // len_penalty_,
                                                                          1.0f,  // repetition_penalty_,
                                                                          stream,
                                                                          cublas_wrapper.get(),
                                                                          allocator.get(),
                                                                          false,
                                                                          cuda_device_prop_ptr.get(),
                                                                          tensor_para_,
                                                                          pipeline_para_,
                                                                          activation_type_,
                                                                          tie_word_embeddings_,
                                                                          custom_all_reduce_comm,
                                                                          enable_custom_all_reduce_));

    return std::unique_ptr<T5TritonModelInstance<T>>(new T5TritonModelInstance<T>(std::move(encoder),
                                                                                  std::move(decoding),
                                                                                  encoder_shared_weights_[device_id],
                                                                                  decoding_shared_weights_[device_id],
                                                                                  std::move(allocator),
                                                                                  std::move(cublas_algo_map),
                                                                                  std::move(cublas_wrapper_mutex),
                                                                                  std::move(cublas_wrapper),
                                                                                  std::move(cuda_device_prop_ptr)));
}

template<typename T>
void T5TritonModel<T>::createSharedWeights(int device_id, int rank)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    const int tensor_para_rank   = rank % tensor_para_size_;
    const int pipeline_para_rank = rank / tensor_para_size_;

    encoder_shared_weights_[device_id] =
        std::make_shared<ft::T5EncoderWeight<T>>(encoder_head_num_,
                                                 encoder_size_per_head_,
                                                 encoder_d_model_,
                                                 encoder_inter_size_,
                                                 encoder_vocab_size_,
                                                 encoder_num_layer_,
                                                 encoder_num_bucket_or_max_pos_seq_len_,
                                                 tensor_para_size_,
                                                 tensor_para_rank,
                                                 pipeline_para_size_,
                                                 pipeline_para_rank,
                                                 t5_with_bias_,
                                                 use_gated_activation_,
                                                 position_embedding_type_);

    decoding_shared_weights_[device_id] =
        std::make_shared<ft::T5DecodingWeight<T>>(decoding_head_num_,
                                                  decoding_size_per_head_,
                                                  decoding_d_model_,
                                                  decoding_inter_size_,
                                                  decoding_vocab_size_,
                                                  decoding_num_layer_,
                                                  encoder_d_model_,
                                                  decoding_num_bucket_or_max_pos_seq_len_,
                                                  tensor_para_size_,
                                                  tensor_para_rank,
                                                  pipeline_para_size_,
                                                  pipeline_para_rank,
                                                  t5_with_bias_,
                                                  use_gated_activation_,
                                                  position_embedding_type_);

    encoder_shared_weights_[device_id]->loadModel(model_dir_);
    decoding_shared_weights_[device_id]->loadModel(model_dir_);
    return;
}

template<typename T>
std::string T5TritonModel<T>::toString()
{
    std::stringstream ss;
    std::string       position_embedding_type_string =
        position_embedding_type_ == ft::PositionEmbeddingType::relative ? "relative" : "absolute";

    ss << "\nModel: "
       << "\n    encoder_head_num_: " << encoder_head_num_ << "\n    encoder_size_per_head_: " << encoder_size_per_head_
       << "\n    encoder_d_model_: " << encoder_d_model_ << "\n    encoder_inter_size_: " << encoder_inter_size_
       << "\n    encoder_num_layer_: " << encoder_num_layer_ << "\n    encoder_vocab_size_: " << encoder_vocab_size_
       << "\n    encoder_num_bucket_or_max_pos_seq_len_: " << encoder_num_bucket_or_max_pos_seq_len_
       << "\n    decoding_head_num_: " << decoding_head_num_
       << "\n    decoding_size_per_head_: " << decoding_size_per_head_
       << "\n    decoding_d_model_: " << decoding_d_model_ << "\n    decoding_inter_size_: " << decoding_inter_size_
       << "\n    decoding_num_layer_: " << decoding_num_layer_ << "\n    decoding_vocab_size_: " << decoding_vocab_size_
       << "\n    decoding_num_bucket_or_max_pos_seq_len_: " << decoding_num_bucket_or_max_pos_seq_len_
       << "\n    t5_with_bias_: " << t5_with_bias_ << "\n    use_gated_activation_: " << use_gated_activation_
       << "\n   position_embedding_type_: " << position_embedding_type_string << "\n    start_id_: " << start_id_
       << "\n    end_id_: " << end_id_ << "\n    model_name_: " << model_name_ << "\n    model_dir_: " << model_dir_
       << std::endl;

    return ss.str();
}

template<typename T>
void T5TritonModel<T>::createCustomComms(std::vector<std::shared_ptr<ft::AbstractCustomComm>>* custom_all_reduce_comms,
                                         int                                                   world_size)
{
    using commDataType = typename ft::CustomARCommTypeConverter<T>::Type;
    ft::initCustomAllReduceComm<commDataType>(custom_all_reduce_comms, enable_custom_all_reduce_, world_size);
}

template<typename T>
int T5TritonModel<T>::getTensorParaSize()
{
    return tensor_para_size_;
}

template<typename T>
int T5TritonModel<T>::getPipelineParaSize()
{
    return pipeline_para_size_;
}

template struct T5TritonModel<float>;
template struct T5TritonModel<half>;
#ifdef ENABLE_BF16
template struct T5TritonModel<__nv_bfloat16>;
#endif
