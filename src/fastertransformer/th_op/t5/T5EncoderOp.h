/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/t5/T5Encoder.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/mpi_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

class IFT5Encoder {
public:
    virtual ~IFT5Encoder() {}
    virtual void forward(size_t batch_size,
                         size_t seq_len,
                         th::Tensor& input,
                         th::Tensor& sequence_lengths,
                         th::Tensor& output,
                         bool removing_padding) = 0;
};

template<typename T>
class FTT5Encoder: public IFT5Encoder {
public:
    FTT5Encoder(int head_num,
                int head_size,
                int inter_size,
                int d_model,
                int layer_num,
                int num_bucket,
                int max_distance,
                bool sparse,
                float q_scaling,
                int tensor_para_size,
                int pipeline_para_size,
                bool t5_with_bias,
                ft::PositionEmbeddingType position_embedding_type,
                const std::vector<th::Tensor>& w):
        _head_num(head_num),
        _head_size(head_size),
        _inter_size(inter_size),
        _d_model(d_model),
        _layer_num(layer_num),
        _num_bucket(num_bucket),
        _max_distance(max_distance),
        _weights(w),
        _sparse(sparse),
        _q_scaling(q_scaling),
        _t5_with_bias(t5_with_bias),
        _position_embedding_type(position_embedding_type)
    {
        tensor_para_.world_size_ = tensor_para_size;
        pipeline_para_.world_size_ = pipeline_para_size;
        init_nccl_comm();
#ifndef SPARSITY_ENABLED
        if (sparse) {
            std::cout << "[WARNING] Sparsity support is not enabled. Will use dense GEMM instead.\n" << std::flush;
        }
#endif
        int hidden_dim = _head_num * _head_size;
        ft::check_cuda_error(cublasLtCreate(&_cublasltHandle));
        sm_ = ft::getSMVersion();
#ifdef SPARSITY_ENABLED
        if (sparse) {
            CHECK_CUSPARSE(cusparseLtInit(&_cusparseLtHandle));
        }
#endif
        std::string sp_config_fname = sparse ? "spgemm_config.in" : "";
        cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in", sp_config_fname);
        cublas_wrapper_mutex_ = new std::mutex();

        t5_encoder_weights.resizeLayer(_layer_num);
        t5_encoder_weights.setT5StructureDiff(t5_with_bias, position_embedding_type);
        for (int i = 0; i < _layer_num; i++) {
            int local_num_layer = (int)(ceil(_layer_num * 1.0f / pipeline_para_.world_size_));
            if (!(i < _layer_num && (i >= local_num_layer * pipeline_para_.rank_)
                  && (i < local_num_layer * (pipeline_para_.rank_ + 1)))) {
                continue;
            }
            const int first_layer_index = local_num_layer * pipeline_para_.rank_;

            t5_encoder_weights.t5_encoder_layer_weights[i]->attn_layernorm_weights.gamma =
                get_ptr<T>(_weights[0]) + _d_model * (i - first_layer_index);
            t5_encoder_weights.t5_encoder_layer_weights[i]->attention_weights.query_weight.kernel =
                get_ptr<T>(_weights[1]) + _d_model * hidden_dim / tensor_para_.world_size_ * (i - first_layer_index);
            t5_encoder_weights.t5_encoder_layer_weights[i]->attention_weights.key_weight.kernel =
                get_ptr<T>(_weights[2]) + _d_model * hidden_dim / tensor_para_.world_size_ * (i - first_layer_index);
            t5_encoder_weights.t5_encoder_layer_weights[i]->attention_weights.value_weight.kernel =
                get_ptr<T>(_weights[3]) + _d_model * hidden_dim / tensor_para_.world_size_ * (i - first_layer_index);
            t5_encoder_weights.t5_encoder_layer_weights[i]->attention_weights.attention_output_weight.kernel =
                get_ptr<T>(_weights[4]) + hidden_dim / tensor_para_.world_size_ * _d_model * (i - first_layer_index);
            t5_encoder_weights.t5_encoder_layer_weights[i]->ffn_layernorm_weights.gamma =
                get_ptr<T>(_weights[5]) + _d_model * (i - first_layer_index);
            t5_encoder_weights.t5_encoder_layer_weights[i]->ffn_weights.intermediate_weight.kernel =
                get_ptr<T>(_weights[6]) + _d_model * _inter_size / tensor_para_.world_size_ * (i - first_layer_index);
            t5_encoder_weights.t5_encoder_layer_weights[i]->ffn_weights.output_weight.kernel =
                get_ptr<T>(_weights[7]) + _inter_size / tensor_para_.world_size_ * _d_model * (i - first_layer_index);
            if (_t5_with_bias) {
                t5_encoder_weights.t5_encoder_layer_weights[i]->attn_layernorm_weights.beta =
                    get_ptr<T>(_weights[11]) + _d_model * (i - first_layer_index);
                t5_encoder_weights.t5_encoder_layer_weights[i]->attention_weights.query_weight.bias =
                    get_ptr<T>(_weights[12]) + hidden_dim / tensor_para_.world_size_ * (i - first_layer_index);
                t5_encoder_weights.t5_encoder_layer_weights[i]->attention_weights.key_weight.bias =
                    get_ptr<T>(_weights[13]) + hidden_dim / tensor_para_.world_size_ * (i - first_layer_index);
                t5_encoder_weights.t5_encoder_layer_weights[i]->attention_weights.value_weight.bias =
                    get_ptr<T>(_weights[14]) + hidden_dim / tensor_para_.world_size_ * (i - first_layer_index);
                t5_encoder_weights.t5_encoder_layer_weights[i]->attention_weights.attention_output_weight.bias =
                    get_ptr<T>(_weights[15]) + _d_model * (i - first_layer_index);
                t5_encoder_weights.t5_encoder_layer_weights[i]->ffn_layernorm_weights.beta =
                    get_ptr<T>(_weights[16]) + _d_model * (i - first_layer_index);
                t5_encoder_weights.t5_encoder_layer_weights[i]->ffn_weights.intermediate_weight.bias =
                    get_ptr<T>(_weights[17]) + _inter_size / tensor_para_.world_size_ * (i - first_layer_index);
                t5_encoder_weights.t5_encoder_layer_weights[i]->ffn_weights.output_weight.bias =
                    get_ptr<T>(_weights[18]) + _d_model * (i - first_layer_index);
            }
        }
        t5_encoder_weights.post_transformer_layernorm_weights.gamma = get_ptr<T>(_weights[8]);
        t5_encoder_weights.absolute_or_relative_position_embedding = get_ptr<T>(_weights[9]);
        t5_encoder_weights.embedding_table = get_ptr<T>(_weights[10]);
        if (_t5_with_bias) {
            t5_encoder_weights.post_transformer_layernorm_weights.beta = get_ptr<T>(_weights[19]);
        }

#ifdef SPARSITY_ENABLED
        if (sparse) {
            auto stream = at::cuda::getCurrentCUDAStream().stream();
            cublasHandle_t _cublasHandle = at::cuda::getCurrentCUDABlasHandle();
            cublasSetStream(_cublasHandle, stream);
            ft::cublasMMWrapper cublas_wrapper = ft::cublasMMWrapper(_cublasHandle,
                                                                     _cublasltHandle,
                                                                     _cusparseLtHandle,
                                                                     stream,
                                                                     cublas_algo_map_,
                                                                     cublas_wrapper_mutex_,
                                                                     nullptr);
            for (int i = 0; i < _layer_num; ++i) {
                t5_encoder_weights.t5_encoder_layer_weights[i]->compress_weights(cublas_wrapper, hidden_dim);
            }
        }
#endif
    }

    ~FTT5Encoder() override
    {
        cublasLtDestroy(_cublasltHandle);
#ifdef SPARSITY_ENABLED
        if (_sparse) {
            cusparseLtDestroy(&_cusparseLtHandle);
        }
#endif
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void init_nccl_comm()
    {
        int mpi_initialized;
        MPICHECK(MPI_Initialized(&mpi_initialized));
        if (!mpi_initialized) {
            printf("[INFO] MPI is not initialized! Skipped the NCCL communication initialization.\n");
            if (tensor_para_.world_size_ != 1) {
                printf("[FATAL] MPI initialization can only be skipped when tensor_para_size=1, but got %d!\n",
                       tensor_para_.world_size_);
            }
            if (pipeline_para_.world_size_ != 1) {
                printf("[FATAL] MPI initialization can only be skipped when pipeline_para_size=1, but got %d!\n",
                       pipeline_para_.world_size_);
            }
            return;
        }

        int rank, world_size;
        MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
        tensor_para_.rank_ = rank % tensor_para_.world_size_;
        pipeline_para_.rank_ = rank / tensor_para_.world_size_;

        ncclUniqueId tensor_para_nccl_uid;
        ncclUniqueId pipeline_para_nccl_uid;

        // assume gpu_num = n * k,
        // tensor parallelism group size is n
        // pipeline parallelism group size is k
        if (tensor_para_.rank_ == 0) {
            // get the uid of each tensor parallelism group
            // here, 0, 1, ..., n-1 are in group 0,
            //       n, ..., 2n - 1 are in group 1.
            NCCLCHECK(ncclGetUniqueId(&tensor_para_nccl_uid));
            for (int i = 1; i < (int)tensor_para_.world_size_; i++) {
                printf("[INFO] rank %d sends tensor_para_nccl_uid to rank %d \n", rank, rank + i);
                MPICHECK(MPI_Send(
                    &tensor_para_nccl_uid, sizeof(tensor_para_nccl_uid), MPI_BYTE, rank + i, 0, MPI_COMM_WORLD));
            }
        }
        else {
            MPI_Status status;
            printf(
                "[INFO] rank %d receives tensor_para_nccl_uid from rank %d \n", rank, rank - (int)tensor_para_.rank_);
            MPICHECK(MPI_Recv(&tensor_para_nccl_uid,
                              sizeof(tensor_para_nccl_uid),
                              MPI_BYTE,
                              rank - tensor_para_.rank_,
                              0,
                              MPI_COMM_WORLD,
                              &status));
        }

        if (pipeline_para_.rank_ == 0) {
            // get the uid of each pipeline parallelism group
            // 0, k, 2k, are in group 0
            // 1, k+1, 2k+1 are in group 1
            NCCLCHECK(ncclGetUniqueId(&pipeline_para_nccl_uid));
            for (int i = 1; i < (int)pipeline_para_.world_size_; i++) {
                printf("[INFO] rank %d sends pipeline_para_nccl_uid to rank %d \n",
                       rank,
                       rank + i * (int)tensor_para_.world_size_);
                MPICHECK(MPI_Send(&pipeline_para_nccl_uid,
                                  sizeof(pipeline_para_nccl_uid),
                                  MPI_BYTE,
                                  rank + i * tensor_para_.world_size_,
                                  0,
                                  MPI_COMM_WORLD));
            }
        }
        else {
            MPI_Status status;
            printf("[INFO] rank %d receives pipeline_para_nccl_uid from rank %d \n",
                   rank,
                   rank % (int)tensor_para_.world_size_);
            MPICHECK(MPI_Recv(&pipeline_para_nccl_uid,
                              sizeof(pipeline_para_nccl_uid),
                              MPI_BYTE,
                              rank % tensor_para_.world_size_,
                              0,
                              MPI_COMM_WORLD,
                              &status));
        }
        NCCLCHECK(ncclCommInitRank(
            &tensor_para_.nccl_comm_, tensor_para_.world_size_, tensor_para_nccl_uid, tensor_para_.rank_));
        NCCLCHECK(ncclCommInitRank(
            &pipeline_para_.nccl_comm_, pipeline_para_.world_size_, pipeline_para_nccl_uid, pipeline_para_.rank_));
    }

    void forward(size_t batch_size,
                 size_t seq_len,
                 th::Tensor& input_ids,
                 th::Tensor& sequence_lengths,
                 th::Tensor& output,
                 bool removing_padding) override
    {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t _cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(_cublasHandle, stream);
        ft::Allocator<ft::AllocatorType::TH>* allocator = new ft::Allocator<ft::AllocatorType::TH>();
        ft::cublasMMWrapper* cublas_wrapper =
#ifdef SPARSITY_ENABLED
            new ft::cublasMMWrapper(_cublasHandle,
                                    _cublasltHandle,
                                    _cusparseLtHandle,
                                    stream,
                                    cublas_algo_map_,
                                    cublas_wrapper_mutex_,
                                    allocator);
#else
            new ft::cublasMMWrapper(
                _cublasHandle, _cublasltHandle, stream, cublas_algo_map_, cublas_wrapper_mutex_, allocator);
#endif

        if (std::is_same<T, half>::value) {
            cublas_wrapper->setFP16GemmConfig();
        }
        else if (std::is_same<T, float>::value) {
            cublas_wrapper->setFP32GemmConfig();
        }

        ft::AttentionType attention_type = ft::getAttentionType<T>(_head_size, sm_, removing_padding, seq_len, false);

        ft::T5Encoder<T>* t5_encoder =
            new ft::T5Encoder<T>(batch_size,
                                 seq_len,
                                 _head_num,
                                 _head_size,
                                 _inter_size,
                                 _d_model,
                                 _layer_num,
                                 _num_bucket,
                                 _max_distance,
                                 sm_,
                                 _q_scaling,
                                 stream,
                                 cublas_wrapper,
                                 allocator,
                                 false,
                                 attention_type,
                                 _sparse,
                                 _t5_with_bias ? ft::ActivationType::Gelu : ft::ActivationType::Relu,
                                 ft::LayerNormType::pre_layernorm,
                                 tensor_para_,
                                 pipeline_para_);

        std::unordered_map<std::string, ft::Tensor> input_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"input_ids", convert_tensor<T>(input_ids)}, {"sequence_length", convert_tensor<int>(sequence_lengths)}};

        std::unordered_map<std::string, ft::Tensor> output_tensors =
            std::unordered_map<std::string, ft::Tensor>{{"output_hidden_state", convert_tensor<T>(output)}};

        try {
            t5_encoder->forward(&output_tensors, &input_tensors, &t5_encoder_weights);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what();
            exit(-1);
        }
        catch (...) {
            std::cout << "Runtime error";
            exit(-1);
        }
        delete t5_encoder;
        delete cublas_wrapper;
        delete allocator;
    }

private:
    const int _head_num;
    const int _head_size;
    const int _inter_size;
    const int _d_model;
    const int _layer_num;
    const int _num_bucket;
    const int _max_distance;
    std::vector<th::Tensor> _weights;
    bool _t5_with_bias;
    ft::PositionEmbeddingType _position_embedding_type;
    bool _sparse;
    const float _q_scaling;
    int sm_;
    cublasLtHandle_t _cublasltHandle;
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t _cusparseLtHandle;
#endif
    std::mutex* cublas_wrapper_mutex_;
    ft::cublasAlgoMap* cublas_algo_map_;
    ft::T5EncoderWeight<T> t5_encoder_weights;

    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;
};

class FasterTransformerT5Encoder: public th::jit::CustomClassHolder {
public:
    FasterTransformerT5Encoder(th::Tensor attr_output_layernorm_gamma,
                               th::Tensor q_kernel,
                               th::Tensor k_kernel,
                               th::Tensor v_kernel,
                               th::Tensor attr_output_kernel,
                               th::Tensor output_layernorm_gamma,
                               th::Tensor inter_kernel,
                               th::Tensor output_kernel,
                               th::Tensor post_transformer_layernorm_gamma,
                               th::Tensor absolute_or_relative_position_embedding,
                               th::Tensor embedding_table,
                               th::Tensor attr_output_layernorm_beta,
                               th::Tensor q_bias,
                               th::Tensor k_bias,
                               th::Tensor v_bias,
                               th::Tensor attr_output_bias,
                               th::Tensor output_layernorm_beta,
                               th::Tensor inter_bias,
                               th::Tensor output_bias,
                               th::Tensor post_transformer_layernorm_beta,
                               int64_t head_num,
                               int64_t head_size,
                               int64_t inter_size,
                               int64_t d_model,
                               bool remove_padding,
                               int64_t layer_num,
                               int64_t num_bucket,
                               int64_t max_distance,
                               bool sparse,
                               double q_scaling,
                               int64_t tensor_para_size,
                               int64_t pipeline_para_size,
                               bool t5_with_bias,
                               int64_t position_embedding_type);

    ~FasterTransformerT5Encoder();

    th::Tensor forward(th::Tensor input, th::Tensor sequence_lengths);

    std::vector<th::Tensor> get_pickle_info() const;

private:
    const at::ScalarType _st;
    bool _remove_padding;
    IFT5Encoder* ft_t5_encoder;
    th::Tensor head_info;
    th::Tensor scaling_info;
    std::vector<th::Tensor> weights;
};

}  // namespace torch_ext
