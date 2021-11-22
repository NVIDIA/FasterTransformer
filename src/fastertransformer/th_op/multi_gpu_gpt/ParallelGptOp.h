/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/utils/mpi_utils.h"

#include <cuda_fp16.h>
#include <iostream>
#include <nvToolsExt.h>
#include <vector>

#include "torch/csrc/cuda/Stream.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include "src/fastertransformer/th_op/th_traits.h"
#include "src/fastertransformer/th_op/th_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

using std::vector;

class IFGpt {
public:
    virtual ~IFGpt() {}
    virtual void forward(th::Tensor& input_ids,
                         th::Tensor& input_lengths,
                         th::Tensor& output_ids,
                         th::Tensor& parent_ids,
                         th::Tensor& sequence_lengths,
                         size_t request_output_len) = 0;
};

template<typename T>
class FTGpt: public IFGpt {
public:
    FTGpt(const size_t max_batch_size,
          const size_t max_seq_len,
          const size_t beam_width,
          const size_t head_num,
          const size_t size_per_head,
          const size_t inter_size,
          const size_t layer_num,
          const size_t vocab_size,
          const int start_id,
          const int end_id,
          const float beam_search_diversity_rate,
          const int top_k,
          const float top_p,
          const unsigned long long random_seed,
          const float temperature,
          const float len_penalty,
          const float repetition_penalty,
          const int tensor_para_size,
          const int pipeline_para_size,
          const int int8_mode,
          const vector<th::Tensor> weights,
          const vector<th::Tensor> int8_weights,
          const vector<th::Tensor> scale):
        max_batch_size_(max_batch_size),
        max_seq_len_(max_seq_len),
        beam_width_(beam_width),
        head_num_(head_num),
        size_per_head_(size_per_head),
        inter_size_(inter_size),
        layer_num_(layer_num),
        vocab_size_(vocab_size),
        start_id_(start_id),
        end_id_(end_id),
        beam_search_diversity_rate_(beam_search_diversity_rate),
        top_k_(top_k),
        top_p_(top_p),
        temperature_(temperature),
        len_penalty_(len_penalty),
        repetition_penalty_(repetition_penalty),
        tensor_para_size_(tensor_para_size),
        pipeline_para_size_(pipeline_para_size),
        int8_mode_(int8_mode),
        weights_(weights),
        int8_weights_(int8_weights),
        scale_(scale)
    {
        check_cuda_error(cublasLtCreate(&cublasltHandle_));
        cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in");
        cublas_wrapper_mutex_ = new std::mutex();

        init_nccl_comm();

        gpt_weights_.resizeLayer(layer_num_);

        for (int i = 0; i < (int)layer_num_; i++) {
            gpt_weights_.decoder_layer_weights[i]->pre_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 0 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->pre_layernorm_weights.beta =
                get_ptr<T>(weights_[i + 1 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.kernel =
                get_ptr<T>(weights_[i + 2 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.bias =
                get_ptr<T>(weights_[i + 3 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(weights_[i + 4 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.bias =
                get_ptr<T>(weights_[i + 5 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attn_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 6 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attn_layernorm_weights.beta =
                get_ptr<T>(weights_[i + 7 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.kernel =
                get_ptr<T>(weights_[i + 8 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.bias =
                get_ptr<T>(weights_[i + 9 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.kernel =
                get_ptr<T>(weights_[i + 10 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.bias =
                get_ptr<T>(weights_[i + 11 * layer_num_]);

            if (int8_mode_ != 0) {
                gpt_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.int8_kernel =
                    get_ptr<int8_t>(int8_weights_[i + 0 * layer_num_]);
                gpt_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.scale =
                    get_ptr<float>(scale_[i + 0 * layer_num_]);
                gpt_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.int8_kernel =
                    get_ptr<int8_t>(int8_weights_[i + 1 * layer_num_]);
                gpt_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.scale =
                    get_ptr<float>(scale_[i + 1 * layer_num_]);
                gpt_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.int8_kernel =
                    get_ptr<int8_t>(int8_weights_[i + 2 * layer_num_]);
                gpt_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.scale =
                    get_ptr<float>(scale_[i + 2 * layer_num_]);
                gpt_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.int8_kernel =
                    get_ptr<int8_t>(int8_weights_[i + 3 * layer_num_]);
                gpt_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.scale =
                    get_ptr<float>(scale_[i + 3 * layer_num_]);
            }
        }

        gpt_weights_.post_decoder_layernorm.gamma = get_ptr<T>(weights_[12 * layer_num_ + 0]);
        gpt_weights_.post_decoder_layernorm.beta = get_ptr<T>(weights_[12 * layer_num_ + 1]);
        gpt_weights_.position_encoding_table = get_ptr<T>(weights_[12 * layer_num_ + 2]);
        gpt_weights_.pre_decoder_embedding_table = get_ptr<T>(weights_[12 * layer_num_ + 3]);
        gpt_weights_.post_decoder_embedding.kernel = get_ptr<T>(weights_[12 * layer_num_ + 4]);

        int rank, world_size;
        MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

        if (rank == 0) {
            random_seed_ = random_seed;
        }
        if (world_size > 1) {
            MPICHECK(MPI_Bcast(&random_seed_, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD));
        }

        int device_id = 0;
        check_cuda_error(cudaGetDevice(&device_id));
        check_cuda_error(cudaGetDeviceProperties(&prop_, device_id));
        printf("Device %s\n", prop_.name);
    }

    ~FTGpt() override
    {
        ncclCommDestroy(tensor_para_comm_);
        ncclCommDestroy(pipeline_para_comm_);
        cublasLtDestroy(cublasltHandle_);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void init_nccl_comm()
    {
        int rank;
        MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        tensor_para_rank_ = rank % tensor_para_size_;
        pipeline_para_rank_ = rank / tensor_para_size_;

        ncclUniqueId tensor_para_nccl_uid;
        ncclUniqueId pipeline_para_nccl_uid;

        // assume gpu_num = n * k,
        // tensor parallelism group size is n
        // pipeline parallelism group size is k
        if (tensor_para_rank_ == 0) {
            // get the uid of each tensor parallelism group
            // here, 0, 1, ..., n-1 are in group 0,
            //       n, ..., 2n - 1 are in group 1.
            NCCLCHECK(ncclGetUniqueId(&tensor_para_nccl_uid));
            for (int i = 1; i < (int)tensor_para_size_; i++) {
                printf("[INFO] rank %d sends tensor_para_nccl_uid to rank %d \n", rank, rank + i);
                MPICHECK(MPI_Send(
                    &tensor_para_nccl_uid, sizeof(tensor_para_nccl_uid), MPI_BYTE, rank + i, 0, MPI_COMM_WORLD));
            }
        }
        else {
            MPI_Status status;
            printf("[INFO] rank %d receives tensor_para_nccl_uid from rank %d \n", rank, rank - (int)tensor_para_rank_);
            MPICHECK(MPI_Recv(&tensor_para_nccl_uid,
                              sizeof(tensor_para_nccl_uid),
                              MPI_BYTE,
                              rank - tensor_para_rank_,
                              0,
                              MPI_COMM_WORLD,
                              &status));
        }

        if (pipeline_para_rank_ == 0) {
            // get the uid of each pipeline parallelism group
            // 0, k, 2k, are in group 0
            // 1, k+1, 2k+1 are in group 1
            NCCLCHECK(ncclGetUniqueId(&pipeline_para_nccl_uid));
            for (int i = 1; i < (int)pipeline_para_size_; i++) {
                printf("[INFO] rank %d sends pipeline_para_nccl_uid to rank %d \n",
                       rank,
                       rank + i * (int)tensor_para_size_);
                MPICHECK(MPI_Send(&pipeline_para_nccl_uid,
                                  sizeof(pipeline_para_nccl_uid),
                                  MPI_BYTE,
                                  rank + i * tensor_para_size_,
                                  0,
                                  MPI_COMM_WORLD));
            }
        }
        else {
            MPI_Status status;
            printf(
                "[INFO] rank %d receives pipeline_para_nccl_uid from rank %d \n", rank, rank % (int)tensor_para_size_);
            MPICHECK(MPI_Recv(&pipeline_para_nccl_uid,
                              sizeof(pipeline_para_nccl_uid),
                              MPI_BYTE,
                              rank % tensor_para_size_,
                              0,
                              MPI_COMM_WORLD,
                              &status));
        }
        NCCLCHECK(ncclCommInitRank(&tensor_para_comm_, tensor_para_size_, tensor_para_nccl_uid, tensor_para_rank_));
        NCCLCHECK(
            ncclCommInitRank(&pipeline_para_comm_, pipeline_para_size_, pipeline_para_nccl_uid, pipeline_para_rank_));
    }

    void forward(th::Tensor& input_ids,
                 th::Tensor& input_lengths,
                 th::Tensor& output_ids,
                 th::Tensor& parent_ids,
                 th::Tensor& sequence_lengths,
                 size_t request_output_len) override
    {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(cublasHandle, stream);
        fastertransformer::Allocator<AllocatorType::TH> allocator = fastertransformer::Allocator<AllocatorType::TH>();
        ft::cublasMMWrapper cublas_wrapper = ft::cublasMMWrapper(
            cublasHandle, cublasltHandle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, &allocator);

        if (std::is_same<T, half>::value) {
            cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
        }
        else if (std::is_same<T, float>::value) {
            cublas_wrapper.setFP32GemmConfig();
        }

        const size_t request_batch_size = (size_t)input_ids.size(0) / beam_width_;
        const size_t max_input_length = (size_t)input_ids.size(1);
        const int total_output_len = (int)(max_input_length + request_output_len);

        ParallelGpt<T> gpt = ParallelGpt<T>(request_batch_size,
                                            total_output_len,
                                            max_input_length,
                                            beam_width_,
                                            head_num_,
                                            size_per_head_,
                                            inter_size_,
                                            layer_num_,
                                            vocab_size_,
                                            start_id_,
                                            end_id_,
                                            0.0f,
                                            top_k_,
                                            top_p_,
                                            random_seed_,  // TODO(bhsueh) add seed argument
                                            temperature_,
                                            len_penalty_,
                                            repetition_penalty_,
                                            tensor_para_size_,
                                            tensor_para_rank_,
                                            tensor_para_comm_,
                                            pipeline_para_size_,
                                            pipeline_para_rank_,
                                            pipeline_para_comm_,
                                            stream,
                                            &cublas_wrapper,
                                            &allocator,
                                            false,
                                            &prop_,
                                            false,
                                            int8_mode_);

        std::vector<Tensor> input_tensors =
            std::vector<Tensor>{Tensor{MEMORY_GPU,
                                       TYPE_INT32,
                                       std::vector<size_t>{request_batch_size * beam_width_, max_input_length},
                                       get_ptr<int>(input_ids)},
                                Tensor{MEMORY_GPU,
                                       TYPE_INT32,
                                       std::vector<size_t>{request_batch_size * beam_width_},
                                       get_ptr<int>(input_lengths)},
                                Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, &total_output_len}};

        std::vector<Tensor> output_tensors =
            std::vector<Tensor>{Tensor{MEMORY_GPU,
                                       TYPE_INT32,
                                       std::vector<size_t>{request_batch_size, beam_width_, (size_t)total_output_len},
                                       get_ptr<int>(output_ids)},
                                Tensor{MEMORY_GPU,
                                       TYPE_INT32,
                                       std::vector<size_t>{(size_t)total_output_len, request_batch_size, beam_width_},
                                       get_ptr<int>(parent_ids)},
                                Tensor{MEMORY_GPU,
                                       TYPE_INT32,
                                       std::vector<size_t>{request_batch_size, beam_width_},
                                       get_ptr<int>(sequence_lengths)},
                                Tensor{MEMORY_GPU,
                                       TYPE_FP32,
                                       std::vector<size_t>{request_output_len, request_batch_size, beam_width_},
                                       nullptr}};

        try {
            gpt.forward(&output_tensors, &input_tensors, &gpt_weights_);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what();
            exit(-1);
        }
        catch (...) {
            std::cout << "Runtime error";
            exit(-1);
        }
    }

private:
    const size_t max_batch_size_;
    const size_t max_seq_len_;
    const size_t beam_width_;
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t layer_num_;
    const size_t vocab_size_;
    const int start_id_;
    const int end_id_;
    const float beam_search_diversity_rate_;
    const int top_k_;
    const float top_p_;
    const float temperature_;
    const float len_penalty_;
    const float repetition_penalty_;

    const int int8_mode_ = 0;

    size_t tensor_para_size_;
    size_t pipeline_para_size_;

    std::vector<th::Tensor> int8_weights_;
    std::vector<th::Tensor> scale_;
    std::vector<th::Tensor> weights_;

    size_t tensor_para_rank_;
    ncclComm_t tensor_para_comm_;
    size_t pipeline_para_rank_;
    ncclComm_t pipeline_para_comm_;

    unsigned long long random_seed_;

    cublasLtHandle_t cublasltHandle_;
    std::mutex* cublas_wrapper_mutex_;
    ft::cublasAlgoMap* cublas_algo_map_;
    struct cudaDeviceProp prop_;
    ParallelGptWeight<T> gpt_weights_;
};

class ParallelGptOp: public th::jit::CustomClassHolder {
public:
    ParallelGptOp(const int64_t max_batch_size,
                  const int64_t max_seq_len,
                  const int64_t beam_width,
                  const int64_t head_num,
                  const int64_t size_per_head,
                  const int64_t inter_size,
                  const int64_t layer_num,
                  const int64_t vocab_size,
                  const int64_t start_id,
                  const int64_t end_id,
                  const double beam_search_diversity_rate,
                  const int64_t top_k,
                  const double top_p,
                  const unsigned long long random_seed,
                  const double temperature,
                  const double len_penalty,
                  const double repetition_penalty,
                  const int64_t tensor_para_size,
                  const int64_t pipeline_para_size,
                  const int64_t int8_mode,
                  const vector<th::Tensor> weights,
                  const vector<th::Tensor> int8_weights,
                  const vector<th::Tensor> scale);

    ~ParallelGptOp();

    vector<th::Tensor> forward(th::Tensor input_ids, th::Tensor input_lengths, const int64_t output_len);

private:
    const at::ScalarType st_;
    IFGpt* ftgpt;
    std::vector<th::Tensor> weights;
};

}  // namespace torch_ext
