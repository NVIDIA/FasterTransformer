/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/t5/T5Decoding.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/mpi_utils.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

class IFTT5Decoding {
public:
    virtual ~IFTT5Decoding() {}
    virtual std::vector<th::Tensor> forward(size_t beam_width,
                                            size_t max_seq_len,
                                            size_t top_k,
                                            float top_p,
                                            float beam_search_diversity_rate,
                                            float temperature,
                                            float len_penalty,
                                            float repetition_penalty,
                                            unsigned long long random_seed,
                                            bool is_return_output_log_probs,
                                            bool is_return_cum_log_probs,
                                            th::Tensor memory,
                                            th::Tensor memory_seq_lens) = 0;
};

template<typename T>
class FTT5Decoding: public IFTT5Decoding {
public:
    FTT5Decoding(int head_num,
                 int size_per_head,
                 int inter_size,
                 int mem_d_model,
                 int d_model,
                 int layer_num,
                 int vocab_size,
                 int num_bucket,
                 int max_distance,
                 double q_scaling,
                 int start_id,
                 int end_id,
                 int tensor_para_size,
                 int pipeline_para_size,
                 bool t5_with_bias,
                 ft::PositionEmbeddingType position_embedding_type,
                 const std::vector<th::Tensor>& w):
        head_num_(head_num),
        size_per_head_(size_per_head),
        inter_size_(inter_size),
        mem_d_model_(mem_d_model),
        d_model_(d_model),
        layer_num_(layer_num),
        vocab_size_(vocab_size),
        num_bucket_(num_bucket),
        max_distance_(max_distance),
        q_scaling_(q_scaling),
        start_id_(start_id),
        end_id_(end_id),
        t5_with_bias_(t5_with_bias),
        position_embedding_type_(position_embedding_type),
        _weights(w)
    {
        tensor_para_.world_size_ = tensor_para_size;
        pipeline_para_.world_size_ = pipeline_para_size;
        init_nccl_comm();

        ft::check_cuda_error(cublasLtCreate(&cublasltHandle_));
        cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in");
        cublas_wrapper_mutex_ = new std::mutex();

        decoding_weights.resizeLayer(layer_num_);
        decoding_weights.setT5StructureDiff(t5_with_bias, position_embedding_type);
        const int hidden_dim = head_num_ * size_per_head_;

        for (int i = 0; i < layer_num_; ++i) {
            int local_num_layer = (int)(ceil(layer_num_ * 1.0f / pipeline_para_.world_size_));
            if (!(i < layer_num_ && (i >= local_num_layer * pipeline_para_.rank_)
                  && (i < local_num_layer * (pipeline_para_.rank_ + 1)))) {
                continue;
            }
            const int first_layer_index = local_num_layer * pipeline_para_.rank_;

            decoding_weights.decoder_layer_weights[i]->pre_layernorm_weights.gamma =
                get_ptr<T>(_weights[0]) + (i - first_layer_index) * d_model;
            decoding_weights.decoder_layer_weights[i]->self_attention_weights.query_weight.kernel =
                get_ptr<T>(_weights[1]) + (i - first_layer_index) * d_model * 3 * hidden_dim / tensor_para_.world_size_;
            decoding_weights.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(_weights[2]) + (i - first_layer_index) * hidden_dim / tensor_para_.world_size_ * d_model;
            decoding_weights.decoder_layer_weights[i]->self_attn_layernorm_weights.gamma =
                get_ptr<T>(_weights[3]) + (i - first_layer_index) * d_model;
            decoding_weights.decoder_layer_weights[i]->cross_attention_weights.query_weight.kernel =
                get_ptr<T>(_weights[4]) + (i - first_layer_index) * d_model * hidden_dim / tensor_para_.world_size_;
            decoding_weights.decoder_layer_weights[i]->cross_attention_weights.key_weight.kernel =
                get_ptr<T>(_weights[5]) + (i - first_layer_index) * mem_d_model * hidden_dim / tensor_para_.world_size_;
            decoding_weights.decoder_layer_weights[i]->cross_attention_weights.value_weight.kernel =
                get_ptr<T>(_weights[6]) + (i - first_layer_index) * mem_d_model * hidden_dim / tensor_para_.world_size_;
            decoding_weights.decoder_layer_weights[i]->cross_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(_weights[7]) + (i - first_layer_index) * hidden_dim / tensor_para_.world_size_ * d_model;
            decoding_weights.decoder_layer_weights[i]->cross_attn_layernorm_weights.gamma =
                get_ptr<T>(_weights[8]) + (i - first_layer_index) * d_model;
            decoding_weights.decoder_layer_weights[i]->ffn_weights.intermediate_weight.kernel =
                get_ptr<T>(_weights[9]) + (i - first_layer_index) * d_model * inter_size_ / tensor_para_.world_size_;
            decoding_weights.decoder_layer_weights[i]->ffn_weights.output_weight.kernel =
                get_ptr<T>(_weights[10]) + (i - first_layer_index) * inter_size_ / tensor_para_.world_size_ * d_model;

            if (t5_with_bias_) {
                decoding_weights.decoder_layer_weights[i]->pre_layernorm_weights.beta =
                    get_ptr<T>(_weights[14]) + (i - first_layer_index) * d_model;
                decoding_weights.decoder_layer_weights[i]->self_attention_weights.query_weight.bias =
                    get_ptr<T>(_weights[15]) + (i - first_layer_index) * 3 * hidden_dim / tensor_para_.world_size_;
                decoding_weights.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.bias =
                    get_ptr<T>(_weights[16]) + (i - first_layer_index) * d_model;
                decoding_weights.decoder_layer_weights[i]->self_attn_layernorm_weights.beta =
                    get_ptr<T>(_weights[17]) + (i - first_layer_index) * d_model;
                decoding_weights.decoder_layer_weights[i]->cross_attention_weights.query_weight.bias =
                    get_ptr<T>(_weights[18]) + (i - first_layer_index) * hidden_dim / tensor_para_.world_size_;
                decoding_weights.decoder_layer_weights[i]->cross_attention_weights.key_weight.bias =
                    get_ptr<T>(_weights[19]) + (i - first_layer_index) * hidden_dim / tensor_para_.world_size_;
                decoding_weights.decoder_layer_weights[i]->cross_attention_weights.value_weight.bias =
                    get_ptr<T>(_weights[20]) + (i - first_layer_index) * hidden_dim / tensor_para_.world_size_;
                decoding_weights.decoder_layer_weights[i]->cross_attention_weights.attention_output_weight.bias =
                    get_ptr<T>(_weights[21]) + (i - first_layer_index) * d_model;
                decoding_weights.decoder_layer_weights[i]->cross_attn_layernorm_weights.beta =
                    get_ptr<T>(_weights[22]) + (i - first_layer_index) * d_model;
                decoding_weights.decoder_layer_weights[i]->ffn_weights.intermediate_weight.bias =
                    get_ptr<T>(_weights[23]) + (i - first_layer_index) * inter_size_ / tensor_para_.world_size_;
                decoding_weights.decoder_layer_weights[i]->ffn_weights.output_weight.bias =
                    get_ptr<T>(_weights[24]) + (i - first_layer_index) * d_model;
            }
        }
        decoding_weights.post_decoder_layernorm.gamma = get_ptr<T>(_weights[11]);
        decoding_weights.pre_decoder_embedding_table = get_ptr<T>(_weights[12]);
        decoding_weights.post_decoder_embedding.kernel = get_ptr<T>(_weights[12]);
        decoding_weights.absolute_or_relative_position_embedding = get_ptr<T>(_weights[13]);
        if (t5_with_bias_) {
            decoding_weights.post_decoder_layernorm.beta = get_ptr<T>(_weights[25]);
            decoding_weights.post_decoder_embedding.bias = get_ptr<T>(_weights[26]);
        }
        int device_id = 0;
        ft::check_cuda_error(cudaGetDevice(&device_id));
        ft::check_cuda_error(cudaGetDeviceProperties(&prop_, device_id));
    }

    void init_nccl_comm()
    {
        int mpi_initialized;
        MPICHECK(MPI_Initialized(&mpi_initialized));
        if (!mpi_initialized) {
            FT_LOG_INFO("MPI is not initialized! Skipped the NCCL communication initialization.\n");
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

    ~FTT5Decoding() override
    {
        cublasLtDestroy(cublasltHandle_);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    std::vector<th::Tensor> forward(size_t beam_width,
                                    size_t max_seq_len,
                                    size_t top_k,
                                    float top_p,
                                    float beam_search_diversity_rate,
                                    float temperature,
                                    float len_penalty,
                                    float repetition_penalty,
                                    unsigned long long random_seed,
                                    bool is_return_output_log_probs,
                                    bool is_return_cum_log_probs,
                                    th::Tensor memory,
                                    th::Tensor memory_seq_lens) override
    {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(cublasHandle, stream);
        ft::Allocator<ft::AllocatorType::TH> allocator = ft::Allocator<ft::AllocatorType::TH>();
        ft::cublasMMWrapper cublas_wrapper = ft::cublasMMWrapper(
            cublasHandle, cublasltHandle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, &allocator);

        if (std::is_same<T, half>::value) {
            cublas_wrapper.setFP16GemmConfig();
        }
        else if (std::is_same<T, float>::value) {
            cublas_wrapper.setFP32GemmConfig();
        }

        const size_t batch_size = (size_t)memory.size(0);
        const size_t mem_max_seq_len = (size_t)memory.size(1);

        ft::T5Decoding<T> decoding =
            ft::T5Decoding<T>(batch_size,
                              max_seq_len,
                              mem_max_seq_len,
                              beam_width,
                              head_num_,
                              size_per_head_,
                              inter_size_,
                              d_model_,
                              layer_num_,
                              vocab_size_,
                              num_bucket_,
                              max_distance_,
                              q_scaling_,
                              start_id_,
                              end_id_,
                              beam_search_diversity_rate,
                              top_k,
                              top_p,
                              temperature,
                              len_penalty,
                              repetition_penalty,
                              stream,
                              &cublas_wrapper,
                              &allocator,
                              false,
                              &prop_,
                              tensor_para_,
                              pipeline_para_,
                              t5_with_bias_ ? ft::ActivationType::Gelu : ft::ActivationType::Relu);
        ft::DataType data_type = ft::getTensorType<T>();

        std::unordered_map<std::string, ft::Tensor> input_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"encoder_output",
             ft::Tensor{ft::MEMORY_GPU,
                        data_type,
                        std::vector<size_t>{(size_t)memory.size(0), (size_t)memory.size(1), (size_t)memory.size(2)},
                        get_ptr<T>(memory)}},
            {"encoder_sequence_length",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{(size_t)memory_seq_lens.size(0)},
                        get_ptr<T>(memory_seq_lens)}}};

        if (top_k == 0 && top_p == 0.0f) {
            ft::FT_CHECK(beam_width > 1);
            input_tensors.insert(
                {"beam_search_diversity_rate",
                 ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate}});
        }
        else {
            if (top_p != 0.0f) {
                input_tensors.insert(
                    {"runtime_top_p", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &top_p}});
            }
            if (top_k != 0) {
                input_tensors.insert(
                    {"runtime_top_k", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1}, &top_k}});
            }
        }
        input_tensors.insert(
            {"temperature", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &temperature}});
        input_tensors.insert(
            {"len_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &len_penalty}});
        input_tensors.insert({"repetition_penalty",
                              ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty}});
        input_tensors.insert(
            {"random_seed", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT64, std::vector<size_t>{1}, &random_seed}});

        auto output_ids = torch::empty({batch_size * beam_width * max_seq_len},
                                       torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
        auto sequence_length = torch::empty({batch_size * beam_width},
                                            torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
        std::vector<th::Tensor> th_output_tensors = {output_ids, sequence_length};

        std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"output_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{batch_size, beam_width, max_seq_len},
                        get_ptr<int>(output_ids)}},
            {"sequence_length",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{batch_size, beam_width},
                        get_ptr<int>(sequence_length)}}};

        if (is_return_output_log_probs) {
            auto output_log_probs = torch::empty({batch_size * beam_width * max_seq_len},
                                                 torch::dtype(torch::kFloat).device(torch::kCUDA).requires_grad(false));
            output_tensors.insert({"output_log_probs",
                                   ft::Tensor{ft::MEMORY_GPU,
                                              ft::TYPE_FP32,
                                              {batch_size, beam_width, max_seq_len},
                                              get_ptr<float>(output_log_probs)}});
            th_output_tensors.push_back(output_log_probs);
        }
        if (is_return_cum_log_probs) {
            auto cum_log_probs = torch::empty({batch_size * beam_width},
                                              torch::dtype(torch::kFloat).device(torch::kCUDA).requires_grad(false));
            output_tensors.insert(
                {"cum_log_probs",
                 ft::Tensor{ft::MEMORY_GPU, ft::TYPE_FP32, {batch_size, beam_width}, get_ptr<float>(cum_log_probs)}});
            th_output_tensors.push_back(cum_log_probs);
        }

        decoding.forward(&output_tensors, &input_tensors, &decoding_weights);
        return th_output_tensors;
    }

private:
    const int head_num_;
    const int size_per_head_;
    const int inter_size_;
    const int mem_d_model_;
    const int d_model_;
    const int layer_num_;
    const int vocab_size_;
    const int num_bucket_;
    const int max_distance_;
    double q_scaling_;
    const int start_id_;
    const int end_id_;
    const bool t5_with_bias_;
    const ft::PositionEmbeddingType position_embedding_type_;

    std::vector<th::Tensor> _weights;
    cublasLtHandle_t cublasltHandle_;
    std::mutex* cublas_wrapper_mutex_;
    ft::cublasAlgoMap* cublas_algo_map_;
    struct cudaDeviceProp prop_;
    ft::T5DecodingWeight<T> decoding_weights;

    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;
};

class FasterTransformerT5Decoding: public torch::jit::CustomClassHolder {
public:
    FasterTransformerT5Decoding(int64_t head_num,
                                int64_t size_per_head,
                                int64_t inter_size,
                                int64_t mem_d_model,
                                int64_t d_model,
                                int64_t layer_num,
                                int64_t vocab_size,
                                int64_t num_bucket,
                                int64_t max_distance,
                                double q_scaling,
                                int64_t start_id,
                                int64_t end_id,
                                int64_t tensor_para_size,
                                int64_t pipeline_para_size,
                                bool t5_with_bias,
                                int64_t position_embedding_type,
                                th::Tensor self_layernorm_gamma,
                                th::Tensor self_kernel_qkv,
                                th::Tensor self_output_kernel,
                                th::Tensor cross_layernorm_gamma,
                                th::Tensor cross_kernel_q,
                                th::Tensor cross_kernel_k,
                                th::Tensor cross_kernel_v,
                                th::Tensor cross_output_kernel,
                                th::Tensor ffn_layernorm_gamma,
                                th::Tensor inter_kernel,
                                th::Tensor output_kernel,
                                th::Tensor decoding_gamma,
                                th::Tensor embedding_table,
                                th::Tensor absolute_or_relative_position_embedding,
                                th::Tensor self_layernorm_beta,
                                th::Tensor self_bias_qkv,
                                th::Tensor self_output_bias,
                                th::Tensor cross_layernorm_beta,
                                th::Tensor cross_bias_q,
                                th::Tensor cross_bias_k,
                                th::Tensor cross_bias_v,
                                th::Tensor cross_output_bias,
                                th::Tensor ffn_layernorm_beta,
                                th::Tensor inter_bias,
                                th::Tensor output_bias,
                                th::Tensor decoding_beta,
                                th::Tensor embedding_bias);

    ~FasterTransformerT5Decoding();

    std::vector<th::Tensor> forward(int64_t beam_width,
                                    int64_t max_seq_len,
                                    int64_t top_k,
                                    double top_p,
                                    double beam_search_diversity_rate,
                                    double temperature,
                                    double len_penalty,
                                    double repetition_penalty,
                                    int64_t random_seed,
                                    bool is_return_output_log_probs,
                                    bool is_return_cum_log_probs,
                                    th::Tensor memory,
                                    th::Tensor memory_seq_lens);

    std::vector<th::Tensor> get_pickle_info() const;

private:
    const at::ScalarType _st;
    torch_ext::IFTT5Decoding* ftdecoding;
    th::Tensor int_info_;
    std::vector<th::Tensor> weights;
};

}  // namespace torch_ext