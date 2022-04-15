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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/tf_op/BaseOp.h"

namespace ft = fastertransformer;
namespace tf = tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Gpt")
    .Input("input_ids: int32")                  // 0 [batch_size * beam_width, max_input_length]
    .Input("input_sequence_length: int32")      // 1 [batch_size * beam_width]
    .Input("pre_beta: N * T")                   // 2
    .Input("pre_gamma: N * T")                  // 3
    .Input("self_qkv_kernel: N * T")            // 4
    .Input("self_qkv_bias: N * T")              // 5
    .Input("self_output_kernel: N * T")         // 6
    .Input("self_output_bias: N * T")           // 7
    .Input("self_beta: N * T")                  // 8
    .Input("self_gamma: N * T")                 // 9
    .Input("ffn_kernel1: N * T")                // 10
    .Input("ffn_bias1: N * T")                  // 11
    .Input("ffn_kernel2: N * T")                // 12
    .Input("ffn_bias2: N * T")                  // 13
    .Input("post_decoder_layernorm_beta: T")    // 14
    .Input("post_decoder_layernorm_gamma: T")   // 15
    .Input("position_encoding_table: T")        // 16
    .Input("pre_decoder_embedding_table: T")    // 17
    .Input("post_decoder_embedding_kernel: T")  // 18
    .Output("output_ids: int32")
    .Output("parent_ids: int32")
    .Output("sequence_length: int32")
    .Output("cum_log_probs: float")
    .Attr("N: int")
    .Attr("T: {float, half}")
    .Attr("max_batch_size: int >= 1")
    .Attr("max_seq_len: int >= 1")
    .Attr("beam_width: int >= 1")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("inter_size: int >= 1")
    .Attr("num_layer: int >= 1")
    .Attr("start_id: int >= 0")
    .Attr("end_id: int >= 0")
    .Attr("beam_search_diversity_rate: float")
    .Attr("top_k: int >= 0")
    .Attr("top_p: float")
    .Attr("temperature: float")
    .Attr("len_penalty: float")
    .Attr("repetition_penalty: float")
    .Attr("output_log_probs: bool = false")
    .Attr("request_output_length: int")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        int beam_width, max_seq_len, request_output_length;
        c->GetAttr("beam_width", &beam_width);
        c->GetAttr("max_seq_len", &max_seq_len);
        c->GetAttr("request_output_length", &request_output_length);

        int rank = c->Rank(c->input(0));
        assert(rank == 2);

        // calculate batch size
        tf::shape_inference::DimensionOrConstant max_seq_dim(max_seq_len);
        tf::shape_inference::DimensionOrConstant beam_dim(beam_width);
        tf::shape_inference::DimensionHandle batch_dim = c->Dim(c->input(0), 0);

        if (beam_width > 1) {
            c->set_output(0, c->MakeShape({batch_dim, beam_dim, max_seq_len}));
            c->set_output(1, c->MakeShape({max_seq_len, batch_dim, beam_dim}));
            c->set_output(2, c->MakeShape({batch_dim, beam_dim}));
        }
        else {
            c->set_output(0, c->MakeShape({batch_dim, max_seq_len}));
            c->set_output(1, c->MakeShape({max_seq_len, batch_dim, 1}));
            c->set_output(2, c->MakeShape({batch_dim}));
            c->set_output(3, c->MakeShape({request_output_length, batch_dim}));
        }

        return tf::Status::OK();
    });

template<typename T>
class TFTraits;

template<>
class TFTraits<float> {
public:
    typedef float DataType;
};

template<>
class TFTraits<Eigen::half> {
public:
    typedef half DataType;
};

#ifdef ENABLE_BF16
template<>
class TFTraits<Eigen::bfloat16> {
public:
    typedef __nv_bfloat16 DataType;
};
#endif

template<typename Device, typename T>
class GptOp: public BaseOp<T> {
public:
    explicit GptOp(tf::OpKernelConstruction* context): BaseOp<T>(context)
    {
        try {
            OP_REQUIRES_OK(context, context->GetAttr("max_batch_size", &max_batch_size_));
            OP_REQUIRES_OK(context, context->GetAttr("max_seq_len", &max_seq_len_));
            OP_REQUIRES_OK(context, context->GetAttr("beam_width", &beam_width_));
            OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
            OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
            OP_REQUIRES_OK(context, context->GetAttr("inter_size", &inter_size_));
            OP_REQUIRES_OK(context, context->GetAttr("num_layer", &num_layer_));
            OP_REQUIRES_OK(context, context->GetAttr("start_id", &start_id_));
            OP_REQUIRES_OK(context, context->GetAttr("end_id", &end_id_));
            OP_REQUIRES_OK(context, context->GetAttr("beam_search_diversity_rate", &beam_search_diversity_rate_));
            OP_REQUIRES_OK(context, context->GetAttr("top_k", &top_k_));
            OP_REQUIRES_OK(context, context->GetAttr("top_p", &top_p_));
            OP_REQUIRES_OK(context, context->GetAttr("temperature", &temperature_));
            OP_REQUIRES_OK(context, context->GetAttr("len_penalty", &len_penalty_));
            OP_REQUIRES_OK(context, context->GetAttr("repetition_penalty", &repetition_penalty_));
            OP_REQUIRES_OK(context, context->GetAttr("output_log_probs", &output_log_probs_));
            OP_REQUIRES_OK(context, context->GetAttr("request_output_length", &request_output_length_));
            cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in");
            ft::check_cuda_error(cudaGetDeviceProperties(&prop_, 0));
        }
        catch (std::runtime_error& error) {
            OP_REQUIRES(context, false, tf::errors::Internal(error.what()));
        }
    }

    ~GptOp()
    {
        delete cublas_algo_map_;
    }

    void Compute(tf::OpKernelContext* context) override
    {
        OP_REQUIRES(context,
                    context->num_inputs() == (num_layer_ * 12) + 7,
                    tf::errors::InvalidArgument("[ERROR] More or Less input arguments"));

        const size_t batch_size = (size_t)context->input(0).dim_size(0);
        const size_t vocab_size = (size_t)(context->input(2 + num_layer_ * 12 + 3).dim_size(0));
        const size_t max_input_length = (size_t)(context->input(0).dim_size(1));

        const cudaStream_t& stream = context->eigen_device<Device>().stream();
        cublasHandle_t cublas_handle = this->get_cublas_handler();
        cublasSetStream(cublas_handle, stream);
        ft::Allocator<ft::AllocatorType::TF> allocator(context, stream);
        ft::cublasMMWrapper cublas_wrapper = ft::cublasMMWrapper(cublas_handle,
                                                                 this->get_cublaslt_handler(),
                                                                 stream,
                                                                 cublas_algo_map_,
                                                                 this->get_cublas_wrapper_mutex(),
                                                                 &allocator);

        if (std::is_same<T, Eigen::half>::value) {
            cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
        }
#ifdef ENABLE_BF16
        else if (std::is_same<T, Eigen::bfloat16>::value) {
            cublas_wrapper.setBF16GemmConfig();
        }
#endif
        else if (std::is_same<T, float>::value) {
            cublas_wrapper.setFP32GemmConfig();
        }

        ft::ParallelGptWeight<DataType> gpt_weidghts;
        gpt_weidghts.resizeLayer(num_layer_);

        for (int i = 0; i < num_layer_; i++) {
            this->get_tensor(context, 2 + i, &gpt_weidghts.decoder_layer_weights[i]->pre_layernorm_weights.beta);
            this->get_tensor(
                context, 2 + num_layer_ * 1 + i, &gpt_weidghts.decoder_layer_weights[i]->pre_layernorm_weights.gamma);

            this->get_tensor(context,
                             2 + num_layer_ * 2 + i,
                             &gpt_weidghts.decoder_layer_weights[i]->self_attention_weights.query_weight.kernel);
            this->get_tensor(context,
                             2 + num_layer_ * 3 + i,
                             &gpt_weidghts.decoder_layer_weights[i]->self_attention_weights.query_weight.bias);
            this->get_tensor(
                context,
                2 + num_layer_ * 4 + i,
                &gpt_weidghts.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.kernel);
            this->get_tensor(
                context,
                2 + num_layer_ * 5 + i,
                &gpt_weidghts.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.bias);
            this->get_tensor(context,
                             2 + num_layer_ * 6 + i,
                             &gpt_weidghts.decoder_layer_weights[i]->self_attn_layernorm_weights.beta);
            this->get_tensor(context,
                             2 + num_layer_ * 7 + i,
                             &gpt_weidghts.decoder_layer_weights[i]->self_attn_layernorm_weights.gamma);

            this->get_tensor(context,
                             2 + num_layer_ * 8 + i,
                             &gpt_weidghts.decoder_layer_weights[i]->ffn_weights.intermediate_weight.kernel);
            this->get_tensor(context,
                             2 + num_layer_ * 9 + i,
                             &gpt_weidghts.decoder_layer_weights[i]->ffn_weights.intermediate_weight.bias);
            this->get_tensor(context,
                             2 + num_layer_ * 10 + i,
                             &gpt_weidghts.decoder_layer_weights[i]->ffn_weights.output_weight.kernel);
            this->get_tensor(context,
                             2 + num_layer_ * 11 + i,
                             &gpt_weidghts.decoder_layer_weights[i]->ffn_weights.output_weight.bias);
        }

        this->get_tensor(context, 2 + num_layer_ * 12 + 0, &gpt_weidghts.post_decoder_layernorm.beta);
        this->get_tensor(context, 2 + num_layer_ * 12 + 1, &gpt_weidghts.post_decoder_layernorm.gamma);
        this->get_tensor(context, 2 + num_layer_ * 12 + 2, &gpt_weidghts.position_encoding_table);
        this->get_tensor(context, 2 + num_layer_ * 12 + 3, &gpt_weidghts.pre_decoder_embedding_table);
        this->get_tensor(context, 2 + num_layer_ * 12 + 4, &gpt_weidghts.post_decoder_embedding.kernel);
        int total_output_length = request_output_length_ + (int)max_input_length;

        tf::Tensor* output_id_tensor = nullptr;
        tf::Tensor* parent_id_tensor = nullptr;
        tf::Tensor* sequence_length_tensor = nullptr;
        tf::Tensor* cum_log_probs = nullptr;
        if (beam_width_ > 1) {
            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    0,
                    {(long long int)batch_size, (long long int)beam_width_, (long long int)total_output_length},
                    &output_id_tensor));
            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    1,
                    {(long long int)total_output_length, (long long int)batch_size, (long long int)beam_width_},
                    &parent_id_tensor));
            OP_REQUIRES_OK(context,
                           context->allocate_output(
                               2, {(long long int)batch_size, (long long int)beam_width_}, &sequence_length_tensor));

            if (this->output_log_probs_) {
                OP_REQUIRES_OK(
                    context,
                    context->allocate_output(
                        3,
                        {(long long int)request_output_length_, (long long int)batch_size, (long long int)beam_width_},
                        &cum_log_probs));
            }
            else {
                OP_REQUIRES_OK(context, context->allocate_output(3, {0}, &cum_log_probs));
            }
        }
        else {
            OP_REQUIRES_OK(context,
                           context->allocate_output(
                               0, {(long long int)batch_size, (long long int)total_output_length}, &output_id_tensor));
            OP_REQUIRES_OK(context,
                           context->allocate_output(
                               1, {(long long int)total_output_length, (long long int)batch_size}, &parent_id_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(2, {(long long int)batch_size}, &sequence_length_tensor));
            if (this->output_log_probs_) {
                OP_REQUIRES_OK(
                    context,
                    context->allocate_output(
                        3, {(long long int)request_output_length_, (long long int)batch_size}, &cum_log_probs));
            }
            else {
                OP_REQUIRES_OK(context, context->allocate_output(3, {0}, &cum_log_probs));
            }
        }
        int* output_ids = (int*)(output_id_tensor->flat<int>().data());
        int* parent_ids = (int*)(parent_id_tensor->flat<int>().data());
        int* sequence_length = (int*)(sequence_length_tensor->flat<int>().data());

        ft::NcclParam tensor_para;
        ft::NcclParam pipeline_para;

        ft::ParallelGpt<DataType> gpt = ft::ParallelGpt<DataType>(batch_size,
                                                                  total_output_length,
                                                                  max_input_length,
                                                                  beam_width_,
                                                                  head_num_,
                                                                  size_per_head_,
                                                                  inter_size_,
                                                                  num_layer_,
                                                                  vocab_size,
                                                                  start_id_,
                                                                  end_id_,
                                                                  beam_search_diversity_rate_,
                                                                  top_k_,
                                                                  top_p_,
                                                                  0,
                                                                  temperature_,
                                                                  len_penalty_,
                                                                  repetition_penalty_,
                                                                  tensor_para,
                                                                  pipeline_para,
                                                                  stream,
                                                                  &cublas_wrapper,
                                                                  &allocator,
                                                                  false,
                                                                  &prop_,
                                                                  false,
                                                                  0);

        std::unordered_map<std::string, ft::Tensor> input_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"input_ids", this->convert_int_tensor(context->input(0))},
            {"input_lengths", this->convert_int_tensor(context->input(1))},
            {"max_output_seq_len",
             ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1}, &total_output_length}}};
        if (top_k_ == 0 && top_p_ == 0.0f) {
            ft::FT_CHECK(beam_width_ > 1);
            input_tensors.insert(
                {"beam_search_diversity_rate",
                 ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate_}});
        }
        else {
            if (top_p_ != 0.0f) {
                input_tensors.insert(
                    {"runtime_top_p", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &top_p_}});
            }
            if (top_k_ != 0) {
                input_tensors.insert(
                    {"runtime_top_k", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1}, &top_k_}});
            }
        }
        input_tensors.insert(
            {"temperature", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &temperature_}});
        input_tensors.insert(
            {"len_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &len_penalty_}});
        input_tensors.insert({"repetition_penalty",
                              ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty_}});
        uint64_t random_seed = 0;
        input_tensors.insert(
            {"random_seed", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT64, std::vector<size_t>{1}, &random_seed}});

        std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"output_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{batch_size, (size_t)beam_width_, (size_t)total_output_length},
                        output_ids}},
            {"parent_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{(size_t)total_output_length, batch_size, (size_t)beam_width_},
                        parent_ids}},
            {"sequence_length",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{batch_size, (size_t)beam_width_},
                        sequence_length}},
            {"output_log_probs",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_FP32,
                        {(size_t)request_output_length_, (size_t)batch_size, (size_t)beam_width_},
                        output_log_probs_ ? reinterpret_cast<float*>(cum_log_probs->flat<float>().data()) : nullptr}}};

        try {
            gpt.forward(&output_tensors, &input_tensors, &gpt_weidghts);
        }
        catch (std::runtime_error& error) {
            std::cout << tf::errors::Internal(error.what());
            exit(-1);
        }
        catch (...) {
            std::cout << tf::errors::Internal("Runtime error \n");
            exit(-1);
        }
    }

private:
    int max_batch_size_ = 0, max_seq_len_ = 0, beam_width_ = 1;
    int head_num_ = 0, size_per_head_ = 0, inter_size_ = 0;
    int num_layer_ = 0, start_id_ = -1, end_id_ = -1;
    float beam_search_diversity_rate_ = 1.0;
    float temperature_;
    float len_penalty_;
    float repetition_penalty_;
    int top_k_ = 0;
    float top_p_ = 0.0f;
    bool output_log_probs_;
    int request_output_length_;
    ft::cublasAlgoMap* cublas_algo_map_;
    struct cudaDeviceProp prop_;
    typedef TFTraits<T> traits_;
    typedef typename traits_::DataType DataType;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                                                                \
    REGISTER_KERNEL_BUILDER(Name("Gpt").Device(tf::DEVICE_GPU).TypeConstraint<T>("T"), GptOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
