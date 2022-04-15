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

#include "src/fastertransformer/models/decoding/Decoding.h"
#include "src/fastertransformer/tf_op/BaseOp.h"

namespace ft = fastertransformer;
namespace tf = tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Decoding")
    .Input("memory_tensor: T")                 // 0 [batch_size * beam_width, mem_max_seq_len, mem_hidden_units]
    .Input("memory_sequence_length: int32")    // 1
    .Input("pre_beta: N * T")                  // 2
    .Input("pre_gamma: N * T")                 // 3
    .Input("self_qkv_kernel: N * T")           // 4
    .Input("self_qkv_bias: N * T")             // 5
    .Input("self_output_kernel: N * T")        // 6
    .Input("self_output_bias: N * T")          // 7
    .Input("self_beta: N * T")                 // 8
    .Input("self_gamma: N * T")                // 9
    .Input("cross_q_kernel: N * T")            // 10
    .Input("cross_q_bias: N * T")              // 11
    .Input("cross_k_kernel: N * T")            // 12
    .Input("cross_k_bias: N * T")              // 13
    .Input("cross_v_kernel: N * T")            // 14
    .Input("cross_v_bias: N * T")              // 15
    .Input("cross_output_kernel: N * T")       // 16
    .Input("cross_output_bias: N * T")         // 17
    .Input("cross_beta: N * T")                // 18
    .Input("cross_gamma: N * T")               // 19
    .Input("ffn_kernel1: N * T")               // 20
    .Input("ffn_bias1: N * T")                 // 21
    .Input("ffn_kernel2: N * T")               // 22
    .Input("ffn_bias2: N * T")                 // 23
    .Input("post_decoder_layernorm_beta: T")   // 24
    .Input("post_decoder_layernorm_gamma: T")  // 25
    .Input("position_encoding_table: T")       // 26
    .Input("pre_decoder_embedding_table: T")   // 27
    .Input("post_decoder_embedding_kernel: T")
    .Input("post_decoder_embedding_bias: T")
    .Output("output_ids: int32")
    .Output("parent_ids: int32")
    .Output("sequence_length: int32")
    .Attr("N: int")
    .Attr("T: {float, half}")
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
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        int beam_width, max_seq_len;
        c->GetAttr("beam_width", &beam_width);
        c->GetAttr("max_seq_len", &max_seq_len);

        int rank = c->Rank(c->input(0));
        assert(rank == 3);

        // calculate batch size
        tf::shape_inference::DimensionOrConstant max_seq_dim(max_seq_len);
        tf::shape_inference::DimensionOrConstant beam_width_dim(beam_width);
        tf::shape_inference::DimensionHandle batchxbeam_dim = c->Dim(c->input(0), 0);
        tf::shape_inference::DimensionHandle batch_dim;
        TF_RETURN_IF_ERROR(c->Divide(batchxbeam_dim, beam_width_dim, true, &batch_dim));

        if (beam_width > 1) {
            c->set_output(0, c->MakeShape({max_seq_len, batch_dim, beam_width_dim}));
            c->set_output(1, c->MakeShape({max_seq_len, batch_dim, beam_width_dim}));
            c->set_output(2, c->MakeShape({batch_dim, beam_width_dim}));
        }
        else {
            c->set_output(0, c->MakeShape({max_seq_len, batch_dim}));
            c->set_output(1, c->MakeShape({max_seq_len, batch_dim}));
            c->set_output(2, c->MakeShape({batch_dim}));
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

template<typename Device, typename T>
class DecodingOp: public BaseOp<T> {
public:
    explicit DecodingOp(tf::OpKernelConstruction* context): BaseOp<T>(context)
    {
        try {
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
            cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in");
            ft::check_cuda_error(cudaGetDeviceProperties(&prop_, 0));
        }
        catch (std::runtime_error& error) {
            OP_REQUIRES(context, false, tf::errors::Internal(error.what()));
        }
    }

    ~DecodingOp()
    {
        delete cublas_algo_map_;
    }

    void Compute(tf::OpKernelContext* context) override
    {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        OP_REQUIRES(context,
                    context->num_inputs() == (num_layer_ * 22) + 8,
                    tf::errors::InvalidArgument("[ERROR] More or Less input arguments"));

        const size_t batch_size = (size_t)(context->input(0).dim_size(0) / beam_width_);
        const size_t mem_max_seq_len = (size_t)(context->input(0).dim_size(1));
        const size_t vocab_size = (size_t)(context->input(2 + num_layer_ * 22 + 3).dim_size(0));

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
            cublas_wrapper.setFP16GemmConfig();
        }
        else if (std::is_same<T, float>::value) {
            cublas_wrapper.setFP32GemmConfig();
        }

        ft::DecodingWeight<DataType> decoding_weights;
        decoding_weights.decoder_layer_weights.resize(num_layer_);

        for (int i = 0; i < num_layer_; i++) {
            this->get_tensor(context, 2 + i, &decoding_weights.decoder_layer_weights[i].pre_layernorm_weights.beta);
            this->get_tensor(context,
                             2 + num_layer_ * 1 + i,
                             &decoding_weights.decoder_layer_weights[i].pre_layernorm_weights.gamma);

            this->get_tensor(context,
                             2 + num_layer_ * 2 + i,
                             &decoding_weights.decoder_layer_weights[i].self_attention_weights.query_weight.kernel);
            this->get_tensor(context,
                             2 + num_layer_ * 3 + i,
                             &decoding_weights.decoder_layer_weights[i].self_attention_weights.query_weight.bias);
            this->get_tensor(
                context,
                2 + num_layer_ * 4 + i,
                &decoding_weights.decoder_layer_weights[i].self_attention_weights.attention_output_weight.kernel);
            this->get_tensor(
                context,
                2 + num_layer_ * 5 + i,
                &decoding_weights.decoder_layer_weights[i].self_attention_weights.attention_output_weight.bias);
            this->get_tensor(context,
                             2 + num_layer_ * 6 + i,
                             &decoding_weights.decoder_layer_weights[i].self_attn_layernorm_weights.beta);
            this->get_tensor(context,
                             2 + num_layer_ * 7 + i,
                             &decoding_weights.decoder_layer_weights[i].self_attn_layernorm_weights.gamma);

            this->get_tensor(context,
                             2 + num_layer_ * 8 + i,
                             &decoding_weights.decoder_layer_weights[i].cross_attention_weights.query_weight.kernel);
            this->get_tensor(context,
                             2 + num_layer_ * 9 + i,
                             &decoding_weights.decoder_layer_weights[i].cross_attention_weights.query_weight.bias);
            this->get_tensor(context,
                             2 + num_layer_ * 10 + i,
                             &decoding_weights.decoder_layer_weights[i].cross_attention_weights.key_weight.kernel);
            this->get_tensor(context,
                             2 + num_layer_ * 11 + i,
                             &decoding_weights.decoder_layer_weights[i].cross_attention_weights.key_weight.bias);
            this->get_tensor(context,
                             2 + num_layer_ * 12 + i,
                             &decoding_weights.decoder_layer_weights[i].cross_attention_weights.value_weight.kernel);
            this->get_tensor(context,
                             2 + num_layer_ * 13 + i,
                             &decoding_weights.decoder_layer_weights[i].cross_attention_weights.value_weight.bias);
            this->get_tensor(
                context,
                2 + num_layer_ * 14 + i,
                &decoding_weights.decoder_layer_weights[i].cross_attention_weights.attention_output_weight.kernel);
            this->get_tensor(
                context,
                2 + num_layer_ * 15 + i,
                &decoding_weights.decoder_layer_weights[i].cross_attention_weights.attention_output_weight.bias);
            this->get_tensor(context,
                             2 + num_layer_ * 16 + i,
                             &decoding_weights.decoder_layer_weights[i].cross_attn_layernorm_weights.beta);
            this->get_tensor(context,
                             2 + num_layer_ * 17 + i,
                             &decoding_weights.decoder_layer_weights[i].cross_attn_layernorm_weights.gamma);

            this->get_tensor(context,
                             2 + num_layer_ * 18 + i,
                             &decoding_weights.decoder_layer_weights[i].ffn_weights.intermediate_weight.kernel);
            this->get_tensor(context,
                             2 + num_layer_ * 19 + i,
                             &decoding_weights.decoder_layer_weights[i].ffn_weights.intermediate_weight.bias);
            this->get_tensor(context,
                             2 + num_layer_ * 20 + i,
                             &decoding_weights.decoder_layer_weights[i].ffn_weights.output_weight.kernel);
            this->get_tensor(context,
                             2 + num_layer_ * 21 + i,
                             &decoding_weights.decoder_layer_weights[i].ffn_weights.output_weight.bias);
        }

        this->get_tensor(context, 2 + num_layer_ * 22 + 0, &decoding_weights.post_decoder_layernorm.beta);
        this->get_tensor(context, 2 + num_layer_ * 22 + 1, &decoding_weights.post_decoder_layernorm.gamma);
        this->get_tensor(context, 2 + num_layer_ * 22 + 2, &decoding_weights.position_encoding_table);
        this->get_tensor(context, 2 + num_layer_ * 22 + 3, &decoding_weights.pre_decoder_embedding_table);
        this->get_tensor(context, 2 + num_layer_ * 22 + 4, &decoding_weights.post_decoder_embedding.kernel);
        this->get_tensor(context, 2 + num_layer_ * 22 + 5, &decoding_weights.post_decoder_embedding.bias);

        tf::Tensor* output_id_tensor = nullptr;
        tf::Tensor* parent_id_tensor = nullptr;
        tf::Tensor* sequence_length_tensor = nullptr;
        if (beam_width_ > 1) {
            OP_REQUIRES_OK(context,
                           context->allocate_output(
                               0,
                               {(long long int)max_seq_len_, (long long int)batch_size, (long long int)beam_width_},
                               &output_id_tensor));
            OP_REQUIRES_OK(context,
                           context->allocate_output(
                               1,
                               {(long long int)max_seq_len_, (long long int)batch_size, (long long int)beam_width_},
                               &parent_id_tensor));
            OP_REQUIRES_OK(context,
                           context->allocate_output(
                               2, {(long long int)batch_size, (long long int)beam_width_}, &sequence_length_tensor));
        }
        else {
            OP_REQUIRES_OK(context,
                           context->allocate_output(
                               0, {(long long int)max_seq_len_, (long long int)batch_size}, &output_id_tensor));
            OP_REQUIRES_OK(context,
                           context->allocate_output(
                               1, {(long long int)max_seq_len_, (long long int)batch_size}, &parent_id_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(2, {(long long int)batch_size}, &sequence_length_tensor));
        }
        int* output_ids = (int*)(output_id_tensor->flat<int>().data());
        int* parent_ids = (int*)(parent_id_tensor->flat<int>().data());
        int* sequence_length = (int*)(sequence_length_tensor->flat<int>().data());

        ft::Decoding<DataType> decoding = ft::Decoding<DataType>(batch_size,
                                                                 max_seq_len_,
                                                                 mem_max_seq_len,
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
                                                                 temperature_,
                                                                 len_penalty_,
                                                                 repetition_penalty_,
                                                                 stream,
                                                                 &cublas_wrapper,
                                                                 &allocator,
                                                                 false,
                                                                 &prop_);

        std::vector<ft::Tensor> input_tensors = std::vector<ft::Tensor>{this->convert_tensor(context->input(0)),
                                                                        this->convert_int_tensor(context->input(1))};

        std::vector<ft::Tensor> output_tensors = std::vector<ft::Tensor>{
            ft::Tensor{
                ft::MEMORY_GPU, ft::TYPE_INT32, {(size_t)max_seq_len_, batch_size, (size_t)beam_width_}, output_ids},
            ft::Tensor{
                ft::MEMORY_GPU, ft::TYPE_INT32, {(size_t)max_seq_len_, batch_size, (size_t)beam_width_}, parent_ids},
            ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, {batch_size, (size_t)beam_width_}, sequence_length}};

        try {
            decoding.forward(&output_tensors, &input_tensors, &decoding_weights);
        }
        catch (std::runtime_error& error) {
            std::cout << tf::errors::Internal(error.what());
            ft::FT_CHECK(false);
        }
        catch (...) {
            std::cout << tf::errors::Internal("Runtime error") << std::endl;
            ft::FT_CHECK(false);
        }
    }

private:
    int max_seq_len_ = 0, beam_width_ = 1;
    int head_num_ = 0, size_per_head_ = 0, inter_size_;
    int num_layer_ = 0, start_id_ = -1, end_id_ = -1;
    float beam_search_diversity_rate_ = 1.0;
    float temperature_;
    float len_penalty_;
    float repetition_penalty_;
    int top_k_ = 0;
    float top_p_ = 0.0f;
    ft::cublasAlgoMap* cublas_algo_map_;
    cudaDeviceProp prop_;
    typedef TFTraits<T> traits_;
    typedef typename traits_::DataType DataType;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                                                                \
    REGISTER_KERNEL_BUILDER(Name("Decoding").Device(tf::DEVICE_GPU).TypeConstraint<T>("T"), DecodingOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
