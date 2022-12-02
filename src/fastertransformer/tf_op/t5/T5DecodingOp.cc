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

#include "src/fastertransformer/models/t5/T5Decoding.h"
#include "src/fastertransformer/tf_op/BaseOp.h"

namespace ft = fastertransformer;
namespace tf = tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("T5Decoding")
    .Input("memory_tensor: T")                    // 0     [batch_size, mem_max_seq_len, mem_hidden_units]
    .Input("memory_sequence_length: int32")       // 1
    .Input("pre_decoder_layernorm_beta: N * T")   // 2     pre-layernorm
    .Input("pre_decoder_layernorm_gamma: N * T")  // 3     pre-layernorm
    .Input("self_qkv_kernel: N * T")              // 4     self attention fused qkv weight
    .Input("self_qkv_bias: N * T")                // 5     self attention fused qkv bias
    .Input("self_output_kernel: N * T")           // 6     self attention output weight
    .Input("self_output_bias: N * T")             // 7     self attention output bias
    .Input("self_beta: N * T")                    // 8     self attention layernorm beta
    .Input("self_gamma: N * T")                   // 9     self attention layernorm gamma
    .Input("cross_q_kernel: N * T")               // 10    cross attention query weights
    .Input("cross_q_bias: N * T")                 // 11    cross attention query bias
    .Input("cross_k_kernel: N * T")               // 12    cross attention key weights
    .Input("cross_k_bias: N * T")                 // 13    cross attention key bias
    .Input("cross_v_kernel: N * T")               // 14    cross attention value weights
    .Input("cross_v_bias: N * T")                 // 15    cross attention value bias
    .Input("cross_output_kernel: N * T")          // 16    cross attention output weights
    .Input("cross_output_bias: N * T")            // 17    cross attention output bias
    .Input("cross_beta: N * T")                   // 18    cross attention layernorm beta
    .Input("cross_gamma: N * T")                  // 19    cross attention layernorm gamma
    .Input("ffn_inter_kernel: N * T")             // 20    dense layer intermediate weights
    .Input("ffn_inter_bias: N * T")               // 21    dense layer intermediate bias
    .Input("ffn_inter2_kernel: N * T")            // 22    dense layer (gated) intermediate weights
    .Input("ffn_inter2_bias: N * T")              // 23    dense layer (gated) intermediate biass
    .Input("ffn_output_kernel: N * T")            // 24    dense layer output weights
    .Input("ffn_output_bias: N * T")              // 25    dense layer output bias
    .Input("post_decoder_layernorm_beta: T")      // 26    output layernorm beta
    .Input("post_decoder_layernorm_gamma: T")     // 27    output layernorm gamma
    .Input("pre_decoder_embedding_table: T")      // 28    pre decoder embedding table
    .Input("post_decoder_embedding_kernel: T")    // 29    post decoder embedding weight
    .Input("post_decoder_embedding_bias: T")      // 30    post decoder embedding bias
    .Input("output_absolute_or_relative_position_embedding: T")  // 31    position embeddings
    .Output("output_ids: int32")
    .Output("sequence_length: int32")
    .Output("ouput_log_probs: float")
    .Output("cum_log_probs: float")
    .Attr("N: int")
    .Attr("T: {float, half}")
    .Attr("max_seq_len: int >= 1")
    .Attr("beam_width: int >= 1")
    .Attr("head_num: int >= 1")
    .Attr("head_size: int >= 1")
    .Attr("inter_size: int >= 1")
    .Attr("num_layer: int >= 1")
    .Attr("d_model: int >= 1")
    .Attr("max_distance: int >= 1")
    .Attr("q_scaling: float = 1.0")
    .Attr("num_bucket: int >= 1")
    .Attr("start_id: int >= 0")
    .Attr("end_id: int >= 0")
    .Attr("beam_search_diversity_rate: float = 0.0")
    .Attr("top_k: int >= 0")
    .Attr("top_p: float")
    .Attr("temperature: float = 1.0")
    .Attr("len_penalty: float = 0.0")
    .Attr("repetition_penalty: float = 1.0")
    .Attr("random_seed: int = 0")
    .Attr("return_output_log_probs: bool = False")
    .Attr("return_cum_log_probs: bool = False")
    .Attr("t5_with_bias: bool = False")
    .Attr("tie_word_embeddings: bool = False")
    .Attr("activation_type: {'relu', 'gated-gelu'}")
    .Attr("position_embedding_type: int = 0")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        int  beam_width, max_seq_len;
        bool return_cum_log_probs, return_output_log_probs;
        c->GetAttr("beam_width", &beam_width);
        c->GetAttr("max_seq_len", &max_seq_len);
        c->GetAttr("return_cum_log_probs", &return_cum_log_probs);
        c->GetAttr("return_output_log_probs", &return_output_log_probs);

        int rank = c->Rank(c->input(0));
        assert(rank == 3);

        // calculate batch size
        tf::shape_inference::DimensionOrConstant max_seq_dim(max_seq_len);
        tf::shape_inference::DimensionOrConstant beam_width_dim(beam_width);
        tf::shape_inference::DimensionHandle     batch_dim = c->Dim(c->input(0), 0);

        if (beam_width > 1) {
            c->set_output(0, c->MakeShape({batch_dim, beam_width_dim, max_seq_len}));
            c->set_output(1, c->MakeShape({batch_dim, beam_width_dim}));
            c->set_output(2, c->MakeShape({batch_dim, beam_width_dim, max_seq_len}));
            c->set_output(3, c->MakeShape({batch_dim, beam_width_dim}));
        }
        else {
            c->set_output(0, c->MakeShape({batch_dim, max_seq_len}));
            c->set_output(1, c->MakeShape({batch_dim}));
            c->set_output(2, c->MakeShape({batch_dim, max_seq_len}));
            c->set_output(3, c->MakeShape({batch_dim}));
        }

        if (!return_output_log_probs) {
            c->set_output(2, c->MakeShape({}));
        }
        if (!return_cum_log_probs) {
            c->set_output(3, c->MakeShape({}));
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
class T5DecodingOp: public BaseOp<T> {
public:
    explicit T5DecodingOp(tf::OpKernelConstruction* context): BaseOp<T>(context)
    {
        try {
            OP_REQUIRES_OK(context, context->GetAttr("max_seq_len", &max_seq_len_));
            OP_REQUIRES_OK(context, context->GetAttr("beam_width", &beam_width_));
            OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
            OP_REQUIRES_OK(context, context->GetAttr("head_size", &head_size_));
            OP_REQUIRES_OK(context, context->GetAttr("inter_size", &inter_size_));
            OP_REQUIRES_OK(context, context->GetAttr("num_layer", &num_layer_));
            OP_REQUIRES_OK(context, context->GetAttr("d_model", &d_model_));
            OP_REQUIRES_OK(context, context->GetAttr("num_bucket", &num_bucket_));
            OP_REQUIRES_OK(context, context->GetAttr("max_distance", &max_distance_));
            OP_REQUIRES_OK(context, context->GetAttr("q_scaling", &q_scaling_));
            OP_REQUIRES_OK(context, context->GetAttr("start_id", &start_id_));
            OP_REQUIRES_OK(context, context->GetAttr("end_id", &end_id_));
            OP_REQUIRES_OK(context, context->GetAttr("beam_search_diversity_rate", &beam_search_diversity_rate_));
            OP_REQUIRES_OK(context, context->GetAttr("top_k", &top_k_));
            OP_REQUIRES_OK(context, context->GetAttr("top_p", &top_p_));
            OP_REQUIRES_OK(context, context->GetAttr("temperature", &temperature_));
            OP_REQUIRES_OK(context, context->GetAttr("len_penalty", &len_penalty_));
            OP_REQUIRES_OK(context, context->GetAttr("repetition_penalty", &repetition_penalty_));
            OP_REQUIRES_OK(context, context->GetAttr("t5_with_bias", &t5_with_bias_));
            OP_REQUIRES_OK(context, context->GetAttr("tie_word_embeddings", &tie_word_embeddings_));
            OP_REQUIRES_OK(context, context->GetAttr("return_output_log_probs", &return_output_log_probs_));
            OP_REQUIRES_OK(context, context->GetAttr("return_cum_log_probs", &return_cum_log_probs_));

            std::string activation_type;
            OP_REQUIRES_OK(context, context->GetAttr("activation_type", &activation_type));
            activation_type_ = ft::getActivationType(activation_type);

            int position_embedding_type;
            OP_REQUIRES_OK(context, context->GetAttr("position_embedding_type", &position_embedding_type));
            position_embedding_type_ = ft::PositionEmbeddingType(position_embedding_type);

            long int random_seed = 0;
            OP_REQUIRES_OK(context, context->GetAttr("random_seed", &random_seed));
            random_seed_ = static_cast<unsigned long long>(random_seed);

            sm_              = ft::getSMVersion();
            cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in");
            ft::check_cuda_error(cudaGetDeviceProperties(&prop_, 0));
        }
        catch (std::runtime_error& error) {
            OP_REQUIRES(context, false, tf::errors::Internal(error.what()));
        }
    }

    ~T5DecodingOp()
    {
        delete cublas_algo_map_;
    }

    void Compute(tf::OpKernelContext* context) override
    {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        OP_REQUIRES(context,
                    context->num_inputs() == (num_layer_ * 24) + 8,
                    tf::errors::InvalidArgument("[ERROR] More or Less input arguments"));

        const size_t batch_size      = (size_t)(context->input(0).dim_size(0));
        const size_t mem_max_seq_len = (size_t)(context->input(0).dim_size(1));
        const size_t vocab_size      = (size_t)(context->input(2 + num_layer_ * 24 + 2).dim_size(0));

        bool use_gated_activation = isGatedActivation(activation_type_);

        const cudaStream_t& stream        = context->eigen_device<Device>().stream();
        cublasHandle_t      cublas_handle = this->get_cublas_handler();
        cublasSetStream(cublas_handle, stream);
        ft::Allocator<ft::AllocatorType::TF> allocator(context, stream);
        ft::cublasMMWrapper                  cublas_wrapper = ft::cublasMMWrapper(cublas_handle,
                                                                 this->get_cublaslt_handler(),
                                                                 stream,
                                                                 cublas_algo_map_,
                                                                 this->get_cublas_wrapper_mutex(),
                                                                 &allocator);

        if (std::is_same<T, Eigen::half>::value) {
            cublas_wrapper.setFP16GemmConfig();
        }
#ifdef ENABLE_BF16
        else if (std::is_same<T, Eigen::bfloat16>::value) {
            cublas_wrapper.setBF16GemmConfig();
        }
#endif
        else if (std::is_same<T, float>::value) {
            cublas_wrapper.setFP32GemmConfig();
        }

        ft::T5DecodingWeight<DataType> t5_decoding_weights;
        t5_decoding_weights.resizeLayer(num_layer_);
        t5_decoding_weights.setT5StructureDiff(t5_with_bias_, use_gated_activation, position_embedding_type_);
        const int hidden_dim = head_num_ * head_size_;

        for (int i = 0; i < num_layer_; i++) {
            ft::T5DecoderLayerWeight<DataType>* layer_weight = t5_decoding_weights.decoder_layer_weights[i];
            this->get_tensor(context, 2 + num_layer_ * 1 + i, &layer_weight->pre_layernorm_weights.gamma);
            this->get_tensor(
                context, 2 + num_layer_ * 2 + i, &layer_weight->self_attention_weights.query_weight.kernel);
            this->get_tensor(
                context, 2 + num_layer_ * 4 + i, &layer_weight->self_attention_weights.attention_output_weight.kernel);
            this->get_tensor(context, 2 + num_layer_ * 7 + i, &layer_weight->self_attn_layernorm_weights.gamma);
            this->get_tensor(
                context, 2 + num_layer_ * 8 + i, &layer_weight->cross_attention_weights.query_weight.kernel);
            this->get_tensor(
                context, 2 + num_layer_ * 10 + i, &layer_weight->cross_attention_weights.key_weight.kernel);
            this->get_tensor(
                context, 2 + num_layer_ * 12 + i, &layer_weight->cross_attention_weights.value_weight.kernel);
            this->get_tensor(context,
                             2 + num_layer_ * 14 + i,
                             &layer_weight->cross_attention_weights.attention_output_weight.kernel);
            this->get_tensor(context, 2 + num_layer_ * 17 + i, &layer_weight->cross_attn_layernorm_weights.gamma);
            this->get_tensor(context, 2 + num_layer_ * 18 + i, &layer_weight->ffn_weights.intermediate_weight.kernel);
            if (use_gated_activation) {
                this->get_tensor(
                    context, 2 + num_layer_ * 20 + i, &layer_weight->ffn_weights.intermediate_weight2.kernel);
            }
            this->get_tensor(context, 2 + num_layer_ * 22 + i, &layer_weight->ffn_weights.output_weight.kernel);

            if (t5_with_bias_) {
                this->get_tensor(context, 2 + num_layer_ * 0 + i, &layer_weight->pre_layernorm_weights.beta);
                this->get_tensor(
                    context, 2 + num_layer_ * 3 + i, &layer_weight->self_attention_weights.query_weight.bias);
                this->get_tensor(context,
                                 2 + num_layer_ * 5 + i,
                                 &layer_weight->self_attention_weights.attention_output_weight.bias);
                this->get_tensor(context, 2 + num_layer_ * 6 + i, &layer_weight->self_attn_layernorm_weights.beta);
                this->get_tensor(
                    context, 2 + num_layer_ * 9 + i, &layer_weight->cross_attention_weights.query_weight.bias);
                this->get_tensor(
                    context, 2 + num_layer_ * 11 + i, &layer_weight->cross_attention_weights.key_weight.bias);
                this->get_tensor(
                    context, 2 + num_layer_ * 13 + i, &layer_weight->cross_attention_weights.value_weight.bias);
                this->get_tensor(context,
                                 2 + num_layer_ * 15 + i,
                                 &t5_decoding_weights.decoder_layer_weights[i]
                                      ->cross_attention_weights.attention_output_weight.bias);
                this->get_tensor(context, 2 + num_layer_ * 16 + i, &layer_weight->cross_attn_layernorm_weights.beta);
                this->get_tensor(context, 2 + num_layer_ * 19 + i, &layer_weight->ffn_weights.intermediate_weight.bias);
                if (use_gated_activation) {
                    this->get_tensor(
                        context, 2 + num_layer_ * 21 + i, &layer_weight->ffn_weights.intermediate_weight2.kernel);
                }
                this->get_tensor(context, 2 + num_layer_ * 23 + i, &layer_weight->ffn_weights.output_weight.bias);
            }
        }

        this->get_tensor(context, 2 + num_layer_ * 24 + 1, &t5_decoding_weights.post_decoder_layernorm.gamma);
        this->get_tensor(context, 2 + num_layer_ * 24 + 2, &t5_decoding_weights.pre_decoder_embedding_table);
        this->get_tensor(context, 2 + num_layer_ * 24 + 3, &t5_decoding_weights.post_decoder_embedding.kernel);
        this->get_tensor(
            context, 2 + num_layer_ * 24 + 5, &t5_decoding_weights.absolute_or_relative_position_embedding);
        if (t5_with_bias_) {
            this->get_tensor(context, 2 + num_layer_ * 24 + 0, &t5_decoding_weights.post_decoder_layernorm.beta);
            this->get_tensor(context, 2 + num_layer_ * 24 + 4, &t5_decoding_weights.post_decoder_embedding.bias);
        }

        tf::Tensor* output_id_tensor        = nullptr;
        tf::Tensor* sequence_length_tensor  = nullptr;
        tf::Tensor* output_log_probs_tensor = nullptr;
        tf::Tensor* cum_log_probs_tensor    = nullptr;
        if (beam_width_ > 1) {
            OP_REQUIRES_OK(context,
                           context->allocate_output(
                               0,
                               {(long long int)batch_size, (long long int)beam_width_, (long long int)max_seq_len_},
                               &output_id_tensor));
            OP_REQUIRES_OK(context,
                           context->allocate_output(
                               1, {(long long int)batch_size, (long long int)beam_width_}, &sequence_length_tensor));
            if (return_output_log_probs_) {
                OP_REQUIRES_OK(context,
                               context->allocate_output(
                                   2,
                                   {(long long int)batch_size, (long long int)beam_width_, (long long int)max_seq_len_},
                                   &output_log_probs_tensor));
            }
            else {
                // allocate dummy variable
                OP_REQUIRES_OK(context, context->allocate_output(2, {}, &output_log_probs_tensor));
            }
            if (return_cum_log_probs_) {
                OP_REQUIRES_OK(context,
                               context->allocate_output(
                                   3, {(long long int)batch_size, (long long int)beam_width_}, &cum_log_probs_tensor));
            }
            else {
                // allocate dummy variable
                OP_REQUIRES_OK(context, context->allocate_output(3, {}, &cum_log_probs_tensor));
            }
        }
        else {
            OP_REQUIRES_OK(context,
                           context->allocate_output(
                               0, {(long long int)batch_size, (long long int)max_seq_len_}, &output_id_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, {(long long int)batch_size}, &sequence_length_tensor));
            if (return_output_log_probs_) {
                OP_REQUIRES_OK(context,
                               context->allocate_output(2,
                                                        {(long long int)batch_size, (long long int)max_seq_len_},
                                                        &output_log_probs_tensor));
            }
            else {
                // allocate dummy variable
                OP_REQUIRES_OK(context, context->allocate_output(2, {}, &output_log_probs_tensor));
            }
            if (return_cum_log_probs_) {
                OP_REQUIRES_OK(context,
                               context->allocate_output(3, {(long long int)batch_size}, &cum_log_probs_tensor));
            }
            else {
                // allocate dummy variable
                OP_REQUIRES_OK(context, context->allocate_output(3, {}, &cum_log_probs_tensor));
            }
        }

        ft::NcclParam tensor_para;
        ft::NcclParam pipeline_para;

        ft::T5Decoding<DataType> decoding = ft::T5Decoding<DataType>(batch_size,
                                                                     max_seq_len_,
                                                                     mem_max_seq_len,
                                                                     beam_width_,
                                                                     head_num_,
                                                                     head_size_,
                                                                     inter_size_,
                                                                     d_model_,
                                                                     num_layer_,
                                                                     vocab_size,
                                                                     num_bucket_,
                                                                     0,  // expert_num
                                                                     max_distance_,
                                                                     0,  // moe_k
                                                                     q_scaling_,
                                                                     start_id_,
                                                                     end_id_,
                                                                     beam_search_diversity_rate_,
                                                                     top_k_,
                                                                     top_p_,
                                                                     temperature_,
                                                                     len_penalty_,
                                                                     repetition_penalty_,
                                                                     {},  // moe_layer_index
                                                                     stream,
                                                                     &cublas_wrapper,
                                                                     &allocator,
                                                                     false,
                                                                     &prop_,
                                                                     tensor_para,
                                                                     pipeline_para,
                                                                     activation_type_,
                                                                     tie_word_embeddings_);

        // assemble input tensors
        ft::TensorMap input_tensors(
            {{"encoder_output", this->convert_tensor(context->input(0))},
             {"encoder_sequence_length", this->convert_int_tensor(context->input(1))},
             {"runtime_top_p", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, {1}, &top_p_}},
             {"runtime_top_k", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT32, {1}, &top_k_}},
             {"temperature", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, {1}, &temperature_}},
             {"len_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, {1}, &len_penalty_}},
             {"repetition_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, {1}, &repetition_penalty_}},
             {"random_seed", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT64, {1}, &random_seed_}}});

        // these are now parsed using the DynamicDecodeLayer, so need to be inserted into input_tensors
        if (beam_width_ > 1) {
            input_tensors.insert({"beam_search_diversity_rate",
                                  ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, {1}, &beam_search_diversity_rate_}});
        }

        // assemble output_tensors
        int* output_ids      = (int*)(output_id_tensor->flat<int>().data());
        int* sequence_length = (int*)(sequence_length_tensor->flat<int>().data());

        ft::TensorMap output_tensors(
            {{"output_ids",
              ft::Tensor{
                  ft::MEMORY_GPU, ft::TYPE_INT32, {batch_size, (size_t)beam_width_, (size_t)max_seq_len_}, output_ids}},
             {"sequence_length",
              ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, {batch_size, (size_t)beam_width_}, sequence_length}}});

        // in case cum_log_probs is requested
        if (return_output_log_probs_) {
            float* output_log_probs = (float*)(output_log_probs_tensor->flat<float>().data());
            output_tensors.insert("output_log_probs",
                                  ft::Tensor{ft::MEMORY_GPU,
                                             ft::TYPE_FP32,
                                             {batch_size, (size_t)beam_width_, (size_t)max_seq_len_},
                                             output_log_probs});
        }

        if (return_cum_log_probs_) {
            float* cum_log_probs = (float*)(cum_log_probs_tensor->flat<float>().data());
            output_tensors.insert(
                "cum_log_probs",
                ft::Tensor{ft::MEMORY_GPU, ft::TYPE_FP32, {batch_size, (size_t)beam_width_}, cum_log_probs});
        }

        try {
            decoding.forward(&output_tensors, &input_tensors, &t5_decoding_weights);
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
    int                       max_seq_len_  = 0;
    int                       beam_width_   = 1;
    int                       head_num_     = 0;
    int                       head_size_    = 0;
    int                       inter_size_   = 0;
    int                       num_layer_    = 0;
    int                       d_model_      = 0;
    int                       num_bucket_   = 0;
    int                       max_distance_ = 0;
    float                     q_scaling_    = 1.0f;
    int                       start_id_;
    int                       end_id_;
    float                     beam_search_diversity_rate_ = 1.0;
    float                     temperature_;
    float                     len_penalty_;
    float                     repetition_penalty_;
    bool                      t5_with_bias_;
    bool                      tie_word_embeddings_;
    bool                      return_output_log_probs_;
    bool                      return_cum_log_probs_;
    int                       top_k_       = 0;
    float                     top_p_       = 0.0f;
    unsigned long long int    random_seed_ = 0;
    ft::PositionEmbeddingType position_embedding_type_;
    ft::ActivationType        activation_type_;

    int                                sm_;
    ft::cublasAlgoMap*                 cublas_algo_map_;
    cudaDeviceProp                     prop_;
    typedef TFTraits<T>                traits_;
    typedef typename traits_::DataType DataType;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                                                                \
    REGISTER_KERNEL_BUILDER(Name("T5Decoding").Device(tf::DEVICE_GPU).TypeConstraint<T>("T"),                          \
                            T5DecodingOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
