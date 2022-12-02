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

#include "src/fastertransformer/models/t5/T5Encoder.h"
#include "src/fastertransformer/tf_op/BaseOp.h"

namespace ft = fastertransformer;
namespace tf = tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("T5Encoder")
    .Input("input_ids: int32")                                   // 0
    .Input("sequence_length: int32")                             // 1
    .Input("attr_output_layernorm_beta: N * T")                  // 2
    .Input("attr_output_layernorm_gamma: N * T")                 // 3
    .Input("attr_q_kernel: N * T")                               // 4
    .Input("attr_q_bias: N * T")                                 // 5
    .Input("attr_k_kernel: N * T")                               // 6
    .Input("attr_k_bias: N * T")                                 // 7
    .Input("attr_v_kernel: N * T")                               // 8
    .Input("attr_v_bias: N * T")                                 // 9
    .Input("attr_output_kernel: N * T")                          // 10
    .Input("attr_output_bias: N * T")                            // 11
    .Input("ffn_layernorm_beta: N * T")                          // 12
    .Input("ffn_layernorm_gamma: N * T")                         // 13
    .Input("ffn_inter_kernel: N * T")                            // 14
    .Input("ffn_inter_bias: N * T")                              // 15
    .Input("ffn_inter2_kernel: N * T")                           // 16
    .Input("ffn_inter2_bias: N * T")                             // 17
    .Input("ffn_output_kernel: N * T")                           // 18
    .Input("ffn_output_bias: N * T")                             // 19
    .Input("output_layernorm_beta: T")                           // 20
    .Input("output_layernorm_gamma: T")                          // 21
    .Input("output_absolute_or_relative_position_embedding: T")  // 22
    .Input("output_embedding_table: T")                          // 23
    .Output("output: T")
    .Attr("N: int")
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    .Attr("head_size: int >= 1")
    .Attr("inter_size: int >= 1")
    .Attr("d_model: int >= 1")
    .Attr("num_layer: int >= 1")
    .Attr("num_bucket: int >= 1")
    .Attr("max_distance: int >= 1")
    .Attr("remove_padding: bool = False")
    .Attr("q_scaling: float")
    .Attr("t5_with_bias: bool = False")
    .Attr("activation_type: {'relu', 'gated-gelu'}")
    .Attr("position_embedding_type: int = 0")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        int d_model;
        c->GetAttr("d_model", &d_model);

        int rank = c->Rank(c->input(0));
        assert(rank == 2);

        // calculate batch size and sequence length from input
        tf::shape_inference::DimensionHandle batch_dim    = c->Dim(c->input(0), 0);
        tf::shape_inference::DimensionHandle from_seq_len = c->Dim(c->input(0), 1);

        c->set_output(0, c->MakeShape({batch_dim, from_seq_len, d_model}));

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

// this part needs to be adapted
template<typename Device, typename T>
class T5EncoderOp: public BaseOp<T> {
public:
    explicit T5EncoderOp(tf::OpKernelConstruction* context): BaseOp<T>(context)
    {
        try {
            OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
            OP_REQUIRES_OK(context, context->GetAttr("head_size", &head_size_));  // head_size in torch version
            OP_REQUIRES_OK(context, context->GetAttr("inter_size", &inter_size_));
            OP_REQUIRES_OK(context, context->GetAttr("num_layer", &num_layer_));
            OP_REQUIRES_OK(context, context->GetAttr("d_model", &d_model_));
            OP_REQUIRES_OK(context, context->GetAttr("num_bucket", &num_bucket_));
            OP_REQUIRES_OK(context, context->GetAttr("max_distance", &max_distance_));

            OP_REQUIRES_OK(context, context->GetAttr("remove_padding", &remove_padding_));
            OP_REQUIRES_OK(context, context->GetAttr("q_scaling", &q_scaling_));
            OP_REQUIRES_OK(context, context->GetAttr("t5_with_bias", &t5_with_bias_));

            std::string activation_type;
            OP_REQUIRES_OK(context, context->GetAttr("activation_type", &activation_type));
            activation_type_ = ft::getActivationType(activation_type);

            int position_embedding_type;
            OP_REQUIRES_OK(context, context->GetAttr("position_embedding_type", &position_embedding_type));
            position_embedding_type_ = ft::PositionEmbeddingType(position_embedding_type);

            sm_              = ft::getSMVersion();
            cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in");
        }
        catch (std::runtime_error& error) {
            OP_REQUIRES(context, false, tf::errors::Internal(error.what()));
        }
    }

    ~T5EncoderOp()
    {
        delete cublas_algo_map_;
    }

    // actual computation goes here
    void Compute(tf::OpKernelContext* context) override
    {
        OP_REQUIRES(context,
                    context->num_inputs() == (num_layer_ * 18) + 6,
                    tf::errors::InvalidArgument("[ERROR] More or Less input arguments"));

        const size_t batch_size_   = (size_t)context->input(0).dim_size(0);
        const size_t from_seq_len_ = (size_t)context->input(0).dim_size(1);

        bool use_gated_activation = isGatedActivation(activation_type_);

        OP_REQUIRES(context,
                    batch_size_ == (size_t)context->input(1).dim_size(0),
                    tf::errors::InvalidArgument("[ERROR] invalid shape"));

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

        ft::T5EncoderWeight<DataType> t5_encoder_weights;
        t5_encoder_weights.resizeLayer(num_layer_);
        t5_encoder_weights.setT5StructureDiff(t5_with_bias_, use_gated_activation, position_embedding_type_);

        // get weights from input tensors
        for (int i = 0; i < num_layer_; i++) {
            ft::T5EncoderLayerWeight<DataType>* layer_weight = t5_encoder_weights.t5_encoder_layer_weights[i];

            this->get_tensor(context, 2 + num_layer_ * 1 + i, &layer_weight->attn_layernorm_weights_.gamma);
            this->get_tensor(context, 2 + num_layer_ * 2 + i, &layer_weight->attention_weights_.query_weight.kernel);
            this->get_tensor(context, 2 + num_layer_ * 4 + i, &layer_weight->attention_weights_.key_weight.kernel);
            this->get_tensor(context, 2 + num_layer_ * 6 + i, &layer_weight->attention_weights_.value_weight.kernel);
            this->get_tensor(
                context, 2 + num_layer_ * 8 + i, &layer_weight->attention_weights_.attention_output_weight.kernel);
            this->get_tensor(context, 2 + num_layer_ * 11 + i, &layer_weight->ffn_layernorm_weights_.gamma);
            this->get_tensor(context, 2 + num_layer_ * 12 + i, &layer_weight->ffn_weights_.intermediate_weight.kernel);
            if (use_gated_activation) {
                this->get_tensor(
                    context, 2 + num_layer_ * 14 + i, &layer_weight->ffn_weights_.intermediate_weight2.kernel);
            }
            this->get_tensor(context, 2 + num_layer_ * 16 + i, &layer_weight->ffn_weights_.output_weight.kernel);

            if (t5_with_bias_) {
                this->get_tensor(context, 2 + num_layer_ * 0 + i, &layer_weight->attn_layernorm_weights_.beta);
                this->get_tensor(context, 2 + num_layer_ * 3 + i, &layer_weight->attention_weights_.query_weight.bias);
                this->get_tensor(context, 2 + num_layer_ * 5 + i, &layer_weight->attention_weights_.key_weight.bias);
                this->get_tensor(context, 2 + num_layer_ * 7 + i, &layer_weight->attention_weights_.value_weight.bias);
                this->get_tensor(
                    context, 2 + num_layer_ * 9 + i, &layer_weight->attention_weights_.attention_output_weight.bias);
                this->get_tensor(context, 2 + num_layer_ * 10 + i, &layer_weight->ffn_layernorm_weights_.beta);
                this->get_tensor(
                    context, 2 + num_layer_ * 13 + i, &layer_weight->ffn_weights_.intermediate_weight.bias);
                if (use_gated_activation) {
                    this->get_tensor(
                        context, 2 + num_layer_ * 15 + i, &layer_weight->ffn_weights_.intermediate_weight2.bias);
                }
                this->get_tensor(context, 2 + num_layer_ * 17 + i, &layer_weight->ffn_weights_.output_weight.bias);
            }
        }

        // post transformer weights
        if (t5_with_bias_) {
            this->get_tensor(
                context, 2 + num_layer_ * 18 + 0, &t5_encoder_weights.post_transformer_layernorm_weights.beta);
        }
        this->get_tensor(
            context, 2 + num_layer_ * 18 + 1, &t5_encoder_weights.post_transformer_layernorm_weights.gamma);
        this->get_tensor(context, 2 + num_layer_ * 18 + 2, &t5_encoder_weights.absolute_or_relative_position_embedding);
        this->get_tensor(context, 2 + num_layer_ * 18 + 3, &t5_encoder_weights.embedding_table);

        // NOTE: fmha doesn't support t5-style relative position bias
        ft::AttentionType attention_type =
            ft::getAttentionType<DataType>(head_size_, sm_, remove_padding_, from_seq_len_, false);

        ft::NcclParam tensor_para;
        ft::NcclParam pipeline_para;

        ft::T5Encoder<DataType> t5_encoder = ft::T5Encoder<DataType>(batch_size_,
                                                                     from_seq_len_,
                                                                     head_num_,
                                                                     head_size_,
                                                                     inter_size_,
                                                                     d_model_,
                                                                     num_layer_,
                                                                     num_bucket_,
                                                                     0,  // expert_num
                                                                     max_distance_,
                                                                     0,  // moe_k
                                                                     sm_,
                                                                     q_scaling_,
                                                                     {},  // moe_layer_index
                                                                     stream,
                                                                     &cublas_wrapper,
                                                                     &allocator,
                                                                     false,
                                                                     attention_type,
                                                                     false,  // sparisty is not supported in TF ops
                                                                     activation_type_,
                                                                     ft::LayerNormType::pre_layernorm,
                                                                     tensor_para,
                                                                     pipeline_para);

        tf::Tensor* output = nullptr;
        OP_REQUIRES_OK(
            context,
            context->allocate_output(0, {(long int)batch_size_, (long int)from_seq_len_, (long int)d_model_}, &output));
        DataType* out_tensor = reinterpret_cast<DataType*>(output->flat<T>().data());

        ft::TensorMap input_tensors({{"input_ids", this->convert_int_tensor(context->input(0))},
                                     {"sequence_length", this->convert_int_tensor(context->input(1))}});

        ft::DataType data_type = ft::getTensorType<DataType>();

        ft::TensorMap output_tensors({{"output_hidden_state",
                                       ft::Tensor{ft::MEMORY_GPU,
                                                  data_type,
                                                  std::vector<size_t>{batch_size_, from_seq_len_, (size_t)d_model_},
                                                  out_tensor}}});

        try {
            t5_encoder.forward(&output_tensors, &input_tensors, &t5_encoder_weights);
        }
        catch (std::runtime_error& error) {
            std::cout << tf::errors::Internal(error.what());
            ft::FT_CHECK(false);
        }
        catch (...) {
            std::cout << tf::errors::Internal("Runtime error");
            ft::FT_CHECK(false);
        }
    }

private:
    int                       head_num_     = 0;
    int                       head_size_    = 0;
    int                       inter_size_   = 0;
    int                       d_model_      = 0;
    int                       num_layer_    = 0;
    int                       num_bucket_   = 0;
    int                       max_distance_ = 0;
    float                     q_scaling_    = 1.0f;
    bool                      remove_padding_;
    bool                      t5_with_bias_;
    ft::PositionEmbeddingType position_embedding_type_;
    ft::ActivationType        activation_type_;

    int                                sm_;
    ft::cublasAlgoMap*                 cublas_algo_map_;
    typedef TFTraits<T>                traits_;
    typedef typename traits_::DataType DataType;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                                                                \
    REGISTER_KERNEL_BUILDER(Name("T5Encoder").Device(tf::DEVICE_GPU).TypeConstraint<T>("T"), T5EncoderOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif