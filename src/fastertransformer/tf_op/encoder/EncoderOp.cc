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

#include "src/fastertransformer/models/bert/Bert.h"
#include "src/fastertransformer/tf_op/BaseOp.h"

namespace ft = fastertransformer;
namespace tf = tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Encoder")
    .Input("from_tensor: T")
    .Input("to_tensor: T")
    .Input("sequence_length: int32")
    .Input("attr_output_layernorm_beta: N * T")
    .Input("attr_output_layernorm_gamma: N * T")
    .Input("attr_q_kernel: N * T")
    .Input("attr_q_bias: N * T")
    .Input("attr_k_kernel: N * T")
    .Input("attr_k_bias: N * T")
    .Input("attr_v_kernel: N * T")
    .Input("attr_v_bias: N * T")
    .Input("attr_output_kernel: N * T")
    .Input("attr_output_bias: N * T")
    .Input("ffn_layernorm_beta: N * T")
    .Input("ffn_layernorm_gamma: N * T")
    .Input("inter_kernel: N * T")
    .Input("inter_bias: N * T")
    .Input("output_kernel: N * T")
    .Input("output_bias: N * T")
    .Input("layernorm_beta: T")
    .Input("layernorm_gamma: T")
    .Output("output: T")
    .Attr("N: int")
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("inter_size: int >= 1")
    .Attr("num_layer: int >= 1")
    .Attr("remove_padding: bool")
    .Attr("q_scaling: float")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
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
class EncoderOp: public BaseOp<T> {
public:
    explicit EncoderOp(tf::OpKernelConstruction* context): BaseOp<T>(context)
    {
        try {
            OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
            OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
            OP_REQUIRES_OK(context, context->GetAttr("inter_size", &inter_size_));
            OP_REQUIRES_OK(context, context->GetAttr("num_layer", &num_layer_));
            OP_REQUIRES_OK(context, context->GetAttr("remove_padding", &remove_padding_));
            OP_REQUIRES_OK(context, context->GetAttr("q_scaling", &q_scaling_));
            sm_ = ft::getSMVersion();
            cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in");
        }
        catch (std::runtime_error& error) {
            OP_REQUIRES(context, false, tf::errors::Internal(error.what()));
        }
    }

    ~EncoderOp()
    {
        delete cublas_algo_map_;
    }

    void Compute(tf::OpKernelContext* context) override
    {
        OP_REQUIRES(context,
                    context->num_inputs() == (num_layer_ * 16) + 5,
                    tf::errors::InvalidArgument("[ERROR] More or Less input arguments"));

        const size_t batch_size_ = (size_t)context->input(0).dim_size(0);
        const size_t from_seq_len_ = (size_t)context->input(0).dim_size(1);

        OP_REQUIRES(context,
                    batch_size_ == (size_t)context->input(2).dim_size(0),
                    tf::errors::InvalidArgument("[ERROR] invalid shape"));

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

        ft::BertWeight<DataType> encoder_weights;
        encoder_weights.bert_layer_weights.resize(num_layer_);

        for (int i = 0; i < num_layer_; i++) {
            this->get_tensor(
                context, 3 + num_layer_ * 0 + i, &encoder_weights.bert_layer_weights[i].attn_layernorm_weights.beta);
            this->get_tensor(
                context, 3 + num_layer_ * 1 + i, &encoder_weights.bert_layer_weights[i].attn_layernorm_weights.gamma);

            this->get_tensor(context,
                             3 + num_layer_ * 2 + i,
                             &encoder_weights.bert_layer_weights[i].attention_weights.query_weight.kernel);
            this->get_tensor(context,
                             3 + num_layer_ * 3 + i,
                             &encoder_weights.bert_layer_weights[i].attention_weights.query_weight.bias);
            this->get_tensor(context,
                             3 + num_layer_ * 4 + i,
                             &encoder_weights.bert_layer_weights[i].attention_weights.key_weight.kernel);
            this->get_tensor(context,
                             3 + num_layer_ * 5 + i,
                             &encoder_weights.bert_layer_weights[i].attention_weights.key_weight.bias);
            this->get_tensor(context,
                             3 + num_layer_ * 6 + i,
                             &encoder_weights.bert_layer_weights[i].attention_weights.value_weight.kernel);
            this->get_tensor(context,
                             3 + num_layer_ * 7 + i,
                             &encoder_weights.bert_layer_weights[i].attention_weights.value_weight.bias);
            this->get_tensor(context,
                             3 + num_layer_ * 8 + i,
                             &encoder_weights.bert_layer_weights[i].attention_weights.attention_output_weight.kernel);
            this->get_tensor(context,
                             3 + num_layer_ * 9 + i,
                             &encoder_weights.bert_layer_weights[i].attention_weights.attention_output_weight.bias);

            this->get_tensor(
                context, 3 + num_layer_ * 10 + i, &encoder_weights.bert_layer_weights[i].ffn_layernorm_weights.beta);
            this->get_tensor(
                context, 3 + num_layer_ * 11 + i, &encoder_weights.bert_layer_weights[i].ffn_layernorm_weights.gamma);
            this->get_tensor(context,
                             3 + num_layer_ * 12 + i,
                             &encoder_weights.bert_layer_weights[i].ffn_weights.intermediate_weight.kernel);
            this->get_tensor(context,
                             3 + num_layer_ * 13 + i,
                             &encoder_weights.bert_layer_weights[i].ffn_weights.intermediate_weight.bias);
            this->get_tensor(context,
                             3 + num_layer_ * 14 + i,
                             &encoder_weights.bert_layer_weights[i].ffn_weights.output_weight.kernel);
            this->get_tensor(context,
                             3 + num_layer_ * 15 + i,
                             &encoder_weights.bert_layer_weights[i].ffn_weights.output_weight.bias);
        }
        this->get_tensor(context, 3 + num_layer_ * 16, &encoder_weights.post_transformer_layernorm_weights.beta);
        this->get_tensor(context, 4 + num_layer_ * 16, &encoder_weights.post_transformer_layernorm_weights.gamma);

        ft::AttentionType attention_type =
            ft::getAttentionType<DataType>(size_per_head_, sm_, remove_padding_, from_seq_len_);

        ft::Bert<DataType> encoder = ft::Bert<DataType>(batch_size_,
                                                        from_seq_len_,
                                                        head_num_,
                                                        size_per_head_,
                                                        inter_size_,
                                                        num_layer_,
                                                        sm_,
                                                        q_scaling_,
                                                        stream,
                                                        &cublas_wrapper,
                                                        &allocator,
                                                        true,
                                                        attention_type,
                                                        false,
                                                        ft::ActivationType::Relu,
                                                        ft::LayerNormType::pre_layernorm);

        tf::Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, context->input(0).shape(), &output));
        DataType* out_tensor = reinterpret_cast<DataType*>(output->flat<T>().data());

        ft::DataType data_type = ft::getTensorType<DataType>();
        std::vector<ft::Tensor> input_tensors = std::vector<ft::Tensor>{this->convert_tensor(context->input(0)),
                                                                        this->convert_int_tensor(context->input(2))};

        std::vector<ft::Tensor> output_tensors = std::vector<ft::Tensor>{
            ft::Tensor{ft::MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{batch_size_, from_seq_len_, (size_t)(head_num_ * size_per_head_)},
                       out_tensor}};

        try {
            encoder.forward(&output_tensors, &input_tensors, &encoder_weights);
        }
        catch (std::runtime_error& error) {
            std::cout << tf::errors::Internal(error.what());
            exit(-1);
        }
        catch (...) {
            std::cout << tf::errors::Internal("Runtime error");
            exit(-1);
        }
    }

private:
    int head_num_ = 0, size_per_head_ = 0, inter_size_ = 0, num_layer_ = 0;
    float q_scaling_ = 1.0f;
    bool remove_padding_;
    int sm_;
    ft::cublasAlgoMap* cublas_algo_map_;
    typedef TFTraits<T> traits_;
    typedef typename traits_::DataType DataType;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                                                                \
    REGISTER_KERNEL_BUILDER(Name("Encoder").Device(tf::DEVICE_GPU).TypeConstraint<T>("T"), EncoderOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
