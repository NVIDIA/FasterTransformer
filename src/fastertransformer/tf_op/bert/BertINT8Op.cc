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

#include "src/fastertransformer/models/bert_int8/BertINT8.h"
#include "src/fastertransformer/tf_op/BaseOp.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

namespace ft = fastertransformer;
namespace tf = tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("BertINT8")
    .Input("from_tensor: T")
    .Input("to_tensor: T")
    .Input("sequence_length: int32")
    .Input("attr_q_kernel: N * T")
    .Input("attr_q_bias: N * T")
    .Input("attr_k_kernel: N * T")
    .Input("attr_k_bias: N * T")
    .Input("attr_v_kernel: N * T")
    .Input("attr_v_bias: N * T")
    .Input("attr_output_kernel: N * T")
    .Input("attr_output_bias: N * T")
    .Input("attr_output_layernorm_beta: N * T")
    .Input("attr_output_layernorm_gamma: N * T")
    .Input("inter_kernel: N * T")
    .Input("inter_bias: N * T")
    .Input("output_kernel: N * T")
    .Input("output_bias: N * T")
    .Input("output_layernorm_beta: N * T")
    .Input("output_layernorm_gamma: N * T")
    .Input("d_scale_list: N * float")
    .Output("output: T")
    .Attr("N: int")
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("inter_size: int >= 1")
    .Attr("num_layer: int >= 1")
    .Attr("int8_mode: int >= 1")
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
class BertINT8Op: public BaseOp<T> {
public:
    explicit BertINT8Op(tf::OpKernelConstruction* context): BaseOp<T>(context)
    {
        try {
            OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
            OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
            OP_REQUIRES_OK(context, context->GetAttr("inter_size", &inter_size_));
            OP_REQUIRES_OK(context, context->GetAttr("num_layer", &num_layer_));
            OP_REQUIRES_OK(context, context->GetAttr("int8_mode", &int8_mode_));
            OP_REQUIRES_OK(context, context->GetAttr("remove_padding", &remove_padding_));
            OP_REQUIRES_OK(context, context->GetAttr("q_scaling", &q_scaling_));
            sm_ = ft::getSMVersion();
            cublas_algo_map_ = new ft::cublasAlgoMap("igemm_config.in");
            set_weight_ = false;
            use_ORDER_COL32_2R_4R4_ = false;
#if (CUDART_VERSION >= 11000)
            if (sm_ >= 80) {
                use_ORDER_COL32_2R_4R4_ = true;
            }
#endif
        }
        catch (std::runtime_error& error) {
            OP_REQUIRES(context, false, tf::errors::Internal(error.what()));
        }
    }

    ~BertINT8Op()
    {
        delete cublas_algo_map_;
        if (set_weight_ && h_scale_list_) {
            free(h_scale_list_);
            h_scale_list_ = nullptr;
            set_weight_ = false;
        }
    }

    void Compute(tf::OpKernelContext* context) override
    {
        OP_REQUIRES(context,
                    context->num_inputs() == (num_layer_ * 17) + 3,
                    tf::errors::InvalidArgument("[ERROR] More or Less input arguments"));

        const size_t batch_size_ = (size_t)context->input(0).dim_size(0);
        const size_t from_seq_len_ = (size_t)context->input(0).dim_size(1);

        OP_REQUIRES(context,
                    batch_size_ == (size_t)context->input(2).dim_size(0),
                    tf::errors::InvalidArgument("[ERROR] invalid shape"));

        const cudaStream_t& stream = context->eigen_device<Device>().stream();
        cublasLtHandle_t cublaslt_handle = this->get_cublaslt_handler();

        ft::cublasINT8MMWrapper cublas_wrapper = ft::cublasINT8MMWrapper(
            cublaslt_handle, stream, cublas_algo_map_, this->get_cublas_wrapper_mutex(), use_ORDER_COL32_2R_4R4_);

        ft::Allocator<ft::AllocatorType::TF> allocator(context, stream);

        if (!set_weight_) {
            const int scale_list_size = ACTIVATION_AMAX_NUM + 9 * head_num_ * size_per_head_ + INT8O_GEMM_NUM
                                        + TRT_AMAX_NUM + SCALE_RESERVE_NUM;
            h_scale_list_ = (float*)malloc(num_layer_ * scale_list_size * sizeof(float));

            bert_layer_weights_.resize(num_layer_);
            for (int i = 0; i < num_layer_; i++) {
                this->get_tensor(context, 3 + i, &bert_layer_weights_[i].attention_weights.query_weight.kernel);
                this->get_tensor(
                    context, 3 + num_layer_ + i, &bert_layer_weights_[i].attention_weights.query_weight.bias);
                this->get_tensor(
                    context, 3 + num_layer_ * 2 + i, &bert_layer_weights_[i].attention_weights.key_weight.kernel);
                this->get_tensor(
                    context, 3 + num_layer_ * 3 + i, &bert_layer_weights_[i].attention_weights.key_weight.bias);
                this->get_tensor(
                    context, 3 + num_layer_ * 4 + i, &bert_layer_weights_[i].attention_weights.value_weight.kernel);
                this->get_tensor(
                    context, 3 + num_layer_ * 5 + i, &bert_layer_weights_[i].attention_weights.value_weight.bias);
                this->get_tensor(context,
                                 3 + num_layer_ * 6 + i,
                                 &bert_layer_weights_[i].attention_weights.attention_output_weight.kernel);
                this->get_tensor(context,
                                 3 + num_layer_ * 7 + i,
                                 &bert_layer_weights_[i].attention_weights.attention_output_weight.bias);

                this->get_tensor(context, 3 + num_layer_ * 8 + i, &bert_layer_weights_[i].attn_layernorm_weights.beta);
                this->get_tensor(context, 3 + num_layer_ * 9 + i, &bert_layer_weights_[i].attn_layernorm_weights.gamma);
                this->get_tensor(
                    context, 3 + num_layer_ * 10 + i, &bert_layer_weights_[i].ffn_weights.intermediate_weight.kernel);
                this->get_tensor(
                    context, 3 + num_layer_ * 11 + i, &bert_layer_weights_[i].ffn_weights.intermediate_weight.bias);
                this->get_tensor(
                    context, 3 + num_layer_ * 12 + i, &bert_layer_weights_[i].ffn_weights.output_weight.kernel);
                this->get_tensor(
                    context, 3 + num_layer_ * 13 + i, &bert_layer_weights_[i].ffn_weights.output_weight.bias);
                this->get_tensor(context, 3 + num_layer_ * 14 + i, &bert_layer_weights_[i].ffn_layernorm_weights.beta);
                this->get_tensor(context, 3 + num_layer_ * 15 + i, &bert_layer_weights_[i].ffn_layernorm_weights.gamma);
                // deal with scale list
                bert_layer_weights_[i].scale_list_.d_scale_list_ =
                    reinterpret_cast<const float*>(context->input(3 + num_layer_ * 16 + i).flat<float>().data());
                bert_layer_weights_[i].scale_list_.size_ = scale_list_size;
                bert_layer_weights_[i].scale_list_.p3_offset_ = ACTIVATION_AMAX_NUM + 9 * head_num_ * size_per_head_;
                bert_layer_weights_[i].scale_list_.p4_offset_ =
                    ACTIVATION_AMAX_NUM + 9 * head_num_ * size_per_head_ + INT8O_GEMM_NUM;
                bert_layer_weights_[i].attention_weights.scale_list_ptr = &(bert_layer_weights_[i].scale_list_);
                bert_layer_weights_[i].ffn_weights.scale_list_ptr = &(bert_layer_weights_[i].scale_list_);
                // copy h_scale_list
                cudaMemcpy(h_scale_list_ + i * scale_list_size,
                           bert_layer_weights_[i].scale_list_.d_scale_list_,
                           sizeof(float) * scale_list_size,
                           cudaMemcpyDeviceToHost);
                bert_layer_weights_[i].scale_list_.h_scale_list_ = h_scale_list_ + i * scale_list_size;
            }
            set_weight_ = true;
        }

        ft::AttentionType attention_type =
            ft::getAttentionTypeINT8<DataType>(size_per_head_, sm_, remove_padding_, from_seq_len_, int8_mode_);
        ft::BertINT8<DataType> bert = ft::BertINT8<DataType>(batch_size_,
                                                             from_seq_len_,
                                                             head_num_,
                                                             size_per_head_,
                                                             inter_size_,
                                                             num_layer_,
                                                             sm_,
                                                             q_scaling_,
                                                             int8_mode_,
                                                             stream,
                                                             &cublas_wrapper,
                                                             &allocator,
                                                             true,
                                                             attention_type);

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
            bert.forward(&output_tensors, &input_tensors, &bert_layer_weights_);
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
    int head_num_ = 0, size_per_head_ = 0, num_layer_ = 0, inter_size_ = 0, int8_mode_ = 1;
    bool remove_padding_;
    bool use_ORDER_COL32_2R_4R4_;
    int sm_;
    float q_scaling_ = 1.0f;
    ft::cublasAlgoMap* cublas_algo_map_;
    float* h_scale_list_ = nullptr;
    bool set_weight_ = false;
    typedef TFTraits<T> traits_;
    typedef typename traits_::DataType DataType;
    std::vector<ft::BertLayerINT8Weight<DataType>> bert_layer_weights_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                                                                \
    REGISTER_KERNEL_BUILDER(Name("BertINT8").Device(tf::DEVICE_GPU).TypeConstraint<T>("T"), BertINT8Op<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
