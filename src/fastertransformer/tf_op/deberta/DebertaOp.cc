/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/deberta/Deberta.h"
#include "src/fastertransformer/tf_op/BaseOp.h"

namespace ft = fastertransformer;
namespace tf = tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Deberta")
    .Input("from_tensor: int32")
    .Input("sequence_length: int32")
    .Input("word_embedding_table: T")
    .Input("word_embedding_layernorm_gamma: T")
    .Input("word_embedding_layernorm_beta: T")
    .Input("relative_embedding_table: T")
    .Input("relative_embedding_layernorm_gamma: T")
    .Input("relative_embedding_layernorm_beta: T")
    .Input("attn_q_kernel: N * T")
    .Input("attn_q_bias: N * T")
    .Input("attn_k_kernel: N * T")
    .Input("attn_k_bias: N * T")
    .Input("attn_v_kernel: N * T")
    .Input("attn_v_bias: N * T")
    .Input("attn_output_kernel: N * T")
    .Input("attn_output_bias: N * T")
    .Input("attn_output_layernorm_gamma: N * T")
    .Input("attn_output_layernorm_beta: N * T")
    .Input("inter_kernel: N * T")
    .Input("inter_bias: N * T")
    .Input("output_kernel: N * T")
    .Input("output_bias: N * T")
    .Input("output_layernorm_gamma: N * T")
    .Input("output_layernorm_beta: N * T")
    .Output("output: T")
    .Attr("N: int")
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("max_relative_positions: int >= 1")
    .Attr("relative_position_buckets: int >= 1")
    .Attr("inter_size: int >= 1")
    .Attr("num_layer: int >= 1")
    .Attr("remove_padding: bool")
    .Attr("q_scaling: float")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        int head_num, size_per_head;
        c->GetAttr("head_num", &head_num);
        c->GetAttr("size_per_head", &size_per_head);

        int rank = c->Rank(c->input(0));
        assert(rank == 2);

        // calculate batch size and sequence length from input
        tf::shape_inference::DimensionHandle batch_dim    = c->Dim(c->input(0), 0);
        tf::shape_inference::DimensionHandle from_seq_len = c->Dim(c->input(0), 1);

        c->set_output(0, c->MakeShape({batch_dim, from_seq_len, head_num * size_per_head}));

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
class DebertaOp: public BaseOp<T> {
public:
    explicit DebertaOp(tf::OpKernelConstruction* context): BaseOp<T>(context)
    {
        try {
            OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
            OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
            OP_REQUIRES_OK(context, context->GetAttr("max_relative_positions", &max_relative_positions_));
            OP_REQUIRES_OK(context, context->GetAttr("relative_position_buckets", &relative_position_buckets_));
            OP_REQUIRES_OK(context, context->GetAttr("inter_size", &inter_size_));
            OP_REQUIRES_OK(context, context->GetAttr("num_layer", &num_layer_));
            OP_REQUIRES_OK(context, context->GetAttr("remove_padding", &remove_padding_));
            OP_REQUIRES_OK(context, context->GetAttr("q_scaling", &q_scaling_));
            cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in");
        }
        catch (std::runtime_error& error) {
            OP_REQUIRES(context, false, tf::errors::Internal(error.what()));
        }
    }

    ~DebertaOp()
    {
        delete cublas_algo_map_;
    }

    void Compute(tf::OpKernelContext* context) override
    {
        // input_tensors:
        //      [0] input_ids [batch, seqlen]
        //      [1] sequence_lengths [batch]
        //      [2] ~ [7] word embedding + layernorm weights & relative embedding + layernorm weights
        //      [...] 16 * L, 16 layer weights per each L encoder layers
        // output tensors:
        //      output_hidden_state [batch, seqlen, hidden]

        OP_REQUIRES(context,
                    context->num_inputs() == 2 + 6 + (num_layer_ * 16),
                    tf::errors::InvalidArgument("[ERROR] More or Less input arguments"));

        const size_t batch_size_   = (size_t)context->input(0).dim_size(0);
        const size_t from_seq_len_ = (size_t)context->input(0).dim_size(1);

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

        ft::DebertaWeight<DataType>
            deberta_weights;  // tensor managed by TF, so only overwrite data pointers of the object. is_maintain_buffer
                              // is by default false, so destructor won't be called later
        deberta_weights.deberta_layer_weights.resize(num_layer_);

        // Model-level weights
        this->get_tensor(context, 2, &deberta_weights.word_embedding_table);
        this->get_tensor(context, 3, &deberta_weights.word_embedding_layernorm_weights.gamma);
        this->get_tensor(context, 4, &deberta_weights.word_embedding_layernorm_weights.beta);
        this->get_tensor(context, 5, &deberta_weights.relative_embedding_table);
        this->get_tensor(context, 6, &deberta_weights.relative_embedding_layernorm_weights.gamma);
        this->get_tensor(context, 7, &deberta_weights.relative_embedding_layernorm_weights.beta);

        // Layer-level weights
        for (int i = 0; i < num_layer_; i++) {
            this->get_tensor(
                context, 8 + i, &deberta_weights.deberta_layer_weights[i].attention_weights.query_weight.kernel);
            this->get_tensor(context,
                             8 + num_layer_ + i,
                             &deberta_weights.deberta_layer_weights[i].attention_weights.query_weight.bias);
            this->get_tensor(context,
                             8 + num_layer_ * 2 + i,
                             &deberta_weights.deberta_layer_weights[i].attention_weights.key_weight.kernel);
            this->get_tensor(context,
                             8 + num_layer_ * 3 + i,
                             &deberta_weights.deberta_layer_weights[i].attention_weights.key_weight.bias);
            this->get_tensor(context,
                             8 + num_layer_ * 4 + i,
                             &deberta_weights.deberta_layer_weights[i].attention_weights.value_weight.kernel);
            this->get_tensor(context,
                             8 + num_layer_ * 5 + i,
                             &deberta_weights.deberta_layer_weights[i].attention_weights.value_weight.bias);
            this->get_tensor(
                context,
                8 + num_layer_ * 6 + i,
                &deberta_weights.deberta_layer_weights[i].attention_weights.attention_output_weight.kernel);
            this->get_tensor(context,
                             8 + num_layer_ * 7 + i,
                             &deberta_weights.deberta_layer_weights[i].attention_weights.attention_output_weight.bias);

            this->get_tensor(context,
                             8 + num_layer_ * 8 + i,
                             &deberta_weights.deberta_layer_weights[i].attn_layernorm_weights.gamma);
            this->get_tensor(
                context, 8 + num_layer_ * 9 + i, &deberta_weights.deberta_layer_weights[i].attn_layernorm_weights.beta);
            this->get_tensor(context,
                             8 + num_layer_ * 10 + i,
                             &deberta_weights.deberta_layer_weights[i].ffn_weights.intermediate_weight.kernel);
            this->get_tensor(context,
                             8 + num_layer_ * 11 + i,
                             &deberta_weights.deberta_layer_weights[i].ffn_weights.intermediate_weight.bias);
            this->get_tensor(context,
                             8 + num_layer_ * 12 + i,
                             &deberta_weights.deberta_layer_weights[i].ffn_weights.output_weight.kernel);
            this->get_tensor(context,
                             8 + num_layer_ * 13 + i,
                             &deberta_weights.deberta_layer_weights[i].ffn_weights.output_weight.bias);
            this->get_tensor(context,
                             8 + num_layer_ * 14 + i,
                             &deberta_weights.deberta_layer_weights[i].ffn_layernorm_weights.gamma);
            this->get_tensor(
                context, 8 + num_layer_ * 15 + i, &deberta_weights.deberta_layer_weights[i].ffn_layernorm_weights.beta);
        }

        ft::Deberta<DataType> deberta = ft::Deberta<DataType>(batch_size_,
                                                              from_seq_len_,
                                                              head_num_,
                                                              size_per_head_,
                                                              max_relative_positions_,
                                                              relative_position_buckets_,
                                                              inter_size_,
                                                              num_layer_,
                                                              q_scaling_,
                                                              stream,
                                                              &cublas_wrapper,
                                                              &allocator,
                                                              true,
                                                              false,
                                                              ft::ActivationType::Gelu,
                                                              ft::LayerNormType::post_layernorm);

        tf::Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(
                           0,
                           {context->input(0).dim_size(0), context->input(0).dim_size(1), head_num_ * size_per_head_},
                           &output));
        DataType* out_tensor = reinterpret_cast<DataType*>(output->flat<T>().data());

        ft::DataType data_type = ft::getTensorType<DataType>();

        ft::TensorMap input_tensors({{"input_ids", this->convert_int_tensor(context->input(0))},
                                     {"sequence_lengths", this->convert_int_tensor(context->input(1))}});

        ft::TensorMap output_tensors(
            {{"output_hidden_state",
              ft::Tensor{ft::MEMORY_GPU,
                         data_type,
                         std::vector<size_t>{batch_size_, from_seq_len_, (size_t)(head_num_ * size_per_head_)},
                         out_tensor}}});

        try {
            deberta.forward(&output_tensors, &input_tensors, &deberta_weights);
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
    int                                head_num_                  = 0;
    int                                size_per_head_             = 0;
    int                                max_relative_positions_    = 0;
    int                                relative_position_buckets_ = 0;
    int                                inter_size_                = 0;
    int                                num_layer_                 = 0;
    float                              q_scaling_                 = 1.0f;
    bool                               remove_padding_;
    ft::cublasAlgoMap*                 cublas_algo_map_;
    typedef TFTraits<T>                traits_;
    typedef typename traits_::DataType DataType;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                                                                \
    REGISTER_KERNEL_BUILDER(Name("Deberta").Device(tf::DEVICE_GPU).TypeConstraint<T>("T"), DebertaOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
