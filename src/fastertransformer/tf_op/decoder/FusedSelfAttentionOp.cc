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

#define EIGEN_USE_GPU

#include "src/fastertransformer/layers/attention_layers/DecoderSelfAttentionLayer.h"
#include "src/fastertransformer/tf_op/BaseOp.h"

namespace ft = fastertransformer;
namespace tf = tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

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

REGISTER_OP("FusedQkvMultiHeadAttention")
    .Input("qkv_tensor: T")
    .Input("qkv_bias: T")
    .Input("k_cache: T")
    .Input("v_cache: T")
    .Output("output: T")
    .Output("new_k_cache: T")
    .Output("new_v_cache: T")
    .Attr("T: {float, half}")
    .Attr("batch_size: int >= 1")
    .Attr("seq_len: int >= 1")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        assert(c->Rank(c->input(0)) == 4);
        assert(c->Rank(c->input(1)) == 3);
        assert(c->Rank(c->input(2)) == 5 || c->Rank(c->input(2)) == 4);
        assert(c->Rank(c->input(3)) == 4);

        int batch_size, head_num, size_per_head;
        c->GetAttr("batch_size", &batch_size);
        c->GetAttr("head_num", &head_num);
        c->GetAttr("size_per_head", &size_per_head);
        c->set_output(0, c->MakeShape({batch_size, 1, head_num, size_per_head}));
        c->set_output(1, c->input(2));
        c->set_output(2, c->input(3));
        return tf::Status::OK();
    });
template<typename Device, typename T>
class FusedQkvMultiHeadAttentionOp: public BaseOp<T> {
public:
    explicit FusedQkvMultiHeadAttentionOp(tf::OpKernelConstruction* context): BaseOp<T>(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batch_size_));
        OP_REQUIRES_OK(context, context->GetAttr("seq_len", &seq_len_));
        OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
        OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
    }

    void Compute(tf::OpKernelContext* context) override
    {
        OP_REQUIRES(context,
                    context->input(0).dims() == 4,
                    tf::errors::InvalidArgument("Invalid rank. The rank of from tensor should be 4 \
                                        ([batch_size, 3, head_num, size_per_head])"));
        OP_REQUIRES(context,
                    context->input(1).dims() == 3,
                    tf::errors::InvalidArgument("Invalid rank. The rank of from tensor should be 4 \
                                        ([3, head_num, size_per_head])"));
        // Detect we use batch major
        bool use_batch_major = (context->input(2).dims() == 5) ? true : false;
        // we use decoder_max_seq_len == -1 to tell the decoder we use seq major cache format
        int decoder_max_seq_len = (use_batch_major) ? (int)context->input(2).dim_size(3) : -1;
        if (use_batch_major) {
            printf("[INFO] use_batch_major \n");
            OP_REQUIRES(context,
                        context->input(2).dims() == 5,
                        tf::errors::InvalidArgument("Invalid rank. The rank of sequence length should be 5 "
                                                    "([batch_size, head_num, size_per_head/x, max_seq_len, x])"));
            OP_REQUIRES(context,
                        context->input(3).dims() == 4,
                        tf::errors::InvalidArgument("Invalid rank. The rank of sequence length should be 4 "
                                                    "([[batch_size, head_num, max_seq_len, size_per_head])"));
        }
        else {
            OP_REQUIRES(context,
                        context->input(2).dims() == 4,
                        tf::errors::InvalidArgument("Invalid rank. The rank of sequence length should be 4 "
                                                    "([seq_len, batch_size, head_num, size_per_head])"));
            OP_REQUIRES(context,
                        context->input(3).dims() == 4,
                        tf::errors::InvalidArgument("Invalid rank. The rank of sequence length should be 4 "
                                                    "([seq_len, batch_size, head_num, size_per_head])"));
        }

        const cudaStream_t& stream = context->eigen_device<Device>().stream();
        const DataType_* qkv_input = reinterpret_cast<const DataType_*>(context->input(0).flat<T>().data());
        OP_REQUIRES(context, qkv_input != nullptr, tf::errors::InvalidArgument("qkv_input is null"));

        const DataType_* qkv_bias = reinterpret_cast<const DataType_*>(context->input(1).flat<T>().data());
        OP_REQUIRES(context, qkv_bias != nullptr, tf::errors::InvalidArgument("qkv_bias is null"));

        tf::Tensor k_cache_tensor = context->input(2);
        context->set_output(1, k_cache_tensor);
        DataType_* k_cache = reinterpret_cast<DataType_*>(k_cache_tensor.flat<T>().data());

        tf::Tensor v_cache_tensor = context->input(3);
        context->set_output(2, v_cache_tensor);
        DataType_* v_cache = reinterpret_cast<DataType_*>(v_cache_tensor.flat<T>().data());

        tf::Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, {batch_size_, 1, head_num_, size_per_head_}, &output));
        DataType_* output_ptr = reinterpret_cast<DataType_*>(output->flat<T>().data());

        try {
            fastertransformer::fusedQKV_masked_attention_dispatch(qkv_input,
                                                                  qkv_bias,
                                                                  k_cache,
                                                                  v_cache,
                                                                  output_ptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  batch_size_,
                                                                  batch_size_,
                                                                  head_num_,
                                                                  size_per_head_,
                                                                  decoder_max_seq_len,
                                                                  0,
                                                                  nullptr,
                                                                  seq_len_,
                                                                  stream);
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
    int batch_size_, head_num_, size_per_head_, seq_len_;
    typedef TFTraits<T> traits_;
    typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                                                                \
    REGISTER_KERNEL_BUILDER(Name("FusedQkvMultiHeadAttention").Device(tf::DEVICE_GPU).TypeConstraint<T>("T"),          \
                            FusedQkvMultiHeadAttentionOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif