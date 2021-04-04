/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

#include "fastertransformer/open_decoder.h"
#include "fastertransformer/tf_op/common_op.h"

namespace tensorflow
{
namespace
{
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Decoder")
    .Input("from_tensor: T")                // # 0
    .Input("memory_tensor: T")              // # 1
    .Input("memory_sequence_length: int32") // # 2
    .Input("self_beta: T")                  // # 3
    .Input("self_gamma: T")                 // # 4
    .Input("self_q_kernel: T")              // # 5
    .Input("self_q_bias: T")                // # 6
    .Input("self_k_kernel: T")              // # 7
    .Input("self_k_bias: T")                // # 8
    .Input("self_v_kernel: T")              // # 9
    .Input("self_v_bias: T")                // # 10
    .Input("self_output_kernel: T")         // # 11
    .Input("self_output_bias: T")           // # 12
    .Input("cross_beta: T")                 // # 13
    .Input("cross_gamma: T")                // # 14
    .Input("cross_q_kernel: T")             // # 15
    .Input("cross_q_bias: T")               // # 16
    .Input("cross_k_kernel: T")             // # 17
    .Input("cross_k_bias: T")               // # 18
    .Input("cross_v_kernel: T")             // # 19
    .Input("cross_v_bias: T")               // # 20
    .Input("cross_output_kernel: T")        // # 21
    .Input("cross_output_bias: T")          // # 22
    .Input("ffn_beta: T")                   // # 23
    .Input("ffn_gamma: T")                  // # 24
    .Input("ffn_kernel1: T")                // # 25
    .Input("ffn_bias1: T")                  // # 26
    .Input("ffn_kernel2: T")                // # 27
    .Input("ffn_bias2: T")                  // # 28
    .Input("old_self_cache: ListT")         // # 29, 30
    .Input("old_mem_cache: T")              // # 31
    .Input("pseudo_input: T")               // # 32, pseudo input, used to prevent the parallel execution for OP and TF
    .Input("step: int32")                   // # 33
    .Output("decoder_output: T")
    .Output("new_self_cache: ListT")
    .Output("new_mem_cache: T")
    .Attr("ListT: list({float, half})")
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("memory_hidden_dim: int >= 1")
    .Attr("is_fuse_qkv: bool")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(29));
      c->set_output(2, c->input(30));
      c->set_output(3, c->input(31));
      return Status::OK();
    });
template <typename Device, typename T>
class DecoderOp : public CommonOp<T>
{
public:
  explicit DecoderOp(OpKernelConstruction *context) : CommonOp<T>(context)
  {
    OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
    OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
    OP_REQUIRES_OK(context, context->GetAttr("memory_hidden_dim", &memory_hidden_dim_));
    OP_REQUIRES_OK(context, context->GetAttr("is_fuse_qkv", &is_fuse_qkv_));
    
    try
    {
      decoder_ = new OpenDecoder<DecoderTraits_::OpType>(head_num_, size_per_head_, memory_hidden_dim_, is_fuse_qkv_);
    }
    catch (std::runtime_error &error)
    {
      OP_REQUIRES(context, false, errors::Internal(error.what()));
    }
  }

  ~DecoderOp()
  {
    delete decoder_;
  }

  void Compute(OpKernelContext *context) override
  {
    // input(1): memory_tensor: [batch_size, memory_max_seq_len, memory_hidden_dim]
    assert((int)(context->input(1).dims()) == 3);
    const int batch_size_ = (int)context->input(1).dim_size(0);
    const int max_mem_seq_len_ = (int)context->input(1).dim_size(1);
    OP_REQUIRES(context, memory_hidden_dim_ == (int)context->input(1).dim_size(2),
      errors::InvalidArgument("[ERROR] memory hidden dimension does not equal to the second dimension of memory tensor"));

    // Detect we use batch major
    bool use_batch_major = (context->input(29).dims() == 5)? true : false;
    // we use decoder_max_seq_len == -1 to tell the decoder we use seq major cache format
    int decoder_max_seq_len = (use_batch_major)? (int)context->input(30).dim_size(2) : -1;

    typedef DecoderTransformerTraits<traits_::OpType> DecoderTraits_;
    const cudaStream_t &stream = context->eigen_device<Device>().stream();
    fastertransformer::Allocator<AllocatorType::TF> allocator_(context, stream);
    OP_REQUIRES(context, context->num_inputs() == 34, errors::InvalidArgument("[ERROR] More or Less input arguments"));

    Tensor *decoder_output_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, {batch_size_, 1, head_num_ * size_per_head_}, &decoder_output_tensor));
    DataType_ *decoder_output = reinterpret_cast<DataType_ *>(decoder_output_tensor->flat<T>().data());

    Tensor self_cache_keys_tensor = context->input(29);
    Tensor self_cache_values_tensor = context->input(30);
    context->set_output(1, self_cache_keys_tensor);
    context->set_output(2, self_cache_values_tensor);

    Tensor memory_cache_tensor = context->input(31);
    context->set_output(3, memory_cache_tensor);
    DataType_ *memory_cache = reinterpret_cast<DataType_ *>(memory_cache_tensor.flat<T>().data());

    const DataType_ *from_tensor = reinterpret_cast<const DataType_ *>(context->input(0).flat<T>().data());
    const DataType_ *memory_tensor = reinterpret_cast<const DataType_ *>(context->input(1).flat<T>().data());
    const int *memory_sequence_length = reinterpret_cast<const int *>(context->input(2).flat<int>().data());

    OP_REQUIRES(context, from_tensor != nullptr, errors::InvalidArgument("from_tensor"));
    OP_REQUIRES(context, memory_tensor != nullptr, errors::InvalidArgument("memory_tensor"));
    OP_REQUIRES(context, memory_sequence_length != nullptr, errors::InvalidArgument("memory_sequence_length"));

    DecoderInitParam<DataType_> params;
    params.cublas_handle = this->get_cublas_handler();
    params.cublaslt_handle = this->get_cublaslt_handler();
    params.stream = stream;
    params.request_max_mem_seq_len = max_mem_seq_len_;
    params.request_batch_size = batch_size_;
    check_cuda_error(cublasSetStream(params.cublas_handle, params.stream));

    const int hidden_units = head_num_ * size_per_head_;
    this->get_tensor(context, 3, &params.self_layernorm.beta);
    this->get_tensor(context, 4, &params.self_layernorm.gamma);

    this->get_tensor(context, 5, &params.self_attention.query_weight.kernel);
    this->get_tensor(context, 6, &params.self_attention.query_weight.bias);
    this->get_tensor(context, 7, &params.self_attention.key_weight.kernel);
    this->get_tensor(context, 8, &params.self_attention.key_weight.bias);
    this->get_tensor(context, 9, &params.self_attention.value_weight.kernel);
    this->get_tensor(context, 10, &params.self_attention.value_weight.bias);
    this->get_tensor(context, 11, &params.self_attention.attention_output_weight.kernel);
    this->get_tensor(context, 12, &params.self_attention.attention_output_weight.bias);

    this->get_tensor(context, 13, &params.cross_layernorm.beta);
    this->get_tensor(context, 14, &params.cross_layernorm.gamma);
    this->get_tensor(context, 15, &params.cross_attention.query_weight.kernel);
    this->get_tensor(context, 16, &params.cross_attention.query_weight.bias);
    this->get_tensor(context, 17, &params.cross_attention.key_weight.kernel);
    this->get_tensor(context, 18, &params.cross_attention.key_weight.bias);
    this->get_tensor(context, 19, &params.cross_attention.value_weight.kernel);
    this->get_tensor(context, 20, &params.cross_attention.value_weight.bias);
    this->get_tensor(context, 21, &params.cross_attention.attention_output_weight.kernel);
    this->get_tensor(context, 22, &params.cross_attention.attention_output_weight.bias);

    this->get_tensor(context, 23, &params.ffn_layernorm.beta);
    this->get_tensor(context, 24, &params.ffn_layernorm.gamma);
    this->get_tensor(context, 25, &params.ffn.intermediate_weight.kernel);
    this->get_tensor(context, 26, &params.ffn.intermediate_weight.bias);
    this->get_tensor(context, 27, &params.ffn.output_weight.kernel);
    this->get_tensor(context, 28, &params.ffn.output_weight.bias);

    const int step = *reinterpret_cast<const int*>(context->input(33).flat<int>().data()) + 1;
    //const int step = (int)context->input(29).dim_size(1);

    DataType_ *K_cache = reinterpret_cast<DataType_ *>(self_cache_keys_tensor.flat<T>().data());
    DataType_ *V_cache = reinterpret_cast<DataType_ *>(self_cache_values_tensor.flat<T>().data());

    DataType_ *K_mem_cache = memory_cache;
    DataType_ *V_mem_cache = memory_cache + batch_size_ * max_mem_seq_len_ * hidden_units;
    decoder_->set_max_batch_size(batch_size_);
    const int decoder_buffer_size = decoder_->getWorkspaceSize() * sizeof(DataType_);
    void *buf = allocator_.malloc(((sizeof(DataType_) == sizeof(half)) ? CUBLAS_WORKSPACE_SIZE : 0) + decoder_buffer_size);
    void *cublas_workspace = nullptr;
    DataType_ *decoder_buffer = (DataType_ *)buf;
    if (sizeof(DataType_) == sizeof(half))
    {
      cublas_workspace = buf;
      decoder_buffer = (DataType_ *)((char*)cublas_workspace + CUBLAS_WORKSPACE_SIZE);
    }

    try
    {
      decoder_->initialize(params, decoder_buffer, cublas_workspace);
      decoder_->forward(from_tensor, memory_tensor,
                        K_cache, V_cache,
                        K_mem_cache, V_mem_cache,
                        memory_sequence_length, decoder_output, step, decoder_max_seq_len,
                        true);
    }
    catch (std::runtime_error &error)
    {
      std::cout << errors::Internal(error.what());
      exit(-1);
    }
    catch (...)
    {
      std::cout << errors::Internal("Runtime error");
      exit(-1);
    }
    allocator_.free(buf);
  }

private:
  int memory_hidden_dim_, max_batch_size_;
  int head_num_, size_per_head_;
  bool is_fuse_qkv_;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef DecoderTransformerTraits<traits_::OpType> DecoderTraits_;
  OpenDecoder<DecoderTraits_::OpType> *decoder_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Decoder").Device(DEVICE_GPU).TypeConstraint<T>("T")  \
      .HostMemory("step"),                                       \
      DecoderOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
} //namespace
} //namespace tensorflow
