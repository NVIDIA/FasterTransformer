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

#include "src/fastertransformer/models/decoder/Decoder.h"
#include "src/fastertransformer/tf_op/BaseOp.h"

namespace ft = fastertransformer;
namespace tf = tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Decoder")
    .Input("from_tensor: T")                 // # 0
    .Input("memory_tensor: T")               // # 1
    .Input("memory_sequence_length: int32")  // # 2
    .Input("old_self_key_cache: T")          // # 3
    .Input("old_self_value_cache: T")        // # 4
    .Input("old_mem_key_cache: T")           // # 5
    .Input("old_mem_value_cache: T")         // # 6
    .Input("pre_beta: N * T")                // # 7
    .Input("pre_gamma: N * T")               // # 8
    .Input("self_qkv_kernel: N * T")         // # 9
    .Input("self_qkv_bias: N * T")           // # 10
    .Input("self_output_kernel: N * T")      // # 11
    .Input("self_output_bias: N * T")        // # 12
    .Input("self_beta: N * T")               // # 13
    .Input("self_gamma: N * T")              // # 14
    .Input("cross_q_kernel: N * T")          // # 15
    .Input("cross_q_bias: N * T")            // # 16
    .Input("cross_k_kernel: N * T")          // # 17
    .Input("cross_k_bias: N * T")            // # 18
    .Input("cross_v_kernel: N * T")          // # 19
    .Input("cross_v_bias: N * T")            // # 20
    .Input("cross_output_kernel: N * T")     // # 21
    .Input("cross_output_bias: N * T")       // # 22
    .Input("cross_beta: N * T")              // # 23
    .Input("cross_gamma: N * T")             // # 24
    .Input("ffn_kernel1: N * T")             // # 25
    .Input("ffn_bias1: N * T")               // # 26
    .Input("ffn_kernel2: N * T")             // # 27
    .Input("ffn_bias2: N * T")               // # 28
    .Input("step: int32")                    // # 29
    .Input("sequence_length: int32")         // # 30
    .Input("psuedo_input: T")                // # 31
    .Output("decoder_output: T")
    .Output("new_self_key_cache: T")
    .Output("new_self_value_cache: T")
    .Output("new_mem_key_cache: T")
    .Output("new_mem_value_cache: T")
    .Attr("N: int")
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("inter_size: int >= 1")
    .Attr("num_layer: int >= 1")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(3));
        c->set_output(2, c->input(4));
        c->set_output(3, c->input(5));
        c->set_output(4, c->input(6));
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
class DecoderOp: public BaseOp<T> {
public:
    explicit DecoderOp(tf::OpKernelConstruction* context): BaseOp<T>(context)
    {
        try {
            OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
            OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
            OP_REQUIRES_OK(context, context->GetAttr("inter_size", &inter_size_));
            OP_REQUIRES_OK(context, context->GetAttr("num_layer", &num_layer_));
            cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in");
        }
        catch (std::runtime_error& error) {
            OP_REQUIRES(context, false, tf::errors::Internal(error.what()));
        }
    }

    ~DecoderOp()
    {
        delete cublas_algo_map_;
    }

    void Compute(tf::OpKernelContext* context) override
    {

        OP_REQUIRES(context,
                    context->num_inputs() == (num_layer_ * 22) + 10,
                    tf::errors::InvalidArgument("[ERROR] More or Less input arguments"));

        const size_t batch_size = (size_t)(context->input(0).dim_size(0));

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

        std::vector<ft::DecoderLayerWeight<DataType>> decoder_layer_weights(num_layer_);

        for (int i = 0; i < num_layer_; i++) {
            this->get_tensor(context, 7 + i, &decoder_layer_weights[i].pre_layernorm_weights.beta);
            this->get_tensor(context, 7 + num_layer_ * 1 + i, &decoder_layer_weights[i].pre_layernorm_weights.gamma);

            this->get_tensor(
                context, 7 + num_layer_ * 2 + i, &decoder_layer_weights[i].self_attention_weights.query_weight.kernel);
            this->get_tensor(
                context, 7 + num_layer_ * 3 + i, &decoder_layer_weights[i].self_attention_weights.query_weight.bias);
            this->get_tensor(context,
                             7 + num_layer_ * 4 + i,
                             &decoder_layer_weights[i].self_attention_weights.attention_output_weight.kernel);
            this->get_tensor(context,
                             7 + num_layer_ * 5 + i,
                             &decoder_layer_weights[i].self_attention_weights.attention_output_weight.bias);
            this->get_tensor(
                context, 7 + num_layer_ * 6 + i, &decoder_layer_weights[i].self_attn_layernorm_weights.beta);
            this->get_tensor(
                context, 7 + num_layer_ * 7 + i, &decoder_layer_weights[i].self_attn_layernorm_weights.gamma);

            this->get_tensor(
                context, 7 + num_layer_ * 8 + i, &decoder_layer_weights[i].cross_attention_weights.query_weight.kernel);
            this->get_tensor(
                context, 7 + num_layer_ * 9 + i, &decoder_layer_weights[i].cross_attention_weights.query_weight.bias);
            this->get_tensor(
                context, 7 + num_layer_ * 10 + i, &decoder_layer_weights[i].cross_attention_weights.key_weight.kernel);
            this->get_tensor(
                context, 7 + num_layer_ * 11 + i, &decoder_layer_weights[i].cross_attention_weights.key_weight.bias);
            this->get_tensor(context,
                             7 + num_layer_ * 12 + i,
                             &decoder_layer_weights[i].cross_attention_weights.value_weight.kernel);
            this->get_tensor(
                context, 7 + num_layer_ * 13 + i, &decoder_layer_weights[i].cross_attention_weights.value_weight.bias);
            this->get_tensor(context,
                             7 + num_layer_ * 14 + i,
                             &decoder_layer_weights[i].cross_attention_weights.attention_output_weight.kernel);
            this->get_tensor(context,
                             7 + num_layer_ * 15 + i,
                             &decoder_layer_weights[i].cross_attention_weights.attention_output_weight.bias);
            this->get_tensor(
                context, 7 + num_layer_ * 16 + i, &decoder_layer_weights[i].cross_attn_layernorm_weights.beta);
            this->get_tensor(
                context, 7 + num_layer_ * 17 + i, &decoder_layer_weights[i].cross_attn_layernorm_weights.gamma);

            this->get_tensor(
                context, 7 + num_layer_ * 18 + i, &decoder_layer_weights[i].ffn_weights.intermediate_weight.kernel);
            this->get_tensor(
                context, 7 + num_layer_ * 19 + i, &decoder_layer_weights[i].ffn_weights.intermediate_weight.bias);
            this->get_tensor(
                context, 7 + num_layer_ * 20 + i, &decoder_layer_weights[i].ffn_weights.output_weight.kernel);
            this->get_tensor(
                context, 7 + num_layer_ * 21 + i, &decoder_layer_weights[i].ffn_weights.output_weight.bias);
        }

        tf::Tensor self_cache_keys_tensor = context->input(3);
        tf::Tensor self_cache_values_tensor = context->input(4);
        tf::Tensor memory_cache_keys_tensor = context->input(5);
        tf::Tensor memory_cache_values_tensor = context->input(6);
        tf::Tensor* output = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, context->input(0).shape(), &output));
        DataType* out_tensor = (DataType*)(output->flat<T>().data());
        context->set_output(1, self_cache_keys_tensor);
        context->set_output(2, self_cache_values_tensor);
        context->set_output(3, memory_cache_keys_tensor);
        context->set_output(4, memory_cache_values_tensor);

        const int* d_step = reinterpret_cast<const int*>(context->input(7 + num_layer_ * 22).flat<int>().data());
        int step;
        cudaMemcpyAsync(&step, d_step, sizeof(int), cudaMemcpyDeviceToHost, stream);
        step += 1;
        tf::Tensor sequence_length_tensor = context->input(8 + num_layer_ * 22);

        ft::Decoder<DataType> decoder = ft::Decoder<DataType>(
            batch_size, head_num_, size_per_head_, inter_size_, num_layer_, stream, &cublas_wrapper, &allocator, true);

        size_t hidden_units = (size_t)(head_num_ * size_per_head_);

        ft::DataType data_type = ft::getTensorType<DataType>();
        std::vector<ft::Tensor> input_tensors = std::vector<ft::Tensor>{
            this->convert_tensor(context->input(0)),
            this->convert_tensor(context->input(1)),
            this->convert_int_tensor(context->input(2)),
            ft::Tensor{ft::MEMORY_GPU, ft::TYPE_BOOL, {batch_size}, nullptr},
            ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, {1}, &step},
            this->convert_int_tensor(sequence_length_tensor),
            ft::Tensor{ft::MEMORY_GPU,
                       ft::TYPE_INT32,
                       {batch_size, 1, step},
                       nullptr}};  // Since we do gather in the Framework, we don't need id of indirection buffer

        std::vector<ft::Tensor> output_tensors =
            std::vector<ft::Tensor>{ft::Tensor{ft::MEMORY_GPU, data_type, {batch_size, hidden_units}, out_tensor},
                                    this->convert_tensor(self_cache_keys_tensor),
                                    this->convert_tensor(self_cache_values_tensor),
                                    this->convert_tensor(memory_cache_keys_tensor),
                                    this->convert_tensor(memory_cache_values_tensor)};

        try {
            decoder.forward(&output_tensors, &input_tensors, &decoder_layer_weights);
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
    ft::cublasAlgoMap* cublas_algo_map_;
    typedef TFTraits<T> traits_;
    typedef typename traits_::DataType DataType;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                                                                \
    REGISTER_KERNEL_BUILDER(Name("Decoder").Device(tf::DEVICE_GPU).TypeConstraint<T>("T"), DecoderOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
