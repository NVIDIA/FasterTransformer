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

#include "src/fastertransformer/models/t5/T5Decoder.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

class IFT5Decoder {
public:
    virtual ~IFT5Decoder() {}
    virtual void forward(size_t      batch_size,
                         size_t      step,
                         th::Tensor& from_tensor,
                         th::Tensor& memory_tensor,
                         th::Tensor& memory_sequence_length,
                         th::Tensor& sequence_length,
                         th::Tensor& output_tensor,
                         th::Tensor& self_cache_keys_tensor,
                         th::Tensor& self_cache_values_tensor,
                         th::Tensor& memory_cache_keys_tensor,
                         th::Tensor& memory_cache_values_tensor,
                         th::Tensor& relative_attention_bias_tensor) = 0;
};

template<typename T>
class FTT5Decoder: public IFT5Decoder {
public:
    FTT5Decoder(int                            head_num,
                int                            head_size,
                int                            inter_size,
                int                            d_model,
                int                            layer_num,
                int                            mem_d_model,
                int                            tensor_para_size,
                int                            pipeline_para_size,
                bool                           t5_with_bias,
                bool                           use_gated_activation,
                int                            position_embedding_type,
                const std::vector<th::Tensor>& w):
        _head_num(head_num),
        _head_size(head_size),
        _inter_size(inter_size),
        _t5_with_bias(t5_with_bias),
        _use_gated_activation(use_gated_activation),
        _position_embedding_type(position_embedding_type),
        _d_model(d_model),
        _weights(w),
        _layer_num(layer_num),
        _mem_d_model(mem_d_model)
    {
        ft::ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size, pipeline_para_size);

        int hidden_dim = _head_num * _head_size;
        ft::check_cuda_error(cublasLtCreate(&_cublasltHandle));
        cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in");

        cublas_wrapper_mutex_ = new std::mutex();
        decoder_layer_weights.clear();
        decoder_layer_weights.resize(_layer_num);

        for (int i = 0; i < _layer_num; ++i) {
            int local_num_layer = (int)(ceil(_layer_num * 1.0f / pipeline_para_.world_size_));
            if (!(i < _layer_num && (i >= local_num_layer * pipeline_para_.rank_)
                  && (i < local_num_layer * (pipeline_para_.rank_ + 1)))) {
                continue;
            }
            const int first_layer_index = local_num_layer * pipeline_para_.rank_;

            decoder_layer_weights[i]->pre_layernorm_weights.gamma =
                get_ptr<T>(_weights[0]) + (i - first_layer_index) * _d_model;
            decoder_layer_weights[i]->self_attention_weights.query_weight.kernel =
                get_ptr<T>(_weights[1])
                + (i - first_layer_index) * _d_model * 3 * hidden_dim / tensor_para_.world_size_;
            decoder_layer_weights[i]->self_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(_weights[2]) + (i - first_layer_index) * hidden_dim / tensor_para_.world_size_ * _d_model;
            decoder_layer_weights[i]->self_attn_layernorm_weights.gamma =
                get_ptr<T>(_weights[3]) + (i - first_layer_index) * _d_model;
            decoder_layer_weights[i]->cross_attention_weights.query_weight.kernel =
                get_ptr<T>(_weights[4]) + (i - first_layer_index) * _d_model * hidden_dim / tensor_para_.world_size_;
            decoder_layer_weights[i]->cross_attention_weights.key_weight.kernel =
                get_ptr<T>(_weights[5])
                + (i - first_layer_index) * _mem_d_model * hidden_dim / tensor_para_.world_size_;
            decoder_layer_weights[i]->cross_attention_weights.value_weight.kernel =
                get_ptr<T>(_weights[6])
                + (i - first_layer_index) * _mem_d_model * hidden_dim / tensor_para_.world_size_;
            decoder_layer_weights[i]->cross_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(_weights[7]) + (i - first_layer_index) * hidden_dim / tensor_para_.world_size_ * _d_model;
            decoder_layer_weights[i]->cross_attn_layernorm_weights.gamma =
                get_ptr<T>(_weights[8]) + (i - first_layer_index) * _d_model;
            decoder_layer_weights[i]->ffn_weights.intermediate_weight.kernel =
                get_ptr<T>(_weights[9]) + (i - first_layer_index) * _d_model * _inter_size / tensor_para_.world_size_;
            if (_use_gated_activation) {
                decoder_layer_weights[i]->ffn_weights.intermediate_weight2.kernel =
                    get_ptr<T>(_weights[10])
                    + (i - first_layer_index) * _d_model * _inter_size / tensor_para_.world_size_;
            }
            decoder_layer_weights[i]->ffn_weights.output_weight.kernel =
                get_ptr<T>(_weights[11]) + (i - first_layer_index) * _inter_size / tensor_para_.world_size_ * _d_model;

            if (_t5_with_bias) {
                decoder_layer_weights[i]->pre_layernorm_weights.beta =
                    get_ptr<T>(_weights[12]) + (i - first_layer_index) * _d_model;
                decoder_layer_weights[i]->self_attention_weights.query_weight.bias =
                    get_ptr<T>(_weights[13]) + (i - first_layer_index) * 3 * hidden_dim / tensor_para_.world_size_;
                decoder_layer_weights[i]->self_attention_weights.attention_output_weight.bias =
                    get_ptr<T>(_weights[14]) + (i - first_layer_index) * _d_model;
                decoder_layer_weights[i]->self_attn_layernorm_weights.beta =
                    get_ptr<T>(_weights[15]) + (i - first_layer_index) * _d_model;
                decoder_layer_weights[i]->cross_attention_weights.query_weight.bias =
                    get_ptr<T>(_weights[16]) + (i - first_layer_index) * hidden_dim / tensor_para_.world_size_;
                decoder_layer_weights[i]->cross_attention_weights.key_weight.bias =
                    get_ptr<T>(_weights[17]) + (i - first_layer_index) * hidden_dim / tensor_para_.world_size_;
                decoder_layer_weights[i]->cross_attention_weights.value_weight.bias =
                    get_ptr<T>(_weights[18]) + (i - first_layer_index) * hidden_dim / tensor_para_.world_size_;
                decoder_layer_weights[i]->cross_attention_weights.attention_output_weight.bias =
                    get_ptr<T>(_weights[19]) + (i - first_layer_index) * _d_model;
                decoder_layer_weights[i]->cross_attn_layernorm_weights.beta =
                    get_ptr<T>(_weights[20]) + (i - first_layer_index) * _d_model;
                decoder_layer_weights[i]->ffn_weights.intermediate_weight.bias =
                    get_ptr<T>(_weights[21]) + (i - first_layer_index) * _inter_size / tensor_para_.world_size_;
                if (_use_gated_activation) {
                    decoder_layer_weights[i]->ffn_weights.intermediate_weight2.bias =
                        get_ptr<T>(_weights[22]) + (i - first_layer_index) * _inter_size / tensor_para_.world_size_;
                }
                decoder_layer_weights[i]->ffn_weights.output_weight.bias =
                    get_ptr<T>(_weights[23]) + (i - first_layer_index) * _d_model;
            }
        }
    }

    ~FTT5Decoder() override
    {
        ft::ftNcclParamDestroy(tensor_para_);
        ft::ftNcclParamDestroy(pipeline_para_);
        cublasLtDestroy(_cublasltHandle);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void forward(size_t      batch_size,
                 size_t      step,
                 th::Tensor& from_tensor,
                 th::Tensor& memory_tensor,
                 th::Tensor& memory_sequence_length,
                 th::Tensor& sequence_length,
                 th::Tensor& output_tensor,
                 th::Tensor& self_cache_keys_tensor,
                 th::Tensor& self_cache_values_tensor,
                 th::Tensor& memory_cache_keys_tensor,
                 th::Tensor& memory_cache_values_tensor,
                 th::Tensor& relative_attention_bias_tensor) override
    {
        auto           stream        = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t _cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(_cublasHandle, stream);
        fastertransformer::Allocator<ft::AllocatorType::TH>* allocator =
            new fastertransformer::Allocator<ft::AllocatorType::TH>();
        ft::cublasMMWrapper* cublas_wrapper = new ft::cublasMMWrapper(
            _cublasHandle, _cublasltHandle, stream, cublas_algo_map_, cublas_wrapper_mutex_, allocator);

        if (std::is_same<T, half>::value) {
            cublas_wrapper->setFP16GemmConfig();
        }
#ifdef ENABLE_BF16
        else if (std::is_same<T, __nv_bfloat16>::value) {
            cublas_wrapper->setBF16GemmConfig();
        }
#endif
        else if (std::is_same<T, float>::value) {
            cublas_wrapper->setFP32GemmConfig();
        }

        // NeMo Megatron T5 with bias and use gelu or geglu.
        ft::ActivationType activation_type =
            _t5_with_bias ? (_use_gated_activation ? ft::ActivationType::GeGLU : ft::ActivationType::Gelu) :
                            ft::ActivationType::Relu;

        ft::T5Decoder<T> decoder = ft::T5Decoder<T>(batch_size,
                                                    _head_num,
                                                    _head_size,
                                                    _inter_size,
                                                    _d_model,
                                                    _layer_num,
                                                    _layernorm_eps,
                                                    stream,
                                                    cublas_wrapper,
                                                    allocator,
                                                    true,
                                                    tensor_para_,
                                                    pipeline_para_,
                                                    activation_type);

        int                     tmp_step = step + 1;
        std::vector<ft::Tensor> input_tensors =
            std::vector<ft::Tensor>{convert_tensor<T>(from_tensor),
                                    convert_tensor<T>(memory_tensor),
                                    convert_tensor<int>(memory_sequence_length),
                                    ft::Tensor{ft::MEMORY_GPU, ft::TYPE_BOOL, {batch_size}, nullptr},
                                    ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, {1}, &tmp_step},
                                    convert_tensor<int>(sequence_length),
                                    convert_tensor<T>(relative_attention_bias_tensor)};

        std::vector<ft::Tensor> output_tensors = std::vector<ft::Tensor>{convert_tensor<T>(output_tensor),
                                                                         convert_tensor<T>(self_cache_keys_tensor),
                                                                         convert_tensor<T>(self_cache_values_tensor),
                                                                         convert_tensor<T>(memory_cache_keys_tensor),
                                                                         convert_tensor<T>(memory_cache_values_tensor)};

        try {
            decoder.forward(&output_tensors, &input_tensors, &decoder_layer_weights);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what();
            exit(-1);
        }
        catch (...) {
            std::cout << "Runtime error";
            exit(-1);
        }
        delete cublas_wrapper;
        delete allocator;
    }

private:
    const int                                 _head_num;
    const int                                 _head_size;
    const int                                 _inter_size;
    const int                                 _d_model;
    std::vector<th::Tensor>                   _weights;
    const int                                 _layer_num;
    static constexpr float                    _layernorm_eps = 1e-6f;
    const int                                 _mem_d_model;
    cublasLtHandle_t                          _cublasltHandle;
    std::mutex*                               cublas_wrapper_mutex_;
    ft::cublasAlgoMap*                        cublas_algo_map_;
    std::vector<ft::T5DecoderLayerWeight<T>*> decoder_layer_weights;

    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;

    bool _t5_with_bias;
    bool _use_gated_activation;
    int  _position_embedding_type;
};

class FasterTransformerT5Decoder: public th::jit::CustomClassHolder {
public:
    FasterTransformerT5Decoder(th::Tensor self_layernorm_gamma,
                               th::Tensor self_kernel_q,
                               th::Tensor self_output_kernel,
                               th::Tensor cross_layernorm_gamma,
                               th::Tensor cross_kernel_q,
                               th::Tensor cross_kernel_k,
                               th::Tensor cross_kernel_v,
                               th::Tensor cross_output_kernel,
                               th::Tensor ffn_layernorm_gamma,
                               th::Tensor inter_kernel,
                               th::Tensor inter_kernel2,
                               th::Tensor output_kernel,
                               th::Tensor self_layernorm_beta,
                               th::Tensor self_bias_qkv,
                               th::Tensor self_output_bias,
                               th::Tensor cross_layernorm_beta,
                               th::Tensor cross_bias_q,
                               th::Tensor cross_bias_k,
                               th::Tensor cross_bias_v,
                               th::Tensor cross_output_bias,
                               th::Tensor ffn_layernorm_beta,
                               th::Tensor inter_bias,
                               th::Tensor inter_bias2,
                               th::Tensor output_bias,
                               int64_t    head_num,
                               int64_t    head_size,
                               int64_t    inter_size,
                               int64_t    d_model,
                               int64_t    layer_num,
                               int64_t    mem_d_model,
                               int64_t    tensor_para_size,
                               int64_t    pipeline_para_size,
                               bool       t5_with_bias,
                               bool       use_gated_activation,
                               int64_t    position_embedding_type);

    ~FasterTransformerT5Decoder();

    std::vector<th::Tensor> forward(int64_t    step,
                                    th::Tensor from_tensor,
                                    th::Tensor memory_tensor,
                                    th::Tensor memory_sequence_length,
                                    th::Tensor sequence_length,
                                    th::Tensor self_cache_keys_tensor,
                                    th::Tensor self_cache_values_tensor,
                                    th::Tensor memory_cache_keys_tensor,
                                    th::Tensor memory_cache_values_tensor,
                                    th::Tensor relative_attention_bias_tensor);

    std::vector<th::Tensor> get_pickle_info() const;

private:
    const at::ScalarType    _st;
    IFT5Decoder*            ftdecoder;
    th::Tensor              head_info;
    std::vector<th::Tensor> weights;
};

}  // namespace torch_ext
