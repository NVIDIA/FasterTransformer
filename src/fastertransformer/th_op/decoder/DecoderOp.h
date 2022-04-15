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
#include "src/fastertransformer/th_op/th_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

class IFDecoder {
public:
    virtual ~IFDecoder() {}
    virtual void forward(size_t batch_size,
                         size_t step,
                         th::Tensor& from_tensor,
                         th::Tensor& memory_tensor,
                         th::Tensor& memory_sequence_length,
                         th::Tensor& sequence_length,
                         th::Tensor& output_tensor,
                         th::Tensor& self_cache_keys_tensor,
                         th::Tensor& self_cache_values_tensor,
                         th::Tensor& memory_cache_keys_tensor,
                         th::Tensor& memory_cache_values_tensor) = 0;
};

template<typename T>
class FTDecoder: public IFDecoder {
public:
    FTDecoder(int head_num,
              int head_size,
              int inter_size,
              int layer_num,
              int mem_hidden_dim,
              const std::vector<th::Tensor>& w):
        _head_num(head_num),
        _head_size(head_size),
        _inter_size(inter_size),
        _weights(w),
        _layer_num(layer_num),
        _mem_hidden_dim(mem_hidden_dim)
    {
        int hidden_dim = _head_num * _head_size;
        ft::check_cuda_error(cublasLtCreate(&_cublasltHandle));
        cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in");

        cublas_wrapper_mutex_ = new std::mutex();
        decoder_layer_weights.clear();
        decoder_layer_weights.resize(_layer_num);

        for (int i = 0; i < _layer_num; ++i) {
            decoder_layer_weights[i].pre_layernorm_weights.gamma = get_ptr<T>(_weights[0]) + i * hidden_dim;
            decoder_layer_weights[i].pre_layernorm_weights.beta = get_ptr<T>(_weights[1]) + i * hidden_dim;
            decoder_layer_weights[i].self_attention_weights.query_weight.kernel =
                get_ptr<T>(_weights[2]) + i * hidden_dim * 3 * hidden_dim;
            decoder_layer_weights[i].self_attention_weights.query_weight.bias =
                get_ptr<T>(_weights[3]) + i * 3 * hidden_dim;
            decoder_layer_weights[i].self_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(_weights[4]) + i * hidden_dim * hidden_dim;
            decoder_layer_weights[i].self_attention_weights.attention_output_weight.bias =
                get_ptr<T>(_weights[5]) + i * hidden_dim;
            decoder_layer_weights[i].self_attn_layernorm_weights.gamma = get_ptr<T>(_weights[6]) + i * hidden_dim;
            decoder_layer_weights[i].self_attn_layernorm_weights.beta = get_ptr<T>(_weights[7]) + i * hidden_dim;
            decoder_layer_weights[i].cross_attention_weights.query_weight.kernel =
                get_ptr<T>(_weights[8]) + i * hidden_dim * hidden_dim;
            decoder_layer_weights[i].cross_attention_weights.key_weight.kernel =
                get_ptr<T>(_weights[9]) + i * mem_hidden_dim * hidden_dim;
            decoder_layer_weights[i].cross_attention_weights.value_weight.kernel =
                get_ptr<T>(_weights[10]) + i * mem_hidden_dim * hidden_dim;
            decoder_layer_weights[i].cross_attention_weights.query_weight.bias =
                get_ptr<T>(_weights[11]) + i * hidden_dim;
            decoder_layer_weights[i].cross_attention_weights.key_weight.bias =
                get_ptr<T>(_weights[12]) + i * hidden_dim;
            decoder_layer_weights[i].cross_attention_weights.value_weight.bias =
                get_ptr<T>(_weights[13]) + i * hidden_dim;
            decoder_layer_weights[i].cross_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(_weights[14]) + i * hidden_dim * hidden_dim;
            decoder_layer_weights[i].cross_attention_weights.attention_output_weight.bias =
                get_ptr<T>(_weights[15]) + i * hidden_dim;
            decoder_layer_weights[i].cross_attn_layernorm_weights.gamma = get_ptr<T>(_weights[16]) + i * hidden_dim;
            decoder_layer_weights[i].cross_attn_layernorm_weights.beta = get_ptr<T>(_weights[17]) + i * hidden_dim;
            decoder_layer_weights[i].ffn_weights.intermediate_weight.kernel =
                get_ptr<T>(_weights[18]) + i * hidden_dim * _inter_size;
            decoder_layer_weights[i].ffn_weights.intermediate_weight.bias = get_ptr<T>(_weights[19]) + i * _inter_size;
            decoder_layer_weights[i].ffn_weights.output_weight.kernel =
                get_ptr<T>(_weights[20]) + i * hidden_dim * _inter_size;
            decoder_layer_weights[i].ffn_weights.output_weight.bias = get_ptr<T>(_weights[21]) + i * hidden_dim;
        }
    }

    ~FTDecoder() override
    {
        cublasLtDestroy(_cublasltHandle);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void forward(size_t batch_size,
                 size_t step,
                 th::Tensor& from_tensor,
                 th::Tensor& memory_tensor,
                 th::Tensor& memory_sequence_length,
                 th::Tensor& sequence_length,
                 th::Tensor& output_tensor,
                 th::Tensor& self_cache_keys_tensor,
                 th::Tensor& self_cache_values_tensor,
                 th::Tensor& memory_cache_keys_tensor,
                 th::Tensor& memory_cache_values_tensor) override
    {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t _cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(_cublasHandle, stream);
        ft::Allocator<ft::AllocatorType::TH>* allocator = new ft::Allocator<ft::AllocatorType::TH>();
        ft::cublasMMWrapper* cublas_wrapper = new ft::cublasMMWrapper(
            _cublasHandle, _cublasltHandle, stream, cublas_algo_map_, cublas_wrapper_mutex_, allocator);

        if (std::is_same<T, half>::value) {
            cublas_wrapper->setFP16GemmConfig();
        }
        else if (std::is_same<T, float>::value) {
            cublas_wrapper->setFP32GemmConfig();
        }

        ft::Decoder<T>* decoder = new ft::Decoder<T>(
            batch_size, _head_num, _head_size, _inter_size, _layer_num, stream, cublas_wrapper, allocator, true);

        int tmp_step = step + 1;
        std::vector<ft::Tensor> input_tensors = std::vector<ft::Tensor>{
            convert_tensor<T>(from_tensor),
            convert_tensor<T>(memory_tensor),
            convert_tensor<int>(memory_sequence_length),
            ft::Tensor{ft::MEMORY_GPU, ft::TYPE_BOOL, {batch_size}, nullptr},
            ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, {1}, &tmp_step},
            convert_tensor<int>(sequence_length),
            ft::Tensor{ft::MEMORY_GPU,
                       ft::TYPE_INT32,
                       {batch_size, 1, (size_t)tmp_step},
                       nullptr}};  // Since we do gather in the Framework, we don't need id of indirection buffer

        std::vector<ft::Tensor> output_tensors = std::vector<ft::Tensor>{convert_tensor<T>(output_tensor),
                                                                         convert_tensor<T>(self_cache_keys_tensor),
                                                                         convert_tensor<T>(self_cache_values_tensor),
                                                                         convert_tensor<T>(memory_cache_keys_tensor),
                                                                         convert_tensor<T>(memory_cache_values_tensor)};

        try {
            decoder->forward(&output_tensors, &input_tensors, &decoder_layer_weights);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what();
            exit(-1);
        }
        catch (...) {
            std::cout << "Runtime error";
            exit(-1);
        }
        delete decoder;
        delete cublas_wrapper;
        delete allocator;
    }

private:
    const int _head_num;
    const int _head_size;
    const int _inter_size;
    std::vector<th::Tensor> _weights;
    const int _layer_num;
    const int _mem_hidden_dim;
    cublasLtHandle_t _cublasltHandle;
    std::mutex* cublas_wrapper_mutex_;
    ft::cublasAlgoMap* cublas_algo_map_;
    std::vector<ft::DecoderLayerWeight<T>> decoder_layer_weights;
};

class FasterTransformerDecoder: public th::jit::CustomClassHolder {
public:
    FasterTransformerDecoder(th::Tensor self_layernorm_gamma,
                             th::Tensor self_layernorm_beta,
                             th::Tensor self_kernel_q,
                             th::Tensor self_bias_q,
                             th::Tensor self_output_kernel,
                             th::Tensor self_output_bias,
                             th::Tensor cross_layernorm_gamma,
                             th::Tensor cross_layernorm_beta,
                             th::Tensor cross_kernel_q,
                             th::Tensor cross_kernel_k,
                             th::Tensor cross_kernel_v,
                             th::Tensor cross_bias_q,
                             th::Tensor cross_bias_k,
                             th::Tensor cross_bias_v,
                             th::Tensor cross_output_kernel,
                             th::Tensor cross_output_bias,
                             th::Tensor ffn_layernorm_gamma,
                             th::Tensor ffn_layernorm_beta,
                             th::Tensor inter_kernel,
                             th::Tensor inter_bias,
                             th::Tensor output_kernel,
                             th::Tensor output_bias,
                             int64_t head_num,
                             int64_t head_size,
                             int64_t inter_size,
                             int64_t layer_num,
                             int64_t mem_hidden_dim);

    ~FasterTransformerDecoder();

    std::vector<th::Tensor> forward(int64_t step,
                                    th::Tensor from_tensor,
                                    th::Tensor memory_tensor,
                                    th::Tensor memory_sequence_length,
                                    th::Tensor sequence_length,
                                    th::Tensor self_cache_keys_tensor,
                                    th::Tensor self_cache_values_tensor,
                                    th::Tensor memory_cache_keys_tensor,
                                    th::Tensor memory_cache_values_tensor);

    std::vector<th::Tensor> get_pickle_info() const;

private:
    const at::ScalarType _st;
    IFDecoder* ftdecoder;
    th::Tensor head_info;
    std::vector<th::Tensor> weights;
};

}  // namespace torch_ext
