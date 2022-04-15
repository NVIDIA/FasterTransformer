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
#include "src/fastertransformer/th_op/th_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

class IFBert {
public:
    virtual ~IFBert() {}
    virtual void forward(size_t batch_size,
                         size_t seq_len,
                         th::Tensor& input,
                         th::Tensor& sequence_lengths,
                         th::Tensor& output,
                         bool removing_padding) = 0;
};

template<typename T>
class FTBert: public IFBert {
public:
    FTBert(int head_num,
           int head_size,
           int inter_size,
           int layer_num,
           bool sparse,
           float q_scaling,
           const std::vector<th::Tensor>& w):
        _head_num(head_num),
        _head_size(head_size),
        _inter_size(inter_size),
        _layer_num(layer_num),
        _weights(w),
        _sparse(sparse),
        _q_scaling(q_scaling)
    {
#ifndef SPARSITY_ENABLED
        if (sparse) {
            std::cout << "[WARNING] Sparsity support is not enabled. Will use dense GEMM instead.\n" << std::flush;
        }
#endif
        int hidden_dim = _head_num * _head_size;
        ft::check_cuda_error(cublasLtCreate(&_cublasltHandle));
        sm_ = ft::getSMVersion();
#ifdef SPARSITY_ENABLED
        if (sparse) {
            CHECK_CUSPARSE(cusparseLtInit(&_cusparseLtHandle));
        }
#endif
        std::string sp_config_fname = sparse ? "spgemm_config.in" : "";
        cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in", sp_config_fname);
        cublas_wrapper_mutex_ = new std::mutex();
        bert_weights.bert_layer_weights.clear();
        bert_weights.bert_layer_weights.resize(_layer_num);

        for (int i = 0; i < _layer_num; i++) {
            bert_weights.bert_layer_weights[i].attention_weights.query_weight.kernel =
                get_ptr<T>(_weights[0]) + hidden_dim * hidden_dim * i;
            bert_weights.bert_layer_weights[i].attention_weights.query_weight.bias =
                get_ptr<T>(_weights[1]) + hidden_dim * i;
            bert_weights.bert_layer_weights[i].attention_weights.key_weight.kernel =
                get_ptr<T>(_weights[2]) + hidden_dim * hidden_dim * i;
            bert_weights.bert_layer_weights[i].attention_weights.key_weight.bias =
                get_ptr<T>(_weights[3]) + hidden_dim * i;
            bert_weights.bert_layer_weights[i].attention_weights.value_weight.kernel =
                get_ptr<T>(_weights[4]) + hidden_dim * hidden_dim * i;
            bert_weights.bert_layer_weights[i].attention_weights.value_weight.bias =
                get_ptr<T>(_weights[5]) + hidden_dim * i;
            bert_weights.bert_layer_weights[i].attention_weights.attention_output_weight.kernel =
                get_ptr<T>(_weights[6]) + hidden_dim * hidden_dim * i;
            bert_weights.bert_layer_weights[i].attention_weights.attention_output_weight.bias =
                get_ptr<T>(_weights[7]) + hidden_dim * i;
            bert_weights.bert_layer_weights[i].attn_layernorm_weights.gamma = get_ptr<T>(_weights[8]) + hidden_dim * i;
            bert_weights.bert_layer_weights[i].attn_layernorm_weights.beta = get_ptr<T>(_weights[9]) + hidden_dim * i;
            bert_weights.bert_layer_weights[i].ffn_weights.intermediate_weight.kernel =
                get_ptr<T>(_weights[10]) + hidden_dim * hidden_dim * 4 * i;
            bert_weights.bert_layer_weights[i].ffn_weights.intermediate_weight.bias =
                get_ptr<T>(_weights[11]) + hidden_dim * 4 * i;
            bert_weights.bert_layer_weights[i].ffn_weights.output_weight.kernel =
                get_ptr<T>(_weights[12]) + hidden_dim * hidden_dim * 4 * i;
            bert_weights.bert_layer_weights[i].ffn_weights.output_weight.bias =
                get_ptr<T>(_weights[13]) + hidden_dim * i;
            bert_weights.bert_layer_weights[i].ffn_layernorm_weights.gamma = get_ptr<T>(_weights[14]) + hidden_dim * i;
            bert_weights.bert_layer_weights[i].ffn_layernorm_weights.beta = get_ptr<T>(_weights[15]) + hidden_dim * i;
        }
#ifdef SPARSITY_ENABLED
        if (sparse) {
            auto stream = at::cuda::getCurrentCUDAStream().stream();
            cublasHandle_t _cublasHandle = at::cuda::getCurrentCUDABlasHandle();
            cublasSetStream(_cublasHandle, stream);
            ft::cublasMMWrapper cublas_wrapper = ft::cublasMMWrapper(_cublasHandle,
                                                                     _cublasltHandle,
                                                                     _cusparseLtHandle,
                                                                     stream,
                                                                     cublas_algo_map_,
                                                                     cublas_wrapper_mutex_,
                                                                     nullptr);
            for (int i = 0; i < _layer_num; ++i) {
                bert_weights.bert_layer_weights[i].compress_weights(cublas_wrapper, hidden_dim);
            }
        }
#endif
    }

    ~FTBert() override
    {
        cublasLtDestroy(_cublasltHandle);
#ifdef SPARSITY_ENABLED
        if (_sparse) {
            cusparseLtDestroy(&_cusparseLtHandle);
        }
#endif
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void forward(size_t batch_size,
                 size_t seq_len,
                 th::Tensor& input,
                 th::Tensor& sequence_lengths,
                 th::Tensor& output,
                 bool removing_padding) override
    {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t _cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(_cublasHandle, stream);
        ft::Allocator<ft::AllocatorType::TH>* allocator = new ft::Allocator<ft::AllocatorType::TH>();
        ft::cublasMMWrapper* cublas_wrapper =
#ifdef SPARSITY_ENABLED
            new ft::cublasMMWrapper(_cublasHandle,
                                    _cublasltHandle,
                                    _cusparseLtHandle,
                                    stream,
                                    cublas_algo_map_,
                                    cublas_wrapper_mutex_,
                                    allocator);
#else
            new ft::cublasMMWrapper(
                _cublasHandle, _cublasltHandle, stream, cublas_algo_map_, cublas_wrapper_mutex_, allocator);
#endif

        if (std::is_same<T, half>::value) {
            cublas_wrapper->setFP16GemmConfig();
        }
        else if (std::is_same<T, float>::value) {
            cublas_wrapper->setFP32GemmConfig();
        }

        ft::AttentionType attention_type = ft::getAttentionType<T>(_head_size, sm_, removing_padding, seq_len);

        ft::Bert<T>* bert = new ft::Bert<T>(batch_size,
                                            seq_len,
                                            _head_num,
                                            _head_size,
                                            _inter_size,
                                            _layer_num,
                                            sm_,
                                            _q_scaling,
                                            stream,
                                            cublas_wrapper,
                                            allocator,
                                            true,
                                            attention_type,
                                            _sparse,
                                            ft::ActivationType::Gelu,
                                            ft::LayerNormType::post_layernorm);

        ft::DataType data_type = ft::getTensorType<T>();
        std::vector<ft::Tensor> input_tensors = std::vector<ft::Tensor>{
            ft::Tensor{ft::MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{batch_size, seq_len, (size_t)(_head_num * _head_size)},
                       get_ptr<T>(input)},
            ft::Tensor{
                ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{batch_size}, get_ptr<int>(sequence_lengths)}};

        std::vector<ft::Tensor> output_tensors = std::vector<ft::Tensor>{
            ft::Tensor{ft::MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{batch_size, seq_len, (size_t)(_head_num * _head_size)},
                       get_ptr<T>(output)}};

        try {
            bert->forward(&output_tensors, &input_tensors, &bert_weights);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what();
            exit(-1);
        }
        catch (...) {
            std::cout << "Runtime error";
            exit(-1);
        }
        delete bert;
        delete cublas_wrapper;
        delete allocator;
    }

private:
    const int _head_num;
    const int _head_size;
    const int _inter_size;
    const int _layer_num;
    std::vector<th::Tensor> _weights;
    bool _sparse;
    const float _q_scaling;
    int sm_;
    cublasLtHandle_t _cublasltHandle;
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t _cusparseLtHandle;
#endif
    std::mutex* cublas_wrapper_mutex_;
    ft::cublasAlgoMap* cublas_algo_map_;
    ft::BertWeight<T> bert_weights;
};

class FasterTransformerBert: public th::jit::CustomClassHolder {
public:
    FasterTransformerBert(th::Tensor q_kernel,
                          th::Tensor q_bias,
                          th::Tensor k_kernel,
                          th::Tensor k_bias,
                          th::Tensor v_kernel,
                          th::Tensor v_bias,
                          th::Tensor attr_output_kernel,
                          th::Tensor attr_output_bias,
                          th::Tensor attr_output_layernorm_gamma,
                          th::Tensor attr_output_layernorm_beta,
                          th::Tensor inter_kernel,
                          th::Tensor inter_bias,
                          th::Tensor output_kernel,
                          th::Tensor output_bias,
                          th::Tensor output_layernorm_gamma,
                          th::Tensor output_layernorm_beta,
                          int64_t head_num,
                          int64_t head_size,
                          int64_t inter_size,
                          bool remove_padding,
                          int64_t layer_num,
                          bool sparse,
                          double q_scaling);

    ~FasterTransformerBert();

    th::Tensor forward(th::Tensor input, th::Tensor sequence_lengths);

    std::vector<th::Tensor> get_pickle_info() const;

private:
    const at::ScalarType _st;
    bool _remove_padding;
    IFBert* ftbert;
    th::Tensor head_info;
    th::Tensor scaling_info;
    std::vector<th::Tensor> weights;
};

}  // namespace torch_ext
