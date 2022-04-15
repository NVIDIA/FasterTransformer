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
#include "src/fastertransformer/th_op/th_utils.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

class IFBertINT8 {
public:
    virtual ~IFBertINT8() {}
    virtual void forward(int batch_size,
                         int seq_len,
                         th::Tensor& input,
                         th::Tensor& sequence_lengths,
                         th::Tensor& output,
                         bool removing_padding) = 0;
};

template<typename T>
class FTBertINT8: public IFBertINT8 {
public:
    FTBertINT8(int head_num,
               int head_size,
               int layer_num,
               const float q_scaling,
               int int8_mode,
               bool sparse,
               const std::vector<th::Tensor>& w):
        _head_num(head_num),
        _head_size(head_size),
        _weights(w),
        _layer_num(layer_num),
        _q_scaling(q_scaling),
        _int8_mode(int8_mode),
        _sparse(sparse)
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
        _use_ORDER_COL32_2R_4R4 = false;
#if (CUDART_VERSION >= 11000)
        if (sm_ >= 80) {
            _use_ORDER_COL32_2R_4R4 = true;
        }
#endif
        std::string sp_config_fname = sparse ? "spigemm_config.in" : "";
        cublas_algo_map_ = new ft::cublasAlgoMap("igemm_config.in", sp_config_fname);
        cublas_wrapper_mutex_ = new std::mutex();
        bert_layer_weights.clear();
        bert_layer_weights.resize(_layer_num);

        for (int i = 0; i < _layer_num; i++) {
            bert_layer_weights[i].attention_weights.query_weight.kernel =
                get_ptr<T>(_weights[0]) + hidden_dim * hidden_dim * i;
            bert_layer_weights[i].attention_weights.query_weight.bias = get_ptr<T>(_weights[1]) + hidden_dim * i;
            bert_layer_weights[i].attention_weights.key_weight.kernel =
                get_ptr<T>(_weights[2]) + hidden_dim * hidden_dim * i;
            bert_layer_weights[i].attention_weights.key_weight.bias = get_ptr<T>(_weights[3]) + hidden_dim * i;
            bert_layer_weights[i].attention_weights.value_weight.kernel =
                get_ptr<T>(_weights[4]) + hidden_dim * hidden_dim * i;
            bert_layer_weights[i].attention_weights.value_weight.bias = get_ptr<T>(_weights[5]) + hidden_dim * i;
            bert_layer_weights[i].attention_weights.attention_output_weight.kernel =
                get_ptr<T>(_weights[6]) + hidden_dim * hidden_dim * i;
            bert_layer_weights[i].attention_weights.attention_output_weight.bias =
                get_ptr<T>(_weights[7]) + hidden_dim * i;
            bert_layer_weights[i].attn_layernorm_weights.gamma = get_ptr<T>(_weights[8]) + hidden_dim * i;
            bert_layer_weights[i].attn_layernorm_weights.beta = get_ptr<T>(_weights[9]) + hidden_dim * i;
            bert_layer_weights[i].ffn_weights.intermediate_weight.kernel =
                get_ptr<T>(_weights[10]) + hidden_dim * hidden_dim * 4 * i;
            bert_layer_weights[i].ffn_weights.intermediate_weight.bias = get_ptr<T>(_weights[11]) + hidden_dim * 4 * i;
            bert_layer_weights[i].ffn_weights.output_weight.kernel =
                get_ptr<T>(_weights[12]) + hidden_dim * hidden_dim * 4 * i;
            bert_layer_weights[i].ffn_weights.output_weight.bias = get_ptr<T>(_weights[13]) + hidden_dim * i;
            bert_layer_weights[i].ffn_layernorm_weights.gamma = get_ptr<T>(_weights[14]) + hidden_dim * i;
            bert_layer_weights[i].ffn_layernorm_weights.beta = get_ptr<T>(_weights[15]) + hidden_dim * i;

            // for scale_list
            bert_layer_weights[i].scale_list_.size_ =
                ACTIVATION_AMAX_NUM + 9 * hidden_dim + INT8O_GEMM_NUM + TRT_AMAX_NUM + SCALE_RESERVE_NUM;
            bert_layer_weights[i].scale_list_.p3_offset_ = ACTIVATION_AMAX_NUM + 9 * hidden_dim;
            bert_layer_weights[i].scale_list_.p4_offset_ = ACTIVATION_AMAX_NUM + 9 * hidden_dim + INT8O_GEMM_NUM;
            bert_layer_weights[i].scale_list_.d_scale_list_ =
                get_ptr<float>(_weights[16]) + i * bert_layer_weights[i].scale_list_.size_;
            bert_layer_weights[i].scale_list_.h_scale_list_ =
                get_ptr<float>(_weights[17]) + i * bert_layer_weights[i].scale_list_.size_;
            bert_layer_weights[i].attention_weights.scale_list_ptr = &(bert_layer_weights[i].scale_list_);
            bert_layer_weights[i].ffn_weights.scale_list_ptr = &(bert_layer_weights[i].scale_list_);
        }
        if (sparse) {
            for (int i = 0; i < _layer_num; ++i) {
                bert_layer_weights[i].attention_weights.query_weight.sp_kernel =
                    bert_layer_weights[i].attention_weights.query_weight.kernel;
                bert_layer_weights[i].attention_weights.key_weight.sp_kernel =
                    bert_layer_weights[i].attention_weights.key_weight.kernel;
                bert_layer_weights[i].attention_weights.value_weight.sp_kernel =
                    bert_layer_weights[i].attention_weights.value_weight.kernel;
                bert_layer_weights[i].attention_weights.attention_output_weight.sp_kernel =
                    bert_layer_weights[i].attention_weights.attention_output_weight.kernel;
                bert_layer_weights[i].ffn_weights.intermediate_weight.sp_kernel =
                    bert_layer_weights[i].ffn_weights.intermediate_weight.kernel;
                bert_layer_weights[i].ffn_weights.output_weight.sp_kernel =
                    bert_layer_weights[i].ffn_weights.output_weight.kernel;
            }
        }
    }

    ~FTBertINT8() override
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

    void forward(int batch_size,
                 int seq_len,
                 th::Tensor& input,
                 th::Tensor& sequence_lengths,
                 th::Tensor& output,
                 bool removing_padding) override
    {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        ft::cublasINT8MMWrapper cublas_wrapper =
#ifdef SPARSITY_ENABLED
            ft::cublasINT8MMWrapper(_cublasltHandle,
                                    _cusparseLtHandle,
                                    stream,
                                    cublas_algo_map_,
                                    cublas_wrapper_mutex_,
                                    _use_ORDER_COL32_2R_4R4);
#else
            ft::cublasINT8MMWrapper(
                _cublasltHandle, stream, cublas_algo_map_, cublas_wrapper_mutex_, _use_ORDER_COL32_2R_4R4);
#endif

        fastertransformer::Allocator<ft::AllocatorType::TH>* allocator =
            new fastertransformer::Allocator<ft::AllocatorType::TH>();
        ft::AttentionType attention_type =
            ft::getAttentionTypeINT8<T>(_head_size, sm_, removing_padding, seq_len, _int8_mode);

        ft::BertINT8<T>* bert_int8 = new ft::BertINT8<T>(batch_size,
                                                         seq_len,
                                                         _head_num,
                                                         _head_size,
                                                         _head_num * _head_size * 4,
                                                         _layer_num,
                                                         sm_,
                                                         _q_scaling,
                                                         _int8_mode,
                                                         stream,
                                                         &cublas_wrapper,
                                                         allocator,
                                                         true,
                                                         attention_type,
                                                         _sparse);

        ft::DataType data_type = ft::getTensorType<T>();
        std::vector<ft::Tensor> input_tensors = std::vector<ft::Tensor>{
            ft::Tensor{ft::MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{(size_t)batch_size, (size_t)seq_len, (size_t)(_head_num * _head_size)},
                       get_ptr<T>(input)},
            ft::Tensor{ft::MEMORY_GPU,
                       ft::TYPE_INT32,
                       std::vector<size_t>{(size_t)batch_size},
                       get_ptr<int>(sequence_lengths)}};

        std::vector<ft::Tensor> output_tensors = std::vector<ft::Tensor>{
            ft::Tensor{ft::MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{(size_t)batch_size, (size_t)seq_len, (size_t)(_head_num * _head_size)},
                       get_ptr<T>(output)}};

        try {
            bert_int8->forward(&output_tensors, &input_tensors, &bert_layer_weights);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what();
            exit(-1);
        }
        catch (...) {
            std::cout << "Runtime error";
            exit(-1);
        }
        delete bert_int8;
        delete allocator;
    }

private:
    const int _head_num;
    const int _head_size;
    std::vector<th::Tensor> _weights;
    const int _layer_num;
    const float _q_scaling;
    int _int8_mode;
    bool _sparse;
    int sm_;
    bool _use_ORDER_COL32_2R_4R4;
    cublasLtHandle_t _cublasltHandle;
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t _cusparseLtHandle;
#endif
    std::mutex* cublas_wrapper_mutex_;
    ft::cublasAlgoMap* cublas_algo_map_;
    std::vector<ft::BertLayerINT8Weight<T>> bert_layer_weights;
};

class FasterTransformerINT8Bert: public th::jit::CustomClassHolder {
public:
    FasterTransformerINT8Bert(th::Tensor q_kernel,
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
                              th::Tensor d_scale_list,
                              th::Tensor h_scale_list,
                              int64_t head_num,
                              int64_t head_size,
                              bool remove_padding,
                              int64_t layer_num,
                              int64_t int8_mode,
                              bool sparse,
                              double q_scaling);

    ~FasterTransformerINT8Bert();
    th::Tensor forward(th::Tensor input, th::Tensor sequence_lengths);

    std::vector<th::Tensor> get_pickle_info() const;

private:
    const at::ScalarType _st;
    bool _remove_padding;
    IFBertINT8* ftbert;
    th::Tensor head_info;
    th::Tensor scaling_info;
    std::vector<th::Tensor> weights;
};

}  // namespace torch_ext
