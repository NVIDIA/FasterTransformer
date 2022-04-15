/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/vit_int8/ViTINT8.h"

#include "src/fastertransformer/th_op/th_utils.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

class IViTFunc {
public:
    virtual ~IViTFunc() {}
    virtual void forward(int batch_size, th::Tensor& input, th::Tensor& output) = 0;
};

template<typename T>
class VisionTransformerINT8Func: public IViTFunc {
public:
    int sm_;
    bool _use_ORDER_COL32_2R_4R4;
    int max_batch_;
    int img_size_;
    int patch_size_;
    int in_chans_;
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    int inter_size_;
    int layer_num_;
    float q_scaling_;
    int int8_mode_;
    bool with_cls_token_;

    VisionTransformerINT8Func(const int max_batch,
                              const int img_size,
                              const int patch_size,
                              const int in_chans,
                              const int embed_dim,
                              const int num_heads,
                              const int inter_size,
                              const int layer_num,
                              const float q_scaling,
                              const int int8_mode,
                              const bool with_cls_token,
                              const std::vector<th::Tensor>& w):
        weights_(w),
        max_batch_(max_batch),
        img_size_(img_size),
        patch_size_(patch_size),
        in_chans_(in_chans),
        embed_dim_(embed_dim),
        num_heads_(num_heads),
        head_dim_(embed_dim / num_heads),
        inter_size_(inter_size),
        layer_num_(layer_num),
        q_scaling_(q_scaling),
        int8_mode_(int8_mode),
        with_cls_token_(with_cls_token)
    {
        ft::check_cuda_error(cublasCreate(&cublas_handle_));
        ft::check_cuda_error(cublasLtCreate(&cublaslt_handle_));
        checkCUDNN(cudnnCreate(&cudnn_handle_));
        sm_ = ft::getSMVersion();

        _use_ORDER_COL32_2R_4R4 = false;

        if (sm_ >= 80) {
            _use_ORDER_COL32_2R_4R4 = true;
        }
        cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in", std::string(""));
        cublas_wrapper_mutex_ = new std::mutex();
        params_.vit_layer_weights.clear();
        params_.vit_layer_weights.resize(layer_num_);

        int idx_w = 0;
        params_.pre_transform_embeds.position_embed = get_ptr<T>(weights_[idx_w++]);
        if (with_cls_token) {
            params_.pre_transform_embeds.class_embed = get_ptr<T>(weights_[idx_w++]);
        }
        params_.pre_encoder_conv_weights.kernel = get_ptr<T>(weights_[idx_w++]);
        params_.pre_encoder_conv_weights.bias = get_ptr<T>(weights_[idx_w++]);

        for (int i = 0; i < layer_num_; i++) {
            auto& layer_weight = params_.vit_layer_weights[i];
            layer_weight.attn_layernorm_weights.gamma = get_ptr<T>(weights_[idx_w++]);
            layer_weight.attn_layernorm_weights.beta = get_ptr<T>(weights_[idx_w++]);
            layer_weight.ffn_layernorm_weights.gamma = get_ptr<T>(weights_[idx_w++]);
            layer_weight.ffn_layernorm_weights.beta = get_ptr<T>(weights_[idx_w++]);
            layer_weight.ffn_weights.intermediate_weight.kernel = get_ptr<T>(weights_[idx_w++]);
            layer_weight.ffn_weights.intermediate_weight.bias = get_ptr<T>(weights_[idx_w++]);
            layer_weight.ffn_weights.output_weight.kernel = get_ptr<T>(weights_[idx_w++]);
            layer_weight.ffn_weights.output_weight.bias = get_ptr<T>(weights_[idx_w++]);
            layer_weight.attention_weights.query_weight.kernel = get_ptr<T>(weights_[idx_w++]);
            layer_weight.attention_weights.query_weight.bias = get_ptr<T>(weights_[idx_w++]);
            layer_weight.attention_weights.key_weight.kernel = get_ptr<T>(weights_[idx_w++]);
            layer_weight.attention_weights.key_weight.bias = get_ptr<T>(weights_[idx_w++]);
            layer_weight.attention_weights.value_weight.kernel = get_ptr<T>(weights_[idx_w++]);
            layer_weight.attention_weights.value_weight.bias = get_ptr<T>(weights_[idx_w++]);
            layer_weight.attention_weights.attention_output_weight.kernel = get_ptr<T>(weights_[idx_w++]);
            layer_weight.attention_weights.attention_output_weight.bias = get_ptr<T>(weights_[idx_w++]);
        }
        params_.post_transformer_layernorm_weights.gamma = get_ptr<T>(weights_[idx_w++]);
        params_.post_transformer_layernorm_weights.beta = get_ptr<T>(weights_[idx_w++]);

        for (int i = 0; i < layer_num_; i++) {
            auto& layer_weight = params_.vit_layer_weights[i];
            layer_weight.scale_list_.size_ = ACTIVATION_AMAX_NUM + 9 * embed_dim + INT8O_GEMM_NUM + TRT_AMAX_NUM;
            layer_weight.scale_list_.p2_offset_ = ACTIVATION_AMAX_NUM;
            layer_weight.scale_list_.p3_offset_ = ACTIVATION_AMAX_NUM + 9 * embed_dim;
            layer_weight.scale_list_.p4_offset_ = ACTIVATION_AMAX_NUM + 9 * embed_dim + INT8O_GEMM_NUM;
            layer_weight.scale_list_.d_scale_list_ = get_ptr<float>(weights_[idx_w++]);
            layer_weight.scale_list_.h_scale_list_ = get_ptr<float>(weights_[idx_w++]);
            layer_weight.attention_weights.scale_list_ptr = &(layer_weight.scale_list_);
            layer_weight.ffn_weights.scale_list_ptr = &(layer_weight.scale_list_);
        }
    }

    ~VisionTransformerINT8Func() override
    {
        ft::check_cuda_error(cublasDestroy(cublas_handle_));
        ft::check_cuda_error(cublasLtDestroy(cublaslt_handle_));
        checkCUDNN(cudnnDestroy(cudnn_handle_));
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void forward(int batch_size, th::Tensor& input, th::Tensor& output) override
    {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        cublasSetStream(cublas_handle_, stream);

        ft::Allocator<ft::AllocatorType::TH>* allocator = new ft::Allocator<ft::AllocatorType::TH>();

        ft::cublasINT8MMWrapper* cublas_wrapper = new ft::cublasINT8MMWrapper(
            cublas_handle_, cublaslt_handle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, _use_ORDER_COL32_2R_4R4);

        if (std::is_same<T, half>::value) {
            cublas_wrapper->setFP16GemmConfig();
        }
        else if (std::is_same<T, float>::value) {
            cublas_wrapper->setFP32GemmConfig();
        }

        int seq_len = (img_size_ / patch_size_) * (img_size_ / patch_size_) + (with_cls_token_ ? 1 : 0);
        ft::AttentionType attention_type = ft::getAttentionType<T>(head_dim_, sm_, true, seq_len);

        auto vit = new ft::ViTTransformerINT8<T>(max_batch_,
                                                 img_size_,
                                                 in_chans_,
                                                 patch_size_,
                                                 embed_dim_,
                                                 num_heads_,
                                                 inter_size_,
                                                 layer_num_,
                                                 with_cls_token_,
                                                 sm_,
                                                 q_scaling_,
                                                 int8_mode_,
                                                 stream,
                                                 cudnn_handle_,
                                                 cublas_wrapper,
                                                 allocator,
                                                 true,
                                                 attention_type);

        ft::DataType data_type = ft::getTensorType<T>();
        int sm_ptr[1] = {sm_};
        std::vector<ft::Tensor> input_tensors = std::vector<ft::Tensor>{
            ft::Tensor{ft::MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{(size_t)batch_size, (size_t)in_chans_, (size_t)img_size_, (size_t)img_size_},
                       get_ptr<T>(input)}};

        std::vector<ft::Tensor> output_tensors = std::vector<ft::Tensor>{
            ft::Tensor{ft::MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{(size_t)batch_size, (size_t)seq_len, (size_t)embed_dim_},
                       get_ptr<T>(output)}};

        vit->forward(&output_tensors, &input_tensors, &params_);

        delete vit;
        delete cublas_wrapper;
        delete allocator;
    }

private:
    std::vector<th::Tensor> weights_;
    cublasHandle_t cublas_handle_ = nullptr;
    cublasLtHandle_t cublaslt_handle_ = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;
    ft::ViTINT8Weight<T> params_;
    std::mutex* cublas_wrapper_mutex_;
    ft::cublasAlgoMap* cublas_algo_map_;
};

class VisionTransformerINT8Class: public torch::jit::CustomClassHolder {
public:
    VisionTransformerINT8Class(std::vector<th::Tensor> w,
                               int64_t max_batch,
                               int64_t img_size,
                               int64_t patch_size,
                               int64_t in_chans,
                               int64_t embed_dim,
                               int64_t num_heads,
                               int64_t inter_size,
                               int64_t layer_num,
                               int64_t with_cls_token,
                               int64_t int8_mode);

    ~VisionTransformerINT8Class();

    th::Tensor forward(th::Tensor input);

    std::vector<th::Tensor> get_pickle_info() const;

private:
    const at::ScalarType st_;
    IViTFunc* vit_func_;
    std::vector<th::Tensor> weights_;
    th::Tensor info_int_;
    int output_seq_len_;
    int output_emb_dim_;
};

}  // namespace torch_ext
