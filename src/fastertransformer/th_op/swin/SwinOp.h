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

#include "src/fastertransformer/kernels/gen_relative_pos_bias.h"
#include "src/fastertransformer/kernels/transform_mask_kernels.h"
#include "src/fastertransformer/models/swin/Swin.h"
#include "src/fastertransformer/th_op/th_utils.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

class ISwinTransformerFunc {
public:
    virtual ~ISwinTransformerFunc() {}
    virtual void forward(int batch_size, th::Tensor& input, th::Tensor& output) = 0;
};

template<typename T>
class SwinTransformerFunc: public ISwinTransformerFunc {
public:
    int   sm_;
    int   max_batch_;
    int   img_size_;
    int   patch_size_;
    int   in_chans_;
    int   embed_dim_;
    int   window_size_;
    int*  depths_;
    int*  num_heads_;
    bool  ape_;
    bool  patch_norm_;
    int   layer_num_;
    float mlp_ratio_;
    bool  qkv_bias_;
    float qk_scale_;
    int   version_;

    SwinTransformerFunc(const int                      max_batch,
                        const int                      img_size,
                        const int                      patch_size,
                        const int                      in_chans,
                        const int                      embed_dim,
                        const int                      window_size,
                        int*                           depths,
                        int*                           num_heads,
                        const bool                     ape,
                        const bool                     patch_norm,
                        const int                      layer_num,
                        const float                    mlp_ratio,
                        const bool                     qkv_bias,
                        const float                    qk_scale,
                        const int                      version,
                        const std::vector<th::Tensor>& w):
        weights_(w),
        max_batch_(max_batch),
        img_size_(img_size),
        patch_size_(patch_size),
        in_chans_(in_chans),
        embed_dim_(embed_dim),
        window_size_(window_size),
        depths_(depths),
        num_heads_(num_heads),
        ape_(ape),
        patch_norm_(patch_norm),
        layer_num_(layer_num),
        mlp_ratio_(mlp_ratio),
        qkv_bias_(qkv_bias),
        qk_scale_(qk_scale),
        version_(version)
    {
        ft::check_cuda_error(cublasCreate(&cublas_handle_));
        ft::check_cuda_error(cublasLtCreate(&cublaslt_handle_));
        checkCUDNN(cudnnCreate(&cudnn_handle_));

        sm_ = ft::getSMVersion();

        cublas_algo_map_      = new ft::cublasAlgoMap(GEMM_CONFIG, "");
        cublas_wrapper_mutex_ = new std::mutex();

        // We arrange weights layer by layer and block by block inside each layer;
        // each block has 14 weights for version 1 or 15 weights for version 2
        // each layer has a block list && 5 weights
        // each swin transformer has a layer list && 6 weights && 3 handles

        int weight_num = 6;
        for (int l = 0; l < layer_num; l++) {
            for (int di = 0; di < depths[l]; di++) {
                weight_num += (version_ == 1 ? 14 : 15);
            }
            weight_num += 5;
        }
        if (weight_num != weights_.size()) {
            printf("[ERROR][SwinTransformerFunc] weights number %lu does not match expected number %d!\n",
                   weights_.size(),
                   weight_num);
            exit(-1);
        }

        int weight_idx = 0;
        for (int l = 0; l < layer_num; l++) {
            ft::SwinTransformerBasicLayerWeight<T> bl;
            for (int di = 0; di < depths[l]; di++) {
                ft::SwinTransformerBlockWeight<T> p;
                p.attention_weights.query_weight.kernel            = get_ptr<T>(weights_[weight_idx++]);
                p.attention_weights.query_weight.bias              = get_ptr<T>(weights_[weight_idx++]);
                p.attention_weights.attention_output_weight.kernel = get_ptr<T>(weights_[weight_idx++]);
                p.attention_weights.attention_output_weight.bias   = get_ptr<T>(weights_[weight_idx++]);
                p.ffn_weights.intermediate_weight.kernel           = get_ptr<T>(weights_[weight_idx++]);
                p.ffn_weights.intermediate_weight.bias             = get_ptr<T>(weights_[weight_idx++]);
                p.ffn_weights.output_weight.kernel                 = get_ptr<T>(weights_[weight_idx++]);
                p.ffn_weights.output_weight.bias                   = get_ptr<T>(weights_[weight_idx++]);
                p.attn_layernorm_weights.gamma                     = get_ptr<T>(weights_[weight_idx++]);
                p.attn_layernorm_weights.beta                      = get_ptr<T>(weights_[weight_idx++]);
                p.ffn_layernorm_weights.gamma                      = get_ptr<T>(weights_[weight_idx++]);
                p.ffn_layernorm_weights.beta                       = get_ptr<T>(weights_[weight_idx++]);
                p.attention_relative_pos_bias                      = get_ptr<T>(weights_[weight_idx++]);
                p.trt_relative_position_bias                       = get_ptr<T>(weights_[weight_idx++]);
                p.attention_logit_scale = (version_ == 1) ? nullptr : get_ptr<T>(weights_[weight_idx++]);
                bl.block_weight_list.push_back(p);
            }
            bl.merge_layernorm_weights.gamma = get_ptr<T>(weights_[weight_idx++]);
            bl.merge_layernorm_weights.beta  = get_ptr<T>(weights_[weight_idx++]);
            bl.merge_linear_weights.kernel   = get_ptr<T>(weights_[weight_idx++]);
            bl.attn_mask                     = get_ptr<T>(weights_[weight_idx++]);
            bl.trt_attn_mask                 = get_ptr<T>(weights_[weight_idx++]);
            params_.basic_layer_weight_list.push_back(bl);
        }
        params_.patchEmbed_linear_weights.kernel = get_ptr<T>(weights_[weight_idx++]);
        params_.patchEmbed_linear_weights.bias   = get_ptr<T>(weights_[weight_idx++]);
        params_.patchEmbed_norm_weights.gamma    = get_ptr<T>(weights_[weight_idx++]);
        params_.patchEmbed_norm_weights.beta     = get_ptr<T>(weights_[weight_idx++]);
        params_.norm_weights.gamma               = get_ptr<T>(weights_[weight_idx++]);
        params_.norm_weights.beta                = get_ptr<T>(weights_[weight_idx++]);
    }

    ~SwinTransformerFunc() override
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

        ft::Allocator<ft::AllocatorType::TH>* allocator = new ft::Allocator<ft::AllocatorType::TH>();

        ft::cublasMMWrapper* cublas_wrapper = new ft::cublasMMWrapper(
            cublas_handle_, cublaslt_handle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, allocator);

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

        ft::SwinTransformer<T>* swin_transformer = new ft::SwinTransformer<T>(max_batch_,
                                                                              img_size_,
                                                                              patch_size_,
                                                                              in_chans_,
                                                                              embed_dim_,
                                                                              window_size_,
                                                                              depths_,
                                                                              num_heads_,
                                                                              ape_,
                                                                              patch_norm_,
                                                                              layer_num_,
                                                                              mlp_ratio_,
                                                                              cudnn_handle_,
                                                                              stream,
                                                                              cublas_wrapper,
                                                                              allocator,
                                                                              true,
                                                                              qkv_bias_,
                                                                              qk_scale_,
                                                                              version_);

        ft::DataType  data_type = ft::getTensorType<T>();
        int           sm_ptr[1] = {sm_};
        ft::TensorMap input_tensors{
            {"input_query",
             ft::Tensor{
                 ft::MEMORY_GPU,
                 data_type,
                 std::vector<size_t>{(size_t)batch_size, (size_t)in_chans_, (size_t)img_size_, (size_t)img_size_},
                 get_ptr<T>(input)}},
            {"additional_params", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT8, std::vector<size_t>{1}, sm_ptr}}};

        ft::TensorMap output_tensors{{"hidden_features",
                                      ft::Tensor{ft::MEMORY_GPU,
                                                 data_type,
                                                 std::vector<size_t>{(size_t)batch_size, (size_t)output.size(1)},
                                                 get_ptr<T>(output)}}};
        swin_transformer->forward(&output_tensors, &input_tensors, params_);
        delete swin_transformer;
        delete cublas_wrapper;
        delete allocator;
    }

private:
    std::vector<th::Tensor>      weights_;
    cublasHandle_t               cublas_handle_   = nullptr;
    cublasLtHandle_t             cublaslt_handle_ = nullptr;
    cudnnHandle_t                cudnn_handle_    = nullptr;
    ft::SwinTransformerWeight<T> params_;
    std::mutex*                  cublas_wrapper_mutex_;
    ft::cublasAlgoMap*           cublas_algo_map_;
};

class SwinTransformerClass: public torch::jit::CustomClassHolder {
public:
    SwinTransformerClass(std::vector<th::Tensor> w,
                         th::Tensor              depths,
                         th::Tensor              num_heads,
                         int64_t                 max_batch,
                         int64_t                 img_size,
                         int64_t                 patch_size,
                         int64_t                 in_chans,
                         int64_t                 embed_dim,
                         int64_t                 window_size,
                         bool                    ape,
                         bool                    patch_norm,
                         int64_t                 layer_num,
                         double                  mlp_ratio,
                         bool                    qkv_bias = true,
                         double                  qk_scale = 1.0,
                         int64_t                 version  = 1);

    ~SwinTransformerClass();

    th::Tensor forward(th::Tensor input);

    std::vector<th::Tensor> get_pickle_info() const;

private:
    const at::ScalarType    st_;
    ISwinTransformerFunc*   swin_transformer_func_;
    std::vector<th::Tensor> weights_;
    th::Tensor              depths_;
    th::Tensor              num_heads_;
    th::Tensor              info_int_;
    th::Tensor              info_float_;
    int                     output_dim_;
};

template<typename T>
th::Tensor gen_relative_pos_bias_impl(th::Tensor table,
                                      th::Tensor relative_position_bias_index,
                                      th::Tensor cpb_mlp_weight1,
                                      th::Tensor cpb_mlp_bias1,
                                      th::Tensor cpb_mlp_weight2,
                                      const int  window_size,
                                      const int  head_num,
                                      const int  version)
{
    auto     stream     = at::cuda::getCurrentCUDAStream().stream();
    int      window_len = window_size * window_size;
    const T* table_ptr  = get_ptr<T>(table);
    CHECK_INPUT(relative_position_bias_index, at::ScalarType::Long);
    const int64_t* relative_position_bias_index_ptr = get_ptr<int64_t>(relative_position_bias_index);
    auto           output                           = torch::empty({head_num, window_len, window_len},
                               torch::dtype(table.dtype()).device(torch::kCUDA).requires_grad(false));
    T*             output_ptr                       = get_ptr<T>(output);

    if (version == 1) {  // version 1
        ft::invokeGenRelativePosBias(
            output_ptr, table_ptr, relative_position_bias_index_ptr, window_size, head_num, stream);
    }
    else if (version == 2) {  // version 2
        const T*  cpb_mlp_weight1_ptr = get_ptr<T>(cpb_mlp_weight1);
        const T*  cpb_mlp_bias1_ptr   = get_ptr<T>(cpb_mlp_bias1);
        const T*  cpb_mlp_weight2_ptr = get_ptr<T>(cpb_mlp_weight2);
        const int cpb_mlp_out_dim     = cpb_mlp_weight1.size(0);
        const int cpb_mlp_in_dim      = cpb_mlp_weight1.size(1);
        ft::invokeGenRelativePosBiasV2(output_ptr,
                                       table_ptr,
                                       relative_position_bias_index_ptr,
                                       cpb_mlp_weight1_ptr,
                                       cpb_mlp_bias1_ptr,
                                       cpb_mlp_weight2_ptr,
                                       window_size,
                                       cpb_mlp_in_dim,
                                       cpb_mlp_out_dim,
                                       head_num,
                                       stream);
    }

    return output;
}

template<typename T>
th::Tensor transform_trt_mask_impl(th::Tensor mask, const int B, const int S, bool use_int8 = false)
{
    int trt_S;
    if (S <= 64) {
        trt_S = 64;
    }
    else if (S <= 128 && use_int8 == false) {
        trt_S = 128;
    }
    else if (S <= 256) {
        trt_S = 256;
    }
    else {
        printf("[ERROR][transform_trt_mask_impl] unsupported seq_len %d\n", S);
        exit(-1);
    }
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto transformed_mask =
        torch::empty({B, trt_S, trt_S}, torch::dtype(mask.dtype()).device(torch::kCUDA).requires_grad(false));
    T*       transformed_mask_ptr = get_ptr<T>(transformed_mask);
    const T* mask_ptr             = get_ptr<T>(mask);
    ft::invokeTransformMask(transformed_mask_ptr, mask_ptr, B, S, stream);
    return transformed_mask;
}

th::Tensor gen_relative_pos_bias(th::Tensor    table,
                                 th::Tensor    relative_position_bias_index,
                                 const int64_t window_size,
                                 const int64_t head_num,
                                 th::Tensor    cpb_mlp_weight1,
                                 th::Tensor    cpb_mlp_bias1,
                                 th::Tensor    cpb_mlp_weight2,
                                 const int64_t version);

th::Tensor transform_trt_mask(th::Tensor mask, const int64_t B, const int64_t S, bool use_int8 = false);
}  // namespace torch_ext
