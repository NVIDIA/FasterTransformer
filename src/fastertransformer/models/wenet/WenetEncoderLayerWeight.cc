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

#include "src/fastertransformer/models/wenet/WenetEncoderLayerWeight.h"
#include "src/fastertransformer/models/wenet/WenetKernels.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
WenetEncoderLayerWeight<T>::WenetEncoderLayerWeight(const size_t layer_id,
                                                    const size_t head_num,
                                                    const size_t size_per_head,
                                                    const size_t inter_size,
                                                    const size_t conv_module_kernel_size,
                                                    const bool   use_layernorm_in_conv_module):
    layer_id_(layer_id),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    conv_module_kernel_size_(conv_module_kernel_size),
    use_layernorm_in_conv_module_(use_layernorm_in_conv_module)
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");
    real_weights_num_ = use_layernorm_in_conv_module_ ? 37 : 35;
    initialize();
    mallocWeights();
    setWeightPtr();
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void WenetEncoderLayerWeight<T>::initialize()
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");
    int hidden_size     = head_num_ * size_per_head_;
    int idx             = 0;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size * inter_size_;
    weights_size[idx++] = inter_size_;
    weights_size[idx++] = hidden_size * inter_size_;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size * 2 * hidden_size;
    weights_size[idx++] = hidden_size * 2;
    weights_size[idx++] = hidden_size * 1 * conv_module_kernel_size_;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size * inter_size_;
    weights_size[idx++] = inter_size_;
    weights_size[idx++] = hidden_size * inter_size_;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size;

    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
WenetEncoderLayerWeight<T>::~WenetEncoderLayerWeight()
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");
    if (is_maintain_buffer == true) {

        norm_ff_macaron_weights.gamma = nullptr;
        norm_ff_macaron_weights.beta  = nullptr;

        feed_forward_macaron_weights.intermediate_weight.kernel = nullptr;
        feed_forward_macaron_weights.intermediate_weight.bias   = nullptr;
        feed_forward_macaron_weights.output_weight.kernel       = nullptr;
        feed_forward_macaron_weights.output_weight.bias         = nullptr;

        attn_layernorm_weights.gamma = nullptr;
        attn_layernorm_weights.beta  = nullptr;

        attention_weights.query_weight.kernel            = nullptr;
        attention_weights.query_weight.bias              = nullptr;
        attention_weights.key_weight.kernel              = nullptr;
        attention_weights.key_weight.bias                = nullptr;
        attention_weights.value_weight.kernel            = nullptr;
        attention_weights.value_weight.bias              = nullptr;
        attention_weights.attention_output_weight.kernel = nullptr;
        attention_weights.attention_output_weight.bias   = nullptr;
        attention_weights.pos_weight.kernel              = nullptr;
        attention_weights.pos_weight.bias                = nullptr;
        attention_weights.pos_bias_u                     = nullptr;
        attention_weights.pos_bias_v                     = nullptr;

        norm_conv_weights.gamma = nullptr;
        norm_conv_weights.beta  = nullptr;

        conv_module_weights.pointwise_conv1_weight.kernel = nullptr;
        conv_module_weights.pointwise_conv1_weight.bias   = nullptr;
        conv_module_weights.depthwise_conv_weight.kernel  = nullptr;
        conv_module_weights.depthwise_conv_weight.bias    = nullptr;
        conv_module_weights.pointwise_conv2_weight.kernel = nullptr;
        conv_module_weights.pointwise_conv2_weight.bias   = nullptr;

        ffn_layernorm_weights.gamma = nullptr;
        ffn_layernorm_weights.beta  = nullptr;

        ffn_weights.intermediate_weight.kernel = nullptr;
        ffn_weights.intermediate_weight.bias   = nullptr;
        ffn_weights.output_weight.kernel       = nullptr;
        ffn_weights.output_weight.bias         = nullptr;

        norm_final_weights.gamma = nullptr;
        norm_final_weights.beta  = nullptr;

        if (use_layernorm_in_conv_module_) {
            conv_module_weights.norm_weights.gamma = nullptr;
            conv_module_weights.norm_weights.beta  = nullptr;
        }

        FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");
    }
}
template<typename T>
WenetEncoderLayerWeight<T>::WenetEncoderLayerWeight(const WenetEncoderLayerWeight& other):
    layer_id_(other.layer_id_),
    head_num_(other.head_num_),
    size_per_head_(other.size_per_head_),
    inter_size_(other.inter_size_),
    conv_module_kernel_size_(other.conv_module_kernel_size_),
    use_layernorm_in_conv_module_(other.use_layernorm_in_conv_module_)
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");
    real_weights_num_ = use_layernorm_in_conv_module_ ? 37 : 35;
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
WenetEncoderLayerWeight<T>& WenetEncoderLayerWeight<T>::operator=(const WenetEncoderLayerWeight<T>& other)
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");
    layer_id_                     = other.layer_id_;
    head_num_                     = other.head_num_;
    size_per_head_                = other.size_per_head_;
    inter_size_                   = other.inter_size_;
    conv_module_kernel_size_      = other.conv_module_kernel_size_;
    real_weights_num_             = other.real_weights_num_;
    use_layernorm_in_conv_module_ = other.use_layernorm_in_conv_module_;
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");

    return *this;
}
/*
#ifdef SPARSITY_ENABLED
template<typename T>
void WenetEncoderLayerWeight<T>::compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim)
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");
    int inter_size = hidden_dim * 4;
    deviceMalloc(&sp_weights_ptr[0], weights_size[0]);
    deviceMalloc(&sp_weights_ptr[1], weights_size[1]);
    deviceMalloc(&sp_weights_ptr[2], weights_size[2]);
    deviceMalloc(&sp_weights_ptr[3], weights_size[3]);
    deviceMalloc(&sp_weights_ptr[4], weights_size[5]);
    deviceMalloc(&sp_weights_ptr[5], weights_size[6]);

    cublas_wrapper.compressMatrix(attention_weights.query_weight.kernel,
                                  sp_weights_ptr[0],
                                  d_model_,
                                  (head_num_ / tensor_para_size_) * size_per_head_);
    cublas_wrapper.compressMatrix(attention_weights.key_weight.kernel,
                                  sp_weights_ptr[1],
                                  d_model_,
                                  (head_num_ / tensor_para_size_) * size_per_head_);
    cublas_wrapper.compressMatrix(attention_weights.value_weight.kernel,
                                  sp_weights_ptr[2],
                                  d_model_,
                                  (head_num_ / tensor_para_size_) * size_per_head_);
    cublas_wrapper.compressMatrix(attention_weights.attention_output_weight.kernel,
                                  sp_weights_ptr[3],
                                  (head_num_ / tensor_para_size_) * size_per_head_,
                                  d_model_);
    cublas_wrapper.compressMatrix(
        ffn_weights.intermediate_weight.kernel, sp_weights_ptr[4], inter_size / tensor_para_size_, d_model_);
    cublas_wrapper.compressMatrix(
        ffn_weights.output_weight.kernel, sp_weights_ptr[5], d_model_, inter_size / tensor_para_size_);
    attention_weights.query_weight.sp_kernel = sp_weights_ptr[0];
    attention_weights.key_weight.sp_kernel = sp_weights_ptr[1];
    attention_weights.value_weight.sp_kernel = sp_weights_ptr[2];
    attention_weights.attention_output_weight.sp_kernel = sp_weights_ptr[3];
    ffn_weights.intermediate_weight.sp_kernel = sp_weights_ptr[4];
    ffn_weights.output_weight.sp_kernel = sp_weights_ptr[5];
    is_maintain_sp_buffer = true;
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");
}
#endif
*/
template<typename T>
void WenetEncoderLayerWeight<T>::setWeightPtr()
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");
    int idx                       = 0;
    norm_ff_macaron_weights.gamma = weights_ptr[idx++];
    norm_ff_macaron_weights.beta  = weights_ptr[idx++];

    feed_forward_macaron_weights.intermediate_weight.kernel = weights_ptr[idx++];
    feed_forward_macaron_weights.intermediate_weight.bias   = weights_ptr[idx++];
    feed_forward_macaron_weights.output_weight.kernel       = weights_ptr[idx++];
    feed_forward_macaron_weights.output_weight.bias         = weights_ptr[idx++];

    attn_layernorm_weights.gamma = weights_ptr[idx++];
    attn_layernorm_weights.beta  = weights_ptr[idx++];

    attention_weights.query_weight.kernel            = weights_ptr[idx++];
    attention_weights.query_weight.bias              = weights_ptr[idx++];
    attention_weights.key_weight.kernel              = weights_ptr[idx++];
    attention_weights.key_weight.bias                = weights_ptr[idx++];
    attention_weights.value_weight.kernel            = weights_ptr[idx++];
    attention_weights.value_weight.bias              = weights_ptr[idx++];
    attention_weights.attention_output_weight.kernel = weights_ptr[idx++];
    attention_weights.attention_output_weight.bias   = weights_ptr[idx++];
    attention_weights.pos_weight.kernel              = weights_ptr[idx++];
    // attention_weights.pos_weight.bias = weights_ptr[idx++];
    attention_weights.pos_bias_u = weights_ptr[idx++];
    attention_weights.pos_bias_v = weights_ptr[idx++];

    norm_conv_weights.gamma = weights_ptr[idx++];
    norm_conv_weights.beta  = weights_ptr[idx++];

    conv_module_weights.pointwise_conv1_weight.kernel = weights_ptr[idx++];
    conv_module_weights.pointwise_conv1_weight.bias   = weights_ptr[idx++];
    conv_module_weights.depthwise_conv_weight.kernel  = weights_ptr[idx++];
    conv_module_weights.depthwise_conv_weight.bias    = weights_ptr[idx++];
    conv_module_weights.pointwise_conv2_weight.kernel = weights_ptr[idx++];
    conv_module_weights.pointwise_conv2_weight.bias   = weights_ptr[idx++];

    ffn_layernorm_weights.gamma = weights_ptr[idx++];
    ffn_layernorm_weights.beta  = weights_ptr[idx++];

    ffn_weights.intermediate_weight.kernel = weights_ptr[idx++];
    ffn_weights.intermediate_weight.bias   = weights_ptr[idx++];
    ffn_weights.output_weight.kernel       = weights_ptr[idx++];
    ffn_weights.output_weight.bias         = weights_ptr[idx++];

    norm_final_weights.gamma = weights_ptr[idx++];
    norm_final_weights.beta  = weights_ptr[idx++];

    if (use_layernorm_in_conv_module_) {
        conv_module_weights.norm_weights.gamma = weights_ptr[idx++];
        conv_module_weights.norm_weights.beta  = weights_ptr[idx++];
    }

    is_maintain_buffer = true;
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void WenetEncoderLayerWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");
    for (int i = 0; i < real_weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }
    is_maintain_buffer = true;
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void WenetEncoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");

    FT_CHECK(is_maintain_buffer == true);

    std::vector<std::string> weights_name;
    std::string              name_prefix = "encoder.encoders.";
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_ff_macaron.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_ff_macaron.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward_macaron.w_1.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward_macaron.w_1.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward_macaron.w_2.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward_macaron.w_2.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_mha.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_mha.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_q.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_q.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_k.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_k.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_v.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_v.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_out.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_out.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_pos.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.pos_bias_u");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.pos_bias_v");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_conv.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_conv.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".conv_module.pointwise_conv1.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".conv_module.pointwise_conv1.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".conv_module.depthwise_conv.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".conv_module.depthwise_conv.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".conv_module.pointwise_conv2.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".conv_module.pointwise_conv2.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_ff.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_ff.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward.w_1.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward.w_1.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward.w_2.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward.w_2.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_final.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_final.bias");

    if (use_layernorm_in_conv_module_) {
        weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".conv_module.norm.weight");
        weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".conv_module.norm.bias");
    }

    for (size_t i = 0; i < weights_name.size(); ++i) {
        loadWeightFromBin<T>(weights_ptr[i], {weights_size[i]}, dir_path + weights_name[i] + ".bin", model_file_type);
    }

    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");
}

template struct WenetEncoderLayerWeight<float>;
template struct WenetEncoderLayerWeight<half>;

}  // namespace fastertransformer
