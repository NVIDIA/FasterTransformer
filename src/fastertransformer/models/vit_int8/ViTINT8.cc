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

#include "src/fastertransformer/models/vit_int8/ViTINT8.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/layernorm_int8_kernels.h"
#include "src/fastertransformer/kernels/layout_transformer_int8_kernels.h"
#include "src/fastertransformer/kernels/vit_kernels.h"

#define SAFE_FREE(x)                                                                                                   \
    if (x) {                                                                                                           \
        delete x;                                                                                                      \
        x = nullptr;                                                                                                   \
    }

namespace fastertransformer {

template<typename T>
void ViTTransformerINT8<T>::initialize()
{
    if (img_size_ % patch_size_ != 0) {
        std::ostringstream buffer;
        buffer << "[FT][ERROR] IMG size & PITCH size missmatch. " << img_size_ << " % " << patch_size_ << " !=0 \n";
        throw std::runtime_error(buffer.str());
    }

    if (head_num_ * head_dim_ != embed_dim_) {
        std::ostringstream buffer;
        buffer << "[FT][ERROR] Embed size and head number mismatch. Embed_dim=" << embed_dim_
               << "; head_num*head_dim = "
               << "(" << head_num_ << "*" << head_dim_ << ")=" << head_num_ * head_dim_ << std::endl;
        throw std::runtime_error(buffer.str());
    }

    max_seq_len_ = request_seq_len_;
    if ((attention_type_ == AttentionType::FUSED_MHA) && std::is_same<T, half>::value == true && head_dim_ <= 384) {
        attention_layer_ = new FusedAttentionLayerINT8<T>(max_batch_size_,
                                                          max_seq_len_,
                                                          head_num_,
                                                          head_dim_,
                                                          sm_,
                                                          q_scaling_,
                                                          int8_mode_,
                                                          stream_,
                                                          cublas_wrapper_,
                                                          allocator_,
                                                          is_free_buffer_after_forward_,
                                                          false);
    }
    else if (attention_type_ == AttentionType::UNFUSED_MHA) {
        if (request_seq_len_ % 32 != 0 && std::is_same<half, T>::value) {
            max_seq_len_ = (request_seq_len_ + 31) / 32 * 32;
            FT_LOG_DEBUG("Request sequence length(%lu) is odd with unfused mha. Padding to %lu\n",
                         request_seq_len_,
                         max_seq_len_);
        }
        attention_layer_ = new UnfusedAttentionLayerINT8<T>(max_batch_size_,
                                                            max_seq_len_,
                                                            head_num_,
                                                            head_dim_,
                                                            q_scaling_,
                                                            int8_mode_,
                                                            stream_,
                                                            cublas_wrapper_,
                                                            allocator_,
                                                            is_free_buffer_after_forward_,
                                                            false);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
    }

    ffn_layer_ = new GeluFfnLayerINT8<T>(max_batch_size_,
                                         max_seq_len_,
                                         head_num_,
                                         head_dim_,
                                         inter_size_,
                                         int8_mode_,
                                         stream_,
                                         cublas_wrapper_,
                                         allocator_,
                                         is_free_buffer_after_forward_,
                                         false);
}

template<typename T>
ViTTransformerINT8<T>::ViTTransformerINT8(size_t max_batch_size,
                                          size_t img_size,
                                          size_t chn_num,
                                          size_t patch_size,
                                          size_t embed_dim,
                                          size_t head_num,
                                          size_t inter_size,
                                          size_t num_layer,
                                          bool with_cls_token,
                                          int sm,
                                          float q_scaling,
                                          int int8_mode,
                                          cudaStream_t stream,
                                          cudnnHandle_t cudnn_handle,
                                          cublasMMWrapper* cublas_wrapper,
                                          IAllocator* allocator,
                                          bool is_free_buffer_after_forward,
                                          AttentionType attention_type):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    img_size_(img_size),
    chn_num_(chn_num),
    patch_size_(patch_size),
    request_seq_len_(img_size * img_size / patch_size / patch_size + (with_cls_token ? 1 : 0)),
    max_seq_len_(0),
    embed_dim_(embed_dim),
    head_num_(head_num),
    head_dim_(embed_dim / head_num),
    inter_size_(inter_size),
    num_layer_(num_layer),
    with_cls_token_(with_cls_token),
    sm_(sm),
    q_scaling_(q_scaling),
    attention_type_(attention_type),
    int8_mode_(int8_mode),
    cudnn_handle_(cudnn_handle)
{
    initialize();
}

template<typename T>
ViTTransformerINT8<T>::ViTTransformerINT8(ViTTransformerINT8<T> const& vit):
    BaseLayer(vit),
    max_batch_size_(vit.max_batch_size_),
    img_size_(vit.img_size_),
    chn_num_(vit.chn_num_),
    patch_size_(vit.patch_size_),
    max_seq_len_(vit.max_seq_len_),
    request_seq_len_(vit.request_seq_len_),
    embed_dim_(vit.embed_dim_),
    head_num_(vit.head_num_),
    head_dim_(vit.head_dim_),
    inter_size_(vit.inter_size_),
    num_layer_(vit.num_layer_),
    with_cls_token_(vit.with_cls_token_),
    sm_(vit.sm_),
    q_scaling_(vit.q_scaling_),
    attention_type_(vit.attention_type_),
    int8_mode_(vit.int8_mode_),
    cudnn_handle_(vit.cudnn_handle_)
{
    initialize();
}

template<typename T>
ViTTransformerINT8<T>::~ViTTransformerINT8()
{
    delete attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void ViTTransformerINT8<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        embed_buf_1_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_, false);
        embed_buf_2_ = (T*)allocator_->malloc(sizeof(int32_t) * max_batch_size_ * max_seq_len_ * embed_dim_, false);
        embed_buf_3_ = (T*)allocator_->malloc(sizeof(int32_t) * max_batch_size_ * max_seq_len_ * embed_dim_, false);
        embed_buf_4_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_, false);
        mask_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_, false);
        padding_offset_ = (int*)allocator_->malloc(sizeof(int) * max_batch_size_ * max_seq_len_, false);
        token_num_ = (size_t*)allocator_->malloc(sizeof(size_t) * 1, false);

        trt_mha_padding_offset_ = (int*)allocator_->malloc(sizeof(int) * (2 * max_batch_size_ + 1), false);
        seq_len_vec_ = (int*)allocator_->malloc(sizeof(int) * max_batch_size_, false);

        setSeqLenVec(max_batch_size_);
        setDefaultMask(max_batch_size_);
        setDefaultPaddingOffset(max_batch_size_);

        is_allocate_buffer_ = true;
    }
}

#define REMALLOC(var, size)                                                                                            \
    {                                                                                                                  \
        var = (decltype(var))allocator_->reMalloc(var, size, false);                                                   \
        FT_LOG_DEBUG("ReMalloc %s, ptr:%x, size:%lu", #var, var, size);                                                \
    }

template<typename T>
void ViTTransformerINT8<T>::allocateBuffer(size_t batch_size)
{
    if (is_allocate_buffer_ && batch_size <= max_batch_size_) {
        return;
    }

    batch_size = batch_size > max_batch_size_ ? batch_size : max_batch_size_;
    embed_buf_1_ = (T*)allocator_->reMalloc(embed_buf_1_, sizeof(T) * batch_size * max_seq_len_ * embed_dim_, false);
    embed_buf_2_ =
        (T*)allocator_->reMalloc(embed_buf_2_, sizeof(int32_t) * batch_size * max_seq_len_ * embed_dim_, false);
    embed_buf_3_ =
        (T*)allocator_->reMalloc(embed_buf_3_, sizeof(int32_t) * batch_size * max_seq_len_ * embed_dim_, false);
    embed_buf_4_ = (T*)allocator_->reMalloc(embed_buf_4_, sizeof(T) * batch_size * max_seq_len_ * embed_dim_, false);
    mask_buf_ = (T*)allocator_->reMalloc(mask_buf_, sizeof(T) * batch_size * max_seq_len_ * max_seq_len_, false);
    REMALLOC(padding_offset_, sizeof(int) * batch_size * max_seq_len_);
    REMALLOC(token_num_, sizeof(size_t) * 1);
    trt_mha_padding_offset_ =
        (int*)allocator_->reMalloc(trt_mha_padding_offset_, sizeof(int) * (2 * batch_size + 1), false);
    seq_len_vec_ = (int*)allocator_->reMalloc(seq_len_vec_, sizeof(int) * batch_size, false);

    resetBatch(batch_size);
    setSeqLenVec(batch_size);
    setDefaultPaddingOffset(batch_size);
    setDefaultMask(batch_size);

    is_allocate_buffer_ = true;
}

template<typename T>
void ViTTransformerINT8<T>::freeBuffer()
{
    allocator_->free(embed_buf_1_);
    allocator_->free(embed_buf_2_);
    allocator_->free(embed_buf_3_);
    allocator_->free(embed_buf_4_);
    allocator_->free(mask_buf_);
    allocator_->free(trt_mha_padding_offset_);
    allocator_->free(seq_len_vec_);
    allocator_->free(padding_offset_);
    allocator_->free(token_num_);

    is_allocate_buffer_ = false;
}

template<typename T>
void ViTTransformerINT8<T>::forward(std::vector<Tensor>* output_tensors,
                                    const std::vector<Tensor>* input_tensors,
                                    const ViTINT8Weight<T>* weights)
{
    // input_tensors:
    //      input_img, BCHW [batch, chn_num, img_size, img_size]
    // output tensors:
    //      output classification [batch, seq_len, embed_dim]

    const size_t input_batch_size = input_tensors->at(0).shape[0];
    const size_t input_chn_num = input_tensors->at(0).shape[1];
    const size_t input_img_size = input_tensors->at(0).shape[2];
    const size_t patch_resol = input_img_size / patch_size_;
    size_t seq_len = patch_resol * patch_resol + (with_cls_token_ ? 1 : 0);
    const bool need_padding =
        (attention_type_ == AttentionType::UNFUSED_MHA && seq_len % 32 != 0 && std::is_same<half, T>::value);

    FT_CHECK(input_img_size == img_size_);
    FT_CHECK(seq_len == request_seq_len_);
    FT_CHECK(input_tensors->size() == 1);
    FT_CHECK(input_tensors->at(0).shape.size() == 4);
    FT_CHECK(output_tensors->size() == 1);
    FT_CHECK(output_tensors->at(0).shape.size() == 3);
    allocateBuffer(input_batch_size);

    const T* input = (const T*)input_tensors->at(0).data;
    T* output = (T*)output_tensors->at(0).data;
    T* encoder_input_ptr = embed_buf_1_;

    // preprocess (patches embedding, concat class embed and add pos embed)
    patchEmbed(need_padding ? embed_buf_2_ : encoder_input_ptr,
               input,
               weights->pre_encoder_conv_weights.kernel,
               weights->pre_encoder_conv_weights.bias,
               weights->pre_transform_embeds.class_embed,
               weights->pre_transform_embeds.position_embed,
               input_batch_size,
               input_img_size,
               patch_size_,
               seq_len,
               input_chn_num,
               embed_dim_);

    DataType data_type = getTensorType<T>();
    size_t h_token_num = input_batch_size * seq_len;
    T* norm_out_buf = embed_buf_2_;
    T* attn_out_buf = embed_buf_3_;
    T* encoder_out_buf = embed_buf_1_;

    // get offsets
    Tensor* offset_tensor_ptr;
    if (attention_type_ == AttentionType::FUSED_MHA) {
        invokeGetTrtPaddingOffset(trt_mha_padding_offset_, seq_len_vec_, input_batch_size, stream_);
        offset_tensor_ptr =
            new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{input_batch_size + 1}, trt_mha_padding_offset_);
    }
    else {
        offset_tensor_ptr = new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{0}, nullptr);
        if (need_padding) {
            seq_len = (seq_len + 31) / 32 * 32;
            h_token_num = seq_len * input_batch_size;
            cudaMemsetAsync(encoder_input_ptr, 0, sizeof(T) * input_batch_size * seq_len * embed_dim_, stream_);
            invokeRebuildPadding(
                encoder_input_ptr, embed_buf_2_, padding_offset_, nopad_token_num_, head_num_ * head_dim_, stream_);
        }
    }
    T* from_buf = embed_buf_4_;
    invokeTransposeMatrixColMajorToCOL32(from_buf, encoder_input_ptr, embed_dim_, h_token_num, stream_);

    for (uint i = 0; i < num_layer_; i++) {
        const ScaleList* scalePtr = &(weights->vit_layer_weights[i].scale_list_);

        invokeLayernormCol32((int8_t*)norm_out_buf,
                             from_buf,
                             weights->vit_layer_weights[i].attn_layernorm_weights.gamma,
                             weights->vit_layer_weights[i].attn_layernorm_weights.beta,
                             h_token_num,
                             embed_dim_,
                             &(scalePtr->d_scale_list_[0 + 3]),
                             stream_);

        // Attention
        {

            std::vector<Tensor> attn_input_tensors{
                Tensor{MEMORY_GPU, TYPE_INT8, std::vector<size_t>{h_token_num, embed_dim_}, norm_out_buf},
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{input_batch_size, 1, seq_len, seq_len}, mask_buf_},
                *offset_tensor_ptr};
            std::vector<Tensor> attn_output_tensors{
                Tensor{MEMORY_GPU, TYPE_INT8, std::vector<size_t>{h_token_num, embed_dim_}, attn_out_buf}};
            attention_layer_->forward(
                &attn_output_tensors, &attn_input_tensors, &weights->vit_layer_weights[i].attention_weights);
        }

        if (int8_mode_ == 1) {
            invokeAddBiasResidualLayerNormCol32_noRes(
                (int8_t*)norm_out_buf,
                (int32_t*)attn_out_buf,
                from_buf,
                weights->vit_layer_weights[i].attention_weights.attention_output_weight.bias,
                weights->vit_layer_weights[i].ffn_layernorm_weights.gamma,
                weights->vit_layer_weights[i].ffn_layernorm_weights.beta,
                h_token_num,
                embed_dim_,
                stream_,
                &(scalePtr->d_scale_list_[scalePtr->p2_offset_ + 3 * embed_dim_]),
                &(scalePtr->d_scale_list_[36]),
                &(scalePtr->d_scale_list_[44 + 3]));
        }
        else if (int8_mode_ == 2) {
            invokeAddBiasResidualLayerNormCol32_noRes(
                (int8_t*)norm_out_buf,
                (int8_t*)attn_out_buf,
                from_buf,
                weights->vit_layer_weights[i].attention_weights.attention_output_weight.bias,
                weights->vit_layer_weights[i].ffn_layernorm_weights.gamma,
                weights->vit_layer_weights[i].ffn_layernorm_weights.beta,
                h_token_num,
                embed_dim_,
                stream_,
                &(scalePtr->d_scale_list_[40 + 1]),
                &(scalePtr->d_scale_list_[44 + 3]));
        }

        // FFN
        T* ffn_out_buf = attn_out_buf;
        {
            std::vector<Tensor> ffn_input_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, embed_dim_}, norm_out_buf}};
            std::vector<Tensor> ffn_output_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, embed_dim_}, ffn_out_buf}};
            ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &weights->vit_layer_weights[i].ffn_weights);
        }

        if (int8_mode_ == 1) {
            invokeAddBiasResidualCol32(from_buf,
                                       (const int32_t*)ffn_out_buf,
                                       from_buf,
                                       weights->vit_layer_weights[i].ffn_weights.output_weight.bias,
                                       h_token_num,
                                       embed_dim_,
                                       stream_,
                                       &(scalePtr->d_scale_list_[scalePtr->p2_offset_ + 8 * embed_dim_]),
                                       &(scalePtr->d_scale_list_[52]),
                                       1);
        }
        else if (int8_mode_ == 2) {
            invokeAddBiasResidualCol32(from_buf,
                                       (const int8_t*)ffn_out_buf,
                                       (const T*)from_buf,
                                       weights->vit_layer_weights[i].ffn_weights.output_weight.bias,
                                       h_token_num,
                                       embed_dim_,
                                       stream_,
                                       &(scalePtr->d_scale_list_[56 + 1]));
        }

        sync_check_cuda_error();
    }

    invokeTransposeMatrixCOL32ToColMajor(attn_out_buf, from_buf, h_token_num, embed_dim_, stream_);

    invokeGeneralLayerNorm(need_padding ? norm_out_buf : output,
                           attn_out_buf,
                           weights->post_transformer_layernorm_weights.gamma,
                           weights->post_transformer_layernorm_weights.beta,
                           h_token_num,
                           embed_dim_,
                           stream_);

    if (need_padding) {
        invokeRemovePadding(output, norm_out_buf, padding_offset_, nopad_token_num_, head_num_ * head_dim_, stream_);
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();

    SAFE_FREE(offset_tensor_ptr);
}

template<typename T>
bool ViTTransformerINT8<T>::resetBatch(size_t batch_size)
{
    if (max_batch_size_ < batch_size) {
        max_batch_size_ = batch_size;
    }

    return true;
}

template<typename T>
bool ViTTransformerINT8<T>::resetSeqLen(size_t seq_len)
{
    if (max_seq_len_ < seq_len) {
        max_seq_len_ = seq_len;
    }

    return true;
}

template<typename T>
bool ViTTransformerINT8<T>::setSeqLenVec(size_t batch_size)
{
    int* seq_len_vec = new int[batch_size];
    for (int i = 0; i < batch_size; i++) {
        seq_len_vec[i] = request_seq_len_;
    }
    cudaMemcpy(seq_len_vec_, seq_len_vec, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
    delete seq_len_vec;
    return true;
}

template<typename T>
void ViTTransformerINT8<T>::setDefaultMask(size_t batch_size)
{
    invokeBuildEncoderAttentionMask(mask_buf_, seq_len_vec_, batch_size, max_seq_len_, stream_);
}

template<typename T>
void ViTTransformerINT8<T>::setDefaultPaddingOffset(size_t batch_size)
{
    invokeGetPaddingOffset(
        &nopad_token_num_, token_num_, padding_offset_, seq_len_vec_, batch_size, max_seq_len_, stream_);
}

template<typename T>
void ViTTransformerINT8<T>::patchEmbed(T* output,
                                       const T* input,
                                       const T* kernel,
                                       const T* bias,
                                       const T* cls_embed,
                                       const T* pos_embed,
                                       const int batch,
                                       const int img_size,
                                       const int patch_size,
                                       const int seq_len,
                                       const int in_chans,
                                       const int embed_dim)
{
    T* tmp_buf = with_cls_token_ ? (output == embed_buf_1_ ? embed_buf_2_ : embed_buf_1_) : output;
    conv2d(
        tmp_buf, input, kernel, batch, img_size, img_size, in_chans, embed_dim, patch_size, patch_size, cudnn_handle_);
    int n = embed_dim;
    int s = seq_len;
    int m = batch * s;
    if (with_cls_token_) {
        FT_CHECK(cls_embed != nullptr);
        invokeAddBiasConcatClsTokenAddPosEmbed(tmp_buf, output, bias, cls_embed, pos_embed, m, n, s, stream_);
    }
    else {
        invokeAddBiasAddPosEmbed(tmp_buf, bias, pos_embed, m, n, s * n, stream_);
    }
}

template class ViTTransformerINT8<float>;
template class ViTTransformerINT8<half>;

}  // namespace fastertransformer
