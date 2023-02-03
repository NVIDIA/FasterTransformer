/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2022.  Authored by Yuqing Ding.
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

#include "src/fastertransformer/models/wenet/WenetEncoder.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/beam_search_topk_kernels.h"
#include "src/fastertransformer/kernels/decoding_kernels.h"

namespace fastertransformer {

template<typename T>
void WenetEncoder<T>::initialize()
{
    check_cuda_error(cudaStreamCreate(&stream2_));
    check_cuda_error(cudaEventCreate(&stream_finished_));
    check_cuda_error(cudaEventCreate(&stream2_finished_));
    check_cuda_error(cudaMallocHost((void**)&h_var_token_num_, sizeof(size_t)));

    attention_layer_ = new RelPositionAttentionLayer<T>(0,
                                                        0,
                                                        head_num_,
                                                        size_per_head_,
                                                        q_scaling_,
                                                        stream_,
                                                        cublas_wrapper_,
                                                        allocator_,
                                                        is_free_buffer_after_forward_,
                                                        sparse_);

    if (activation_type_ == ActivationType::Gelu) {
        ffn_layer_ = new GeluFfnLayer<T>(0,
                                         0,
                                         head_num_,
                                         size_per_head_,
                                         0,  // expert_num
                                         inter_size_,
                                         stream_,
                                         cublas_wrapper_,
                                         allocator_,
                                         is_free_buffer_after_forward_,
                                         sparse_);
    }
    else if (activation_type_ == ActivationType::Relu) {
        ffn_layer_ = new ReluFfnLayer<T>(0,
                                         0,
                                         head_num_,
                                         size_per_head_,
                                         0,  // expert_num
                                         inter_size_,
                                         stream_,
                                         cublas_wrapper_,
                                         allocator_,
                                         is_free_buffer_after_forward_,
                                         sparse_);
    }
    else if (activation_type_ == ActivationType::Silu) {
        ffn_layer_ = new SiluFfnLayer<T>(0,
                                         0,
                                         head_num_,
                                         size_per_head_,
                                         0,  // expert_num
                                         inter_size_,
                                         stream_,
                                         cublas_wrapper_,
                                         allocator_,
                                         is_free_buffer_after_forward_,
                                         sparse_);
    }

    conformer_conv_layer_ = new ConformerConvLayer<T>(0,
                                                      0,
                                                      head_num_,
                                                      size_per_head_,
                                                      conv_module_kernel_size_,
                                                      stream_,
                                                      cublas_wrapper_,
                                                      allocator_,
                                                      is_free_buffer_after_forward_,
                                                      sparse_,
                                                      0,  // int8_mode, not supported yet
                                                      use_layernorm_in_conv_module_);
}

template<typename T>
WenetEncoder<T>::WenetEncoder(size_t           max_batch_size,
                              size_t           max_seq_len,
                              size_t           head_num,
                              size_t           size_per_head,
                              size_t           feature_size,
                              size_t           max_len,
                              size_t           inter_size,
                              size_t           d_model,
                              size_t           num_layer,
                              size_t           vocab_size,
                              size_t           conv_module_kernel_size,
                              int              sm,
                              float            q_scaling,
                              cudnnHandle_t    cudnn_handle,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              AttentionType    attention_type,
                              bool             sparse,
                              ActivationType   activation_type,
                              bool             use_layernorm_in_conv_module):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    feature_size_(feature_size),
    max_len_(max_len),
    d_model_(d_model),
    hidden_units_(head_num_ * size_per_head_),
    num_layer_(num_layer),
    sm_(sm),
    q_scaling_(q_scaling),
    sparse_(sparse),
    activation_type_(activation_type),
    vocab_size_(vocab_size),
    conv_module_kernel_size_(conv_module_kernel_size),
    cudnn_handle_(cudnn_handle),
    use_layernorm_in_conv_module_(use_layernorm_in_conv_module)
{
    initialize();
}

template<typename T>
WenetEncoder<T>::WenetEncoder(WenetEncoder<T> const& wenet_encoder):
    BaseLayer(wenet_encoder),
    head_num_(wenet_encoder.head_num_),
    size_per_head_(wenet_encoder.size_per_head_),
    feature_size_(wenet_encoder.feature_size_),
    max_len_(wenet_encoder.max_len_),
    inter_size_(wenet_encoder.inter_size_),
    d_model_(wenet_encoder.d_model_),
    hidden_units_(wenet_encoder.hidden_units_),
    num_layer_(wenet_encoder.num_layer_),
    sm_(wenet_encoder.sm_),
    q_scaling_(wenet_encoder.q_scaling_),
    sparse_(wenet_encoder.sparse_),
    activation_type_(wenet_encoder.activation_type_),
    vocab_size_(wenet_encoder.vocab_size_),
    conv_module_kernel_size_(wenet_encoder.conv_module_kernel_size_),
    cudnn_handle_(wenet_encoder.cudnn_handle_),
    use_layernorm_in_conv_module_(wenet_encoder.use_layernorm_in_conv_module_)
{
    initialize();
}

template<typename T>
WenetEncoder<T>::~WenetEncoder()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    delete attention_layer_;
    delete ffn_layer_;
    delete conformer_conv_layer_;

    check_cuda_error(cudaFreeHost(h_var_token_num_));
    check_cuda_error(cudaEventDestroy(stream2_finished_));
    check_cuda_error(cudaEventDestroy(stream_finished_));
    check_cuda_error(cudaStreamDestroy(stream2_));

    freeBuffer();
}

template<typename T>
void WenetEncoder<T>::setStream(cudaStream_t stream)
{
    attention_layer_->setStream(stream);
    ffn_layer_->setStream(stream);
    conformer_conv_layer_->setStream(stream);
    BaseLayer::setStream(stream);
}

template<typename T>
void WenetEncoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void WenetEncoder<T>::allocateBuffer(
    size_t batch_size, size_t seq_len, size_t feature_size, size_t kernel_size, size_t stride)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t feature_size1   = (feature_size_ - kernel_size) / stride + 1;
    size_t feature_size2   = (feature_size1 - kernel_size) / stride + 1;
    size_t seq_len1        = (seq_len - kernel_size) / stride + 1;
    size_t seq_len2        = (seq_len1 - kernel_size) / stride + 1;
    size_t cur_tensor_size = batch_size * seq_len2 * hidden_units_;

    // Position Embed
    inter_conv1_input_buf_ =
        (T*)allocator_->reMalloc(inter_conv1_input_buf_, sizeof(T) * batch_size * seq_len * feature_size_, false);
    inter_conv1_output_buf_ = (T*)allocator_->reMalloc(
        inter_conv1_output_buf_, sizeof(T) * batch_size * d_model_ * seq_len1 * feature_size1, false);
    inter_conv2_output_buf_ = (T*)allocator_->reMalloc(
        inter_conv2_output_buf_, sizeof(T) * batch_size * d_model_ * seq_len2 * feature_size2, false);
    inter_fc_input_buf_ = (T*)allocator_->reMalloc(
        inter_fc_input_buf_, sizeof(T) * batch_size * seq_len2 * d_model_ * feature_size2, false);
    // Current workspace used for CuDNN Convolution is 1 << 29
    conv_workspace_ = (T*)allocator_->reMalloc(conv_workspace_, 1 << 29, false);
    // Position Embed

    input_hidden_state_ =
        (T*)allocator_->reMalloc(input_hidden_state_, sizeof(T) * batch_size * seq_len2 * hidden_units_, false);
    pos_emb_tensor_ = (T*)allocator_->reMalloc(pos_emb_tensor_, sizeof(T) * cur_tensor_size, false);
    attention_mask_ =
        (T*)allocator_->reMalloc(attention_mask_, sizeof(T) * batch_size * 1 * seq_len2 * seq_len2, false);

    token_num_      = (size_t*)allocator_->reMalloc(token_num_, sizeof(size_t) * 1, false);
    padding_offset_ = (int*)allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * seq_len2, false);
    bid_start_end_  = (int*)allocator_->reMalloc(bid_start_end_, sizeof(int) * batch_size * seq_len2 * 3, false);

    normed_from_tensor_ = (T*)allocator_->reMalloc(normed_from_tensor_, sizeof(T) * cur_tensor_size, false);

    ffn_out_buf_        = (T*)allocator_->reMalloc(ffn_out_buf_, sizeof(T) * cur_tensor_size, false);
    normed_ffn_out_buf_ = (T*)allocator_->reMalloc(normed_ffn_out_buf_, sizeof(T) * cur_tensor_size, false);

    attn_out_buf_        = (T*)allocator_->reMalloc(attn_out_buf_, sizeof(T) * cur_tensor_size, false);
    normed_attn_out_buf_ = (T*)allocator_->reMalloc(normed_attn_out_buf_, sizeof(T) * cur_tensor_size, false);

    conv_out_buf_        = (T*)allocator_->reMalloc(conv_out_buf_, sizeof(T) * cur_tensor_size, false);
    normed_conv_out_buf_ = (T*)allocator_->reMalloc(normed_conv_out_buf_, sizeof(T) * cur_tensor_size, false);

    ffn2_out_buf_ = (T*)allocator_->reMalloc(ffn2_out_buf_, sizeof(T) * cur_tensor_size, false);

    ctc_lo_out_buf_ = (T*)allocator_->reMalloc(ctc_lo_out_buf_, sizeof(T) * batch_size * seq_len2 * vocab_size_, false);
    log_softmax_out_buf_ =
        (T*)allocator_->reMalloc(log_softmax_out_buf_, sizeof(T) * batch_size * seq_len2 * vocab_size_, false);
}

template<typename T>
void WenetEncoder<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    allocator_->free((void**)(&inter_conv1_input_buf_));
    allocator_->free((void**)(&inter_conv1_output_buf_));
    allocator_->free((void**)(&inter_conv2_output_buf_));
    allocator_->free((void**)(&inter_fc_input_buf_));
    allocator_->free((void**)(&conv_workspace_));

    allocator_->free((void**)(&input_hidden_state_));
    allocator_->free((void**)(&pos_emb_tensor_));
    allocator_->free((void**)(&attention_mask_));
    allocator_->free((void**)(&token_num_));
    allocator_->free((void**)(&padding_offset_));
    allocator_->free((void**)(&bid_start_end_));

    allocator_->free((void**)(&normed_from_tensor_));

    allocator_->free((void**)(&ffn_out_buf_));
    allocator_->free((void**)(&normed_ffn_out_buf_));

    allocator_->free((void**)(&attn_out_buf_));
    allocator_->free((void**)(&normed_attn_out_buf_));

    allocator_->free((void**)(&conv_out_buf_));
    allocator_->free((void**)(&normed_conv_out_buf_));
    allocator_->free((void**)(&ffn2_out_buf_));

    allocator_->free((void**)(&ctc_lo_out_buf_));
    allocator_->free((void**)(&log_softmax_out_buf_));
}

template<typename T>
void WenetEncoder<T>::forward(std::vector<Tensor>*         output_tensors,
                              const std::vector<Tensor>*   input_tensors,
                              const WenetEncoderWeight<T>* encoder_weights)
{
    TensorMap input_tensors_map =
        TensorMap({{"speech", input_tensors->at(0)}, {"sequence_length", input_tensors->at(1)}});

    TensorMap output_tensors_map = TensorMap({
        {"output_hidden_state", output_tensors->at(0)},
        {"encoder_out_lens", output_tensors->at(1)},
        {"ctc_log_probs", output_tensors->at(2)},
    });

    forward(&output_tensors_map, &input_tensors_map, encoder_weights);
}

template<typename T>
void WenetEncoder<T>::forward(TensorMap*                   output_tensors,
                              TensorMap*                   input_tensors,
                              const WenetEncoderWeight<T>* encoder_weights)
{
    // input_tensors:
    //      speech [batch, seq_len, feature_size]
    //      sequence_length [batch]
    // output tensors:
    //      output_hidden_state [batch, seq_len2, hidden_units]
    //      encoder_out_lens [batch]
    //      ctc_log_probs [batch, seq_len2, vocab_size]

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() == 2);
    const size_t batch_size = input_tensors->at("speech").shape[0];
    const size_t seq_len    = input_tensors->at("speech").shape[1];

    T*     speech_tensor_ptr = (T*)input_tensors->at("speech").data;
    Tensor seq_len_tensor    = input_tensors->at("sequence_length");

    FT_CHECK(batch_size == seq_len_tensor.shape[0]);
    FT_CHECK(input_tensors->at("speech").shape.size() == 3);
    FT_CHECK(seq_len_tensor.shape.size() == 1);

    const size_t in_channel    = 1;
    const size_t kernel_size   = 3;
    const size_t stride        = 2;
    const T      scale         = sqrt(d_model_);
    const size_t feature_size1 = (feature_size_ - kernel_size) / stride + 1;
    const size_t feature_size2 = (feature_size1 - kernel_size) / stride + 1;
    const size_t seq_len1      = (seq_len - kernel_size) / stride + 1;
    const size_t seq_len2      = (seq_len1 - kernel_size) / stride + 1;
    allocateBuffer(batch_size, seq_len, feature_size_, kernel_size, stride);

    Tensor     encoder_out_lens_tensor = output_tensors->at("encoder_out_lens");
    const int* sequence_lengths_in     = seq_len_tensor.getPtr<int>();
    int*       sequence_lengths        = encoder_out_lens_tensor.getPtr<int>();
    invokeGetWenetOutLens(
        sequence_lengths, sequence_lengths_in, batch_size, input_tensors->at("speech").shape[1], stream_);

    DataType data_type = getTensorType<T>();

    // Position Embed
    {
        invokeCMVN<T>(inter_conv1_input_buf_,
                      speech_tensor_ptr,
                      encoder_weights->cmvn_weights.mean,
                      encoder_weights->cmvn_weights.istd,
                      batch_size * seq_len,
                      feature_size_,
                      stream_);

        conv2d(inter_conv1_output_buf_,
               inter_conv1_input_buf_,
               conv_workspace_,
               0,
               encoder_weights->embed_conv1_weights.kernel,
               encoder_weights->embed_conv1_weights.bias,
               batch_size,
               seq_len,
               feature_size_,
               in_channel,
               d_model_,
               kernel_size,
               stride,
               cudnn_handle_,
               stream_);
        conv2d(inter_conv2_output_buf_,
               inter_conv1_output_buf_,
               conv_workspace_,
               1,
               encoder_weights->embed_conv2_weights.kernel,
               encoder_weights->embed_conv2_weights.bias,
               batch_size,
               seq_len1,
               feature_size1,
               d_model_,
               d_model_,
               kernel_size,
               stride,
               cudnn_handle_,
               stream_);
        invokeTranspose0213<T>(
            inter_fc_input_buf_, inter_conv2_output_buf_, batch_size, seq_len2, d_model_, feature_size2, stream_);
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              d_model_,
                              batch_size * seq_len2,
                              feature_size2 * d_model_,
                              encoder_weights->embed_out_weights.kernel,
                              d_model_,
                              inter_fc_input_buf_,
                              feature_size2 * d_model_,
                              input_hidden_state_,
                              d_model_);
        invokeAddBiasMul<T>(input_hidden_state_,
                            encoder_weights->embed_out_weights.bias,
                            scale,
                            batch_size * seq_len2,
                            d_model_,
                            stream_);
        invokeSlice<T>(pos_emb_tensor_,
                       encoder_weights->positional_encoding_weights.data,
                       batch_size,
                       seq_len2,
                       d_model_,
                       stream_);
    }

    size_t h_token_num       = batch_size * seq_len2;
    T*     output_ptr        = output_tensors->at("output_hidden_state").getPtr<T>();
    float* ctc_log_probs_ptr = output_tensors->at("ctc_log_probs").getPtr<float>();

    invokeBuildEncoderAttentionMask(attention_mask_, sequence_lengths, batch_size, seq_len2, stream_);
    sync_check_cuda_error();

    bool use_varlen   = false;
    *h_var_token_num_ = h_token_num;
    if (use_varlen) {
        FT_CHECK(false);
        //     invokeGetPaddingOffset(token_num_, padding_offset_, sequence_lengths, batch_size, seq_len, stream_);
        //     sync_check_cuda_error();
        //     check_cuda_error(cudaEventRecord(stream_finished_, stream_));

        //     check_cuda_error(cudaStreamWaitEvent(stream2_, stream_finished_, 0));
        //     check_cuda_error(
        //         cudaMemcpyAsync(h_var_token_num_, token_num_, sizeof(size_t), ::cudaMemcpyDeviceToHost, stream2_));
        //     sync_check_cuda_error();
        //     check_cuda_error(cudaEventRecord(stream2_finished_, stream2_));
        //     invokeGetBatchIDStartEnd(bid_start_end_, sequence_lengths, batch_size, seq_len, stream_);

        //     sync_check_cuda_error();
    }

    for (uint i = 0; i < num_layer_; i++) {
        const T* from_tensor = (const T*)(i == 0 ? input_hidden_state_ : output_ptr);
        T*       out_tensor  = output_ptr;

        invokeGeneralLayerNorm(normed_from_tensor_,
                               from_tensor,
                               encoder_weights->encoder_layer_weights[i]->norm_ff_macaron_weights.gamma,
                               encoder_weights->encoder_layer_weights[i]->norm_ff_macaron_weights.beta,
                               1e-6f,
                               h_token_num,
                               hidden_units_,
                               (float*)nullptr,
                               0,
                               stream_);
        sync_check_cuda_error();

        // feed_forward_macaron
        {
            std::vector<Tensor> ffn_input_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, normed_from_tensor_}};
            std::vector<Tensor> ffn_output_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, ffn_out_buf_}};
            ffn_layer_->forward(&ffn_output_tensors,
                                &ffn_input_tensors,
                                &encoder_weights->encoder_layer_weights[i]->feed_forward_macaron_weights);
        }

        invokeGeneralScaleAddBiasResidualPreLayerNorm(
            ffn_out_buf_,
            normed_ffn_out_buf_,
            from_tensor,
            encoder_weights->encoder_layer_weights[i]->attn_layernorm_weights.gamma,
            encoder_weights->encoder_layer_weights[i]->attn_layernorm_weights.beta,
            encoder_weights->encoder_layer_weights[i]->feed_forward_macaron_weights.output_weight.bias,
            h_token_num,
            hidden_units_,
            stream_,
            2,
            0.5f,
            1.0f);
        sync_check_cuda_error();

        // attn
        {
            TensorMap attn_input_tensors{
                {"normed_ffn_out",
                 Tensor{MEMORY_GPU,
                        data_type,
                        std::vector<size_t>{batch_size * seq_len2, hidden_units_},
                        normed_ffn_out_buf_}},
                {"attention_mask",
                 Tensor{
                     MEMORY_GPU, data_type, std::vector<size_t>{batch_size, 1, seq_len2, seq_len2}, attention_mask_}},
                // {"padding_offset", Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batch_size * seq_len2},
                // nullptr}}, // not supported yet
                {"pos_emb",
                 Tensor{MEMORY_GPU,
                        data_type,
                        std::vector<size_t>{batch_size * seq_len2, hidden_units_},
                        pos_emb_tensor_}}};
            TensorMap attn_output_tensors{
                {"attention_out",
                 Tensor{
                     MEMORY_GPU, data_type, std::vector<size_t>{batch_size * seq_len2, hidden_units_}, attn_out_buf_}}};
            attention_layer_->forward(&attn_output_tensors,
                                      &attn_input_tensors,
                                      &encoder_weights->encoder_layer_weights[i]->attention_weights);
        }

        invokeGeneralScaleAddBiasResidualPreLayerNorm(
            attn_out_buf_,
            normed_attn_out_buf_,
            ffn_out_buf_,
            encoder_weights->encoder_layer_weights[i]->norm_conv_weights.gamma,
            encoder_weights->encoder_layer_weights[i]->norm_conv_weights.beta,
            encoder_weights->encoder_layer_weights[i]->attention_weights.attention_output_weight.bias,
            h_token_num,
            hidden_units_,
            stream_,
            2,
            1.0f,
            1.0f);
        sync_check_cuda_error();

        // conv
        {
            // if (i == 0 && use_varlen) {
            //     check_cuda_error(cudaStreamWaitEvent(stream_, stream2_finished_, 0));
            // }
            TensorMap conv_input_tensors{
                {"input_tensor",
                 Tensor{MEMORY_GPU,
                        data_type,
                        std::vector<size_t>{batch_size, seq_len2, hidden_units_},
                        normed_attn_out_buf_}},
                {"attention_mask",
                 Tensor{
                     MEMORY_GPU, data_type, std::vector<size_t>{batch_size, 1, seq_len2, seq_len2}, attention_mask_}},
                {"padding_offset",
                 Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{*h_var_token_num_}, padding_offset_}},
                {"bid_start_end",
                 Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{(*h_var_token_num_) * 3}, bid_start_end_}}};

            TensorMap conv_output_tensors{
                {"output_tensor",
                 Tensor{
                     MEMORY_GPU, data_type, std::vector<size_t>{batch_size, seq_len2, hidden_units_}, conv_out_buf_}}};
            conformer_conv_layer_->forward(&conv_output_tensors,
                                           &conv_input_tensors,
                                           &encoder_weights->encoder_layer_weights[i]->conv_module_weights);
        }

        T* bias_nullptr = nullptr;
        invokeGeneralScaleAddBiasResidualPreLayerNorm(
            conv_out_buf_,
            normed_conv_out_buf_,
            attn_out_buf_,
            encoder_weights->encoder_layer_weights[i]->ffn_layernorm_weights.gamma,
            encoder_weights->encoder_layer_weights[i]->ffn_layernorm_weights.beta,
            bias_nullptr,
            h_token_num,
            hidden_units_,
            stream_,
            2,
            1.0f,
            1.0f);
        sync_check_cuda_error();

        // ffn
        {
            std::vector<Tensor> ffn_input_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, normed_conv_out_buf_}};
            std::vector<Tensor> ffn_output_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, ffn2_out_buf_}};
            ffn_layer_->forward(
                &ffn_output_tensors, &ffn_input_tensors, &encoder_weights->encoder_layer_weights[i]->ffn_weights);
        }

        invokeGeneralScaleAddBiasResidualPreLayerNorm(
            ffn2_out_buf_,
            out_tensor,
            conv_out_buf_,
            encoder_weights->encoder_layer_weights[i]->norm_final_weights.gamma,
            encoder_weights->encoder_layer_weights[i]->norm_final_weights.beta,
            encoder_weights->encoder_layer_weights[i]->ffn_weights.output_weight.bias,
            h_token_num,
            hidden_units_,
            stream_,
            2,
            0.5f,
            1.0f);
        sync_check_cuda_error();
    }
    invokeGeneralLayerNorm(output_ptr,
                           output_ptr,
                           encoder_weights->post_transformer_layernorm_weights.gamma,
                           encoder_weights->post_transformer_layernorm_weights.beta,
                           1e-6f,
                           h_token_num,
                           hidden_units_,
                           (float*)nullptr,
                           0,
                           stream_);
    sync_check_cuda_error();

    int n = vocab_size_;
    int m = h_token_num;
    int k = hidden_units_;

    cublas_wrapper_->Gemm(
        CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, encoder_weights->ctc_lo_weight.kernel, n, output_ptr, k, ctc_lo_out_buf_, n);
    sync_check_cuda_error();
    invokeBiasLogSoftmax<T>(ctc_log_probs_ptr,
                            ctc_lo_out_buf_,
                            encoder_weights->ctc_lo_weight.bias,
                            nullptr,  // sequence_lengths,
                            seq_len2,
                            batch_size,
                            vocab_size_,
                            vocab_size_,
                            true,
                            stream_);
    sync_check_cuda_error();

    // TODO(mengw): add TopK kernel
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template class WenetEncoder<float>;
template class WenetEncoder<half>;

}  // namespace fastertransformer
