/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "3rdparty/trt_fused_multihead_attention/qkvToContext.h"
#include "src/fastertransformer/kernels/layout_transformer_int8_kernels.h"
#include "src/fastertransformer/kernels/normalize_kernels.h"
#include "src/fastertransformer/kernels/reverse_roll_kernels.h"
#include "src/fastertransformer/kernels/softmax_int8_kernels.h"
#include "src/fastertransformer/kernels/transform_mask_kernels.h"
#include "src/fastertransformer/kernels/transpose_int8_kernels.h"
#include "src/fastertransformer/kernels/unfused_attention_int8_kernels.h"
#include "src/fastertransformer/layers/attention_layers/BaseAttentionLayer.h"
#include "src/fastertransformer/models/swin_int8/SwinINT8Weight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cublasINT8MMWrapper.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>

#define TRT_MAX_LEN 384

namespace fastertransformer {

template<typename T>
class WindowAttentionINT8: public BaseAttentionLayer<T> {
private:
    int   max_batch_   = 1;
    int   window_size_ = 7;
    bool  qkv_bias_    = true;
    float qk_scale_    = 1.0f;
    bool  use_trt_     = false;
    int   version_     = 1;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(int batch, int window_num, int window_len, int embed_dim, int num_head);

    int                        dispatcher_int8_num_head_ = -1;
    std::unique_ptr<MHARunner> dispatcher_int8_;

    using BaseAttentionLayer<T>::stream_;
    using BaseAttentionLayer<T>::is_free_buffer_after_forward_;
    using BaseAttentionLayer<T>::is_allocate_buffer_;
    using BaseAttentionLayer<T>::cublas_wrapper_;
    using BaseAttentionLayer<T>::allocator_;

    int8_t *buf_ = nullptr, *Q_buf_ = nullptr, *K_buf_ = nullptr, *V_buf_ = nullptr;
    int8_t *q_buf_ = nullptr, *k_buf_ = nullptr, *v_buf_ = nullptr, *qk_buf_ = nullptr, *dst_buf_ = nullptr;

    static int trt_getS(const int actual_seqlen);

public:
    WindowAttentionINT8(int              max_batch,
                        int              window_size,
                        cudaStream_t     stream,
                        cublasMMWrapper* cublas_wrapper,
                        IAllocator*      allocator,
                        bool             is_free_buffer_after_forward = false,
                        bool             qkv_bias                     = true,
                        float            qk_scale                     = 1.0f,
                        int              version                      = 1);

    ~WindowAttentionINT8();

    void
    forward(TensorMap* output_tensors, TensorMap* input_tensors, const AttentionWeight<T>* attention_weights) override;

};  // class WindowAttentionINT8
}  // namespace fastertransformer
