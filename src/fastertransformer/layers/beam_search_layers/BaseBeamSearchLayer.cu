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

#include "src/fastertransformer/kernels/beam_search_penalty_kernels.h"
#include "src/fastertransformer/layers/beam_search_layers/BaseBeamSearchLayer.h"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

template<typename T>
__global__ void update_KV_batch_major_cache_kernel(const T* __restrict key_src_cache,
                                                   T* key_tgt_cache,
                                                   const T* __restrict value_src_cache,
                                                   T* value_tgt_cache,
                                                   const int* beam_ids,
                                                   const bool* finished,
                                                   const int beam_width,
                                                   const int size_per_head,
                                                   const size_t cache_size,
                                                   const int step,
                                                   const int max_seq_len,
                                                   const int ite)
{
    int layer_id = blockIdx.z;
    int head_id = blockIdx.y;
    int bb_id = blockIdx.x;
    int batch_id = bb_id / beam_width;
    int beam_id = bb_id % beam_width;

    if (finished[batch_id * beam_width + beam_id])
        return;

    const int hidden_dim = size_per_head * gridDim.y;

    int64_t src_offset =
        layer_id * cache_size
        + ((gridDim.x * ite + batch_id * beam_width + beam_ids[gridDim.x * ite + batch_id * beam_width + beam_id])
               * hidden_dim
           + head_id * size_per_head)
              * max_seq_len;
    int64_t tgt_offset =
        layer_id * cache_size
        + ((gridDim.x * ite + batch_id * beam_width + beam_id) * hidden_dim + head_id * size_per_head) * max_seq_len;

    // for better memory access always do 16 byte loads.
    // [B, H, Dh/x, L, x]  and [B, H, L, Dh/x, x] (i.e. [B, H, L, Dh])
    auto key_src_ptr = reinterpret_cast<const uint4*>(key_src_cache + src_offset);
    auto value_src_ptr = reinterpret_cast<const uint4*>(value_src_cache + src_offset);
    auto key_tgt_ptr = reinterpret_cast<uint4*>(key_tgt_cache + tgt_offset);
    auto value_tgt_ptr = reinterpret_cast<uint4*>(value_tgt_cache + tgt_offset);
    constexpr int x = (sizeof(T) == 4) ? 4 : 8;

// step starts from 1
#if 0
    constexpr int WARP_SIZE = 32;
    const int num_warps = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    for (int dhx = warp_id; dhx < size_per_head/x; dhx += num_warps)
    {
      for (int tid = lane_id; tid < step; tid += WARP_SIZE)
      {
        key_tgt_ptr[dhx * max_seq_len + tid] = key_src_ptr[dhx * max_seq_len + tid];
      }
    }
#else
    // seems to be a bit faster
    for (int tid = threadIdx.x; tid < max_seq_len * size_per_head / x; tid += blockDim.x) {
        // could consider fast int division here
        if (tid % max_seq_len < step) {
            key_tgt_ptr[tid] = key_src_ptr[tid];
        }
    }
#endif

    for (int tid = threadIdx.x; tid < step * size_per_head / x; tid += blockDim.x) {
        value_tgt_ptr[tid] = value_src_ptr[tid];
    }
}

template<typename T>
void update_KV_cache_kernelLauncher(T* key_cache_output,
                                    T* value_cache_output,
                                    const T* key_cache_input,
                                    const T* value_cache_input,
                                    const int* beam_ids,
                                    const bool* finished,
                                    const int max_batch_size,
                                    const int local_batch_size,
                                    const int beam_width,
                                    const int head_num,
                                    const int size_per_head,
                                    const int step,
                                    const int decoder_max_seq_len,
                                    const int cache_size,
                                    const int decoder_layers,
                                    const int ite,
                                    cudaStream_t stream)
{
    dim3 grid(local_batch_size * beam_width, head_num, decoder_layers);
    constexpr int block_sz = 128;

    update_KV_batch_major_cache_kernel<<<grid, block_sz, 0, stream>>>(key_cache_input,
                                                                      key_cache_output,
                                                                      value_cache_input,
                                                                      value_cache_output,
                                                                      beam_ids,
                                                                      finished,
                                                                      beam_width,
                                                                      size_per_head,
                                                                      (size_t)cache_size,
                                                                      step,
                                                                      decoder_max_seq_len,
                                                                      ite);
}

template void update_KV_cache_kernelLauncher(float* key_cache_output,
                                             float* value_cache_output,
                                             const float* key_cache_input,
                                             const float* value_cache_input,
                                             const int* beam_ids,
                                             const bool* finished,
                                             const int max_batch_size,
                                             const int local_batch_size,
                                             const int beam_width,
                                             const int head_num,
                                             const int size_per_head,
                                             const int step,
                                             const int decoder_max_seq_len,
                                             const int cache_size,
                                             const int decoder_layers,
                                             const int ite,
                                             cudaStream_t stream);

template void update_KV_cache_kernelLauncher(half* key_cache_output,
                                             half* value_cache_output,
                                             const half* key_cache_input,
                                             const half* value_cache_input,
                                             const int* beam_ids,
                                             const bool* finished,
                                             const int max_batch_size,
                                             const int local_batch_size,
                                             const int beam_width,
                                             const int head_num,
                                             const int size_per_head,
                                             const int step,
                                             const int decoder_max_seq_len,
                                             const int cache_size,
                                             const int decoder_layers,
                                             const int ite,
                                             cudaStream_t stream);
template<typename T>
BaseBeamSearchLayer<T>::BaseBeamSearchLayer(size_t max_batch_size,
                                            size_t head_num,
                                            size_t size_per_head,
                                            size_t beam_width,
                                            size_t vocab_size,
                                            size_t vocab_size_padded,
                                            int end_id,
                                            float diversity_rate,
                                            float temperature,
                                            float len_penalty,
                                            float repetition_penalty,
                                            cudaStream_t stream,
                                            cublasMMWrapper* cublas_wrapper,
                                            IAllocator* allocator,
                                            bool is_free_buffer_after_forward):
    DynamicDecodeBaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr),
    max_batch_size_(max_batch_size),
    head_num_(head_num),
    size_per_head_(size_per_head),
    beam_width_(beam_width),
    vocab_size_(vocab_size),
    vocab_size_padded_(vocab_size_padded),
    end_id_(end_id),
    diversity_rate_(diversity_rate),
    temperature_(temperature),
    len_penalty_(len_penalty),
    repetition_penalty_(repetition_penalty)
{
    hidden_units_ = head_num_ * size_per_head_;
}

template<typename T>
bool BaseBeamSearchLayer<T>::isValidBatchSize(size_t batch_size)
{
    if (batch_size <= max_batch_size_) {
        return true;
    }
    else {
        freeBuffer();
        max_batch_size_ = batch_size * 1.2;
        return true;
    }
}

template<typename T>
BaseBeamSearchLayer<T>::BaseBeamSearchLayer(BaseBeamSearchLayer<T> const& beam_search_layer):
    DynamicDecodeBaseLayer(beam_search_layer),
    max_batch_size_(beam_search_layer.max_batch_size_),
    head_num_(beam_search_layer.head_num_),
    size_per_head_(beam_search_layer.size_per_head_),
    beam_width_(beam_search_layer.beam_width_),
    vocab_size_(beam_search_layer.vocab_size_),
    vocab_size_padded_(beam_search_layer.vocab_size_padded_),
    end_id_(beam_search_layer.end_id_),
    diversity_rate_(beam_search_layer.diversity_rate_),
    hidden_units_(beam_search_layer.hidden_units_),
    topk_softmax_workspace_size_(beam_search_layer.topk_softmax_workspace_size_),
    temperature_(beam_search_layer.temperature_),
    len_penalty_(beam_search_layer.len_penalty_),
    repetition_penalty_(beam_search_layer.repetition_penalty_)
{
}

template<typename T>
BaseBeamSearchLayer<T>::~BaseBeamSearchLayer()
{
    freeBuffer();
}

template<typename T>
void BaseBeamSearchLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(topk_softmax_workspace_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void BaseBeamSearchLayer<T>::forward(std::vector<Tensor>* output_tensors, const std::vector<Tensor>* input_tensors)
{
    // input_tensors:
    //      logits [local_batch_size, beam_width_, vocab_size_padded]
    //      embedding_bias [vocab_size_padded]
    //      step [1] on cpu
    //      src_key_cache [num_layer, batch_size * beam_width, head_num, size_per_head // x, max_seq_len, x]
    //      src_value_cache [num_layer, batch_size * beam_width, head_num, max_seq_len, size_per_head]
    //      max_input_length [1] on cpu
    //      input_lengths [local_batch_size * beam_width_]
    //      ite [1] on cpu

    // output_tensors:
    //      output_ids [max_seq_len, batch_size, beam_width]
    //      finished [local_batch_size * beam_width]
    //      cum_logits [local_batch_size * beam_width]
    //      parent_ids [max_seq_len, batch_size * beam_width]
    //      sequence_length [local_batch_size * beam_width]
    //      tgt_key_cache [num_layer, batch_size * beam_width, head_num, size_per_head // x, max_seq_len, x]
    //      tgt_value_cache [num_layer, batch_size * beam_width, head_num, max_seq_len, size_per_head]

    FT_CHECK(input_tensors->size() == 8);
    FT_CHECK(output_tensors->size() == 7);
    isValidBatchSize(input_tensors->at(0).shape[0]);
    allocateBuffer();

    const int batch_size = output_tensors->at(0).shape[1];
    const int step = *((int*)input_tensors->at(2).data);
    const int ite = *((int*)input_tensors->at(7).data);
    const int local_batch_size = input_tensors->at(0).shape[0];

    if (input_tensors->at(1).data != nullptr || temperature_ != 1.0f || len_penalty_ != 1.0f
        || repetition_penalty_ != 1.0f) {
        invokeAddBiasApplyPenalties(step,
                                    (T*)input_tensors->at(0).data,
                                    (const int*)output_tensors->at(0).data + (step - 1) * batch_size * beam_width_
                                        + ite * local_batch_size * beam_width_,
                                    ((const int*)output_tensors->at(0).data),
                                    ((const int*)output_tensors->at(3).data),
                                    (const int*)input_tensors->at(6).data,
                                    (const T*)input_tensors->at(1).data,
                                    ite,
                                    *(int*)input_tensors->at(5).data,
                                    local_batch_size,
                                    batch_size,
                                    beam_width_,
                                    vocab_size_,
                                    vocab_size_padded_,
                                    end_id_,
                                    temperature_,
                                    len_penalty_,
                                    repetition_penalty_,
                                    stream_);
        sync_check_cuda_error();
    }

    invokeSoftMax(output_tensors, input_tensors);

    if (beam_width_ > 1) {
        const int max_seq_len = output_tensors->at(0).shape[0];
        const int decoder_layer = output_tensors->at(5).shape[0];
        const int decoder_max_seq_len = output_tensors->at(5).shape[4];
        FT_CHECK(max_seq_len == decoder_max_seq_len);
        int cache_size = 1;
        for (auto t = output_tensors->at(5).shape.begin() + 1; t != output_tensors->at(5).shape.end(); ++t) {
            cache_size *= *t;
        }

        if (output_tensors->at(5).type == DataType::TYPE_FP32) {
            update_KV_cache_kernelLauncher((float*)output_tensors->at(5).data,
                                           (float*)output_tensors->at(6).data,
                                           (const float*)input_tensors->at(3).data,
                                           (const float*)input_tensors->at(4).data,
                                           ((int*)output_tensors->at(3).data) + step * batch_size * beam_width_,
                                           (bool*)output_tensors->at(1).data,
                                           batch_size,
                                           local_batch_size,
                                           beam_width_,
                                           head_num_,
                                           size_per_head_,
                                           step,
                                           decoder_max_seq_len,
                                           cache_size,
                                           decoder_layer,
                                           ite,
                                           stream_);
        }
        else if (output_tensors->at(5).type == DataType::TYPE_FP16) {
            update_KV_cache_kernelLauncher((half*)output_tensors->at(5).data,
                                           (half*)output_tensors->at(6).data,
                                           (const half*)input_tensors->at(3).data,
                                           (const half*)input_tensors->at(4).data,
                                           ((int*)output_tensors->at(3).data) + step * batch_size * beam_width_,
                                           (bool*)output_tensors->at(1).data,
                                           batch_size,
                                           local_batch_size,
                                           beam_width_,
                                           head_num_,
                                           size_per_head_,
                                           step,
                                           decoder_max_seq_len,
                                           cache_size,
                                           decoder_layer,
                                           ite,
                                           stream_);
        }
    }
    sync_check_cuda_error();
    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template class BaseBeamSearchLayer<float>;
template class BaseBeamSearchLayer<half>;

}  // namespace fastertransformer
