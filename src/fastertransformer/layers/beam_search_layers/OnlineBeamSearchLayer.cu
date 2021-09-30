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

#include "src/fastertransformer/layers/beam_search_layers/OnlineBeamSearchLayer.h"

namespace fastertransformer {

static const int SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS = 128;
static const int MAX_K = 4;

template<typename T>
__global__ void update_kernel(bool* finished,
                              int* parent_ids,
                              int* sequence_length,
                              int* word_ids,
                              int* output_ids,
                              const int vocab_size,
                              const int end_id,
                              const int local_batch_size,
                              const int beam_width)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < local_batch_size * beam_width;
         index += blockDim.x * gridDim.x) {

        int batch_id = index / beam_width;
        sequence_length[index] = finished[index] ? sequence_length[index] : sequence_length[index] + 1;

        int beam_id = (word_ids[index] / vocab_size) % beam_width;
        int word_id = word_ids[index] % vocab_size;

        sequence_length[index] = sequence_length[batch_id * beam_width + beam_id];
        finished[index] = word_id == end_id ? 1 : 0;
        parent_ids[index] = beam_id;
        word_ids[index] = word_id;
        output_ids[index] = word_id;
    }
}

void invokeUpdate(bool* finished,
                  int* parent_ids,
                  int* sequence_length,
                  int* word_ids,
                  int* output_ids,
                  const int local_batch_size,
                  const int beam_width,
                  const int vocab_size_padded,
                  const int end_id,
                  cudaStream_t stream)
{
    dim3 grid((int)ceil(local_batch_size * beam_width * 1.0 / 256));
    dim3 block(256);

    update_kernel<float><<<grid, block, 0, stream>>>(finished,
                                                     parent_ids,
                                                     sequence_length,
                                                     word_ids,
                                                     output_ids,
                                                     vocab_size_padded,
                                                     end_id,
                                                     local_batch_size,
                                                     beam_width);
}

template<typename T>
void OnlineBeamSearchLayer<T>::invokeSoftMax(std::vector<Tensor>* output_tensors,
                                             const std::vector<Tensor>* input_tensors)
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

    const int batch_size = output_tensors->at(0).shape[1];
    const int step = *((int*)input_tensors->at(2).data);
    const int ite = *((int*)input_tensors->at(7).data);
    const int local_batch_size = input_tensors->at(0).shape[0];

    const int id_offset = step * batch_size * beam_width_ + local_batch_size * ite * beam_width_;
    invokeTopkSoftMax((const T*)input_tensors->at(0).data,
                      (const T*)(nullptr),
                      (const bool*)output_tensors->at(1).data,
                      (float*)output_tensors->at(2).data,
                      ((int*)output_tensors->at(0).data) + id_offset,
                      topk_softmax_workspace_,
                      topk_softmax_workspace_size_,
                      local_batch_size,
                      beam_width_,
                      vocab_size_padded_,
                      end_id_,
                      diversity_rate_,
                      stream_);
    sync_check_cuda_error();

    invokeUpdate((bool*)output_tensors->at(1).data,
                 ((int*)output_tensors->at(3).data) + id_offset,
                 (int*)output_tensors->at(4).data,
                 ((int*)output_tensors->at(0).data) + id_offset,
                 ((int*)output_tensors->at(0).data) + id_offset,
                 local_batch_size,
                 beam_width_,
                 vocab_size_padded_,
                 end_id_,
                 stream_);
    sync_check_cuda_error();
}

template<typename T>
void OnlineBeamSearchLayer<T>::allocateBuffer()
{
    topk_softmax_workspace_size_ =
        (size_t)(ceil(max_batch_size_ * beam_width_ * beam_width_ / 4.) * 4 * 2
                 + ceil(max_batch_size_ * beam_width_ * SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * MAX_K + 2) / 4.) * 4);

    if (is_allocate_buffer_ == false) {
        topk_softmax_workspace_ =
            reinterpret_cast<float*>(allocator_->malloc(sizeof(float) * topk_softmax_workspace_size_, false));
        is_allocate_buffer_ = true;
    }
}

template<typename T>
OnlineBeamSearchLayer<T>::OnlineBeamSearchLayer(size_t max_batch_size,
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
    BaseBeamSearchLayer<T>(max_batch_size,
                           head_num,
                           size_per_head,
                           beam_width,
                           vocab_size,
                           vocab_size_padded,
                           end_id,
                           diversity_rate,
                           temperature,
                           len_penalty,
                           repetition_penalty,
                           stream,
                           cublas_wrapper,
                           allocator,
                           is_free_buffer_after_forward)
{
}

template<typename T>
OnlineBeamSearchLayer<T>::OnlineBeamSearchLayer(OnlineBeamSearchLayer<T> const& beam_search_layer):
    BaseBeamSearchLayer<T>(beam_search_layer)
{
}

template<typename T>
OnlineBeamSearchLayer<T>::~OnlineBeamSearchLayer()
{
}

template class OnlineBeamSearchLayer<float>;
template class OnlineBeamSearchLayer<half>;

}  // namespace fastertransformer