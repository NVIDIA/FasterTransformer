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
                              const int* end_ids,
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
        finished[index] = word_id == end_ids[index / beam_width] ? 1 : 0;
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
                  const int* end_ids,
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
                                                     end_ids,
                                                     local_batch_size,
                                                     beam_width);
}

template<typename T>
void OnlineBeamSearchLayer<T>::invokeSoftMax(std::vector<Tensor>* output_tensors,
                                             const std::vector<Tensor>* input_tensors)
{
    // input_tensors:
    //      logits [local_batch_size, beam_width, vocab_size_padded]
    //      embedding_bias [vocab_size_padded]
    //      step [1] on cpu
    //      src_cache_indirection [local_batch_size, beam_width, max_seq_len]
    //      max_input_length [1] on cpu
    //      input_lengths [local_batch_size * beam_width]
    //      ite [1] on cpu

    // output_tensors:
    //      output_ids [max_seq_len, batch_size, beam_width]
    //      finished [local_batch_size * beam_width]
    //      cum_log_probs [local_batch_size * beam_width]
    //      parent_ids [max_seq_len, batch_size * beam_width]
    //      sequence_length [local_batch_size * beam_width]
    //      tgt_cache_indirection [local_batch_size, beam_width, max_seq_len]

    std::unordered_map<std::string, Tensor> input_tensors_map{{"logits", input_tensors->at(0)},
                                                              {"embedding_bias", input_tensors->at(1)},
                                                              {"step", input_tensors->at(2)},
                                                              {"src_cache_indirection", input_tensors->at(3)},
                                                              {"max_input_length", input_tensors->at(4)},
                                                              {"input_lengths", input_tensors->at(5)},
                                                              {"ite", input_tensors->at(6)}};

    std::unordered_map<std::string, Tensor> output_tensors_map{{"output_ids", output_tensors->at(0)},
                                                               {"finished", output_tensors->at(1)},
                                                               {"cum_log_probs", output_tensors->at(2)},
                                                               {"parent_ids", output_tensors->at(3)},
                                                               {"sequence_length", output_tensors->at(4)},
                                                               {"tgt_cache_indirection", output_tensors->at(5)}};
    invokeSoftMax(&output_tensors_map, &input_tensors_map);
}

template<typename T>
void OnlineBeamSearchLayer<T>::invokeSoftMax(std::unordered_map<std::string, Tensor>* output_tensors,
                                             const std::unordered_map<std::string, Tensor>* input_tensors)
{
    // input_tensors:
    //      logits [local_batch_size, beam_width, vocab_size_padded]
    //      embedding_bias [vocab_size_padded]
    //      step [1] on cpu
    //      src_cache_indirection [local_batch_size, beam_width, max_seq_len]
    //      max_input_length [1] on cpu
    //      input_lengths [local_batch_size * beam_width]
    //      ite [1] on cpu
    //      beam_search_diversity_rate [1] on cpu, optional
    //      temperature [1] on cpu, optional
    //      len_penalty [1] on cpu, optional
    //      repetition_penalty [1] on cpu, optional

    // output_tensors:
    //      output_ids [max_seq_len, batch_size, beam_width]
    //      finished [local_batch_size * beam_width]
    //      cum_log_probs [local_batch_size * beam_width]
    //      parent_ids [max_seq_len, batch_size * beam_width]
    //      sequence_length [local_batch_size * beam_width]
    //      tgt_cache_indirection [local_batch_size, beam_width, max_seq_len]
    //      output_log_probs [local_batch_size * beam_width]

    FT_CHECK(input_tensors->size() >= 7);
    FT_CHECK(output_tensors->size() >= 6);

    const int batch_size = output_tensors->at("output_ids").shape[1];
    const int beam_width = output_tensors->at("output_ids").shape[2];
    const int step = *((int*)input_tensors->at("step").data);
    const int ite = *((int*)input_tensors->at("ite").data);
    const int local_batch_size = input_tensors->at("logits").shape[0];
    const float diversity_rate = input_tensors->count("beam_search_diversity_rate") ?
                                     input_tensors->at("beam_search_diversity_rate").getVal<float>() :
                                     0.0f;
    float* output_log_probs =
        output_tensors->count("output_log_probs") ? (float*)output_tensors->at("output_log_probs").data : nullptr;
    const int id_offset = step * batch_size * beam_width + local_batch_size * ite * beam_width;
    invokeTopkSoftMax((const T*)input_tensors->at("logits").data,
                      (const T*)(nullptr),
                      (const bool*)output_tensors->at("finished").data,
                      (float*)output_tensors->at("cum_log_probs").data,
                      output_log_probs,
                      ((int*)output_tensors->at("output_ids").data) + id_offset,
                      topk_softmax_workspace_,
                      topk_softmax_workspace_size_,
                      local_batch_size,
                      beam_width,
                      vocab_size_padded_,
                      (const int*)input_tensors->at("end_id").data,
                      diversity_rate,
                      stream_);
    sync_check_cuda_error();

    invokeUpdate((bool*)output_tensors->at("finished").data,
                 ((int*)output_tensors->at("parent_ids").data) + id_offset,
                 (int*)output_tensors->at("sequence_length").data,
                 ((int*)output_tensors->at("output_ids").data) + id_offset,
                 ((int*)output_tensors->at("output_ids").data) + id_offset,
                 local_batch_size,
                 beam_width,
                 vocab_size_padded_,
                 (const int*)input_tensors->at("end_id").data,
                 stream_);
    sync_check_cuda_error();
}

template<typename T>
void OnlineBeamSearchLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void OnlineBeamSearchLayer<T>::allocateBuffer(size_t batch_size, size_t beam_width)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    topk_softmax_workspace_size_ =
        (size_t)(ceil(batch_size * beam_width * beam_width / 4.) * 4 * 2
                 + ceil(batch_size * beam_width * SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * MAX_K + 2) / 4.) * 4);

    topk_softmax_workspace_ = reinterpret_cast<float*>(
        allocator_->reMalloc(topk_softmax_workspace_, sizeof(float) * topk_softmax_workspace_size_, false));
    is_allocate_buffer_ = true;
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