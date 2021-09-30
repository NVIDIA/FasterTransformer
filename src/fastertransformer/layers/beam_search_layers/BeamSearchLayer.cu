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

#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/layers/beam_search_layers/BeamSearchLayer.h"

namespace fastertransformer {

template<typename T>
__global__ void logProbAddCumLogProb(
    float* log_probs, const T* logits, const float* cum_log_probs, const int end_id, const bool* finished, const int n)
{
    int bid = blockIdx.x;
    bool finish = finished[bid];
    int offset = bid * n;

    float max_val = -1 * FLT_MAX;
    __shared__ float s_max_val;
    __shared__ float s_sum_val;

    if (finish) {
        for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
            log_probs[offset + tid] = (tid == end_id) ? cum_log_probs[bid] : -FLT_MAX;
        }
    }
    else {
        for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
            log_probs[offset + tid] = (float)(logits[offset + tid]);
            max_val = max(max_val, log_probs[offset + tid]);
        }

        max_val = blockReduceMax(max_val);
        if (threadIdx.x == 0)
            s_max_val = max_val;
        __syncthreads();

        float sum_val = 0.0f;
        for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
            log_probs[offset + tid] = __expf(log_probs[offset + tid] - s_max_val);
            sum_val += log_probs[offset + tid];
        }

        sum_val = blockReduceSum(sum_val);
        if (threadIdx.x == 0)
            s_sum_val = sum_val + 1e-6f;
        __syncthreads();

        for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
            log_probs[offset + tid] = logf(log_probs[offset + tid] / s_sum_val) + cum_log_probs[bid];
        }
    }
}

template<typename T>
void invokeLogProbAddCumLogProb(float* log_probs,
                                const T* logits,
                                const float* cum_log_probs,
                                const int end_id,
                                const bool* finished,
                                const int m,
                                const int n,
                                cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(min(n, 1024));
    /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
    logProbAddCumLogProb<<<grid, block, 0, stream>>>(log_probs, logits, cum_log_probs, end_id, finished, n);
}

template<typename T>
__global__ void updateStatesKernel(T* log_probs,
                                   T* cum_log_probs,
                                   bool* finished,
                                   int* parent_ids,
                                   int* sequence_length,
                                   int* word_ids,
                                   int* output_ids,
                                   const int local_batch_size,
                                   const int beam_width,
                                   const int vocab_size,
                                   const int end_id)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < local_batch_size * beam_width;
         index += blockDim.x * gridDim.x) {

        int batch_id = index / beam_width;
        sequence_length[index] = finished[index] ? sequence_length[index] : sequence_length[index] + 1;

        int beam_id = (word_ids[index] / vocab_size) % beam_width;
        int word_id = word_ids[index] % vocab_size;

        cum_log_probs[index] = log_probs[batch_id * beam_width * vocab_size + beam_id * vocab_size + word_id];
        sequence_length[index] = sequence_length[batch_id * beam_width + beam_id];
        finished[index] = word_id == end_id ? 1 : 0;
        parent_ids[index] = beam_id;
        word_ids[index] = word_id;
        output_ids[index] = word_id;
    }
}

void invokeUpdateStates(float* log_probs,
                        float* cum_log_probs,
                        bool* finished,
                        int* parent_ids,
                        int* sequence_length,
                        int* word_ids,
                        int* output_ids,
                        const int local_batch_size,
                        const int beam_width,
                        const int vocab_size,
                        const int end_id,
                        cudaStream_t stream)
{
    dim3 grid((int)ceil(local_batch_size * beam_width * 1.0 / 256));
    dim3 block(256);

    updateStatesKernel<float><<<grid, block, 0, stream>>>(log_probs,
                                                          cum_log_probs,
                                                          finished,
                                                          parent_ids,
                                                          sequence_length,
                                                          word_ids,
                                                          output_ids,
                                                          local_batch_size,
                                                          beam_width,
                                                          vocab_size,
                                                          end_id);
}

template<typename T>
void BeamSearchLayer<T>::invokeSoftMax(std::vector<Tensor>* output_tensors, const std::vector<Tensor>* input_tensors)
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

    const int id_offset = step * batch_size * beam_width_ + ite * local_batch_size * beam_width_;
    invokeLogProbAddCumLogProb(float_log_prob_buf_,
                               (T*)input_tensors->at(0).data,
                               (float*)output_tensors->at(2).data,
                               end_id_,
                               (bool*)output_tensors->at(1).data,
                               local_batch_size * beam_width_,
                               vocab_size_padded_,
                               stream_);
    sync_check_cuda_error();

    invokeTopkBeamSearch<float>(topk_softmax_workspace_,
                                topk_softmax_workspace_size_,
                                float_log_prob_buf_,
                                ((int*)output_tensors->at(0).data) + id_offset,
                                (bool*)output_tensors->at(1).data,
                                local_batch_size,
                                beam_width_,
                                vocab_size_padded_,
                                diversity_rate_,
                                end_id_,
                                stream_);
    sync_check_cuda_error();

    invokeUpdateStates(float_log_prob_buf_,
                       (float*)output_tensors->at(2).data,
                       (bool*)output_tensors->at(1).data,
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
void BeamSearchLayer<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        invokeTopkBeamSearch<float>(nullptr,
                                    topk_softmax_workspace_size_,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    max_batch_size_,
                                    beam_width_,
                                    vocab_size_padded_,
                                    0.0f,
                                    end_id_,
                                    stream_);
        topk_softmax_workspace_ = reinterpret_cast<float*>(allocator_->malloc(
            topk_softmax_workspace_size_ + sizeof(float) * max_batch_size_ * beam_width_ * vocab_size_padded_, false));
        float_log_prob_buf_ = (float*)((char*)topk_softmax_workspace_ + topk_softmax_workspace_size_);
        is_allocate_buffer_ = true;
    }
}

template<typename T>
BeamSearchLayer<T>::BeamSearchLayer(size_t max_batch_size,
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
BeamSearchLayer<T>::BeamSearchLayer(BeamSearchLayer<T> const& beam_search_layer):
    BaseBeamSearchLayer<T>(beam_search_layer)
{
}

template<typename T>
BeamSearchLayer<T>::~BeamSearchLayer()
{
}

template class BeamSearchLayer<float>;
template class BeamSearchLayer<half>;

}  // namespace fastertransformer