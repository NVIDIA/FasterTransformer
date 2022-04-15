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

#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/layers/beam_search_layers/BeamSearchLayer.h"

namespace fastertransformer {

template<typename T>
__global__ void logProbAddCumLogProb(float* log_probs,
                                     const T* logits,
                                     const float* cum_log_probs,
                                     const int* end_ids,
                                     const bool* finished,
                                     const int beam_width,
                                     const int n)
{
    int bid = blockIdx.x;
    bool finish = finished[bid];
    int offset = bid * n;

    float max_val = -1 * FLT_MAX;
    __shared__ float s_max_val;
    __shared__ float s_sum_val;

    if (finish) {
        for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
            log_probs[offset + tid] = (tid == end_ids[bid / beam_width]) ? cum_log_probs[bid] : -FLT_MAX;
        }
    }
    else {
        for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
            log_probs[offset + tid] = (float)(logits[offset + tid]);
            max_val = max(max_val, log_probs[offset + tid]);
        }

        max_val = blockReduceMax(max_val);
        if (threadIdx.x == 0) {
            s_max_val = max_val;
        }
        __syncthreads();

        float sum_val = 0.0f;
        for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
            log_probs[offset + tid] = __expf(log_probs[offset + tid] - s_max_val);
            sum_val += log_probs[offset + tid];
        }

        sum_val = blockReduceSum(sum_val);
        if (threadIdx.x == 0) {
            s_sum_val = sum_val + 1e-6f;
        }
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
                                const int* end_ids,
                                const bool* finished,
                                const int m,
                                const int beam_width,
                                const int n,
                                cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(min(n, 1024));
    /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
    logProbAddCumLogProb<<<grid, block, 0, stream>>>(
        log_probs, logits, cum_log_probs, end_ids, finished, beam_width, n);
}

template<typename T>
__global__ void updateStatesKernel(T* log_probs,
                                   T* cum_log_probs,
                                   float* output_log_probs,
                                   bool* finished,
                                   int* parent_ids,
                                   int* sequence_length,
                                   int* word_ids,
                                   int* output_ids,
                                   const int local_batch_size,
                                   const int beam_width,
                                   const int vocab_size,
                                   const int* end_ids)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < local_batch_size * beam_width;
         index += blockDim.x * gridDim.x) {

        int batch_id = index / beam_width;
        sequence_length[index] = finished[index] ? sequence_length[index] : sequence_length[index] + 1;

        int beam_id = (word_ids[index] / vocab_size) % beam_width;
        int word_id = word_ids[index] % vocab_size;

        if (output_log_probs != nullptr) {
            // get the cum_log_probs of previous run
            output_log_probs[index] = log_probs[batch_id * beam_width * vocab_size + beam_id * vocab_size + word_id]
                                      - cum_log_probs[batch_id * beam_width + beam_id];
        }
        cum_log_probs[index] = log_probs[batch_id * beam_width * vocab_size + beam_id * vocab_size + word_id];
        sequence_length[index] = sequence_length[batch_id * beam_width + beam_id];
        finished[index] = word_id == end_ids[batch_id] ? 1 : 0;
        parent_ids[index] = beam_id;
        word_ids[index] = word_id;
        output_ids[index] = word_id;
    }
}

void invokeUpdateStates(float* log_probs,
                        float* cum_log_probs,
                        float* output_log_probs,
                        bool* finished,
                        int* parent_ids,
                        int* sequence_length,
                        int* word_ids,
                        int* output_ids,
                        const int local_batch_size,
                        const int beam_width,
                        const int vocab_size,
                        const int* end_ids,
                        cudaStream_t stream)
{
    dim3 grid((int)ceil(local_batch_size * beam_width * 1.0 / 256));
    dim3 block(256);

    updateStatesKernel<float><<<grid, block, 0, stream>>>(log_probs,
                                                          cum_log_probs,
                                                          output_log_probs,
                                                          finished,
                                                          parent_ids,
                                                          sequence_length,
                                                          word_ids,
                                                          output_ids,
                                                          local_batch_size,
                                                          beam_width,
                                                          vocab_size,
                                                          end_ids);
}

template<typename T>
void BeamSearchLayer<T>::invokeSoftMax(std::vector<Tensor>* output_tensors, const std::vector<Tensor>* input_tensors)
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
void BeamSearchLayer<T>::invokeSoftMax(std::unordered_map<std::string, Tensor>* output_tensors,
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
    //      output_log_probs [local_batch_size * beam_width], optional

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

    const int id_offset = step * batch_size * beam_width + ite * local_batch_size * beam_width;
    invokeLogProbAddCumLogProb(float_log_prob_buf_,
                               (T*)input_tensors->at("logits").data,
                               (float*)output_tensors->at("cum_log_probs").data,
                               (const int*)input_tensors->at("end_id").data,
                               (bool*)output_tensors->at("finished").data,
                               local_batch_size * beam_width,
                               beam_width,
                               vocab_size_padded_,
                               stream_);
    sync_check_cuda_error();

    invokeTopkBeamSearch<float>(topk_softmax_workspace_,
                                topk_softmax_workspace_size_,
                                float_log_prob_buf_,
                                ((int*)output_tensors->at("output_ids").data) + id_offset,
                                (bool*)output_tensors->at("finished").data,
                                local_batch_size,
                                beam_width,
                                vocab_size_padded_,
                                diversity_rate,
                                (const int*)input_tensors->at("end_id").data,
                                stream_);
    sync_check_cuda_error();

    float* output_log_probs =
        output_tensors->count("output_log_probs") ? (float*)output_tensors->at("output_log_probs").data : nullptr;
    invokeUpdateStates(float_log_prob_buf_,
                       (float*)output_tensors->at("cum_log_probs").data,
                       output_log_probs,
                       (bool*)output_tensors->at("finished").data,
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
void BeamSearchLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void BeamSearchLayer<T>::allocateBuffer(size_t batch_size, size_t beam_width)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    invokeTopkBeamSearch<float>(nullptr,
                                topk_softmax_workspace_size_,
                                nullptr,
                                nullptr,
                                nullptr,
                                batch_size,
                                beam_width,
                                vocab_size_padded_,
                                0.0f,
                                nullptr,
                                stream_);
    topk_softmax_workspace_ = reinterpret_cast<float*>(allocator_->reMalloc(
        topk_softmax_workspace_,
        topk_softmax_workspace_size_ + sizeof(float) * batch_size * beam_width * vocab_size_padded_,
        false));
    float_log_prob_buf_ = (float*)((char*)topk_softmax_workspace_ + topk_softmax_workspace_size_);
    is_allocate_buffer_ = true;
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