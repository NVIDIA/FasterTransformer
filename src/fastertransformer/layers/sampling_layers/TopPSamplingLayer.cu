/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include <algorithm>
#include <float.h>

#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/kernels/sampling_topk_kernels.h"
#include "src/fastertransformer/kernels/sampling_topp_kernels.h"
#include "src/fastertransformer/layers/sampling_layers/TopPSamplingLayer.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

static __global__ void set_topp_runtime_args(int    batch_size,
                                             uint   top_k,
                                             uint*  top_ks,
                                             int    top_ks_size,
                                             float  top_p,
                                             float* top_ps,
                                             int    top_ps_size,
                                             bool*  skip_decode)
{
    int index = blockIdx.x * gridDim.x + threadIdx.x;
    for (int i = index; i < batch_size; i += gridDim.x * blockDim.x) {
        uint  k = top_ks_size > 1 ? top_ks[i] : top_k;
        float p = top_ps_size > 1 ? top_ps[i] : top_p;
        if (k == 0 && p == 0.0f) {
            // Invalid runtime topk/topp combination. Use a greedy
            // decoding as default and topk sampling will handle.
            printf("[WARNING] Invalid runtime topk/topp combination for token %d (topk: %d, topp: %f). "
                   "Use a greedy decoding as default.\n",
                   i,
                   k,
                   p);
            k = 1;
        }
        top_ks[i] = k;
        // Clip p value if it is out of range. range = [0.0, 1.0].
        top_ps[i] = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
        if (p < 0.0f || p > 1.0f) {
            printf("[WARNING] topp (%f) is out of range ([0.0, 1.0f]) for token %d"
                   " clip to closest number %f.\n",
                   p,
                   i,
                   top_ps[i]);
        }
        skip_decode[i] = k > 0;
    }
}

template<typename T>
void TopPSamplingLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void TopPSamplingLayer<T>::allocateBuffer(size_t batch_size, Tensor top_k, Tensor top_p)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    BaseSamplingLayer<T>::allocateBuffer(batch_size, top_k, top_p);
    invokeTopPSampling<T>(nullptr,  // workspace
                          sampling_workspace_size_,
                          cub_temp_storage_size_,
                          nullptr,  // output_ids
                          nullptr,  // sequence_length
                          nullptr,  // finished_buffer
                          nullptr,  // cum_log_probs
                          nullptr,  // output_log_probs
                          nullptr,  // log_probs
                          topp_id_vals_buf_,
                          topp_offset_buf_,
                          begin_topp_offset_buf_,
                          curandstate_buf_,
                          batch_size,
                          vocab_size_padded_,
                          nullptr,
                          top_p.size() > 0 ? top_p.max<float>() : 0.0f,
                          stream_,
                          cuda_device_prop_,
                          skip_decode_buf_);
    sampling_workspace_ = allocator_->reMalloc(sampling_workspace_, sampling_workspace_size_, true);
    runtime_top_k_buf_ =
        reinterpret_cast<uint*>(allocator_->reMalloc(runtime_top_k_buf_, sizeof(uint) * batch_size, false));
    runtime_top_p_buf_ =
        reinterpret_cast<float*>(allocator_->reMalloc(runtime_top_p_buf_, sizeof(float) * batch_size, false));
    topp_id_vals_buf_ = reinterpret_cast<int*>(
        allocator_->reMalloc(topp_id_vals_buf_, sizeof(int) * batch_size * vocab_size_padded_, false));
    topp_offset_buf_ =
        reinterpret_cast<int*>(allocator_->reMalloc(topp_offset_buf_, sizeof(int) * (batch_size + 1), false));
    begin_topp_offset_buf_ =
        reinterpret_cast<int*>(allocator_->reMalloc(begin_topp_offset_buf_, sizeof(int) * (batch_size + 1), false));
    is_allocate_buffer_ = true;
}

template<typename T>
void TopPSamplingLayer<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&sampling_workspace_));
        allocator_->free((void**)(&topp_id_vals_buf_));
        allocator_->free((void**)(&topp_offset_buf_));
        allocator_->free((void**)(&begin_topp_offset_buf_));
        allocator_->free((void**)(&runtime_top_k_buf_));
        allocator_->free((void**)(&runtime_top_p_buf_));
    }
    BaseSamplingLayer<T>::freeBuffer();
    is_allocate_buffer_ = false;
}

template<typename T>
void TopPSamplingLayer<T>::setup(const size_t                                   batch_size,
                                 const size_t                                   beam_width,
                                 const std::unordered_map<std::string, Tensor>* runtime_args)
{
    // Set up the sampling layer for given runtime arguments.
    //
    // runtime_args:
    //     runtime_top_k [1] or [batch_size] on cpu, optional.
    //     runtime_top_p [1] or [batch_size] on cpu, optional
    //     temperature [1] or [batch_size] on cpu, optional
    //     repetition_penalty [1] or [batch_size] on cpu, optional

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    BaseSamplingLayer<T>::setup(batch_size, beam_width, runtime_args);
    const Tensor runtime_top_p = runtime_args->count("runtime_top_p") ? runtime_args->at("runtime_top_p") : Tensor();
    const size_t runtime_top_p_size = runtime_top_p.size();
    if (runtime_top_p_size == 0) {
        std::fill_n(skip_decode_, batch_size, true);
        return;
    }

    uint         tmp_top_k          = 0;
    const Tensor runtime_top_k      = runtime_args->count("runtime_top_k") ?
                                          runtime_args->at("runtime_top_k") :
                                          Tensor(MEMORY_CPU, TYPE_UINT32, {1}, &tmp_top_k);
    const size_t runtime_top_k_size = runtime_top_k.size();

    uint  top_k = runtime_top_k.getVal<uint>();
    float top_p = runtime_top_p.getVal<float>();

    if (runtime_top_k_size > 1) {
        FT_CHECK(runtime_top_k.size() == batch_size);
        cudaH2Dcpy(runtime_top_k_buf_, runtime_top_k.getPtr<uint>(), batch_size);
    }
    if (runtime_top_p_size > 1) {
        FT_CHECK(runtime_top_p.size() == batch_size);
        cudaH2Dcpy(runtime_top_p_buf_, runtime_top_p.getPtr<float>(), batch_size);
    }

    dim3 block(std::min((int)batch_size, 1024));
    dim3 grid(div_up((int)batch_size, (int)block.x));
    set_topp_runtime_args<<<grid, block, 0, stream_>>>(batch_size,
                                                       top_k,
                                                       runtime_top_k_buf_,
                                                       runtime_top_k_size,
                                                       top_p,
                                                       runtime_top_p_buf_,
                                                       runtime_top_p_size,
                                                       skip_decode_buf_);
    cudaAutoCpy(skip_decode_, skip_decode_buf_, batch_size, stream_);
    float* runtime_top_ps = new float[batch_size];
    cudaAutoCpy(runtime_top_ps, runtime_top_p_buf_, batch_size, stream_);
    runtime_max_top_p_ = *std::max_element(runtime_top_ps, runtime_top_ps + batch_size);
    delete[] runtime_top_ps;
}

template<typename T>
void TopPSamplingLayer<T>::runSampling(std::vector<fastertransformer::Tensor>*       output_tensors,
                                       const std::vector<fastertransformer::Tensor>* input_tensors)
{
    // input_tensors:
    //      logits [local_batch_size, vocab_size_padded]
    //      embedding_bias [vocab_size_padded]
    //      step [1] on cpu
    //      max_input_length [1] on cpu
    //      input_lengths [local_batch_size]
    //      ite [1] on cpu
    //      random_seed [1] on cpu

    // output_tensors:
    //      output_ids [max_seq_len, batch_size]
    //      finished [local_batch_size]
    //      sequence_length [local_batch_size]
    //      cum_log_probs [batch_size], must be float*
    //          The cumultative log probability of generated tokens.
    //      output_log_probs [local_batch_size], must be float*
    //          The log probs at the current step.

    FT_CHECK(false);  // TODO deprecated, need to remove
    std::unordered_map<std::string, Tensor> input_tensors_map{{"logits", input_tensors->at(0)},
                                                              {"embedding_bias", input_tensors->at(1)},
                                                              {"step", input_tensors->at(2)},
                                                              {"max_input_length", input_tensors->at(3)},
                                                              {"input_lengths", input_tensors->at(4)},
                                                              {"ite", input_tensors->at(5)}};
    if (input_tensors->size() == 7) {
        input_tensors_map.insert({"random_seed", input_tensors->at(6)});
    }

    std::unordered_map<std::string, Tensor> output_tensors_map{{"output_ids", output_tensors->at(0)},
                                                               {"finished", output_tensors->at(1)},
                                                               {"sequence_length", output_tensors->at(2)},
                                                               {"cum_log_probs", output_tensors->at(3)},
                                                               {"output_log_probs", output_tensors->at(4)}};
    runSampling(&output_tensors_map, &input_tensors_map);
}

template<typename T>
void TopPSamplingLayer<T>::runSampling(std::unordered_map<std::string, Tensor>*       output_tensors,
                                       const std::unordered_map<std::string, Tensor>* input_tensors)
{
    // input_tensors:
    //      logits [local_batch_size, vocab_size_padded]
    //      embedding_bias [vocab_size_padded]
    //      step [1] on cpu
    //      max_input_length [1] on cpu
    //      input_lengths [local_batch_size]
    //      ite [1] on cpu

    // output_tensors:
    //      output_ids [max_seq_len, batch_size]
    //      finished [local_batch_size]
    //      sequence_length [local_batch_size]
    //      cum_log_probs [batch_size], must be float*, optional
    //          The cumultative log probability of generated tokens.
    //      output_log_probs [local_batch_size], must be float*, optional
    //          The log probs at the current step.

    FT_CHECK(input_tensors->size() >= 6);
    FT_CHECK(output_tensors->size() >= 3);

    const int batch_size       = output_tensors->at("output_ids").shape[1];
    const int local_batch_size = input_tensors->at("logits").shape[0];
    const int step             = *((int*)input_tensors->at("step").data);
    const int ite              = *((int*)input_tensors->at("ite").data);

    // in case of skip any, the logit value is already copied and processed.
    T* logits = !skip_any_ ? input_tensors->at("logits").getPtr<T>() : runtime_logits_buf_;

    invokeTopPInitialize(
        topp_id_vals_buf_, topp_offset_buf_, begin_topp_offset_buf_, local_batch_size, vocab_size_padded_, stream_);
    sync_check_cuda_error();

    invokeAddBiasSoftMax(logits,
                         (T*)(nullptr),
                         input_tensors->at("end_id").getPtr<int>(),
                         output_tensors->at("finished").getPtr<bool>(),
                         local_batch_size,
                         vocab_size_padded_,
                         vocab_size_,
                         stream_);
    sync_check_cuda_error();

    float* cum_log_probs =
        output_tensors->count("cum_log_probs") ? output_tensors->at("cum_log_probs").getPtr<float>() : nullptr;
    float* output_log_probs =
        output_tensors->count("output_log_probs") ? output_tensors->at("output_log_probs").getPtr<float>() : nullptr;

    invokeBatchTopPSampling<T>(
        sampling_workspace_,
        sampling_workspace_size_,
        cub_temp_storage_size_,
        output_tensors->at("output_ids").getPtrWithOffset<int>(step * batch_size + ite * local_batch_size),
        output_tensors->at("sequence_length").getPtr<int>(),
        output_tensors->at("finished").getPtr<bool>(),
        cum_log_probs,
        output_log_probs,
        logits,
        topp_id_vals_buf_,
        topp_offset_buf_,
        begin_topp_offset_buf_,
        curandstate_buf_ + ite * local_batch_size,
        local_batch_size,
        vocab_size_padded_,
        input_tensors->at("end_id").getPtr<int>(),
        runtime_max_top_p_,
        runtime_top_p_buf_ + ite * local_batch_size,
        stream_,
        cuda_device_prop_,
        skip_decode_buf_ + ite * local_batch_size);
    sync_check_cuda_error();
}

template<typename T>
TopPSamplingLayer<T>::TopPSamplingLayer(size_t             max_batch_size,
                                        size_t             vocab_size,
                                        size_t             vocab_size_padded,
                                        int                end_id,
                                        float              top_p,
                                        unsigned long long random_seed,
                                        float              temperature,
                                        float              len_penalty,
                                        float              repetition_penalty,
                                        cudaStream_t       stream,
                                        cublasMMWrapper*   cublas_wrapper,
                                        IAllocator*        allocator,
                                        bool               is_free_buffer_after_forward,
                                        cudaDeviceProp*    cuda_device_prop):
    BaseSamplingLayer<T>(max_batch_size,
                         vocab_size,
                         vocab_size_padded,
                         end_id,
                         0,
                         top_p,
                         random_seed,
                         temperature,
                         len_penalty,
                         repetition_penalty,
                         stream,
                         cublas_wrapper,
                         allocator,
                         is_free_buffer_after_forward,
                         cuda_device_prop)
{
}

template<typename T>
TopPSamplingLayer<T>::TopPSamplingLayer(TopPSamplingLayer<T> const& top_p_sampling_layer):
    BaseSamplingLayer<T>(top_p_sampling_layer)
{
}

template<typename T>
TopPSamplingLayer<T>::~TopPSamplingLayer()
{
    freeBuffer();
}

template class TopPSamplingLayer<float>;
template class TopPSamplingLayer<half>;

}  // namespace fastertransformer
