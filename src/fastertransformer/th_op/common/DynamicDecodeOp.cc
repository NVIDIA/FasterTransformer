/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/th_op/common/DynamicDecodeOp.h"

namespace th = torch;
namespace ft = fastertransformer;

namespace torch_ext {

template<typename T>
FtDynamicDecode<T>::FtDynamicDecode(const size_t vocab_size,
                                    const size_t vocab_size_padded,
                                    const int    tensor_para_size,
                                    const int    pipeline_para_size):
    vocab_size_(vocab_size), vocab_size_padded_(vocab_size_padded)
{
    ft::FT_CHECK_WITH_INFO(
        vocab_size_padded_ % tensor_para_size == 0,
        ft::fmtstr("vocab_size (%ld) is not multiple of tensor_para_size (%d).", vocab_size_padded_, tensor_para_size));
    ft::ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size, pipeline_para_size);

    allocator_ = new ft::Allocator<ft::AllocatorType::TH>();
    ft::check_cuda_error(cublasLtCreate(&cublaslt_handle_));
    cublas_wrapper_mutex_ = new std::mutex();
    cublas_algo_map_      = new ft::cublasAlgoMap(GEMM_CONFIG);

    auto stream        = at::cuda::getCurrentCUDAStream().stream();
    auto cublas_handle = at::cuda::getCurrentCUDABlasHandle();

    cublas_wrapper_ = new ft::cublasMMWrapper(
        cublas_handle, cublaslt_handle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, allocator_);

    cudaDeviceProp prop;
    ft::check_cuda_error(cudaGetDeviceProperties(&prop, 0));

    dynamic_decode_layer_ = new ft::DynamicDecodeLayer<T>(
        vocab_size_, vocab_size_padded_, 0, stream, cublas_wrapper_, allocator_, false, &prop_);
}

template<typename T>
FtDynamicDecode<T>::~FtDynamicDecode()
{
    delete dynamic_decode_layer_;
    ft::ftNcclParamDestroy(tensor_para_);
    ft::ftNcclParamDestroy(pipeline_para_);
    cublasLtDestroy(cublaslt_handle_);
    delete cublas_wrapper_mutex_;
    delete cublas_algo_map_;
    delete allocator_;
}

template<typename T>
void FtDynamicDecode<T>::setup(size_t                   batch_size,
                               size_t                   beam_width,
                               th::optional<th::Tensor> runtime_top_k_opt,
                               th::optional<th::Tensor> runtime_top_p_opt,
                               th::optional<th::Tensor> temperature_opt,
                               th::optional<th::Tensor> repetition_penalty_opt,
                               th::optional<th::Tensor> length_penalty_opt,
                               th::optional<th::Tensor> beam_search_diversity_rate_opt,
                               th::optional<th::Tensor> random_seed_opt,
                               th::optional<th::Tensor> top_p_decay_opt,
                               th::optional<th::Tensor> top_p_min_opt,
                               th::optional<th::Tensor> top_p_reset_ids_opt)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cublas_wrapper_->setStream(stream);
    dynamic_decode_layer_->setStream(stream);

#define SAFE_INSERT(map_, name_, type_, optional_)                                                                     \
    if (optional_.has_value()) {                                                                                       \
        map_.insert(name_, convert_tensor<type_>(optional_.value()));                                                  \
    }

    ft::TensorMap runtime_args;
    SAFE_INSERT(runtime_args, "runtime_top_k", uint, runtime_top_k_opt);
    SAFE_INSERT(runtime_args, "runtime_top_p", float, runtime_top_p_opt);
    SAFE_INSERT(runtime_args, "temperature", float, temperature_opt);
    SAFE_INSERT(runtime_args, "repetition_penalty", float, repetition_penalty_opt);
    SAFE_INSERT(runtime_args, "len_penalty", float, length_penalty_opt);
    SAFE_INSERT(runtime_args, "beam_search_diversity_rate", float, beam_search_diversity_rate_opt);
    SAFE_INSERT(runtime_args, "random_seed", unsigned long long, random_seed_opt);
    SAFE_INSERT(runtime_args, "top_p_decay_opt", float, top_p_decay_opt);
    SAFE_INSERT(runtime_args, "top_p_min", float, top_p_min_opt);
    SAFE_INSERT(runtime_args, "top_p_reset_ids", uint, top_p_reset_ids_opt);

    dynamic_decode_layer_->setup(batch_size, beam_width, &runtime_args);

#undef SAFE_INSERT
}

template<typename T>
void FtDynamicDecode<T>::forward(th::Tensor&              logits,  // (batch_size, beam_width, hidden_size)
                                 int                      step,
                                 int                      max_input_length,
                                 uint                     ite,
                                 int                      local_batch_size,
                                 th::Tensor               end_id,
                                 th::optional<th::Tensor> runtime_top_k_opt,
                                 th::optional<th::Tensor> runtime_top_p_opt,
                                 th::optional<th::Tensor> temperature_opt,
                                 th::optional<th::Tensor> repetition_penalty_opt,
                                 th::optional<th::Tensor> length_penalty_opt,
                                 th::optional<th::Tensor> beam_search_diversity_rate_opt,
                                 th::optional<th::Tensor> top_p_decay_opt,
                                 th::optional<th::Tensor> top_p_min_opt,
                                 th::optional<th::Tensor> top_p_reset_ids_opt,
                                 th::optional<th::Tensor> embedding_bias_opt,
                                 th::optional<th::Tensor> input_lengths_opt,
                                 th::optional<th::Tensor> sequence_limit_length_opt,
                                 th::optional<th::Tensor> stop_words_list_opt,
                                 th::optional<th::Tensor> bad_words_list_opt,
                                 th::optional<th::Tensor> src_cache_indirection_opt,
                                 // Outputs
                                 th::Tensor&              output_token_ids,
                                 th::Tensor&              should_stop,
                                 th::optional<th::Tensor> finished_opt,
                                 th::optional<th::Tensor> sequence_lengths_opt,
                                 th::optional<th::Tensor> cum_log_probs_opt,
                                 th::optional<th::Tensor> output_log_probs_opt,
                                 th::optional<th::Tensor> parent_ids_opt,
                                 th::optional<th::Tensor> tgt_cache_indirection_opt)
{
    ft::TensorMap input_tensors{
        {"logits", convert_tensor<T>(logits)},
        {"step", ft::Tensor(ft::MEMORY_CPU, ft::TYPE_INT32, {1}, &step)},
        {"max_input_length", ft::Tensor(ft::MEMORY_CPU, ft::TYPE_INT32, {1}, &max_input_length)},
        {"ite", ft::Tensor(ft::MEMORY_CPU, ft::TYPE_UINT32, {1}, &ite)},
        {"local_batch_size", ft::Tensor(ft::MEMORY_CPU, ft::TYPE_INT32, {1}, &local_batch_size)},
        {"end_id", convert_tensor<int>(end_id)}};

#define SAFE_INSERT(map_, name_, type_, optional_)                                                                     \
    if (optional_.has_value()) {                                                                                       \
        map_.insert(name_, convert_tensor<type_>(optional_.value()));                                                  \
    }

    SAFE_INSERT(input_tensors, "runtime_top_k", uint, runtime_top_k_opt);
    SAFE_INSERT(input_tensors, "runtime_top_p", float, runtime_top_p_opt);
    SAFE_INSERT(input_tensors, "temperature", float, temperature_opt);
    SAFE_INSERT(input_tensors, "repetition_penalty", float, repetition_penalty_opt);
    SAFE_INSERT(input_tensors, "len_penalty", float, length_penalty_opt);
    SAFE_INSERT(input_tensors, "beam_search_diversity_rate", float, beam_search_diversity_rate_opt);
    SAFE_INSERT(input_tensors, "top_p_decay_opt", float, top_p_decay_opt);
    SAFE_INSERT(input_tensors, "top_p_min", float, top_p_min_opt);
    SAFE_INSERT(input_tensors, "top_p_reset_ids", uint, top_p_reset_ids_opt);

    SAFE_INSERT(input_tensors, "embedding_bias", T, embedding_bias_opt);
    SAFE_INSERT(input_tensors, "input_lengths", int, input_lengths_opt);
    SAFE_INSERT(input_tensors, "sequence_limit_length", int, sequence_limit_length_opt);
    SAFE_INSERT(input_tensors, "stop_words_list", int, stop_words_list_opt);
    SAFE_INSERT(input_tensors, "bad_words_list", int, bad_words_list_opt);
    SAFE_INSERT(input_tensors, "src_cache_indirection", int, src_cache_indirection_opt);

    ft::TensorMap output_tensors{{"output_ids", convert_tensor<int>(output_token_ids)},
                                 {"should_stop", convert_tensor<bool>(should_stop)}};
    SAFE_INSERT(output_tensors, "finished", bool, finished_opt);
    SAFE_INSERT(output_tensors, "sequence_length", int, sequence_lengths_opt);
    SAFE_INSERT(output_tensors, "parent_ids", int, parent_ids_opt);
    SAFE_INSERT(output_tensors, "cum_log_probs", float, cum_log_probs_opt);
    SAFE_INSERT(output_tensors, "output_log_probs", float, output_log_probs_opt);
    SAFE_INSERT(output_tensors, "tgt_cache_indirection", int, tgt_cache_indirection_opt);
#undef SAFE_INSERT

    dynamic_decode_layer_->forward(&output_tensors, &input_tensors);
}

template<typename T>
void FtDynamicDecode<T>::broadcastFromLastPipeline(std::vector<th::Tensor> tensors)
{
    if (pipeline_para_.world_size_ == 1) {
        return;
    }

    auto stream  = at::cuda::getCurrentCUDAStream().stream();
    auto pp_rank = pipeline_para_.rank_;
    auto pp_size = pipeline_para_.world_size_;

    ft::ftNcclGroupStart();
    for (auto const& tensor : tensors) {
        char* buffer = reinterpret_cast<char*>(tensor.data_ptr());
        ft::ftNcclBroadCast(buffer, sizeBytes(tensor), pp_size - 1, pipeline_para_, stream);
    }
    ft::ftNcclGroupEnd();
    // throw errors when detected
    ft::ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream);
    ft::sync_check_cuda_error();
}

DynamicDecodeOp::DynamicDecodeOp(const int64_t  vocab_size,
                                 const int64_t  vocab_size_padded,
                                 const int64_t  tensor_para_size,
                                 const int64_t  pipeline_para_size,
                                 at::ScalarType scalar_type):
    vocab_size_(static_cast<size_t>(vocab_size)),
    vocab_size_padded_(static_cast<size_t>(vocab_size_padded)),
    tensor_para_size_(static_cast<int>(tensor_para_size)),
    pipeline_para_size_(static_cast<int>(pipeline_para_size)),
    scalar_type_(scalar_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    createInstance();
}

DynamicDecodeOp::~DynamicDecodeOp()
{
    // Do nothing.
}

void DynamicDecodeOp::createInstance()
{
    dynamic_decode_.reset();
    switch (scalar_type_) {
        case at::ScalarType::Float:
            dynamic_decode_ = std::make_unique<FtDynamicDecode<float>>(
                vocab_size_, vocab_size_padded_, tensor_para_size_, pipeline_para_size_);
            break;
        case at::ScalarType::Half:
            dynamic_decode_ = std::make_unique<FtDynamicDecode<half>>(
                vocab_size_, vocab_size_padded_, tensor_para_size_, pipeline_para_size_);
            break;
        default:
            throw std::runtime_error("Wrong tensor type.");
    }
}

void DynamicDecodeOp::setup(int64_t                  batch_size,
                            int64_t                  beam_width,
                            th::optional<th::Tensor> runtime_top_k_opt,
                            th::optional<th::Tensor> runtime_top_p_opt,
                            th::optional<th::Tensor> temperature_opt,
                            th::optional<th::Tensor> repetition_penalty_opt,
                            th::optional<th::Tensor> length_penalty_opt,
                            th::optional<th::Tensor> beam_search_diversity_rate_opt,
                            th::optional<th::Tensor> random_seed_opt,
                            th::optional<th::Tensor> top_p_decay_opt,
                            th::optional<th::Tensor> top_p_min_opt,
                            th::optional<th::Tensor> top_p_reset_ids_opt)
{
    // TODO: Revise DynamicDecodeLayer and make the decode arguments consistent.
    CHECK_OPTIONAL_CPU_INPUT(runtime_top_k_opt, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(runtime_top_p_opt, torch::kFloat);

    CHECK_OPTIONAL_CPU_INPUT(temperature_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(repetition_penalty_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(length_penalty_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(beam_search_diversity_rate_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(random_seed_opt, torch::kInt64);
    CHECK_OPTIONAL_INPUT(top_p_decay_opt, torch::kFloat);
    CHECK_OPTIONAL_INPUT(top_p_min_opt, torch::kFloat);
    CHECK_OPTIONAL_INPUT(top_p_reset_ids_opt, torch::kInt32);

    dynamic_decode_->setup(static_cast<size_t>(batch_size),
                           static_cast<size_t>(beam_width),
                           runtime_top_k_opt,
                           runtime_top_p_opt,
                           temperature_opt,
                           repetition_penalty_opt,
                           length_penalty_opt,
                           beam_search_diversity_rate_opt,
                           random_seed_opt,
                           top_p_decay_opt,
                           top_p_min_opt,
                           top_p_reset_ids_opt);
}

th::Tensor DynamicDecodeOp::forward(th::Tensor               logits,
                                    int64_t                  step,
                                    int64_t                  max_input_length,
                                    int64_t                  ite,
                                    int64_t                  local_batch_size,
                                    th::Tensor               end_id,
                                    th::optional<th::Tensor> runtime_top_k_opt,
                                    th::optional<th::Tensor> runtime_top_p_opt,
                                    th::optional<th::Tensor> temperature_opt,
                                    th::optional<th::Tensor> repetition_penalty_opt,
                                    th::optional<th::Tensor> length_penalty_opt,
                                    th::optional<th::Tensor> beam_search_diversity_rate_opt,
                                    th::optional<th::Tensor> top_p_decay_opt,
                                    th::optional<th::Tensor> top_p_min_opt,
                                    th::optional<th::Tensor> top_p_reset_ids_opt,
                                    th::optional<th::Tensor> embedding_bias_opt,
                                    th::optional<th::Tensor> input_lengths_opt,  // length of input contexts.
                                    th::optional<th::Tensor> sequence_limit_length_opt,
                                    th::optional<th::Tensor> stop_words_list_opt,
                                    th::optional<th::Tensor> bad_words_list_opt,
                                    th::optional<th::Tensor> src_cache_indirection_opt,
                                    // output buffers.
                                    th::Tensor               output_token_ids,
                                    th::optional<th::Tensor> finished_opt,
                                    th::optional<th::Tensor> seuqence_lengths_opt,  // length of the current sequences.
                                    th::optional<th::Tensor> cum_log_probs_opt,
                                    th::optional<th::Tensor> output_log_probs_opt,
                                    th::optional<th::Tensor> parent_ids_opt,
                                    th::optional<th::Tensor> tgt_cache_indirection_opt)
{
    // Input Arguments:
    //     logits: [batch_size, beam_width, vocab_size_padded], T
    //     end_id: [batch_size], int, optional
    //     runtime_top_k: [batch_size], int, optional
    //     runtime_top_p: [batch_size], float, optional
    //     temperature: [batch_size], float, optional
    //     repetition_penalty: [batch_size], float, optional
    //     length_penalty: [batch_size], float, optional
    //     beam_search_diversity_rate: [batch_size], float, optional
    //     top_p_decay: [batch_size], float, optional
    //     top_p_min: [batch_size], float, optional
    //     top_p_reset_ids: [batch_size], int, optional
    //     embedding_bias: [vocab_size_padded], T, optional
    //     input_lengths: [batch_size * beam_width], int, optional
    //     sequence_limit_length: [batch_size], int, optional
    //     stop_words_list: [batch_size, 2, stop_words_length], int, optional
    //     bad_words_list: [2, stop_words_length], int, optional
    //     src_cache_indirection: [local_batch_size, beam_width, memory_length], int, optional
    //     output_token_ids: [max_seq_length, batch_size, beam_width], int
    //     finished: [batch_size * beam_width], bool, optional
    //     sequence_lengths: [batch_size * beam_width], int, optional
    //     cum_log_probs: [batch_size * beam_width], float, optional
    //     output_log_probs: [gen_length, batch_size, beam_width], float, optional
    //     parent_ids: [gen_length, batch_size, beam_width], float, optional
    //     tgt_cache_indirection: [local_batch_size, beam_width, memory_length], float, optional

    CHECK_INPUT(logits, scalar_type_);
    ft::FT_CHECK_WITH_INFO(logits.dim() == 3,
                           ft::fmtstr("logits is of shape (batch_size, beam_width, vocab_size_padded), "
                                      "but got dim=%d shape=%s",
                                      (int)logits.dim(),
                                      ft::vec2str(convert_shape(logits)).c_str())
                               .c_str());
    ft::FT_CHECK_WITH_INFO(
        static_cast<size_t>(logits.size(2)) == vocab_size_padded_,
        ft::fmtstr("logits is of shape (batch_size, beam_width, vocab_size(%ld)), but got the last dim=%d.",
                   vocab_size_padded_,
                   static_cast<size_t>(logits.size(2))));

    CHECK_INPUT(end_id, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(runtime_top_k_opt, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(runtime_top_p_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(temperature_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(repetition_penalty_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(length_penalty_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(beam_search_diversity_rate_opt, torch::kFloat);
    CHECK_OPTIONAL_INPUT(top_p_decay_opt, torch::kFloat);
    CHECK_OPTIONAL_INPUT(top_p_min_opt, torch::kFloat);
    CHECK_OPTIONAL_INPUT(top_p_reset_ids_opt, torch::kInt32);

    CHECK_OPTIONAL_INPUT(input_lengths_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(sequence_limit_length_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(stop_words_list_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(bad_words_list_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(src_cache_indirection_opt, torch::kInt32);

    CHECK_INPUT(output_token_ids, torch::kInt32);
    CHECK_OPTIONAL_INPUT(finished_opt, torch::kBool);
    CHECK_OPTIONAL_INPUT(seuqence_lengths_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(cum_log_probs_opt, torch::kFloat32);
    CHECK_OPTIONAL_INPUT(output_log_probs_opt, torch::kFloat32);
    CHECK_OPTIONAL_INPUT(parent_ids_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(tgt_cache_indirection_opt, torch::kInt32);

    th::Tensor should_stop = torch::zeros({}, torch::dtype(torch::kBool).requires_grad(false));

    dynamic_decode_->forward(
        // Inputs
        logits,
        static_cast<int>(step),
        static_cast<int>(max_input_length),
        static_cast<uint>(ite),
        static_cast<int>(local_batch_size),
        end_id,
        runtime_top_k_opt,
        runtime_top_p_opt,
        temperature_opt,
        repetition_penalty_opt,
        length_penalty_opt,
        beam_search_diversity_rate_opt,
        top_p_decay_opt,
        top_p_min_opt,
        top_p_reset_ids_opt,
        embedding_bias_opt,
        input_lengths_opt,
        sequence_limit_length_opt,
        stop_words_list_opt,
        bad_words_list_opt,
        src_cache_indirection_opt,
        // Outputs
        output_token_ids,
        should_stop,
        finished_opt,
        seuqence_lengths_opt,
        cum_log_probs_opt,
        output_log_probs_opt,
        parent_ids_opt,
        tgt_cache_indirection_opt);

    return should_stop;
}

void DynamicDecodeOp::broadcastFromLastPipeline(std::vector<th::Tensor> tensors)
{
    for (size_t i = 0; i < tensors.size(); ++i) {
        CHECK_TH_CUDA(tensors[i]);
        CHECK_CONTIGUOUS(tensors[i]);
    }
    dynamic_decode_->broadcastFromLastPipeline(tensors);
}
}  // namespace torch_ext

static auto fasterTransformerGptContextDecoderTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::DynamicDecodeOp>("FasterTransformerDynamicDecodeOp")
#else
    torch::jit::class_<torch_ext::DynamicDecodeOp>("FasterTransformer", "DynamicDecodeOp")
#endif
        .def(torch::jit::init<int64_t, int64_t, int64_t, int64_t, at::ScalarType>())
        .def("setup", &torch_ext::DynamicDecodeOp::setup)
        .def("forward", &torch_ext::DynamicDecodeOp::forward)
        .def("broadcast_from_last_pipeline", &torch_ext::DynamicDecodeOp::broadcastFromLastPipeline);
