/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/bert/Bert.h"
#include "src/fastertransformer/models/bert_fp8/BertFP8.h"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include "src/fastertransformer/utils/logger.h"

using namespace fastertransformer;

template<typename T1, typename T2>
int bertExample(size_t max_batch_size,
                size_t num_layers,
                size_t head_num,
                size_t size_per_head,
                bool   is_remove_padding,
                bool   allow_gemm_test = false);

int main(int argc, char** argv)
{
    if (argc != 8 && argc != 9) {
        FT_LOG_ERROR(
            "bert_fp8_example_squad max_batch_size num_layers seq_len head_num size_per_head is_fp16 is_remove_padding");
        FT_LOG_ERROR("e.g., ./bin/bert_fp8_example_squad 32 24 384 16 64 1 1");
        return 0;
    }
    bool allow_gemm_test = false;
    if (argc == 9) {
        allow_gemm_test = (atoi(argv[8]) == 1) ? true : false;
    }

    int  max_batch_size    = atoi(argv[1]);
    int  num_layers        = atoi(argv[2]);
    int  seq_len           = atoi(argv[3]);
    int  head_num          = atoi(argv[4]);
    int  size_per_head     = atoi(argv[5]);
    bool is_remove_padding = static_cast<bool>(atoi(argv[7]));

    return bertExample<__nv_fp8_e4m3, __nv_bfloat16>(
        max_batch_size, num_layers, head_num, size_per_head, is_remove_padding, allow_gemm_test);
}

template<typename T1, typename T2>
int bertExample(size_t max_batch_size,
                size_t num_layers,
                size_t head_num,
                size_t size_per_head,
                bool   is_remove_padding,
                bool   allow_gemm_test)
{
    const int fp8_mode = 2;
    FT_LOG_INFO("fp8_mode: %d", fp8_mode);
    FT_LOG_INFO("Device: %s \n", getDeviceName().c_str());

    const size_t hidden_units            = head_num * size_per_head;
    const size_t inter_size              = 4 * hidden_units;
    const size_t vocab_size              = 30522;  // Fixed by bert model config
    const size_t max_position_embeddings = 512;    // Fixed by bert model config
    const size_t token_type_vocab_size   = 2;      // Fixed by bert model config

    cudaStream_t     stream;
    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparselt_handle;
    CHECK_CUSPARSE(cusparseLtInit(&cusparselt_handle));
#endif
    cublasSetStream(cublas_handle, stream);
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap("gemm_config.in", "");

    Allocator<AllocatorType::CUDA> allocator(getDevice());

    std::mutex* cublas_wrapper_mutex = new std::mutex();
#ifdef SPARSITY_ENABLED
    // cublasFP8MMWrapper cublas_wrapper = cublasFP8MMWrapper(
    //     cublas_handle, cublaslt_handle, cusparselt_handle, stream, cublas_algo_map, cublas_wrapper_mutex,
    //     &allocator);
#else
    cublasFP8MMWrapper cublas_wrapper =
        cublasFP8MMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
#endif

    // Prepare FP8 bert model and weight
    BertFP8Weight<T1, T2> bert_weights(hidden_units,
                                       head_num,
                                       size_per_head,
                                       inter_size,
                                       num_layers,
                                       vocab_size,
                                       max_position_embeddings,
                                       token_type_vocab_size,
                                       1,
                                       1,
                                       fp8_mode,
                                       true,
                                       true);
    bert_weights.loadModel("/home/scratch.bhsueh_sw_1/FP8/mlperf/ft_fp8/");
    bert_weights.transposeWeight();

    NcclParam       tensor_para;
    NcclParam       pipeline_para;
    AttentionType   attention_type = getAttentionType<T1>(size_per_head, getSMVersion(), is_remove_padding, 384);
    BertFP8<T1, T2> bert           = BertFP8<T1, T2>(head_num,
                                           size_per_head,
                                           head_num * size_per_head,
                                           inter_size,
                                           num_layers,
                                           tensor_para,
                                           pipeline_para,
                                           getSMVersion(),
                                           1.0f,
                                           stream,
                                           &cublas_wrapper,
                                           &allocator,
                                           false,
                                           attention_type,
                                           false,
                                           ActivationType::Gelu,
                                           LayerNormType::post_layernorm,
                                           fp8_mode);

    // SQuAD test settings
    const int TOTAL_SENTENCE_NUM = 10833;
    const int seq_len            = 384;

    // unit test settings
    FT_CHECK(seq_len == 384);

    half* out_tensor;
    deviceMalloc(&out_tensor, max_batch_size * seq_len * hidden_units, false);
    T2* ft_pooled_output;
    deviceMalloc(&ft_pooled_output, max_batch_size * 1 * hidden_units, false);
    int* d_input_ids;
    deviceMalloc(&d_input_ids, TOTAL_SENTENCE_NUM * seq_len, false);
    int* d_sequence_lengths;
    deviceMalloc(&d_sequence_lengths, TOTAL_SENTENCE_NUM, false);
    int* d_token_type_ids;
    deviceMalloc(&d_token_type_ids, TOTAL_SENTENCE_NUM * seq_len, false);
    half* h_results_fp8 = new half[max_batch_size * seq_len * hidden_units];

    loadWeightFromBin(d_input_ids,
                      {(int)(TOTAL_SENTENCE_NUM * seq_len)},
                      "/home/scratch.bhsueh_sw_1/FP8/bert_input/bs_0_input_ids.bin");

    loadWeightFromBin(d_sequence_lengths,
                      {(int)TOTAL_SENTENCE_NUM},
                      "/home/scratch.bhsueh_sw_1/FP8/bert_input/bs_0_input_lengths.bin");

    loadWeightFromBin(d_token_type_ids,
                      {(int)(TOTAL_SENTENCE_NUM * seq_len)},
                      "/home/scratch.bhsueh_sw_1/FP8/bert_input/bs_0_token_type_ids.bin");

    std::string output_path            = "./bert_output/";
    int         remain_sentences       = TOTAL_SENTENCE_NUM;
    int*        d_input_ids_ptr        = d_input_ids;
    int*        d_sequence_lengths_ptr = d_sequence_lengths;
    int*        d_token_type_ids_ptr   = d_token_type_ids;
    int         batch_count            = 0;
    while (remain_sentences > 0) {
        FT_LOG_INFO("remain_sentences = (%5d/%5d), batch_count: %5d/%d",
                    remain_sentences,
                    TOTAL_SENTENCE_NUM,
                    batch_count,
                    (TOTAL_SENTENCE_NUM + (max_batch_size - 1)) / max_batch_size);
        const size_t batch_size = remain_sentences > max_batch_size ? max_batch_size : remain_sentences;
        const size_t input_size = batch_size * seq_len * hidden_units;

        TensorMap input_tensors = TensorMap(std::unordered_map<std::string, Tensor>{
            {"input_ids", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size, seq_len}, d_input_ids_ptr}},
            {"sequence_lengths",
             Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, d_sequence_lengths_ptr}},
            {"token_type_ids",
             Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size, seq_len}, d_token_type_ids_ptr}}});

        TensorMap output_tensors = TensorMap(
            std::unordered_map<std::string, Tensor>{{"output_hidden_state",
                                                     Tensor{MEMORY_GPU,
                                                            getTensorType<half>(),
                                                            {batch_size, seq_len, (size_t)(head_num * size_per_head)},
                                                            out_tensor}},
                                                    {"ft_pooled_output",
                                                     Tensor{MEMORY_GPU,
                                                            getTensorType<T2>(),
                                                            {batch_size, (size_t)(head_num * size_per_head)},
                                                            ft_pooled_output}}});
        bert.forward(&output_tensors, &input_tensors, &bert_weights);

        {
            cudaD2Hcpy(h_results_fp8, out_tensor, input_size);
            std::string   filename = output_path + "/bs_" + std::to_string(batch_count) + "_outputs.bin";
            std::ofstream out(filename, std::ios::out | std::ios::binary);
            if (!out.is_open()) {
                FT_LOG_WARNING("file %s cannot be opened, loading model fails! \n", filename.c_str());
                return 0;
            }

            out.write((char*)h_results_fp8, input_size * sizeof(half));
        }
        d_input_ids_ptr += batch_size * seq_len;
        d_sequence_lengths_ptr += batch_size;
        d_token_type_ids_ptr += batch_size * seq_len;
        remain_sentences -= batch_size;
        batch_count++;
    }

    deviceFree(out_tensor);
    deviceFree(ft_pooled_output);
    deviceFree(d_input_ids);
    deviceFree(d_sequence_lengths);
    deviceFree(d_token_type_ids);
    delete[] h_results_fp8;

#ifdef SPARSITY_ENABLED
    // cusparseLtDestroy(&cusparselt_handle);
#endif
    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
    return 0;
}
