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
#include "src/fastertransformer/utils/nvtx_utils.h"
#include <cuda_profiler_api.h>
#include <thread>

using namespace fastertransformer;

template<typename T1, typename T2>
int bertExample(size_t       batch_size,
                size_t       num_layers,
                size_t       seq_len,
                size_t       head_num,
                size_t       size_per_head,
                bool         is_remove_padding,
                std::string  model_path,
                const float& mean_rel_diff_threshold);

template<typename T1, typename T2>
void compareTwoTensorV2(const T1*         pred,
                        const T2*         ref,
                        const int         batch_size,
                        const int*        sequence_lengths,
                        const int         hidden_unit,
                        const int         max_length,
                        const int         print_size              = 0,
                        const std::string filename                = "",
                        const float&      mean_rel_diff_threshold = -1.0f)
{
    std::vector<int> v_sequence_lengths(sequence_lengths, sequence_lengths + batch_size);
    const int        size   = batch_size * max_length * hidden_unit;
    T1*              h_pred = new T1[size];
    T2*              h_ref  = new T2[size];
    check_cuda_error(cudaMemcpy(h_pred, pred, size * sizeof(T1), cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_ref, ref, size * sizeof(T2), cudaMemcpyDeviceToHost));

    FILE* fd = nullptr;
    if (filename != "") {
        fd = fopen(filename.c_str(), "w");
        fprintf(fd, "| %10s | %10s | %10s | %10s | \n", "pred", "ref", "abs_diff", "rel_diff(%)");
    }

    if (print_size > 0) {
        FT_LOG_INFO("  id |   pred  |   ref   |abs diff | rel diff (%) |");
    }
    float mean_abs_diff = 0.0f;
    float mean_rel_diff = 0.0f;
    int   count         = 0;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < sequence_lengths[i] * hidden_unit; j++) {
            const int idx = i * max_length * hidden_unit + j;
            if (idx < print_size) {
                FT_LOG_INFO("%4d | % 6.4f | % 6.4f | % 6.4f | % 7.4f |\r",
                            i,
                            (float)h_pred[idx],
                            (float)h_ref[idx],
                            abs((float)h_pred[idx] - (float)h_ref[idx]),
                            abs((float)h_pred[idx] - (float)h_ref[idx]) / (abs((float)h_ref[idx]) + 1e-6f) * 100.f);
            }

            if ((float)h_pred[idx] == 0) {
                continue;
            }
            count += 1;
            mean_abs_diff += abs((float)h_pred[idx] - (float)h_ref[idx]);
            mean_rel_diff += abs((float)h_pred[idx] - (float)h_ref[idx]) / (abs((float)h_ref[idx]) + 1e-6f) * 100.f;

            if (fd != nullptr) {
                fprintf(fd,
                        "| %10.5f | %10.5f | %10.5f | %11.5f |\n",
                        (float)h_pred[idx],
                        (float)h_ref[idx],
                        abs((float)h_pred[idx] - (float)h_ref[idx]),
                        abs((float)h_pred[idx] - (float)h_ref[idx]) / (abs((float)h_ref[idx]) + 1e-6f) * 100.f);
            }
        }
    }

    mean_abs_diff = mean_abs_diff / (float)count;
    mean_rel_diff = mean_rel_diff / (float)count;
    FT_LOG_INFO("mean_abs_diff: % 6.4f, mean_rel_diff: % 6.4f (%%)", mean_abs_diff, mean_rel_diff);

    if (fd != nullptr) {
        fprintf(fd, "mean_abs_diff: % 6.4f, mean_rel_diff: % 6.4f (%%)", mean_abs_diff, mean_rel_diff);
        fclose(fd);
    }
    delete[] h_pred;
    delete[] h_ref;

    if (mean_rel_diff_threshold > 0.0f) {
        FT_CHECK_WITH_INFO(mean_rel_diff_threshold >= mean_rel_diff,
                           fmtstr("mean_rel_diff (%f) is larger than mean_rel_diff_threshold (%f)",
                                  mean_rel_diff,
                                  mean_rel_diff_threshold));
        FT_LOG_INFO("Test PASS!");
    }
}

int main(int argc, char** argv)
{
    if (argc < 8 || argc > 10) {
        FT_LOG_ERROR("bert_example batch_size num_layers seq_len head_num size_per_head is_remove_padding thread_num");
        FT_LOG_ERROR("e.g., ./bin/bert_example 32 12 32 12 64 0 1");
        return 0;
    }

    int         batch_size              = atoi(argv[1]);
    int         num_layers              = atoi(argv[2]);
    int         seq_len                 = atoi(argv[3]);
    int         head_num                = atoi(argv[4]);
    int         size_per_head           = atoi(argv[5]);
    bool        is_remove_padding       = static_cast<bool>(atoi(argv[6]));
    const int   thread_num              = atoi(argv[7]);
    std::string model_path              = argc >= 9 ? std::string(argv[8]) : "";
    const float mean_rel_diff_threshold = argc >= 10 ? atof(argv[9]) : -1.0f;

    std::vector<std::thread> threads;
    threads.clear();

    for (int i = 0; i < thread_num; i++) {
        threads.push_back(std::thread(bertExample<__nv_fp8_e4m3, __nv_bfloat16>,
                                      batch_size,
                                      num_layers,
                                      seq_len,
                                      head_num,
                                      size_per_head,
                                      is_remove_padding,
                                      model_path,
                                      mean_rel_diff_threshold));
    }
    for (auto& t : threads) {
        t.join();
    }

    return 0;
}

template<typename T1, typename T2>
int bertExample(size_t       batch_size,
                size_t       num_layers,
                size_t       seq_len,
                size_t       head_num,
                size_t       size_per_head,
                bool         is_remove_padding,
                std::string  model_path,
                const float& mean_rel_diff_threshold)
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
    // cublasFP8MMWrapper cublas_wrapper = cublasMMWrapper(
    //     cublas_handle, cublaslt_handle, cusparselt_handle, stream, cublas_algo_map, cublas_wrapper_mutex,
    //     &allocator);
#else
    cublasFP8MMWrapper cublas_wrapper =
        cublasFP8MMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
#endif

    const size_t input_size     = batch_size * seq_len * head_num * size_per_head;
    std::string  fp32_ckpt_path = model_path + "/ft_fp32/1-gpu/";
    std::string  fp8_ckpt_path  = model_path + "/ft_fp8/1-gpu/";
    std::string  input_path     = model_path + "/bert_input_unit_test/bs_";  // use random ids

    // Preare FP16 bert model and weight
    BertWeight<half> f_bert_weights(hidden_units, inter_size, num_layers);
    f_bert_weights.loadModel(fp32_ckpt_path);

    AttentionType fp8_attention_type = getAttentionType<T1>(size_per_head, getSMVersion(), is_remove_padding, seq_len);
    AttentionType f_attention_type = getAttentionType<half>(size_per_head, getSMVersion(), is_remove_padding, seq_len);
    cublasMMWrapper f_cublas_wrapper =
        cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
    f_cublas_wrapper.setFP16GemmConfig();

    Bert<half> f_bert = Bert<half>(batch_size,
                                   seq_len,
                                   head_num,
                                   size_per_head,
                                   inter_size,
                                   num_layers,
                                   getSMVersion(),
                                   1.0f,
                                   stream,
                                   &f_cublas_wrapper,
                                   &allocator,
                                   false,
                                   f_attention_type,
                                   false,
                                   ActivationType::Gelu,
                                   LayerNormType::post_layernorm);

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
    bert_weights.loadModel(fp8_ckpt_path);
    bert_weights.transposeWeight();

    NcclParam tensor_para;
    NcclParam pipeline_para;

    BertFP8<T1, T2> bert = BertFP8<T1, T2>(head_num,
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
                                           fp8_attention_type,
                                           false,
                                           ActivationType::Gelu,
                                           LayerNormType::post_layernorm,
                                           fp8_mode);

    half* f_from_tensor;
    half* f_out_tensor;
    half* f_ref_tensor;
    deviceMalloc(&f_from_tensor, input_size, true);
    deviceMalloc(&f_out_tensor, input_size, false);
    deviceMalloc(&f_ref_tensor, input_size, false);

    half* h_results_baseline = new half[input_size];
    half* h_results_fp8      = new half[input_size];
    int*  h_sequence_lengths = new int[batch_size];

    half* out_tensor;
    T2*   ft_pooled_output;
    deviceMalloc(&out_tensor, input_size, false);
    deviceMalloc(&ft_pooled_output, batch_size * 1 * head_num * size_per_head, false);

    int* d_input_ids;
    int* d_sequence_lengths;
    int* d_token_type_ids;

    deviceMalloc(&d_input_ids, batch_size * seq_len, false);
    deviceMalloc(&d_sequence_lengths, batch_size, false);
    deviceMalloc(&d_token_type_ids, batch_size * seq_len, false);

    if (checkIfFileExist(input_path + std::to_string(0) + "_input_ids.bin")
        && checkIfFileExist(input_path + std::to_string(0) + "_input_lengths.bin")
        && checkIfFileExist(input_path + std::to_string(0) + "_token_type_ids.bin")) {

        loadWeightFromBin(d_input_ids, {(batch_size * seq_len)}, input_path + std::to_string(0) + "_input_ids.bin");
        loadWeightFromBin(d_sequence_lengths, {batch_size}, input_path + std::to_string(0) + "_input_lengths.bin");
        loadWeightFromBin(
            d_token_type_ids, {(batch_size * seq_len)}, input_path + std::to_string(0) + "_token_type_ids.bin");

        cudaD2Hcpy(h_sequence_lengths, d_sequence_lengths, batch_size);
    }
    else {
        FT_LOG_WARNING(fmtstr("Cannot load real sequence length data, set all are %d for running benchmark", seq_len));
        for (int i = 0; i < batch_size; i++) {
            h_sequence_lengths[i] = seq_len;
        }
        cudaH2Dcpy(d_sequence_lengths, h_sequence_lengths, batch_size);
        deviceFill(d_input_ids, (batch_size * seq_len), 0);
        deviceFill(d_token_type_ids, (batch_size * seq_len), 0);
    }

    // run embedding lookup for FP16 baseline
    {
        T1* tmp;
        cudaMalloc(&tmp, sizeof(T1) * input_size);
        {
            RemovePaddingEmbLookupLayerNormFP8OutParam<T1, T2> param{
                tmp,
                d_input_ids,
                nullptr,
                d_token_type_ids,
                nullptr,
                bert_weights.word_embeddings,
                bert_weights.position_embeddings,
                bert_weights.token_type_embeddings,
                bert_weights.embeddings_layernorm.gamma,
                bert_weights.embeddings_layernorm.beta,
                bert_weights.bert_layer_weights[0].attention_weights.query_weight.input_scale_inv,
                d_sequence_lengths,
                (int)(batch_size * seq_len),
                (int)hidden_units,
                (int)batch_size,
                (int)seq_len,
                0,  // pad_token_id
                stream,
                false,
            };
            PUSH_RANGE("invokeRemovePaddingEmbLookupLayerNormFP8Out");
            invokeRemovePaddingEmbLookupLayerNormFP8Out(param);
            sync_check_cuda_error();
            POP_RANGE;
        }
        {
            QuantizeMatrixRebuildPaddingParam<half, T1, QUANTIZE_MODE::PER_TENSOR> param{
                f_from_tensor,
                tmp,
                nullptr,
                (int)(batch_size * seq_len),
                (int)hidden_units,
                bert_weights.bert_layer_weights[0].attention_weights.query_weight.input_scale,
                stream};
            PUSH_RANGE("invokeQuantizeMatrixRebuildPadding");
            invokeQuantizeMatrixRebuildPadding<half, T1, QUANTIZE_MODE::PER_TENSOR>(param);
            sync_check_cuda_error();
            POP_RANGE;
        }
        cudaFree(tmp);
    }

    TensorMap input_tensors = TensorMap(std::unordered_map<std::string, Tensor>{
        {"input_ids", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size, seq_len}, d_input_ids}},
        {"sequence_lengths", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, d_sequence_lengths}},
        {"token_type_ids",
         Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size, seq_len}, d_token_type_ids}}});

    TensorMap output_tensors = TensorMap(std::unordered_map<std::string, Tensor>{
        {"output_hidden_state",
         Tensor{
             MEMORY_GPU, getTensorType<half>(), {batch_size, seq_len, (size_t)(head_num * size_per_head)}, out_tensor}},
        {"ft_pooled_output",
         Tensor{MEMORY_GPU, getTensorType<T2>(), {batch_size, (size_t)(head_num * size_per_head)}, ft_pooled_output}}});

    std::vector<Tensor> f_input_tensors =
        std::vector<Tensor>{Tensor{MEMORY_GPU,
                                   getTensorType<half>(),
                                   std::vector<size_t>{batch_size, seq_len, (size_t)(head_num * size_per_head)},
                                   f_from_tensor},
                            Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, d_sequence_lengths}};

    std::vector<Tensor> f_output_tensors =
        std::vector<Tensor>{Tensor{MEMORY_GPU,
                                   getTensorType<half>(),
                                   std::vector<size_t>{batch_size, seq_len, (size_t)(head_num * size_per_head)},
                                   f_out_tensor}};

    FT_LOG_INFO("run FP8 bert");

    // warmup
    for (int i = 0; i < 0; i++) {
        bert.forward(&output_tensors, &input_tensors, &bert_weights);
    }

    cudaProfilerStart();
    CudaTimer cuda_timer(stream);
    ft_nvtx::resetScope();
    ft_nvtx::addScope("Bert");
    cuda_timer.start();

    const int ite = 1;
    for (int i = 0; i < ite; i++) {
        bert.forward(&output_tensors, &input_tensors, &bert_weights);
    }
    float total_time = cuda_timer.stop();
    cudaProfilerStop();

    FT_LOG_INFO("batch_size %ld seq_len %ld layer %ld FT-CPP-time %.2f ms (%d iterations) ",
                batch_size,
                seq_len,
                num_layers,
                total_time / ite,
                ite);

    // verify correctness
    {
        FT_LOG_INFO("run FP16 bert");
        f_bert.forward(&f_output_tensors, &f_input_tensors, &f_bert_weights);
        compareTwoTensorV2(
            out_tensor, f_out_tensor, batch_size, h_sequence_lengths, hidden_units, seq_len, 0, "diff.fp8.fp16.txt");
    }

    if (checkIfFileExist(input_path + std::to_string(0) + "_layer_" + std::to_string(num_layers - 1) + "_output.bin")) {
        loadWeightFromBin(f_ref_tensor,
                          {input_size},
                          input_path + std::to_string(0) + "_layer_" + std::to_string(num_layers - 1) + "_output.bin",
                          FtCudaDataType::FP32);
        compareTwoTensorV2(out_tensor,
                           f_ref_tensor,
                           batch_size,
                           h_sequence_lengths,
                           hidden_units,
                           seq_len,
                           0,
                           "diff.fp8.hf-fp16.txt",
                           mean_rel_diff_threshold);
    }

    delete[] h_results_baseline;
    delete[] h_results_fp8;
    delete[] h_sequence_lengths;

#ifdef SPARSITY_ENABLED
    // cusparseLtDestroy(&cusparselt_handle);
#endif
    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
    return 0;
}
