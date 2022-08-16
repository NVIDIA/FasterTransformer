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

#include <algorithm>   // std::min, std::max
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdexcept>
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include "src/fastertransformer/kernels/beam_search_penalty_kernels.h"
#include "src/fastertransformer/kernels/sampling_penalty_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"

#include "tests/unittests/unittest_utils.h"

using namespace fastertransformer;

struct TemperatureTestCase {
    size_t batch_size;
    size_t vocab_size;
    float temperature;

    std::string toString() {
        char buf[200];
        snprintf(buf, sizeof(buf),
                 "TemperatureTestCase[batch=%ld, vocab=%ld, temperature=%4.2f]",
                 batch_size, vocab_size, temperature);
        return buf;
    }

    void print() {
        FT_LOG_INFO(toString());
    }
};

struct RepetitionTestCase {
    size_t batch_size;
    size_t vocab_size;
    size_t max_input_length;
    float repetition_penalty;

    std::string toString() {
        char buf[200];
        snprintf(buf, sizeof(buf),
                 "RepetitionTestCase[batch=%ld, vocab=%ld, max_input_length=%ld, repetition_penalty=%4.2f]",
                 batch_size, vocab_size, max_input_length, repetition_penalty);
        return buf;
    }

    void print() {
        FT_LOG_INFO(toString());
    }
};

size_t pad_vocab_size(size_t vocab_size, size_t pad = 8) {
    return (vocab_size + pad - 1) / pad * pad;
}

void checkTemperatureValidity(float temperature) {
    if (temperature <= 0.0f) {
        throw std::domain_error(
            fmtstr("temperature should be positive but got %.2f.", temperature));
    }
}

template<typename T>
void applyTemperature(T* logits,
                      const T* bias,
                      const float temperature,
                      const size_t batch_size,
                      const size_t vocab_size,
                      const size_t vocab_size_padded)
{
    checkTemperatureValidity(temperature);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < vocab_size; ++j) {
            size_t index = i * vocab_size_padded + j;
            float logit = static_cast<float>(logits[index]);
            if (bias != nullptr) {
                logit += static_cast<float>(bias[j]);
            }
            logits[index] = static_cast<T>(logit / temperature);
        }
    }
}

template<typename T>
void batchApplyTemperature(T* logits,
                           const T* bias,
                           const float* temperatures,
                           const size_t batch_size,
                           const size_t vocab_size,
                           const size_t vocab_size_padded)
{
    for (size_t i = 0; i < batch_size; ++i) {
        float temperature = temperatures[i];
        checkTemperatureValidity(temperature);
        for (size_t j = 0; j < vocab_size; ++j) {
            size_t index = i * vocab_size_padded + j;
            float logit = static_cast<float>(logits[index]);
            if (bias != nullptr) {
                logit += static_cast<float>(bias[j]);
            }
            logits[index] = static_cast<T>(logit / temperature);
        }
    }
}

template<typename T>
void applyRepetitonPenalty(T* logits,
                           const int* output_ids,
                           const int* input_lengths,
                           const float repetition_penalty,
                           const size_t step,
                           const size_t max_input_length,
                           const size_t batch_size,
                           const size_t vocab_size,
                           const size_t vocab_size_padded)
{
    bool* penalized = new bool[vocab_size];
    for (size_t i = 0; i < batch_size; ++i) {
        std::fill_n(penalized, vocab_size, false);
        size_t length = std::min<int>(step, input_lengths[i]);
        size_t offset = i * vocab_size_padded;
        for (size_t t = 0; t < step; ++t) {
            if (t >= (size_t)input_lengths[i] && t < max_input_length) {
                continue;
            }
            int token_id = output_ids[i + t * batch_size];
            if (!penalized[token_id]) {
                float logit = static_cast<float>(logits[offset + token_id]);
                logits[offset + token_id] = static_cast<T>(logit < 0.0f ?
                    logit * repetition_penalty : logit / repetition_penalty);
                penalized[token_id] = true;
            }
        }
    }
    delete[] penalized;
}

template<typename T>
void batchApplyRepetitonPenalty(T* logits,
                                const int* output_ids,
                                const int* input_lengths,
                                const float* repetition_penalties,
                                const size_t step,
                                const size_t max_input_length,
                                const size_t batch_size,
                                const size_t vocab_size,
                                const size_t vocab_size_padded)
{
    bool* penalized = new bool[vocab_size];
    for (size_t i = 0; i < batch_size; ++i) {
        float repetition_penalty = repetition_penalties[i];
        std::fill_n(penalized, vocab_size, false);
        size_t offset = i * vocab_size_padded;
        for (size_t t = 0; t < step; ++t) {
            if (t >= (size_t)input_lengths[i] && t < max_input_length) {
                continue;
            }
            int token_id = output_ids[i + t * batch_size];
            if (!penalized[token_id]) {
                float logit = static_cast<float>(logits[offset + token_id]);
                logits[offset + token_id] = static_cast<T>(logit < 0.0f ?
                    logit * repetition_penalty : logit / repetition_penalty);
                penalized[token_id] = true;
            }
        }
    }
    delete[] penalized;
}

template<typename T>
void initLogitsAndBias(T* logits,
                       T* bias,
                       const size_t batch_size,
                       const size_t vocab_size,
                       const size_t vocab_size_padded)
{
    initRandom(logits, batch_size * vocab_size_padded, -5.0f, 5.0f);
    if (bias != nullptr) {
        initRandom(bias, vocab_size, -5.0f, 5.0f);
    }
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < vocab_size_padded; ++j) {
            if (j >= vocab_size) {
                logits[i * vocab_size_padded + j] = static_cast<T>(isHalf<T>() ? -65504.f : -FLT_MAX);
                if (bias != nullptr && i == 0) {
                    bias[j] = (T)0.0f;
                }
            }
        }
    }
}


/////////////////////////////////// Tests //////////////////////////////////////////

template<typename T>
void testApplyTemperaturePenaltyKernel(TemperatureTestCase tc) {
    // Set up test
    const size_t batch_size = tc.batch_size;
    const size_t vocab_size = tc.vocab_size;
    const size_t vocab_size_padded = pad_vocab_size(vocab_size);

    const float temperature = tc.temperature;
    T* h_logits = new T[batch_size * vocab_size_padded];
    T* h_bias = new T[vocab_size_padded];
    initLogitsAndBias(h_logits, h_bias, batch_size, vocab_size, vocab_size_padded);

    T* d_logits;
    T* d_bias;
    check_cuda_error(cudaMalloc(&d_logits, sizeof(T) * batch_size * vocab_size_padded));
    check_cuda_error(cudaMalloc(&d_bias, sizeof(T) * vocab_size_padded));
    check_cuda_error(cudaMemcpy(d_logits, h_logits, sizeof(T) * batch_size * vocab_size_padded, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_bias, h_bias, sizeof(T) * vocab_size_padded, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));

    // Do test
    invokeApplyTemperaturePenalty(d_logits,
                                  d_bias,
                                  temperature,
                                  batch_size,
                                  vocab_size,
                                  vocab_size_padded,
                                  stream);

    applyTemperature(h_logits, h_bias, temperature, batch_size, vocab_size, vocab_size_padded);
    std::string tag = "Correctness " + tc.toString() + (isHalf<T>() ? " (FP16)" : " (FP32)");
    bool passed = checkResult(tag, d_logits, h_logits, batch_size * vocab_size_padded);

    // Tear down test
    check_cuda_error(cudaStreamDestroy(stream));
    check_cuda_error(cudaFree(d_logits));
    check_cuda_error(cudaFree(d_bias));
    delete[] h_logits;
    delete[] h_bias;

    EXPECT_TRUE(passed);
}

template<typename T>
void testBatchApplyTemperaturePenaltyKernel(TemperatureTestCase tc) {
    // Set up test
    const size_t batch_size = tc.batch_size;
    const size_t vocab_size = tc.vocab_size;
    const size_t vocab_size_padded = pad_vocab_size(vocab_size);

    float* h_temperatures = new float[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        h_temperatures[i] = i % 2 == 0 ? tc.temperature : 0.1f * tc.temperature;
    }
    T* h_logits = new T[batch_size * vocab_size_padded];
    T* h_bias = new T[vocab_size_padded];
    initLogitsAndBias(h_logits, h_bias, batch_size, vocab_size, vocab_size_padded);

    float* d_temperatures;
    T* d_logits;
    T* d_bias;
    check_cuda_error(cudaMalloc(&d_temperatures, sizeof(float) * batch_size));
    check_cuda_error(cudaMalloc(&d_logits, sizeof(T) * batch_size * vocab_size_padded));
    check_cuda_error(cudaMalloc(&d_bias, sizeof(T) * vocab_size_padded));
    check_cuda_error(cudaMemcpy(d_temperatures, h_temperatures, sizeof(float) * batch_size, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_logits, h_logits, sizeof(T) * batch_size * vocab_size_padded, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_bias, h_bias, sizeof(T) * vocab_size_padded, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));

    // Do test
    invokeBatchApplyTemperaturePenalty(d_logits,
                                       d_bias,
                                       d_temperatures,
                                       batch_size,
                                       vocab_size,
                                       vocab_size_padded,
                                       stream);

    batchApplyTemperature(h_logits, h_bias, h_temperatures, batch_size, vocab_size, vocab_size_padded);
    std::string tag = "Correctness Batch " + tc.toString() + (isHalf<T>() ? " (FP16)" : " (FP32)");
    bool passed = checkResult(tag, d_logits, h_logits, batch_size * vocab_size_padded);

    // Tear down test
    check_cuda_error(cudaStreamDestroy(stream));
    check_cuda_error(cudaFree(d_logits));
    check_cuda_error(cudaFree(d_bias));
    check_cuda_error(cudaFree(d_temperatures));
    delete[] h_logits;
    delete[] h_bias;
    delete[] h_temperatures;

    EXPECT_TRUE(passed);
}

template<typename T>
void testConsistencyTemperaturePenaltyKernel(TemperatureTestCase tc) {
    // Set up test
    const size_t batch_size = tc.batch_size;
    const size_t vocab_size = tc.vocab_size;
    const size_t vocab_size_padded = pad_vocab_size(vocab_size);

    float temperature = tc.temperature;
    float* h_temperatures = new float[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        h_temperatures[i] = temperature;
    }
    T* h_logits = new T[batch_size * vocab_size_padded];
    T* h_bias = new T[vocab_size_padded];
    initLogitsAndBias(h_logits, h_bias, batch_size, vocab_size, vocab_size_padded);

    float* d_temperatures;
    check_cuda_error(cudaMalloc(&d_temperatures, sizeof(float) * batch_size));
    check_cuda_error(cudaMemcpy(d_temperatures, h_temperatures, sizeof(float) * batch_size, cudaMemcpyHostToDevice));

    T* d_logits_single;
    T* d_bias_single;
    check_cuda_error(cudaMalloc(&d_logits_single, sizeof(T) * batch_size * vocab_size_padded));
    check_cuda_error(cudaMalloc(&d_bias_single, sizeof(T) * vocab_size_padded));
    check_cuda_error(cudaMemcpy(d_logits_single, h_logits, sizeof(T) * batch_size * vocab_size_padded, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_bias_single, h_bias, sizeof(T) * vocab_size_padded, cudaMemcpyHostToDevice));

    T* d_logits_batch;
    T* d_bias_batch;
    check_cuda_error(cudaMalloc(&d_logits_batch, sizeof(T) * batch_size * vocab_size_padded));
    check_cuda_error(cudaMalloc(&d_bias_batch, sizeof(T) * vocab_size_padded));
    check_cuda_error(cudaMemcpy(d_logits_batch, h_logits, sizeof(T) * batch_size * vocab_size_padded, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_bias_batch, h_bias, sizeof(T) * vocab_size_padded, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));

    // Do test
    invokeApplyTemperaturePenalty(d_logits_single,
                                  d_bias_single,
                                  temperature,
                                  batch_size,
                                  vocab_size,
                                  vocab_size_padded,
                                  stream);
    invokeBatchApplyTemperaturePenalty(d_logits_batch,
                                       d_bias_batch,
                                       d_temperatures,
                                       batch_size,
                                       vocab_size,
                                       vocab_size_padded,
                                       stream);
    std::string tag = "Consistency " + tc.toString() + (isHalf<T>() ? " (FP16)" : " (FP32)");
    bool passed = checkResult(tag, d_logits_single, d_logits_batch, batch_size * vocab_size_padded, true, true);

    // Tear down test
    check_cuda_error(cudaStreamDestroy(stream));
    check_cuda_error(cudaFree(d_logits_single));
    check_cuda_error(cudaFree(d_bias_single));
    check_cuda_error(cudaFree(d_logits_batch));
    check_cuda_error(cudaFree(d_bias_batch));
    check_cuda_error(cudaFree(d_temperatures));
    delete[] h_logits;
    delete[] h_bias;
    delete[] h_temperatures;

    EXPECT_TRUE(passed);
}

template<typename T>
void testApplyRepetitonPenaltyKernel(RepetitionTestCase tc) {
    // Set up test
    const size_t batch_size = tc.batch_size;
    const size_t vocab_size = tc.vocab_size;
    const size_t vocab_size_padded = pad_vocab_size(vocab_size);
    const size_t max_input_length = tc.max_input_length;
    const size_t sequence_length = 2 * max_input_length;  // input + output
    const size_t step = sequence_length * 0.5;
    const float repetition_penalty = tc.repetition_penalty;
    T* h_logits = new T[batch_size * vocab_size_padded];
    int* h_output_ids = new int[sequence_length * batch_size];
    int* h_input_lengths = new int[batch_size];
    initLogitsAndBias(h_logits, (T*)nullptr, batch_size, vocab_size, vocab_size_padded);
    initRandomInt(h_output_ids, sequence_length * batch_size, 0, vocab_size);
    initRandomInt(h_input_lengths, batch_size, 1, max_input_length);

    T* d_logits;
    check_cuda_error(cudaMalloc(&d_logits, sizeof(T) * batch_size * vocab_size_padded));
    check_cuda_error(cudaMemcpy(d_logits, h_logits, sizeof(T) * batch_size * vocab_size_padded, cudaMemcpyHostToDevice));
    int* d_output_ids;
    check_cuda_error(cudaMalloc(&d_output_ids, sizeof(int) * sequence_length * batch_size));
    check_cuda_error(cudaMemcpy(d_output_ids, h_output_ids, sizeof(int) * sequence_length * batch_size, cudaMemcpyHostToDevice));
    int* d_input_lengths;
    check_cuda_error(cudaMalloc(&d_input_lengths, sizeof(int) * batch_size));
    check_cuda_error(cudaMemcpy(d_input_lengths, h_input_lengths, sizeof(int) * batch_size, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));

    // Do test
    invokeApplyRepetitionPenalty(d_logits,
                                 repetition_penalty,
                                 nullptr,
                                 d_output_ids,
                                 batch_size,
                                 batch_size,
                                 vocab_size,
                                 vocab_size_padded,
                                 d_input_lengths,
                                 max_input_length,
                                 step,
                                 stream);

    applyRepetitonPenalty(h_logits,
                          h_output_ids,
                          h_input_lengths,
                          repetition_penalty,
                          step,
                          max_input_length,
                          batch_size,
                          vocab_size,
                          vocab_size_padded);

    std::string tag = "Correctness " + tc.toString() + (isHalf<T>() ? " (FP16)" : " (FP32)");
    bool passed = checkResult(tag, d_logits, h_logits, batch_size * vocab_size_padded);

    // Tear down test
    check_cuda_error(cudaStreamDestroy(stream));
    check_cuda_error(cudaFree(d_logits));
    check_cuda_error(cudaFree(d_output_ids));
    check_cuda_error(cudaFree(d_input_lengths));
    delete[] h_logits;
    delete[] h_output_ids;
    delete[] h_input_lengths;

    EXPECT_TRUE(passed);
}

template<typename T>
void testBatchApplyRepetitonPenaltyKernel(RepetitionTestCase tc) {
    // Set up test
    const size_t batch_size = tc.batch_size;
    const size_t vocab_size = tc.vocab_size;
    const size_t vocab_size_padded = pad_vocab_size(vocab_size);
    const size_t max_input_length = tc.max_input_length;
    const size_t sequence_length = 2 * tc.max_input_length;
    const size_t step = sequence_length * 0.8;
    const float repetition_penalty = tc.repetition_penalty;
    float* h_repetition_penalties = new float[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        h_repetition_penalties[i] = i % 2 == 0 ? repetition_penalty : 0.1f * repetition_penalty;
    }

    T* h_logits = new T[batch_size * vocab_size_padded];
    int* h_output_ids = new int[sequence_length * batch_size];
    int* h_input_lengths = new int[batch_size];
    initLogitsAndBias(h_logits, (T*)nullptr, batch_size, vocab_size, vocab_size_padded);
    initRandomInt(h_output_ids, sequence_length * batch_size, 0, vocab_size);
    initRandomInt(h_input_lengths, batch_size, 1, max_input_length);

    T* d_logits;
    check_cuda_error(cudaMalloc(&d_logits, sizeof(T) * batch_size * vocab_size_padded));
    check_cuda_error(cudaMemcpy(d_logits, h_logits, sizeof(T) * batch_size * vocab_size_padded, cudaMemcpyHostToDevice));
    int* d_output_ids;
    check_cuda_error(cudaMalloc(&d_output_ids, sizeof(int) * sequence_length * batch_size));
    check_cuda_error(cudaMemcpy(d_output_ids, h_output_ids, sizeof(int) * sequence_length * batch_size, cudaMemcpyHostToDevice));
    int* d_input_lengths;
    check_cuda_error(cudaMalloc(&d_input_lengths, sizeof(int) * batch_size));
    check_cuda_error(cudaMemcpy(d_input_lengths, h_input_lengths, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    float* d_repetition_penalties;
    check_cuda_error(cudaMalloc(&d_repetition_penalties, sizeof(float) * batch_size));
    check_cuda_error(cudaMemcpy(d_repetition_penalties, h_repetition_penalties, sizeof(float) * batch_size, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));

    // Do test
    invokeBatchApplyRepetitionPenalty(d_logits,
                                      d_repetition_penalties,
                                      d_output_ids,
                                      batch_size,
                                      batch_size,
                                      vocab_size_padded,
                                      d_input_lengths,
                                      max_input_length,
                                      step,
                                      stream);

    batchApplyRepetitonPenalty(h_logits,
                               h_output_ids,
                               h_input_lengths,
                               h_repetition_penalties,
                               step,
                               max_input_length,
                               batch_size,
                               vocab_size,
                               vocab_size_padded);

    std::string tag = "Correctness Batch " + tc.toString() + (isHalf<T>() ? " (FP16)" : " (FP32)");
    bool passed = checkResult(tag, d_logits, h_logits, batch_size * vocab_size_padded);

    // Tear down test
    check_cuda_error(cudaStreamDestroy(stream));
    check_cuda_error(cudaFree(d_logits));
    check_cuda_error(cudaFree(d_output_ids));
    check_cuda_error(cudaFree(d_input_lengths));
    check_cuda_error(cudaFree(d_repetition_penalties));
    delete[] h_repetition_penalties;
    delete[] h_logits;
    delete[] h_output_ids;
    delete[] h_input_lengths;

    EXPECT_TRUE(passed);
}

template<typename T>
void testBatchApplyRepetitonPenaltyKernelWithLocalBatch(RepetitionTestCase tc) {
    // Set up test
    const size_t batch_size = tc.batch_size;
    if (batch_size % 2 != 0) {
        FT_LOG_WARNING("Skip testApplyRepetitonPenaltyKernelWithLocalBatch (batch_size % 2 != 0).");
        return;
    }
    const size_t local_batch_size = batch_size / 2;
    const size_t vocab_size = tc.vocab_size;
    const size_t vocab_size_padded = pad_vocab_size(vocab_size);
    const size_t max_input_length = tc.max_input_length;
    const size_t sequence_length = 2 * tc.max_input_length; // input + output
    const size_t step = sequence_length * 0.8;
    const float repetition_penalty = tc.repetition_penalty;
    float* h_repetition_penalties = new float[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        h_repetition_penalties[i] = i % 2 == 0 ? repetition_penalty : 0.1f * repetition_penalty;
    }

    T* h_logits = new T[batch_size * vocab_size_padded];
    int* h_output_ids = new int[sequence_length * batch_size];
    int* h_input_lengths = new int[batch_size];
    initLogitsAndBias(h_logits, (T*)nullptr, batch_size, vocab_size, vocab_size_padded);
    initRandomInt(h_output_ids, sequence_length * batch_size, 0, vocab_size);
    initRandomInt(h_input_lengths, batch_size, 1, max_input_length);

    T* d_logits;
    check_cuda_error(cudaMalloc(&d_logits, sizeof(T) * batch_size * vocab_size_padded));
    check_cuda_error(cudaMemcpy(d_logits, h_logits, sizeof(T) * batch_size * vocab_size_padded, cudaMemcpyHostToDevice));
    int* d_output_ids;
    check_cuda_error(cudaMalloc(&d_output_ids, sizeof(int) * sequence_length * batch_size));
    check_cuda_error(cudaMemcpy(d_output_ids, h_output_ids, sizeof(int) * sequence_length * batch_size, cudaMemcpyHostToDevice));
    int* d_input_lengths;
    check_cuda_error(cudaMalloc(&d_input_lengths, sizeof(int) * batch_size));
    check_cuda_error(cudaMemcpy(d_input_lengths, h_input_lengths, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    float* d_repetition_penalties;
    check_cuda_error(cudaMalloc(&d_repetition_penalties, sizeof(float) * batch_size));
    check_cuda_error(cudaMemcpy(d_repetition_penalties, h_repetition_penalties, sizeof(float) * batch_size, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));

    // Do test
    int ite = 1;
    invokeBatchApplyRepetitionPenalty(d_logits + ite * local_batch_size * vocab_size_padded,
                                      d_repetition_penalties + ite * local_batch_size,
                                      d_output_ids + ite * local_batch_size,
                                      batch_size,
                                      local_batch_size,
                                      vocab_size_padded,
                                      d_input_lengths + ite * local_batch_size,
                                      max_input_length,
                                      step,
                                      stream);
    batchApplyRepetitonPenalty(h_logits,
                               h_output_ids,
                               h_input_lengths,
                               h_repetition_penalties,
                               step,
                               max_input_length,
                               batch_size,
                               vocab_size,
                               vocab_size_padded);

    std::string tag = "Correctness (local batch) " + tc.toString() + (isHalf<T>() ? " (FP16)" : " (FP32)");
    bool passed = checkResult(tag,
                              d_logits + ite * local_batch_size * vocab_size_padded,
                              h_logits + ite * local_batch_size * vocab_size_padded,
                              local_batch_size * vocab_size_padded);

    // Tear down test
    check_cuda_error(cudaStreamDestroy(stream));
    check_cuda_error(cudaFree(d_logits));
    check_cuda_error(cudaFree(d_output_ids));
    check_cuda_error(cudaFree(d_input_lengths));
    check_cuda_error(cudaFree(d_repetition_penalties));
    delete[] h_repetition_penalties;
    delete[] h_logits;
    delete[] h_output_ids;
    delete[] h_input_lengths;

    EXPECT_TRUE(passed);
}

template<typename T>
void testConsistencyRepetitionPenaltyKernel(RepetitionTestCase tc) {
    // Set up test
    const size_t batch_size = tc.batch_size;
    const size_t vocab_size = tc.vocab_size;
    const size_t vocab_size_padded = pad_vocab_size(vocab_size);
    const size_t max_input_length = tc.max_input_length;
    const size_t sequence_length = 2 * max_input_length;
    const size_t step = max_input_length * 0.8;
    const float repetition_penalty = tc.repetition_penalty;
    float* h_repetition_penalties = new float[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        h_repetition_penalties[i] = repetition_penalty;
    }

    T* h_logits = new T[batch_size * vocab_size_padded];
    int* h_output_ids = new int[sequence_length * batch_size];
    int* h_input_lengths = new int[batch_size];
    initLogitsAndBias(h_logits, (T*)nullptr, batch_size, vocab_size, vocab_size_padded);
    initRandomInt(h_output_ids, sequence_length * batch_size, 0, vocab_size);
    initRandomInt(h_input_lengths, batch_size, 1, max_input_length);

    T* d_logits_single;
    check_cuda_error(cudaMalloc(&d_logits_single, sizeof(T) * batch_size * vocab_size_padded));
    check_cuda_error(cudaMemcpy(d_logits_single, h_logits, sizeof(T) * batch_size * vocab_size_padded, cudaMemcpyHostToDevice));
    T* d_logits_batch;
    check_cuda_error(cudaMalloc(&d_logits_batch, sizeof(T) * batch_size * vocab_size_padded));
    check_cuda_error(cudaMemcpy(d_logits_batch, h_logits, sizeof(T) * batch_size * vocab_size_padded, cudaMemcpyHostToDevice));

    int* d_output_ids;
    check_cuda_error(cudaMalloc(&d_output_ids, sizeof(int) * sequence_length * batch_size));
    check_cuda_error(cudaMemcpy(d_output_ids, h_output_ids, sizeof(int) * sequence_length * batch_size, cudaMemcpyHostToDevice));
    int* d_input_lengths;
    check_cuda_error(cudaMalloc(&d_input_lengths, sizeof(int) * batch_size));
    check_cuda_error(cudaMemcpy(d_input_lengths, h_input_lengths, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    float* d_repetition_penalties;
    check_cuda_error(cudaMalloc(&d_repetition_penalties, sizeof(float) * batch_size));
    check_cuda_error(cudaMemcpy(d_repetition_penalties, h_repetition_penalties, sizeof(float) * batch_size, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));

    // Do test
    invokeApplyRepetitionPenalty(d_logits_single,
                                 repetition_penalty,
                                 nullptr,
                                 d_output_ids,
                                 batch_size,
                                 batch_size,
                                 vocab_size,
                                 vocab_size_padded,
                                 d_input_lengths,
                                 max_input_length,
                                 step,
                                 stream);

    invokeBatchApplyRepetitionPenalty(d_logits_batch,
                                      d_repetition_penalties,
                                      d_output_ids,
                                      batch_size,
                                      batch_size,
                                      vocab_size_padded,
                                      d_input_lengths,
                                      max_input_length,
                                      step,
                                      stream);

    std::string tag = "Consistency " + tc.toString() + (isHalf<T>() ? " (FP16)" : " (FP32)");
    bool passed = checkResult(tag, d_logits_single, d_logits_batch, batch_size * vocab_size_padded, true, true);

    // Tear down test
    check_cuda_error(cudaStreamDestroy(stream));
    check_cuda_error(cudaFree(d_logits_single));
    check_cuda_error(cudaFree(d_logits_batch));
    check_cuda_error(cudaFree(d_output_ids));
    check_cuda_error(cudaFree(d_input_lengths));
    check_cuda_error(cudaFree(d_repetition_penalties));
    delete[] h_logits;
    delete[] h_output_ids;
    delete[] h_repetition_penalties;
    delete[] h_input_lengths;
    EXPECT_TRUE(passed);
}

template<typename T>
void testBeamPenaltyKernelCorrectness() {
    // Set up test
    const size_t batch_size = 2;
    const size_t beam_width = 3;
    const size_t batchxbeam = batch_size * beam_width;
    const size_t vocab_size = 4;
    const size_t vocab_size_padded = 8;
    const size_t max_input_length = 2;
    const size_t local_batch_size = batch_size;
    const int ite = 0;
    const int step = 4;
    assert(step > max_input_length);
    int* h_end_ids = new int[batch_size]{0, 2};
    int* h_input_lengths = new int[batchxbeam]{2, 2, 2, 2, 2, 2};
    const T MASK_VAL = static_cast<T>(isHalf<T>() ? -65504.f : -FLT_MAX);
    T* h_logits = new T[batchxbeam * vocab_size_padded]{
         4.0f, -2.0f,  5.0f,  9.0f, MASK_VAL, MASK_VAL, MASK_VAL, MASK_VAL,
         4.0f, -2.0f,  5.0f,  9.0f, MASK_VAL, MASK_VAL, MASK_VAL, MASK_VAL,
         4.0f, -2.0f,  5.0f,  9.0f, MASK_VAL, MASK_VAL, MASK_VAL, MASK_VAL,
        -2.0f,  1.0f, -3.0f, -2.0f, MASK_VAL, MASK_VAL, MASK_VAL, MASK_VAL,
        -2.0f,  1.0f, -3.0f, -2.0f, MASK_VAL, MASK_VAL, MASK_VAL, MASK_VAL,
        -2.0f,  1.0f, -3.0f, -2.0f, MASK_VAL, MASK_VAL, MASK_VAL, MASK_VAL
    };
    T* h_bias = new T[vocab_size_padded]{
        0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    int* h_previous_ids = new int[(step - 1) * batchxbeam]{
        3, 3, 2, 3, 0, 2, // step 0 [b1 b1 b1 b2 b2 b2]
        1, 2, 1, 1, 1, 2, // step 1 [b1 b1 b1 b2 b2 b2]
        2, 0, 1, 1, 2, 1, // step 2
    };
    int* h_current_ids = new int[batchxbeam]{0, 3, 1, 0, 0, 2};  // step 3.
    int* h_parent_ids = new int[(step - 1) * batchxbeam]{
        0, 1, 2, 0, 1, 1,  // step 0 [b1 b1 b1 b2 b2 b2]
        0, 2, 2, 2, 1, 1,  // step 1 [b1 b1 b1 b2 b2 b2]
        2, 0, 1, 2, 2, 1,  // step 2
    };
    // final output sequence [batch, beam]
    //   [0, 0]: 2 1 1 0
    //   [0, 1]: 3 1 2 3
    //   [0, 2]: 2 1 0 1
    //   [1, 0]: 0 1 1 0
    //   [1, 1]: 0 1 2 0
    //   [1, 2]: 0 1 2 2

    float temperature = 2.0f;
    float repetition_penalty = 2.0f;

    T* h_expected = new T[batchxbeam * vocab_size_padded]{
         1.0f, -2.0f,  1.5f,  4.0f, MASK_VAL, MASK_VAL, MASK_VAL, MASK_VAL,
         2.0f, -2.0f,  1.5f,  2.0f, MASK_VAL, MASK_VAL, MASK_VAL, MASK_VAL,
         1.0f, -2.0f,  1.5f,  4.0f, MASK_VAL, MASK_VAL, MASK_VAL, MASK_VAL,
        -2.0f, 0.25f, -1.0f, -1.5f, MASK_VAL, MASK_VAL, MASK_VAL, MASK_VAL,
        -2.0f, 0.25f, -1.0f, -1.5f, MASK_VAL, MASK_VAL, MASK_VAL, MASK_VAL,
        -2.0f, 0.25f, -2.0f, -1.5f, MASK_VAL, MASK_VAL, MASK_VAL, MASK_VAL
    };

    T *d_logits, *d_bias;
    check_cuda_error(cudaMalloc(&d_logits, sizeof(T) * batchxbeam * vocab_size_padded));
    check_cuda_error(cudaMemcpy(
        d_logits, h_logits, sizeof(T) * batchxbeam * vocab_size_padded, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMalloc(&d_bias, sizeof(T) * vocab_size_padded));
    check_cuda_error(cudaMemcpy(d_bias, h_bias, sizeof(T) * vocab_size_padded, cudaMemcpyHostToDevice));
    int *d_previous_ids, *d_current_ids, *d_parent_ids;
    check_cuda_error(cudaMalloc(&d_previous_ids, sizeof(int) * (step - 1) * batchxbeam));
    check_cuda_error(cudaMemcpy(
        d_previous_ids, h_previous_ids, sizeof(int) * (step - 1) * batchxbeam, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMalloc(&d_current_ids, sizeof(int) * batchxbeam));
    check_cuda_error(cudaMemcpy(
        d_current_ids, h_current_ids, sizeof(int) * batchxbeam, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMalloc(&d_parent_ids, sizeof(int) * (step - 1) * batchxbeam));
    check_cuda_error(cudaMemcpy(
        d_parent_ids, h_parent_ids, sizeof(int) * (step - 1) * batchxbeam, cudaMemcpyHostToDevice));
    int *d_end_ids;
    check_cuda_error(cudaMalloc(&d_end_ids, sizeof(int) * batch_size));
    check_cuda_error(cudaMemcpy(d_end_ids, h_end_ids, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    int* d_input_lengths;
    check_cuda_error(cudaMalloc(&d_input_lengths, sizeof(int) * batchxbeam));
    check_cuda_error(cudaMemcpy(
        d_input_lengths, h_input_lengths, sizeof(int) * batchxbeam, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));

    // Do test
    invokeAddBiasApplyPenalties(step,
                                d_logits + ite * vocab_size_padded,
                                d_current_ids,
                                d_previous_ids,
                                d_parent_ids, // + ite * local_batch_size * beam_width,
                                d_input_lengths + ite * local_batch_size * beam_width,
                                d_bias,
                                ite,
                                max_input_length,
                                local_batch_size,
                                batch_size,
                                beam_width,
                                vocab_size,
                                vocab_size_padded,
                                d_end_ids,
                                temperature,
                                repetition_penalty,
                                stream);
    std::string tag = std::string("Beamsearch Penalty Kernel Correctness")
                      + (isHalf<T>() ? " (FP16)" : " (FP32)");
    bool passed = checkResult(tag, d_logits, h_expected, batchxbeam * vocab_size_padded);

    // Tear down test
    check_cuda_error(cudaStreamDestroy(stream));
    check_cuda_error(cudaFree(d_logits));
    check_cuda_error(cudaFree(d_bias));
    check_cuda_error(cudaFree(d_current_ids));
    check_cuda_error(cudaFree(d_previous_ids));
    check_cuda_error(cudaFree(d_parent_ids));
    check_cuda_error(cudaFree(d_input_lengths));
    check_cuda_error(cudaFree(d_end_ids));
    delete[] h_logits;
    delete[] h_bias;
    delete[] h_current_ids;
    delete[] h_previous_ids;
    delete[] h_parent_ids;
    delete[] h_input_lengths;
    delete[] h_end_ids;
    EXPECT_TRUE(passed);
}

int main() {
    std::vector<TemperatureTestCase> temperature_test_cases {
        // TC: name / batch / vocab / temperature / repetition
        {6,  4, 0.53f},
        {6,  4, 1.0f},
        {6,  4, 2.01f},
        {6,  50001, 2.01f},
        {128,  51200, 2.01f}
    };

    for (auto &tc : temperature_test_cases) {
        testApplyTemperaturePenaltyKernel<float>(tc);
        testApplyTemperaturePenaltyKernel<half>(tc);
        testBatchApplyTemperaturePenaltyKernel<float>(tc);
        testBatchApplyTemperaturePenaltyKernel<half>(tc);
        testConsistencyTemperaturePenaltyKernel<float>(tc);
        testConsistencyTemperaturePenaltyKernel<half>(tc);
    }
    FT_LOG_INFO("test TemperaturePenaltyKernel done");

    std::vector<RepetitionTestCase> repetition_test_cases {
        {6, 4, 10, 0.53f},
        {6, 4, 10, 1.0f},
        {6, 4, 10, 2.01f},
        {6, 50001, 10, 2.01f},
        {128, 51200, 1024, 2.01f},
        {128, 51200, 2048, 2.01f}
    };
    for (auto& tc : repetition_test_cases) {
        testApplyRepetitonPenaltyKernel<float>(tc);
        testApplyRepetitonPenaltyKernel<half>(tc);
        testBatchApplyRepetitonPenaltyKernel<float>(tc);
        testBatchApplyRepetitonPenaltyKernel<half>(tc);
        testBatchApplyRepetitonPenaltyKernelWithLocalBatch<float>(tc);
        testBatchApplyRepetitonPenaltyKernelWithLocalBatch<half>(tc);
        testConsistencyRepetitionPenaltyKernel<float>(tc);
        testConsistencyRepetitionPenaltyKernel<half>(tc);
    }
    FT_LOG_INFO("test RepetitionPenaltyKernel done");

    testBeamPenaltyKernelCorrectness<float>();
    testBeamPenaltyKernelCorrectness<half>();
    FT_LOG_INFO("test BeamPenaltyKernelCorrectness done");

    return 0;
}
