#include <algorithm>   // std::min, std::max
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include "src/fastertransformer/kernels/sampling_topk_kernels.h"
#include "src/fastertransformer/kernels/sampling_topp_kernels.h"
#include "src/fastertransformer/layers/DynamicDecodeLayer.h"
#include "src/fastertransformer/layers/sampling_layers/TopKSamplingLayer.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/Tensor.h"

#include "tests/unittests/unittest_utils.h"

using namespace fastertransformer;

struct TestCase {
    std::string name;
    size_t batch_size;
    size_t vocab_size;
    size_t beam_width;
    size_t top_k;
    float top_p;
    size_t output_len;

    std::string toString() {
        char buf[100];
        snprintf(buf, sizeof(buf),
                 "TestCase[name=%s, batch=%ld, vocab=%ld, beam=%ld, k=%ld, p=%3.1f, output_len=%ld]",
                 name.c_str(), batch_size, vocab_size, beam_width, top_k, top_p, output_len);
        return buf;
    }

    void print() {
        FT_LOG_INFO(toString());
    }
};

template<typename T>
void computeProb(T* probs, T* logits, int batch_size, int vocab_size) {
    // Compute the log probability from logits.
    //   logits = batch_size x vocab_size vector.
    //   logprobs = log(softmax(logits)) (softmax along with vocab dimension)
    for (int bidx = 0; bidx < batch_size; ++bidx) {
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            sum += expf((float)logits[bidx * vocab_size + i]);
        }
        for (int i = 0; i < vocab_size; ++i) {
            int idx = bidx * vocab_size + i;
            probs[idx] = static_cast<T>(expf((float)logits[idx]) / (sum + EPSILON));
        }
    }
}

template<typename T>
void computeLogProb(T* logprobs, T* logits, int batch_size, int vocab_size) {
    // Compute the log probability from logits.
    //   logits = batch_size x vocab_size vector.
    //   logprobs = log(softmax(logits)) (softmax along with vocab dimension)
    for (int bidx = 0; bidx < batch_size; ++bidx) {
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            sum += expf(logits[bidx * vocab_size + i]);
        }
        for (int i = 0; i < vocab_size; ++i) {
            int idx = bidx * vocab_size + i;
            logprobs[idx] = static_cast<T>(logf(expf(logits[idx]) / (sum + EPSILON) + EPSILON));
        }
    }
}

std::string toTestTag(std::string name, TestCase tc, bool is_fp32) {
    return name + " " + tc.toString() + (is_fp32 ? " (fp32)" : " (fp16)");
}

/////////////////////////////////// Tests //////////////////////////////////////////

template<typename T>
void testTopKSamplingKernel(TestCase tc) {

    bool is_fp32 = std::is_same<T, float>::value;

    size_t top_k = tc.top_k;
    unsigned long long seed = 0;

    size_t batch_size = tc.batch_size;
    size_t vocab_size = tc.vocab_size;

    int end_id = 3;
    size_t max_output_len = tc.output_len;
    size_t max_seq_len = max_output_len;

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));
    Allocator<AllocatorType::CUDA>* allocator = new Allocator<AllocatorType::CUDA>(getDevice());
    allocator->setStream(stream);

    // Logit values in the host of shape (batch_size x vocab_size).
    T* h_logits = new T[batch_size * vocab_size];
    T* h_probs = new T[batch_size * vocab_size];
    T* h_log_probs = new T[batch_size * vocab_size];
    float* h_cum_log_probs = new float[batch_size];
    float* h_output_log_probs = new float[batch_size];
    float* expected_cum_log_probs = new float[batch_size];
    int* h_output_ids = new int[batch_size];
    int* h_seq_lengths = new int[batch_size];
    bool* h_finished = new bool[batch_size];

    initRandom(h_logits, batch_size * vocab_size, -10.0f, -1.0f);
    memset(expected_cum_log_probs, 0, sizeof(float) * batch_size);

    curandState_t* curand_states = reinterpret_cast<curandState_t*>(
        allocator->malloc(sizeof(curandState_t) * batch_size, false));
    invokeCurandInitialize(curand_states, batch_size, seed, stream);

    size_t workspace_size = 0;
    // retrieve the workspace size of the top-k sampling kernel.
    invokeTopKSampling<T>(nullptr,
                          workspace_size,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          top_k,
                          1.0f,
                          vocab_size,
                          nullptr,
                          stream,
                          batch_size,
                          nullptr);
    void* workspace = allocator->malloc(workspace_size, false);
    int* sequence_lengths = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
    bool* finished = reinterpret_cast<bool*>(allocator->malloc(sizeof(bool) * batch_size));
    int* end_ids = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size, false));

    T* probs = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batch_size * vocab_size, true));
    float* cum_log_probs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));
    float* output_log_probs = reinterpret_cast<float*>(
        allocator->malloc(sizeof(float) * max_output_len * batch_size));
    int* output_ids = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * max_seq_len * batch_size));

    // Init by zero.
    deviceFill(sequence_lengths, batch_size, 0);
    deviceFill(finished, batch_size, false);
    deviceFill(end_ids, batch_size, end_id);

    deviceFill(cum_log_probs, batch_size, 0.0f);
    deviceFill(output_log_probs, max_output_len * batch_size, 0.0f);
    deviceFill(output_ids, max_seq_len * batch_size, 0);

    void* h_worksapce = malloc(workspace_size);

    for (size_t step = 0; step < max_output_len; ++step) {
        initRandom(h_logits, batch_size * vocab_size, -10.0f, -1.0f);
        computeProb(h_probs, h_logits, batch_size, vocab_size);
        cudaH2Dcpy(probs, h_probs, batch_size * vocab_size);
        invokeTopKSampling(workspace,
                           workspace_size,
                           // Note that the kernel needs vocab probs instead of
                           // log-prob if cum_log_probs or output_log_probs are
                           // provided. It's because the sampling layer already
                           // preprocesses log_prob_buf when those are provided.
                           probs,
                           output_ids + step * batch_size,
                           sequence_lengths,
                           finished,
                           cum_log_probs,
                           output_log_probs + step * batch_size,
                           curand_states,
                           top_k,
                           1.0f,
                           vocab_size,
                           end_ids,
                           stream,
                           batch_size,
                           nullptr);

        // Compute reference.
        cudaD2Hcpy(h_output_ids, output_ids + step * batch_size, batch_size);
        cudaD2Hcpy(h_output_log_probs, output_log_probs + step * batch_size, batch_size);
        cudaD2Hcpy(h_cum_log_probs, cum_log_probs, batch_size);
        cudaD2Hcpy(h_seq_lengths, sequence_lengths, batch_size);
        cudaD2Hcpy(h_finished, finished, batch_size);
        computeLogProb(h_log_probs, h_logits, batch_size, vocab_size);
        for (size_t i = 0; i < batch_size; ++i) {
            int idx = i * vocab_size + h_output_ids[i];
            bool expected_finished = h_output_ids[i] == end_id;
            float expected_log_prob = (int)step < h_seq_lengths[i] ? (float)h_log_probs[idx] : 0.0f;
            expected_cum_log_probs[i] += expected_log_prob;
            EXPECT_TRUE(h_finished[i] == expected_finished);
        }
    }
    std::string tag = toTestTag("TestTopKSamplingKernel", tc, is_fp32);
    bool passed = checkResult(tag, cum_log_probs, expected_cum_log_probs, batch_size);
    EXPECT_TRUE(passed);

    delete[] expected_cum_log_probs;
    delete[] h_seq_lengths;
    delete[] h_output_log_probs;
    delete[] h_cum_log_probs;
    delete[] h_logits;
    delete[] h_log_probs;
    delete[] h_probs;
    delete[] h_output_ids;
    delete allocator;
    check_cuda_error(cudaStreamDestroy(stream));
}

template<typename T>
void testBatchTopKSamplingKernel(TestCase tc, bool has_diff_runtime_args) {

    bool is_fp32 = std::is_same<T, float>::value;

    unsigned long long seed = 0;

    size_t batch_size = tc.batch_size;
    size_t vocab_size = tc.vocab_size;

    int top_k = (int)tc.top_k;
    int* h_top_ks = new int[batch_size];
    // Initialize runtime top k values.
    for (size_t i = 0; i < batch_size; ++i) {
        h_top_ks[i] = has_diff_runtime_args ? std::max(1, top_k - int(i % 3)) : top_k;
    }
    int max_top_k = *std::max_element(h_top_ks, h_top_ks + batch_size);
    int end_id = 3;
    size_t max_output_len = tc.output_len;
    size_t max_seq_len = max_output_len;

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));
    Allocator<AllocatorType::CUDA>* allocator = new Allocator<AllocatorType::CUDA>(getDevice());
    allocator->setStream(stream);

    // Logit values in the host of shape (batch_size x vocab_size).
    T* h_logits = new T[batch_size * vocab_size];
    T* h_probs = new T[batch_size * vocab_size];
    T* h_log_probs = new T[batch_size * vocab_size];
    float* h_cum_log_probs = new float[batch_size];
    float* h_output_log_probs = new float[batch_size];
    float* expected_cum_log_probs = new float[batch_size];
    int* h_output_ids = new int[batch_size];
    int* h_seq_lengths = new int[batch_size];
    bool* h_finished = new bool[batch_size];

    initRandom(h_logits, batch_size * vocab_size, -10.0f, -1.0f);
    memset(expected_cum_log_probs, 0, sizeof(float) * batch_size);

    curandState_t* curand_states = reinterpret_cast<curandState_t*>(
        allocator->malloc(sizeof(curandState_t) * batch_size, false));
    invokeCurandInitialize(curand_states, batch_size, seed, stream);

    size_t workspace_size = 0;
    // retrieve the workspace size of the top-k sampling kernel.
    invokeBatchTopKSampling<T>(nullptr,  // workspace
                               workspace_size,
                               nullptr,  // log_probs
                               nullptr,  // ids
                               nullptr,  // sequence_lengths
                               nullptr,  // finished
                               nullptr,  // cum_log_probs
                               nullptr,  // output_log_probs
                               nullptr,  // curandstates
                               max_top_k,
                               nullptr,  // top_ks
                               1.0f,
                               nullptr,
                               vocab_size,
                               nullptr,  // end_ids
                               stream,
                               batch_size,
                               nullptr);

    void* workspace = allocator->malloc(workspace_size, false);
    int* top_ks = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
    int* end_ids = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
    int* sequence_lengths = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
    bool* finished = reinterpret_cast<bool*>(allocator->malloc(sizeof(bool) * batch_size));
    T* probs = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batch_size * vocab_size, true));
    float* cum_log_probs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));
    float* output_log_probs = reinterpret_cast<float*>(
        allocator->malloc(sizeof(float) * max_output_len * batch_size));
    int* output_ids = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * max_seq_len * batch_size));

    // Initialize.
    cudaH2Dcpy(top_ks, h_top_ks, batch_size);
    deviceFill(end_ids, batch_size, end_id);
    deviceFill(sequence_lengths, batch_size, 0);
    deviceFill(finished, batch_size, false);
    deviceFill(cum_log_probs, batch_size, 0.0f);
    deviceFill(output_log_probs, max_output_len * batch_size, 0.0f);
    deviceFill(output_ids, max_seq_len * batch_size, 0);

    for (size_t step = 0; step < max_output_len; ++step) {
        initRandom(h_logits, batch_size * vocab_size, -10.0f, -1.0f);
        computeProb(h_probs, h_logits, batch_size, vocab_size);
        cudaH2Dcpy(probs, h_probs, batch_size * vocab_size);

        invokeBatchTopKSampling(workspace,
                                workspace_size,
                                // Note that the kernel needs vocab probs instead of
                                // log-prob if cum_log_probs or output_log_probs are
                                // provided. It's because the sampling layer already
                                // preprocesses log_prob_buf when those are provided.
                                probs,
                                output_ids + step * batch_size,
                                sequence_lengths,
                                finished,
                                cum_log_probs,
                                output_log_probs + step * batch_size,
                                curand_states,
                                max_top_k,
                                top_ks,
                                1.0f,
                                nullptr,
                                vocab_size,
                                end_ids,
                                stream,
                                batch_size,
                                nullptr);

        // Compute reference.
        cudaD2Hcpy(h_output_ids, output_ids + step * batch_size, batch_size);
        cudaD2Hcpy(h_output_log_probs, output_log_probs + step * batch_size, batch_size);
        cudaD2Hcpy(h_cum_log_probs, cum_log_probs, batch_size);
        cudaD2Hcpy(h_seq_lengths, sequence_lengths, batch_size);
        cudaD2Hcpy(h_finished, finished, batch_size);
        computeLogProb(h_log_probs, h_logits, batch_size, vocab_size);
        for (size_t i = 0; i < batch_size; ++i) {
            int idx = i * vocab_size + h_output_ids[i];
            bool expected_finished = h_output_ids[i] == end_id;
            float expected_log_prob = (int)step < h_seq_lengths[i] ? (float)h_log_probs[idx] : 0.0f;
            expected_cum_log_probs[i] += expected_log_prob;
            EXPECT_TRUE(h_finished[i] == expected_finished);
        }
    }
    std::string tag = toTestTag("TestBatchTopKSamplingKernel", tc, is_fp32)
        + (has_diff_runtime_args ? " (diff_args)" : "");
    bool passed = checkResult(tag, cum_log_probs, expected_cum_log_probs, batch_size);
    EXPECT_TRUE(passed);

    delete[] expected_cum_log_probs;
    delete[] h_seq_lengths;
    delete[] h_output_log_probs;
    delete[] h_cum_log_probs;
    delete[] h_logits;
    delete[] h_log_probs;
    delete[] h_probs;
    delete[] h_output_ids;
    delete[] h_top_ks;
    delete allocator;
    check_cuda_error(cudaStreamDestroy(stream));
}

template<typename T>
void testBatchTopKSamplingWithSkipDecode(TestCase tc) {

    bool is_fp32 = std::is_same<T, float>::value;

    unsigned long long seed = 0;

    size_t batch_size = tc.batch_size;
    size_t vocab_size = tc.vocab_size;

    int top_k = (int)tc.top_k;
    int* h_top_ks = new int[batch_size];
    // Initialize runtime top k values.
    for (size_t i = 0; i < batch_size; ++i) {
        h_top_ks[i] = i % 3 == 0 ? top_k : 1;
    }
    int max_top_k = *std::max_element(h_top_ks, h_top_ks + batch_size);
    int end_id = 0;
    size_t max_output_len = tc.output_len;
    size_t max_seq_len = max_output_len;

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));
    Allocator<AllocatorType::CUDA>* allocator = new Allocator<AllocatorType::CUDA>(getDevice());
    allocator->setStream(stream);

    // Logit values in the host of shape (batch_size x vocab_size).
    T* h_logits = new T[batch_size * vocab_size];
    T* h_probs = new T[batch_size * vocab_size];
    T* h_log_probs = new T[batch_size * vocab_size];
    float* h_cum_log_probs = new float[batch_size];
    float* h_output_log_probs = new float[batch_size];
    float* expected_cum_log_probs = new float[batch_size];
    int* h_output_ids = new int[batch_size];
    int* h_seq_lengths = new int[batch_size];
    bool* h_finished = new bool[batch_size];
    bool* h_skip_decode = new bool[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        h_skip_decode[i] = i % 2 == 0;
    }

    initRandom(h_logits, batch_size * vocab_size, -3.0f, 3.0f);
    memset(expected_cum_log_probs, 0, sizeof(float) * batch_size);

    curandState_t* curand_states = reinterpret_cast<curandState_t*>(
        allocator->malloc(sizeof(curandState_t) * batch_size, false));
    invokeCurandInitialize(curand_states, batch_size, seed, stream);

    size_t workspace_size = 0;
    // retrieve the workspace size of the top-k sampling kernel.
    invokeBatchTopKSampling<T>(nullptr,  // workspace
                               workspace_size,
                               nullptr,  // log_probs
                               nullptr,  // ids
                               nullptr,  // sequence_lengths
                               nullptr,  // finished
                               nullptr,  // cum_log_probs
                               nullptr,  // output_log_probs
                               nullptr,  // curandstates
                               max_top_k,
                               nullptr,  // top_ks
                               1.0f,
                               nullptr,
                               vocab_size,
                               nullptr,  // end_ids
                               stream,
                               batch_size,
                               nullptr);

    void* workspace = allocator->malloc(workspace_size, false);
    int* top_ks = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
    int* end_ids = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
    int* sequence_lengths = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
    bool* finished = reinterpret_cast<bool*>(allocator->malloc(sizeof(bool) * batch_size));
    T* probs = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batch_size * vocab_size, true));
    float* cum_log_probs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));
    float* output_log_probs = reinterpret_cast<float*>(
        allocator->malloc(sizeof(float) * max_output_len * batch_size));
    int* output_ids = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * max_seq_len * batch_size));
    bool* skip_decode = reinterpret_cast<bool*>(allocator->malloc(sizeof(bool) * batch_size));
    cudaH2Dcpy(skip_decode, h_skip_decode, batch_size);

    // Initialize.
    cudaH2Dcpy(top_ks, h_top_ks, batch_size);
    deviceFill(end_ids, batch_size, end_id);
    deviceFill(sequence_lengths, batch_size, 0);
    deviceFill(finished, batch_size, false);
    deviceFill(cum_log_probs, batch_size, 0.0f);
    deviceFill(output_log_probs, max_output_len * batch_size, 0.0f);
    deviceFill(output_ids, max_seq_len * batch_size, 0);

    for (size_t step = 0; step < max_output_len; ++step) {
        initRandom(h_logits, batch_size * vocab_size, -10.0f, -1.0f);
        computeProb(h_probs, h_logits, batch_size, vocab_size);
        cudaH2Dcpy(probs, h_probs, batch_size * vocab_size);

        invokeBatchTopKSampling(workspace,
                                workspace_size,
                                // Note that the kernel needs vocab probs instead of
                                // log-prob if cum_log_probs or output_log_probs are
                                // provided. It's because the sampling layer already
                                // preprocesses log_prob_buf when those are provided.
                                probs,
                                output_ids + step * batch_size,
                                sequence_lengths,
                                finished,
                                cum_log_probs,
                                output_log_probs + step * batch_size,
                                curand_states,
                                max_top_k,
                                top_ks,
                                1.0f,
                                nullptr,
                                vocab_size,
                                end_ids,
                                stream,
                                batch_size,
                                skip_decode);

        // Compute reference.
        cudaD2Hcpy(h_output_ids, output_ids + step * batch_size, batch_size);
        cudaD2Hcpy(h_output_log_probs, output_log_probs + step * batch_size, batch_size);
        cudaD2Hcpy(h_cum_log_probs, cum_log_probs, batch_size);
        cudaD2Hcpy(h_seq_lengths, sequence_lengths, batch_size);
        cudaD2Hcpy(h_finished, finished, batch_size);
        computeLogProb(h_log_probs, h_logits, batch_size, vocab_size);
        for (size_t i = 0; i < batch_size; ++i) {
            if (!h_skip_decode[i]) {
                int idx = i * vocab_size + h_output_ids[i];
                bool expected_finished = h_output_ids[i] == end_id;
                float expected_log_prob = (int)step < h_seq_lengths[i] ? (float)h_log_probs[idx] : 0.0f;
                expected_cum_log_probs[i] += expected_log_prob;
                EXPECT_TRUE(h_finished[i] == expected_finished);
            }
        }
    }
    std::string tag = toTestTag("TestBatchTopKSamplingWithSkip", tc, is_fp32);
    bool passed = checkResult(tag, cum_log_probs, expected_cum_log_probs, batch_size);
    EXPECT_TRUE(passed);

    delete[] expected_cum_log_probs;
    delete[] h_seq_lengths;
    delete[] h_output_log_probs;
    delete[] h_cum_log_probs;
    delete[] h_logits;
    delete[] h_log_probs;
    delete[] h_probs;
    delete[] h_output_ids;
    delete[] h_top_ks;
    delete allocator;
    check_cuda_error(cudaStreamDestroy(stream));
}

template<typename T>
inline T clip(T val, T minval, T maxval) {
    if (val < minval) return minval;
    if (val > maxval) return maxval;
    return val;
}

template<typename T>
void testBatchTopPSamplingKernel(TestCase tc, bool has_diff_runtime_args) {
    unsigned long long seed = 0;

    size_t batch_size = tc.batch_size;
    size_t vocab_size = tc.vocab_size;

    float top_p = tc.top_p;
    float* h_top_ps = new float[batch_size];
    // Initialize runtime top k values.
    for (size_t i = 0; i < batch_size; ++i) {
        h_top_ps[i] = top_p;
        if (has_diff_runtime_args) {
            h_top_ps[i] = clip<float>(h_top_ps[i] + ((i % 2 == 0) ? -0.1 : 0.1), 0.1f, 0.9f);
        }
    }
    int max_top_p = *std::max_element(h_top_ps, h_top_ps + batch_size);
    int end_id = 3;
    size_t max_output_len = tc.output_len;
    size_t max_seq_len = max_output_len;

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));
    Allocator<AllocatorType::CUDA>* allocator = new Allocator<AllocatorType::CUDA>(getDevice());
    allocator->setStream(stream);

    // Logit values in the host of shape (batch_size x vocab_size).
    T* h_logits = new T[batch_size * vocab_size];
    T* h_probs = new T[batch_size * vocab_size];
    T* h_log_probs = new T[batch_size * vocab_size];
    float* h_cum_log_probs = new float[batch_size];
    float* h_output_log_probs = new float[batch_size];
    float* expected_cum_log_probs = new float[batch_size];
    int* h_output_ids = new int[batch_size];
    int* h_seq_lengths = new int[batch_size];
    bool* h_finished = new bool[batch_size];

    initRandom(h_logits, batch_size * vocab_size, -10.0f, -1.0f);
    memset(expected_cum_log_probs, 0, sizeof(float) * batch_size);

    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device);

    curandState_t* curand_states = reinterpret_cast<curandState_t*>(
        allocator->malloc(sizeof(curandState_t) * batch_size, false));
    invokeCurandInitialize(curand_states, batch_size, seed, stream);

    float* top_ps = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));
    int* end_ids = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
    int* sequence_lengths = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
    bool* finished = reinterpret_cast<bool*>(allocator->malloc(sizeof(bool) * batch_size));
    T* probs = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batch_size * vocab_size, true));
    float* cum_log_probs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));
    float* output_log_probs = reinterpret_cast<float*>(
        allocator->malloc(sizeof(float) * max_output_len * batch_size));
    int* output_ids = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * max_seq_len * batch_size));

    int* begin_offsets = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * (batch_size + 1)));
    int* end_offsets = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * (batch_size + 1)));
    int* topp_id_vals_buf = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size * vocab_size));

    size_t workspace_size = 0;
    size_t cub_temp_storage_size = 0;
    // retrieve the workspace size of the top-k sampling kernel.
    invokeBatchTopPSampling<T>(nullptr,  // workspace
                            workspace_size,
                            cub_temp_storage_size,
                            nullptr,  // output_ids
                            nullptr,  // sequence_length
                            nullptr,  // finished_buffer
                            nullptr,  // cum_log_probs
                            nullptr,  // output_log_probs
                            (T*)nullptr,  // log_probs
                            topp_id_vals_buf,
                            end_offsets,
                            begin_offsets,
                            curand_states,
                            batch_size,
                            vocab_size,
                            nullptr,
                            max_top_p,
                            top_ps,
                            stream,
                            &device_prop,
                            nullptr);
    void* workspace = allocator->malloc(workspace_size, false);

    // Initialize.
    cudaH2Dcpy(top_ps, h_top_ps, batch_size);
    deviceFill(end_ids, batch_size, end_id);
    deviceFill(sequence_lengths, batch_size, 0);
    deviceFill(finished, batch_size, false);
    deviceFill(cum_log_probs, batch_size, 0.0f);
    deviceFill(output_log_probs, max_output_len * batch_size, 0.0f);
    deviceFill(output_ids, max_seq_len * batch_size, 0);

    for (size_t step = 0; step < max_output_len; ++step) {
        initRandom(h_logits, batch_size * vocab_size, -10.0f, -1.0f);
        computeProb(h_probs, h_logits, batch_size, vocab_size);
        cudaH2Dcpy(probs, h_probs, batch_size * vocab_size);

        invokeTopPInitialize(topp_id_vals_buf,
                             end_offsets,
                             begin_offsets,
                             batch_size,
                             vocab_size,
                             stream);

        invokeBatchTopPSampling<T>(workspace,
                                   workspace_size,
                                   cub_temp_storage_size,
                                   output_ids + step * batch_size,
                                   sequence_lengths,
                                   finished,
                                   cum_log_probs,
                                   output_log_probs + step * batch_size,
                                   // Note that the kernel needs vocab probs instead of
                                   // log-prob if cum_log_probs or output_log_probs are
                                   // provided. It's because the sampling layer already
                                   // preprocesses log_prob_buf when those are provided.
                                   probs,
                                   topp_id_vals_buf,
                                   end_offsets,
                                   begin_offsets,
                                   curand_states,
                                   batch_size,
                                   vocab_size,
                                   end_ids,
                                   max_top_p,
                                   top_ps,
                                   stream,
                                   &device_prop,
                                   nullptr);

        // Compute reference.
        cudaD2Hcpy(h_output_ids, output_ids + step * batch_size, batch_size);
        cudaD2Hcpy(h_output_log_probs, output_log_probs + step * batch_size, batch_size);
        cudaD2Hcpy(h_cum_log_probs, cum_log_probs, batch_size);
        cudaD2Hcpy(h_seq_lengths, sequence_lengths, batch_size);
        cudaD2Hcpy(h_finished, finished, batch_size);
        computeLogProb(h_log_probs, h_logits, batch_size, vocab_size);
        for (size_t i = 0; i < batch_size; ++i) {
            int idx = i * vocab_size + h_output_ids[i];
            bool expected_finished = h_output_ids[i] == end_id;
            float expected_log_prob = (int)step < h_seq_lengths[i] ? (float)h_log_probs[idx] : 0.0f;
            expected_cum_log_probs[i] += expected_log_prob;
            EXPECT_TRUE(h_finished[i] == expected_finished);
        }
    }
    std::string tag = toTestTag("TestBatchTopPSamplingKernel", tc, std::is_same<T, float>::value);
    bool passed = checkResult(tag, cum_log_probs, expected_cum_log_probs, batch_size);
    EXPECT_TRUE(passed);

    delete[] expected_cum_log_probs;
    delete[] h_seq_lengths;
    delete[] h_output_log_probs;
    delete[] h_cum_log_probs;
    delete[] h_logits;
    delete[] h_log_probs;
    delete[] h_probs;
    delete[] h_output_ids;
    delete[] h_top_ps;
    delete allocator;
    check_cuda_error(cudaStreamDestroy(stream));
}

template<typename T>
void testBatchTopPSamplingWithSkipDecode(TestCase tc) {
    unsigned long long seed = 0;

    size_t batch_size = tc.batch_size;
    size_t vocab_size = tc.vocab_size;

    float top_p = tc.top_p;
    float* h_top_ps = new float[batch_size];
    // Initialize runtime top k values.
    for (size_t i = 0; i < batch_size; ++i) {
        h_top_ps[i] = i % 2 == 0 ? top_p : 0.3f * top_p;
    }
    int max_top_p = *std::max_element(h_top_ps, h_top_ps + batch_size);
    int end_id = 3;
    size_t max_output_len = tc.output_len;
    size_t max_seq_len = max_output_len;

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));
    Allocator<AllocatorType::CUDA>* allocator = new Allocator<AllocatorType::CUDA>(getDevice());
    allocator->setStream(stream);

    // Logit values in the host of shape (batch_size x vocab_size).
    T* h_logits = new T[batch_size * vocab_size];
    T* h_probs = new T[batch_size * vocab_size];
    T* h_log_probs = new T[batch_size * vocab_size];
    float* h_cum_log_probs = new float[batch_size];
    float* h_output_log_probs = new float[batch_size];
    float* expected_cum_log_probs = new float[batch_size];
    int* h_output_ids = new int[batch_size];
    int* h_seq_lengths = new int[batch_size];
    bool* h_finished = new bool[batch_size];
    bool* h_skip_decode = new bool[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        h_skip_decode[i] = i % 2 == 0;
    }

    initRandom(h_logits, batch_size * vocab_size, -3.0f, -3.0f);
    memset(expected_cum_log_probs, 0, sizeof(float) * batch_size);

    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device);

    curandState_t* curand_states = reinterpret_cast<curandState_t*>(
        allocator->malloc(sizeof(curandState_t) * batch_size, false));
    invokeCurandInitialize(curand_states, batch_size, seed, stream);

    float* top_ps = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));
    int* end_ids = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
    int* sequence_lengths = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size));
    bool* finished = reinterpret_cast<bool*>(allocator->malloc(sizeof(bool) * batch_size));
    T* probs = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batch_size * vocab_size, true));
    float* cum_log_probs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size));
    float* output_log_probs = reinterpret_cast<float*>(
        allocator->malloc(sizeof(float) * max_output_len * batch_size));
    int* output_ids = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * max_seq_len * batch_size));

    int* begin_offsets = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * (batch_size + 1)));
    int* end_offsets = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * (batch_size + 1)));
    int* topp_id_vals_buf = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size * vocab_size));

    bool* skip_decode = reinterpret_cast<bool*>(allocator->malloc(sizeof(bool) * batch_size));
    cudaH2Dcpy(skip_decode, h_skip_decode, batch_size);

    size_t workspace_size = 0;
    size_t cub_temp_storage_size = 0;
    // retrieve the workspace size of the top-k sampling kernel.
    invokeBatchTopPSampling<T>(nullptr,  // workspace
                               workspace_size,
                               cub_temp_storage_size,
                               nullptr,  // output_ids
                               nullptr,  // sequence_length
                               nullptr,  // finished_buffer
                               nullptr,  // cum_log_probs
                               nullptr,  // output_log_probs
                               (T*)nullptr,  // log_probs
                               topp_id_vals_buf,
                               end_offsets,
                               begin_offsets,
                               curand_states,
                               batch_size,
                               vocab_size,
                               nullptr,
                               max_top_p,
                               top_ps,
                               stream,
                               &device_prop,
                               nullptr);
    void* workspace = allocator->malloc(workspace_size, false);

    // Initialize.
    cudaH2Dcpy(top_ps, h_top_ps, batch_size);
    deviceFill(end_ids, batch_size, end_id);
    deviceFill(sequence_lengths, batch_size, 0);
    deviceFill(finished, batch_size, false);
    deviceFill(cum_log_probs, batch_size, 0.0f);
    deviceFill(output_log_probs, max_output_len * batch_size, 0.0f);
    deviceFill(output_ids, max_seq_len * batch_size, 0);

    for (size_t step = 0; step < max_output_len; ++step) {
        initRandom(h_logits, batch_size * vocab_size, -3.0f, 3.0f);
        computeProb(h_probs, h_logits, batch_size, vocab_size);
        cudaH2Dcpy(probs, h_probs, batch_size * vocab_size);

        invokeTopPInitialize(topp_id_vals_buf,
                             end_offsets,
                             begin_offsets,
                             batch_size,
                             vocab_size,
                             stream);

        invokeBatchTopPSampling<T>(workspace,
                                   workspace_size,
                                   cub_temp_storage_size,
                                   output_ids + step * batch_size,
                                   sequence_lengths,
                                   finished,
                                   cum_log_probs,
                                   output_log_probs + step * batch_size,
                                   // Note that the kernel needs vocab probs instead of
                                   // log-prob if cum_log_probs or output_log_probs are
                                   // provided. It's because the sampling layer already
                                   // preprocesses log_prob_buf when those are provided.
                                   probs,
                                   topp_id_vals_buf,
                                   end_offsets,
                                   begin_offsets,
                                   curand_states,
                                   batch_size,
                                   vocab_size,
                                   end_ids,
                                   max_top_p,
                                   top_ps,
                                   stream,
                                   &device_prop,
                                   skip_decode);

        // Compute reference.
        cudaD2Hcpy(h_output_ids, output_ids + step * batch_size, batch_size);
        cudaD2Hcpy(h_output_log_probs, output_log_probs + step * batch_size, batch_size);
        cudaD2Hcpy(h_cum_log_probs, cum_log_probs, batch_size);
        cudaD2Hcpy(h_seq_lengths, sequence_lengths, batch_size);
        cudaD2Hcpy(h_finished, finished, batch_size);
        computeLogProb(h_log_probs, h_logits, batch_size, vocab_size);
        for (size_t i = 0; i < batch_size; ++i) {
            if (!h_skip_decode[i]) {
                int idx = i * vocab_size + h_output_ids[i];
                bool expected_finished = h_output_ids[i] == end_id;
                float expected_log_prob = (int)step < h_seq_lengths[i] ? (float)h_log_probs[idx] : 0.0f;
                expected_cum_log_probs[i] += expected_log_prob;
                EXPECT_TRUE(h_finished[i] == expected_finished);
            }
        }
    }
    std::string tag = toTestTag("TestBatchTopPSamplingWithSkipDecode", tc, std::is_same<T, float>::value);
    bool passed = checkResult(tag, cum_log_probs, expected_cum_log_probs, batch_size);

    delete[] expected_cum_log_probs;
    delete[] h_seq_lengths;
    delete[] h_output_log_probs;
    delete[] h_cum_log_probs;
    delete[] h_logits;
    delete[] h_log_probs;
    delete[] h_probs;
    delete[] h_output_ids;
    delete[] h_top_ps;
    delete allocator;
    check_cuda_error(cudaStreamDestroy(stream));
    EXPECT_TRUE(passed);
}

int main() {
    std::vector<TestCase> topk_test_cases {
        // TC: name / batch / vocab / beam / k / p / outlen
        TestCase{"topk", 6,   4,     1, 1,  0.0f, 1},
        TestCase{"topk", 6,   4,     1, 4,  0.0f, 1},
        TestCase{"topk", 128, 51200, 1, 1,  0.0f, 8},
        TestCase{"topk", 128, 51200, 1, 63, 0.0f, 8}
    };
    for (auto &tc : topk_test_cases) {
        testTopKSamplingKernel<float>(tc);
        testTopKSamplingKernel<half>(tc);
        testBatchTopKSamplingKernel<float>(tc, false);
        testBatchTopKSamplingKernel<half>(tc, false);
        testBatchTopKSamplingKernel<float>(tc, true);
        testBatchTopKSamplingKernel<half>(tc, true);
        testBatchTopKSamplingWithSkipDecode<float>(tc);
        testBatchTopKSamplingWithSkipDecode<half>(tc);
    }

    std::vector<TestCase> topp_test_cases {
        // TC: name / batch / vocab / beam / k / p / outlen
        TestCase{"topp", 6,   4,     1, 0,  0.2f, 1},
        TestCase{"topp", 6,   4,     1, 0,  0.9f, 1},
        TestCase{"topp", 6,   4,     1, 0,  1.0f, 1},
        TestCase{"topp", 128, 51200, 1, 0,  0.8f, 16},
        TestCase{"topp", 128, 51200, 1, 0,  1.0f, 16}
    };

    for (auto &tc : topp_test_cases) {
        testBatchTopPSamplingKernel<float>(tc, false);
        testBatchTopPSamplingKernel<half>(tc, false);
        testBatchTopPSamplingKernel<float>(tc, true);
        testBatchTopPSamplingKernel<half>(tc, true);
        testBatchTopPSamplingWithSkipDecode<float>(tc);
        testBatchTopPSamplingWithSkipDecode<half>(tc);
    }

    FT_LOG_INFO("testTopKSamplingKernel done");
    return 0;
}
