#include <algorithm>   // std::min, std::max
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include "src/fastertransformer/layers/DynamicDecodeLayer.h"
#include "src/fastertransformer/layers/sampling_layers/TopKSamplingLayer.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/Tensor.h"

// namespace ft = fastertransformer;
using namespace fastertransformer;


#define PRINT_LIMIT 16
#define EPSILON (1e-20)
#define EPSILON_FP16 (1e-10)

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

bool almostEqual(float a, float b, float atol = 1e-5, float rtol = 1e-8)
{
    // Params: a = value to compare and b = reference
    // This function follows implementation of numpy.isclose(), which checks
    //   abs(a - b) <= (atol + rtol * abs(b)).
    // Note that the inequality above is asymmetric where b is considered as
    // a reference value. To account into both absolute/relative errors, it
    // uses absolute tolerance and relative tolerance at the same time. The
    // default values of atol and rtol borrowed from numpy.isclose(). For the
    // case of nan value, the result will be true.
    if (isnan(a) && isnan(b)) {
        return true;
    }
    return fabs(a - b) <= (atol + rtol * fabs(b));
}

template<typename T>
bool checkResult(std::string name, T* out, T*ref, size_t size, float atol, float rtol) {
    size_t failures = 0;
    float relative_gap = 0.0f;

    T* h_out = reinterpret_cast<T*>(malloc(sizeof(T) * size));
    cudaMemcpy(h_out, out, sizeof(T) * size, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < size; ++i) {
        // The values for the output and the reference.
        float a = (float)h_out[i];
        float b = (float)ref[i];

        bool ok = almostEqual(a, b, atol, rtol);
        // Print the error.
        if (!ok && failures < 4) {
            FT_LOG_ERROR(">> invalid result for i=%lu:", i);
            FT_LOG_ERROR(">>    found......: %10.6f", a);
            FT_LOG_ERROR(">>    expected...: %10.6f", b);
            FT_LOG_ERROR(">>    error......: %.6f", fabsf(a - b));
            FT_LOG_ERROR(">>    tol........: %.6f", atol + rtol * fabs(b));
        }
        // Update the number of failures.
        failures += ok ? 0 : 1;
        // Update the relative gap.
        relative_gap += fabsf(a - b) / (fabsf(b) + EPSILON);
    }

    relative_gap /= size;

    // Allow not matched up to 1% elements.
    size_t tol_failures = (size_t)(0.01 * size);
    FT_LOG_INFO("check.......%-30s : %s (failures: %.2f%% atol: %.2e rtol: %.2e rel_gap: %.2e%%)",
                name.c_str(), failures <= tol_failures ? "OK" : "FAILED",
                100. * failures / size, atol, rtol, 100. * relative_gap);
    return failures <= tol_failures;
}

template<typename T>
bool checkResult(std::string name, T* out, T* ref, size_t size) {
    bool is_fp32 = sizeof(T) == 4;
    // float atol = is_fp32 ? 1e-6f : 1e-3f;
    // float rtol = is_fp32 ? 1e-4f : 1e-1f;
    float atol = is_fp32 ? 1e-4f : 1e-3f;
    float rtol = is_fp32 ? 1e-2f : 1e-1f;
    bool is_ok = checkResult(name, out, ref, size, atol, rtol);
    return is_ok;
}

template<typename T>
void initRandom(T* ptr, size_t size, float minval, float maxval) {
    for (size_t i = 0; i < size; ++i) {
        float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        val *= (maxval - minval);
        ptr[i] = static_cast<T>(minval + val);
    }
}

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

template<typename T>
static inline void printMatrixHightPrecision(T* ptr, int m, int k, int stride, bool is_device_ptr)
{
    T* tmp;
    if (is_device_ptr) {
        // k < stride ; stride = col-dimension.
        tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
        check_cuda_error(cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    }
    else {
        tmp = ptr;
    }

    for (int ii = -1; ii < m; ++ii) {
        if (ii >= 0) {
            printf("%02d ", ii);
        }
        else {
            printf("   ");
        }

        for (int jj = 0; jj < k; jj += 1) {
            if (ii >= 0) {
                printf("%9.6f ", (float)tmp[ii * stride + jj]);
            }
            else {
                printf("%9d ", jj);
            }
        }
        printf("\n");
    }
    if (is_device_ptr) {
        free(tmp);
    }
}

template<typename T>
static inline void printMatrixWithLimit(T* ptr, int m, int k, int stride, bool is_device_ptr) {
    printMatrixHightPrecision(ptr, std::min(PRINT_LIMIT, m), std::min(PRINT_LIMIT, k), stride, is_device_ptr);
}

template<typename T>
void testDynamicDecoingLayer(TestCase tc) {

    bool is_fp32 = std::is_same<T, float>::value;

    size_t beam_width = tc.beam_width;
    size_t top_k = tc.top_k;
    float top_p = tc.top_p;
    unsigned long long seed = 0;
    // use default values having no effect.
    float temperature = 1.0f;
    float len_penalty = 1.0f;
    float repetition_penalty = 1.0f;

    size_t batch_size = tc.batch_size;
    size_t vocab_size = tc.vocab_size;
    int end_id = 3;
    size_t max_input_len = 0;  // has no effect.
    size_t max_output_len = tc.output_len;
    size_t max_seq_len = max_input_len + max_output_len;

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));

    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    check_cuda_error(cudaStreamCreate(&stream));
    check_cuda_error(cublasCreate(&cublas_handle));
    check_cuda_error(cublasLtCreate(&cublaslt_handle));
    check_cuda_error(cublasSetStream(cublas_handle, stream));

    cublasAlgoMap cublas_algo_map(GEMM_CONFIG);
    Allocator<AllocatorType::CUDA> * allocator = new Allocator<AllocatorType::CUDA>(getDevice());
    allocator->setStream(stream);

    std::mutex* cublas_wrapper_mutex = new std::mutex();
    cublasMMWrapper *cublas_wrapper = new cublasMMWrapper(cublas_handle,
                                   cublaslt_handle,
                                   stream,
                                   &cublas_algo_map,
                                   cublas_wrapper_mutex,
                                   allocator);

    DynamicDecodeLayer<T> dynamic_decode_layer(vocab_size,
                                               vocab_size,
                                               end_id,
                                               stream,
                                               cublas_wrapper,
                                               allocator,
                                               false,   // is_free_buffer_after_forward
                                               &prop);  // cuda_device_prop

    const DataType data_type = getTensorType<T>();
    size_t logits_size = batch_size * beam_width * vocab_size;
    T* logits_buf = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * logits_size, true));

    // Logit values in the host of shape ((batch_size x beam) x vocab_size) where beam = 1.
    T* h_logits = reinterpret_cast<T*>(malloc(sizeof(T) * batch_size * beam_width * vocab_size));
    T* h_probs = reinterpret_cast<T*>(malloc(sizeof(T) * batch_size * beam_width * vocab_size));
    T* h_log_probs = reinterpret_cast<T*>(malloc(sizeof(T) * batch_size * beam_width * vocab_size));
    float* h_cum_log_probs = reinterpret_cast<float*>(malloc(sizeof(float) * batch_size * beam_width));
    float* h_output_log_probs = reinterpret_cast<float*>(
        malloc(sizeof(float) * max_output_len * batch_size * beam_width));
    float* expected_cum_log_probs = reinterpret_cast<float*>(malloc(sizeof(float) * batch_size * beam_width));
    initRandom(h_logits, batch_size * beam_width * vocab_size, -10.0f / vocab_size, -1.0f);
    computeProb(h_probs, h_logits, batch_size * beam_width, vocab_size);
    computeLogProb(h_log_probs, h_logits, batch_size * beam_width, vocab_size);
    memset(expected_cum_log_probs, 0, sizeof(float) * batch_size * beam_width);

#ifndef NDEBUG
    FT_LOG_DEBUG("logit values");
    printMatrixWithLimit(h_logits, batch_size * beam_width, vocab_size, vocab_size, false);
    FT_LOG_DEBUG("\nprob values");
    printMatrixWithLimit(h_probs, batch_size * beam_width, vocab_size, vocab_size, false);
    FT_LOG_DEBUG("\nlog-prob values");
    printMatrixWithLimit(h_log_probs, batch_size * beam_width, vocab_size, vocab_size, false);
#endif

    int* tiled_input_lengths_buf = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size * beam_width));
    float* cum_log_probs = reinterpret_cast<float*>(allocator->malloc(sizeof(float) * batch_size * beam_width));
    float* output_log_probs = reinterpret_cast<float*>(
        allocator->malloc(sizeof(float) * max_output_len * batch_size * beam_width));
    bool has_diff_runtime_args = false;
    bool is_initialize_random_table = true;

    int* output_ids = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * max_seq_len * batch_size * beam_width));
    int* h_output_ids = reinterpret_cast<int*>(malloc(sizeof(int) * batch_size * beam_width));

    // Init by zero.
    cudaMemset(cum_log_probs, 0, sizeof(float) * batch_size * beam_width);
    cudaMemset(output_log_probs, 0, sizeof(float) * max_output_len * batch_size * beam_width);
    cudaMemset(output_ids, 0, sizeof(int) * max_seq_len * batch_size * beam_width);

    for (size_t step = max_input_len; step < max_output_len; ++step) {
        uint ite = 0;
        seed += step;

        // Reset by the test value since the sampling layer internally update the logit buffer (making it log-prob).
        cudaH2Dcpy(logits_buf, h_logits, logits_size);
        std::unordered_map<std::string, Tensor> dynamic_decode_input_tensors{
            {"logits", Tensor{MEMORY_GPU, TYPE_FP32, {batch_size, beam_width, vocab_size}, logits_buf}},
            {"embedding_bias", Tensor{MEMORY_GPU, data_type, {vocab_size}, nullptr}},
            {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}},
            {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_len}},
            {"input_lengths",
                Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width}, tiled_input_lengths_buf}},
            {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}},
            {"has_diff_runtime_args", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &has_diff_runtime_args}},
            {"local_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &batch_size}},
            {"is_initialize_random_table", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_initialize_random_table}},
            {"end_id", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &end_id}},
            {"random_seed", {MEMORY_CPU, TYPE_INT32, {1}, &seed}},
            {"runtime_top_k", {MEMORY_CPU, TYPE_INT32, {1}, &top_k}},
            {"runtime_top_p", {MEMORY_CPU, TYPE_FP32, {1}, &top_p}},
            {"temperature", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &temperature}},
            {"len_penalty", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &len_penalty}},
            {"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &repetition_penalty}}
        };

        // common outputs
        std::unordered_map<std::string, Tensor> dynamic_decode_output_tensors{
            {"output_ids", Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len, batch_size, beam_width}, output_ids}},
            {"finished", Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width}, nullptr}},
            {"cum_log_probs", Tensor{MEMORY_GPU, TYPE_FP32, {batch_size * beam_width}, cum_log_probs}},
            {"output_log_probs",
                Tensor{MEMORY_GPU, TYPE_FP32, {max_seq_len, batch_size, beam_width}, output_log_probs}},
            {"parent_ids", Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len, batch_size, beam_width}, nullptr}},
            {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, nullptr}},
            // necessary for beam search.
            {"tgt_cache_indirection",
                Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width, max_output_len}, nullptr}}};

        dynamic_decode_layer.forward(&dynamic_decode_output_tensors,
                                     &dynamic_decode_input_tensors);

        FT_LOG_DEBUG("Step %2d generated ids", step);
        cudaD2Hcpy(h_output_ids,
                   (int*)dynamic_decode_output_tensors
                       .at("output_ids")
                       .getPtrWithOffset(step * (batch_size * beam_width)),
                   batch_size * beam_width);
        cudaD2Hcpy(h_cum_log_probs, cum_log_probs, batch_size * beam_width);
        cudaD2Hcpy(h_output_log_probs, output_log_probs, max_output_len * batch_size * beam_width);
        for (size_t i = 0; i < batch_size * beam_width; ++i) {
            int idx = i * vocab_size + h_output_ids[i];
            expected_cum_log_probs[i] += (float)h_log_probs[idx];
            FT_LOG_DEBUG(
                "| step %2d batch %2d idx %7d id %6d | log-prob %9.4f (expt: %9.4f) "
                "| cum-log-prob %9.4f (expt: %9.4f) | prob %9.4e",
                (int)step, (int)i, (int)idx, (int)h_output_ids[i],
                h_output_log_probs[step * batch_size * beam_width + i], (float)h_log_probs[idx],
                h_cum_log_probs[i], expected_cum_log_probs[i], (float)h_probs[idx]);
        }
        FT_LOG_DEBUG("");

#ifndef NDEBUG
        // print output ids
        for (size_t s = max_input_len; s < max_seq_len; ++s) {
            cudaD2Hcpy(h_output_ids,
                       (int*)dynamic_decode_output_tensors
                           .at("output_ids")
                           .getPtrWithOffset(s * (batch_size * beam_width)),
                       batch_size * beam_width);
            printf("%02d ", (int)s);
            for (size_t b = 0; b < batch_size; ++b) {
                printf("%3d ", (int)h_output_ids[b]);
            }
            printf("\n");
        }
#endif
    }
    std::string tag = tc.toString() + (is_fp32 ? " (fp32)" : " (fp16)");
    checkResult(tag, cum_log_probs, expected_cum_log_probs, batch_size * beam_width);

    free(expected_cum_log_probs);
    free(h_output_log_probs);
    free(h_cum_log_probs);
    free(h_logits);
    free(h_log_probs);
    free(h_probs);
    free(h_output_ids);
    allocator->free(tiled_input_lengths_buf);
    allocator->free(cum_log_probs);
    allocator->free(output_ids);
    allocator->free(logits_buf);
    allocator->free(output_log_probs);

    delete cublas_wrapper;
    delete allocator;
    check_cuda_error(cudaStreamDestroy(stream));
    check_cuda_error(cublasDestroy(cublas_handle));
    check_cuda_error(cublasLtDestroy(cublaslt_handle));
}

int main() {
    std::vector<TestCase> test_cases {
        // TC: name / batch / vocab / beam / k / p / outlen
        TestCase{"topk",      6,  4,     1, 1,  0.0f, 4},
        TestCase{"topk",      6,  4,     1, 4,  0.0f, 4},
        TestCase{"topk",      6,  51200, 1, 31, 0.0f, 16},
        TestCase{"topk",      32, 51200, 1, 63, 0.0f, 16},
        TestCase{"topk",      32, 51200, 1, 64, 0.0f, 16},
        TestCase{"topp",      6,  4,     1, 0,  0.8f, 4},
        TestCase{"topp",      6,  4,     1, 0,  1.0f, 4},
        TestCase{"topp",      6,  51200, 1, 0,  0.8f, 16},
        TestCase{"topp",      32, 51200, 1, 0,  0.8f, 16},
        TestCase{"topp",      32, 51200, 1, 0,  1.0f, 16},
        TestCase{"topk_topp", 6,  4,     1, 1,  0.8f, 16},
        TestCase{"topk_topp", 6,  4,     1, 4,  1.0f, 16},
        TestCase{"topk_topp", 6,  51200, 1, 31, 0.8f, 16},
        TestCase{"topk_topp", 32, 51200, 1, 63, 0.8f, 16},
        TestCase{"topk_topp", 32, 51200, 1, 64, 1.0f, 16},
    };

    for (auto &tc : test_cases) {
        testDynamicDecoingLayer<float>(tc);
        testDynamicDecoingLayer<half>(tc);  // T5 model uses DynamicDecodingLayer<T>.
    }
    FT_LOG_INFO("Test Done");
    return 0;
}
