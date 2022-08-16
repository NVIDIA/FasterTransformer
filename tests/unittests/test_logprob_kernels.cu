#include <assert.h>
#include <math.h>
#include <float.h>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <sys/time.h>

#include "src/fastertransformer/kernels/logprob_kernels.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/memory_utils.h"

#include "tests/unittests/unittest_utils.h"

using namespace fastertransformer;

#define EXPECT_ALMOST_EQUAL(name, dtype, ctype, out, ref)       \
    do {                                                        \
        bool is_ok = checkResult<dtype,ctype>(name, out, ref);  \
        if(!is_ok) {                                            \
            FT_LOG_ERROR("TEST FAIL [%s] at %s:%d",             \
                        __func__, __FILE__, __LINE__);          \
            throw TestFailureError(__func__);                   \
        }                                                       \
    } while(false)

////////////////////////////////////////////////////////////////////////////////////

struct TestCase {
    std::string name;
    size_t max_input_length;
    size_t batch_size;
    size_t vocab_size;
    size_t beam_width;

    std::string toString() {
        char buf[100];
        snprintf(buf, sizeof(buf),
                 "TestCase[name=%s, max_input_length=%ld, batch=%ld, vocab=%ld, beam_width=%ld]",
                 name.c_str(), max_input_length, batch_size, vocab_size, beam_width);
        return buf;
    }

    void print() {
        FT_LOG_INFO(toString());
    }
};

template<typename T>
void computeCumLogProbs(float* cum_log_probs,
                        float* log_probs,
                        const T* logits,
                        const int* input_ids,
                        const int* input_lengths,
                        const size_t max_input_length,
                        const size_t batch_size,
                        const size_t vocab_size,
                        const size_t vocab_size_padded)
{
    for (size_t step = 0; step < max_input_length; ++step) {
        for (size_t i = 0; i < batch_size; ++i) {
            if ((int)step == 0) {
                if (log_probs != nullptr) {
                    log_probs[i] = 0.0f;
                }
                cum_log_probs[i] = 0.0f;
            }
            else if ((int)step < input_lengths[i]) {
                size_t step_offset = (step - 1) * batch_size * vocab_size_padded;
                const T* vec = logits + step_offset + i * vocab_size_padded;
                float max_logits = -FLT_MAX;
                for (size_t v = 0; v < vocab_size; ++v) {
                    float val = static_cast<float>(vec[v]);
                    if (val > max_logits) {
                        max_logits = val;
                    }
                }
                float sum = 0.0f;
                for (size_t v = 0; v < vocab_size; ++v) {
                    sum += expf(static_cast<float>(vec[v]) - max_logits);
                }
                int token_id = input_ids[step * batch_size + i];
                float log_prob = static_cast<float>(vec[token_id]) - max_logits - log(sum);
                if (log_probs != nullptr) {
                    log_probs[step * batch_size + i] = log_prob;
                }
                cum_log_probs[i] += log_prob;
            }
        }
    }
}

template<typename T>
void computeCumLogProbsBatchFirst(float* cum_log_probs,
                                  float* log_probs,
                                  const T* logits,
                                  const int* input_ids,
                                  const int* input_lengths,
                                  const size_t max_input_length,
                                  const size_t batch_size,
                                  const size_t vocab_size,
                                  const size_t vocab_size_padded)
{
    for (size_t i = 0; i < batch_size; ++i) {
        size_t batch_offset = i * max_input_length * vocab_size_padded;
        for (size_t step = 0; step < max_input_length; ++step) {
            if ((int)step == 0) {
                if (log_probs != nullptr) {
                    log_probs[i * max_input_length] = 0.0f;
                }
                cum_log_probs[i] = 0.0f;
            }
            else if ((int)step < input_lengths[i]) {
                const T* vec = logits + batch_offset + (step - 1) * vocab_size_padded;
                float max_logits = -FLT_MAX;
                for (size_t v = 0; v < vocab_size; ++v) {
                    float val = static_cast<float>(vec[v]);
                    if (val > max_logits) {
                        max_logits = val;
                    }
                }
                float sum = 0.0f;
                for (size_t v = 0; v < vocab_size; ++v) {
                    sum += expf(static_cast<float>(vec[v]) - max_logits);
                }
                int token_id = input_ids[i * max_input_length + step];
                float log_prob = static_cast<float>(vec[token_id]) - max_logits - log(sum);
                if (log_probs != nullptr) {
                    log_probs[i * max_input_length + step] = log_prob;
                }
                cum_log_probs[i] += log_prob;
            }
        }
    }
}

/////////////////////////////////// Unittests //////////////////////////////////////////

template<typename T>
void testCumLogProbCorrectness(TestCase tc) {
    size_t max_input_length = tc.max_input_length;
    size_t batchxbeam = tc.batch_size * tc.beam_width;
    size_t vocab_size = tc.vocab_size;
    // Make multiple of 8 as GPT does.
    size_t vocab_size_padded = static_cast<size_t>(ceil(vocab_size / 8.f) * 8);

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));
    Allocator<AllocatorType::CUDA> allocator(getDevice());

    // input values
    T* h_logits = new T[max_input_length * batchxbeam * vocab_size];
    int* h_input_ids = new int[max_input_length * batchxbeam];
    int* h_input_lengths = new int[batchxbeam];

    // outupt buffers
    float* expected_cum_log_probs = new float[batchxbeam];

    // initialize host buffers
    initRandom(h_logits, max_input_length * batchxbeam * vocab_size, -10.0f / vocab_size, -1.0f);
    initRandomInt(h_input_ids, max_input_length * batchxbeam, 0, vocab_size);
    initRandomInt(h_input_lengths, batchxbeam, 1, max_input_length + 1);
    memset(expected_cum_log_probs, 0, sizeof(float) * batchxbeam);

    // device buffers
    T* d_logits = reinterpret_cast<T*>(allocator.malloc(sizeof(T) * max_input_length * batchxbeam * vocab_size));
    int *d_input_ids = reinterpret_cast<int*>(allocator.malloc(sizeof(int) * max_input_length * batchxbeam));
    int *d_input_lengths = reinterpret_cast<int*>(allocator.malloc(sizeof(int) * batchxbeam));
    float* d_cum_log_probs = reinterpret_cast<float*>(allocator.malloc(sizeof(float) * batchxbeam));

    // initialize device buffers
    cudaH2Dcpy(d_logits, h_logits, max_input_length * batchxbeam * vocab_size);
    cudaH2Dcpy(d_input_ids, h_input_ids, max_input_length * batchxbeam);
    cudaH2Dcpy(d_input_lengths, h_input_lengths, batchxbeam);
    check_cuda_error(cudaMemset(d_cum_log_probs, 0, sizeof(float) * batchxbeam));

    size_t workspace_size = sizeof(float) * max_input_length * batchxbeam;
    void* workspace = allocator.malloc(workspace_size);
    invokeLogProbFromLogits(d_cum_log_probs,
                            d_logits,
                            d_input_ids,
                            d_input_lengths,
                            max_input_length,
                            batchxbeam,
                            vocab_size,
                            vocab_size_padded,
                            workspace,
                            workspace_size,
                            stream,
                            false);
    computeCumLogProbs(expected_cum_log_probs,
                       nullptr,
                       h_logits,
                       h_input_ids,
                       h_input_lengths,
                       max_input_length,
                       batchxbeam,
                       vocab_size,
                       vocab_size_padded);
    std::string tag = tc.toString() + (std::is_same<T, float>::value ? " (fp32)" : " (fp16)");
    bool passed = checkResult(tag.c_str(), d_cum_log_probs, expected_cum_log_probs, batchxbeam);
    EXPECT_TRUE(passed);

    FT_LOG_DEBUG("free host buffers");
    delete[] expected_cum_log_probs;
    delete[] h_input_lengths;
    delete[] h_input_ids;
    delete[] h_logits;

    FT_LOG_DEBUG("free device buffers");
    allocator.free((void**)(&d_cum_log_probs));
    allocator.free((void**)(&d_input_lengths));
    allocator.free((void**)(&d_input_ids));
    allocator.free((void**)(&d_logits));
    check_cuda_error(cudaStreamDestroy(stream));
}

template<typename T>
void testBatchFirstCumLogProbCorrectness(TestCase tc) {
    size_t max_input_length = tc.max_input_length;
    size_t batchxbeam = tc.batch_size * tc.beam_width;
    size_t vocab_size = tc.vocab_size;
    // Make multiple of 8 as GPT does.
    size_t vocab_size_padded = static_cast<size_t>(ceil(vocab_size / 8.f) * 8);

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));
    Allocator<AllocatorType::CUDA> allocator(getDevice());

    // input values
    T* h_logits = new T[max_input_length * batchxbeam * vocab_size_padded];
    int* h_input_ids = new int[max_input_length * batchxbeam];
    int* h_input_lengths = new int[batchxbeam];

    // outupt buffers
    float* expected_cum_log_probs = new float[batchxbeam];

    // initialize host buffers
    initRandom(h_logits, max_input_length * batchxbeam * vocab_size_padded, -10.0f / vocab_size, -1.0f);
    initRandomInt(h_input_ids, max_input_length * batchxbeam, 0, vocab_size);
    initRandomInt(h_input_lengths, batchxbeam, 1, max_input_length + 1);
    memset(expected_cum_log_probs, 0, sizeof(float) * batchxbeam);

    // device buffers
    T* d_logits = reinterpret_cast<T*>(allocator.malloc(sizeof(T) * max_input_length * batchxbeam * vocab_size_padded));
    int *d_input_ids = reinterpret_cast<int*>(allocator.malloc(sizeof(int) * max_input_length * batchxbeam));
    int *d_input_lengths = reinterpret_cast<int*>(allocator.malloc(sizeof(int) * batchxbeam));
    float* d_cum_log_probs = reinterpret_cast<float*>(allocator.malloc(sizeof(float) * batchxbeam));

    // initialize device buffers
    cudaH2Dcpy(d_logits, h_logits, max_input_length * batchxbeam * vocab_size_padded);
    cudaH2Dcpy(d_input_ids, h_input_ids, max_input_length * batchxbeam);
    cudaH2Dcpy(d_input_lengths, h_input_lengths, batchxbeam);
    check_cuda_error(cudaMemset(d_cum_log_probs, 0, sizeof(float) * batchxbeam));

    size_t workspace_size = sizeof(float) * max_input_length * batchxbeam;
    void* workspace = allocator.malloc(workspace_size);
    invokeLogProbFromLogits(d_cum_log_probs,
                            d_logits,
                            d_input_ids,
                            d_input_lengths,
                            max_input_length,
                            batchxbeam,
                            vocab_size,
                            vocab_size_padded,
                            workspace,
                            workspace_size,
                            stream,
                            true);

    computeCumLogProbsBatchFirst(expected_cum_log_probs,
                                 nullptr,
                                 h_logits,
                                 h_input_ids,
                                 h_input_lengths,
                                 max_input_length,
                                 batchxbeam,
                                 vocab_size,
                                 vocab_size_padded);
    std::string tag = tc.toString() + (std::is_same<T, float>::value ? " (fp32)" : " (fp16)");
    bool passed = checkResult(tag.c_str(), d_cum_log_probs, expected_cum_log_probs, batchxbeam);
    EXPECT_TRUE(passed);

    FT_LOG_DEBUG("free host buffers");
    delete[] expected_cum_log_probs;
    delete[] h_input_lengths;
    delete[] h_input_ids;
    delete[] h_logits;

    FT_LOG_DEBUG("free device buffers");
    allocator.free((void**)(&d_cum_log_probs));
    allocator.free((void**)(&d_input_lengths));
    allocator.free((void**)(&d_input_ids));
    allocator.free((void**)(&d_logits));
    check_cuda_error(cudaStreamDestroy(stream));
}

int main(int argc, char* argv[]) {
    std::vector<TestCase> test_cases {
        // TC: name / max_input_seq / batch / vocab / beam
        TestCase{"cum_logprob test", 1,    32, 16,    1},
        TestCase{"cum_logprob test", 129,  8,  50211, 1},
        TestCase{"cum_logprob test", 255,  8,  51200, 1},
        TestCase{"cum_logprob test", 500,  8,  51200, 1},
        TestCase{"cum_logprob test", 1023, 8,  51200, 1},
    };

    for (auto &tc : test_cases) {
        testCumLogProbCorrectness<float>(tc);
        testCumLogProbCorrectness<half>(tc);
    }
    FT_LOG_INFO("Test Done");

    for (auto &tc : test_cases) {
        tc.name = "batch first test";
        testBatchFirstCumLogProbCorrectness<float>(tc);
        testBatchFirstCumLogProbCorrectness<half>(tc);
    }
    FT_LOG_INFO("Test Done");

    return 0;
}
