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

using namespace fastertransformer;

#define PRINT_LIMIT 16
#define EPSILON (1e-20)

// Can be replaced by the function provided by a test framework

class TestFailureError : public std::exception {
private:
    std::string msg_;
public:
    explicit TestFailureError() = default;
    explicit TestFailureError(std::string name, std::string msg = "") {
        msg_ = fmtstr("TEST FAIL [%s] %s", name.c_str(), msg.c_str());
    }
	const char* what () const throw () {
    	return msg_.c_str();
    }
};

#define EXPECT_TRUE(cond)                           \
    do { if(!(cond)) {                              \
        FT_LOG_ERROR("TEST FAIL [%s] at %s:%d",     \
                     __func__, __FILE__, __LINE__); \
        throw TestFailureError(__func__);           \
    } } while(false)

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
        size_t step_offset = step * batch_size * vocab_size_padded;
        for (size_t i = 0; i < batch_size; ++i) {
            if ((int)step < input_lengths[i]) {
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
            if ((int)step < input_lengths[i]) {
                const T* vec = logits + batch_offset + step * vocab_size_padded;
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
    check_cuda_error(cudaMemcpy(h_out, out, sizeof(T) * size, cudaMemcpyDeviceToHost));

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
    float atol = is_fp32 ? 1e-6f : 1e-3f;
    float rtol = is_fp32 ? 1e-4f : 1e-1f;
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

void initRandomInt(int* ptr, size_t size, int minval, int maxval) {
    assert(minval < maxval);
    int mod = maxval - minval;
    for (size_t i = 0; i < size; ++i) {
        ptr[i] = minval + rand() % mod;
    }
}

template<typename T>
static inline void printMatrixScientificFormat(T* ptr, int m, int k, int stride, bool is_device_ptr)
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
                printf("%11.4e ", (float)tmp[ii * stride + jj]);
            }
            else {
                printf("%11d ", jj);
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
    printMatrixScientificFormat(ptr, std::min(PRINT_LIMIT, m), std::min(PRINT_LIMIT, k), stride, is_device_ptr);
}


/////////////////////////////////// Unittests //////////////////////////////////////////

template<typename T>
void testCumLogProbCorrectness(TestCase tc) {
    size_t max_input_length = tc.max_input_length;
    size_t batchxbeam = tc.batch_size * tc.beam_width;
    size_t vocab_size = tc.vocab_size;
    // Make mulitple of 8 as GPT does.
    size_t vocab_size_padded = static_cast<size_t>(ceil(vocab_size / 8.f) * 8);

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));
    Allocator<AllocatorType::CUDA> allocator(getDevice());

    // input values
    T* h_logits = reinterpret_cast<T*>(malloc(sizeof(T) * max_input_length * batchxbeam * vocab_size));
    int* h_input_ids = reinterpret_cast<int*>(malloc(sizeof(int) * max_input_length * batchxbeam));
    int* h_input_lengths = reinterpret_cast<int*>(malloc(sizeof(int) * batchxbeam));

    // outupt buffers
    float* expected_cum_log_probs = reinterpret_cast<float*>(malloc(sizeof(float) * batchxbeam));

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
    checkResult(tc.toString().c_str(), d_cum_log_probs, expected_cum_log_probs, batchxbeam);

    FT_LOG_DEBUG("free host buffers");
    free(expected_cum_log_probs);
    free(h_input_lengths);
    free(h_input_ids);
    free(h_logits);

    FT_LOG_DEBUG("free device buffers");
    allocator.free(d_cum_log_probs);
    allocator.free(d_input_lengths);
    allocator.free(d_input_ids);
    allocator.free(d_logits);
    check_cuda_error(cudaStreamDestroy(stream));
}

template<typename T>
void testBatchFirstCumLogProbCorrectness(TestCase tc) {
    size_t max_input_length = tc.max_input_length;
    size_t batchxbeam = tc.batch_size * tc.beam_width;
    size_t vocab_size = tc.vocab_size;
    // Make mulitple of 8 as GPT does.
    size_t vocab_size_padded = static_cast<size_t>(ceil(vocab_size / 8.f) * 8);

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));
    Allocator<AllocatorType::CUDA> allocator(getDevice());

    // input values
    T* h_logits = reinterpret_cast<T*>(malloc(sizeof(T) * max_input_length * batchxbeam * vocab_size_padded));
    int* h_input_ids = reinterpret_cast<int*>(malloc(sizeof(int) * max_input_length * batchxbeam));
    int* h_input_lengths = reinterpret_cast<int*>(malloc(sizeof(int) * batchxbeam));

    // outupt buffers
    float* expected_cum_log_probs = reinterpret_cast<float*>(malloc(sizeof(float) * batchxbeam));

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
    checkResult(tc.toString().c_str(), d_cum_log_probs, expected_cum_log_probs, batchxbeam);

    FT_LOG_DEBUG("free host buffers");
    free(expected_cum_log_probs);
    free(h_input_lengths);
    free(h_input_ids);
    free(h_logits);

    FT_LOG_DEBUG("free device buffers");
    allocator.free(d_cum_log_probs);
    allocator.free(d_input_lengths);
    allocator.free(d_input_ids);
    allocator.free(d_logits);
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
