#include "worker.hpp"
#include <string>
#include <cublasLt.h>
#include <cuda_bf16.h>
// #include "3rdparty/swarm-review-30931053/cudart/cuda_fp8.h"
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include "cuda_utils.h"
#include <iostream>
#include <vector>
namespace {

static constexpr uint64_t LOOP_COUNTER = 1ull;


template<typename TA, typename TB>
void checkMat(TA* A, TB* B, int size, std::string mark)
{
    float max_diff = -10000.0f;
    float max_diff_a, max_diff_b;
    TA* matA = (TA*)malloc(sizeof(TA) * size);
    TB* matB = B;
    int not_passed = 0;
    cudaMemcpy(matA, A, sizeof(TA) * size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(matB, B, sizeof(TB) * size, cudaMemcpyDeviceToHost);
    printf("[INFO] A  B  abs_diff rel_diff\n");
    for (int jjj = 0; jjj < size; jjj++) {
        float diff = fabs(float(matA[jjj]) - float(matB[jjj]));
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_a = float(matA[jjj]);
            max_diff_b = float(matB[jjj]);
        }
        // if (fabs(float(matA[jjj]) - float(matB[jjj])) > 0.001) {
            not_passed += 1;
            if (not_passed < 100)
                printf("%4d %10.4f %10.4f %10.4f (%7.4f %% percent)\n",
                       jjj,
                       float(matA[jjj]),
                       float(matB[jjj]),
                       diff,
                       (diff) / (float(matA[jjj] + 1e-6f)) * 100.f);
        // }
    }
    printf("[%s] max diff : %f ; a : %f ; b : %f\n", mark.c_str(), max_diff, max_diff_a, max_diff_b);
    if (not_passed != 0)
        printf("[%s] different elements : %d \n", mark.c_str(), not_passed);
    else
        printf("[%s] check pass!\n", mark.c_str());
    free(matA);
    // free(matB);
}

void getAMax(float* amax_ptr, float* input, const int m, const int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0) {
                // amax_ptr[j] = fabs(input[i * n + j]) / 127.f;
                amax_ptr[j] = 1.0f;
            }
            // else {
            amax_ptr[j] = std::max(amax_ptr[j], fabs(input[i * n + j]) / 127.f);
            // }
        }
    }
}

//------------------------------------------------

void freeResources(void** a, void** b, void** c, void** d, void** ws)
{
    auto cleaner = [](void** ptr) {
        if (*ptr != nullptr) {
            cudaFree(*ptr);
            *ptr = nullptr;
        }
    };
    cleaner(a);
    cleaner(b);
    cleaner(c);
    cleaner(d);
    cleaner(ws);
}

//------------------------------------------------

inline void checkCublasStatus(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        // throw std::logic_error("cuBLAS API failed");
        // printf("cuBLAS API failed");
    }
}

inline void checkCudaStatus(cudaError_t status)
{
    if (status != cudaSuccess) {
        printf("CUDA API failed with status %d\n", status);
        throw std::logic_error("CUDA API failed");
    }
}

//------------------------------------------------

template<typename T>
bool fillTensorWithValue(void* devPtr, const size_t sizeBytes, const T value)
{
    std::vector<T> hData(sizeBytes / sizeof(T), value);
    const auto status =
        cudaMemcpy(static_cast<uint8_t*>(devPtr), hData.data(), hData.size() * sizeof(T), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        return false;
    }
    return true;
};

//------------------------------------------------

template<typename T>
void printDevBuffer(void* devPtr, const size_t sizeBytes, const size_t row)
{
    std::vector<T> hBuffer(sizeBytes / sizeof(T), static_cast<T>(0.0));
    const auto status = cudaMemcpy(hBuffer.data(), devPtr, hBuffer.size() * sizeof(T), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        std::cout << "Failed to copy dev buffer's content!" << std::endl;
        return;
    }
    printf("hBuffer.size(): %d \n", (int)hBuffer.size());
    for (size_t i = 0; i < hBuffer.size(); i++) {
        if (!(i % row)) {
            std::cout << std::endl;
        }
        std::cout << (float)hBuffer[i] << " ";
    }
    std::cout << std::endl << std::endl;
}

//------------------------------------------------

#if 1
struct Timer {
    Timer(const char* _msg): msg(_msg)
    {
        _start = getTick();
    }
    ~Timer()
    {
        const auto duration = getTick() - _start;
        printf("^^^^^^^^ [Timer] %s: %lu\n", msg, duration);
    }

private:
    uint64_t getTick()
    {
        uint64_t low, high;
        __asm__ volatile("rdtsc" : "=a"(low), "=d"(high));
        return (high << 32) | low;
    }
    uint64_t _start = 0;
    const char* msg;
};
#else
class Timer {
public:
    Timer(std::string&& _msg)
    {
        start = Clock::now();
        msg = std::move(_msg);
    }
    ~Timer()
    {
        TimePoint end = Clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        std::cout << "[Timer] " << msg << ": " << dur.count() << " ns" << std::endl;
    }

private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;  // std::chrono::time_point;
    TimePoint start;
    std::string msg;
};
#endif

}  // anonymous namespace

template<typename T>
void printMatrix(T* devPtr, const int m, const int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", (float)(devPtr[i * n + j]));
        }
        printf("\n");
    }
    printf("\n\n");
}

template void printMatrix(float* devPtr, const int m, const int n);

//=====================================================================================================================
// public API

int32_t cublasTester(const ProblemDesc& p)
{
    // Initialize input A, B and D
    // A: weight matrix, k * n
    // B: input matrix, m * k
    // D: output matrix, m * n
    float* f_hevAPtr = new float[p.batch * p.k * p.n];
    float* f_hevBPtr = new float[p.batch * p.m * p.k];
    float* f_hevDPtr = new float[p.batch * p.m * p.n];
    float* f_hevAPtr_trans = new float[p.batch * p.n * p.k]; // For TN GEMM, the weight should be transposed

    // const float max_rand_num = 0.01f;
    for (int b = 0; b < (int)p.batch; b++) {
        for (int i = 0; i < (int)(p.k); i++) {
            for (int j = 0; j < (int)(p.n); j++) {
                // f_hevAPtr[i * p.n + j] = (random() % 1000) / (1000.f / max_rand_num) - (max_rand_num / 2);
                // f_hevAPtr[i * p.n + j] = (random() % 1000) / 1000.f * 2.0f - 1.0f;
                // f_hevAPtr[i * p.n + j] = pow(-1, random() % 2) * (1 << (random() % 8)) * ((random() % 4) * 0.25 + 1.f);
                float exp_num = 1 << (random() % 8);
                float man_num = random() % 4 * 0.25 + 1.f;
                exp_num = random() % 2 == 0 ? exp_num * 1.0f : 1.0f / exp_num;
                man_num = random() % 2 == 0 ? man_num : -1.0f * man_num;
                f_hevAPtr[b * p.k * p.n + i * p.n + j] = exp_num * man_num;
                f_hevAPtr_trans[b * p.k * p.n + j * p.k + i] = f_hevAPtr[b * p.k * p.n + i * p.n + j];
            }
        }
    }

    for (int i = 0; i < (int)(p.batch * p.m * p.k); i++) {
        // f_hevBPtr[i] = (random() % 1000) / 1000.f * 2.0f - 1.0f;
        f_hevBPtr[i] = 1.0f;
    }
    for (int i = 0; i < (int)(p.batch * p.m * p.n); i++) {
        f_hevDPtr[i] = 0.0f;
    }

    // Baseline
    for (int b = 0; b < (int)p.batch; b++) {
        for (int i = 0; i < (int)p.m; i++) {
            for (int j = 0; j < (int)p.n; j++) {
                for (int k = 0; k < (int)p.k; k++) {
                    f_hevDPtr[b * p.m * p.n + i * p.n + j] =
                        f_hevDPtr[b * p.m * p.n + i * p.n + j] + f_hevBPtr[b * p.m * p.k + i * p.k + k] * p.bScale * f_hevAPtr[b * p.k * p.n + k * p.n + j] * p.aScale;
                }
            }
        }
    }
    
    printf("[INFO] fp32 input: \n");
    printMatrix(f_hevBPtr, p.m, p.k);
    printf("[INFO] fp32 weight: \n");
    printMatrix(f_hevAPtr, p.k, p.n);
    printf("[INFO] fp32 output: \n");
    printMatrix(f_hevDPtr, p.batch * p.m, p.n);

    // Collect amax scalar
    float* h_input_amax = new float(0.0f); // per tensor
    float* h_output_scaling = new float[p.n];
    float* h_weight_amax = new float[p.n];

    getAMax(h_input_amax, f_hevBPtr, (int)(p.batch * p.m * p.k), 1);
    // getAMax(h_weight_amax, f_hevAPtr, (int)p.k, (int)p.n);
    for (int i = 0; i < (int)(p.n); i++) h_weight_amax[i] = 1.0f; // No scaling

    for (int i = 0; i < (int)p.n; i++) {
        h_output_scaling[i] = (*h_input_amax) * h_weight_amax[i];
    }

    printf("h_input_amax: %f\n", *h_input_amax);
    for (int i = 0; i < (int)p.n; i++) {
        printf("h_weight_amax[%d]: %f\n", i, h_weight_amax[i]);
    }

    // CUDA side
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::cout << "aScale factor is: " << p.aScale << std::endl;
    std::cout << "bScale factor is: " << p.bScale << std::endl;
    std::cout << "cScale factor is: " << p.cScale << std::endl;
    std::cout << "dScale factor is: " << p.dScale << std::endl;
    std::cout << "epilogueAuxScale factor is: " << p.epilogueAuxScale << std::endl << std::endl;

    const size_t aSizeBytes = p.batch * p.n * p.k * p.aElemSize;
    const size_t bSizeBytes = p.batch * p.m * p.k * p.bElemSize;
    const size_t cSizeBytes = p.batch * p.m * p.n * p.cElemSize;
    const size_t dSizeBytes = p.batch * p.m * p.n * p.dElemSize;
    const size_t wsSizeBytes = 4096 * 1024;

    void* devAPtr = nullptr;
    void* devBPtr = nullptr;
    void* devCPtr = nullptr;
    void* devDPtr = nullptr;
    void* devWsPtr = nullptr;
    checkCudaStatus(cudaMalloc(&devAPtr, aSizeBytes)); // n * k
    checkCudaStatus(cudaMalloc(&devBPtr, bSizeBytes)); // m * k
    checkCudaStatus(cudaMalloc(&devCPtr, cSizeBytes)); // m * n
    checkCudaStatus(cudaMalloc(&devDPtr, dSizeBytes)); // m * n
    checkCudaStatus(cudaMalloc(&devWsPtr, wsSizeBytes));

    void* d_input_amax = nullptr;
    void* d_weight_amax = nullptr;
    void* d_output_scaling = nullptr;
    checkCudaStatus(cudaMalloc(&d_input_amax, sizeof(float) * p.k));
    checkCudaStatus(cudaMalloc(&d_weight_amax, sizeof(float) * p.n));
    checkCudaStatus(cudaMalloc(&d_output_scaling, sizeof(float) * p.n));

    void* f_devAPtr = nullptr;
    void* f_devBPtr = nullptr;
    checkCudaStatus(cudaMalloc(&f_devAPtr, sizeof(float) * p.batch * p.n * p.k));
    checkCudaStatus(cudaMalloc(&f_devBPtr, sizeof(float) * p.batch * p.m * p.k));

    checkCudaStatus(cudaMemcpy((float*)f_devAPtr, (float*)f_hevAPtr_trans, sizeof(float) * p.batch * p.n * p.k, cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemcpy((float*)f_devBPtr, (float*)f_hevBPtr, sizeof(float) * p.batch * p.m * p.k, cudaMemcpyHostToDevice));
    
    checkCudaStatus(cudaMemcpy((float*)d_input_amax, (float*)h_input_amax, sizeof(float) * p.k, cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemcpy((float*)d_weight_amax, (float*)h_weight_amax, sizeof(float) * p.n, cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemcpy((float*)d_output_scaling, (float*)h_output_scaling, sizeof(float) * p.n, cudaMemcpyHostToDevice));
    
    invokeQuatizeVectorE4M3<float, 0>((__nv_fp8_e4m3*)devAPtr, (float*)d_weight_amax, (float*)f_devAPtr, p.batch * p.k * p.n, p.n, stream);
    invokeQuatizeVectorE4M3<float, 0>((__nv_fp8_e4m3*)devBPtr, (float*)d_input_amax, (float*)f_devBPtr, p.batch * p.m * p.k, 1, stream);

    void* devAscalePtr = nullptr;
    void* devBscalePtr = nullptr;
    void* devCscalePtr = nullptr;
    void* devDscalePtr = nullptr;
    void* devEpilogueAuxScalePtr = nullptr;
    void* devAmaxdPtr = nullptr;
    void* devAmaxEpilogueAuxPtr = nullptr;

    // using aDataType = __nv_fp8_e4m3;
    // using bDataType = __nv_fp8_e4m3;
    using cDataType = __nv_bfloat16;
    using dDataType = __nv_bfloat16;
    using computeDataType = float;

    const auto aType = CUDA_R_8F_E4M3;
    const auto bType = CUDA_R_8F_E4M3;
    // const auto cType = CUDA_R_16BF;
    const auto dType = CUDA_R_16BF;
    const auto computeType = CUBLAS_COMPUTE_32F;
    const auto scaleType = CUDA_R_32F;
    const auto epilogueAuxType = CUDA_R_16BF;

    // fillTensorWithValue<aDataType>(devAPtr, aSizeBytes, static_cast<aDataType>(1.0));
    // fillTensorWithValue<bDataType>(devBPtr, bSizeBytes, static_cast<bDataType>(1.0));
    fillTensorWithValue<cDataType>(devCPtr, cSizeBytes, static_cast<cDataType>(1.0));
    fillTensorWithValue<dDataType>(devDPtr, dSizeBytes, static_cast<dDataType>(0.0));

    //----------- handle
    cublasLtHandle_t ltHandle = {0};
    checkCublasStatus(cublasLtCreate(&ltHandle));
    for (int id = 35; id < 36; id++) {
        printf("[INFO] id = %d \n", id);

        //------- init, desc & tensors
        cublasLtMatmulDesc_t matmulDesc;
        cublasLtMatrixLayout_t Adesc;
        cublasLtMatrixLayout_t Bdesc;
        // cublasLtMatrixLayout_t Cdesc;
        cublasLtMatrixLayout_t Ddesc;

        if (p.initExtraPtrs) {
            checkCudaStatus(cudaMalloc(&devAscalePtr, sizeof(computeDataType)));
            checkCudaStatus(
                cudaMemcpy(static_cast<uint8_t*>(devAscalePtr), &p.aScale, sizeof(p.aScale), cudaMemcpyHostToDevice));

            checkCudaStatus(cudaMalloc(&devBscalePtr, sizeof(computeDataType)));
            checkCudaStatus(
                cudaMemcpy(static_cast<uint8_t*>(devBscalePtr), &p.bScale, sizeof(p.bScale), cudaMemcpyHostToDevice));

            checkCudaStatus(cudaMalloc(&devCscalePtr, sizeof(computeDataType)));
            checkCudaStatus(
                cudaMemcpy(static_cast<uint8_t*>(devCscalePtr), &p.cScale, sizeof(p.cScale), cudaMemcpyHostToDevice));

            checkCudaStatus(cudaMalloc(&devDscalePtr, sizeof(computeDataType)));
            checkCudaStatus(
                cudaMemcpy(static_cast<uint8_t*>(devDscalePtr), &p.dScale, sizeof(p.dScale), cudaMemcpyHostToDevice));

            checkCudaStatus(cudaMalloc(&devEpilogueAuxScalePtr, sizeof(computeDataType)));
            checkCudaStatus(cudaMemcpy(static_cast<uint8_t*>(devEpilogueAuxScalePtr),
                                       &p.epilogueAuxScale,
                                       sizeof(p.epilogueAuxScale),
                                       cudaMemcpyHostToDevice));

            const float amax = -99.0f;
            checkCudaStatus(cudaMalloc(&devAmaxdPtr, sizeof(computeDataType)));
            checkCudaStatus(
                cudaMemcpy(static_cast<uint8_t*>(devAmaxdPtr), &amax, sizeof(amax), cudaMemcpyHostToDevice));

            checkCudaStatus(cudaMalloc(&devAmaxEpilogueAuxPtr, sizeof(computeDataType)));
            checkCudaStatus(
                cudaMemcpy(static_cast<uint8_t*>(devAmaxEpilogueAuxPtr), &amax, sizeof(amax), cudaMemcpyHostToDevice));
        }

        {
            checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType));
            checkCublasStatus(
                cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &p.tA, sizeof(p.tA)));
            checkCublasStatus(
                cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &p.tB, sizeof(p.tB)));

            checkCublasStatus(cublasLtMatmulDescSetAttribute(
                matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &devAscalePtr, sizeof(devAscalePtr)));
            checkCublasStatus(cublasLtMatmulDescSetAttribute(
                matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &devBscalePtr, sizeof(devBscalePtr)));
            // checkCublasStatus(cublasLtMatmulDescSetAttribute(
            //     matmulDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &devCscalePtr, sizeof(devCscalePtr)));
            checkCublasStatus(cublasLtMatmulDescSetAttribute(
                matmulDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &devDscalePtr, sizeof(devDscalePtr)));
            checkCublasStatus(cublasLtMatmulDescSetAttribute(
                matmulDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &devAmaxdPtr, sizeof(devAmaxdPtr)));

            checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc,
                                                             CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
                                                             &devEpilogueAuxScalePtr,
                                                             sizeof(devEpilogueAuxScalePtr)));
            checkCublasStatus(cublasLtMatmulDescSetAttribute(
                matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &epilogueAuxType, sizeof(epilogueAuxType)));
            checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc,
                                                             CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER,
                                                             &devAmaxEpilogueAuxPtr,
                                                             sizeof(devAmaxEpilogueAuxPtr)));
        }

        {
            const int64_t lda = p.k;
            const int64_t ldb = p.k;
            // const int64_t ldc = p.n;
            const int64_t ldd = p.n;
            const int batch = p.batch;
            const int64_t strideA = p.k * p.n;
            const int64_t strideB = p.m * p.k;
            // const int64_t strideC = p.m * p.n;
            const int64_t strideD = p.m * p.n;

            // create matrix descriptors, we are good with the details here so no need
            // to set any extra attributes
            checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, aType, p.tA == CUBLAS_OP_N ? p.n : p.k, p.tA == CUBLAS_OP_N ? p.k : p.n, lda));
            checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
            checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));

            checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, bType, p.tB == CUBLAS_OP_N ? p.k : p.m, p.tB == CUBLAS_OP_N ? p.m : p.k, ldb));
            checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
            checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));

            // checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, cType, p.n, p.m, ldc));
            // checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
            // checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));

            checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, dType, p.n, p.m, ldd));
            checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
            checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideD, sizeof(strideD)));
        }

        // cublasLtMatmulAlgo_t algo;

        // size_t sizeWritten = 0;
        // cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL, 0, &sizeWritten);
        // int nbTiles = int(sizeWritten / sizeof(int));
        // printf("[INFO] sizeWritten: %ld, nbTiles: %d \n", sizeWritten, nbTiles);

        // const int32_t algoId = id;

        // checkCublasStatus(cublasLtMatmulAlgoInit(ltHandle,     //
        //                                          computeType,  // compute
        //                                          scaleType,    // scale
        //                                          aType,        // A
        //                                          bType,        // B
        //                                          cType,        // C, this is suspicious! check type matrix!
        //                                          dType,        // D
        //                                          algoId,
        //                                          &algo));

        //---------- algo desc

        for (int tile_id = 0; tile_id < 1; tile_id++) {
            for (int stage_id = 0; stage_id < 1; stage_id++) {
                for (uint16_t gemm_shape = 0; gemm_shape < 1; gemm_shape++) {
                    for (uint16_t cga_shape = 0; cga_shape < 1; cga_shape++) {
                        for (uint16_t scheduling_mode = 0; scheduling_mode < 1; scheduling_mode++) {

                            // sm90_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x128_cgasize1x2x1_stage4_gmmastage2_warpgroupsize1x1x1_tensor64x128x32
                            // cask_5x::ARCH_90, cask_5x::PROBLEM_LAYOUT_TN, CUBLASLT_MATMUL_TILE_128x128,
                            // CUBLASLT_MATMUL_STAGES_128x4, cask_5x::ALIGN_128, cask_5x::GMMA_UNDEFINED,
                            // cask_5x::CGA_1x2x1, cask_5x::SCHEDULING_STATIC, cask_5x::DATA_LAYOUT_REGULAR,
                            // cask_5x::EPILOGUE_DEFAULT,
                            // const cublasLtMatmulTile_t tileId = static_cast<cublasLtMatmulTile_t>(tile_id);
                            // const cublasLtMatmulStages_t stagesId = static_cast<cublasLtMatmulStages_t>(stage_id);
                            // const uint16_t gmmaShape = gemm_shape;
                            // const uint16_t cgaShape = cga_shape;
                            // const uint16_t schedulingMode = scheduling_mode;
                            // const uint16_t dataLayout = 0; // not exposed, yet

                            // {
                                // checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(
                                //     &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileId, sizeof(tileId)));
                                // checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(
                                //     &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesId, sizeof(stagesId)));
                                // checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(
                                //     &algo, CUBLASLT_ALGO_CONFIG_GMMA_SHAPE_ID, &gmmaShape, sizeof(gmmaShape)));
                                // checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(
                                //     &algo, CUBLASLT_ALGO_CONFIG_CGA_SHAPE_ID, &cgaShape, sizeof(cgaShape)));
                                // checkCublasStatus(
                                //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                //                                          CUBLASLT_ALGO_CONFIG_SCHEDULING_MODE,
                                //                                          &schedulingMode,
                                //                                          sizeof(schedulingMode)));
                            // }

                            // cublasLtMatmulHeuristicResult_t heurResult;
                            // cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck(
                            //     ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc, &algo, &heurResult);
                            // if (algoStatus != CUBLAS_STATUS_SUCCESS) {
                            //     // printf("[INFO] algoId: %d, tile_id: %d, stage_id: %d, cublasLtMatmulAlgoCheck fail
                            //     // \r", algoId, tile_id, stage_id);
                            //     continue;
                            // }
                            // else {
                            //     printf(
                            //         "[INFO] algoId: %d, tile_id: %d, stage_id: %d,gemm_shape %d, cga_shape %d, 
                            //         scheduling_mode %d, cublasLtMatmulAlgoCheck success \n", algoId, tile_id,
                            //         stage_id,
                            //         gemm_shape,
                            //         cga_shape,
                            //         scheduling_mode);
                            // }

                            {
                                const float alpha = 1.0f;
                                const float beta = 0.0f;

                                cublasStatus_t status = cublasLtMatmul(ltHandle,
                                                                       matmulDesc,
                                                                       &alpha,
                                                                       devAPtr,
                                                                       Adesc,
                                                                       devBPtr,
                                                                       Bdesc,
                                                                       &beta,
                                                                       devDPtr,
                                                                       Ddesc,
                                                                       devDPtr,
                                                                       Ddesc,
                                                                       nullptr, // &algo,
                                                                       devWsPtr,
                                                                       wsSizeBytes,
                                                                       0);
                                if (status == CUBLAS_STATUS_SUCCESS) {
                                    // printDevBuffer<dDataType>(devDPtr, dSizeBytes, p.m);
                                    // invokeDequatizeVectorE4M3<float, 0>((float*)f_devDPtr, (float*)d_output_amax, (__nv_fp8_e4m3*)devDPtr, (uint32_t)(p.m * p.n), (uint32_t)p.n, stream);
                                    
                                    // invokeDequatizeVectorE4M3<__nv_bfloat16, __nv_bfloat16, 0>((__nv_bfloat16*)devDPtr, (float*)d_output_scaling, (__nv_bfloat16*)devDPtr, (uint32_t)(p.m * p.n), (uint32_t)p.n, stream);
                                    printDevBuffer<dDataType>(devDPtr, dSizeBytes, p.n);
                                    checkMat((__nv_bfloat16*)devDPtr, f_hevDPtr, (int)(p.batch * p.m * p.n), "comparison");
                                    exit(0);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (Ddesc)
            checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
        // if (Cdesc)
        //     checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
        if (Bdesc)
            checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
        if (Adesc)
            checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
        if (matmulDesc)
            checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
    }
    checkCublasStatus(cublasLtDestroy(ltHandle));

    freeResources(&devAPtr, &devBPtr, &devCPtr, &devDPtr, &devWsPtr);

    if (p.initExtraPtrs) {
        checkCudaStatus(cudaFree(devAscalePtr));
        checkCudaStatus(cudaFree(devBscalePtr));
        checkCudaStatus(cudaFree(devCscalePtr));
        checkCudaStatus(cudaFree(devDscalePtr));
        checkCudaStatus(cudaFree(devEpilogueAuxScalePtr));

        computeDataType amax = -99.9f;
        auto status = cudaMemcpy(&amax, devAmaxdPtr, sizeof(computeDataType), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            std::cout << "Failed to copy dev buffer's content!" << std::endl;
        }
        else {
            std::cout << "amaxD is: " << amax << std::endl;
        }
        checkCudaStatus(cudaFree(devAmaxdPtr));

        amax = -99.9f;
        status = cudaMemcpy(&amax, devAmaxEpilogueAuxPtr, sizeof(computeDataType), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            std::cout << "Failed to copy dev buffer's content!" << std::endl;
        }
        else {
            std::cout << "amaxEpilogueAux is: " << amax << std::endl;
        }
        checkCudaStatus(cudaFree(devAmaxEpilogueAuxPtr));
    }

    std::cout << std::endl;
    return 0;
}
