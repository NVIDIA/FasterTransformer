#include "worker.hpp"
#include <cuda_bf16.h>
// #include "3rdparty/swarm-review-30931053/cudart/cuda_fp8.h"
#include <cuda_fp8.h>
#include <iostream>
#include "cuda_utils.h"

//=====================================================================================================================

int main()
{
    const int32_t m = 16;
    const int32_t n = 32;
    const int32_t k = 64;
    const int32_t batch = 2;
    const cublasOperation_t tA = CUBLAS_OP_T;
    const cublasOperation_t tB = CUBLAS_OP_N;
    const bool isStridedBatch = true;
    const size_t aElemSize = sizeof(__nv_fp8_e4m3);
    const size_t bElemSize = sizeof(__nv_fp8_e4m3);
    const size_t cElemSize = sizeof(__nv_bfloat16);
    const size_t dElemSize = sizeof(__nv_bfloat16);
    const float aScale = 1.0f;
    const float bScale = 2.0f;
    const float cScale = 0.0f;
    const float dScale = 1.0f;
    const float epilogueAuxScale = 1.0f;
    const bool initExtraPtrs = true;

    const auto result = cublasTester({m, n, k, tA, tB, batch, isStridedBatch, aElemSize, bElemSize, cElemSize,
    dElemSize, aScale, bScale, cScale, dScale, epilogueAuxScale, initExtraPtrs});

    if (result != 0) {
        std::cout << "In the end, something went wrong!" << std::endl;
    }
    else {
        std::cout << "Everything is ok" << std::endl;
    }
}
