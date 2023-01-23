#ifndef _WORKER_HPP
#define _WORKER_HPP

#include <cstdint>
#include <cstdlib>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


/* **************************** debug tools ********************************* */
static const char* _cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorString(error);
}

static const char* _cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

inline void syncAndCheck(const char* const file, int const line)
{
#ifndef NDEBUG
    cudaDeviceSynchronize();
    cudaError_t result = cudaGetLastError();
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
#endif
}

#define sync_check_cuda_error() syncAndCheck(__FILE__, __LINE__)


struct ProblemDesc {
  size_t m;
  size_t n;
  size_t k;
  cublasOperation_t tA;
  cublasOperation_t tB;
  size_t batch;
  bool isStridedBatch;
  size_t aElemSize;
  size_t bElemSize;
  size_t cElemSize;
  size_t dElemSize;
  float aScale;
  float bScale;
  float cScale;
  float dScale;
  float epilogueAuxScale;
  bool initExtraPtrs;
};

//------------------------------------------------

template<typename T>
void silence_unused(const T&) {}

//------------------------------------------------

int32_t cublasTester(const ProblemDesc& problem);

template<typename TA, typename TB, typename TC, typename TD>
int32_t cublasTesterNew(const ProblemDesc& p);

#endif  // _WORKER_HPP
