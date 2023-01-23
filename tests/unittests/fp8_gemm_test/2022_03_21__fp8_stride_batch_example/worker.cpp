#include "worker.hpp"

#include <cublasLt.h>
#include <cuda_runtime_api.h>

//------------------------------------------------

void checkCublasStatus(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("cuBLAS API failed with status %d\n", status);
    throw std::logic_error("cuBLAS API failed");
  }
}

void checkCudaStatus(cudaError_t status) {
  if (status != cudaSuccess) {
    printf("CUDA API failed with status %d\n", status);
    throw std::logic_error("CUDA API failed");
  }
}

//------------------------------------------------

void freeResources(void** a, void** b, void** c, void** d, void** ws) {
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
