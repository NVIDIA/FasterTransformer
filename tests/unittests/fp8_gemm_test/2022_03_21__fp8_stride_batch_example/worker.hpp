#ifndef _WORKER_HPP
#define _WORKER_HPP

#include <cstdint>
#include <cstdlib>

#include <cublasLt.h>
#include <library_types.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <vector>

//--------------------------------------------------------------------------------------------------

struct ProblemDesc {
  size_t m;
  size_t n;
  size_t k;
  int32_t tA;
  int32_t tB;
  int32_t batch;
  bool isStridedBatch;
  float aScale;
  float bScale;
  float cScale;
  float dScale;
  float epilogueAuxScale;
  bool initExtraPtrs;
  bool useHeuristics;
};

//--------------------------------------------------------------------------------------------------

template <typename T>
bool fillTensorWithValue(void* devPtr, const size_t sizeBytes, const T value) {
  std::vector<T> hData(sizeBytes / sizeof(T), value);
  const auto status = cudaMemcpy(static_cast<uint8_t*>(devPtr), hData.data(), hData.size() * sizeof(T), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    return false;
  }
  return true;
};

template <typename T>
void printDevBuffer(void* devPtr, const size_t sizeBytes, const size_t row) {
  std::vector<T> hBuffer(sizeBytes / sizeof(T), static_cast<T>(0.0));
  const auto status = cudaMemcpy(hBuffer.data(), devPtr, hBuffer.size() * sizeof(T), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    std::cout << "Failed to copy dev buffer's content!" << std::endl;
    return;
  }
  for (size_t i = 0; i < hBuffer.size(); i++) {
    if (!(i % row)) {
      std::cout << std::endl;
    }
    std::cout << (float)hBuffer[i] << " ";
  }
  std::cout << std::endl << std::endl;
}

//--------------------------------------------------------------------------------------------------

template<typename T>
void silence_unused(const T&) {}

//--------------------------------------------------------------------------------------------------

void checkCublasStatus(cublasStatus_t status);
void checkCudaStatus(cudaError_t status);
void freeResources(void** a, void** b, void** c, void** d, void** ws);

//--------------------------------------------------------------------------------------------------

struct Timer {
  Timer(const char* _msg) : msg(_msg) { _start = getTick(); }
  ~Timer() {
    const auto duration = getTick() - _start;
    printf("^^^^^^^^ [Timer] %s: %lu\n", msg, duration);
  }

 private:
  uint64_t getTick() {
    uint64_t low, high;
    __asm__ volatile("rdtsc" : "=a"(low), "=d"(high));
    return (high << 32) | low;
  }
  uint64_t _start = 0;
  const char* msg;
};

//--------------------------------------------------------------------------------------------------

template<typename T>
struct TypeTrait;

template<>
struct TypeTrait<__nv_fp8_e4m3>{
  static constexpr cudaDataType_t cudaType = CUDA_R_8F_E4M3;
  static constexpr cublasComputeType_t compType = CUBLAS_COMPUTE_32F;
};
template<>
struct TypeTrait<__nv_fp8_e5m2>{
  static constexpr cudaDataType_t cudaType = CUDA_R_8F_E5M2;
  static constexpr cublasComputeType_t compType = CUBLAS_COMPUTE_32F;
};
template<>
struct TypeTrait<__nv_bfloat16>{
  static constexpr cudaDataType_t cudaType = CUDA_R_16BF;
  static constexpr cublasComputeType_t compType = CUBLAS_COMPUTE_32F;
};
template<>
struct TypeTrait<float>{
  static constexpr cudaDataType_t cudaType = CUDA_R_32F;
  static constexpr cublasComputeType_t compType = CUBLAS_COMPUTE_32F;
};

//--------------------------------------------------------------------------------------------------

template<typename aType, typename bType, typename cType, typename dType, typename computeType>
int32_t cublasTester(const ProblemDesc& p) {
  const size_t aSizeBytes = p.batch * p.m * p.k * sizeof(aType);
  const size_t bSizeBytes = p.batch * p.k * p.n * sizeof(bType);
  const size_t cSizeBytes = p.batch * p.m * p.n * sizeof(cType);
  const size_t dSizeBytes = p.batch * p.m * p.n * sizeof(dType);
  const size_t wsSizeBytes = 128 * 1024 * 1024;

  void* devAPtr = nullptr;
  void* devBPtr = nullptr;
  void* devCPtr = nullptr;
  void* devDPtr = nullptr;
  void* devWsPtr = nullptr;
  checkCudaStatus(cudaMalloc(&devAPtr, aSizeBytes));
  checkCudaStatus(cudaMalloc(&devBPtr, bSizeBytes));
  checkCudaStatus(cudaMalloc(&devCPtr, cSizeBytes));
  checkCudaStatus(cudaMalloc(&devDPtr, dSizeBytes));
  checkCudaStatus(cudaMalloc(&devWsPtr, wsSizeBytes));

  void* devAscalePtr = nullptr;
  void* devBscalePtr = nullptr;
  void* devCscalePtr = nullptr;
  void* devDscalePtr = nullptr;
  void* devEpilogueAuxScalePtr = nullptr;
  void* devAmaxdPtr = nullptr;
  void* devAmaxEpilogueAuxPtr = nullptr;

  const auto aTypeLabel = TypeTrait<aType>::cudaType;
  const auto bTypeLabel = TypeTrait<bType>::cudaType;
  const auto cTypeLabel = TypeTrait<cType>::cudaType;
  const auto dTypeLabel = TypeTrait<dType>::cudaType;
  const auto computeTypeLabel = TypeTrait<computeType>::compType;
  const auto scaleTypeLabel = TypeTrait<computeType>::cudaType;
  const auto epilogueAuxTypeLabel = TypeTrait<dType>::cudaType;

  fillTensorWithValue<aType>(devAPtr, aSizeBytes, static_cast<aType>(1.0));
  fillTensorWithValue<bType>(devBPtr, bSizeBytes, static_cast<bType>(1.0));
  fillTensorWithValue<cType>(devCPtr, cSizeBytes, static_cast<cType>(1.0));
  fillTensorWithValue<dType>(devDPtr, dSizeBytes, static_cast<dType>(0.0));

  //------- init, desc & tensors
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t Adesc;
  cublasLtMatrixLayout_t Bdesc;
  cublasLtMatrixLayout_t Cdesc;
  cublasLtMatrixLayout_t Ddesc;

  if (p.initExtraPtrs) {
    std::cout << "aScale factor is: " << p.aScale << std::endl;
    std::cout << "bScale factor is: " << p.bScale << std::endl;
    std::cout << "cScale factor is: " << p.cScale << std::endl;
    std::cout << "dScale factor is: " << p.dScale << std::endl;
    std::cout << "epilogueAuxScale factor is: " << p.epilogueAuxScale << std::endl << std::endl;

    checkCudaStatus(cudaMalloc(&devAscalePtr, sizeof(computeType)));
    checkCudaStatus(cudaMemcpy(static_cast<uint8_t*>(devAscalePtr), &p.aScale, sizeof(p.aScale), cudaMemcpyHostToDevice));

    checkCudaStatus(cudaMalloc(&devBscalePtr, sizeof(computeType)));
    checkCudaStatus(cudaMemcpy(static_cast<uint8_t*>(devBscalePtr), &p.bScale, sizeof(p.bScale), cudaMemcpyHostToDevice));

    checkCudaStatus(cudaMalloc(&devCscalePtr, sizeof(computeType)));
    checkCudaStatus(cudaMemcpy(static_cast<uint8_t*>(devCscalePtr), &p.cScale, sizeof(p.cScale), cudaMemcpyHostToDevice));

    checkCudaStatus(cudaMalloc(&devDscalePtr, sizeof(computeType)));
    checkCudaStatus(cudaMemcpy(static_cast<uint8_t*>(devDscalePtr), &p.dScale, sizeof(p.dScale), cudaMemcpyHostToDevice));

    checkCudaStatus( cudaMalloc(&devEpilogueAuxScalePtr, sizeof(computeType)));
    checkCudaStatus(cudaMemcpy(static_cast<uint8_t*>(devEpilogueAuxScalePtr), &p.epilogueAuxScale, sizeof(p.epilogueAuxScale), cudaMemcpyHostToDevice));

    const float amax = -99.0f;
    checkCudaStatus(cudaMalloc(&devAmaxdPtr, sizeof(computeType)));
    checkCudaStatus(cudaMemcpy(static_cast<uint8_t*>(devAmaxdPtr), &amax, sizeof(amax), cudaMemcpyHostToDevice));

    checkCudaStatus( cudaMalloc(&devAmaxEpilogueAuxPtr, sizeof(computeType)));
    checkCudaStatus(cudaMemcpy(static_cast<uint8_t*>(devAmaxEpilogueAuxPtr), &amax, sizeof(amax), cudaMemcpyHostToDevice));
  }

  {
    checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, computeTypeLabel, scaleTypeLabel));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &p.tA, sizeof(p.tA)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &p.tB, sizeof(p.tB)));

    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &devAscalePtr, sizeof(devAscalePtr)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &devBscalePtr, sizeof(devBscalePtr)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &devCscalePtr, sizeof(devCscalePtr)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &devDscalePtr, sizeof(devDscalePtr)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &devAmaxdPtr, sizeof(devAmaxdPtr)));

    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER, &devEpilogueAuxScalePtr, sizeof(devEpilogueAuxScalePtr)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &epilogueAuxTypeLabel, sizeof(epilogueAuxTypeLabel)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER, &devAmaxEpilogueAuxPtr, sizeof(devAmaxEpilogueAuxPtr)));
  }

  {
    const int64_t lda = p.m;
    const int64_t ldb = p.k;
    const int64_t ldc = p.m;
    const int64_t ldd = p.m;
    const int64_t strideA = p.m * p.k;
    const int64_t strideB = p.k * p.n;
    const int64_t strideC = p.m * p.n;
    const int64_t strideD = p.m * p.n;

    // create matrix descriptors, we are good with the details here so no need
    // to set any extra attributes
    checkCublasStatus(cublasLtMatrixLayoutCreate(
        &Adesc, aTypeLabel, p.tA == CUBLAS_OP_N ? p.m : p.k,
        p.tA == CUBLAS_OP_N ? p.k : p.m, lda));
    if (p.batch > 1) {
      checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc,
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &p.batch, sizeof(p.batch)));
      checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc,
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
    }

    checkCublasStatus(cublasLtMatrixLayoutCreate(
        &Bdesc, bTypeLabel, p.tB == CUBLAS_OP_N ? p.k : p.n,
        p.tB == CUBLAS_OP_N ? p.n : p.k, ldb));
    if (p.batch > 1) {
      checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc,
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &p.batch, sizeof(p.batch)));
      checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc,
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
    }

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, cTypeLabel, p.m, p.n, ldc));
    if (p.batch > 1) {
      checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc,
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &p.batch, sizeof(p.batch)));
      checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc,
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));
    }

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, dTypeLabel, p.m, p.n, ldd));
    if (p.batch > 1) {
      checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Ddesc,
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &p.batch, sizeof(p.batch)));
      checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Ddesc,
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideD, sizeof(strideD)));
    }
  }

  //----------- handle
  cublasLtHandle_t ltHandle = {0};
  checkCublasStatus(cublasLtCreate(&ltHandle));
  

  //----------- heuristics / explicit kernel
  cublasLtMatmulAlgo_t* algoPtr = nullptr;
  cublasLtMatmulAlgo_t algo;

  if (!p.useHeuristics) {
    const int32_t algoId = 35;
  
    checkCublasStatus(cublasLtMatmulAlgoInit(
        ltHandle,          //
        computeTypeLabel,  // compute
        scaleTypeLabel,    // scale
        aTypeLabel,        // A
        bTypeLabel,        // B
        cTypeLabel,        // C, this is suspicious! check type matrix!
        dTypeLabel,        // D
        algoId, &algo));
  
    // sm90_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x128_cgasize1x2x1_stage4_gmmastage2_warpgroupsize1x1x1_tensor64x128x32
    // ARCH_90, BLAS_OP_GEMM, PROBLEM_LAYOUT_TN, CUBLASLT_MATMUL_TILE_128x128, CUBLASLT_MATMUL_STAGES_128x4, ALIGN_128, GMMA_UNDEFINED, CGA_1x2x1, SCHEDULING_STATIC, DATA_LAYOUT_REGULAR, EPILOGUE_DEFAULT
    const cublasLtMatmulTile_t tileId = static_cast<cublasLtMatmulTile_t>(20);
    const cublasLtMatmulStages_t stagesId = static_cast<cublasLtMatmulStages_t>(22);
    const uint16_t gmmaShape = 0;
    const uint16_t cgaShape = 2;
    const uint16_t schedulingMode = 0;
    // const uint16_t dataLayout = 0; // not exposed, yet

    checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileId, sizeof(tileId)));
    checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesId, sizeof(stagesId)));
    checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_GMMA_SHAPE_ID, &gmmaShape, sizeof(gmmaShape)));
    checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CGA_SHAPE_ID, &cgaShape, sizeof(cgaShape)));
    checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SCHEDULING_MODE, &schedulingMode, sizeof(schedulingMode)));

    algoPtr = &algo;
  }

  {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // in-place matrix-multi -> C == D & CDesc == DDesc
    checkCublasStatus(cublasLtMatmul(
        ltHandle, matmulDesc, &alpha, devAPtr, Adesc, devBPtr, Bdesc, &beta,
        devDPtr, Ddesc, devDPtr, Ddesc, algoPtr, devWsPtr, wsSizeBytes, 0));
  }

  printDevBuffer<dType>(devDPtr, dSizeBytes, p.m);

  if (Ddesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (matmulDesc) checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
  checkCublasStatus(cublasLtDestroy(ltHandle));

  freeResources(&devAPtr, &devBPtr, &devCPtr, &devDPtr, &devWsPtr);

  if (p.initExtraPtrs) {
    checkCudaStatus(cudaFree(devAscalePtr));
    checkCudaStatus(cudaFree(devBscalePtr));
    checkCudaStatus(cudaFree(devCscalePtr));
    checkCudaStatus(cudaFree(devDscalePtr));
    checkCudaStatus(cudaFree(devEpilogueAuxScalePtr));

    computeType amax = -99.9f;
    auto status = cudaMemcpy(&amax, devAmaxdPtr, sizeof(computeType), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      std::cout << "Failed to copy dev buffer's content!" << std::endl;
    } else {
      std::cout << "amaxD is: " << amax << std::endl;
    }
    checkCudaStatus(cudaFree(devAmaxdPtr));

    amax = -99.9f;
    status = cudaMemcpy(&amax, devAmaxEpilogueAuxPtr, sizeof(computeType), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      std::cout << "Failed to copy dev buffer's content!" << std::endl;
    } else {
      std::cout << "amaxEpilogueAux is: " << amax << std::endl;
    }
    checkCudaStatus(cudaFree(devAmaxEpilogueAuxPtr));
  }

  std::cout << std::endl;
  return 0;
}

#endif  // _WORKER_HPP
