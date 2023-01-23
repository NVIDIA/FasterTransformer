#include "worker.hpp"
#include <iostream>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

//=====================================================================================================================

int main() {
  const int32_t m = 64;
  const int32_t n = 63;
  const int32_t k = 64;
  const int32_t batch = 2;
  const bool tA = true;
  const bool tB = false;
  const bool isStridedBatch = true;
  const float aScale = 1.0f;
  const float bScale = 1.0f;
  const float cScale = 1.0f;
  const float dScale = 1.0f;
  const float epilogueAuxScale = 1.0f;
  const bool initExtraPtrs = true;

  const bool useHeuristics = false;

  const auto result = cublasTester<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16, float>
    ({m, n, k, tA, tB, batch, isStridedBatch, aScale, bScale, cScale, dScale, epilogueAuxScale, initExtraPtrs, useHeuristics});

  if (result != 0) {
    std::cout << "In the end, something went wrong!" << std::endl;
  }
  else {
    std::cout << "Everything is ok" << std::endl;
  }
}
