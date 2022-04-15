/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/utils/cuda_utils.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <utility>

#pragma once
namespace fastertransformer {

#define GEMM_NUM 6
#define GEMM_CONFIG "gemm_config.in"
#define IGEMM_CONFIG "igemm_config.in"
#define SPGEMM_CONFIG "spgemm_config.in"
#define SPIGEMM_CONFIG "spigemm_config.in"

typedef struct {
    int algoId, customOption, tile, splitK_val;
    int swizzle, reductionScheme, workspaceSize;
    // only used in cublasLt >= 11.0
    int stages;
    float exec_time;
} cublasLtMatmulAlgo_info;

/* Structure to store information about different run trials */
typedef struct {
    cublasLtMatmulAlgo_t algo;
    cublasStatus_t status;
    float time;
    size_t workspaceSize;  // actual memory workspace needed
    cublasMath_t mathMode;
    cublasLtReductionScheme_t reductionScheme;
    int customOption;
    float wavesCount;
} customMatmulPerf_t;

class cublasAlgoMap {
private:
    std::map<std::string, cublasLtMatmulAlgo_info> algo_map_;
    std::string config_filename_;
    std::string sp_config_filename_;
    std::map<std::string, int> sp_algo_map_;

public:
    explicit cublasAlgoMap(const std::string filename, const std::string sp_config_filename = "");
    cublasAlgoMap(const cublasAlgoMap& map);
    ~cublasAlgoMap();
    void loadGemmConfig();
    void loadSpGemmConfig();
    int getSpAlgo(const int batch_count, const int m, const int n, const int k);
    bool isUseSparse(const int batch_count, const int m, const int n, const int k);

    bool isExist(const int batch_count, const int m, const int n, const int k, const CublasDataType data_type);

    cublasLtMatmulAlgo_info
    getAlgo(const int batch_count, const int m, const int n, const int k, const CublasDataType data_type);
};

}  // namespace fastertransformer
