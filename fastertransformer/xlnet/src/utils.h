/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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




#pragma once
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <assert.h>
#include"cnpy.h"
#include <nvml.h>
#include <string>
#include <iomanip>
#include "string.h"

/*************MACRO FUNCTION**************/
#define FINAL_MASK 0xffffffff
#define MAX_HIDDEN_DIM 1024
#define NUM_CUBLAS_FUNC 10
#define FULL_GEMM_LENGTH 23
#define VERIFY_NUM 7

//#define SIZE_DICTION 32000
//#define DATA_TYPE __half
//#define DATA_TYPE float

enum cublasFunction{
    GEMM_STRIDE=0,
    GEMM_A_0=1,
    GEMM_B_0=2
};

enum RUN_MODE{
    FP16_TIME_TEST=0,
    FP16_CORRECTNESS_TEST=1,
    FP32_TIME_TEST=2,
    FP32_CORRECTNESS_TEST=3,
};

/*************Error Handling**************/
bool check(cudaError_t e, int iLine, const char *szFile);
bool check(cublasStatus_t e, int iLine, const char *szFile);
#define ck(call) check(call, __LINE__, __FILE__)
#define PRINT_FUNC_NAME_() do{\
    std::cout << "[FT][CALL] " << __FUNCTION__ << " " << std::endl; \
} while (0)

/*************Time Handling**************/
class CudaTimer{
    private:
        cudaEvent_t event_start;
        cudaEvent_t event_stop;
        cudaStream_t stream;
        float time;
    public:
        CudaTimer(cudaStream_t stream=0);
        void start();
        float stop();
        ~CudaTimer();
};
/*************Useful functions***********************/
int blockNum(int size, int blockSize);
int next_pow2(int a);
template <typename T> int numPerThread();
template <typename T> void deviceMalloc(T** ptr, int size);
template <typename T> void deviceMemset(T* ptr, int value, int size);
template <typename T> void deviceFree(T* & ptr);
template <typename T> void deviceMemcpyHtoD(cudaStream_t stream, T* d_ptr,T* h_ptr, int size);
template <typename T> float castToFloat(T input);
 
/*********************Npz &Npy File Process functions***********************/
std::string paraName(int i_layer, std::string sub_para);
std::string paraName(std::string s);
void setByNpz(cudaStream_t stream, cnpy::npz_t & my_npz, std::string name, int* d_ptr, int size, int offset=0);
void setByNpz(cudaStream_t stream, cnpy::npz_t & my_npz, std::string name, float* d_ptr, int size, int offset=0);
void setByNpz(cudaStream_t stream, cnpy::npz_t & my_npz, std::string name, __half* d_ptr, int size, int offset=0);
void setByNpy(cudaStream_t stream, float* d_ptr, int size,std::string dir="./", std::string fname="tmp.npy");
void setByNpy(cudaStream_t stream, __half* d_ptr, int size,std::string dir="./", std::string fname="tmp.npy");

template <typename T> void setByNpz(cnpy::npz_t & my_npz, std::string name, T* h_ptr, int size, int offset=0);

void checkByNpy(cudaStream_t stream, float* d_ptr, int size,std::string dir="./", std::string fname="tmp.npy");
void checkByNpy(cudaStream_t stream, __half* d_ptr, int size,std::string dir="./", std::string fname="tmp.npy");
void checkByNpz(cudaStream_t stream,std::string data_fname, std::string name, float* d_ptr, int size);
void checkByNpz(cudaStream_t stream,std::string data_fname, std::string name, __half* d_ptr, int size);

template <typename T> bool checkByNpz(cnpy::npz_t& data_npz,cudaStream_t stream,std::string name, T* d_ptr, int size);
void printKey(cnpy::npz_t & npz);
