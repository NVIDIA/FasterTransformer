/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "fastertransformer/open_decoder.h"
#include "fastertransformer/gpt.h"
#include "fastertransformer/utils/INIReader.h"
#include "fastertransformer/utils/common.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_profiler_api.h>
#include <iostream>
#include <sys/time.h>
#include <cuda_fp16.h>

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <memory>
#include <utility>

using namespace fastertransformer;


#ifdef USE_TRITONSERVER_DATATYPE

#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"
typedef TRITONSERVER_DataType DataType;
typedef TRITONSERVER_MemoryType MemoryType;

constexpr TRITONSERVER_DataType TYPE_INVALID = TRITONSERVER_TYPE_INVALID;
constexpr TRITONSERVER_DataType TYPE_BOOL    = TRITONSERVER_TYPE_BOOL;
constexpr TRITONSERVER_DataType TYPE_UINT8   = TRITONSERVER_TYPE_UINT8;
constexpr TRITONSERVER_DataType TYPE_UINT16  = TRITONSERVER_TYPE_UINT16;
constexpr TRITONSERVER_DataType TYPE_UINT32  = TRITONSERVER_TYPE_UINT32;
constexpr TRITONSERVER_DataType TYPE_UINT64  = TRITONSERVER_TYPE_UINT64;
constexpr TRITONSERVER_DataType TYPE_INT8    = TRITONSERVER_TYPE_INT8;
constexpr TRITONSERVER_DataType TYPE_INT16   = TRITONSERVER_TYPE_INT16;
constexpr TRITONSERVER_DataType TYPE_INT32   = TRITONSERVER_TYPE_INT32;
constexpr TRITONSERVER_DataType TYPE_INT64   = TRITONSERVER_TYPE_INT64;
constexpr TRITONSERVER_DataType TYPE_FP16    = TRITONSERVER_TYPE_FP16;
constexpr TRITONSERVER_DataType TYPE_FP32    = TRITONSERVER_TYPE_FP32;
constexpr TRITONSERVER_DataType TYPE_FP64    = TRITONSERVER_TYPE_FP64;
constexpr TRITONSERVER_DataType TYPE_BYTES   = TRITONSERVER_TYPE_BYTES;
constexpr TRITONSERVER_MemoryType MEMORY_CPU        = TRITONSERVER_MEMORY_CPU;
constexpr TRITONSERVER_MemoryType MEMORY_CPU_PINNED = TRITONSERVER_MEMORY_CPU_PINNED;
constexpr TRITONSERVER_MemoryType MEMORY_GPU        = TRITONSERVER_MEMORY_GPU;

#else
typedef enum datatype_enum {
  TYPE_INVALID,
  TYPE_BOOL,
  TYPE_UINT8,
  TYPE_UINT16,
  TYPE_UINT32,
  TYPE_UINT64,
  TYPE_INT8,
  TYPE_INT16,
  TYPE_INT32,
  TYPE_INT64,
  TYPE_FP16,
  TYPE_FP32,
  TYPE_FP64,
  TYPE_BYTES
} DataType;

typedef enum memorytype_enum {
  MEMORY_CPU,
  MEMORY_CPU_PINNED,
  MEMORY_GPU
} MemoryType;

#endif

//struct Request {
//  //  Request(int* start_ids, std::vector<int> shape) : start_ids(start_ids), shape(shape) {}
//  int* start_ids;
//  std::vector<int> shape;
//};

struct Tensor {
  //std::string name;
  const MemoryType where;
  const DataType type;
  const std::vector<int64_t> shape;
  const void* data;
};

struct AbstractParamInstance
{
  virtual AbstractParam* get_param_ptr(std::string param_name) = 0;
  virtual void free_model_param() = 0;
  virtual void init_nccl_from_ids(std::vector<ncclUniqueId> nccl_ids) = 0;
  virtual void init_nccl_from_comms(ncclComm_t tensor_para_nccl_comm, ncclComm_t layer_para_nccl_comm) = 0;
};

struct AbstractTransformerModelInstance
{
  virtual std::shared_ptr<std::vector<Tensor>> forward(std::shared_ptr<std::vector<Tensor>> input_tensors) = 0;
  virtual void set_param(AbstractParamInstance* param_instance) = 0;
};

struct AbstractTransformerModel {
  static std::shared_ptr<AbstractTransformerModel> createGptModel (std::string inifile);
  virtual std::vector<ncclUniqueId> create_nccl_ids(const uint32_t world_size) = 0;
  virtual std::unique_ptr<AbstractTransformerModelInstance> createModelInstance (int nodeId, int deviceId, int world_size, cudaStream_t stream) = 0;
  virtual std::unique_ptr<AbstractParamInstance> createParamInstance(int nodeId, int deviceId, int world_size, cudaStream_t stream, std::vector<ncclUniqueId> nccl_ids) = 0;
  virtual std::string to_string() = 0;
  virtual std::pair<uint32_t, uint32_t> get_max_batch_seqlen() = 0;
  virtual int get_tensor_para_size() = 0;
  virtual int get_layer_para_size() = 0;
};
