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

#include "fastertransformer/triton_backend/transformer.hpp"

using namespace fastertransformer;

template <fastertransformer::OperationType OpType>
struct GptModel : public AbstractTransformerModel {
  GptModel
  (size_t batch_size = 0,
   size_t candidate_num = 0,
   size_t head_num = 0,
   size_t size_per_head = 0,
   size_t vocab_size = 0,
   size_t max_seq_len = 0,
   size_t decoder_layers = 0,
   size_t tensor_para_size = 0,
   size_t layer_para_size = 0,
   size_t layer_para_batch_size = 0,
   const float probability_threshold = 0.0,
   const bool is_fuse_QKV = true,
   const std::string model_name = "",
   const std::string model_path_prefix = "")
    : batch_size(batch_size),
      candidate_num(candidate_num),
      head_num(head_num),
      size_per_head(size_per_head),
      vocab_size(vocab_size),
      max_seq_len(max_seq_len),
      decoder_layers(decoder_layers),
      tensor_para_size(tensor_para_size),
      layer_para_size(layer_para_size),
      layer_para_batch_size(layer_para_batch_size),
      probability_threshold(probability_threshold),
      is_fuse_QKV(is_fuse_QKV),
      model_name(model_name),
      model_path_prefix(model_path_prefix){}

  typedef DecoderTransformerTraits<OpType> Traits;
  typedef typename Traits::DataType DataType;
  size_t batch_size;
  size_t candidate_num;
  size_t head_num;
  size_t size_per_head;
  size_t vocab_size;
  size_t max_seq_len;
  size_t decoder_layers;
  size_t tensor_para_size;
  size_t layer_para_size;
  size_t layer_para_batch_size;
  const float probability_threshold;
  const bool is_fuse_QKV;
  const std::string model_name;
  const std::string model_path_prefix;

  virtual std::unique_ptr<AbstractTransformerModelInstance> createModelInstance (int nodeId, int deviceId, int world_size, cudaStream_t stream);
  virtual std::unique_ptr<AbstractParamInstance> createParamInstance(int nodeId, int deviceId, int world_size, cudaStream_t stream, std::vector<ncclUniqueId> nccl_ids);
  virtual std::string to_string() {
    std::stringstream ss;
    ss << "Model:"
       << "\nbatch_size: : " << batch_size
       << "\ncandidate_num: " << candidate_num
       << "\nhead_num: " << head_num
       << "\nsize_per_head: " << size_per_head
       << "\nvocab_size: " << vocab_size
       << "\nmax_seq_len: " << max_seq_len
       << "\ndecoder_layers: " << decoder_layers
       << "\ntensor_para_size: " << tensor_para_size
       << "\nlayer_para_size: " << layer_para_size
       << "\nlayer_para_batch_size: " << layer_para_batch_size
       << "\nprobability_threshold: " << probability_threshold
       << "\nis_fuse_QKV: " << is_fuse_QKV
       << "\nmodel_name: " << model_name
       << "\nmodel_path_prefix: " << model_path_prefix << std::endl;
    return ss.str();
  }

  virtual std::vector<ncclUniqueId> create_nccl_ids(const uint32_t world_size)
  {
    assert(world_size % tensor_para_size == 0);
    if(world_size != tensor_para_size * layer_para_size)
    {
      printf("[ERROR] world_size (%d) should equal to tensor_para_size * layer_para_size (%ld) \n", world_size, tensor_para_size * layer_para_size);
      exit(-1);
    }
    std::vector<ncclUniqueId> nccl_ids(tensor_para_size + layer_para_size);
    for(uint32_t i = 0; i < nccl_ids.size(); i++)
    {
      NCCLCHECK(ncclGetUniqueId(&nccl_ids[i]));
    }
    return nccl_ids;
  }

  virtual std::pair<uint32_t, uint32_t> get_max_batch_seqlen()
  {
    return std::pair<uint32_t, uint32_t>(batch_size, max_seq_len);
  }
  virtual int get_tensor_para_size()
  {
    return tensor_para_size;
  }
  virtual int get_layer_para_size()
  {
    return layer_para_size;
  }

private:
};

template <fastertransformer::OperationType OpType>
struct GptParamInstance : AbstractParamInstance
{
  typedef DecoderTransformerTraits<OpType> Traits;
  typedef typename Traits::DataType DataType;

  std::unique_ptr<DecoderInitParam<DataType>[]> decoder_params;
  DecodingInitParam<DataType> decoding_params;
  LayerParallelParam layer_parallel_params;
  TensorParallelParam tensor_parallel_params;
  cudaStream_t stream;
  cublasHandle_t cublasHandle;

  uint64_t batch_size_;
  uint64_t head_num_;
  uint64_t size_per_head_;
  uint64_t tensor_para_size_;
  uint64_t layer_para_size_;
  uint64_t world_size_;
  uint64_t rank_;
  uint64_t decoder_layers_;
  uint64_t vocab_size_;
  uint64_t max_seq_len_;
  uint64_t layer_para_batch_size_;
  std::string model_path_prefix_;

  GptParamInstance(uint64_t batch_size,
                   uint64_t head_num,
                   uint64_t size_per_head,
                   uint64_t tensor_para_size,
                   uint64_t layer_para_size,
                   uint64_t world_size,
                   uint64_t rank,
                   uint64_t decoder_layers,
                   uint64_t vocab_size,
                   uint64_t max_seq_len,
                   uint64_t layer_para_batch_size,
                   std::string model_path_prefix,
                   cudaStream_t stream,
                   std::vector<ncclUniqueId> nccl_ids) :
                          batch_size_(batch_size),
                          head_num_(head_num),
                          size_per_head_(size_per_head),
                          tensor_para_size_(tensor_para_size),
                          layer_para_size_(layer_para_size),
                          world_size_(world_size),
                          rank_(rank),
                          decoder_layers_(decoder_layers),
                          vocab_size_(vocab_size),
                          max_seq_len_(max_seq_len),
                          layer_para_batch_size_(layer_para_batch_size),
                          model_path_prefix_(model_path_prefix),
                          stream(stream)
  {
    setup_parallel_param_ranks();
    //    setup_parallel_param_nccls(nccl_ids);
    load_gpt_model_param();
  }

  ~GptParamInstance()
  {
    free_model_param();
  }

  void inline free_param(const DataType** p)
  {
    cudaFree(const_cast<DataType*>(*p));
    *p = nullptr;
  }

  void inline free_param(const int** p)
  {
    cudaFree(const_cast<int*>(*p));
    *p = nullptr;
  }

  void inline free_param(int** p)
  {
    cudaFree(*p);
    *p = nullptr;
  }

  inline std::string path_to_weights(const char *file, int layernum = -1, int gpu_num = 1)
  {
    if (layernum == -1)
      return model_path_prefix_ + std::to_string(gpu_num) + "-gpu/model." + file;
    else
    {
      return model_path_prefix_ + std::to_string(gpu_num) + "-gpu/model.layers." + std::to_string(layernum) + "." + file;
    }
  }

  inline std::string add_rank_to_path(std::string str, int rank)
  {
    return str + std::to_string(rank) + ".bin";
  }

  virtual AbstractParam* get_param_ptr(std::string param_name)
  {
    if(param_name.find("decoding_params") != std::string::npos)
      return &decoding_params;
    else if(param_name.find("decoder_params") != std::string::npos)
      return dynamic_cast<AbstractParam*>(decoder_params.get());
    else if(param_name.find("tensor_parallel_params") != std::string::npos)
      return &tensor_parallel_params;
    else if(param_name.find("layer_parallel_params") != std::string::npos)
      return &layer_parallel_params;
    else
    {
      printf("[ERROR] no parameters %s. \n", param_name.c_str());
      exit(-1);
    }
  }

  virtual void free_model_param();

  void setup_parallel_param(std::vector<ncclUniqueId> nccl_ids);
  void load_gpt_model_param();

  void setup_parallel_param_ranks();
  void setup_parallel_param_nccls(std::vector<ncclUniqueId> nccl_ids);
  virtual void init_nccl_from_ids(std::vector<ncclUniqueId> nccl_ids) {setup_parallel_param_nccls(nccl_ids);}
  virtual void init_nccl_from_comms(ncclComm_t tensor_para_nccl_comm, ncclComm_t layer_para_nccl_comm)
  {
    tensor_parallel_params.nccl_comm = tensor_para_nccl_comm;
    layer_parallel_params.nccl_comm = layer_para_nccl_comm;
  }

  int init_device_from_bin(DataType **ptr, std::vector<uint64_t> shape, std::string filename, int split = 1);
  int init_device_from_csv(DataType **ptr, std::vector<uint64_t> shape, std::string filename, int split = 1);
  int init_device_from_file(DataType **ptr, std::vector<uint64_t> shape, std::string filename, int split = 1, std::string type="bin");
};

template <fastertransformer::OperationType OpType>
struct GptModelInstance : public AbstractTransformerModelInstance
{
  typedef DecoderTransformerTraits<OpType> Traits;
  typedef typename Traits::DataType DataType;
  GptModelInstance
  (const cudaStream_t stream,
   std::unique_ptr<fastertransformer::Allocator<AllocatorType::CUDA>> allocator,
   std::unique_ptr<DecodingGpt<OpType>> decoding,
   const int batch_size,
   const int max_seq_len)
      : stream(stream),
        allocator(std::move(allocator)),
        decoding(std::move(decoding)),
        batch_size(batch_size),
        max_seq_len(max_seq_len) {}

  const cudaStream_t stream;
  std::unique_ptr<fastertransformer::Allocator<AllocatorType::CUDA>> allocator;
  std::unique_ptr<DecoderInitParam<DataType>[]> decoder_param;
  DecodingInitParam<DataType> decoding_params;
  DecodingInitParam<DataType> decoding_params_2;
  const std::unique_ptr<DecodingGpt<OpType>> decoding;
  const int batch_size;
  const int max_seq_len;
  const int start_id = 50256; // In fact, there is no start id in GPT model, I use ' ' token here.
  const int end_id = 50256;

  virtual std::shared_ptr<std::vector<Tensor>> forward(std::shared_ptr<std::vector<Tensor>> input_tensors) {

    decoding_params.request_batch_size = input_tensors->at(0).shape[0];
    decoding_params.max_input_len = input_tensors->at(0).shape[1];
    for(int i = 0; i < decoding->get_num_layer(); i++)
    {
      decoder_param.get()[i].request_batch_size = input_tensors->at(0).shape[0];
    }

    decoding_params.request_input_len = ((int*)input_tensors->at(1).data)[0];
    for(int i = 1; i < decoding_params.request_batch_size; i++)
    {
      decoding_params.request_input_len = decoding_params.request_input_len < ((int*)input_tensors->at(1).data)[i] ?
                                          decoding_params.request_input_len : ((int*)input_tensors->at(1).data)[i];
    }
    decoding_params.request_output_len = ((int*)input_tensors->at(2).data)[0];
    for(int i = 1; i < decoding_params.request_batch_size; i++)
    {
      decoding_params.request_output_len = decoding_params.request_output_len > ((int*)input_tensors->at(2).data)[i] ?
                                           decoding_params.request_output_len : ((int*)input_tensors->at(2).data)[i];
    }

    auto d_inputs = prepareRequestAttentionMask(input_tensors, decoding_params.request_input_len);
    decoding_params.d_start_ids = d_inputs.first;
    decoding_params.d_attn_mask = (DataType *) d_inputs.second;
    cudaDeviceSynchronize();
    
    // TODO: Here, we set the local batch size to request batch size
    // because triton backend only supports single nodes and we don't
    // need local_batch_size currently.
    decoding->set_local_batch_size(decoding_params.request_batch_size);

    // printf("[INFO] start to forward \n");
    // struct timeval start, end;
    // gettimeofday(&start, NULL);
    decoding->forward_context(decoder_param.get(), decoding_params);
    decoding->forward(decoder_param.get(), decoding_params);
    // gettimeofday(&end, NULL);
    // printf("[INFO] inference time: %.2f ms \n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001);

    cudaFree(d_inputs.first);
    cudaFree(d_inputs.second);
    
    return std::shared_ptr<std::vector<Tensor>> (new std::vector<Tensor>{
        Tensor {MEMORY_GPU, TYPE_UINT32,
            std::vector<int64_t>{batch_size, max_seq_len},
              (void *) decoding_params.output_ids}});
  }

  virtual void set_param(AbstractParamInstance* param_instance)
  {
    decoding_params = (*dynamic_cast<DecodingInitParam<DataType>*>(param_instance->get_param_ptr("decoding_params")));
    decoder_param = std::unique_ptr<DecoderInitParam<DataType>[]>(dynamic_cast<DecoderInitParam<DataType>*>(param_instance->get_param_ptr("decoder_params")));
    decoding->set_tensor_parallel_param(*dynamic_cast<TensorParallelParam*>(param_instance->get_param_ptr("tensor_parallel_params")));
    decoding->set_layer_parallel_param(*dynamic_cast<LayerParallelParam*>(param_instance->get_param_ptr("layer_parallel_params")));
  }

  ~GptModelInstance()
  {
  }

private:
  std::pair<int*, void*> prepareRequestAttentionMask(std::shared_ptr<std::vector<Tensor>> input_tensors, const int input_len);
};

std::shared_ptr<std::vector<Tensor>> prepareRequest(std::string request_config_filename, std::string start_id_filename = std::string("../sample/cpp/start_ids.csv"));
void check_inputs(std::shared_ptr<std::vector<Tensor>> input, const char* filename="in");
void check_outputs(std::shared_ptr<std::vector<Tensor>> output, const char* finename="out");
