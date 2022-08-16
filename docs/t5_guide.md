# FasterTransformer T5

The FasterTransformer T5 implements the huggingface t5 model (https://huggingface.co/t5-base).

## Table Of Contents

- [FasterTransformer T5](#fastertransformer-t5)
  - [Table Of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Supported features](#supported-features)
  - [Model architecture](#model-architecture)
    - [Workflow](#workflow)
    - [Optimization](#optimization)
  - [Setup](#setup)
    - [Requirements](#requirements)
    - [Build the FasterTransformer](#build-the-fastertransformer)
      - [Prepare](#prepare)
      - [Build the project](#build-the-project)
  - [How to use](#how-to-use)
    - [Translation process](#translation-process)
    - [Running UL2 on FasterTransformer Pytorch op](#running-ul2-on-fastertransformer-pytorch-op)
    - [Running t5-v1.1](#running-t5-v11)
    - [Running mt5](#running-mt5)
  - [Performance](#performance)
    - [End-to-end translation performance on PyTorch](#end-to-end-translation-performance-on-pytorch)
      - [T5-3B on A100-80GB](#t5-3b-on-a100-80gb)
      - [T5-base on A100-40GB](#t5-base-on-a100-40gb)
      - [T5-base on V100-16GB](#t5-base-on-v100-16gb)
      - [T5-small on V100-16GB](#t5-small-on-v100-16gb)

## Introduction

This document describes what FasterTransformer provides for the `T5` model, explaining the workflow and optimization. We also provide a guide to help users to run the `T5` model on FasterTransformer. Finally, we provide benchmark to demonstrate the speed of FasterTransformer on `T5`. 

### Supported features

* Checkpoint converter
  * Huggingface
  * Megatron
  * NeMo Megatron
* Data type
  * FP32
  * FP16
  * BF16
* Feature
  * Multi-GPU multi-node inference
  * Dynamic random seed
  * Stop tokens
  * Beam search and sampling are both supported
  * Loading FP32 or FP16 weights
* Frameworks
  * PyTorch
  * Triton backend

## Model architecture

### Workflow

The source codes are put in `src/fastertransformer/models/t5`.

* Constructor of T5 Encoder

| Classification |             Name             |     Data Type      |                                                                                                            Description                                                                                                            |
| :------------: | :--------------------------: | :----------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      [0]       |        max_batch_size        |       size_t       |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [1]       |         max_seq_len          |       size_t       |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [2]       |           head_num           |       size_t       |                                                                                                Head number for model configuration                                                                                                |
|      [3]       |        size_per_head         |       size_t       |                                                                                               Size per head for model configuration                                                                                               |
|      [4]       |          inter_size          |       size_t       |                                                                     The inter size of feed forward network. It is often set to 4 * head_num * size_per_head.                                                                      |
|      [5]       |           d_model            |       size_t       |                                                                                         The dimension of embedding of transformer input.                                                                                          |
|      [6]       |          num_layer           |       size_t       |                                                                                       Number of transformer layers for model configuration                                                                                        |
|      [7]       |  num_bucket_or_max_seq_len   |       size_t       |                                                              Number of bucket in relative position embedding, or max sequence length for absolute position embedding                                                              |
|      [8]       |         max_distance         |       size_t       |                                                                                           Max distance for relative position embedding                                                                                            |
|      [9]       |              sm              |        int         |                                                                                                    The compute capacity of GPU                                                                                                    |
|      [10]      |          q_scaling           |       float        |                                                                          It is used to scale the query before the batch multiplication of query and key                                                                           |
|      [11]      |            stream            |    cudaStream_t    |                                                                                                            CUDA stream                                                                                                            |
|      [12]      |        cublas_wrapper        |  cublasMMWrapper*  |                                                                  Pointer of cuBLAS wrapper, which is declared in `src/fastertransformer/utils/cublasMMWrapper.h`                                                                  |
|      [13]      |          allocator           |    IAllocator*     |                                                                    Pointer of memory allocator, which is declared in `src/fastertransformer/utils/allocator.h`                                                                    |
|      [14]      | is_free_buffer_after_forward |        bool        | If setting to be `true`, FasterTransformer will allocate buffer before forward, and free buffer after forward. When the allocator is based on memory pool, setting to `true` may help reducing the memory usage during inference. |
|      [15]      |        attention_type        |   AttentionType    |                                      Determine fusing the attention or not, remove padding or not, which is declared in `src/fastertransformer/layers/attention_layers/BaseAttentionLayer.h`                                      |
|      [16]      |            sparse            |        bool        |                                                                                            Is using sparsity. **Experimental feature**                                                                                            |
|      [17]      |       activation_type        |   ActivationType   |                                                         Determine the activation in FFN, which is declared in `src/fastertransformer/layers/attention_layers/FfnLayer.h`                                                          |
|      [18]      |        layernorm_type        |   LayerNormType    |                                                     Determine using pre-layernorm or post-layernorm, which is declared in `src/fastertransformer/kernels/layernorm_kernels.h`                                                     |
|      [19]      |         tensor_para          |     NcclParam      |                                                                   Tensor Parallel information, which is declared in `src/fastertransformer/utils/nccl_utils.h`                                                                    |
|      [20]      |        pipeline_para         |     NcclParam      |                                                                  Pipeline Parallel information, which is declared in `src/fastertransformer/utils/nccl_utils.h`                                                                   |
|      [21]      |    custom_all_reduce_comm    | AbstractCustomComm |                                                Custom all reduction communication for custom all reduction in model parallelism. It is only supported in 8-way tensor parallelism                                                 |
|      [22]      |   enable_custom_all_reduce   |        int         |                                                                                           Flag of enabling custom all reduction or not                                                                                            |

* Input of T5 Encoder

|      Name       |     Tensor/Parameter Shape     | Location |   Data Type    |                                                         Description                                                         |
| :-------------: | :----------------------------: | :------: | :------------: | :-------------------------------------------------------------------------------------------------------------------------: |
|    input_ids    |     [batch_size, seq_len]      |   GPU    |      int       |                                                        The input ids                                                        |
| sequence_length |          [batch_size]          |   GPU    |      int       |                                                  The lengths of input ids                                                   |
|  inputs_embeds  | [batch_size, seq_len, d_model] |   GPU    | fp32/fp16/bf16 | **Optional**. The embedding after embedding lookup. If this input is not null, using this embedding as input of transformer |

* Output of T5 Encoder

|        Name         |         Tensor/Parameter Shape          | Location |   Data Type    |           Description           |
| :-----------------: | :-------------------------------------: | :------: | :------------: | :-----------------------------: |
| output_hidden_state | [batch_size, sequence_length, d_model_] |   GPU    | fp32/fp16/bf16 | The output of transformer layer |

* Constructor of T5 Decoding

| Classification |             Name             |     Data Type      |                                                                                                            Description                                                                                                            |
| :------------: | :--------------------------: | :----------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      [0]       |        max_batch_size        |       size_t       |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [1]       |         max_seq_len          |       size_t       |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [2]       |       mem_max_seq_len        |       size_t       |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [3]       |          beam_width          |       size_t       |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [4]       |           head_num           |       size_t       |                                                                                                Head number for model configuration                                                                                                |
|      [5]       |        size_per_head         |       size_t       |                                                                                               Size per head for model configuration                                                                                               |
|      [6]       |          inter_size          |       size_t       |                                                                     The inter size of feed forward network. It is often set to 4 * head_num * size_per_head.                                                                      |
|      [7]       |           d_model            |       size_t       |                                                                                         The dimension of embedding of transformer input.                                                                                          |
|      [8]       |          num_layer           |       size_t       |                                                                                       Number of transformer layers for model configuration                                                                                        |
|      [9]       |          vocab_size          |       size_t       |                                                                                              Vocabulary size for model configuration                                                                                              |
|      [10]      |          num_bucket          |       size_t       |                                                              Number of bucket in relative position embedding, or max sequence length for absolute position embedding                                                              |
|      [11]      |         max_distance         |       size_t       |                                                                                           Max distance for relative position embedding                                                                                            |
|      [12]      |          q_scaling           |       float        |                                                                          It is used to scale the query before the batch multiplication of query and key                                                                           |
|      [13]      |           start_id           |        int         |                                                                                                      Start id for vocabulary                                                                                                      |
|      [14]      |            end_id            |        int         |                                                                                                       End id for vocabulary                                                                                                       |
|      [15]      |  beam_search_diversity_rate  |       float        |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [16]      |            top_k             |       size_t       |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [17]      |            top_p             |       float        |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [18]      |         temperature          |       float        |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [19]      |         len_penalty          |       float        |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [20]      |      repetition_penalty      |       float        |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [21]      |            stream            |    cudaStream_t    |                                                                                                            CUDA stream                                                                                                            |
|      [22]      |        cublas_wrapper        |  cublasMMWrapper*  |                                                                  Pointer of cuBLAS wrapper, which is declared in `src/fastertransformer/utils/cublasMMWrapper.h`                                                                  |
|      [23]      |          allocator           |    IAllocator*     |                                                                    Pointer of memory allocator, which is declared in `src/fastertransformer/utils/allocator.h`                                                                    |
|      [24]      | is_free_buffer_after_forward |        bool        | If setting to be `true`, FasterTransformer will allocate buffer before forward, and free buffer after forward. When the allocator is based on memory pool, setting to `true` may help reducing the memory usage during inference. |
|      [25]      |       cuda_device_prop       |  cudaDeviceProp*   |                                                           Pointer of CUDA device properties, which is used to get the properties of hardware like size of shared memory                                                           |
|      [26]      |         tensor_para          |     NcclParam      |                                                                   Tensor Parallel information, which is declared in `src/fastertransformer/utils/nccl_utils.h`                                                                    |
|      [27]      |        pipeline_para         |     NcclParam      |                                                                  Pipeline Parallel information, which is declared in `src/fastertransformer/utils/nccl_utils.h`                                                                   |
|      [28]      |       activation_type        |   ActivationType   |                                                         Determine the activation in FFN, which is declared in `src/fastertransformer/layers/attention_layers/FfnLayer.h`                                                          |
|      [29]      |     tie_word_embeddings      |        bool        |                                                                                        A flag controlling the scale of transformer output                                                                                         |
|      [30]      |    custom_all_reduce_comm    | AbstractCustomComm |                                                Custom all reduction communication for custom all reduction in model parallelism. It is only supported in 8-way tensor parallelism                                                 |
|      [31]      |   enable_custom_all_reduce   |        int         |                                                                                           Flag of enabling custom all reduction or not                                                                                            |

* Input of T5 Decoding

|            Name            |            Tensor/Parameter Shape             | Location |       Data Type        |                                                              Description                                                               |
| :------------------------: | :-------------------------------------------: | :------: | :--------------------: | :------------------------------------------------------------------------------------------------------------------------------------: |
|       encoder_output       | [batch_size, mem_max_seq_len, memory_d_model] |   GPU    |     fp32/fp16/bf16     |                                                        The output of T5 Encoder                                                        |
|  encoder_sequence_length   |                 [batch_size]                  |   GPU    |          int           |                                              The sequence length of encoder input/output                                               |
|      stop_words_list       |      [batch_size, 2, stop_words_length]       |   GPU    |          int           |                **Optional**. When FT generates words in this list, it will stop the generation. An extension of stop id                |
|       bad_words_list       |       [batch_size, 2, bad_words_length]       |   GPU    |          int           | **Optional**. The words in the list will be When FT generates words in this list, it will stop the generation. An extension of stop id |
|          start_id          |                 [batch_size]                  |   CPU    |          int           |                            **Optional**. If FT receives this input, FT will replace default start id by it                             |
|           end_id           |                 [batch_size]                  |   CPU    |          int           |                             **Optional**. If FT receives this input, FT will replace default end id by it                              |
|       runtime_top_k        |              [1] or [batch_size]              |   CPU    |          uint          |                                              **Optional**. top_k value for top k sampling                                              |
|       runtime_top_p        |              [1] or [batch_size]              |   CPU    |         float          |                                              **Optional**. top_p value for top p sampling                                              |
| beam_search_diversity_rate |              [1] or [batch_size]              |   CPU    |         float          |               **Optional**. A hyper hyper-parameter for [simple diverse decoding](https://arxiv.org/pdf/1611.08562.pdf)                |
|        temperature         |              [1] or [batch_size]              |   CPU    |         float          |                             **Optional**. Temperature applied to logits for both beam search and sampling                              |
|        len_penalty         |              [1] or [batch_size]              |   CPU    |         float          |                                  **Optional**. Length penalty applied to logits for only beam search                                   |
|     repetition_penalty     |              [1] or [batch_size]              |   CPU    |         float          |                          **Optional**. Repetition penalty applied to logits for both beam search and sampling                          |
|        random_seed         |              [1] or [batch_size]              |   CPU    | unsigned long long int |                                 **Optional**. Random seed to initialize the random table in sampling.                                  |

* Output of T5 Decoding

|       Name       |                                               Tensor/Parameter Shape                                                | Location | Data Type |                                    Description                                    |
| :--------------: | :-----------------------------------------------------------------------------------------------------------------: | :------: | :-------: | :-------------------------------------------------------------------------------: |
|    output_ids    |                                    [batch_size, beam_width, max_output_seq_len]                                     |   GPU    |    int    |            The output ids. It contains the input_ids and generated ids            |
| sequence_length  |                                              [batch_size, beam_width]                                               |   GPU    |    int    |                             The lengths of output ids                             |
| output_log_probs |                                  [batch_size, beam_width, request_output_seq_len]                                   |   GPU    |   float   | **Optional**. It records the log probability of logits at each step for sampling. |
|  cum_log_probs   |                                              [batch_size, beam_width]                                               |   GPU    |   float   |          **Optional**. Cumulative log probability of generated sentences          |
| cross_attentions | [num_layer / pipeline_para_size, batch_size, beam_width, head_num / tensor_para_size, max_seq_len, mem_max_seq_len] |   GPU    |   float   |               **Optional**. The attention scores of cross attention               |

### Optimization

1.	Kernel optimization: First, since the sequence length of query in `SelfAttention` and `CrossAttention` is always 1, we use customed fused multi-head attention kernel to optimize. Second, we fuse many small operations into one kernel. For example, `AddBiasResidualLayerNorm` combines the adding bias, adding residual of previous block and the computation of layer normalization into 1 kernel. Third, we optimize top k operation and sampling to accelerate the beam search and sampling. Finally, to prevent from recomputing the previous keys and values, we allocate a buffer to store them at each step. Although it takes some additional memory usage, we can save the cost of recomputing, allocating buffer at each step, and the cost of concatenation.

## Setup

The following section lists the requirements to use FasterTransformer.

### Requirements

- CMake >= 3.13 for PyTorch
- CUDA 11.0 or newer version
- NCCL 2.10 or newer version
- Python: Only verify on Python 3.
- PyTorch: Verify on 1.10.0, >= 1.5.0 should work.

Recommend use nvcr image like `nvcr.io/nvidia/pytorch:22.07-py3`.

Ensure you have the following components:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) and NGC container are recommended
- [NVIDIA Pascal](https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/) or [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) or [Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/) based GPU 

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:

- [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
- [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
- [Running PyTorch](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)

For those unable to use the NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

### Build the FasterTransformer

#### Prepare

You can choose the pytorch version and python version you want. Here, we suggest image `nvcr.io/nvidia/pytorch:22.07-py3`, which contains the PyTorch 1.8.0 and python 3.8.

    ```bash
    nvidia-docker run -ti --rm nvcr.io/nvidia/pytorch:22.07-py3 bash
    git clone https://github.com/NVIDIA/FasterTransformer.git
    mkdir -p FasterTransformer/build
    cd FasterTransformer/build
    git submodule init && git submodule update
    ```

#### Build the project

* Note: the `xx` of `-DSM=xx` in following scripts means the compute capability of your GPU. The following table shows the compute capability of common GPUs.

|  GPU  | compute capacity |
| :---: | :--------------: |
|  P40  |        60        |
|  P4   |        61        |
| V100  |        70        |
|  T4   |        75        |
| A100  |        80        |
|  A30  |        80        |
|  A10  |        86        |

By default, `-DSM` is set by 70, 75, 80 and 86. When users set more kinds of `-DSM`, it requires longer time to compile. So, we suggest setting the `-DSM` for the device you use only. Here, we use `xx` as an example due to convenience.

1. build with PyTorch

    ```bash
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
    make
    ```

    This will build the TorchScript custom class. Please make sure that the `PyTorch >= 1.5.0`.

2. build with TensorRT
  
    Can use `nvcr.io/nvidia/pytorch:22.07-py3` docker image, too.

    ```bash
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_TRT=ON -DBUILD_MULTI_GPU=ON ..
    make
    ```

## How to use

### Translation process

1. Run FasterTransformer T5 on PyTorch

    Please install utils first before running the demos by

    ```bash
    pip install -r ../examples/pytorch/t5/requirement.txt
    ```

    1.1 Generate the `gemm_config.in` file:

    `./bin/t5_gemm` can generate the best GEMM configuration.

    Assume the settings of decoding are as follows.

    - `batch_size` = 8
    - `beam_width` = 4
    - `max_mem_seq_len` = 32
    - `encoder_d_model` = 512
    - `encoder_head_num` = 8
    - `encoder_size_per_head` = 64
    - `encoder_inter_size` = 2048
    - `decoder_d_model` = 512
    - `decoder_head_num` = 8
    - `decoder_size_per_head` = 64
    - `decoder_inter_size` = 2048
    - `decoder_vocab_size` = 32128
    - `data_type` = 0 (FP32) or 1 (FP16) or 2 (BF16)
    - `tensor_para_size` = 2

    Then the following scripts can generate the best GEMM configuration under such settings and record the configuration into the `gemm_config.in` file.

    ```bash
    ./bin/t5_gemm 8 4 32 512 8 64 2048 512 8 64 2048 32128 0 2 1
    ```

    1.2 Run the PyTorch T5 example: 

    ```bash
    python ../examples/pytorch/t5/translate_example.py \
            --batch_size 32 \
            --beam_width 4 \
            --max_seq_len 128 \
            --data_type fp32 \
            --test_time 0123 \
            --sampling_topk 4 \
            --model t5-small
    ```

    Data Type can be `fp32`, `fp16` and `bf16`

    The outputs should be like to the following:

    ```bash
    [INFO] hf-beamsearch translates 94 batches taking 157.58 sec to translate 62591 tokens, BLEU score: 26.21, 397 tokens/sec.
    [INFO] ft-beamsearch translates 94 batches taking 14.45 sec to translate 61763 tokens, BLEU score: 26.45, 4274 tokens/sec.
    [INFO] hf-sampling translates 94 batches taking 99.17 sec to translate 62022 tokens, BLEU score: 25.35, 625 tokens/sec.
    [INFO] ft-sampling translates 94 batches taking 7.93 sec to translate 62096 tokens, BLEU score: 17.61, 7827 tokens/sec.
    ```

    1.3 Run T5 with model parallel

    ```bash
    mpirun -n 4 --allow-run-as-root \
      python ../examples/pytorch/t5/translate_example.py \
            --batch_size 32 \
            --beam_width 4 \
            --max_seq_len 128 \
            --data_type fp32 \
            --test_time 0123 \
            --sampling_topk 4 \
            --model t5-small \
            --tensor_para_size 2 \
            --pipeline_para_size 2
    ```

2. Run FasterTransformer T5 on TensorRT

    Please install transformers first before running the demos by

    ```bash
    pip install -r ../examples/pytorch/t5/requirement.txt
    ```

    ```bash
    # get T5Model weight for test (need Internet or pre-downloaded model)
    # Note that the model is saved in ./ft_t5_small/1-gpu, but not ./ft_t5_small
    python ../examples/tensorrt/t5/extractT5ModelToBIN.py \
            -in_file t5-small \
            -saved_dir ./ft_t5_small

    python ../examples/tensorrt/t5/testT5Plugin.py \
            --batch_size 32 \
            --beam_width 4 \
            --max_seq_len 128 \
            --data_type fp16 \
            --ckpt_path ./ft_t5_small/1-gpu
    ```
* Input/Output Tensor/Parameter of T5Encoder Plugin

| Classification  |      Tensor/Parameter Shape      |   Data Type    |              Description               |
| :-------------: | :------------------------------: | :------------: | :------------------------------------: |
|  input tensor   |                                  |                |                                        |
|       [0]       |     [batch_size,max_seq_len]     |     int32      |     input token after tokenization     |
|       [1]       |           [batch_size]           |     int32      |   real sequence length of each input   |
| input parameter |                                  |                |                                        |
|       [0]       |                []                |     int32      |             max_batch_size             |
|       [1]       |                []                |     int32      |              max_seq_len               |
|       [2]       |                []                |     int32      | beam_width (keep the same as decoding) |
|       [3]       |                []                |     int32      |                   sm                   |
|       [4]       |                []                |     int32      |                useFP16                 |
|       [5]       |                []                |     string     | checkpoint path of converted FT model  |
|  output tensor  |                                  |                |                                        |
|       [0]       | [batch_size,max_seq_len,d_model] | foat32/float16 |             encoder output             |

* Input/Output Tensor/Parameter of T5Decoding Plugin

| Classification  |       Tensor/Parameter Shape        |    Data Type    |              Description              |
| :-------------: | :---------------------------------: | :-------------: | :-----------------------------------: |
|  input tensor   |                                     |                 |                                       |
|       [0]       |  [batch_size,max_seq_len,d_model]   | foat32/float16  |            encoder output             |
|       [1]       |            [batch_size]             |      int32      |  real sequence length of each input   |
|       [2]       |         [1] or [batch_size]         |      int32      |                 top_k                 |
|       [3]       |         [1] or [batch_size]         |     float32     |                 top_p                 |
|       [4]       |         [1] or [batch_size]         |     float32     |      beam_search_diversity_rate       |
|       [5]       |         [1] or [batch_size]         |     float32     |              temperature              |
|       [6]       |         [1] or [batch_size]         |     float32     |              len_penalty              |
|       [7]       |         [1] or [batch_size]         |     float32     |          repetition_penalty           |
| input parameter |                                     |                 |                                       |
|       [0]       |                 []                  |      int32      |            max_batch_size             |
|       [1]       |                 []                  |      int32      |              max_seq_len              |
|       [2]       |                 []                  |      int32      |            mem_max_seq_len            |
|       [3]       |                 []                  |      int32      |              beam_width               |
|       [4]       |                 []                  |      int32      |               usaeFP16                |
|       [5]       |                 []                  |     string      | checkpoint path of converted FT model |
|  output tensor  |                                     |                 |                                       |
|       [0]       | [batch_size,beam_width,max_seq_len] | float32/float16 |            decoding output            |
|       [1]       |       [batch_size,beam_width]       | float32/float16 |  real sequence length of each output  |

The model configuration are stored in `config.ini` of checkpoint path. For example, after running, 

```
python ../examples/tensorrt/t5/extractT5ModelToBIN.py \
            -in_file t5-small \
            -saved_dir ./ft_t5_small`
```

users can see the model configuration in `./ft_t5_small/1-gpu/config.ini`

### Running UL2 on FasterTransformer Pytorch op

[UL2](https://arxiv.org/pdf/2205.05131v1.pdf) (Unifying Language Learning Paradigms) is published by Google. The following is its introduction:

> UL2 is a unified framework for pretraining models that are universally effective across datasets and setups. UL2 uses Mixture-of-Denoisers (MoD), apre-training objective that combines diverse pre-training paradigms together. UL2 introduces a notion of mode switching, wherein downstream fine-tuning is associated with specific pre-training schemes.

We show how to sever UL2 by FasterTransformer PyTorch op on huggingface's model in this section.

    3.1 Download model (It requires some time because the model size is about 40GBs)

    ```
    sudo apt-get install git-lfs
    git lfs install
    git lfs clone https://huggingface.co/google/ul2
    ```

    3.2 Convert the checkpoint to FT

    Because loading UL2 model on pytorch and do prprocessing takes long time, and `summarization.py` only supports loading FT's model from binary files, we convert the pytorch checkpoint to FasterTransformer by converter `huggingface_t5_ckpt_convert.py`.

    ```
    python3 ../examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
            -saved_dir ul2/c-models \
            -in_file ul2/ \
            -inference_tensor_para_size 2 \
            -weight_data_type fp32
    ```

    3.3 Run UL2 on summarization task

    ```
    mpirun -n 2 python3 ../examples/pytorch/t5/summarization.py  \
                          --ft_model_location ul2/c-models/ \
                          --hf_model_location ul2/ \
                          --test_ft \
                          --data_type bf16 \
                          --tensor_para_size 2
    ```

    The results would be like

    ```
    rouge1 : 23.673944166014593
    rouge2 : 5.946485383012474
    rougeL : 14.749827731626247
    rougeLsum : 20.217932008044144
    ```

### Running t5-v1.1

    3.1 Download model (It requires some time because the model size is about 40GBs)

    ```
    sudo apt-get install git-lfs
    git lfs install
    git lfs clone https://huggingface.co/google/t5-v1_1-base
    ```


    3.2 Convert the checkpoint to FT

    ```
    python3 ../examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
            -saved_dir t5-v1_1-base/c-models \
            -in_file t5-v1_1-base/ \
            -inference_tensor_para_size 1 \
            -weight_data_type fp32
    ```

    3.3 Run t5-v1.1 on summarization task

    ```
    python3 ../examples/pytorch/t5/summarization.py  \
            --ft_model_location t5-v1_1-base/c-models/ \
            --hf_model_location t5-v1_1-base/ \
            --test_ft \
            --test_hf
    ```

    The results would be like

    ```
    Hugging Face (total latency: 21.826529 sec)
    rouge1 : 10.786476875527406
    rouge2 : 1.8231246974441166
    rougeL : 8.652689713627165
    rougeLsum : 10.326607305635523
    Faster Transformers (total latency: 7.036808000000001 sec)
    rouge1 : 10.91735083630513
    rouge2 : 1.8454654301092783
    rougeL : 8.76872604148143
    rougeLsum : 10.453229536094794
    ```

    * Note that these models are not fine-tuned, so running with FP16 or setting topk > 1 may lead to unstable results.

### Running mt5

    3.1 Download model (It requires some time because the model size is about 40GBs)

    ```
    sudo apt-get install git-lfs
    git lfs install
    git lfs clone https://huggingface.co/google/mt5-base
    ```


    3.2 Convert the checkpoint to FT

    ```
    python3 ../examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
            -saved_dir mt5-base/c-models \
            -in_file mt5-base/ \
            -inference_tensor_para_size 1 \
            -weight_data_type fp32
    ```

    3.3 Run mt5 on summarization task

    ```
    python3 ../examples/pytorch/t5/summarization.py  \
            --ft_model_location mt5-base/c-models/ \
            --hf_model_location mt5-base/ \
            --test_ft \
            --test_hf
    ```

    The results would be like

    ```
    Hugging Face (total latency: 3.143815 sec)
    rouge1 : 4.636193727758547
    rouge2 : 0.20661157024793395
    rougeL : 3.7990194456844026
    rougeLsum : 4.274724726798723
    Faster Transformers (total latency: 1.3952859999999998 sec)
    rouge1 : 4.726148174547172
    rouge2 : 0.20818875780707846
    rougeL : 3.8698557495145516
    rougeLsum : 4.3507453221528
    ```

    * Note that these models are not fine-tuned, so running with FP16 or setting topk > 1 may lead to unstable results.

## Performance

Hardware settings: 
* CPU: Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz
* V100-16GB (with mclk 877MHz, pclk 1380MHz) with Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz (dgx-1 server)
* A100-40GB
* A100-80GB (with mclk 1593, pclk 1410) with AMD EPYC 7742 64-Core Processor

To run the following benchmark, we need to install the unix computing tool "bc" by

```bash
apt-get install bc
```

### End-to-end translation performance on PyTorch

We demonstrate the throughput of huggingface and FT for end-to-end translation on V100 and A100. We also skip the BLEU score because the score of PyTorch, FT Decoder and FT Decoding are close.

Although the bleu scores of all methods are close, the results may be little different, and the number of generated tokens may be also different. So, we use throughput but not latency to show the performance in this benchmark.

#### T5-3B on A100-80GB

* T5-3B on FP16 with beamsearch

| Batch Size | beamsearch | Precision | FT Decoding <br/> Throughput (token/sec) |
| :--------: | :--------: | :-------: | :--------------------------------------: |
|     1      |     4      |   FP16    |                   192                    |
|     1      |     32     |   FP16    |                   140                    |
|     8      |     4      |   FP16    |                   787                    |
|     8      |     32     |   FP16    |                   271                    |
|     32     |     4      |   FP16    |                   1540                   |
|     32     |     32     |   FP16    |                   OOM                    |
|    128     |     4      |   FP16    |                   1907                   |
|    128     |     32     |   FP16    |                   OOM                    |

When batch size is 32, beam width is 32, the k/v caches require about 90GBs and lead to OOM.

* T5-3B on FP16 with sampling

| Batch Size | sampling | Precision | FT Decoding <br/> Throughput (token/sec) |
| :--------: | :------: | :-------: | :--------------------------------------: |
|     1      |    4     |   FP16    |                   218                    |
|     1      |   0.5    |   FP16    |                   217                    |
|     8      |    4     |   FP16    |                   932                    |
|     8      |   0.5    |   FP16    |                   908                    |
|     32     |    4     |   FP16    |                   2416                   |
|     32     |   0.5    |   FP16    |                   2344                   |
|    128     |    4     |   FP16    |                   5004                   |
|    128     |   0.5    |   FP16    |                   4891                   |

#### T5-base on A100-40GB

* T5-base on FP32 with beamsearch

| Batch Size | beamsearch | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
| :--------: | :--------: | :-------: | :--------------------------------------: | :--------------------------------------: | :-----------------------: |
|     1      |     4      |   FP32    |                    46                    |                   422                    |           9.17            |
|     1      |     32     |   FP32    |                    34                    |                   339                    |           9.97            |
|     8      |     4      |   FP32    |                   194                    |                   1779                   |           9.17            |
|     8      |     32     |   FP32    |                    98                    |                   516                    |           5.26            |
|     32     |     4      |   FP32    |                   486                    |                   2939                   |           6.04            |
|     32     |     32     |   FP32    |                   OOM                    |                   OOM                    |             -             |
|    128     |     4      |   FP32    |                   810                    |                   3445                   |           4.25            |
|    128     |     32     |   FP32    |                   OOM                    |                   OOM                    |             -             |

* T5-base on FP16 with beamsearch

| Batch Size | beamsearch | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
| :--------: | :--------: | :-------: | :--------------------------------------: | :--------------------------------------: | :-----------------------: |
|     1      |     4      |   FP16    |                    44                    |                   671                    |           15.25           |
|     1      |     32     |   FP16    |                    25                    |                   517                    |           20.68           |
|     8      |     4      |   FP16    |                   139                    |                   2807                   |           20.19           |
|     8      |     32     |   FP16    |                    77                    |                   1573                   |           20.42           |
|     32     |     4      |   FP16    |                   368                    |                   7102                   |           19.29           |
|     32     |     32     |   FP16    |                   123                    |                   1830                   |           14.87           |
|    128     |     4      |   FP16    |                   656                    |                  11312                   |           17.24           |
|    128     |     32     |   FP16    |                   OOM                    |                   1845                   |             -             |

* T5-base on FP32 with sampling

| Batch Size | sampling | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
| :--------: | :------: | :-------: | :--------------------------------------: | :--------------------------------------: | :-----------------------: |
|     1      |    4     |   FP32    |                    66                    |                   334                    |           5.06            |
|     1      |   0.5    |   FP32    |                    65                    |                   323                    |           4.97            |
|     8      |    4     |   FP32    |                   217                    |                   1887                   |           8.70            |
|     8      |   0.5    |   FP32    |                   200                    |                   1765                   |           8.83            |
|     32     |    4     |   FP32    |                   718                    |                   5211                   |           7.26            |
|     32     |   0.5    |   FP32    |                   656                    |                   4731                   |           7.21            |
|    128     |    4     |   FP32    |                   2115                   |                   8782                   |           4.15            |
|    128     |   0.5    |   FP32    |                   1805                   |                   8212                   |           4.55            |

* T5-base on FP16 with sampling

| Batch Size | sampling | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
| :--------: | :------: | :-------: | :--------------------------------------: | :--------------------------------------: | :-----------------------: |
|     1      |    4     |   FP16    |                    46                    |                   746                    |           16.21           |
|     1      |   0.5    |   FP16    |                    43                    |                   706                    |           16.41           |
|     8      |    4     |   FP16    |                   212                    |                   3293                   |           15.53           |
|     8      |   0.5    |   FP16    |                   191                    |                   3049                   |           15.96           |
|     32     |    4     |   FP16    |                   501                    |                   8783                   |           17.53           |
|     32     |   0.5    |   FP16    |                   432                    |                   7961                   |           18.42           |
|    128     |    4     |   FP16    |                   1426                   |                  18137                   |           12.71           |
|    128     |   0.5    |   FP16    |                   1414                   |                  16680                   |           11.79           |

#### T5-base on V100-16GB

* T5-base on FP32 with beamsearch

| Batch Size | beamsearch | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
| :--------: | :--------: | :-------: | :--------------------------------------: | :--------------------------------------: | :-----------------------: |
|     1      |     4      |   FP32    |                    28                    |                   257                    |           9.17            |
|     1      |     32     |   FP32    |                    20                    |                   175                    |           8.75            |
|     8      |     4      |   FP32    |                   105                    |                   953                    |           9.07            |
|     8      |     32     |   FP32    |                    50                    |                   196                    |           3.92            |
|     32     |     4      |   FP32    |                   247                    |                   1400                   |           5.66            |
|     32     |     32     |   FP32    |                    0                     |                   OOM                    |             x             |
|    128     |     4      |   FP32    |                    0                     |                   1448                   |             x             |
|    128     |     32     |   FP32    |                   OOM                    |                   OOM                    |             x             |

* T5-base on FP16 with beam search

| Batch Size | beamsearch | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
| :--------: | :--------: | :-------: | :--------------------------------------: | :--------------------------------------: | :-----------------------: |
|     1      |     4      |   FP16    |                    21                    |                   359                    |           17.09           |
|     1      |     32     |   FP16    |                    14                    |                   250                    |           17.85           |
|     8      |     4      |   FP16    |                    76                    |                   1418                   |           18.65           |
|     8      |     32     |   FP16    |                    40                    |                   526                    |           13.15           |
|     32     |     4      |   FP16    |                   221                    |                   2962                   |           13.40           |
|     32     |     32     |   FP16    |                   OOM                    |                   684                    |             x             |
|    128     |     4      |   FP16    |                   345                    |                   4079                   |           11.82           |
|    128     |     32     |   FP16    |                   OOM                    |                   OOM                    |             x             |

* T5-base on FP32 with sampling

| Batch Size | sampling | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
| :--------: | :------: | :-------: | :--------------------------------------: | :--------------------------------------: | :-----------------------: |
|     1      |    4     |   FP32    |                    26                    |                   226                    |           8.69            |
|     1      |   0.5    |   FP32    |                    27                    |                   219                    |           8.11            |
|     8      |    4     |   FP32    |                   115                    |                   1153                   |           10.02           |
|     8      |   0.5    |   FP32    |                   130                    |                   1075                   |           8.26            |
|     32     |    4     |   FP32    |                   327                    |                   3021                   |           9.23            |
|     32     |   0.5    |   FP32    |                   297                    |                   2773                   |           9.33            |
|    128     |    4     |   FP32    |                   1162                   |                   4184                   |           3.60            |
|    128     |   0.5    |   FP32    |                   797                    |                   3975                   |           4.98            |

* T5-base on FP16 with sampling

| Batch Size | sampling | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
| :--------: | :------: | :-------: | :--------------------------------------: | :--------------------------------------: | :-----------------------: |
|     1      |    4     |   FP16    |                    19                    |                   364                    |           19.15           |
|     1      |   0.5    |   FP16    |                    20                    |                   353                    |           17.65           |
|     8      |    4     |   FP16    |                    83                    |                   1733                   |           20.87           |
|     8      |   0.5    |   FP16    |                    98                    |                   1599                   |           16.31           |
|     32     |    4     |   FP16    |                   337                    |                   4517                   |           13.40           |
|     32     |   0.5    |   FP16    |                   301                    |                   4207                   |           13.97           |
|    128     |    4     |   FP16    |                   956                    |                   8519                   |           8.91            |
|    128     |   0.5    |   FP16    |                   723                    |                   7997                   |           11.06           |

#### T5-small on V100-16GB

* T5-small on FP32 with beamsearch

| Batch Size | beamsearch | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
| :--------: | :--------: | :-------: | :--------------------------------------: | :--------------------------------------: | :-----------------------: |
|     1      |     4      |   FP32    |                    51                    |                   626                    |           12.27           |
|     1      |     32     |   FP32    |                    30                    |                   413                    |           13.76           |
|     8      |     4      |   FP32    |                   192                    |                   2462                   |           12.82           |
|     8      |     32     |   FP32    |                    72                    |                   563                    |           7.81            |
|     32     |     4      |   FP32    |                   383                    |                   4316                   |           11.26           |
|     32     |     32     |   FP32    |                   104                    |                   668                    |           6.42            |
|    128     |     4      |   FP32    |                   554                    |                   4747                   |           8.56            |
|    128     |     32     |   FP32    |                   OOM                    |                   OOM                    |             x             |

* T5-small on FP16 with beamsearch

| Batch Size | beamsearch | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
| :--------: | :--------: | :-------: | :--------------------------------------: | :--------------------------------------: | :-----------------------: |
|     1      |     4      |   FP16    |                    35                    |                   776                    |           22.17           |
|     1      |     32     |   FP16    |                    28                    |                   553                    |           19.75           |
|     8      |     4      |   FP16    |                   163                    |                   3467                   |           21.26           |
|     8      |     32     |   FP16    |                    71                    |                   1140                   |           16.05           |
|     32     |     4      |   FP16    |                   365                    |                   7154                   |           19.60           |
|     32     |     32     |   FP16    |                   108                    |                   1359                   |           12.58           |
|    128     |     4      |   FP16    |                   524                    |                  11285                   |           21.53           |
|    128     |     32     |   FP16    |                    0                     |                   942※                   |           0.00            |

※: Out of memory on single GPU. Run by 2 ways tensor parallel.

* T5-small on FP32 with sampling

| Batch Size | sampling | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
| :--------: | :------: | :-------: | :--------------------------------------: | :--------------------------------------: | :-----------------------: |
|     1      |    4     |   FP32    |                    60                    |                   577                    |           9.61            |
|     1      |   0.5    |   FP32    |                    57                    |                   524                    |           9.19            |
|     8      |    4     |   FP32    |                   243                    |                   2821                   |           11.60           |
|     8      |   0.5    |   FP32    |                   221                    |                   2345                   |           10.61           |
|     32     |    4     |   FP32    |                   765                    |                   7865                   |           10.28           |
|     32     |   0.5    |   FP32    |                   634                    |                   6365                   |           10.03           |
|    128     |    4     |   FP32    |                   2238                   |                  12134                   |           5.42            |
|    128     |   0.5    |   FP32    |                   1611                   |                  10439                   |           6.47            |

* T5-small on FP16 with sampling

| Batch Size | sampling | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
| :--------: | :------: | :-------: | :--------------------------------------: | :--------------------------------------: | :-----------------------: |
|     1      |    4     |   FP16    |                    46                    |                   934                    |           20.30           |
|     1      |   0.5    |   FP16    |                    42                    |                   862                    |           20.52           |
|     8      |    4     |   FP16    |                   194                    |                   3510                   |           18.09           |
|     8      |   0.5    |   FP16    |                   182                    |                   3235                   |           17.77           |
|     32     |    4     |   FP16    |                   592                    |                  10692                   |           18.06           |
|     32     |   0.5    |   FP16    |                   553                    |                   9008                   |           16.28           |
|    128     |    4     |   FP16    |                   1921                   |                  19446                   |           10.12           |
|    128     |   0.5    |   FP16    |                   1307                   |                  16810                   |           12.86           |
