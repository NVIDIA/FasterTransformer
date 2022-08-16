# GPT-J

## Table Of Contents

- [GPT-J](#gpt-j)
  - [Table Of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Supported features](#supported-features)
  - [Setup](#setup)
    - [Requirements](#requirements)
    - [Docker image](#docker-image)
    - [Build project](#build-project)
    - [Download the model](#download-the-model)
    - [Download tables](#download-tables)
    - [Run GPT-J](#run-gpt-j)
    - [Run GPTJ with prompts](#run-gptj-with-prompts)
    - [Compare with reference implementation](#compare-with-reference-implementation)
    - [gpt-j with triton backend](#gpt-j-with-triton-backend)

## Introduction

This document describes the step to run the GPT-J model on FasterTransformer.
GPT-J was developed by EleutherAI and trained on The Pile, a 825GB dataset from curated sources (e.g. Wikipedia, arXiv, GitHub, StackExchange, PubMed, ...).
With 6 billion parameters, GPT-J is one of the largest GPT-like publicly released models as of 2021.

Optimization in GPT-j are similar to optimization in GPT, describing in the [gpt_guide.md](gpt_guide.md#optimization).

* Constructor of GPT-j

| Classification |             Name             |     Data Type      |                                                                                                            Description                                                                                                            |
| :------------: | :--------------------------: | :----------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      [0]       |        max_batch_size        |       size_t       |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [1]       |         max_seq_len          |       size_t       |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [2]       |        max_input_len         |       size_t       |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [3]       |          beam_width          |       size_t       |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [4]       |           head_num           |       size_t       |                                                                                                Head number for model configuration                                                                                                |
|      [5]       |        size_per_head         |       size_t       |                                                                                               Size per head for model configuration                                                                                               |
|      [6]       |          inter_size          |       size_t       |                                                                     The inter size of feed forward network. It is often set to 4 * head_num * size_per_head.                                                                      |
|      [7]       |          num_layer           |       size_t       |                                                                                       Number of transformer layers for model configuration                                                                                        |
|      [8]       |          vocab_size          |       size_t       |                                                                                              Vocabulary size for model configuration                                                                                              |
|      [9]       |     rotary_embeeding_dim     |       size_t       |                                                                          Rotary embedding dimension of rotary position embedding for model configuration                                                                          |
|      [10]      |           start_id           |        int         |                                                                                                      Start id for vocabulary                                                                                                      |
|      [11]      |            end_id            |        int         |                                                                                                       End id for vocabulary                                                                                                       |
|      [12]      |   prompt_learning_start_id   |        int         |                                                                                         The start id of virtual token in p/prompt-tuning                                                                                          |
|      [13]      |     prompt_learning_type     | PromptLearningType |                                   The type of prompt learning when we load the prompt embedding in constructor. FT supports `no_prompt`, `soft_prompt`, `prefix_prompt`, `p_prompt_tuning` now                                    |
|      [14]      |  beam_search_diversity_rate  |       float        |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [15]      |            top_k             |       size_t       |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [16]      |            top_p             |       float        |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [17]      |         random_seed          | unsigned long long |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [18]      |         temperature          |       float        |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [19]      |         len_penalty          |       float        |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [20]      |      repetition_penalty      |       float        |                                                                                                   **Deprecated, move to input**                                                                                                   |
|      [21]      |         tensor_para          |     NcclParam      |                                                                   Tensor Parallel information, which is declared in `src/fastertransformer/utils/nccl_utils.h`                                                                    |
|      [22]      |        pipeline_para         |     NcclParam      |                                                                  Pipeline Parallel information, which is declared in `src/fastertransformer/utils/nccl_utils.h`                                                                   |
|      [23]      |            stream            |    cudaStream_t    |                                                                                                            CUDA stream                                                                                                            |
|      [24]      |        cublas_wrapper        |  cublasMMWrapper*  |                                                                  Pointer of cuBLAS wrapper, which is declared in `src/fastertransformer/utils/cublasMMWrapper.h`                                                                  |
|      [25]      |          allocator           |    IAllocator*     |                                                                    Pointer of memory allocator, which is declared in `src/fastertransformer/utils/allocator.h`                                                                    |
|      [26]      | is_free_buffer_after_forward |        bool        | If setting to be `true`, FasterTransformer will allocate buffer before forward, and free buffer after forward. When the allocator is based on memory pool, setting to `true` may help reducing the memory usage during inference. |
|      [27]      |       cuda_device_prop       |  cudaDeviceProp*   |                                                           Pointer of CUDA device properties, which is used to get the properties of hardware like size of shared memory                                                           |
|      [28]      |    custom_all_reduce_comm    | AbstractCustomComm |                                                Custom all reduction communication for custom all reduction in model parallelism. It is only supported in 8-way tensor parallelism                                                 |
|      [29]      |   enable_custom_all_reduce   |        int         |                                                                                           Flag of enabling custom all reduction or not                                                                                            |

* Input of GPT-j

|             Name              |            Tensor/Parameter Shape             | Location |       Data Type        |                                                               Description                                                               |
| :---------------------------: | :-------------------------------------------: | :------: | :--------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
|           input_ids           |        [batch_size, max_input_length]         |   GPU    |          int           |                                                         The input ids (context)                                                         |
|         input_lengths         |                 [batch_size]                  |   GPU    |          int           |                                                        The lengths of input ids                                                         |
| prompt_learning_task_name_ids |                 [batch_size]                  |   CPU    |          int           |                                            **Optional**. Task name ids for prompt learning.                                             |
|        output_seq_len         |                 [batch_size]                  |   CPU    |        uint32_t        |                        The largest number of tokens you hope for results. Note that it contains the input length                        |
|           start_id            |                 [batch_size]                  |   CPU    |          int           |                             **Optional**. If FT receives this input, FT will replace default start id by it                             |
|            end_id             |                 [batch_size]                  |   CPU    |          int           |                              **Optional**. If FT receives this input, FT will replace default end id by it                              |
|        stop_words_list        |      [batch_size, 2, stop_words_length]       |   GPU    |          int           |                                       **Optional**. FT would not generate the tokens in the list.                                       |
|        bad_words_list         |       [batch_size, 2, bad_words_length]       |   GPU    |          int           | **Optional**. The words in the list will be When FT generates words in this list, it will stop the generation. An extension of stop id  |
|         runtime_top_k         |              [1] or [batch_size]              |   CPU    |          uint          |                                              **Optional**. top_k value for top k sampling                                               |
|         runtime_top_p         |              [1] or [batch_size]              |   CPU    |         float          |                                              **Optional**. top_p value for top p sampling                                               |
|  beam_search_diversity_rate   |              [1] or [batch_size]              |   CPU    |         float          |                **Optional**. A hyper hyper-parameter for [simple diverse decoding](https://arxiv.org/pdf/1611.08562.pdf)                |
|          temperature          |              [1] or [batch_size]              |   CPU    |         float          |                              **Optional**. Temperature applied to logits for both beam search and sampling                              |
|          len_penalty          |              [1] or [batch_size]              |   CPU    |         float          |                                   **Optional**. Length penalty applied to logits for only beam search                                   |
|      repetition_penalty       |              [1] or [batch_size]              |   CPU    |         float          |                          **Optional**. Repetition penalty applied to logits for both beam search and sampling                           |
|          random_seed          |              [1] or [batch_size]              |   CPU    | unsigned long long int |                                  **Optional**. Random seed to initialize the random table in sampling.                                  |
|    request_prompt_lengths     |                 [batch_size],                 |   CPU    |          int           |     **Optional**. Length of prefix soft prompt embedding. This describes how many tokens of soft prompt embedding in each sentence.     |
|   request_prompt_embedding    | [batch_size, max_prompt_length, hidden_units] |   GPU    |         float          |                 **Optional**. Prefix soft prompt embedding. FT will concat them with results of embedding lookup kernel                 |
|      request_prompt_type      |                 [batch_size]                  |   CPU    |          int           |                  **Optional**. Prompt type of request. This is necessary when user pass the prompt embedding by input                   |
|          memory_len           |                      [1]                      |   CPU    |         uint32         | **Optional**. The maximum time memory used in attention modules. Reduces the memory footprint but quality of generation might degrades. |

* Output of GPT-j

|       Name       |              Tensor/Parameter Shape              | Location | Data Type |                                    Description                                    |
| :--------------: | :----------------------------------------------: | :------: | :-------: | :-------------------------------------------------------------------------------: |
|    output_ids    |   [batch_size, beam_width, max_output_seq_len]   |   GPU    |    int    |            The output ids. It contains the input_ids and generated ids            |
| sequence_length  |             [batch_size, beam_width]             |   GPU    |    int    |                             The lengths of output ids                             |
| output_log_probs | [batch_size, beam_width, request_output_seq_len] |   GPU    |   float   | **Optional**. It records the log probability of logits at each step for sampling. |
|  cum_log_probs   |             [batch_size, beam_width]             |   GPU    |   float   |          **Optional**. Cumulative log probability of generated sentences          |

The `beam_width` value is set by the output shape directly. When the `beam_width` of `output_ids` is larger than 1, FT will use beam search to generate tokens; otherwise, FT will use topk or topp sampling. When the inputs of beam search and sampling is invalid, like beam width 1, top k 0, top p 0.0, FT will run greedy search automatically.

### Supported features

* Checkpoint converter
  * EleutherAI
  * Huggingface
* Data type
  * FP32
  * FP16
  * BF16
* Feature
  * Multi-GPU multi-node inference
  * Dynamic random seed
  * Stop tokens
  * Bad words list
  * Beam search and sampling are both supported
* Frameworks
  * Triton backend

## Setup

### Requirements

- CMake >= 3.13 for PyTorch
- CUDA 11.0 or newer version
- NCCL 2.10 or newer version
- Python: Only verify on python 3
- PyTorch: Verify on 1.8.0, >= 1.5.0 should work.

Recommend use nvcr image like `nvcr.io/nvidia/pytorch:22.07-py3`.

These components are readily available within the NGC Docker image below.

Ensure you have the following components:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) and NGC container are recommended
- [NVIDIA Pascal](https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/) or [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) or [Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/) based GPU 

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:

- [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
- [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
- [Running PyTorch](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)

For those unable to use the NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

### Docker image

* The model was built and tested with the use nvcr image `nvcr.io/nvidia/pytorch:22.07-py3`. e.g.

    ```bash
    nvidia-docker run -ti --rm nvcr.io/nvidia/pytorch:22.07-py3 bash
    ```

### Build project

* Get the code and install all dependencies:

    ```bash
    git clone https://github.com/NVIDIA/FasterTransformer.git
    mkdir -p FasterTransformer/build
    cd FasterTransformer/build
    git submodule init && git submodule update
    pip3 install fire jax jaxlib
    ```

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

    ```bash
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_MULTI_GPU=ON ..
    make -j
    ```

### Download the model

* Download the mystic public model and convert

    ```bash
    wget https://mystic.the-eye.eu/public/AI/GPT-J-6B/step_383500_slim.tar.zstd
    unzstd step_383500_slim.tar.zstd
    tar -axf step_383500_slim.tar
    python3 ../examples/pytorch/gptj/utils/gptj_ckpt_convert.py --output-dir ../models/j6b_ckpt --ckpt-dir ./step_383500/
    ```

The script accepts the following arguments:
1. `--output-dir` is the path of the base directory where the weight binary files will be saved. If `--output-dir` terminates with `.pt` the script just converts the checkpoint to a Pytorch model file that can be read by the [GPT-J implementation in HuggingFace's transformer](https://github.com/finetuneanon/transformers).
2. `--ckpt-dir` is the path to the extracted checkpoint. If `--ckpt-dir` terminates with `.pt` then the script reads the Pytorch model file instead than the public checkpoint, which is faster.
3. `--n-inference-gpus` number of GPUs used for inference, defaults to 1. The binary model parameters are saved to `${output-dir}/${n-inference-gpus}-gpu/`

* Download the huggingface gptj model and convert

    ```bash
    sudo apt-get install git-lfs
    git lfs install
    git clone https://huggingface.co/EleutherAI/gpt-j-6B
    python3 ../examples/pytorch/gptj/utils/huggingface_gptj_ckpt_convert.py --ckpt-dir gpt-j-6B/ --output-dir gpt-j-6B/c-models/ --n-inference-gpus 1
    ```

The script accepts the following arguments:
1. `--output-dir` is the path of the base directory where the weight binary files will be saved.
2. `--ckpt-dir` is the path to the extracted checkpoint.
3. `--n-inference-gpus` number of GPUs used for inference, defaults to 1. The binary model parameters are saved to `${output-dir}/${n-inference-gpus}-gpu/`

### Download tables

* The vocabolary and merge tables are the same as for GPT

    ```bash
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -P ../models
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -P ../models
    ```

### Run GPT-J

* Generate the `gemm_config.in` file.\
  Data Type = 0 (FP32) or 1 (FP16) or 2 (BF16)
    ```bash
    ./bin/gpt_gemm <batch_size> <beam_width> <max_input_len> <head_number> <size_per_head> <inter_size> <vocab_size> <data_type> <tensor_para_size>
    E.g., ./bin/gpt_gemm 8 1 32 16 256 16384 50400 1 1
    ```

* Run GPT on C++

    Users can see the details of arguments in `examples/cpp/gptj/gptj_config.ini`. It controls the model path, model size, tensor parallelism size, and some hyper-parameters.

    ```bash
    mpirun -n 1 --allow-run-as-root ./bin/gptj_example
    ```

E.g. by setting the `data_type` of `gpt_config.ini` to `fp16` or `bf16`, users can run gpt model under fp16/bf16.

* Convert the token ids to sentence.

    ```bash
    python ../examples/pytorch/gpt/utils/gpt_token_converter.py
    ```

* The model supports both pipeline and tensor parallelism. Users can use `tensor_para_size` and `pipeline_para_size` in `../examples/cpp/gptj/gptj_config.ini` to control the size of model parallel. Note that the number of processes must equal to `tensor_para_size * pipeline_para_size`. For tensor parallelism, the model parameters need to be prepared with the `gptj_ckpt_convert.py` script and `--n-inference-gpus=$NGPUS` as described above.

* Provide a bad tokens list that should not be generated (optional). You can use the script `../examples/pytorch/gpt/utils/word_list.py` to convert a python `List[List[int]]` to a format understandable by FasterTransformer. Beware of tokenizer subtleties, "word" and "\<Space\>word" are usually mapped to two uniques token.

* There is also an example of running GPT-J as a Triton model. This example does not involve a client.

    ```bash
    export CUDA_VISIBLE_DEVICES=0
    mpirun -n 1 --allow-run-as-root ./bin/gptj_triton_example
    ```

    To run with tensor and/or pipeline parallelism, make more GPUs visible, edit the `../examples/cpp/gptj/gptj_config.ini` and generate the parameter files with  `gptj_ckpt_convert.py` accordingly.


### Run GPTJ with prompts

GPTJ now supports prefix_prompt.

1.  Convert the prompt weights

    You need to transpose the prefix prompt weights to the shape [num_layers, 2, num_heads, perfix_seq_len, size_per_head], and save it by numpy. The naming style is like ` model.prefix_prompt.<task_name>.weights.<tensor_para_size>.bin`.

    Note that you need to specify `start_id`, `end_id` by yourself in order to make sure that it is consistent with the tokenizer.

2.  Run GPT with C++ example

    You need to specify the example gpt_config.ini like below to enable the p/prompt_tuning feature.

    ```ini
    [gpt_124M]
    head_num=12
    size_per_head=64
    vocab_size=50257
    decoder_layers=12
    start_id=50256
    end_id=50256
    inter_size=3072
    num_tasks=3
    prompt_learning_type=3

    [gpt_124M_task_0]
    task_name = squad
    prompt_length = 10

    [gpt_124M_task_1]
    task_name = sentiment
    prompt_length = 10

    [gpt_124M_task_2]
    task_name = intent_and_slot
    prompt_length = 10
    ```

    `task_name` and `prompt_length` are specified for loading prompt weights.

    **prompt_learning_type**:

    - no prompt: 0
    - soft_prompt: 1
    - prefix_prompt: 2
    - p/prompt_tuning: 3


### Compare with reference implementation

* Install the reference implementation from finetuneanon:

    ```bash
    git clone https://github.com/finetuneanon/transformers
    pip3 install -e ./transformers
    ```

* Convert the checkpoint to a Pytorch model file (assuming the checkpoint `step_383500_slim` was already downloaded and extracted):

    ```bash
    python3 ../examples/pytorch/gptj/utils/gptj_ckpt_convert.py --output-dir j6b_ckpt.pt --ckpt-dir ./step_383500
    ```

* Run the model:

    ```bash
    python3 ../examples/pytorch/gptj/utils/reference_gptj.py
    ```

Testing was performed by comparing the logits for the model given the context.

### gpt-j with triton backend

Details are in [transformer_backend](https://github.com/triton-inference-server/fastertransformer_backend)
