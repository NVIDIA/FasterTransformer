# GPT

## Table Of Contents

- [GPT](#gpt)
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
    - [Prepare](#prepare-1)
      - [Download openai-gpt model and convert](#download-openai-gpt-model-and-convert)
      - [Download megatron model and convert](#download-megatron-model-and-convert)
      - [Download onnx model and convert](#download-onnx-model-and-convert)
      - [Download huggingface gpt model and convert](#download-huggingface-gpt-model-and-convert)
    - [Run GPT](#run-gpt)
    - [Run GPT with prompts](#run-gpt-with-prompts)
    - [Run Meta OPT](#run-meta-opt)
    - [gpt with triton backend](#gpt-with-triton-backend)
    - [Advanced features](#advanced-features)
      - [generate different sentences and enable shared context](#generate-different-sentences-and-enable-shared-context)
      - [Interactive generation](#interactive-generation)
  - [Performance](#performance)
    - [Large model inference with model parallel](#large-model-inference-with-model-parallel)
      - [Performance of Megatron-530B](#performance-of-megatron-530b)
      - [Performance of GPT-175B](#performance-of-gpt-175b)
      - [Perofrmance of GPT-89B](#perofrmance-of-gpt-89b)
      - [Performance of GPT-20B](#performance-of-gpt-20b)
      - [Performance of GPT-6.7B](#performance-of-gpt-67b)
      - [Performance of GPT-1.3B](#performance-of-gpt-13b)
      - [Performance of GPT-350M](#performance-of-gpt-350m)

## Introduction

This document describes what FasterTransformer provides for the GPT model, explaining the workflow and optimization. We also provide a guide to help users to run the GPT model on FasterTransformer. Finally, we provide benchmark to demonstrate the speed of FasterTransformer on GPT. 

GPT is a variant of Decoding model, which does not have the encoder module, cross multi-head attention, and uses GeLU as the activation. In 2020, OpenAI shows that using very giant model and lots of training data can significantly improve the capacity of GPT model in [their paper](https://arxiv.org/abs/2005.14165). However, it is impossible to put such model into a single GPU. For example, the largest model, GPT-3, has 175 billion parameters, which takes about 350 GBs under half data type. Therefore, multi-gpus, even multi-nodes, is necessary. To solve the bottleneck of latency and memory due to the model size, FasterTransformer provides kernels with high efficiency, optimized memory usage, and model parallelism on multiple frameworks. 

### Supported features

* Checkpoint converter
  * Huggingface
  * Megatron
  * Nemo Megatron
  * TensorFlow 
* Data type
  * FP32
  * FP16
  * BF16
  * INT8 weight only PTQ for bs 1 and 2
* Feature
  * Multi-GPU multi-node inference
  * Dynamic random seed
  * Stop tokens
  * Beam search and sampling are both supported
  * Loading FP32 or FP16 weights
* Frameworks
  * TensorFlow
  * PyTorch
  * C++
  * Triton backend

## Model architecture

### Workflow

<div align=center><img width=600 src ="images/gpt/gpt.png "/></div>
<div align=center> Fig 1. Workflow of GPT model.</div>
<br/><br/>

Fig 1 demonstrates the workflow of FasterTransformer GPT. Different from BERT and encoder-decoder structure, GPT receive some input ids as context, and generates the respective output ids as response. In this workflow, the major bottleneck is the GptDecoderLayer (transformer block) because the time increase linearly when we increase the number of layers. In GPT-3, the GptDecoderLayer takes about 95% of total time. 

FasterTransformer splits the whole workflow into 2 parts. The first one is “computing the k/v cache of context (input ids), and the second part is “auto-regressive generating the output ids”. The operations of these two parts are similar, but the shapes of tensors in the `SelfAttention` is different. So, we use 2 different implementations to handle two different cases, as demonstrating in Fig 2. In `DecoderSelfAttention`, the sequence length of query is always 1, so we used customed fused masked multi-head attention kernel to handle. On the other hand, the sequence length of query in the `ContextSelfAttention` is maximum input length, so we use cuBLAS to leverage the tensor core. 

<div align=center>
  <img width=400 src ="images/gpt/gpt_context.png "/> &ensp;&ensp;&ensp;&ensp;&ensp;
  <img width=400 src ="images/gpt/parallelgpt.png "/>
</div>
<div align=center> 
  Fig 2. Comparison between different self attention. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;
  Fig 3. Workflow of GPT with tensor parallelism.
</div>
<br/><br/>

The following examples demonstrating how to run multi-GPU and multi-node GPT model.
1. `examples/cpp/multi_gpu_gpt_example.cc`: It uses MPI to organize all GPUs.
2. `examples/cpp/multi_gpu_gpt_triton_example.cc`: It uses threading for intra node, and MPI for inter node. This example also demonstrates how to use Triton backend API of FasterTransformer to run the GPT model.
3. `examples/pytorch/gpt/multi_gpu_gpt_example.py`: This example is similar to `examples/cpp/multi_gpu_gpt_example.cc`, but encapsulate the instance of FasterTransformer by PyTorch OP.

In summary, the workflow to run the GPT model is:
1.	Initializing the NCCL comm and setting ranks of tensor parallel and pipeline parallel by MPI or threading
2.	Load weights by the ranks of tensor parallel, pipeline parallel and other model hyper-parameters.
3.	Create the instance of `ParalelGpt` by the ranks of tensor parallel, pipeline parallel and other model hyper-parameters.
4.	Receive the request from client and convert the request to the format of input tensors for ParallelGpt.
5.	Run forward
6.	Convert the output tensors of ParallelGpt to response of client and return the response. 
In c++ example codes, we skip the step 4 and step 6, loading the request by `examples/cpp/multi_gpu_gpt/start_ids.csv`. In PyTorch example codes, the request comes from the PyTorch side. In Triton example codes, we have a completed examples from step 1 to step 6.

The source codes are put in `src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.cc`. The arguments, input tensors and output tensors of GPT:

* Constructor of GPT

| Classification |             Name             |     Data Type      |                                                                                                                         Description                                                                                                                          |
| :------------: | :--------------------------: | :----------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      [0]       |        max_batch_size        |       size_t       |                                                                                                                **Deprecated, move to input**                                                                                                                 |
|      [1]       |         max_seq_len          |       size_t       |                                                                                                                **Deprecated, move to input**                                                                                                                 |
|      [2]       |        max_input_len         |       size_t       |                                                                                                                **Deprecated, move to input**                                                                                                                 |
|      [3]       |          beam_width          |       size_t       |                                                                                                                **Deprecated, move to input**                                                                                                                 |
|      [4]       |           head_num           |       size_t       |                                                                                                             Head number for model configuration                                                                                                              |
|      [5]       |        size_per_head         |       size_t       |                                                                                                            Size per head for model configuration                                                                                                             |
|      [6]       |          inter_size          |       size_t       |                                                                                   The inter size of feed forward network. It is often set to 4 * head_num * size_per_head.                                                                                   |
|      [7]       |          num_layer           |       size_t       |                                                                                                     Number of transformer layers for model configuration                                                                                                     |
|      [8]       |          vocab_size          |        int         |                                                                                                           Vocabulary size for model configuration                                                                                                            |
|      [9]       |           start_id           |        int         |                                                                                                                   Start id for vocabulary                                                                                                                    |
|      [10]      |            end_id            |        int         |                                                                                                                    End id for vocabulary                                                                                                                     |
|      [11]      |   prompt_learning_start_id   |        int         |                                                                                                       The start id of virtual token in p/prompt-tuning                                                                                                       |
|      [12]      |     prompt_learning_type     | PromptLearningType |                                                 The type of prompt learning when we load the prompt embedding in constructor. FT supports `no_prompt`, `soft_prompt`, `prefix_prompt`, `p_prompt_tuning` now                                                 |
|      [13]      |      gpt_variant_params      |  gptVariantParams  |                                                                            This structure defines some hyper-parameters of gpt layers, including type of layernorm and activation                                                                            |
|      [14]      |  beam_search_diversity_rate  |       float        |                                                                                                                **Deprecated, move to input**                                                                                                                 |
|      [15]      |            top_k             |       size_t       |                                                                                                                **Deprecated, move to input**                                                                                                                 |
|      [16]      |            top_p             |       float        |                                                                                                                **Deprecated, move to input**                                                                                                                 |
|      [17]      |         random_seed          | unsigned long long |                                                                                                                **Deprecated, move to input**                                                                                                                 |
|      [18]      |         temperature          |       float        |                                                                                                                **Deprecated, move to input**                                                                                                                 |
|      [19]      |         len_penalty          |       float        |                                                                                                                **Deprecated, move to input**                                                                                                                 |
|      [20]      |      repetition_penalty      |       float        |                                                                                                                **Deprecated, move to input**                                                                                                                 |
|      [21]      |         tensor_para          |     NcclParam      |                                                                                 Tensor Parallel information, which is declared in `src/fastertransformer/utils/nccl_utils.h`                                                                                 |
|      [22]      |        pipeline_para         |     NcclParam      |                                                                                Pipeline Parallel information, which is declared in `src/fastertransformer/utils/nccl_utils.h`                                                                                |
|      [23]      |            stream            |    cudaStream_t    |                                                                                                                         CUDA stream                                                                                                                          |
|      [24]      |        cublas_wrapper        |  cublasMMWrapper*  |                                                                               Pointer of cuBLAS wrapper, which is declared in `src/fastertransformer/utils/cublasMMWrapper.h`                                                                                |
|      [25]      |          allocator           |    IAllocator*     |                                                                                 Pointer of memory allocator, which is declared in `src/fastertransformer/utils/allocator.h`                                                                                  |
|      [26]      | is_free_buffer_after_forward |        bool        |              If setting to be `true`, FasterTransformer will allocate buffer before forward, and free buffer after forward. When the allocator is based on memory pool, setting to `true` may help reducing the memory usage during inference.               |
|      [27]      |       cuda_device_prop       |  cudaDeviceProp*   |                                                                        Pointer of CUDA device properties, which is used to get the properties of hardware like size of shared memory                                                                         |
|      [28]      |            sparse            |        bool        |                                                                                                         Is using sparsity. **Experimental feature**                                                                                                          |
|      [29]      |          int8_mode           |        int         |                                                                                             Using int8 weight only quantization or not. **Experimental feature**                                                                                             |
|      [30]      |    custom_all_reduce_comm    | AbstractCustomComm |                                                              Custom all reduction communication for custom all reduction in model parallelism. It is only supported in 8-way tensor parallelism                                                              |
|      [31]      |   enable_custom_all_reduce   |        int         |                                                                                                         Flag of enabling custom all reduction or not                                                                                                         |
|      [32]      |        remove_padding        |        bool        |                                                                                                   Remove the padding of input ids or not in context phase.                                                                                                   |
|      [33]      |    shared_contexts_ratio     |       float        | Ratio that controls the use of the shared contexts optimization. If the compact size (that accounts only for unique prompts) is less than ratio * batch size, use the optimized implementation. Setting shared_contexts_ratio=0 deactivate the optimization. |

* Input of GPT

|              Name               |            Tensor/Parameter Shape             | Location |       Data Type        |                                                                                         Description                                                                                         |
| :-----------------------------: | :-------------------------------------------: | :------: | :--------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|            input_ids            |        [batch_size, max_input_length]         |   GPU    |          int           |                                                                                   The input ids (context)                                                                                   |
|          input_lengths          |                 [batch_size]                  |   GPU    |          int           |                                                                                  The lengths of input ids                                                                                   |
|  prompt_learning_task_name_ids  |                 [batch_size]                  |   CPU    |          int           |                                                                      **Optional**. Task name ids for prompt learning.                                                                       |
|         output_seq_len          |                 [batch_size]                  |   CPU    |        uint32_t        |                                                  The largest number of tokens you hope for results. Note that it contains the input length                                                  |
|         stop_words_list         |      [batch_size, 2, stop_words_length]       |   GPU    |          int           |                                          **Optional**. When FT generates words in this list, it will stop the generation. An extension of stop id                                           |
|         bad_words_list          |       [batch_size, 2, bad_words_length]       |   GPU    |          int           |                           **Optional**. The words in the list will be When FT generates words in this list, it will stop the generation. An extension of stop id                            |
|            start_id             |                 [batch_size]                  |   CPU    |          int           |                                                       **Optional**. If FT receives this input, FT will replace default start id by it                                                       |
|             end_id              |                 [batch_size]                  |   CPU    |          int           |                                                        **Optional**. If FT receives this input, FT will replace default end id by it                                                        |
|          runtime_top_k          |              [1] or [batch_size]              |   CPU    |          uint          |                                                                        **Optional**. top_k value for top k sampling                                                                         |
|          runtime_top_p          |              [1] or [batch_size]              |   CPU    |         float          |                                                                        **Optional**. top_p value for top p sampling                                                                         |
|   beam_search_diversity_rate    |              [1] or [batch_size]              |   CPU    |         float          |                                          **Optional**. A hyper hyper-parameter for [simple diverse decoding](https://arxiv.org/pdf/1611.08562.pdf)                                          |
|           temperature           |              [1] or [batch_size]              |   CPU    |         float          |                                                        **Optional**. Temperature applied to logits for both beam search and sampling                                                        |
|           len_penalty           |              [1] or [batch_size]              |   CPU    |         float          |                                                             **Optional**. Length penalty applied to logits for only beam search                                                             |
|       repetition_penalty        |              [1] or [batch_size]              |   CPU    |         float          |                                                    **Optional**. Repetition penalty applied to logits for both beam search and sampling                                                     |
|           random_seed           |              [1] or [batch_size]              |   CPU    | unsigned long long int |                                                            **Optional**. Random seed to initialize the random table in sampling.                                                            |
|     request_prompt_lengths      |                 [batch_size],                 |   CPU    |          int           |                               **Optional**. Length of prefix soft prompt embedding. This describes how many tokens of soft prompt embedding in each sentence.                               |
|    request_prompt_embedding     | [batch_size, max_prompt_length, hidden_units] |   GPU    |  float/half/bfloat16   | **Optional**. FT will concat them with results of embedding lookup kernel. For prefix soft prompt embedding, the type must be float; for p/prompt tuning, the type is same to weight. |
|       request_prompt_type       |                 [batch_size]                  |   CPU    |          int           |                                            **Optional**. Prompt type of request. This is necessary when user pass the prompt embedding by input                                             |
| is_return_context_cum_log_probs |                      [1]                      |   CPU    |          bool          |                                                            **Optional**. Return the cumulative log probability of context or not                                                            |
|           session_len           |                      [1]                      |   CPU    |         uint32         |                             **Optional**. The maximum time length allowed during the whole interactive generation. Only used for interactive generation feature                             |
|          continue_gen           |                      [1]                      |   CPU    |          bool          |   **Optional**. A flag to tell FasterTransformer to not discard previous tokens and continue producing token based on previous generations. Only used for interactive generation feature    |
|           memory_len            |                      [1]                      |   CPU    |         uint32         |                           **Optional**. The maximum time memory used in attention modules. Reduces the memory footprint but quality of generation might degrades.                           |

* Output of GPT

|       Name       |              Tensor/Parameter Shape              | Location | Data Type |                                    Description                                    |
| :--------------: | :----------------------------------------------: | :------: | :-------: | :-------------------------------------------------------------------------------: |
|    output_ids    |   [batch_size, beam_width, max_output_seq_len]   |   GPU    |    int    |            The output ids. It contains the input_ids and generated ids            |
| sequence_length  |             [batch_size, beam_width]             |   GPU    |    int    |                             The lengths of output ids                             |
| output_log_probs | [batch_size, beam_width, request_output_seq_len] |   GPU    |   float   | **Optional**. It records the log probability of logits at each step for sampling. |
|  cum_log_probs   |             [batch_size, beam_width]             |   GPU    |   float   |          **Optional**. Cumulative log probability of generated sentences          |

The `beam_width` value is set by the output shape directly. When the `beam_width` of `output_ids` is larger than 1, FT will use beam search to generate tokens; otherwise, FT will use topk or topp sampling. When the inputs of beam search and sampling is invalid, like beam width 1, top k 0, top p 0.0, FT will run greedy search automatically.

### Optimization

1.	Kernel optimization: many kernels are based on the kernels of decoder and decoding modules, which are already highly optimized. To prevent from recomputing the previous keys and values, we will allocate a buffer to store them at each step. Although it takes some additional memory usage, we can save the cost of recomputing, allocating buffer at each step, and the cost of concatenation.
2.	Memory optimization: Different to traditional models like BERT, GPT-3 has 175 billion parameters, taking 350 GBs even if we store the model by half precision. Therefore, we must reduce the memory usage for other parts. In FasterTransformer, we will reuse the memory buffer of different decoder layers. Since the number of layers in GPT-3 is 96, we only need 1/96 memory.
3.	Model parallelism: In GPT model, FasterTransormer provides both tensor parallelism and pipeline parallelism. For tensor parallelism, FasterTransformer follows the idea of [Megatron]( https://arxiv.org/pdf/1909.08053.pdf). For both self-attention block and feed forward network block, we split the weights of first matrix multiplication by row and split the weights of the second matrix multiplication by column. By optimization, we can reduce the reduction operation to 2 times for each transformer block. The workflow is demonstrated in Fig 3. For pipeline parallelism, FasterTransformer splits the whole batch of request into multiple micro batches and hide the bubble of communication. FasterTransformer will adjust the micro batch size automatically for different cases. Users can adjust the model parallelism by modifying the `gpt_config.ini` file. We recommend to use tensor parallel intra node, and use pipeline parallel inter node because tensor parallel requires more NCCL communication.
4.	Multiple frameworks: Except the source codes on c, FasterTransformer also provide the TensorFlow op, PyTorch op and Triton backend. Currently, TensorFlow op only supports the single GPU, while PyTorch op and Triton backend support multi-GPU and multi-node. To prevent the additional work of splitting model for model parallelism, FasterTransformer also provides a tool to split and convert the model of Megatron to binary files, then FasterTransformer can load the model in binary directly.

## Setup

The following guide demonstrates how to run the examples of c++, PyTorch and Triton backend.

### Requirements

- CMake >= 3.8 for Tensorflow, CMake >= 3.13 for PyTorch
- CUDA 11.0 or newer version
- NCCL 2.10 or newer version
- Python: Only verify on python 3
- Tensorflow: Verify on 1.15, 1.13 and 1.14 should work.
- PyTorch: Verify on 1.8.0, >= 1.5.0 should work.

Recommend use nvcr image like `nvcr.io/nvidia/tensorflow:22.07-tf1-py3` or `nvcr.io/nvidia/pytorch:22.07-py3`.

These components are readily available within the NGC TensorFlow Docker image below.

Ensure you have the following components:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) and NGC container are recommended
- [NVIDIA Pascal](https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/) or [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) or [Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/) based GPU 

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:

- [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
- [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
- [Running TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running)
- [Running PyTorch](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)

For those unable to use the NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).


### Build the FasterTransformer

#### Prepare

    You can choose the tensorflow version and python version you want. Here, we list some possible images:

    To achieve best performance, we recommend to use the latest image. For example, running image `nvcr.io/nvidia/tensorflow:22.07-tf1-py3` by 

    ```bash
    nvidia-docker run -ti --rm nvcr.io/nvidia/tensorflow:22.07-tf1-py3 bash
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

1. build with C++

    ```bash
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_MULTI_GPU=ON ..
    make
    ```

2. build with TensorFlow 

    Uses need to set the path of TensorFlow. For example, if we use `nvcr.io/nvidia/tensorflow:22.07-tf1-py3`, then

    ```bash
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_TF=ON -DTF_PATH=/usr/local/lib/python3.8/dist-packages/tensorflow_core/ -DBUILD_MULTI_GPU=ON ..
    make 
    ```

3. build with PyTorch

    ```bash
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
    make
    ```

    This will build the TorchScript custom class. Please make sure that the `PyTorch >= 1.5.0`.

## How to use

### Prepare

* Install required tools

```bash
pip install -r ../examples/pytorch/gpt/requirement.txt
```

To run the GPT on c, users need to convert the checkpoint of TensorFlow or PyTorch to binary files, and then load by FasterTransformer c api. Unfortunately, there is no published large model. So, users are only able to verify the correctness by smaller model. Currently, FasterTransformer provides two kinds of samples. First one is using the checkpoint of [OpenAI GPT-2 model](https://github.com/openai/gpt-2) (which is trained by TensorFlow); Another choice is using the checkpoint of [Megatron](https://github.com/NVIDIA/Megatron-LM) (which is trained by pytorch).

* Download vocab and merge table

They can be used in both OpenAI GPT-2 and Megatron.

```bash
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -P ../models
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -P ../models
```

#### Download openai-gpt model and convert

To convert the OpenAI GPT model to binary, FasterTransformer provides a tool `sample/tensorflow/utils/openai_gpt_ckpt_convert.py` to convert the checkpoint. The converter requires the following arguments: 

1. `-i`: The path of megatron model
2. `-o`: The output path of converted model
3. `-t_g`: The tensor parallel size to train the model
4. `-i_g`: The tensor parallel size we hope for inference
5. `-h_n`: Number of heads, which is the hyper-parameter of the model

```bash
mkdir -p ../models/openai-gpt-models/
python tensorflow/utils/download_gpt2_model.py <model_name>
e.g. python ../examples/tensorflow/gpt/utils/download_gpt2_model.py 124M
mv models/124M ../models/openai-gpt-models/
python ../examples/tensorflow/gpt/utils/openai_gpt_ckpt_converter.py -o ../models/openai-gpt-models/c-model/124m/ -i ../models/openai-gpt-models/124M/model.ckpt -g 1 # convert 124M model with 1 TP mode
python ../examples/tensorflow/gpt/utils/openai_gpt_ckpt_converter.py -o ../models/openai-gpt-models/c-model/124m/ -i ../models/openai-gpt-models/124M/model.ckpt -g 4 # convert 124M model with 4 TP mode
```

In the repo of OpenAI, they provide many models, including `124M`, `355M`, `774M` and `1558M`

#### Download megatron model and convert

To convert the Megatron GPT model to binary, FasterTransformer provides a tool `examples/pytorch/utils/megatron_ckpt_convert.py` to convert the checkpoint.

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
mkdir -p ../models/megatron-models/345m
unzip megatron_lm_345m_v0.0.zip -d ../models/megatron-models/345m
export PYTHONPATH=$PWD/..:${PYTHONPATH}
python ../examples/pytorch/gpt/utils/megatron_ckpt_convert.py \
        -head_num 16 \
        -i ../models/megatron-models/345m/release/ \
        -o ../models/megatron-models/c-model/345m/ \
        -t_g 1 \
        -i_g 1 \
        --vocab-path ../models/gpt2-vocab.json \
        --merges-path ../models/gpt2-merges.txt
python ../examples/pytorch/gpt/utils/megatron_ckpt_convert.py \
        -head_num 16 \
        -i ../models/megatron-models/345m/release/ \
        -o ../models/megatron-models/c-model/345m/ \
        -t_g 1 \
        -i_g 8 \
        --vocab-path ../models/gpt2-vocab.json \
        --merges-path ../models/gpt2-merges.txt
```

where `t_g` means the number GPUs of TP during training, and `i_g` means the number of GPUs for TP during inference.

Note that there are different checkpoint version of Megatron. The version of the checkpoint above is 0.

For model trained by pipeline parallelism or the checkpoint version is 3, you don't need to specify head_num or checkpoint_version as it can retrieve from model_args.

```bash
python ../examples/pytorch/gpt/utils/megatron_ckpt_convert.py -i ../models/megatron-models/345m/release/ -o ../models/megatron-models/c-model/345m/ -i_g 1
```

#### Download onnx model and convert

Note that the original `gpt2-10.onnx` model at `https://github.com/onnx/models/raw/master/text/machine_comprehension/gpt-2/model/gpt2-10.onnx` is removed. And new link `https://github.com/onnx/models/blob/main/text/machine_comprehension/gpt-2/model/gpt2-10.onnx` cannot be loaded by onnx successfully.

To convert the ONNX GPT model to binary, FasterTransformer provides a tool `examples/onnx/multi_gpu_gpt/onnx_ckpt_convert.py` to convert the checkpoint.

```bash
wget https://github.com/onnx/models/blob/main/text/machine_comprehension/gpt-2/model/gpt2-10.onnx
python ../examples/onnx/multi_gpu_gpt/onnx_ckpt_convert.py -i gpt2-10.onnx -o ../models/onnx-models/c-model/124m/ -i_g 1
python ../examples/onnx/multi_gpu_gpt/onnx_ckpt_convert.py -i gpt2-10.onnx -o ../models/onnx-models/c-model/124m/ -i_g 4
```

#### Download huggingface gpt model and convert

```bash
git clone https://huggingface.co/gpt2-xl
python ../examples/pytorch/gpt/utils/huggingface_gpt_convert.py -i gpt2-xl/ -o ../models/huggingface-models/c-model/gpt2-xl -i_g 1
```

### Run GPT

1. Run GPT under on C++ with multiple gpu

    1.1 Generate the `gemm_config.in` file.\
    Data Type = 0 (FP32) or 1 (FP16) or 2 (BF16)

    ```bash
    ./bin/gpt_gemm <batch_size> <beam_width> <max_input_len> <head_number> <size_per_head> <inter_size> <vocab_size> <data_type> <tensor_para_size>
    E.g., ./bin/gpt_gemm 8 1 32 12 128 6144 51200 1 1
    ```

    Note: We remove the `local_batch_size` argument since v5.0. When users use pipeline parallelism, FT will determine the `local_batch_size` automatically. 

    1.2 Run GPT on C++

    Users can see the details of arguments in `examples/cpp/multi_gpu_gpt/gpt_config.ini`. It controls the model path, model size, tensor parallelism size, and some hyper-parameters.

    ```bash
    ./bin/multi_gpu_gpt_example
    ```

    then use following script to convert the token ids to sentence.

    ```bash
    python ../examples/pytorch/gpt/utils/gpt_token_converter.py --vocab_file=../models/gpt2-vocab.json  --bpe_file=../models/gpt2-merges.txt
    ```

    By setting the `data_type` of `gpt_config.ini` to `fp16` or `bf16`, users can run gpt model under fp16 or bf16.

    1.3 Run with tensor parallelism (TP), pipeline parallelism (PP)

    Users can use `tensor_para_size` and `pipeline_para_size` in `gpt_config.ini` to control the size of model parallel. Note that the number of processes must equal to `tensor_para_size * pipeline_para_size`.

    ```bash
    mpirun -n 8 ./bin/multi_gpu_gpt_example
    python ../examples/pytorch/gpt/utils/gpt_token_converter.py --vocab_file=../models/gpt2-vocab.json  --bpe_file=../models/gpt2-merges.txt
    ```

    1.4 Run gpt on multi-nodes

    Since the c sample codes use the MPI to communicate, it can extend to multi-nodes easily, except that users need to setup some network environment to communicate between multi-nodes. The following scripts are an example to show how to run multi-nodes inference on slurm.

    ```bash
    srun -N2 -n2 -t 600 --pty bash # Assume we get 2 nodes: prm-dgx-09 and prm-dgx-10
    srun -N2 -n2 docker pull nvcr.io/nvidia/tensorflow:22.07-tf1-py3 

    srun -N2 -n2  nvidia-docker run -itd --rm --privileged --network=host --pid=host --cap-add=IPC_LOCK --device=/dev/infiniband -v $PWD:$PWD -w $PWD --name ft-test nvcr.io/nvidia/tensorflow:22.07-tf1-py3 /bin/bash

    srun -N2 -n2  nvidia-docker exec -i --env SLURM_NTASKS --env SLURM_NODEID --env SLURM_PROCID --env SLURM_STEP_NODELIST --env SLURMD_NODENAME --privileged ft-test bash -c "mkdir /root/.ssh && cp $PWD/ssh/* /root/.ssh && chmod 700 /root/.ssh && chmod 640 /root/.ssh/authorized_keys2 && chmod 400 /root/.ssh/id_rsa && apt-get update && apt-get install ssh -y && mkdir /run/sshd/ && /usr/sbin/sshd -p 11068 && nvidia-smi -lgc 1530"

    nvidia-docker exec -ti ft-test bash 
    cd FasterTransformer/build
    mpirun --allow-run-as-root -np 2 -H prm-dgx-09:1,prm-dgx-10:1 -mca plm_rsh_args "-p 11068" ./bin/multi_gpu_gpt_example
    srun -N2 -n2 docker stop ft-test
    ```

2. Run GPT on PyTorch

    Basically, `gpt_example.py` includes the example how to declare a model, load a ckeckpoint, and forward context inputs and get generated outputs in Pytorch.

    For generating outputs based on context inputs, create a text file including the context inputs (line by line) and set `--sample_file_input` to the text file path. (By default, the script will generate outputs without context inputs.) Set `--sample_file_output` to write the outputs to a file. Use `--data_type fp16/bf16` to run in FP16 or BF16.

    Run with `-h` to see more settings.
    ```bash
    python ../examples/pytorch/gpt/gpt_example.py -h
    ```

    2.1 Run GPT with TP and PP on single node (NVIDIA DGX A100). Note that the number of processes must equal to `tensor_para_size * pipeline_para_size`.
    ```bash
    # No parallelism (tensor_para_size=1, pipeline_para_size=1)
    python ../examples/pytorch/gpt/gpt_example.py

    # TP (tensor_para_size=8, pipeline_para_size=1)
    mpirun -n 8 --allow-run-as-root python ../examples/pytorch/gpt/multi_gpu_gpt_example.py --tensor_para_size=8 --pipeline_para_size=1 --ckpt_path="/workspace/fastertransformer/models/megatron-models/c-model/345m/8-gpu"

    # LP (tensor_para_size=1, pipeline_para_size=8)
    mpirun -n 8 --allow-run-as-root python ../examples/pytorch/gpt/multi_gpu_gpt_example.py --tensor_para_size=1 --pipeline_para_size=8 --ckpt_path="/workspace/fastertransformer/models/megatron-models/c-model/345m/1-gpu"

    # TP and LP (tensor_para_size=4, pipeline_para_size=2)
    mpirun -n 8 --allow-run-as-root python ../examples/pytorch/gpt/multi_gpu_gpt_example.py --tensor_para_size=4 --pipeline_para_size=2 --ckpt_path="/workspace/fastertransformer/models/megatron-models/c-model/345m/4-gpu"
    ```

    2.2 Run GPT with TP and PP on single-node/multi-node (NVIDIA SuperPOD)
    #### Set up in interactive mode

    ```bash
    srun -A devtech -J devtech-gpt:gpt -p luna -N1 --mpi=pmix --ntasks-per-node=8 --container-image nvcr.io/nvidia/pytorch:22.07-py3 --container-mounts /lustre/fsw/devtech/hpc-devtech/dahn/FasterTransformer:/workspace/fastertransformer --container-workdir /workspace/fastertransformer --pty bash

    mkdir build && cd build
    cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON .. && make -j12
    ```

    #### Run on singe-node
    * tensor_para_size=8, pipeline_para_size=1

    ```bash
    srun -A devtech -p luna -N1 --mpi=pmix --ntasks-per-node=8 --container-image nvcr.io/nvidia/pytorch:22.07-py3 --container-mounts /lustre/fsw/devtech/hpc-devtech/dahn/FasterTransformer:/workspace/fastertransformer --container-workdir /workspace/fastertransformer/build python ../examples/pytorch/gpt/multi_gpu_gpt_example.py --tensor_para_size=8 --pipeline_para_size=1 --ckpt_path="/workspace/fastertransformer/models/megatron-models/c-model/345m/8-gpu"
    ```

    #### Run on multi-node
    * tensor_para_size=8, pipeline_para_size=2

    ```bash
    srun -A devtech -p luna -N2 --mpi=pmix --ntasks-per-node=8 --container-image nvcr.io/nvidia/pytorch:22.07-py3 --container-mounts /lustre/fsw/devtech/hpc-devtech/dahn/FasterTransformer:/workspace/fastertransformer --container-workdir /workspace/fastertransformer/build python ../examples/pytorch/gpt/multi_gpu_gpt_example.py --tensor_para_size=8 --pipeline_para_size=2 --ckpt_path="/workspace/fastertransformer/models/megatron-models/c-model/345m/8-gpu"
    ```

   2.2 Run LAMBADA test on PyTorch

    download data set:

    ```bash
    wget https://github.com/cybertronai/bflm/raw/master/lambada_test.jsonl -P ../models/megatron-models
    export PYTHONPATH=$PWD/../:$PYTHONPATH
    python ../examples/pytorch/gpt/utils/update_gpt_config.py \
            --model-dir ../models/megatron-models/c-model/345m/1-gpu/ \
            --config-ini-path ../models/megatron-models/c-model/345m/1-gpu/config.ini \
            --pipeline-para-size 1 \
            --tensor-para-size 1 \
            --max-seq-len 512 \
            --beam-width 1 \
            --sampling-top-k 1 \
            --sampling-top-p 0 \
            --data-type fp16
    python ../examples/pytorch/gpt/lambada_task_example.py \
           --batch-size 64 \
           --checkpoint-path ../models/megatron-models/c-model/345m/1-gpu/ \
           --lib-path lib/libth_parallel_gpt.so \
           --lambada-path ../models/megatron-models/lambada_test.jsonl 
    ```

3. Run GPT on tensorflow

    Follow [Download openai-gpt model and convert](#download-openai-gpt-model-and-convert) to prepare the model. Assume the TF model is put in `../models/openai-gpt-models/`.

    ```bash
    ./bin/gpt_gemm 4 1 32 12 64 3072 50257 1 1
    python ../examples/tensorflow/gpt/gpt_example.py --batch_size=4 \
                                                     --length=32 \
                                                     --top_k=4 \
                                                     --top_p=0.6 \
                                                     --data_type=fp16 \
                                                     --models_dir=../models/openai-gpt-models/
    ```

    Note that the tensorflow op only supports single gpu.

### Run GPT with prompts

GPT now supports p/prompt-tuning. It works with [nemo checkpoint and prompt learning](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/prompt_learning.html).

1.  Convert the prompt weights

    Use the `examples/pytorch/gpt/utils/nemo_ckpt_convert.py` to convert the NeMo Megatron Prompt Weights. 
    It will automatically generate configuration needed for triton backend inference.

    Note that you need to specify `start_id`, `end_id` by yourself in order to make sure that it is consistent with the tokenizer.

2.  Run GPT with C++ example

    You need to specify the example gpt_config.ini like below to enable the p/prompt_tuning feature.

    ```ini
    [gptj_6B]
    head_num=16
    size_per_head=256
    vocab_size=50400
    decoder_layers=28
    rotary_embedding=64
    start_id=50256
    end_id=50256
    inter_size=16384
    num_tasks=2
    prompt_learning_type=2

    ;prompt learning example (soft prompt doesn't need it)
    [gptj_6B_task_0]
    task_name=task_0
    prompt_length=5

    [gptj_6B_task_1]
    task_name=task_1
    prompt_length=10
    ```

    `task_name` and `prompt_length` are specified for loading prompt weights.
    `prompt_learning_start_id` is needed for checking whether ids are prompts or normal input ids.

    **prompt_learning_type**:

    - no prompt: 0
    - soft_prompt: 1
    - prefix_prompt: 2
    - p/prompt_tuning: 3

### Run Meta OPT

Meta OPT and OpenAI GPT do not have big differences in terms of structures, so they are sharing the same model and triton backend classes. \
You need to convert the Huggingface Meta Opt models to fastertransformer format by `examples/pytorch/gpt/utils/huggingface_opt_convert.py`.

1. Run OPT under on C++ with multiple gpu

    Users can see the details of arguments in `examples/cpp/multi_gpu_gpt/gpt_config.ini`. It controls the model path, model size, tensor parallelism size, and some hyper-parameters.\
    In order to run with Meta Opt models, you need to add additional configuraitons: `model_variant`, which controls the `layernorm_eps, layernorm_type, activation_type, has_post_decoder_layernorm`.

    For example, the opt 125m model configuraitons would be like:
    ```ini
    [opt_125M]
    head_num=12
    size_per_head=64
    vocab_size=50272
    decoder_layers=12
    start_id=2
    end_id=2
    inter_size=3072
    model_variant=opt-pre ;define variant structure
    ```
    There are two model types: opt-pre = [pre_layernorm](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L332), opt_post = [post_layernorm](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L323)\
    **Note that:** [the model has post decoder layernorm when layernorm_type is pre_layernorm](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L498).

2. Run OPT on PyTorch

    We can run summarization task examples of meta opt models. See `examples/pytorch/gpt/opt_summarization.py`.

    Note that the summarization test are ran by topk = 2, so the rouge score of HF and FT are often different.
    
    * Run on opt-125m model

    ```bash
    git lfs clone https://huggingface.co/facebook/opt-125m
    python ../examples/pytorch/gpt/utils/huggingface_opt_convert.py \
          -i opt-125m/ \
          -o opt-125m/c-model/ \
          -i_g 1
    python3 ../examples/pytorch/gpt/opt_summarization.py \
            --summarize \
            --test_hf \
            --max_ite 20 \
            --ft_model_location opt-125m/c-model \
            --hf_model_name opt-125m
    ```

    The results are similar to:

    ```
    Hugging Face (total latency: 9.258284 sec)
    rouge1 : 20.36984889475218
    rouge2 : 4.854345624891912
    rougeL : 14.82866480289381
    rougeLsum : 18.23638863809613
    Faster Transformers (total latency: 3.9376330000000004 sec)
    rouge1 : 26.676168312282357
    rouge2 : 10.004052949342602
    rougeL : 19.20934213532261
    rougeLsum : 24.243496576656323
    ```

    * Run on opt-350m model

    ```bash
    git lfs clone https://huggingface.co/facebook/opt-350m
    python ../examples/pytorch/gpt/utils/huggingface_opt_convert.py \
          -i opt-350m/ \
          -o opt-350m/c-model/ \
          -i_g 1
    python3 ../examples/pytorch/gpt/opt_summarization.py \
            --summarize \
            --test_hf \
            --max_ite 20 \
            --ft_model_location opt-350m/c-model \
            --hf_model_name opt-350m \
            --data_type fp16
    ```

    The results are similar to:

    ```
    Hugging Face (total latency: 21.961627 sec)
    rouge1 : 28.939621379501467
    rouge2 : 9.858278077813752
    rougeL : 19.159853526952528
    rougeLsum : 26.120654334830885
    Faster Transformers (total latency: 6.293255999999998 sec)
    rouge1 : 26.80687566772978
    rouge2 : 8.639787737378661
    rougeL : 18.90520115636779
    rougeLsum : 24.372302912676407
    ```

3. Run OPT with Triton Backends

    Model configurations have been automatically generated when converting the [meta opt models](https://huggingface.co/docs/transformers/model_doc/opt).\
    Then, you can use the converted weights and configuration file to serve the model by triton servers.
    Example of the `config.ini` when converting the model:
    ```ini
    [gpt]
    model_name = opt-350m/
    head_num = 16
    size_per_head = 64
    inter_size = 4096
    max_pos_seq_len = 2048
    num_layer = 24
    layernorm_eps = 1e-5
    layernorm_type = post_layernorm
    activation_type = Relu
    has_post_decoder_layernorm = 0
    vocab_size = 50272
    start_id = 2
    end_id = 2
    weight_data_type = fp32
    ```

### gpt with triton backend

Details are in [transformer_backend](https://github.com/triton-inference-server/fastertransformer_backend)

### Advanced features

#### generate different sentences and enable shared context

The model downloading and conversion are described in [Download megatron model and convert](#download-megatron-model-and-convert).

A common request is, we have single input request, and hope to reply multiple results with different random seed. To achieve this target, we can mulpitle the inputs by several times, and set different random seed for different sentences in a batch. You can enable it by adding `--enable_random_seed`. Otherwise, all random seed would be set to 0 by default.

For example, we prepare a input with batch size 4, and the sentences are all same.

```bash
for i in {1..4} ; do echo " Article :  (CNN)James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV's \"The Dukes of Hazzard,\" died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he'd been a busy actor for decades in theater and in Hollywood, Best didn't become famous until 1979, when \"The Dukes of Hazzard's\" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best's Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his \"hot pursuit\" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive \"kew-kew-kew\" chuckle and for goofy catchphrases such as \"cuff 'em and stuff 'em! \" upon making an arrest. Among the most popular shows on TV in the early '80s, \"The Dukes of Hazzard\" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best's \"Hazzard\" co-stars paid tribute to the late actor on social media. \"I laughed and learned more from Jimmie in one hour than from anyone else in a whole year,\" co-star John Schneider, who played Bo Duke, said on Twitter. \"Give Uncle Jesse my love when you see him dear friend.\" \"Jimmy Best was the most constantly creative person I have ever known,\" said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. \"Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life's many passions.\" Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career.  TL;DR: " >> sample_input.txt ; done
```

Then, we run the `multi_gpu_gpt_example.py` with `--enable_random_seed`:

```bash
python3 ../examples/pytorch/gpt/multi_gpu_gpt_example.py  \
        --ckpt_path ../models/megatron-models/c-model/345m/1-gpu/ \
        --vocab_file ../models/gpt2-vocab.json  \
        --merges_file ../models/gpt2-merges.txt  \
        --sample_input_file sample_input.txt \
        --max_batch_size 4 \
        --time  \
        --top_p 0.9 \
        --top_k 0 \
        --shared_contexts_ratio 0.0 \
        --enable_random_seed \
        --output_len 8
```

You can see the results are little different, and the program will show the time cost like:

```bash
[INFO] GPT time costs: 64.25 ms
```

Although this method can achieve our target, but computing same duplicated inputs is waste. So, we can set `--shared_contexts_ratio` to compute the duplicated inputs once in context phase:

```bash
python3 ../examples/pytorch/gpt/multi_gpu_gpt_example.py  \
        --ckpt_path ../models/megatron-models/c-model/345m/1-gpu/ \
        --vocab_file ../models/gpt2-vocab.json  \
        --merges_file ../models/gpt2-merges.txt  \
        --sample_input_file sample_input.txt \
        --max_batch_size 4 \
        --time  \
        --top_p 0.9 \
        --top_k 0 \
        --shared_contexts_ratio 1.0 \
        --enable_random_seed \
        --output_len 8
```

You can see the inference is faster than original one like:

```bash
[INFO] GPT time costs: 41.69 ms
```

Notes:
1. The results of enabling `shared_context` and disabling `shared_context` may be different because the shape of GEMM are changed. But it does not affect the qualities of generation.
2. We use short `output_len` in this example to demonstarte the benefit of `shared_context`. In real application, the more duplicated input, longer input length compared to output length, the more speedup `shared_context` brings.
3. Since the additional overhead of enabling `shared_context` is ignorable, we enable it by default. 

#### Interactive generation

<div align=center>
  <img width=1000 src ="images/gpt/gpt_interactive_generation.0.png "/> &ensp;&ensp;&ensp;&ensp;&ensp;
</div>
<div align=center> 
  Fig 4. GPT generate some outputs by some inputs
</div>
<br/><br/>

<div align=center>
  <img width=600 src ="images/gpt/gpt_interactive_generation.1.png "/> &ensp;&ensp;&ensp;&ensp;&ensp;
</div>
<div align=center> 
  Fig 5. New inputs with previous texts and some additional new input ids.
</div>
<br/><br/>

In some scenarios (like chatting), the new requests are related to previous requests. Currently, users can pass all previous inputs and outputs as a new inputs into FT to make FT generate new reply from these previous texts, like what we see in Fig 4 and Fig 5. However, this means that we need to re-compute the k/v cache of all previous inputs and outputs again, which is time wasting when the context is very long.

<div align=center>
  <img width=1000 src ="images/gpt/gpt_interactive_generation.2.png "/> &ensp;&ensp;&ensp;&ensp;&ensp;
</div>
<div align=center> 
  Fig 6. The workflow of generation with interactive generation
</div>
<br/><br/>

To achieve better performance and prevent useless computing, we add a new flag `continue_gen` into GPT. When this flag is on, FT keeps all results during generation and assume the users will provide some more texts. And FT would not compute the k/v cache of the results it already has, but only compute the k/v cache of new ids. The workflow would become what we demonstrate in Fig 6. To prevent allocate the memory buffer again, users also need to set the `session_len` to be the maximum sequence length of the final sentence, but not only for intermediate sentence.

We will use `multi_gpu_gpt_interactive_example` to demonstarte how to use this feature. In this example, we load the `examples/cpp/multi_gpu_gpt/start_ids.csv` first (the input length are all 8):

```
818, 262, 938, 3155, 286, 1528, 11, 257
198, 464, 968, 8221, 2732, 286, 15198, 318
464, 968, 1971, 12056, 423, 257, 649, 1182
464, 968, 1971, 3782, 468, 3199, 663, 5079
818, 257, 1445, 326, 481, 1884, 787, 340
464, 968, 1971, 12056, 6, 5859, 41683, 423
198, 198, 464, 5398, 4332, 628, 628, 198
464, 717, 640, 314, 2497, 262, 3807, 11
```

then generates 32 tokens with setting `continue_gen=true` to get an intermediate results (the results are saved in `out.interm`):

```
818 262 938 3155 286 1528 11 257 1256 286 661 423 587 4737 502 546 262 649 1492 11 290 314 1053 587 2111 284 3280 617 286 262 2683 326 661 423 587 4737 502 13 198 198 
198 464 968 8221 2732 286 15198 318 1762 351 262 1181 338 9358 5011 284 5004 262 1266 835 284 1445 262 4979 13 198 1 1135 821 1016 284 307 2045 379 262 1266 835 284 1445 262 
464 968 1971 12056 423 257 649 1182 3985 11 290 339 338 257 3516 508 338 587 1088 262 4652 329 257 890 640 13 679 338 257 3516 508 338 587 1088 262 4652 329 257 890 640 
464 968 1971 3782 468 3199 663 5079 1351 286 262 995 338 749 14212 661 13 198 464 1351 11 543 373 14102 416 262 968 1971 3782 11 318 1912 319 257 5526 286 517 621 352 11 
818 257 1445 326 481 1884 787 340 4577 329 262 1664 284 3677 663 7303 11 262 1664 468 4987 284 3677 663 10171 287 262 1664 284 257 1448 286 7713 2957 416 262 2839 13598 4081 309 
464 968 1971 12056 6 5859 41683 423 587 257 1263 636 286 262 1074 338 1943 428 1622 13 198 464 12056 423 587 1498 284 1057 262 2613 6840 11 290 484 423 587 1498 284 1057 262 
198 198 464 5398 4332 628 628 198 198 464 5398 4332 628 628 198 198 464 5398 4332 628 628 198 198 464 5398 4332 628 628 198 198 464 5398 4332 628 628 198 198 464 5398 4332 
464 717 640 314 2497 262 3807 11 314 373 588 11 705 5812 616 1793 11 428 318 523 3608 2637 314 373 588 11 705 40 765 284 307 287 428 3807 2637 314 373 588 11 705 
```

Next, we load another inputs from `examples/cpp/multi_gpu+gpt/interactive_inputs_ids` (the input length are all 8 again):

```
5962, 11, 314, 561, 588, 284, 910, 326
11125, 286, 2844, 291, 5028, 422, 262, 7627
392, 257, 1913, 1998, 351, 1353, 12, 28282
830, 34643, 11, 7602, 11, 4708, 6332, 1938
5, 38328, 763, 13, 1119, 481, 2148, 257
3245, 355, 257, 22080, 1074, 13, 4042, 286
14150, 26443, 262, 1230, 338, 1410, 284, 3958
5195, 4398, 470, 314, 7342, 340, 2961, 30
```

and pass into FT again (note that we only need to pass new ids because FT already records all previous ids). Then FT will concatenate these new ids into output ids, compute k/v caches for only these new ids, and then generate another 32 tokens as a new response (the results are saved in `out`):

```
818 262 938 3155 286 1528 11 257 1256 286 661 423 587 4737 502 546 262 649 1492 11 290 314 1053 587 2111 284 3280 617 286 262 2683 326 661 423 587 4737 502 13 198 198 5962 11 314 561 588 284 910 326 314 1101 407 257 4336 286 262 1492 13 314 892 340 338 257 1310 1165 881 286 257 366 10919 611 1 1492 13 314 892 340 338 257 1310 1165 
198 464 968 8221 2732 286 15198 318 1762 351 262 1181 338 9358 5011 284 5004 262 1266 835 284 1445 262 4979 13 198 1 1135 821 1016 284 307 2045 379 262 1266 835 284 1445 262 11125 286 2844 291 5028 422 262 7627 7784 15296 284 262 7421 7784 15296 553 531 42743 6523 3899 1024 33246 271 13 198 464 42743 318 635 2045 379 262 5885 286 3867 262 4979 422 262 7421 
464 968 1971 12056 423 257 649 1182 3985 11 290 339 338 257 3516 508 338 587 1088 262 4652 329 257 890 640 13 679 338 257 3516 508 338 587 1088 262 4652 329 257 890 640 392 257 1913 1998 351 1353 12 28282 18370 13 679 338 257 3516 508 338 587 1088 262 4652 329 257 890 640 13 679 338 257 3516 508 338 587 1088 262 4652 329 257 890 640 13 
464 968 1971 3782 468 3199 663 5079 1351 286 262 995 338 749 14212 661 13 198 464 1351 11 543 373 14102 416 262 968 1971 3782 11 318 1912 319 257 5526 286 517 621 352 11 830 34643 11 7602 11 4708 6332 1938 290 584 14212 661 13 198 464 1351 318 14102 416 262 968 1971 3782 290 318 3199 319 262 3052 286 262 7533 13 198 464 1351 318 20633 416 262 
818 257 1445 326 481 1884 787 340 4577 329 262 1664 284 3677 663 7303 11 262 1664 468 4987 284 3677 663 10171 287 262 1664 284 257 1448 286 7713 2957 416 262 2839 13598 4081 309 5 38328 763 13 1119 481 2148 257 2472 286 720 16 13 20 2997 287 5003 290 4283 13 198 464 1730 318 2938 284 1969 287 262 1218 2063 286 428 614 13 198 464 1664 531 340 
464 968 1971 12056 6 5859 41683 423 587 257 1263 636 286 262 1074 338 1943 428 1622 13 198 464 12056 423 587 1498 284 1057 262 2613 6840 11 290 484 423 587 1498 284 1057 262 3245 355 257 22080 1074 13 4042 286 262 640 11 262 12056 423 587 1498 284 1057 262 2613 6840 11 290 484 423 587 1498 284 1057 262 3245 355 257 22080 1074 13 198 464 12056 423 
198 198 464 5398 4332 628 628 198 198 464 5398 4332 628 628 198 198 464 5398 4332 628 628 198 198 464 5398 4332 628 628 198 198 464 5398 4332 628 628 198 198 464 5398 4332 14150 26443 262 1230 338 1410 284 3958 262 779 286 262 1573 366 16991 1 287 262 1499 338 1743 3303 13 198 198 464 1230 338 1410 284 3958 262 779 286 262 1573 366 16991 1 287 
464 717 640 314 2497 262 3807 11 314 373 588 11 705 5812 616 1793 11 428 318 523 3608 2637 314 373 588 11 705 40 765 284 307 287 428 3807 2637 314 373 588 11 705 5195 4398 470 314 7342 340 2961 30 4162 4398 470 314 1775 340 878 8348 314 373 588 11 705 40 765 284 307 287 428 3807 2637 314 373 588 11 705 40 765 284 307 287 428 
```

## Performance

Hardware settings (A100 SuperPod architecture):

* Intra node: 8xA100-80GBs (with mclk 1593MHz, pclk 1410MHz) with AMD EPYC 7742 64-Core Processor, linked by NVSwitch
* Inter node: Linked by Infiniband, 8x200Gb/s NICs

### Large model inference with model parallel

We demonstrate the inference time of Megatron and FasterTransformer on Triton, and show the speedup of FasterTransformer compare to Megatron for GPT-175B and GPT-89B. In the experiments of GPT, we updated the following parameters:

#### Performance of Megatron-530B

* head_num = 128
* size_per_head = 160
* num_layers = 105
* data_type = FP16
* vocab_size = 51200
* top_p = 0.9

TP means tensor parallelism, PP means pipeline parallelism.

<div align=center><img width=800 src ="images/gpt/Megatron_530B_benchmark_1.png "/></div>
<div align=center> Fig 7. Latency on input length 60, output length 20. TP means tensor parallelism and PP means pipeline parallelism. </div>
<br/><br/>

<div align=center><img width=800 src ="images/gpt/Megatron_530B_benchmark_2.png "/></div>
<div align=center> Fig 8. Throughput per GPU on input length 60, output length 20. TP means tensor parallelism and PP means pipeline parallelism. </div>
<br/><br/>

<div align=center><img width=800 src ="images/gpt/Megatron_530B_benchmark_3.png "/></div>
<div align=center> Fig 9. Latency on fixing output length 20, 16 ways tensor parallelism, different input length and batch size. </div>
<br/><br/>

<div align=center><img width=800 src ="images/gpt/Megatron_530B_benchmark_4.png "/></div>
<div align=center> Fig 10. Latency on fixing input length 128, 16 ways tensor parallelism, different output length and batch size. </div>
<br/><br/>

| Batch Size | Input Length | Output Length | Latency of TP-16, PP-1 (ms) | Latency of TP-32, PP-1 (ms) | Latency of TP-8, PP-3 (ms) |
| :--------: | :----------: | :-----------: | :-------------------------: | :-------------------------: | :------------------------: |
|     1      |      20      |       8       |             565             |             431             |            842             |
|     2      |      20      |       8       |             598             |             455             |            860             |
|     4      |      20      |       8       |             616             |             493             |            867             |
|     8      |      20      |       8       |             660             |             523             |            929             |
|     16     |      20      |       8       |             730             |             575             |            1049            |
|     32     |      20      |       8       |             865             |             672             |            1283            |
|     64     |      20      |       8       |            1191             |             942             |            1722            |
|    128     |      20      |       8       |            1862             |            1431             |            2124            |
|    256     |      20      |       8       |            3341             |            2483             |            3140            |
|            |              |               |                             |                             |                            |
|     1      |      60      |      20       |            1379             |            1037             |            2085            |
|     2      |      60      |      20       |            1515             |            1110             |            2122            |
|     4      |      60      |      20       |            1512             |            1198             |            2184            |
|     8      |      60      |      20       |            1631             |            1295             |            2367            |
|     16     |      60      |      20       |            1868             |            1454             |            2753            |
|     32     |      60      |      20       |            2361             |            1804             |            3543            |
|     64     |      60      |      20       |            3383             |            2646             |            4117            |
|    128     |      60      |      20       |            5406             |            4099             |            5319            |
|    256     |      60      |      20       |             OOM             |            7203             |            8318            |
|            |              |               |                             |                             |                            |
|     1      |     128      |       8       |             585             |            451             |             866             |
|     2      |     128      |       8       |             667             |            508             |             932             |
|     4      |     128      |       8       |             765             |            606             |            1097             |
|     8      |     128      |       8       |             990             |            766             |            1434             |
|     16     |     128      |       8       |            1377             |            1074            |            2104             |
|     32     |     128      |       8       |            2251             |            1741            |            2623             |
|     64     |     128      |       8       |            4002             |            3114            |            3578             |
|    128     |     128      |       8       |             OOM             |            5784            |            5512             |
|    256     |     128      |       8       |             OOM             |           11232            |            9614             |

#### Performance of GPT-175B

* head_num = 96
* size_per_head = 128
* num_layers = 96
* data_type = FP16
* vocab_size = 51200
* top_p = 0.9
* tensor_parallel_size = 8 with NVLink

| Batch_size | Input Seqlen | Output Seqlen | Megatron <br/> Latency (ms) | FT <br/> Latency (ms) | FT <br/> Speedup |
| :--------: | :----------: | :-----------: | :-------------------------: | :-------------------: | :--------------: |
|     1      |     128      |       8       |           660.38            |        488.86         |       1.35       |
|     2      |     128      |       8       |           687.34            |        509.47         |       1.35       |
|     4      |     128      |       8       |           1004.88           |        629.64         |       1.60       |
|     8      |     128      |       8       |           1705.07           |        749.86         |       2.27       |
|     12     |     128      |       8       |           2365.02           |        886.24         |       2.67       |
|     16     |     128      |       8       |           3111.57           |        1037.47        |       3.00       |
|     20     |     128      |       8       |           3723.73           |        1135.72        |       3.28       |
|     32     |     128      |       8       |           5778.72           |        1547.44        |       3.73       |
|            |              |               |                             |                       |                  |
|     1      |     512      |      32       |           2384.78           |        1719.96        |       1.39       |
|     2      |     512      |      32       |           2503.24           |        1830.56        |       1.37       |
|     4      |     512      |      32       |           3658.65           |        2092.56        |       1.75       |
|     8      |     512      |      32       |           6238.79           |        2629.97        |       2.37       |
|     16     |     512      |      32       |          11409.53           |        3706.23        |       3.08       |

#### Perofrmance of GPT-89B

* head_num = 96
* size_per_head = 128
* num_layers = 48
* data_type = FP16
* vocab_size = 51200
* top_p = 0.9
* tensor_parallel_size = 8 with NVLink

| Batch_size | Input Seqlen | Output Seqlen | Megatron <br/> Latency (ms) | FT <br/> Latency (ms) | FT <br/> Speedup |
| :--------: | :----------: | :-----------: | :-------------------------: | :-------------------: | :--------------: |
|     1      |     128      |       8       |           342.86            |        279.44         |       1.23       |
|     2      |     128      |       8       |           369.43            |        280.24         |       1.32       |
|     4      |     128      |       8       |           540.97            |        317.71         |       1.70       |
|     8      |     128      |       8       |           912.46            |        377.50         |       2.42       |
|     12     |     128      |       8       |           1263.39           |        445.46         |       2.84       |
|     16     |     128      |       8       |           1663.39           |        524.80         |       3.17       |
|     20     |     128      |       8       |           1991.16           |        575.83         |       3.46       |
|     32     |     128      |       8       |           3086.85           |        786.57         |       3.92       |
|            |              |               |                             |                       |                  |
|     1      |     512      |      32       |           1244.81           |        887.52         |       1.40       |
|     2      |     512      |      32       |           1357.54           |        940.11         |       1.44       |
|     4      |     512      |      32       |           1970.08           |        1133.22        |       1.74       |
|     8      |     512      |      32       |           3341.66           |        1415.02        |       2.36       |
|     16     |     512      |      32       |           6090.07           |        1952.2         |       3.12       |

#### Performance of GPT-20B

* head_num = 48
* size_per_head = 128
* num_layers = 44
* data_type = FP16
* vocab_size = 51200
* top_p = 0.9

TP means tensor parallelism

| Batch_size | Input Length | Output Length | Latency of <br/> single GPU (ms) | Latency of <br/> 2-way TP (ms) | Latency of <br/> 4-way TP (ms) | Latency of <br/> 8-way TP (ms) |
| :--------: | :----------: | :-----------: | :------------------------------: | :----------------------------: | :----------------------------: | :----------------------------: |
|     1      |      20      |       8       |               225                |              147               |              102               |               89               |
|     2      |      20      |       8       |               225                |              152               |              108               |               94               |
|     4      |      20      |       8       |               228                |              158               |              113               |              100               |
|     8      |      20      |       8       |               239                |              169               |              121               |              107               |
|     16     |      20      |       8       |               268                |              191               |              133               |              113               |
|     32     |      20      |       8       |               331                |              230               |              155               |              127               |
|     64     |      20      |       8       |               452                |              314               |              200               |              169               |
|    128     |      20      |       8       |               726                |              484               |              318               |              256               |
|    256     |      20      |       8       |               1352               |              844               |              533               |              416               |
|            |              |               |                                  |                                |                                |                                |
|     1      |      60      |      20       |               560                |              358               |              248               |              212               |
|     2      |      60      |      20       |               562                |              378               |              262               |              222               |
|     4      |      60      |      20       |               582                |              393               |              274               |              236               |
|     8      |      60      |      20       |               635                |              429               |              299               |              247               |
|     16     |      60      |      20       |               748                |              510               |              345               |              272               |
|     32     |      60      |      20       |               933                |              620               |              418               |              325               |
|     64     |      60      |      20       |               1352               |              887               |              574               |              454               |
|    128     |      60      |      20       |               2218               |              1384              |              928               |              699               |
|    256     |      60      |      20       |               4141               |              2424              |              1574              |              1152              |
|            |              |               |                                  |                                |                                |                                |
|     1      |     128      |      20       |               566                |              362               |              254               |              217               |
|     2      |     128      |      20       |               580                |              385               |              267               |              227               |
|     4      |     128      |      20       |               629                |              421               |              290               |              244               |
|     8      |     128      |      20       |               740                |              487               |              333               |              267               |
|     16     |     128      |      20       |               931                |              618               |              405               |              312               |
|     32     |     128      |      20       |               1335               |              862               |              547               |              418               |
|     64     |     128      |      20       |               2157               |              1379              |              832               |              634               |
|    128     |     128      |      20       |               3830               |              2365              |              1439              |              1072              |
|    256     |     128      |      20       |               OOM                |              4414              |              2639              |              1943              |
|            |              |               |                                  |                                |                                |                                |
|     1      |      80      |      200      |               5609               |              3532              |              2438              |              2053              |
|     2      |      80      |      200      |               5588               |              3682              |              2544              |              2095              |
|     4      |      80      |      200      |               5661               |              3797              |              2646              |              2206              |
|     8      |      80      |      200      |               5838               |              3984              |              2741              |              2268              |
|     16     |      80      |      200      |               6167               |              4356              |              2964              |              2307              |
|     32     |      80      |      200      |               6864               |              4817              |              3233              |              2566              |
|     64     |      80      |      200      |               8290               |              6003              |              3815              |              3173              |
|    128     |      80      |      200      |               OOM                |              7884              |              5239              |              4303              |
|    256     |      80      |      200      |               OOM                |             12007              |              7603              |              6087              |
|            |              |               |                                  |                                |                                |                                |
|     1      |     200      |      200      |               5648               |              3544              |              2481              |              2080              |
|     2      |     200      |      200      |               5686               |              3739              |              2597              |              2131              |
|     4      |     200      |      200      |               5830               |              3876              |              2719              |              2249              |
|     8      |     200      |      200      |               6146               |              4123              |              2851              |              2338              |
|     16     |     200      |      200      |               6815               |              4672              |              3152              |              2475              |
|     32     |     200      |      200      |               8111               |              5488              |              3634              |              2811              |
|     64     |     200      |      200      |              10766               |              7256              |              4536              |              3621              |
|    128     |     200      |      200      |               OOM                |             10538              |              6618              |              5229              |
|    256     |     200      |      200      |               OOM                |              OOM               |             10447              |              7895              |

#### Performance of GPT-6.7B

* head_num = 32
* size_per_head = 128
* num_layers = 32
* data_type = FP16
* vocab_size = 51200
* top_p = 0.9
* tensor_para_size = 1

| Batch_size | Input Seqlen | Output Seqlen | FT <br/> Latency (ms) | Memory Usage (GB) |
| :--------: | :----------: | :-----------: | :-------------------: | :---------------: |
|     1      |     128      |       8       |         98.29         |       15.55       |
|     2      |     128      |       8       |        106.74         |       15.66       |
|     4      |     128      |       8       |        123.47         |       15.87       |
|     8      |     128      |       8       |        162.51         |       16.31       |
|     16     |     128      |       8       |        241.16         |       17.19       |
|     32     |     128      |       8       |        400.35         |       18.84       |
|     64     |     128      |       8       |        718.07         |       22.17       |
|            |              |               |                       |                   |
|     1      |     512      |      32       |        384.70         |       15.96       |
|     2      |     512      |      32       |        425.88         |       16.30       |
|     4      |     512      |      32       |        514.93         |       16.99       |
|     8      |     512      |      32       |        699.62         |       18.72       |
|     16     |     512      |      32       |        1068.88        |       22.17       |
|     32     |     512      |      32       |        1814.03        |       28.73       |
|     64     |     512      |      32       |        3306.41        |       41.84       |

#### Performance of GPT-1.3B

* head_num = 32
* size_per_head = 64
* num_layers = 24
* data_type = FP16
* vocab_size = 51200
* top_p = 0.9
* tensor_para_size = 1

| Batch_size | Input Seqlen | Output Seqlen | FT <br/> Latency (ms) | Memory Usage (GB) |
| :--------: | :----------: | :-----------: | :-------------------: | :---------------: |
|     1      |     128      |       8       |         36.76         |       8.67        |
|     2      |     128      |       8       |         39.16         |       5.39        |
|     4      |     128      |       8       |         43.32         |       5.49        |
|     8      |     128      |       8       |         52.92         |       5.66        |
|     16     |     128      |       8       |         74.44         |       6.00        |
|     32     |     128      |       8       |        116.74         |       6.66        |
|     64     |     128      |       8       |        201.71         |       7.97        |
|            |              |               |                       |                   |
|     1      |     512      |      32       |        135.85         |       5.58        |
|     2      |     512      |      32       |        150.57         |       5.71        |
|     4      |     512      |      32       |        178.25         |       5.97        |
|     8      |     512      |      32       |        232.11         |       6.64        |
|     16     |     512      |      32       |        345.96         |       7.98        |
|     32     |     512      |      32       |        578.52         |       10.52       |
|     64     |     512      |      32       |        1036.21        |       15.61       |

#### Performance of GPT-350M

* head_num = 16
* size_per_head = 64
* num_layers = 24
* data_type = FP16
* vocab_size = 51200
* top_p = 0.9
* tensor_para_size = 1

| Batch_size | Input Seqlen | Output Seqlen | FT <br/> Latency (ms) | Memory Usage (GB) |
| :--------: | :----------: | :-----------: | :-------------------: | :---------------: |
|     1      |     128      |       8       |         25.43         |       3.43        |
|     2      |     128      |       8       |         26.42         |       3.46        |
|     4      |     128      |       8       |         28.00         |       3.51        |
|     8      |     128      |       8       |         32.56         |       3.61        |
|     16     |     128      |       8       |         42.87         |       3.78        |
|     32     |     128      |       8       |         62.61         |       4.13        |
|     64     |     128      |       8       |        104.51         |       4.81        |
|            |              |               |                       |                   |
|     1      |     512      |      32       |         92.01         |       3.57        |
|     2      |     512      |      32       |         97.87         |       3.65        |
|     4      |     512      |      32       |        110.70         |       3.78        |
|     8      |     512      |      32       |        136.45         |       4.12        |
|     16     |     512      |      32       |        189.91         |       4.80        |
|     32     |     512      |      32       |        296.15         |       6.09        |
|     64     |     512      |      32       |        529.18         |       8.67        |
