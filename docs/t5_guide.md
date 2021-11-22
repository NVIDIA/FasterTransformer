# FasterTransformer T5

The FasterTransformer T5 implements the huggingface t5 model (https://huggingface.co/t5-base).

## Table Of Contents

- [FasterTransformer T5](#fastertransformer-t5)
  - [Table Of Contents](#table-of-contents)
  - [Introduction](#introduction)
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
  - [Performance](#performance)
    - [End-to-end translation performance on PyTorch](#end-to-end-translation-performance-on-pytorch)

## Introduction

This document describes what FasterTransformer provides for the `T5` model, explaining the workflow and optimization. We also provide a guide to help users to run the `T5` model on FasterTransformer. Finally, we provide benchmark to demonstrate the speed of FasterTransformer on `T5`. 

## Model architecture

### Workflow
<!-- 
Fig 1 demonstrates the workflow of FasterTransformer Decoder and Decoding. They receive some results from encoder as the inputs of CrossAttention, using the start ids or the generated ids of previous step as the inputs of Decoding and generates the respective output ids as response.

<div align=center><img  width='600' src ="images/decoding/decoding.png "/></div>
<div align=center>Fig. 1 Flowchart of Decoding and GPT.</div>

The following examples demonstrating how to run multi-GPU and multi-node GPT model.
1. `examples/cpp/decoding.cc`: An example to run the Decoding with random weights and inputs in C++.
2. `examples/tensorflow/decoding/translate_example.py`: An example to run the end-to-end translation task with FasterTransformer Decoder/Decoding in TensorFlow. We also use the FasterTransformer encoder op in this example.  -->

The source codes are put in `src/fastertransformer/models/t5`.

### Optimization

1.	Kernel optimization: First, since the sequence length of query in `SelfAttention` and `CrossAttention` is always 1, we use customed fused multi-head attention kernel to optimize. Second, we fuse many small operations into one kernel. For example, `AddBiasResidualLayerNorm` combines the adding bias, adding residual of previous block and the computation of layer normalization into 1 kernel. Third, we optimize top k operation and sampling to accelerate the beam search and sampling. Finally, to prevent from recomputing the previous keys and values, we allocate a buffer to store them at each step. Although it takes some additional memory usage, we can save the cost of recomputing, allocating buffer at each step, and the cost of concatenation.

## Setup

The following section lists the requirements to use FasterTransformer.

### Requirements

- CMake >= 3.8 for Tensorflow, CMake >= 3.13 for PyTorch
- CUDA 10.1 or newer version
- Python 3 is recommended because some features are not supported in python 2
- Tensorflow 1.13 or 1.14 or 1.15
- PyTorch >= 1.4.0

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

You can choose the tensorflow version and python version you want. Here, we suggest image `nvcr.io/nvidia/pytorch:21.02-py3`, which contains the PyTorch 1.8.0 and python 3.8.

    ```bash
    nvidia-docker run -ti --rm nvcr.io/nvidia/pytorch:21.02-py3 bash
    git clone https://github.com/NVIDIA/FasterTransformer.git
    mkdir -p FasterTransformer/build
    cd FasterTransformer/build
    git submodule init && git submodule update
    ```

#### Build the project

* Note: the `xx` of `-DSM=xx` in following scripts means the compute capability of your GPU. For example, 60 (P40) or 61 (P4) or 70 (V100) or 75(T4) or 80 (A100).  Default setting is including 70, 75, 80 and 86.

1. build with PyTorch

    ```bash
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_GPT=ON ..
    make
    ```

    This will build the TorchScript custom class. Please make sure that the `PyTorch >= 1.5.0`.

## How to use

### Translation process

1. Run FasterTransformer T5 on PyTorch

    Please install transformers first before running the demos by
    ```bash
    pip install transformers
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
    - `data_type` = fp32
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

## Performance

Hardware settings: 
* CPU: Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz
* V100 (with mclk 877MHz, pclk 1380MHz) with Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz (dgx-1 server)

To run the following benchmark, we need to install the unix computing tool "bc" by

```bash
apt-get install bc
```

### End-to-end translation performance on PyTorch

We demonstrate the throughput of huggingface and FT for end-to-end translation on V100. We also skip the BLEU score because the score of PyTorch, FT Decoder and FT Decoding are close.

Although the bleu scores of all methods are close, the results may be little different, and the number of generated tokens may be also different. So, we use throughput but not latency to show the performance in this benchmark.

#### T5-base

* T5-base on FP32 with beamsearch

| Batch Size | beamsearch | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
|:----------:|:--------------:|:---------:|:----------------------------------------:|:----------------------------------------:|:-------------------------:|
|   1 |    4 | FP32 |    28 |   257 |  9.17 |
|   1 |   32 | FP32 |    20 |   175 |  8.75 |
|   8 |    4 | FP32 |   105 |   953 |  9.07 |
|   8 |   32 | FP32 |    50 |   196 |  3.92 |
|  32 |    4 | FP32 |   247 |  1400 |  5.66 |
|  32 |   32 | FP32 |     0 |   OOM |    x  |
| 128 |    4 | FP32 |     0 |  1448 |    x  |
| 128 |   32 | FP32 |   OOM |   OOM |    x  |

* T5-base on FP16 with beam search

| Batch Size | beamsearch | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
|:----------:|:--------------:|:---------:|:----------------------------------------:|:----------------------------------------:|:-------------------------:|
|   1 |    4 | FP16 |    21 |   359 | 17.09 |
|   1 |   32 | FP16 |    14 |   250 | 17.85 |
|   8 |    4 | FP16 |    76 |  1418 | 18.65 |
|   8 |   32 | FP16 |    40 |   526 | 13.15 |
|  32 |    4 | FP16 |   221 |  2962 | 13.40 |
|  32 |   32 | FP16 |   OOM |   684 |    x  |
| 128 |    4 | FP16 |   345 |  4079 | 11.82 |
| 128 |   32 | FP16 |   OOM |   OOM |    x  |

* T5-base on FP32 with sampling

| Batch Size | sampling | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
|:----------:|:--------------:|:---------:|:----------------------------------------:|:----------------------------------------:|:-------------------------:|
|   1 |    4 | FP32 |    26 |   226 |  8.69 |
|   1 |  0.5 | FP32 |    27 |   219 |  8.11 |
|   8 |    4 | FP32 |   115 |  1153 | 10.02 |
|   8 |  0.5 | FP32 |   130 |  1075 |  8.26 |
|  32 |    4 | FP32 |   327 |  3021 |  9.23 |
|  32 |  0.5 | FP32 |   297 |  2773 |  9.33 |
| 128 |    4 | FP32 |  1162 |  4184 |  3.60 |
| 128 |  0.5 | FP32 |   797 |  3975 |  4.98 |

* T5-base on FP16 with sampling

| Batch Size | sampling | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
|:----------:|:--------------:|:---------:|:----------------------------------------:|:----------------------------------------:|:-------------------------:|
|   1 |    4 | FP16 |    19 |   364 | 19.15 |
|   1 |  0.5 | FP16 |    20 |   353 | 17.65 |
|   8 |    4 | FP16 |    83 |  1733 | 20.87 |
|   8 |  0.5 | FP16 |    98 |  1599 | 16.31 |
|  32 |    4 | FP16 |   337 |  4517 | 13.40 |
|  32 |  0.5 | FP16 |   301 |  4207 | 13.97 |
| 128 |    4 | FP16 |   956 |  8519 |  8.91 |
| 128 |  0.5 | FP16 |   723 |  7997 | 11.06 |

#### T5-small

* T5-small on FP32 with beamsearch

| Batch Size | beamsearch | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
|:----------:|:--------------:|:---------:|:----------------------------------------:|:----------------------------------------:|:-------------------------:|
|   1 |    4 | FP32 |    51 |   626 | 12.27 |
|   1 |   32 | FP32 |    30 |   413 | 13.76 |
|   8 |    4 | FP32 |   192 |  2462 | 12.82 |
|   8 |   32 | FP32 |    72 |   563 |  7.81 |
|  32 |    4 | FP32 |   383 |  4316 | 11.26 |
|  32 |   32 | FP32 |   104 |   668 |  6.42 |
| 128 |    4 | FP32 |   554 |  4747 |  8.56 |
| 128 |   32 | FP32 |   OOM |   OOM |   x   |

* T5-small on FP16 with beamsearch

| Batch Size | beamsearch | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
|:----------:|:--------------:|:---------:|:----------------------------------------:|:----------------------------------------:|:-------------------------:|
|   1 |    4 | FP16 |    35 |   776 | 22.17 |
|   1 |   32 | FP16 |    28 |   553 | 19.75 |
|   8 |    4 | FP16 |   163 |  3467 | 21.26 |
|   8 |   32 | FP16 |    71 |  1140 | 16.05 |
|  32 |    4 | FP16 |   365 |  7154 | 19.60 |
|  32 |   32 | FP16 |   108 |  1359 | 12.58 |
| 128 |    4 | FP16 |   524 | 11285 | 21.53 |
| 128 |   32 | FP16 |     0 |  942※|  0.00 |

※: Out of memory on single GPU. Run by 2 ways tensor parallel.

* T5-small on FP32 with sampling

| Batch Size | sampling | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
|:----------:|:--------------:|:---------:|:----------------------------------------:|:----------------------------------------:|:-------------------------:|
|   1 |    4 | FP32 |    60 |   577 |  9.61 |
|   1 |  0.5 | FP32 |    57 |   524 |  9.19 |
|   8 |    4 | FP32 |   243 |  2821 | 11.60 |
|   8 |  0.5 | FP32 |   221 |  2345 | 10.61 |
|  32 |    4 | FP32 |   765 |  7865 | 10.28 |
|  32 |  0.5 | FP32 |   634 |  6365 | 10.03 |
| 128 |    4 | FP32 |  2238 | 12134 |  5.42 |
| 128 |  0.5 | FP32 |  1611 | 10439 |  6.47 |

* T5-small on FP16 with sampling

| Batch Size | sampling | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |
|:----------:|:--------------:|:---------:|:----------------------------------------:|:----------------------------------------:|:-------------------------:|
|   1 |    4 | FP16 |    46 |   934 | 20.30 |
|   1 |  0.5 | FP16 |    42 |   862 | 20.52 |
|   8 |    4 | FP16 |   194 |  3510 | 18.09 |
|   8 |  0.5 | FP16 |   182 |  3235 | 17.77 |
|  32 |    4 | FP16 |   592 | 10692 | 18.06 |
|  32 |  0.5 | FP16 |   553 |  9008 | 16.28 |
| 128 |    4 | FP16 |  1921 | 19446 | 10.12 |
| 128 |  0.5 | FP16 |  1307 | 16810 | 12.86 |
