# FasterTransformer DeBERTa

The FasterTransformer DeBERTa implements the huggingface DeBERTa-V2 model (https://huggingface.co/docs/transformers/model_doc/deberta-v2).

## Table Of Contents

- [FasterTransformer DeBERTa](#fastertransformer-deberta)
  - [Table Of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Supported features](#supported-features)
    - [Optimization](#optimization)
  - [Setup](#setup)
    - [Requirements](#requirements)
    - [Build FasterTransformer](#build-fastertransformer)
      - [Prepare](#prepare)
      - [Build the project](#build-the-project)
  - [How to use](#how-to-use)

## Introduction

This document describes what FasterTransformer provides for the `DeBERTa` model, explaining the workflow and optimization. We also provide a guide to help users to run the `DeBERTa` model on FasterTransformer.

### Supported features

* Checkpoint loading
  * Huggingface
* Data type
  * FP32
  * FP16
  * BF16
* Feature
  * Multi-GPU multi-node inference (implemented, not verified yet)
  * Disentangled attention mechanism support with fused kernels
* Frameworks
  * PyTorch
  * TensorFlow

### Optimization

We implemented an efficient algorithm to perform the calculation of disentangled attention matrices for DeBERTa-variant types of Transformers.

Unlike [BERT](https://arxiv.org/abs/1810.04805) where each word is represented by one vector that sums the content embedding and position embedding, [DeBERTa](https://arxiv.org/abs/2006.03654) design first proposed the concept of disentangled attention, which uses two vectors to encode content and position respectively and forms attention weights by summing disentangled matrices. Performance gap has been identified between the new attention scheme and the original self-attention, mainly due to extra indexing and gather opertaions. Major optimizations implemented in this plugin includes: (i) fusion of gather and pointwise operataions (ii) utilizing the pattern of relative position matrix and shortcuting out-of-boundary index calculation (iii) parallel index calculation. 

The disentangled attention support is primarily intended to be used together with DeBERTa network (with HuggingFace [DeBERTa](https://huggingface.co/docs/transformers/model_doc/deberta) and [DeBERTa-V2](https://huggingface.co/docs/transformers/model_doc/deberta-v2) implementation), but also applies to generic architectures that adopt disentangeld attention.

## Setup

The following section lists the requirements to use FasterTransformer.

### Requirements

- CMake >= 3.13 for PyTorch
- CUDA 11.0 or newer version
- NCCL 2.10 or newer version
- Python: Only verify on Python 3.
- TensorFlow 2.0: Verify on 2.10.0.

Ensure you have the following components:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) and NGC container are recommended
- [NVIDIA Pascal](https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/) or [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) or [Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/) based GPU 

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:

- [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
- [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
- [Running PyTorch](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)

For those unable to use the NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

### Build FasterTransformer

#### Prepare

You can choose the pytorch version and python version you want. Here, we suggest image `nvcr.io/nvidia/pytorch:22.09-py3`, which contains the PyTorch 1.13.0 and python 3.8.

    ```bash
    nvidia-docker run -ti --shm-size 5g --rm nvcr.io/nvidia/pytorch:22.09-py3 bash
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

1. build with TensorFlow

    ```bash
    docker build -f docker/Dockerfile.tf2 --build-arg SM=XX --tag=ft-tf2 .
    docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm ft-tf2:latest
    
    mkdir build && cd build
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_MULTI_GPU=ON -DBUILD_TF2=ON -DTF_PATH=/usr/local/lib/python3.8/dist-packages/tensorflow/ ..
    make -j12
    ```
    This will build the TensorFlow custom class. Please make sure that the `TensorFlow >= 2.0`.

2. build with PyTorch

    ```bash
    docker build -f docker/Dockerfile.torch --build-arg SM=XX --tag=ft-pytorch .

    mkdir build && cd build
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
    make -j12
    ```

    This will build the TorchScript custom class. Please make sure that the `PyTorch >= 1.5.0`.

## How to use
Please refer to [DeBERTa examples](../examples/tensorflow/deberta/) for demo of FT DeBERTa usage. Meanwhile, task specific examples are under development.