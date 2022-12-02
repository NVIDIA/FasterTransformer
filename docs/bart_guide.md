# FasterTransformer BART

The FasterTransformer BART implements the huggingface BART model (https://huggingface.co/docs/transformers/model_doc/bart).

## Table Of Contents

- [FasterTransformer BART](#fastertransformer-bart)
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

This document describes what FasterTransformer provides for the `BART` model, explaining the workflow and optimization. We also provide a guide to help users to run the `BART` model on FasterTransformer. Finally, we provide benchmark to demonstrate the speed of FasterTransformer on `BART`. 

### Supported features

* Checkpoint loading
  * Huggingface
* Data type
  * FP32
  * FP16
  * BF16
* Feature
  * Multi-GPU multi-node inference
  * Dynamic random seed
  * Stop tokens
  * Beam search and sampling are both supported
* Frameworks
  * PyTorch

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

Recommend use nvcr image like `nvcr.io/nvidia/pytorch:22.09-py3`.

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

1. build with PyTorch

    ```bash
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
    make -j12
    ```

    This will build the TorchScript custom class. Please make sure that the `PyTorch >= 1.5.0`.

## How to use
Please refer to [BART Jupyter notebook](../examples/pytorch/bart/bart.ipynb) for demo of FT BART usage. Meanwhile, task specific examples are under development.