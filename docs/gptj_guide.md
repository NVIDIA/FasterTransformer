# GPT-J

## Table Of Contents

- [GPT-J](#gpt-j)
  - [Table Of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Supported features](#supported-features)
  - [Setup](#setup)
    - [Requirements](#requirements)
    - [Docker image](#docker-image)
    - [Setup](#setup-1)
    - [Build](#build)
    - [Download the model](#download-the-model)
    - [Run GPT-J](#run-gpt-j)
    - [Compare with reference implementation](#compare-with-reference-implementation)
    - [gpt-j with triton backend](#gpt-j-with-triton-backend)


## Introduction

This document describes the step to run the GPT-J model on FasterTransformer.
GPT-J was developed by EleutherAI and trained on The Pile, a 825GB dataset from curated sources (e.g. Wikipedia, arXiv, GitHub, StackExchange, PubMed, ...).
With 6 billion parameters, GPT-J is one of the largest GPT-like publicly released models as of 2021.

### Supported features

* Checkpoint converter
  * EleutherAI
* Data type
  * FP32
  * FP16
* Feature
  * Multi-GPU multi-node inference
  * Dynamic random seed
  * Stop tokens
  * Bad words list
  * Beam search and sampling are both supported
* Frameworks
  * Triton backend

Optimization in GPT-j are similar to optimization in GPT, describing in the [gpt_guide.md](gpt_guide.md#optimization).

* Arguments:
  1. Maximum batch size (Deprecated, move to input)
  2. Maximum sequence length (Deprecated, move to input)
  3. Maximum input sequence length (Deprecated, move to input)
  4. beam width for beam search. If setting b to be 1, then we don’t use beam search but use sampling. (Deprecated, move to input)
  5. Head number
  6. Size per head
  7. Intermediate size. The inter size of feed forward network. It is often set to 4 * head_num * size_per_head.
  8. Number of decoder layers.
  9. Vocab size.
  10. Rotary embedding for attetnion.
  11. Start id of the vocabulary.
  12. End id of the vocabulary.
  13. Diversity rate of beam search. A hyper hyper-parameter for [simple diverse decoding](https://arxiv.org/pdf/1611.08562.pdf). (Deprecated, move to input)
  14. top_k value for top k sampling. (Deprecated, move to input)
  15. top_p value for top p sampling. (Deprecated, move to input)
  16. Random seed for sampling. (Deprecated, move to input)
  17. Temperature for logit. Setting to be 1.0 if you don’t want to apply the temperature. (Deprecated, move to input)
  18. Length penalty for logit. Setting to be 1.0 if you don’t want to apply the length penalty. (Deprecated, move to input)
  19. Repetition penalty for logit. Setting to be 1.0 if you don’t want to apply the repetition penalty. (Deprecated, move to input)
  20. Tensor Parallel information, which is declared in `src/fastertransformer/utils/nccl_utils.h`.
  21. Pipeline Parallel information, which is declared in `src/fastertransformer/utils/nccl_utils.h`.
  22. CUDA stream.
  23. Pointer of cuBLAS wrapper, which is declared in `src/fastertransformer/utils/cublasMMWrapper.h`.
  24. Pointer of memory allocator, which is declared in `src/fastertransformer/utils/allocator.h`
  25. “is_free_buffer_after_forward” flag. If setting to be true, FasterTransformer will allocate buffer before forward, and free buffer after forward. If the memory is controlled by memory pool and the cost of allocating/releasing memory is small, setting the flag to be true can save some memory.
  26. Pointer of CUDA device properties, which is used to get the properties of hardware like size of shared memory.
  27. Custom all reduction communication for custom all reduction in model parallelism. It is only supported in 8-way tensor parallelism.
  28. Flag of enable custom all reduction or not. 
* Input tensors:
  1. Input ids (context). The shape is \[ request batch size * beam width, request maximum input length \].
  2. Input lengths. The shape is \[ request batch size * beam width \].
  3. Maximum output sequence length. An integer to describe the largest number of tokens you hope for results. Note that it includes the input ids.
  4. Start id in runtime. The shape is \[batch_size\] on cpu, optional. If FT receives this input, FT will replace default start id by it, optional. 
  5. End id in runtime. The shape is \[batch_size\] on cpu, optional. If FT receives this input, FT will replace default end id by it, optional. 
  6. Stop word list. When FT generates words in this list, it will stop the generation. An extension of stop id, optional.  
  7. Bad word list. FT won't generates words in this list, optional.  
  8. top_k value for top k sampling. The shape is \[1\] or \[batch_size, 1\] on cpu, optional.
  9. top_p value for top p sampling. The shape is \[1\] or \[batch_size, 1\] on cpu, optional.
  10. Diversity rate of beam search (beam_search_diversity_rate). A hyper hyper-parameter for [simple diverse decoding](https://arxiv.org/pdf/1611.08562.pdf). [1] or \[batch_size, 1\] on cpu, optional.
  11. Temperature for logit (temperature). The sahpe \[1\] or \[batch_size, 1\] on cpu, optional.
  12. Length penalty for logit (len_penalty). The shape is \[1\] or \[batch_size, 1\] on cpu, optional
  13. Repetition penalty for logit (repetition_penalty). The shape is \[1\] or \[batch_size, 1\] on cpu, optional
  14. Random_seed \[1\] or \[batch_size, 1\] on cpu, optional
  15. Length of prefix soft prompt embedding. This describes how many tokens of soft prompt embedding in each sentence. The shape is \[batch_size\], optional.
  16. Prefix soft prompt embedding. FT will concat them with results of embedding lookup kernel. The shape is \[batch_size, max_prefix_soft_prompt_length, hidden_units\], optional.
* Output tensors:
  1. Output ids. The shape is \[batch size, beam width, maximum output sequence length \].
  2. Sequence lengths. The shape is \[batch size * beam width\]. It records the final sequence lengths of all sentences.
  3. Log probability for sampling. The shape is \[requested token number, batch size, beam \]. It records the log probability of logits at each step. Optional outputs in FP32.

The beam_width value is set by the output shape directly. When the beam_width is larger than 1, FT will use beam search to generate tokens; otherwise, FT will use topk or topp sampling.

## Setup

### Requirements

- CMake >= 3.13 for PyTorch
- CUDA 11.0 or newer version
- NCCL 2.10 or newer version
- Python 3 is recommended because some features are not supported in python 2
- PyTorch: Verify on 1.8.0, >= 1.5.0 should work.

Recommend use nvcr image like `nvcr.io/nvidia/pytorch:21.11-py3`.

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

* The model was built and tested with the use nvcr image `nvcr.io/nvidia/pytorch:21.07-py3`. e.g.

    ```bash
    nvidia-docker run -ti --rm nvcr.io/nvidia/pytorch:21.07-py3 bash
    ```

### Setup

* Get the code and install all dependencies:

    ```bash
    git clone https://github.com/NVIDIA/FasterTransformer.git
    mkdir -p FasterTransformer/build
    cd FasterTransformer/build
    git submodule init && git submodule update
    pip3 install fire jax jaxlib
    ```

### Build

* Note: the `xx` of `-DSM=xx` in following scripts means the compute capability of your GPU. For example, 60 (P40) or 61 (P4) or 70 (V100) or 75(T4) or 80 (A100).  Default setting is including 70, 75, 80 and 86.


    ```bash
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_MULTI_GPU=ON ..
    make -j
    ```

### Download the model

* Download the public model and convert

    ```bash
    wget https://mystic.the-eye.eu/public/AI/GPT-J-6B/step_383500_slim.tar.zstd
    tar -axf step_383500_slim.tar.gz
    python3 ../examples/pytorch/gptj/utils/gptj_ckpt_convert.py --output-dir ../models/j6b_ckpt --ckpt-dir ./step_383500/
    ```

The script accepts the following arguments:
1. `--output-dir` is the path of the base directory where the weight binary files will be saved. If `--output-dir` terminates with `.pt` the script just converts the checkpoint to a Pytorch model file that can be read by the [GPT-J implementation in HuggingFace's transformer](https://github.com/finetuneanon/transformers).
2. `--ckpt-dir` is the path to the extracted checkpoint. If `--ckpt-dir` terminates with `.pt` then the script reads the Pytorch model file instead than the public checkpoint, which is faster.
3. `--n-inference-gpus` number of GPUs used for inference, defaults to 1. The binary model parameters are saved to `${output-dir}/${n-inference-gpus}-gpu/`


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
    E.g., ./bin/gpt_gemm 8 1 32 16 128 16384 50400 1 1
    ```

* Run GPT on C++

    Users can see the details of arguments in `examples/cpp/gptj/gptj_config.ini`. It controls the model path, model size, tensor parallelism size, and some hyper-parameters.

    ```bash
    mpirun -n 1 --allow-run-as-root ./bin/gptj_example
    ```

E.g. by setting the `is_half` of `gpt_config.ini` to 1, users can run gpt model under fp16.

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
