# GPT

## Table Of Contents

- [GPT](#gpt)
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
    - [Prepare](#prepare-1)
    - [Run GPT](#run-gpt)
    - [gpt with triton backend](#gpt-with-triton-backend)
  - [Performance](#performance)
    - [Large model inference with model parallel](#large-model-inference-with-model-parallel)
      - [Performance of GPT-175B](#performance-of-gpt-175b)
      - [Perofrmance of GPT-89B](#perofrmance-of-gpt-89b)
      - [Performance of GPT-1.3B](#performance-of-gpt-13b)
      - [Performance of GPT-6.7B](#performance-of-gpt-67b)
      - [Performance of GPT-350M](#performance-of-gpt-350m)
  - [TODO](#todo)

## Introduction

This document describes what FasterTransformer provides for the GPT model, explaining the workflow and optimization. We also provide a guide to help users to run the GPT model on FasterTransformer. Finally, we provide benchmark to demonstrate the speed of FasterTransformer on GPT. 

GPT is a variant of Decoding model, which does not have the encoder module, cross multi-head attention, and uses GeLU as the activation. In 2020, OpenAI shows that using very giant model and lots of training data can significantly improve the capacity of GPT model in [their paper](https://arxiv.org/abs/2005.14165). However, it is impossible to put such model into a single GPU. For example, the largest model, GPT-3, has 175 billion parameters, which takes about 350 GBs under half data type. Therefore, multi-gpus, even multi-nodes, is necessary. To solve the bottleneck of latency and memory due to the model size, FasterTransformer provides kernels with high efficiency, optimized memory usage, and model parallelism on multiple frameworks. 
 
## Model architecture

### Workflow

<div align=center><img width=600 src ="images/gpt/gpt.png "/></div>
<div align=center> Fig 1. Workflow of GPT model.</div>

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

* Arguments:
  1. Maximum batch size
  2. Maximum sequence length
  3. Maximum input sequence length
  4. beam width for beam search. If setting b to be 1, then we don’t use beam search but use sampling.
  5. Head number
  6. Size per head
  7. Intermediate size. The inter size of feed forward network. It is often set to 4 * head_num * size_per_head.
  8. Number of decoder layers
  9. Vocab size
  10. Start id of the vocabulary
  11. End id of the vocabulary
  12. Diversity rate of beam search. A hyper hyper-parameter for [simple diverse decoding](https://arxiv.org/pdf/1611.08562.pdf).
  13. top_k value for top k sampling.
  14. top_p value for top p sampling
  15. Random seed for sampling
  16. Temperature for logit. Setting to be 1.0 if you don’t want to apply the temperature.
  17. Length penalty for logit. Setting to be 1.0 if you don’t want to apply the length penalty.
  18. Repetition penalty for logit. Setting to be 1.0 if you don’t want to apply the repetition penalty.
  19. Size of tensor parallelism. Must be able to divide the head number.
  20. Rank of tensor parallelism.
  21. Comm of tensor parallelism.
  22. Size of pipeline parallelism. Must be able to devide the number of decoder layers.
  23. Rank of pipeline parallelism.
  24. Comm of pipeline parallelism.
  25. CUDA stream.
  26. Pointer of cuBLAS wrapper, which is declared in `src/fastertransformer/utils/cublasMMWrapper.h`.
  27. Pointer of memory allocator, which is declared in `src/fastertransformer/utils/allocator.h`
  28. “is_free_buffer_after_forward” flag. If setting to be true, FasterTransformer will allocate buffer before forward, and free buffer after forward. If the memory is controlled by memory pool and the cost of allocating/releasing memory is small, setting the flag to be true can save some memory.
  29. Pointer of CUDA device properties, which is used to get the properties of hardware like size of shared memory.
* Input tensors:
  1. Input ids (context). The shape is \[ request batch size * beam width, request maximum input length \].
  2. Input lengths. The shape is \[ request batch size * beam width \].
  3. Maximum output sequence length. An integer to describe the largest number of tokens you hope for results. Note that it includes the input ids.
* Output tensors:
  1. Output ids. The shape is \[maximum output sequence length, batch size, beam width \].
  2. Parent ids. It is used to find the best path in beam search. It is deprecated now. 
  3. Sequence lengths. The shape is \[batch size * beam width\]. It records the final sequence lengths of all sentences.
  4. Log probability for sampling. The shape is \[requested token number, batch size, beam \]. It records the log probability of logits at each step.  

Although there are many arguments, most of them are fixed. For example, arguments 5 ~ 11 are model hyper-parameters and fixed after we determine the model hyper-parameters. Arguments 19 ~ 27 and 29 are some settings about CUDA and NCCL, and progress are fixed.

We also provide the module `Gpt` in `src/fastertransformer/models/gpt/Gpt.cc`, which is a GPT model without model parallelism. It does not need the arguments 20 ~ 24, and other are same.

### Optimization

1.	Kernel optimization: many kernels are based on the kernels of decoder and decoding modules, which are already highly optimized. To prevent from recomputing the previous keys and values, we will allocate a buffer to store them at each step. Although it takes some additional memory usage, we can save the cost of recomputing, allocating buffer at each step, and the cost of concatenation.
2.	Memory optimization: Different to traditional models like BERT, GPT-3 has 175 billion parameters, taking 350 GBs even if we store the model by half precision. Therefore, we must reduce the memory usage for other parts. In FasterTransformer, we will reuse the memory buffer of different decoder layers. Since the number of layers in GPT-3 is 96, we only need 1/96 memory.
3.	Model parallelism: In GPT model, FasterTransormer provides both tensor parallelism and pipeline parallelism. For tensor parallelism, FasterTransformer follows the idea of [Megatron]( https://arxiv.org/pdf/1909.08053.pdf). For both self-attention block and feed forward network block, we split the weights of first matrix multiplication by row and split the weights of the second matrix multiplication by column. By optimization, we can reduce the reduction operation to 2 times for each transformer block. The workflow is demonstrated in Fig 3. For pipeline parallelism, FasterTransformer splits the whole batch of request into multiple micro batches and hide the bubble of communication. FasterTransformer will adjust the micro batch size automatically for different cases. Users can adjust the model parallelism by modifying the `gpt_config.ini` file. We recommend to use tensor parallel intra node, and use pipeline parallel inter node because tensor parallel requires more NCCL communication.
4.	Multiple frameworks: Except the source codes on c, FasterTransformer also provide the TensorFlow op, PyTorch op and Triton backend. Currently, TensorFlow op only supports the single GPU, while PyTorch op and Triton backend support multi-GPU and multi-node. To prevent the additional work of splitting model for model parallelism, FasterTransformer also provides a tool to split and convert the model of Megatron to binary files, then FasterTransformer can load the model in binary directly.

## Setup

The following guide demonstrates how to run the examples of c++, PyTorch and Triton backend.

### Requirements

- CMake >= 3.8 for Tensorflow, CMake >= 3.13 for PyTorch
- CUDA 10.1 or newer version
- Python 3 is recommended because some features are not supported in python 2
- Tensorflow 1.13 or 1.14 or 1.15
- PyTorch >= 1.5.0

Recommend use nvcr image like `nvcr.io/nvidia/tensorflow:21.02-tf1-py3` or `nvcr.io/nvidia/pytorch:21.02-py3`.

### Build the FasterTransformer

#### Prepare

    You can choose the tensorflow version and python version you want. Here, we list some possible images:

    - `nvcr.io/nvidia/tensorflow:19.07-py2` contains the TensorFlow 1.14 and python 2.7. 
    - `nvcr.io/nvidia/tensorflow:20.12-tf1-py3` contains the TensorFlow 1.15 and python 3.8. 
    - `nvcr.io/nvidia/pytorch:20.03-py3` contains the PyTorch 1.5.0 and python 3.6
    - `nvcr.io/nvidia/pytorch:20.07-py3` contains the PyTorch 1.6.0 and python 3.6
    - `nvcr.io/nvidia/pytorch:20.12-py3` contains the PyTorch 1.8.0 and python 3.8

    To achieve best performance, we recommand to use the latest image. For example, running image `nvcr.io/nvidia/tensorflow:20.12-tf1-py3` by 

    ```bash
    nvidia-docker run -ti --rm nvcr.io/nvidia/tensorflow:20.12-tf1-py3 bash
    git clone https://github.com/NVIDIA/FasterTransformer.git
    mkdir -p FasterTransformer/build
    cd FasterTransformer/build
    git submodule init && git submodule update
    ```

#### Build the project

* Note: the `xx` of `-DSM=xx` in following scripts means the compute capability of your GPU. For example, 60 (P40) or 61 (P4) or 70 (V100) or 75(T4) or 80 (A100).  Default setting is including 70, 75, 80 and 86.

1. build with C++

    ```bash
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_GPT=ON ..
    make
    ```

2. build with TensorFlow 

    Uses need to set the path of TensorFlow. For example, if we use `nvcr.io/nvidia/tensorflow:20.12-tf1-py3`, then

    ```bash
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_TF=ON -DTF_PATH=/usr/local/lib/python3.8/dist-packages/tensorflow_core/ -DBUILD_GPT=ON ..
    make 
    ```

3. build with PyTorch

    ```bash
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_GPT=ON ..
    make
    ```

    This will build the TorchScript custom class. Please make sure that the `PyTorch >= 1.5.0`.

## How to use

### Prepare

* Install required tools

```bash
pip install -r ../examples/pytorch/requirement.txt
```

To run the GPT on c, users need to convert the checkpoint of TensorFlow or PyTorch to binary files, and then load by FasterTransformer c api. Unfortunately, there is no published large model. So, users are only able to verify the correctness by smaller model. Currently, FasterTransformer provides two kinds of samples. First one is using the checkpoint of [OpenAI GPT-2 model](https://github.com/openai/gpt-2) (which is trained by TensorFlow); Another choice is using the checkpoint of [Megatron](https://github.com/NVIDIA/Megatron-LM) (which is trained by pytorch).

* Download vocab and merge table

They can be used in both OpenAI GPT-2 and Megatron.

```bash
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -P ../models
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -P ../models
```

* Downlaod openai-gpt model and convert

To convert the OpenAI GPT model to binary, FasterTransformer provides a tool `sample/tensorflow/utils/openai_gpt_ckpt_convert.py` to convert the checkpoint. The converter requires the following arguemtns: 

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

* Download megatron model and convert

To convert the Megatron GPT model to binary, FasterTransformer provides a tool `examples/pytorch/utils/megatron_ckpt_convert.py` to convert the checkpoint.

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
mkdir -p ../models/megatron-models/345m
unzip megatron_lm_345m_v0.0.zip -d ../models/megatron-models/345m
python ../examples/pytorch/gpt/utils/megatron_ckpt_convert.py -head_num 16 -i ../models/megatron-models/345m/release/ -o ../models/megatron-models/c-model/345m/ -t_g 1 -i_g 1
python ../examples/pytorch/gpt/utils/megatron_ckpt_convert.py -head_num 16 -i ../models/megatron-models/345m/release/ -o ../models/megatron-models/c-model/345m/ -t_g 1 -i_g 8
```

where `t_g` means the number GPUs of TP during training, and `i_g` means the number of GPUs for TP during inference.

Note that there are different checkpoint version of Megatron. The version of the checkpoint above is 0. If users have trained a model by themselves, the default version of latest Megatron is 3. To convert the checkpoint with version 3, please add `-checkpoint_version 3`.

For model trained by pipeline parallelism, please use new checkpoint converter `megatron_ckpt_convert_2.py`. This converter is only able to convert the newer version of checkpoint.

```bash
python ../examples/pytorch/gpt/utils/megatron_ckpt_convert_2.py -i ../models/megatron-models/345m/release/ -o ../models/megatron-models/c-model/345m/ -i_g 1
```

* How to use `checkpoint_saver_fastertransformer.py` to convert the megatron model. Note that this tool is only available for newer checkpoint. Need to get more details from ADLR team.

```bash
git clone -b checkpoint_util https://gitlab-master.nvidia.com/ADLR/megatron-lm.git
cp ../examples/pytorch/gpt/utils/checkpoint_saver_fastertransformer.py megatron-lm/tools
cd megatron-lm
python tools/checkpoint_util.py --model-type GPT --loader megatron --saver fastertransformer --input ../megatron_new_ckpt/357m-pipeline-2-tensor-2/ --output ../tmp  --target-tensor-parallel-size 1
```

* Download onnx model and convert

To convert the Megatron GPT model to binary, FasterTransformer provides a tool `examples/onnx/multi_gpu_gpt/onnx_ckpt_convert.py` to convert the checkpoint.

```bash
wget https://github.com/onnx/models/raw/master/text/machine_comprehension/gpt-2/model/gpt2-10.onnx
python ../examples/onnx/multi_gpu_gpt/onnx_ckpt_convert.py -i gpt2-10.onnx -o ../models/onnx-models/c-model/124m/ -i_g 1
python ../examples/onnx/multi_gpu_gpt/onnx_ckpt_convert.py -i gpt2-10.onnx -o ../models/onnx-models/c-model/124m/ -i_g 4
```

### Run GPT

1. Run GPT under on C++ with multiple gpu

    1.1 Generate the `gemm_config.in` file.

    ```bash
    ./bin/gpt_gemm <batch_size> <beam_width> <max_input_len> <head_number> <size_per_head> <inter_size> <vocab_size> <is_fp16> <tensor_para_size>
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

    By setting the `is_half` of `gpt_config.ini` to 1, users can run gpt model under fp16.

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
    srun -N2 -n2 docker pull nvcr.io/nvidia/tensorflow:20.07-tf1-py3 

    srun -N2 -n2  nvidia-docker run -itd --rm --privileged --network=host --pid=host --cap-add=IPC_LOCK --device=/dev/infiniband -v $PWD:$PWD -w $PWD --name ft-test nvcr.io/nvidia/tensorflow:20.12-tf1-py3 /bin/bash

    srun -N2 -n2  nvidia-docker exec -i --env SLURM_NTASKS --env SLURM_NODEID --env SLURM_PROCID --env SLURM_STEP_NODELIST --env SLURMD_NODENAME --privileged ft-test bash -c "mkdir /root/.ssh && cp $PWD/ssh/* /root/.ssh && chmod 700 /root/.ssh && chmod 640 /root/.ssh/authorized_keys2 && chmod 400 /root/.ssh/id_rsa && apt-get update && apt-get install ssh -y && mkdir /run/sshd/ && /usr/sbin/sshd -p 11068 && nvidia-smi -lgc 1530"

    nvidia-docker exec -ti ft-test bash 
    cd FasterTransformer/build
    mpirun --allow-run-as-root -np 2 -H prm-dgx-09:1,prm-dgx-10:1 -mca plm_rsh_args "-p 11068" ./bin/multi_gpu_gpt_example
    srun -N2 -n2 docker stop ft-test
    ```

2. Run GPT on PyTorch

    Basically, `gpt_example.py` includes the example how to declare a model, load a ckeckpoint, and forward context inputs and get generated outputs in Pytorch.

    For generating outputs based on context inputs, create a text file including the context inputs (line by line) and set `--sample_file_input` to the text file path. (By default, the script will generate outputs without context inputs.) Set `--sample_file_output` to write the outputs to a file. Use `--fp_16` to run in FP16.

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
    srun -A devtech -J devtech-gpt:gpt -p luna -N1 --mpi=pmix --ntasks-per-node=8 --container-image nvcr.io/nvidia/pytorch:20.12-py3 --container-mounts /lustre/fsw/devtech/hpc-devtech/dahn/FasterTransformer:/workspace/fastertransformer --container-workdir /workspace/fastertransformer --pty bash

    mkdir build && cd build
    cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON .. && make -j12
    ```

    #### Run on singe-node
    * tensor_para_size=8, pipeline_para_size=1

    ```bash
    srun -A devtech -p luna -N1 --mpi=pmix --ntasks-per-node=8 --container-image nvcr.io/nvidia/pytorch:20.12-py3 --container-mounts /lustre/fsw/devtech/hpc-devtech/dahn/FasterTransformer:/workspace/fastertransformer --container-workdir /workspace/fastertransformer/build python ../examples/pytorch/gpt/multi_gpu_gpt_example.py --tensor_para_size=8 --pipeline_para_size=1 --ckpt_path="/workspace/fastertransformer/models/megatron-models/c-model/345m/8-gpu"
    ```

    #### Run on multi-node
    * tensor_para_size=8, pipeline_para_size=2

    ```bash
    srun -A devtech -p luna -N2 --mpi=pmix --ntasks-per-node=8 --container-image nvcr.io/nvidia/pytorch:20.12-py3 --container-mounts /lustre/fsw/devtech/hpc-devtech/dahn/FasterTransformer:/workspace/fastertransformer --container-workdir /workspace/fastertransformer/build python ../examples/pytorch/gpt/multi_gpu_gpt_example.py --tensor_para_size=8 --pipeline_para_size=2 --ckpt_path="/workspace/fastertransformer/models/megatron-models/c-model/345m/8-gpu"
    ```

3. Run GPT on tensorflow

    Note that the tensorflow op only supports single gpu.

    ```bash
    ./bin/gpt_gemm 4 1 32 12 64 3072 50257 1 1
    python ../examples/tensorflow/gpt/gpt_example.py --batch_size=4 \
                                                     --length=32 \
                                                     --top_k=4 \
                                                     --top_p=0.6 \
                                                     --data_type=fp16
    ```

4. Run LAMBADA test

    download data set:

    ```bash
    wget https://github.com/cybertronai/bflm/raw/master/lambada_test.jsonl -P ../models/megatron-models
    bash ../examples/pytorch/gpt/scripts/evaluate_zeroshot_gpt.sh
    ```

### gpt with triton backend

Details are in [transformer_backend](https://github.com/triton-inference-server/fastertransformer_backend)

## Performance

Hardware settings: 
* 8xA100-80GBs (with mclk 1593MHz, pclk 1410MHz) with AMD EPYC 7742 64-Core Processor

### Large model inference with model parallel

We demonstrate the inference time of Megatron and FasterTransformer on Triton, and show the speedup of FasterTransformer compare to Megatron for GPT-175B and GPT-89B. In the experiments of GPT, we updated the following parameters:

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

## TODO

* [ ] Add benchmark for more model size, batch size with different model parallel size.
* [ ] Add benchmark for Pipeline parallel. (how local batch size influences: some benchmarks to show the trade off between micro batches computing and the total batch computing)
* [ ] Add benchmark for With and without recompute
