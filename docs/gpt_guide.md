# GPT

## Table Of Contents

- [GPT](#gpt)
  - [Table Of Contents](#table-of-contents)
  - [Model architecture](#model-architecture)
  - [Introduction](#introduction)
  - [Setup](#setup)
    - [Requirements](#requirements)
  - [How to use](#how-to-use)
    - [Prepare](#prepare)
    - [Run GPT](#run-gpt)
    - [gpt with triton backend](#gpt-with-triton-backend)
  - [Performance](#performance)
    - [Perofrmance of GPT-89B](#perofrmance-of-gpt-89b)
    - [Performance of GPT-175B](#performance-of-gpt-175b)

## Model architecture

<div align=center><img  width='616' height='1256' src ="images/gpt_flowchart.png "/></div>
<div align=center>Fig. 1 Flowchart of GPT model.</div>

## Introduction

GPT model is a variant of Decoding model. GPT model does not require the results from encoder and the cross multi-head attention, and use GeLU as the activation. However, OpenAI shows that using very giant model and lots of training data can significantly improve the capacity of GPT model in [their paper](https://arxiv.org/abs/2005.14165). However, it is impossible to put such model into a single GPU. For example, the largest model, GPT-3, has 175 billion parameters, which takes about 350GBs under half data type. Therefore, multi-gpus, even multi-nodes, is necessary.

In FasterTransformer 4.0, we propose the multi-gpu inference library to run GPT-3. FasterTransformer supports `Tensor Parallel` and `Layer Parallel` in the same time and provides the api of cpp, TensorFlow/PyTorch op and triton backend. In cpp and PyTorch op, users can use MPI to run multiple gpus on multiple nodes. For example, using 4 dgx-1 V100 nodes (16 GBs memory per GPU) to run the GPT-3 model. To be convenient on serving, we also provide the triton backend. However, this backend only supports single nodes, multi-gpus currently. For TensorFlow op, FasterTransformer only supports single gpu now.

The arguments, inputs, and outputs of GPT: 

* Arguments:
  1. Maximum batch size (B)
  2. Maximum sequence length (S)
  3. Top k value (K)
  4. Top p value (P)
  5. Head number (H)
  6. Size per head (N)
  7. Number of decoder layers
  8. Start id of the vocabulary
  9. End id of the vocabulary
  10. Vocab size (V)
  11. Tensor parallel size
  12. Layer parallel size
* Inputs:
  1. The table for embedding lookup. The shape is \[ V, H x N \].
  2. The weights of all parameters.
  3. Position encoding table. The shape is \[ S, H x N \].
  4. Inputs contexts. The shape is \[ b, s \], where b <= B, s <= S.
* Outputs:
  1. The output ids. The shape is \[b, S \].

## Setup

### Requirements

- CMake >= 3.8 for Tensorflow, CMake >= 3.13 for PyTorch
- CUDA 10.1 or newer version
- Python 3 is recommended because some features are not supported in python 2
- Tensorflow 1.13 or 1.14 or 1.15
- PyTorch >= 1.5.0

Recommend use nvcr image like `nvcr.io/nvidia/tensorflow:20.12-tf1-py3` or `nvcr.io/nvidia/pytorch:20.12-py3`.

## How to use

### Prepare

* Install required tools

```bash
pip install -r ../requirement.txt
```

To run the GPT on c, users need to convert the checkpoint of TensorFlow or PyTorch to binary files, and then load by FasterTransformer c api. Unfortunately, there is no published large model. So, users are only able to verify the correctness by smaller model. Currently, FasterTransformer provides two kinds of samples. First one is using the checkpoint of [OpenAI GPT-2 model](https://github.com/openai/gpt-2) (which is trained by TensorFlow); Another choice is using the checkpoint of [Megatron](https://github.com/NVIDIA/Megatron-LM) (which is trained by pytorch).

* Download vocab and merge table

They can be used in both OpenAI GPT-2 and Megatron.

```bash
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -P models
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -P models
```

* Downlaod openai-gpt model and convert

To convert the OpenAI GPT model to binary, FasterTransformer provides a tool `sample/tensorflow/utils/openai_gpt_ckpt_convert.py` to convert the checkpoint.

```bash
python tensorflow/utils/download_gpt2_model.py <model_name>
e.g. python tensorflow/utils/download_gpt2_model.py 124M
python ../sample/tensorflow/utils/openai_gpt_ckpt_convert.py -o models/openai-gpt-models/c-model/124m/ -i models/124M/model.ckpt -g 1 # convert 124M model with 1 TP mode
python ../sample/tensorflow/utils/openai_gpt_ckpt_convert.py -o models/openai-gpt-models/c-model/124m/ -i models/124M/model.ckpt -g 4 # convert 124M model with 4 TP mode
```

In the repo of OpenAI, they provide many models, including `124M`, `355M`, `774M` and `1558M`

* Download megatron model and convert

To convert the Megatron GPT model to binary, FasterTransformer provides a tool `sample/pytorch/utils/megatron_ckpt_convert.py` to convert the checkpoint.

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
mkdir -p models/megatron-models/345m
unzip megatron_lm_345m_v0.0.zip -d models/megatron-models/345m
git clone https://github.com/NVIDIA/Megatron-LM.git
python ../sample/pytorch/utils/megatron_ckpt_convert.py -head_num 16 -i ./models/megatron-models/345m/release/ -o ./models/megatron-models/c-model/345m/ -t_g 1 -i_g 1
python ../sample/pytorch/utils/megatron_ckpt_convert.py -head_num 16 -i ./models/megatron-models/345m/release/ -o ./models/megatron-models/c-model/345m/ -t_g 1 -i_g 8
```

where `t_g` means the number GPUs of TP during training, and `i_g` means the number of GPUs for TP during inference.

Note that there are different checkpoint version of Megatron. The version of the checkpoint above is 0. If users have trained a model by themselves, the default version of latest Megatron is 3. To convert the checkpoint with version 3, please add `-checkpoint_version 3`.

### Run GPT

1. Run GPT under on C++ with multiple gpu

    1.1 Generate the `decoding_gemm_config.in` file.

    ```bash
    ./bin/gpt_gemm <local_batch_size> <context_local_batch_size> <head_number> <size_per_head> <vocab_size> <start_len> <tensor_para_size> <is_fp16>
    E.g., ./bin/gpt_gemm 8 8 12 64 50257 32 1 1
    ```

    Here, `local_batch_size` can be set as `batch_size` if users do not use the layer parallelism. If users use layer parallelism, we recommand to set `local_batch_size` to be smaller than `batch_size` to hide the bubble. But this requires larger `batch_size`. `context_local_batch_size` is used for computing the k/v cache of input. Similar to `local_batch_size`, users can use `batch_size` directly if you don't use layer parallelism, and setting to be smaller than `batch_size` when you use layer parallelism. 

    1.2 Run GPT on C++

    Users can see the details of arguments in `sample/cpp/gpt_config.ini`. It controls the model path, model size, tensor parallelism size, and some hyper-parameters.

    ```bash
    ./bin/gpt_sample
    ```

    then use following script to convert the token ids to sentence.

    ```bash
    python ../sample/pytorch/utils/convert_gpt_token.py --vocab_file=./models/gpt2-vocab.json  --bpe_file=./models/gpt2-merges.txt
    ```

    By setting the `is_half` of `gpt_config.ini` to 1, users can run gpt model under fp16.

    1.3 Run with tensor parallelism (TP), layer parallelism (LP) and pipeline parallelism (PP)

    Users can use `tensor_para_size` and `layer_para_size` in `gpt_config.ini` to control the size of model parallel. Besides, in the layer parallelism, we can use pipeline parallelism to reduce the bubbles. We can set the `layer_para_batch_size` to determine the real batch size for each forward. For example, if the total batch size is 4, and layer_para_batch_size is 1, then we will split the total batch into 4 parts, and each time we only use 1 batch size. Users can set them in the `gpt_config.ini`.

    Note that we split the definition of LP and PP here, but we often combine them to hide the cost of bubble.

    ```bash
    mpirun -n 8 ./bin/gpt_sample
    python ../sample/pytorch/utils/convert_gpt_token.py --vocab_file=./models/gpt2-vocab.json  --bpe_file=./models/gpt2-merges.txt
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
    mpirun --allow-run-as-root -np 2 -H prm-dgx-09:1,prm-dgx-10:1 -mca plm_rsh_args "-p 11068" ./bin/gpt_sample
    srun -N2 -n2 docker stop ft-test
    ```

2. Run GPT on PyTorch

    Basically, `gpt_sample.py` includes the example how to declare a model, load a ckeckpoint, and forward context inputs and get generated outputs in Pytorch.

    For generating outputs based on context inputs, create a text file including the context inputs (line by line) and set `--sample_file_input` to the text file path. (By default, the script will generate outputs without context inputs.) Set `--sample_file_output` to write the outputs to a file. Use `--fp_16` to run in FP16.

    Run with `-h` to see more settings.
    ```bash
    python ./pytorch/gpt_sample.py -h
    ```

    2.1 Run GPT with TP and PP on single node (NVIDIA DGX A100)
    ```bash
    # No parallelism (tensor_para_size=1, layer_para_size=1)
    mpirun -n 1 --allow-run-as-root python ./pytorch/gpt_sample.py

    # TP (tensor_para_size=8, layer_para_size=1)
    mpirun -n 8 --allow-run-as-root python ./pytorch/gpt_sample.py --tensor_para_size=8 --layer_para_size=1 --ckpt_path="/workspace/fastertransformer/models/megatron-models/c-model/345m/8-gpu"

    # LP (tensor_para_size=1, layer_para_size=8)
    mpirun -n 8 --allow-run-as-root python ./pytorch/gpt_sample.py --tensor_para_size=1 --layer_para_size=8 --ckpt_path="/workspace/fastertransformer/models/megatron-models/c-model/345m/1-gpu"

    # TP and LP (tensor_para_size=4, layer_para_size=2)
    mpirun -n 8 --allow-run-as-root python ./pytorch/gpt_sample.py --tensor_para_size=4 --layer_para_size=2 --ckpt_path="/workspace/fastertransformer/models/megatron-models/c-model/345m/4-gpu"
    ```

    For PP, set `--layer_para_batch_size` so that batch_size >= layer_para_batch_size.

    2.2 Run GPT with TP and PP on single-node/multi-node (NVIDIA SuperPOD)
    #### Set up in interactive mode

    ```bash
    cd <FasterTransformer root path>
    srun -N1 --mpi=pmix --ntasks-per-node=8 --container-image nvcr.io/nvidia/pytorch:20.12-py3 --container-mounts <FasterTransformer root path>:/workspace/fastertransformer --container-workdir /workspace/fastertransformer --pty bash

    mkdir build && cd build
    cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON .. && make -j12
    ```

    #### Run on singe-node
    * tensor_para_size=8, layer_para_size=1

    ```bash
    srun -N1 --mpi=pmix --ntasks-per-node=8 --container-image nvcr.io/nvidia/pytorch:20.12-py3 --container-mounts <FasterTransformer root path>:/workspace/fastertransformer --container-workdir /workspace/fastertransformer/build python ./pytorch/gpt_sample.py --tensor_para_size=8 --layer_para_size=1 --ckpt_path="/workspace/fastertransformer/models/megatron-models/c-model/345m/8-gpu"
    ```

    #### Run on multi-node
    * tensor_para_size=8, layer_para_size=2

    ```bash
    srun -N2 --mpi=pmix --ntasks-per-node=8 --container-image nvcr.io/nvidia/pytorch:20.12-py3 --container-mounts <FasterTransformer root path>:/workspace/fastertransformer --container-workdir /workspace/fastertransformer/build python ./pytorch/gpt_sample.py --tensor_para_size=8 --layer_para_size=2 --ckpt_path="/workspace/fastertransformer/models/megatron-models/c-model/345m/8-gpu"
    ```

3. Run GPT on tensorflow

    Note that the tensorflow op only supports single gpu.

    ```bash
    ./bin/gpt_gemm 4 4 12 64 50257 1 1 1
    python tensorflow/gpt_sample.py --batch_size=4 \
                                    --length=32 \
                                    --top_k=4 \
                                    --top_p=0.6 \
                                    --data_type=fp16
    ```

### gpt with triton backend

Details are in [transformer_backend](https://github.com/triton-inference-server/fastertransformer_backend)

## Performance

Hardware settings: 
* 8xA100-80GBs (with mclk 1593MHz, pclk 1410MHz) with AMD EPYC 7742 64-Core Processor

We demonstrate the inference time of Megatron and FasterTransformer on Triton, and show the speedup of FasterTransformer compare to Megatron. In the experiments of encoder, we updated the following parameters:

* head_num = 96
* size_per_head = 128
* num_layers = 48 for GPT-89B model, 96 for GPT-175B model
* data_type = FP16
* vocab_size = 51200
* top_p = 0.9
* tensor parallel size = 8

### Perofrmance of GPT-89B

| Batch_size | Input Seqlen | Output Seqlen | Megatron <br/> Latency (ms) | FT <br/> Latency (ms) | FT <br/> Speedup |
|:----------:|:------------:|:-------------:|:---------------------------:|:---------------------:|:----------------:|
| 1  | 128 | 8  | 342.86  | 279.44  | 1.23 |
| 2  | 128 | 8  | 369.43  | 280.24  | 1.32 |
| 4  | 128 | 8  | 540.97  | 317.71  | 1.70 |
| 8  | 128 | 8  | 912.46  | 377.50  | 2.42 |
| 12 | 128 | 8  | 1263.39 | 445.46  | 2.84 |
| 16 | 128 | 8  | 1663.39 | 524.80  | 3.17 |
| 20 | 128 | 8  | 1991.16 | 575.83  | 3.46 |
| 32 | 128 | 8  | 3086.85 | 786.57  | 3.92 |
|    |     |    |         |         |      |
| 1  | 512 | 32 | 1244.81 | 887.52  | 1.40 |
| 2  | 512 | 32 | 1357.54 | 940.11  | 1.44 |
| 4  | 512 | 32 | 1970.08 | 1133.22 | 1.74 |
| 8  | 512 | 32 | 3341.66 | 1415.02 | 2.36 |
| 16 | 512 | 32 | 6090.07 | 1952.2  | 3.12 |

### Performance of GPT-175B

| Batch_size | Input Seqlen | Output Seqlen | Megatron <br/> Latency (ms) | FT <br/> Latency (ms) | FT <br/> Speedup |
|:----------:|:------------:|:-------------:|:---------------------------:|:---------------------:|:----------------:|
| 1  | 128 | 8  | 660.38   | 488.86  | 1.35 |
| 2  | 128 | 8  | 687.34   | 509.47  | 1.35 |
| 4  | 128 | 8  | 1004.88  | 629.64  | 1.60 |
| 8  | 128 | 8  | 1705.07  | 749.86  | 2.27 |
| 12 | 128 | 8  | 2365.02  | 886.24  | 2.67 |
| 16 | 128 | 8  | 3111.57  | 1037.47 | 3.00 |
| 20 | 128 | 8  | 3723.73  | 1135.72 | 3.28 |
| 32 | 128 | 8  | 5778.72  | 1547.44 | 3.73 |
|    |     |    |          |         |      |
| 1  | 512 | 32 | 2384.78  | 1719.96 | 1.39 |
| 2  | 512 | 32 | 2503.24  | 1830.56 | 1.37 |
| 4  | 512 | 32 | 3658.65  | 2092.56 | 1.75 |
| 8  | 512 | 32 | 6238.79  | 2629.97 | 2.37 |
| 16 | 512 | 32 | 11409.53 | 3706.23 | 3.08 |
