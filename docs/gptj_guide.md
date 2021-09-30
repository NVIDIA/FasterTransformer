# GPT-J

## Introduction

This document describes the step to run the GPT-J model on FasterTransformer.
GPT-J was developed by EleutherAI and trained on The Pile, a 825GB dataset from curated sources (e.g. Wikipedia, arXiv, GitHub, StackExchange, PubMed, ...).
With 6 billion parameters, GPT-J is one of the largest GPT-like publicly released models as of 2021.


## Setup

### Docker image

* The model was built and tested with the use nvcr image `nvcr.io/nvidia/pytorch:21.07-py3`. e.g.

    ```bash
    nvidia-docker run -ti --rm nvcr.io/nvidia/pytorch:21.07-py3 bash
    ```

* (internal) On Selene:

    ```bash
    srun -A devtech -p luna -N 1 --mpi=pmix --ntasks-per-node=8        \
    --container-image=nvcr.io#nvidia/pytorch:21.07-py3                 \
    --container-mounts=$(pwd):$(pwd),/lustre/fsw/adlr:/lustre/fsw/adlr \
    --container-workdir=$(pwd) --pty /bin/bash
    ```

### Setup

* Get the code and install all dependencies:

    ```bash
    git clone git clone https://gitlab-master.nvidia.com/zehuanw/FasterTransformer -b v5.0-dev-gptj
    mkdir -p FasterTransformer/build
    cd FasterTransformer/build
    git submodule init && git submodule update
    pip3 install fire jax jaxlib
    ```

### Build

* Note: the `xx` of `-DSM=xx` in following scripts means the compute capability of your GPU. For example, 60 (P40) or 61 (P4) or 70 (V100) or 75(T4) or 80 (A100).  Default setting is including 70, 75, 80 and 86.


    ```bash
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_GPT=ON ..
    make -j
    ```

### Download the model

To run the GPT on c, users need to convert the checkpoint of TensorFlow or PyTorch to binary files, and then load by FasterTransformer c api. Unfortunately, there is no published large model. So, users are only able to verify the correctness by smaller model. Currently, FasterTransformer provides two kinds of samples. First one is using the checkpoint of [OpenAI GPT-2 model](https://github.com/openai/gpt-2) (which is trained by TensorFlow); Another choice is using the checkpoint of [Megatron](https://github.com/NVIDIA/Megatron-LM) (which is trained by pytorch).

* Download openai-gpt model and convert


    ```bash
    wget https://the-eye.eu/public/AI/GPT-J-6B/step_383500_slim.tar.gz
    tar -axf step_383500_slim.tar.zstd
    python3 ../examples/pytorch/gptj/utils/gptj_ckpt_convert.py --output-dir ../models/j6b_ckpt --ckpt-dir ./step_383500 
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

* Generate the `gemm_config.in` file.

    ```bash
    ./bin/gpt_gemm <batch_size> <beam_width> <max_input_len> <head_number> <size_per_head> <inter_size> <vocab_size> <is_fp16> <tensor_para_size>
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

### Run through tritonserver

* This step requires setting up a new Docker instance with `tritonserver`. The following was tested with `nvcr.io#nvidia/tritonserver:21.07-py3` and install dependencies:

    ```bash
    # Needed for server and compilation:
    apt-get update
    apt-get install --yes python3-dev rapidjson-dev
    wget https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1-linux-x86_64.tar.gz
    tar -axf cmake-3.21.1-linux-x86_64.tar.gz
    export PATH=`pwd`/cmake-3.21.1-linux-x86_64/bin/:$PATH
    # Needed for client and token conversion:
    pip3 install tritonclient[all] fire regex
    ```

* Get up to date code (FasterTransformer is not needed, only used for token conversion):

    ```bash
    git clone https://gitlab-master.nvidia.com/bhsueh/fastertransformer_backend -b dev-gptj
    git clone https://gitlab-master.nvidia.com/zehuanw/FasterTransformer        -b v5.0-dev-gptj
    git clone https://github.com/triton-inference-server/server.git # We need some tools when we test this backend
    ln -s server/qa/common .
    ```

* Set up env variables:

    ```bash
    export WORKSPACE=$(pwd)
    export SRC_MODELS_DIR=${WORKSPACE}/models
    export TRITON_MODELS_STORE=${WORKSPACE}/triton-model-store
    export CONTAINER_VERSION=21.07
    export TRITON_DOCKER_IMAGE=triton_with_ft:${CONTAINER_VERSION}
    ```

* Install FT backend:

    ```bash
    mkdir -p fastertransformer_backend/build
    cd fastertransformer_backend/build
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/tritonserver -DTRITON_COMMON_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" -DTRITON_CORE_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" -DTRITON_BACKEND_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" ..
    make -j install
    ```

* Prepare the model store directory:

    ```bash
    cd ${WORKSPACE}
    mkdir -p ${TRITON_MODELS_STORE}/fastertransformer/1
    cp fastertransformer_backend/all_models/fastertransformer/config.pbtxt ${TRITON_MODELS_STORE}/fastertransformer
    ```
    The model parameter files should be placed in `${TRITON_MODELS_STORE}/fastertransformer/1/$NGPUS-gpu`. E.g.:
    ```bash
    python3 FasterTransformer/examples/pytorch/gptj/utils/gptj_ckpt_convert.py --output-dir ${TRITON_MODELS_STORE}/fastertransformer/1/ --ckpt-dir /path/to/step_383500
    ```

* Run server:

    ```bash
    mpirun -n 1 --allow-run-as-root tritonserver --model-repository=${HOME}/triton-model-store &
    ```

* Run client:

    ```bash
    bash fastertransformer_backend/tools/run_client.sh
    ```

* Decode:
    ```bash
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
    python3 FasterTransformer/examples/pytorch/gpt/utils/gpt_token_converter.py --out_file=triton_out --vocab_file=gpt2-vocab.json --bpe_file=gpt2-merges.txt
    ```
