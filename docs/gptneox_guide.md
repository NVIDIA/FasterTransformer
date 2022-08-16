# GPT-NeoX

## Table Of Contents

- [GPT-NeoX](#gpt-neox)
  - [Table Of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Supported features](#supported-features)
  - [Setup](#setup)
    - [Requirements](#requirements)
    - [Download the model](#download-the-model)
    - [Tokenizer](#tokenizer)
    - [Run GPT-NeoX](#run-gpt-neox)

## Introduction

This document describes the steps to run the GPT-NeoX model on FasterTransformer.
GPT-NeoX is a model developed by EleutherAI, available publicly on their GitHub [repository](https://github.com/EleutherAI/gpt-neox).
For the time being, only the 20B parameter version has been tested.

More details are listed in [gptj_guide.md](gptj_guide.md#introduction).

Optimization in gpt-neox are similar to optimization in GPT, describing in the [gpt_guide.md](gpt_guide.md#optimization).

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

## Setup

### Requirements

See common requirements such as in [gptj_guide.md](gptj_guide.md#requirements).

### Download the model

First download a pytorch checkpoint, as provided by [EleutherAI](https://github.com/EleutherAI/gpt-neox#download-links):

```bash
wget --cut-dirs=5 -nH -r --no-parent --reject "index.html*" https://mystic.the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/ -P 20B_checkpoints
```

Then use the script provided by FasterTransformer to convert the checkpoint to raw weights, understood by FT.

```bash
python ../examples/pytorch/gptneox/utils/eleutherai_gpt_neox_convert.py 20B_checkpoints ../models/gptneox -t 2
```

### Tokenizer

You may download the tokenizer config [here](https://mystic.the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/20B_tokenizer.json).

To tokenize/detokenize files, use the script found in `examples/pytorch/gptneox/utils/hftokenizer.py`. You may need to pass the path to the tokenizer config with the `--tokenizer` flag.

### Run GPT-NeoX

* Generate the `gemm_config.in` file.\
  Data Type = 0 (FP32) or 1 (FP16) or 2 (BF16)
    ```bash
    ./bin/gpt_gemm <batch_size> <beam_width> <max_input_len> <head_number> <size_per_head> <inter_size> <vocab_size> <data_type> <tensor_para_size>
    E.g., ./bin/gpt_gemm 8 1 32 64 96 24576 50432 1 2
    ```

* Run GPT on C++

    Users can see the details of arguments in `examples/cpp/gptneox/gptneox_config.ini`. It controls the model path, model size, tensor parallelism size, and some hyper-parameters.

    ```bash
    mpirun -n 2 --allow-run-as-root ./bin/gptneox_example
    ```

E.g. by setting the `data_type` of `gptneox_config.ini` to `fp16`, users can run gpt model under fp16.

You can then decode the `out` file with the tokenizer:

  ```bash
  wget https://mystic.the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/20B_tokenizer.json
  ../examples/pytorch/gptneox/utils/hftokenizer.py out --tokenizer 20B_tokenizer.json
  ```
<!-- This converter only works for customed checkpoint -->
<!-- ### Run GPT-NeoX with prompts

GPT-NeoX now supports prefix_prompt.

1.  Convert the prompt weights

    Convert the model and prompt weights by `examples/pytorch/gptneox/utils/huggingface_jp_gptneox_convert.py`, and it will automatically generate configuration needed for triton backend inference.

    Note that you need to specify `start_id`, `end_id` by yourself in order to make sure that it is consistent with the tokenizer.

2.  Run GPT-NeoX with C++ example

    You need to specify the example gpt_config.ini like below to enable the p/prompt_tuning feature.

    ```ini
    [gptneox_20B]
    head_num=64
    size_per_head=96
    vocab_size=50432
    decoder_layers=44
    rotary_embedding=24
    start_id=0
    end_id=2
    inter_size=24576
    use_gptj_residual=1
    num_tasks=2
    prompt_learning_type=2

    [gptneox_20B_task_0]
    task_name = squad
    prompt_length = 10

    [gptneox_20B_task_1]
    task_name = sentiment
    prompt_length = 10
    ```

    `task_name` and `prompt_length` are specified for loading prompt weights.

    **prompt_learning_type**:

    - no prompt: 0
    - soft_prompt: 1
    - prefix_prompt: 2
    - p/prompt_tuning: 3 -->
