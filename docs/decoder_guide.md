# FasterTransformer Decoder

The FasterTransformer Decoder contains the transformer decoder block, whole decoding progress, and GPT model.

## Table Of Contents

- [FasterTransformer Decoder](#fastertransformer-decoder)
  - [Table Of Contents](#table-of-contents)
  - [Model architecture](#model-architecture)
    - [Decoder](#decoder)
    - [Decoding progress](#decoding-progress)
    - [Decoder and Decoding](#decoder-and-decoding)
    - [GPT](#gpt)
  - [Setup](#setup)
    - [Requirements](#requirements)
  - [How to use](#how-to-use)
    - [Decoder and decoding process](#decoder-and-decoding-process)
    - [Translation process](#translation-process)
  - [Performance](#performance)
    - [End to end translation performance on TensorFlow](#end-to-end-translation-performance-on-tensorflow)
      - [Beamsearch performance on V100 and TensorFlow](#beamsearch-performance-on-v100-and-tensorflow)
      - [Sampling performance on V100 and TensorFlow](#sampling-performance-on-v100-and-tensorflow)
      - [Beamsearch performance on T4 and TensorFlow](#beamsearch-performance-on-t4-and-tensorflow)
      - [Sampling performance on T4 and TensorFlow](#sampling-performance-on-t4-and-tensorflow)
    - [End to end translation performance on PyTorch](#end-to-end-translation-performance-on-pytorch)
      - [Beamsearch performance on V100 and PyTorch](#beamsearch-performance-on-v100-and-pytorch)
      - [Beamsearch performance on T4 and PyTorch](#beamsearch-performance-on-t4-and-pytorch)

## Model architecture

<div align=center><img  width='616' height='1256' src ="images/decoding_flowchart.png "/></div>
<div align=center>Fig. 1 Flowchart of Decoding and GPT.</div>

### Decoder

Here, Decoder means the decoder transformer block, which contains two attention block and a feed-forward network. The modules in the red block of Fig. 1 demonstrate Decoder block.

The arguments, inputs, and outputs of decoder: 

* Arguments:
  1. Head number (H)
  2. size per head (N)
* Inputs:
  1. The features vector obtained by looking up the embedding table, or the previous result of the decoder. The shape is \[ B<sub>1</sub> x B<sub>2</sub>, 1, H x N \].
  2. The output of the encoder.
  3. The sequence length of the source sentence. Note that the lengths should be expanded by beam width times. 
  4. A memory cache space to store the K, V of masked multi-head attention. The size will grow for each step.
  5. A memory cache space to store the K, V of cross attention. Since K, V is computed by the encoder result, we only compute them in the first step, storing them into the cache, and then reuse in the other steps. 
  6. The weights of all parameters.
  7. To prevent the parallel computing of TensorFlow decoder and FasterTransformer Decoder, we put the TensorFlow result as a pseudo input in the TensorFlow OP. Otherwise, the results of FasterTransformer Decoder will incorrect. This input is useless for computing. Users can remove it when applying Decoder into a real application.  
* Outputs:
  1. Memory cache of masked multi-head attention. 
  2. Memory cache of cross attention. 
  3. The decoder output feature. The shape is \[ B<sub>1</sub> x B<sub>2</sub>, 1, H x N \].

### Decoding progress

Decoding refers to the whole translating process, including position encoding, embedding lookup, and beam search or sampling method to choose the token. Fig. 1 shows the different between decoding with beam search and sampling.

The arguments, inputs, and outputs of decoding with beam search: 

* Arguments:
  1. Beam width (B<sub>2</sub>)
  2. Maximum sequence length (S)
  3. Head number (H)
  4. Size per head (N)
  5. Number of decoder layers
  6. Start id of the vocabulary
  7. End id of the vocabulary
  8. Beam search diversity rate of [simple diverse decoding](https://arxiv.org/pdf/1611.08562.pdf)
* Inputs:
  1. The output of the encoder. The shape is \[ B<sub>1</sub>, memory sequence length, H x N \]. Where B<sub>1</sub> means the batch size.
  2. The sequence length of the source sentence. Note that the lengths should be expanded by beam width times.
  3. The table for embedding lookup. The shape is \[ V, H x N \].
  4. The weights of all parameters.
  5. Position encoding table. The shape is \[ S, H x N \].
* Outputs:
  1. The output ids. The shape is \[ S, B<sub>1</sub> x B<sub>2</sub> \].
  2. The parent ids, which are the chosen beam ids. The shape is \[ S, B<sub>1</sub> x B<sub>2</sub> \].
  3. The sequence lengths of each sentence. The shape is \[B<sub>1</sub> x B<sub>2</sub> \].

Note that the results of decoding with beam search are required to be finalized by TensorFlow's `tf.contrib.seq2seq.gather_tree` or other progress. 

The arguments, inputs, and outputs of decoding with sampling: 

* Arguments:
  1. Maximum sequence length (S)
  2. Top k value (K)
  3. Top p value (P)
  4. Head number (H)
  5. Size per head (N)
  6. Number of decoder layers
  7. Start id of the vocabulary
  8. End id of the vocabulary
* Inputs:
  1. The output of the encoder. The shape is \[ B, memory sequence length, H x N \]. Where B means the batch size.
  2. The sequence length of the source sentence. Note that the lengths should be expanded by beam width times.
  3. The table for embedding lookup. The shape is \[ V, H x N \].
  4. The weights of all parameters.
  5. Position encoding table. The shape is \[ S, H x N \].
* Outputs:
  1. The output ids. The shape is \[ S, B \].
  2. The sequence lengths of each sentence. The shape is \[ B \].

Note that K and P cannot be zero or non-zero value at the same time. FasterTransformer chooses the non-zero one to determine to use top k sampling or top p sampling. 

### Decoder and Decoding

Although the decoding process of most methods is similar, we find that there are lots of different kinds to compute the probability and implement the beam search. Therefore, if your chosen beam search algorithm is different from our implementation and it is hard for you to modify the beam search kernel, TensorFlow decoding with FasterTransformer Decoder is the recommended choice. However, the performance of the TensorFlow decoding with the FasterTransformer Decoder is worse than the performance of the FasterTransformer Decoding, especially for small batch sizes.

### GPT

The GPT model is based on [OpenAI gpt-2 project](https://github.com/openai/gpt-2). GPT is a special case of Decoding, it does not require the cross-attention block and the results from encoder. Users can put some started words into GPT, and GPT will use these words to generate the next word. By this method, GPT can translate the sentence, reply the questions, and do many different applications. Fig. 1 shows the difference between GPT standard decoding model. More details are put in [`docs/gpt_guide.md`](gpt_guide.md).

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

## How to use

### Decoder and decoding process

1. Run FasterTransformer decoding on C++

    1.1 Generate the `decoding_gemm_config.in` file. 

    `./bin/decoding_gemm` can generate the best GEMM configuration. The arguments of `decoding_gemm` are:

    ```bash
    ./bin/decoding_gemm <batch_size> <beam_width> <head_number> <size_per_head> <vocab_size> <sequence_length> <encoder_hidden_dim> <is_use_fp16>
    ```

    Assume the settings of decoding are as follows.

    - `batch_size`=32
    - `beam_width`=4
    - `head_number`=8
    - `size_per_head`=64 
    - `vocabulary_size`=30000
    - `sequence_length`=32
    - `encoder's hidden dimension`=512
    - `data_type`=FP32

    Then the following scripts can generate the best GEMM configuration under such settings, and record the configuration into the `decoding_gemm_config.in` file.

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 0
    ```

    1.2 Run decoding under FP32 on C++

    Assume the settings are the same as above, and the decoder contains 6 transformer layers. 

    In the decoding, we provide two kinds of methods to choose the tokens from the candidates. The first kind of method is the beam search algorithm. The second kind of method is sampling algorithm. 

    For beam search, we provide a simple diverse decoding of [link](https://arxiv.org/pdf/1611.08562.pdf). When the diversity rate is set to 0, it is equivalent to the naive beam search. 

    For sampling, we provide the top k sampling and top p sampling. Here, k is an integer number and p is a float point number. Note that we cannot use both of them at the same time. So, only one of both can be non-zero value. 

    `./bin/decoding_beamsearch_sample` runs the decoding with beam search in the `C++`. The arguments of `decoding_beamsearch_sample` is:

    ```bash
    ./bin/decoding_beamsearch_sample <batch_size> <beam_width> <head_number> <size_per_head> <vocab_size> <sequence_length> <num_layers> <encoder_hidden_dim> <is_use_fp16>
    ```

    Then the following scripts can run the decoding with beam search under the above settings. 

    ```bash
    ./bin/decoding_beamsearch_sample 32 4 8 64 30000 32 6 512 0
    ```

    The outputs should be like to the following:

    ```bash 
    Device Tesla V100-PCIE-32GB
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-CPP-decoding-beamsearch-time 73.36 ms
    ```

    `./bin/decoding_sampling_sample` runs the decoding with sampling in the `C++`. The arguments of `decoding_sampling_sample` is:

    ```bash
    ./bin/decoding_sampling_sample <batch_size> <candidate_num> <probability_threshold> <head_number> <size_per_head> <vocab_size> <sequence_length> <num_layers> <encoder_hidden_dim> <is_use_fp16>
    ```

    where `candidate_num` is the k value of top k, while `probability_threshold` is the p value of top p.

    Note that the beam width of sampling algorithm is always 1, so we need to generate the new configuration.

    The following scripts can run the decoding with top k sampling with under the above settings. 

    ```bash
    ./bin/decoding_gemm 32 1 8 64 30000 32 512 0
    ./bin/decoding_sampling_sample 32 4 0.0 8 64 30000 32 6 512 0
    ```

    The outputs should be like to the following:

    ```bash 
    Device Tesla V100-PCIE-32GB
    [INFO] batch_size 32 topk 4 topp 0.000000 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-CPP-decoding-sampling-time 41.65 ms
    ```

    The following scripts can run the decoding with top p sampling with under the above settings. 

    ```bash
    ./bin/decoding_gemm 32 1 8 64 30000 32 512 0
    ./bin/decoding_sampling_sample 32 0 0.01 8 64 30000 32 6 512 0
    ```

    The outputs should be like to the following:

    ```bash 
    Device Tesla V100-PCIE-32GB
    [INFO] batch_size 32 topk 0 topp 0.010000 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-CPP-decoding-sampling-time 61.63 ms
    ```

    1.3 Run decoding under FP16 on C++

    So far, we use the FP32 to run the FasterTransformer. If we use the volta or newer NVIDIA GPU, we can use tensor core to accelerate when we use the FP16. 

    To use the FP16, we only need to set the `<is_use_fp16>` flag to 1 like following:

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 1
    ./bin/decoding_beamsearch_sample 32 4 8 64 30000 32 6 512 1
    ```

    Note that the configuration of FP32 and FP16 are different, so we need to generate the configuration again. 

    The outputs should be like to the following:  

    ```bash
    Device Tesla V100-PCIE-32GB
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-CPP-decoding-beamsearch-time 47.89 ms
    ```

2. Run FasterTransformer decoder/decoding on TensorFlow

    2.1 Run FasterTransformer decoder under FP32 on TensorFlow

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 0
    python tensorflow/decoder_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp32 \
            --decoder_type 2
    ```

    The outputs should be like to the following:

    ```bash 
    [[INFO][PYTHON] step:][29][True][max abs diff: ][4.17232513e-06][ op val: ][1.23598516][ tf val: ][1.23598933]
    [[INFO][PYTHON] step:][30][True][max abs diff: ][4.05311584e-06][ op val: ][-2.40530682][ tf val: ][-2.40531087]
    [[INFO][PYTHON] step:][31][False][max abs diff: ][3.7997961e-06][ op val: ][-0.120998174][ tf val: ][-0.121001974]
    ```

    The results show that the differences between the decoder of TensorFlow and decoder are smaller than threshold. Sometimes, the differences are larger than the threshold and the checking will return "False", but it does not affect the results.

    The argument `decoder_type` decides to use the decoder of TensorFlow or decoder of FasterTransformer. `decoder_type 2` uses both decoders and compares their results. 

    The following script demonstrates the execution time of the FasterTransformer decoder.

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 0
    python tensorflow/decoder_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp32 \
            --decoder_type 1 \
            --test_time 1
    ```

    The outputs should be like to the following:

    ```bash
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-OP-decoder-time 138.90 ms.
    ```

    The following script demonstrates the execution time of the TensorFlow decoder.

    ```bash 
    python tensorflow/decoder_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp32 \
            --decoder_type 0 \
            --test_time 1
    ```

    The outputs should be like to the following:

    ```bash
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 TF-decoding-beamsearch-time 564.37 ms.
    ```

    2.2 Run FasterTransformer decoder under FP16 on TensorFlow

    To use the FP16 in TensorFlow, we only need to set the `--data_type fp16` like following:

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 1
    python tensorflow/decoder_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp16 \
            --decoder_type 2
    ```

    The outputs should be like to the following:

    ```bash 
    [[INFO][PYTHON] step:][29][True][max abs diff: ][0.01171875][ op val: ][2.03125][ tf val: ][2.04296875]
    [[INFO][PYTHON] step:][30][True][max abs diff: ][0.01171875][ op val: ][2.3671875][ tf val: ][2.35546875]
    [[INFO][PYTHON] step:][31][True][max abs diff: ][0.01171875][ op val: ][2.33398438][ tf val: ][2.32226562]
    ```

    The following script demonstrates the execution time of the FasterTransformer decoder.

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 1
    python tensorflow/decoder_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp16 \
            --decoder_type 1 \
            --test_time 1
    ```

    The outputs should be like to the following:

    ```bash
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-OP-decoder-time 132.48 ms.
    ```

    The following script demonstrates the execution time of the TensorFlow decoder.

    ```bash 
    python tensorflow/decoder_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp16 \
            --decoder_type 0 \
            --test_time 1
    ```

    The outputs should be like to the following:

    ```bash
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 TF-decoding-beamsearch-time 503.52 ms.
    ```

    Note that when the batch size is small, using FP16 may cause the inference speed to become slower. This is because that decoding is not computing bound and using FP16 in TensorFlow leads to some additional operation and casting. 

    2.3 Run FasterTransformer decoding under FP32 on TensorFlow

    In the decoding, we provide two kinds of methods to choose the tokens from the candidates. The first kind of method is the beam search algorithm. The second kind of method is sampling algorithm. 

    For beam search, we provide a simple diverse decoding of [link](https://arxiv.org/pdf/1611.08562.pdf). When the `--beam_search_diversity_rate` is set to 0, it is equivalent to the naive beam search. 

    For sampling, we provide the top k sampling and top p sampling, which are set by the arguments `--sampling_topk` and `--sampling_topp`. Here, k is an integer number and p is a float point number. Note that we cannot use both at the same time. So, only one of both can be non-zero value. 

    The following script uses diverse decoding with diversity rate 0 and top k sampling with k = 4:

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 0
    python tensorflow/decoding_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp32 \
            --beam_search_diversity_rate 0 \
            --sampling_topk 4 \
            --sampling_topp 0.0 \
            --test_time 0123
    ```
    
    The outputs should be like to the following:

    ```bash
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 TF-decoding-beamsearch-time 555.87 ms.
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-OP-decoding-beamsearch-time  75.80 ms.
    [INFO] batch_size 32 topk 4 topp 0.0 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 TF-decoding-sampling-time 432.40 ms.
    [INFO] batch_size 32 topk 4 topp 0.0 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-OP-decoding-sampling-time  46.68 ms.
    ```

    Note that the results of FasterTransformer may be different, especially when the batch size is larger.

    Here, we use same configuration to run the decoding with beam search and sampling at the same time. This is not correct because the beam width of decoding with sampling is always 1, so the configurations of them are same only when the beam width is 1. However, this only little reduce the speed of decoding with sampling, so we ignore this problem here. 

    Here, the meaning of argument `--test_time` is different. 0 means testing the TensorFlow with beam search; 1 means testing the FasterTransformer with beam search; 2 means testing the TensorFlow with sampling; 3 means testing the FasterTransformer with sampling. 

    The following script uses diverse decoding with diversity rate -1.3 and top p sampling with p = 0.01:

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 0
    python tensorflow/decoding_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp32 \
            --beam_search_diversity_rate -1.3 \
            --sampling_topk 0 \
            --sampling_topp 0.01 \
            --test_time 0123
    ```

    The outputs should be like to the following:

    ```bash
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 TF-decoding-beamsearch-time 525.55 ms.
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-OP-decoding-beamsearch-time  76.79 ms.
    [INFO] batch_size 32 topk 4 topp 0.0 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 TF-decoding-sampling-time 420.98 ms.
    [INFO] batch_size 32 topk 4 topp 0.0 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-OP-decoding-sampling-time  46.37 ms.
    ```

    For the sampling algorithm, the results of TensorFlow and FasterTransformer are often different. 

    2.4 Run FasterTransformer decoding under FP16 on TensorFlow

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 1
    python tensorflow/decoding_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp16 \
            --beam_search_diversity_rate 0.0 \
            --sampling_topk 4 \
            --sampling_topp 0.00 \
            --test_time 0123
    ```

    The outputs should be like to the following:

    ```bash
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 TF-decoding-beamsearch-time 494.23 ms.
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-OP-decoding-beamsearch-time  50.43 ms.
    [INFO] batch_size 32 topk 4 topp 0.0 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 TF-decoding-sampling-time 382.34 ms.
    [INFO] batch_size 32 topk 4 topp 0.0 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-OP-decoding-sampling-time  33.19 ms.
    ```

    Note that the results of FasterTransformer may be different, especially when the batch size is larger.

3. Run FasterTransformer decoder/decoding on PyTorch

    Please install OpenNMT-py first before running the demos by
    ```bash
    pip install opennmt-py==1.1.1
    ```

    3.1 Generate the `decoding_gemm_config.in` file:

    ```bash
    ./bin/decoding_gemm <batch_size> <beam_size> <head_number> <size_per_head> <vocab_size> <seq_len> <memory_hidden_dim> <is_fp16>
    ./bin/decoding_gemm 8 4 8 64 31538 32 512 1
    ```
    If you want to use the library in other directory, please generate this file according to your setting and copy it to your working directory.

    3.2 Run the PyTorch decoder sample: 

    ```bash
    python pytorch/decoder_sample.py <batch_size> <layer_num> <sequence_length> <head_number> <size_per_head> <--fp16> <--time>
    python pytorch/decoder_sample.py 8 6 32 8 64 --fp16 --time
    ```
    Remove `--fp16` for fp32 mode.

    The outputs should be like to the following:

    ```bash
    step: 30     Mean relative diff: 0.01395416259765625     Max relative diff: 1.38671875     Min relative diff: 0.0
    step: 31     Mean relative diff: 0.0148468017578125     Max relative diff: 2.880859375     Min relative diff: 0.0
    [INFO] ONMTDecoder time costs: 218.37 ms
    [INFO] FTDecoder time costs: 25.15 ms
    ```

    Note that the relative diff is very large. It is caused by the random initial weights and inputs, and it does not affect the result of translation.

    3.3 Run the PyTorch decoding sample: 

    ```bash
    python pytorch/decoding_sample.py <batch_size> <layer_num> <sequence_length> <head_number> <size_per_head> <beam_size> <vocab_size> <--fp16> <--time>
    python pytorch/decoding_sample.py 8 6 32 8 64 4 31538 --fp16 --time
    ```
    Remove `--fp16` for fp32 mode.

    The outputs should be like to the following:

    ```bash
    [INFO] TorchDecoding time costs: 289.08 ms
    [INFO] TorchDecoding (with FTDecoder) time costs: 104.15 ms
    [INFO] FTDecoding time costs: 30.57 ms
    ```

    Random initialized parameters may lead to different results. You can download the pretrained model following the instruction in the next part, and add `--use_pretrained`, then you can get the same results.

### Translation process

1. Translation with FasterTransformer on TensorFlow

    This subsection demonstrates how to use FasterTransformer decoding to translate a sentence. We use the pretrained model and testing data in [OpenNMT-tf](https://opennmt.net/Models-tf/), which translates from English to German. 

    Because the FasterTransformer Encoder is based on BERT, we cannot restore the model of encoder of OpenNMT to FasterTransformer Encoder. Therefore, we use OpenNMT-tf to build the encoder and preprocess the source sentence.

    Another problem is that the implementation of FasterTransformer Decoder and decoder of OpenNMT-tf is a little different. For example, the decoder of OpenNMT-tf uses one convolution to compute query, key, and value in masked-multi-head-attention; but FasterTransformer Decoder splits them into three gemms. One method is using the tool `utils/dump_model.py` to convert the pretrained model to fit the model structure of FasterTransformer Decoder. Another method is Splitting the weights during inference.

    `download_model_data.sh` will install the OpenNMT-tf v1, downloading the pretrained model into the `translation` folder, and convert the model. 

    ```bash
    bash tensorflow/utils/translation/download_model_data.sh
    ```

    Then run the translation sample by the following script:

    ```bash
    ./bin/decoding_gemm 128 4 8 64 32001 100 512 0
    python tensorflow/translate_sample.py \
            --batch_size 128 \
            --beam_width 4 \
            --encoder_head_number 8 \
            --encoder_size_per_head 64 \
            --decoder_head_number 8 \
            --decoder_size_per_head 64 \
            --max_seq_len 32 \
            --encoder_num_layer 6 \
            --decoder_num_layer 6 \
            --data_type fp32 \
            --beam_search_diversity_rate 0.0 \
            --sampling_topk 1 \
            --sampling_topp 0.00 \
            --test_time 012345
    ```

    The outputs of should be similar to the following:

    ```bash
    [INFO] tf-decoding-beamsearch translates 24 batches taking 31.39 ms to translate 67092 tokens, BLEU score: 26.29, 2137 tokens/sec.
    [INFO] op-decoder-beamsearch translates 24 batches taking 10.37 ms to translate 67092 tokens, BLEU score: 26.29, 6473 tokens/sec.
    [INFO] op-decoding-beamsearch translates 24 batches taking 7.88 ms to translate 67124 tokens, BLEU score: 26.31, 8513 tokens/sec.
    [INFO] tf-decoding-sampling translates 24 batches taking 16.23 ms to translate 67813 tokens, BLEU score: 25.79, 4178 tokens/sec.
    [INFO] op-decoder-sampling translates 24 batches taking 6.29 ms to translate 67813 tokens, BLEU score: 25.79, 10781 tokens/sec.
    [INFO] op-decoding-sampling translates 24 batches taking 4.10 ms to translate 67813 tokens, BLEU score: 25.79, 16524 tokens/sec.
    ```

    The scripts of running under FP16 is following:

    ```bash
    python tensorflow/tensorflow_bert/ckpt_type_convert.py --init_checkpoint=translation/ckpt/model.ckpt-500000 --fp16_checkpoint=translation/ckpt/fp16_model.ckpt-500000
    ./bin/decoding_gemm 128 4 8 64 32001 100 512 1
    python tensorflow/translate_sample.py \
          --batch_size 128 \
          --beam_width 4 \
          --encoder_head_number 8 \
          --encoder_size_per_head 64 \
          --decoder_head_number 8 \
          --decoder_size_per_head 64 \
          --max_seq_len 32 \
          --encoder_num_layer 6 \
          --decoder_num_layer 6 \
          --data_type fp16 \
          --beam_search_diversity_rate 0.0 \
          --sampling_topk 1 \
          --sampling_topp 0.00 \
          --test_time 012345
    ```

    The outputs of should be similar to the following:

    ```bash
    [INFO] tf-decoding-beamsearch translates 24 batches taking 22.75 ms to translate 67094 tokens, BLEU score: 26.31, 2949 tokens/sec.
    [INFO] op-decoder-beamsearch translates 24 batches taking 7.73 ms to translate 67089 tokens, BLEU score: 26.30, 8682 tokens/sec.
    [INFO] op-decoding-beamsearch translates 24 batches taking 5.27 ms to translate 67130 tokens, BLEU score: 26.33, 12746 tokens/sec.
    [INFO] tf-decoding-sampling translates 24 batches taking 13.65 ms to translate 67828 tokens, BLEU score: 25.83, 4968 tokens/sec.
    [INFO] op-decoder-sampling translates 24 batches taking 4.92 ms to translate 67831 tokens, BLEU score: 25.80, 13773 tokens/sec.
    [INFO] op-decoding-sampling translates 24 batches taking 2.54 ms to translate 67844 tokens, BLEU score: 25.82, 26718 tokens/sec.
    ```

2.  Translation with FasterTransformer on PyTorch

    We have a translation demo for En-De translation.

    You need to download the pretrained_model first by:

    ```bash
    bash pytorch/scripts/download_translation_model.sh
    ```

    Then you can run the demo by:

    ```bash
    python pytorch/run_translation.py --batch_size <batch_size> --beam_size <beam_size> --model_type <model_type> --data_type <data_type> --output_file <output_file>
    ```
    you can also use `--input_file` to set the input file to be translated.

    the `<model_type>` can be:
    <!-- - `ori`: original OpenNMT model -->
    <!-- - `decoder_ext`: replace the decoder in OpenNMT model with our FasterTransformer decoder -->
    - `decoding_ext`: using our FasterTransformer decoding module
    - `torch_decoding`: PyTorch version decoding with the method FasterTransformer decoding uses
    - `torch_decoding_with_decoder_ext`: PyTorch version decoding with the method FasterTransformer decoding uses but replace the decoder with the FasterTransformer decoder

    the `<data_type>` can be `fp32` or `fp16`

    if you do not specify the output file, it only print to the stdout.

    If you want to evaluate the BLEU score, please recover the BPE first by:
    ```bash
    python pytorch/utils/recover_bpe.py <ref_file> <debpe_ref_file>
    python pytorch/utils/recover_bpe.py <output_file> <debpe_output_file>
    ```
    the `<ref_file>` for our demo is `pytorch/translation/data/test.de`, the `<output_file>` is the output from `run_translation.py`.

    Then you can evalute the BLEU score, for example, through `sacrebleu`:
    ```bash
    pip install sacrebleu
    cat <debpe_output_file> | sacrebleu <debpe_ref_file>
    ```

    The following scripts run translation under FP32 and get the bleu score:

    ```bash
    ./bin/decoding_gemm 128 4 8 64 31538 100 512 0
    python pytorch/run_translation.py --batch_size 128 --beam_size 4 --model_type decoding_ext --data_type fp32 --output_file output.txt
    python pytorch/utils/recover_bpe.py pytorch/translation/data/test.de debpe_ref.txt
    python pytorch/utils/recover_bpe.py output.txt debpe_output.txt
    pip install sacrebleu
    cat debpe_output.txt | sacrebleu debpe_ref.txt
    ```

## Performance

Hardware settings: 
* CPU: Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz
* T4 (with mclk 5000MHz, pclk 1590MHz) with Intel(R) Xeon(R) CPU E5-2603 v4 @ 1.70GHz
* V100 (with mclk 877MHz, pclk 1380MHz) with Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz (dgx-1 server)

To run the following benchmark, we need to install the unix computing tool "bc" by

```bash
apt-get install bc
```

To understand the speedup on real application, we use real end to end model and task in this benchmark on both TensorFlow and PyTorch. It is hard to compare the performance of v3.1 and v4.0 this the benchmark directly. But by our testing, compared to v3.1, v4.0 brings at most 50% speedup, especially for large batch size.

### End to end translation performance on TensorFlow

We demonstrate the throughput of TensorFlow (`TF`), `FT Decoder` and `FT Decoding` for end to end translation. Here, TensorFlow means that the program fully runs on TensorFlow. FT Decoder means that we replace the decoder transformer layer by FasterTransformer. FT Decoding means that we replace the whole procedure of decoder by FasterTransformer. Besides, we also replace the encoder transformer layer by FasterTransformer Encoder in FT Decoding.

We do not demonstrate the performance of TensorFlow with XLA since we did not find that using XLA has obvious speedup. We also skip the BLEU score because the score of TensorFlow, FT Decoder and FT Decoding are close.

Althought the bleu scores of all methods are close, the results may be little different and the number of generated tokens may be also different. So, we use throughput but not latency to show the peformance in this benchmark.

The benchmark of beamsearch were obtained by running the `sample/tensorflow/scripts/profile_decoding_beamsearch_performance.sh`; while The benchmark of sampling were obtained by running the `sample/tensorflow/scripts/profile_decoding_sampling_performance.sh`..

In this benchmark, we updated the following parameters:

* head_num = 8 for both encoder and decoder
* size_per_head = 64 for both encoder and decoder
* num_layers = 6 for both encoder and decoder
* vocabulary_size = 32001
* max_seq_len = 128

#### Beamsearch performance on V100 and TensorFlow

* Performance on FP32

| Batch Size | Beam Width | Precision | TF <br/> Throughput (token/sec) | FT Decoder <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoder <br/> Speedup | FT Decoding <br/> Speedup |
|:----------:|:----------:|:---------:|:-------------------------------:|:---------------------------------------:|:----------------------------------------:|:------------------------:|:-------------------------:|
|   1 |   1 | FP32 |    95 |   351 |   800 | 3.69 | 8.42 |
|   1 |   4 | FP32 |   110 |   341 |   763 | 3.10 | 6.93 |
|   1 |  32 | FP32 |    78 |   171 |   489 | 2.19 | 6.26 |
|   8 |   1 | FP32 |   484 |  1645 |  3694 | 3.39 | 7.63 |
|   8 |   4 | FP32 |   511 |  1435 |  3068 | 2.80 | 6.00 |
|   8 |  32 | FP32 |   231 |   427 |   916 | 1.84 | 3.96 |
| 128 |   1 | FP32 |  3157 |  8373 | 19803 | 2.65 | 6.27 |
| 128 |   4 | FP32 |  1773 |  3648 |  7848 | 2.05 | 4.42 |

* Performance on FP16

| Batch Size | Beam Width | Precision | TF <br/> Throughput (token/sec) | FT Decoder <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoder <br/> Speedup | FT Decoding <br/> Speedup |
|:----------:|:----------:|:---------:|:-------------------------------:|:---------------------------------------:|:----------------------------------------:|:------------------------:|:-------------------------:|
|   1 |   1 | FP16 |   153 |   360 |  1043 | 2.35 | 6.81 |
|   1 |   4 | FP16 |   143 |   333 |   915 | 2.32 | 6.39 |
|   1 |  32 | FP16 |   102 |   179 |   630 | 1.75 | 6.17 |
|   8 |   1 | FP16 |   662 |  1652 |  4863 | 2.49 | 7.34 |
|   8 |   4 | FP16 |   619 |  1457 |  3995 | 2.35 | 6.45 |
|   8 |  32 | FP16 |   359 |   504 |  1413 | 1.40 | 3.93 |
| 128 |   1 | FP16 |  5693 | 10454 | 30890 | 1.83 | 5.42 |
| 128 |   4 | FP16 |  3316 |  5231 | 16856 | 1.57 | 5.08 |

#### Sampling performance on V100 and TensorFlow

* Performance on FP32

| Batch Size | Topk/Topp | Precision | TF <br/> Throughput (token/sec) | FT Decoder <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoder <br/> Speedup | FT Decoding <br/> Speedup |
|:----------:|:---------:|:---------:|:-------------------------------:|:---------------------------------------:|:----------------------------------------:|:------------------------:|:-------------------------:|
|   1 |  4   | FP32 |   119 |   379 |   759 | 3.18 | 6.37 |
|   1 | 32   | FP32 |   103 |   368 |   739 | 3.57 | 7.17 |
|   1 | 0.75 | FP32 |   111 |   324 |   619 | 2.91 | 5.57 |
|   8 |  4   | FP32 |   491 |  1765 |  3475 | 3.59 | 7.07 |
|   8 | 32   | FP32 |   483 |  1637 |  3395 | 3.38 | 7.02 |
|   8 | 0.75 | FP32 |   460 |  1460 |  2645 | 3.17 | 5.75 |
| 128 |  4   | FP32 |  3387 |  9203 | 18165 | 2.71 | 5.36 |
| 128 | 32   | FP32 |  3380 |  8605 | 17541 | 2.54 | 5.18 |
| 128 | 0.75 | FP32 |  3194 |  6898 | 13925 | 2.15 | 4.35 |

* Performance on FP16

| Batch Size | Topk/Topp | Precision | TF <br/> Throughput (token/sec) | FT Decoder <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoder <br/> Speedup | FT Decoding <br/> Speedup |
|:----------:|:---------:|:---------:|:-------------------------------:|:---------------------------------------:|:----------------------------------------:|:------------------------:|:-------------------------:|
|   1 |  4   | FP16 |   169 |   412 |   992 | 2.43 | 5.86 |
|   1 | 32   | FP16 |   167 |   376 |   970 | 2.25 | 5.80 |
|   1 | 0.75 | FP16 |   160 |   350 |   845 | 2.18 | 5.28 |
|   8 |  4   | FP16 |   739 |  1802 |  4620 | 2.43 | 6.25 |
|   8 | 32   | FP16 |   785 |  1754 |  4425 | 2.23 | 5.63 |
|   8 | 0.75 | FP16 |   715 |  1586 |  3634 | 2.21 | 5.08 |
| 128 |  4   | FP16 |  6217 | 11392 | 29409 | 1.83 | 4.73 |
| 128 | 32   | FP16 |  5937 | 10366 | 27995 | 1.74 | 4.71 |
| 128 | 0.75 | FP16 |  5129 |  8423 | 22094 | 1.64 | 4.30 |

#### Beamsearch performance on T4 and TensorFlow

* Performance on FP32

| Batch Size | Beam Width | Precision | TF <br/> Throughput (token/sec) | FT Decoder <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoder <br/> Speedup | FT Decoding <br/> Speedup |
|:----------:|:----------:|:---------:|:-------------------------------:|:---------------------------------------:|:----------------------------------------:|:------------------------:|:-------------------------:|
|   1 |   1 | FP32 |    40 |   151 |   599 | 3.77 | 14.97 |
|   1 |   4 | FP32 |    34 |   137 |   563 | 4.02 | 16.55 |
|   1 |  32 | FP32 |    37 |    91 |   330 | 2.45 | 8.91  |
|   8 |   1 | FP32 |   193 |   807 |  2868 | 4.18 | 14.86 |
|   8 |   4 | FP32 |   198 |   644 |  2205 | 3.25 | 11.13 |
|   8 |  32 | FP32 |    94 |   209 |   366 | 2.22 | 3.89  |
| 128 |   1 | FP32 |  1234 |  3420 | 10313 | 2.77 | 8.35  |
| 128 |   4 | FP32 |   677 |  1260 |  3114 | 1.86 | 4.59  |

* Performance on FP16

| Batch Size | Beam Width | Precision | TF <br/> Throughput (token/sec) | FT Decoder <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoder <br/> Speedup | FT Decoding <br/> Speedup |
|:----------:|:----------:|:---------:|:-------------------------------:|:---------------------------------------:|:----------------------------------------:|:------------------------:|:-------------------------:|
|   1 |   1 | FP16 |    57 |   175 |   786 | 3.07 | 13.78 |
|   1 |   4 | FP16 |    55 |   169 |   766 | 3.07 | 13.92 |
|   1 |  32 | FP16 |    45 |    94 |   465 | 2.08 | 10.33 |
|   8 |   1 | FP16 |   226 |   683 |  4077 | 3.02 | 18.03 |
|   8 |   4 | FP16 |   217 |   631 |  3440 | 2.90 | 15.85 |
|   8 |  32 | FP16 |   151 |   259 |   619 | 1.71 | 4.09  |
| 128 |   1 | FP16 |  2060 |  4474 | 21675 | 2.17 | 10.52 |
| 128 |   4 | FP16 |  1250 |  1948 |  8796 | 1.55 | 7.03  |

#### Sampling performance on T4 and TensorFlow

* Performance on FP32

| Batch Size | Topk/Topp | Precision | TF <br/> Throughput (token/sec) | FT Decoder <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoder <br/> Speedup | FT Decoding <br/> Speedup |
|:----------:|:---------:|:---------:|:-------------------------------:|:---------------------------------------:|:----------------------------------------:|:------------------------:|:-------------------------:|
|   1 |  4   | FP32 |    49 |   201 |   584 | 4.10 | 11.91 |
|   1 | 32   | FP32 |    50 |   175 |   568 | 3.50 | 11.36 |
|   1 | 0.75 | FP32 |    48 |   156 |   494 | 3.25 | 10.29 |
|   8 |  4   | FP32 |   226 |   791 |  2753 | 3.50 | 12.18 |
|   8 | 32   | FP32 |   230 |   859 |  2643 | 3.73 | 11.49 |
|   8 | 0.75 | FP32 |   230 |   706 |  2225 | 3.06 | 9.67  |
| 128 |  4   | FP32 |  1443 |  3729 |  8822 | 2.58 | 6.11  |
| 128 | 32   | FP32 |  1372 |  3396 |  8694 | 2.47 | 6.33  |
| 128 | 0.75 | FP32 |  1259 |  2640 |  7127 | 2.09 | 5.66  |

* Performance on FP16

| Batch Size | Topk/Topp | Precision | TF <br/> Throughput (token/sec) | FT Decoder <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoder <br/> Speedup | FT Decoding <br/> Speedup |
|:----------:|:---------:|:---------:|:-------------------------------:|:---------------------------------------:|:----------------------------------------:|:------------------------:|:-------------------------:|
|   1 |  4   | FP16 |    70 |   211 |   765 | 3.01 | 10.92 |
|   1 | 32   | FP16 |    68 |   201 |   756 | 2.95 | 11.11 |
|   1 | 0.75 | FP16 |    65 |   163 |   658 | 2.50 | 10.12 |
|   8 |  4   | FP16 |   296 |   904 |  3821 | 3.05 | 12.90 |
|   8 | 32   | FP16 |   291 |   851 |  3929 | 2.92 | 13.50 |
|   8 | 0.75 | FP16 |   280 |   723 |  3168 | 2.58 | 11.31 |
| 128 |  4   | FP16 |  2649 |  4810 | 21185 | 1.81 | 7.99  |
| 128 | 32   | FP16 |  2337 |  4632 | 18966 | 1.98 | 8.11  |
| 128 | 0.75 | FP16 |  1937 |  3269 | 15599 | 1.68 | 8.05  |

### End to end translation performance on PyTorch

We demonstrate the throughput of PyTorch, FT Decoder and FT Decoding for end to end translation. Here, PyTorch means that the program fully runs on PyTorch. FT Decoder means that we replace the decoder transformer layer by FasterTransformer. FT Decoding means that we replace the whole procedure of decoder by FasterTransformer.

We also skip the BLEU score because the score of PyTorch, FT Decoder and FT Decoding are close.

Althought the bleu scores of all methods are close, the results may be little different and the number of generated tokens may be also different. So, we use throughput but not latency to show the peformance in this benchmark.

This benchmark were obtained by running the `../sample/pytorch/scripts/profile_decoder_decoding.sh`.

In this benchmark, we updated the following parameters:

* head_num = 8 for both encoder and decoder
* size_per_head = 64 for both encoder and decoder
* num_layers = 6 for both encoder and decoder
* vocabulary_size = 31538
* max_seq_len = 128

#### Beamsearch performance on V100 and PyTorch

* Perofrmance on FP32

| Batch Size | Beam Width | Precision | PyTorch <br/> Throughput (token/sec) | FT Decoder <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoder <br/> Speedup | FT Decoding <br/> Speedup |
|:----------:|:----------:|:---------:|:------------------------------------:|:---------------------------------------:|:----------------------------------------:|:------------------------:|:-------------------------:|
|   1 |   1 | FP32 |    92 |   277 |   699 | 3.00 | 7.56 |
|   1 |   4 | FP32 |    80 |   226 |   703 | 2.82 | 8.76 |
|   1 |  32 | FP32 |    69 |   217 |   471 | 3.12 | 6.76 |
|   8 |   1 | FP32 |   385 |  1232 |  3225 | 3.20 | 8.37 |
|   8 |   4 | FP32 |   352 |  1121 |  2756 | 3.18 | 7.81 |
|   8 |  32 | FP32 |   262 |   465 |   950 | 1.77 | 3.62 |
| 128 |   1 | FP32 |  2968 |  6213 | 12848 | 2.09 | 4.32 |
| 128 |   4 | FP32 |  1953 |  2447 |  6759 | 1.25 | 3.46 |

* Performance on FP16

| Batch Size | Beam Width | Precision | PyTorch <br/> Throughput (token/sec) | FT Decoder <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoder <br/> Speedup | FT Decoding <br/> Speedup |
|:----------:|:----------:|:---------:|:------------------------------------:|:---------------------------------------:|:----------------------------------------:|:------------------------:|:-------------------------:|
|   1 |   1 | FP16 |    78 |   267 |   967 | 3.40 | 12.39 |
|   1 |   4 | FP16 |    76 |   251 |   868 | 3.29 | 11.39 |
|   1 |  32 | FP16 |    70 |   217 |   635 | 3.10 | 9.07  |
|   8 |   1 | FP16 |   357 |  1242 |  4508 | 3.47 | 12.61 |
|   8 |   4 | FP16 |   336 |   886 |  3769 | 2.63 | 11.20 |
|   8 |  32 | FP16 |   265 |   575 |  1454 | 2.17 | 5.48  |
| 128 |   1 | FP16 |  3193 |  7396 | 19264 | 2.31 | 6.03  |
| 128 |   4 | FP16 |  2141 |  3141 | 12609 | 1.46 | 5.88  |


#### Beamsearch performance on T4 and PyTorch

* Perofrmance on FP32

| Batch Size | Beam Width | Precision | PyTorch <br/> Throughput (token/sec) | FT Decoder <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoder <br/> Speedup | FT Decoding <br/> Speedup |
|:----------:|:----------:|:---------:|:------------------------------------:|:---------------------------------------:|:----------------------------------------:|:------------------------:|:-------------------------:|
|   1 |   1 | FP32 |    62 |   179 |   566 | 2.85 | 8.99 |
|   1 |   4 | FP32 |    56 |   158 |   535 | 2.79 | 9.46 |
|   1 |  32 | FP32 |    47 |   144 |   312 | 3.06 | 6.62 |
|   8 |   1 | FP32 |   259 |   764 |  2418 | 2.94 | 9.30 |
|   8 |   4 | FP32 |   239 |   711 |  1914 | 2.97 | 7.99 |
|   8 |  32 | FP32 |   140 |   183 |   358 | 1.30 | 2.54 |
| 128 |   1 | FP32 |  1803 |  2885 |  6400 | 1.60 | 3.54 |
| 128 |   4 | FP32 |   690 |   836 |  2519 | 1.21 | 3.64 |

* Performance on FP16

| Batch Size | Beam Width | Precision | PyTorch <br/> Throughput (token/sec) | FT Decoder <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoder <br/> Speedup | FT Decoding <br/> Speedup |
|:----------:|:----------:|:---------:|:------------------------------------:|:---------------------------------------:|:----------------------------------------:|:------------------------:|:-------------------------:|
|   1 |   1 | FP16 |    60 |   176 |   774 | 2.93 | 12.81 |
|   1 |   4 | FP16 |    55 |   170 |   699 | 3.08 | 12.68 |
|   1 |  32 | FP16 |    46 |   147 |   468 | 3.17 | 10.06 |
|   8 |   1 | FP16 |   254 |   832 |  3389 | 3.27 | 13.32 |
|   8 |   4 | FP16 |   237 |   759 |  2981 | 3.19 | 12.53 |
|   8 |  32 | FP16 |   164 |   256 |   636 | 1.56 | 3.87  |
| 128 |   1 | FP16 |  2035 |  4000 | 10836 | 1.96 | 5.32  |
| 128 |   4 | FP16 |   977 |  1192 |  6369 | 1.21 | 6.51  |

<!-- #### Decoding performance on A100 and TensorFlow

* Performance of FP32

* Performance of FP16 -->
