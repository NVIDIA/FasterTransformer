# Faster Swin-Transformer
The Faster Swin-Transformer contains the Swin-Transformer model, a state-of-the-art vision transformer model which was presented in [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030). The abstract of the paper is the following:

>This paper presents a new vision Transformer, called
Swin Transformer, that capably serves as a general-purpose
backbone for computer vision. Challenges in adapting
Transformer from language to vision arise from differences
between the two domains, such as large variations in the
scale of visual entities and the high resolution of pixels
in images compared to words in text. To address these
differences, we propose a hierarchical Transformer whose
representation is computed with Shifted windows. The
shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local
windows while also allowing for cross-window connection.
This hierarchical architecture has the flexibility to model
at various scales and has linear computational complexity
with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision
tasks, including image classification (87.3 top-1 accuracy
on ImageNet-1K) and dense prediction tasks such as object
detection (58.7 box AP and 51.1 mask AP on COCO testdev) and semantic segmentation (53.5 mIoU on ADE20K
val). Its performance surpasses the previous state-of-theart by a large margin of +2.7 box AP and +2.6 mask AP on
COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones.
The hierarchical design and the shifted window approach
also prove beneficial for all-MLP architectures. The code
and models are publicly available at https://github.com/microsoft/Swin-Transformer.

Our implementation is aligned with the official PyTorch implementation [Swin-Transformer Github](https://github.com/microsoft/Swin-Transformer).

## Table of Contents
- [Faster Swin-Transformer](#faster-swin-transformer)
  - [Table of Contents](#table-of-contents)
  - [Swin-Transformer Computation Flow](#swin-transformer-computation-flow)
  - [Demo](#demo)
    - [Requirements](#requirements)
    - [Setup](#setup)
    - [Run](#run)
      - [Run Swin-Transformer on C++](#run-swin-transformer-on-c)
      - [Run with PyTorch op](#run-with-pytorch-op)
      - [Run with TensorRT plugin](#run-with-tensorrt-plugin)
  - [Performance](#performance)
    - [Swin Performance on T4](#swin-performance-on-t4)
      - [Swin-V1 FP32](#swin-v1-fp32)
      - [Swin-V1 FP16](#swin-v1-fp16)
      - [Swin-V1 INT8](#swin-v1-int8)
      - [INT8 vs. FP16 speedup on Swin-V1 TINY/SMALL/BASE/LARGE:](#int8-vs-fp16-speedup-on-swin-v1-tinysmallbaselarge)
      - [Swin-V2 FP32](#swin-v2-fp32)
      - [Swin-V2 FP16](#swin-v2-fp16)
      - [Swin-V2 INT8](#swin-v2-int8)
      - [INT8 vs. FP16 speedup of Swin-V2:](#int8-vs-fp16-speedup-of-swin-v2)
    - [Swin Performance on A10](#swin-performance-on-a10)
      - [Swin-V1 TF32](#swin-v1-tf32)
      - [Swin-V1 FP16](#swin-v1-fp16-1)
      - [Swin-V1 INT8](#swin-v1-int8-1)
      - [INT8 vs. FP16 speedup on Swin-V1 TINY/SMALL/BASE/LARGE:](#int8-vs-fp16-speedup-on-swin-v1-tinysmallbaselarge-1)
      - [Swin-V2 TF32](#swin-v2-tf32)
      - [Swin-V2 FP16](#swin-v2-fp16-1)
      - [Swin-V2 INT8](#swin-v2-int8-1)
      - [INT8 vs. FP16 speedup of Swin-V2:](#int8-vs-fp16-speedup-of-swin-v2-1)
    - [Swin Performance on A100](#swin-performance-on-a100)
      - [Swin-V1 TF32](#swin-v1-tf32-1)
      - [Swin-V1 FP16](#swin-v1-fp16-2)
      - [Swin-V1 INT8](#swin-v1-int8-2)
      - [INT8 vs. FP16 speedup on Swin-V1 TINY/SMALL/BASE/LARGE:](#int8-vs-fp16-speedup-on-swin-v1-tinysmallbaselarge-2)
      - [Swin-V2 TF32](#swin-v2-tf32-1)
      - [Swin-V2 FP16](#swin-v2-fp16-2)
      - [Swin-V2 INT8](#swin-v2-int8-2)
      - [INT8 vs. FP16 speedup of Swin-V2:](#int8-vs-fp16-speedup-of-swin-v2-2)

## Swin-Transformer Computation Flow
<div align=center><img width=80% src ="images/FP-swin-flowchart.png"/></div>
<div align=center>Fig. 1 Flowchart of FP16/FP32 Swin-Transformer.</div>

<div align=center><img width=80% src ="images/INT8-swin-flowchart.png"/></div>
<div align=center>Fig. 2 Flowchart of INT8 Swin-Transformer (with fused MHA and int8-mode=1).</div>

## Demo

In this demo, you can run Faster Swin-Transformer as a C++ program.

### Requirements

- CMake >= 3.13 for PyTorch
- CUDA 11.0 or newer version
- NCCL 2.10 or newer version
- Python 3 is recommended because some features are not supported in python 2
- PyTorch: Verify on 1.10.0, >= 1.5.0 should work.

Recommend to use image `nvcr.io/nvidia/pytorch:22.09-py3`.  

### Setup

1. Start the docker container, ensure mounting the project directory into it. For example:
    ```bash
    docker run \
        -it \
        --shm-size 5g \
        --rm \
        --gpus=all \
        -v {YOUR_FASTER_TRANSFORMER_PROJECT_DIR_ON_HOST}:/workspace/FasterTransformer \
        --workdir /workspace/FasterTransformer \
        nvcr.io/nvidia/pytorch:22.09-py3 bash
    export WORKSPACE = /workspace/FasterTransformer
    ```

Here, we use `nvcr.io/nvidia/pytorch:22.09-py3`, you can also switch it to another CUDA-enabled PyTorch containers, but need to comply with the previous requirements.

2. Build the FasterTransformer with C++:
    ```bash
    cd $WORKSPACE
    git submodule update --init
    mkdir -p build
    cd build
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_TRT=ON ..
    make -j12
    ```
Note: **xx** is the compute capability of your GPU. For example, 60 (P40) or 61 (P4) or 70 (V100) or 75(T4) or 80 (A100).

### Run  

#### Run Swin-Transformer on C++

Firstly we use `./bin/swin_gemm` as the tool to search the best GEMM configuration. And then run `./bin/swin_example` or `./bin/swin_int8_example`.\
For FP32/FP16 swin v1 & swin v2
```bash
# data_type=0 indicates FP32, data_type=1 indicates FP16
# model_type={0,1,2,3}
#   0: tiny
#   1: small
#   2: base
#   3: large
./bin/swin_gemm <batch_size> <image_width> <window_width> <head_number of the first block> <size_per_head> <data_type> <is_use_int8> 
# ./bin/swin_example version(1/2) data_type(0/1, fp32/fp16) model_type(0-3) window_size(7/8/12/16/24) img_size(192/224/256/384) batch_size
./bin/swin_example <version[1-2]> <data_type> <model_type[0-3]> <window_size> <img_size> <batch_size>
```
For INT8 swin v1 & swin v2
```bash
# For INT8 cases, should set `is_use_int8=1`.
# `datatype` can be set either 0 or 1 because it does not have effect now.
# model_type={0,1,2,3}
#   0: tiny
#   1: small
#   2: base
#   3: large
./bin/swin_gemm <batch_size> <image_width> <window_width> <head_number of the first block> <size_per_head> <data_type> <is_use_int8>
# ./bin/swin_int8_example version(1/2) int8_mode(1/2) model_type(0-3) window_size(7/8/12/16/24) img_size(192/224/256/384) batch_size
./bin/swin_int8_example <version[1-2]> <int8_mode> <model_type[0-3]> <window_size> <img_size> <batch_size>
```

Take swin-TINY with batch=32 as an example:
```bash
# Run Swin-Transformer(TINY) version 2 under FP32 on C++:
./bin/swin_gemm 32 256 8 3 32 0 0
./bin/swin_example 2 0 0 8 256 32 

# Run Swin-Transformer(TINY) version 2 under FP16 on C++
./bin/swin_gemm 32 256 8 3 32 1 0
./bin/swin_example 2 1 0 8 256 32 

# Run Swin-Transformer(TINY) version 2 under INT8 on C++
./bin/swin_gemm 32 256 8 3 32 0 1
./bin/swin_int8_example 2 1 0 8 256 32
```

#### Run with PyTorch op
Download checkpoint
```bash
cd $WORKSPACE/examples/pytorch/swin/Swin-Transformer-Quantization
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
wget https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth
```

**Run FP16/FP32 pytorch op of swin v1 and swin v2** 
```bash
cd $WORKSPACE/examples/pytorch/swin
pip install timm==0.4.12
pip install termcolor==1.1.0

#test with random input
bash -x run_test_v1.sh <batch_size> ##profile of FP16/FP32 model with version 1
bash -x run_test_v2.sh <batch_size> ##profile of FP16/FP32 model with version 2

#test with ImageNet dataset
bash -x run_test_fp32_accuracy.sh <path_to_imagenet_dataset>
bash -x run_test_fp16_accuracy.sh <path_to_imagenet_dataset>
```

**Run INT8 pytorch op of swin v1** 
1. Get calibrated checkpoint

    Refer to [Guide of Swin-Transformer Quantization Toolkit](../examples/pytorch/swin/Swin-Transformer-Quantization/README.md#usage) when installing dependencies and setting up datasets
```bash
cd $WORKSPACE/examples/pytorch/swin/Swin-Transformer-Quantization
python -m torch.distributed.launch --nproc_per_node 1 \
  --master_port 12345 main.py \
  --calib \
  --cfg SwinTransformer/configs/swin/swin_tiny_patch4_window7_224.yaml \
  --resume swin_tiny_patch4_window7_224.pth \
  --data-path <imagenet-path> \
  --num-calib-batch 10 \
  --calib-batchsz 8 \
  --int8-mode 1 \
  --version 1\
  --calib-output-path calib-checkpoint

```
**NOTE**: If you **ONLY** want to use PTQ instead of QAT: when calibrating TINY/SMALL/BASE model, `--int8-mode 1` suffices. When calibrating LARGE model, we have to specify `--int8-mode 2` instead of `--int8-mode 1`. The reason is, `Swin-L` is much harder to quantize, and we have to disable more quantization nodes in order to obtain satisfactory PTQ accuracy results. 

(When `int8_mode=1`, all GEMMs are INT8-in-INT8-out.
When `int8_mode=2`, GEMM of all `fc2` layers and `patchMerge` are relaxed to INT8-in-INT32-out, while other GEMMs keep INT8-I/O. )

If you want to insists on using `--int8-mode 1` for LARGE model (because speed of mode=1 is much faster), we recommend using QAT to finetune parameters of LARGE checkpoint.

`--int8-mode 2` is only developed for swin-v1-LARGE, when it comes to swin-v2, the CUDA implementation only provides `--int8-mode 1` and setting `--int8-mode 2` does not change the behavior.

2. Run test
```bash
cd $WORKSPACE/examples/pytorch/swin
pip install timm==0.4.12
pip install termcolor==1.1.0

bash -x run_test_v1_int8.sh <batch_size> ##profile of swin-v1 INT8 model
bash -x run_test_v1_int8_accuracy.sh <batch_size> ##test accuracy of swin-v1 INT8 model

bash -x run_test_v2_int8.sh <batch_size> ##profile of swin-v2 INT8 model
bash -x run_test_v2_int8_accuracy.sh <batch_size> ##test accuracy of swin-v2 INT8 model
```
Note: When testing PTQ accuracy for INT8 swin-v1-LARGE, we have to specify `--int8-mode 2` instead of `--int8-mode 1` in **run_test_int8.sh**.

However, if you have finetuned a Swin-LARGE using QAT and `--int8-mode 1`, then you can do inference with `--int8-mode 1` too. In a word, consistency have to be ensured.

3. Accuracy loss of INT8 Post-Training-Quantization(PTQ) compared with FP16

**swin-v1**

|  name  |   pretrain   | resolution | acc@1  | acc@5  |    acc@1(I)    |    acc@5(I)    |
| :----: | :----------: | :--------: | :----: | :----: | :------------: | :------------: |
| Swin-T | ImageNet-1K  |  224x224   | 81.182 | 95.522 | 80.748(-0.434) | 95.328(-0.194) |
| Swin-S | ImageNet-1K  |  224x224   | 83.214 | 96.242 | 82.904(-0.288) | 96.196(-0.036) |
| Swin-B | ImageNet-1K  |  224x224   | 83.424 | 96.446 | 83.116(-0.354) | 96.322(-0.144) |
| Swin-B | ImageNet-1K  |  384x384   | 84.474 | 96.956 | 84.034(-0.440) | 96.782(-0.174) |
| Swin-L | ImageNet-22K |  224x224   | 86.246 | 97.876 | 85.892(-0.354) | 97.822(-0.054) |
| Swin-L | ImageNet-22K |  384x384   | 87.246 | 98.248 | 86.916(-0.330) | 98.174(-0.074) |

**swin-v2**

|   name   |   pretrain   | resolution | window | acc@1 |   acc@1(I)   |
| :------: | :----------: | :--------: | :----: | :---: | :----------: |
| SwinV2-T | ImageNet-1K  |  256x256   |  8x8   | 81.8  | 81.15(-0.65) |
| SwinV2-S | ImageNet-1K  |  256x256   |  8x8   | 83.7  | 83.48(-0.22) |
| SwinV2-B | ImageNet-1K  |  256x256   |  8x8   | 84.2  | 84.17(-0.03) |
| SwinV2-T | ImageNet-1K  |  256x256   | 16x16  | 82.8  | 82.22(-0.58) |
| SwinV2-S | ImageNet-1K  |  256x256   | 16x16  | 84.1  | 83.90(-0.30) |
| SwinV2-B | ImageNet-1K  |  256x256   | 16x16  | 84.6  | 84.53(-0.07) |
| SwinV2-B | ImageNet-22K |  256x256   | 16x16  | 86.2  | 85.61(-0.59) |
| SwinV2-B | ImageNet-22K |  384x384   | 24x24  | 87.1  | 86.46(-0.64) |
| SwinV2-L | ImageNet-22K |  256x256   | 16x16  | 86.9  | 86.35(-0.55) |
| SwinV2-L | ImageNet-22K |  384x384   | 24x24  | 87.6  | 86.90(-0.70) |


#### Run with TensorRT plugin
**FP16/FP32 TensorRT plugin of swin v1 and swin v2** 
```bash
cd $WORKSPACE/examples/tensorrt/swin

#build FP32/FP16 trt engine
sh run_builder_fp32_v1.sh # build engine of swin v1
sh run_builder_fp32_v2.sh # build engine of swin v2

sh run_builder_fp16_v1.sh # build engine of swin v1
sh run_builder_fp16_v2.sh # build engine of swin v2

#infer
sh run_infer_fp32.sh <version> <batch_size>
sh run_infer_fp16.sh <version> <batch_size>
```

**INT8 TensorRT plugin of swin v1 and swin v2** 
```bash
cd $WORKSPACE/examples/tensorrt/swin

#INT8 engine build
sh run_builder_int8_v1.sh # build engine of swin v1
sh run_builder_int8_v2.sh # build engine of swin v2

#infer
sh run_infer_int8.sh <version> <batch_size>
```

## Performance  

Hardware settings:
* T4 (with mclk 5000MHz, pclk 1590MHz) with  Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz
* A100 (with mclk 1215, pclk 1410MHz) with  Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz

Software settings:
* CUDA 11.4

We here compared the performance between Swin-Transformer and FT Swin-Transformer on T4 & A100. Here we used Swin-TINY as an example, and the hyper-parameters of the model are:

* head_num = {3,6,12,24}
* size_per_head = 32
* num_of_blocks = {2,2,6,2}

### Swin Performance on T4
Here, `torch.jit.trace` means using tracing to convert Torch model to TorchScript model and then profile its performance. 
#### Swin-V1 FP32
| Batch_size | torch.jit.trace |  cpp   | speedup | trt plugin | speedup | torch op | speedup |
| :--------: | :-------------: | :----: | :-----: | :--------: | :-----: | :------: | :-----: |
|     1      |      11.70      |  4.49  |  2.61   |    4.64    |  2.52   |   4.68   |  2.50   |
|     8      |      36.89      | 27.39  |  1.35   |   29.23    |  1.26   |  29.04   |  1.27   |
|     16     |      72.26      | 52.55  |  1.38   |   57.32    |  1.26   |  56.90   |  1.27   |
|     32     |     140.80      | 102.15 |  1.38   |   111.31   |  1.27   |  111.33  |  1.26   |

#### Swin-V1 FP16
| Batch_size | torch.jit.trace |  cpp  | speedup | trt plugin | speedup | torch op | speedup |
| :--------: | :-------------: | :---: | :-----: | :--------: | :-----: | :------: | :-----: |
|     1      |      10.49      | 1.43  |  7.34   |    1.59    |  6.60   |   2.12   |  4.95   |
|     8      |      18.73      | 6.60  |  2.84   |    7.47    |  2.51   |   7.03   |  2.66   |
|     16     |      37.60      | 13.12 |  2.87   |   14.84    |  2.53   |  13.87   |  2.71   |
|     32     |      74.85      | 25.25 |  2.96   |   28.74    |  2.60   |  27.64   |  2.71   |

#### Swin-V1 INT8
| Batch_size | torch.jit.trace |  cpp  | speedup(vs FP16) | trt plugin | speedup(vs FP16) | torch op | speedup(vs FP16) |
| :--------: | :-------------: | :---: | :--------------: | :--------: | :--------------: | :------: | :--------------: |
|     1      |                 | 1.18  |       1.21       |    1.40    |       1.14       |   1.59   |       1.33       |
|     8      |                 | 4.26  |       1.55       |    5.26    |       1.42       |   4.87   |       1.44       |
|     16     |                 | 8.57  |       1.53       |   10.58    |       1.40       |   9.80   |       1.42       |
|     32     |                 | 16.81 |       1.50       |   20.48    |       1.40       |  19.36   |       1.43       |

#### INT8 vs. FP16 speedup on Swin-V1 TINY/SMALL/BASE/LARGE:
| Batch_size | TINY (FP16) | TINY (INT8) | Speedup | SMALL (FP16) | SMALL (INT8) | Speedup | BASE (FP16) | BASE (INT8) | Speedup | LARGE (FP16) | LARGE (INT8) | Speedup |
| :--------: | :---------: | :---------: | :-----: | :----------: | :----------: | :-----: | :---------: | :---------: | :-----: | :----------: | :----------: | :-----: |
|     1      |    1.43     |    1.18     |  1.21   |     2.90     |     2.15     |  1.34   |    3.41     |    2.60     |  1.31   |     5.98     |     3.67     |  1.63   |
|     8      |    6.60     |    4.26     |  1.55   |    11.02     |     6.63     |  1.66   |    16.62    |    9.78     |  1.70   |    30.81     |    16.95     |  1.82   |
|     16     |    13.12    |    8.57     |  1.53   |    21.85     |    13.59     |  1.61   |    31.03    |    19.12    |  1.62   |    59.76     |    33.86     |  1.76   |
|     32     |    25.25    |    16.81    |  1.50   |    42.41     |    26.87     |  1.58   |    60.99    |    38.21    |  1.60   |    115.07    |    70.23     |  1.64   |

#### Swin-V2 FP32
| Batch_size | torch.jit.trace |  cpp   | speedup | trt plugin | speedup | torch op | speedup |
| :--------: | :-------------: | :----: | :-----: | :--------: | :-----: | :------: | :-----: |
|     1      |      12.33      |  5.47  |  2.25   |    5.69    |  2.17   |   5.70   |  2.16   |
|     8      |      49.57      | 37.17  |  1.33   |   39.01    |  1.27   |  38.73   |  1.28   |
|     16     |      97.15      | 72.03  |  1.35   |   75.50    |  1.29   |  75.60   |  1.29   |
|     32     |     193.09      | 141.31 |  1.37   |   148.65   |  1.30   |  150.63  |  1.28   |

#### Swin-V2 FP16
| Batch_size | torch.jit.trace |  cpp  | speedup | trt plugin | speedup | torch op | speedup |
| :--------: | :-------------: | :---: | :-----: | :--------: | :-----: | :------: | :-----: |
|     1      |      16.38      | 1.66  |  9.86   |    1.86    |  8.80   |   2.09   |  7.84   |
|     8      |      24.51      | 9.10  |  2.69   |   10.20    |  2.40   |   9.60   |  2.55   |
|     16     |      48.93      | 17.64 |  2.77   |   19.75    |  2.48   |  19.08   |  2.56   |
|     32     |      98.46      | 34.67 |  2.84   |   39.58    |  2.48   |  38.76   |  2.54   |

#### Swin-V2 INT8
| Batch_size | torch.jit.trace |  cpp  | speedup(vs FP16) | trt plugin | speedup(vs FP16) | torch op | speedup(vs FP16) |
| :--------: | :-------------: | :---: | :--------------: | :--------: | :--------------: | :------: | :--------------: |
|     1      |                 | 1.32  |       1.25       |    1.52    |       1.22       |   1.61   |       1.30       |
|     8      |                 | 5.81  |       1.57       |    7.12    |       1.43       |   6.61   |       1.45       |
|     16     |                 | 11.84 |       1.49       |   14.35    |       1.38       |  13.36   |       1.43       |
|     32     |                 | 23.59 |       1.47       |   28.23    |       1.40       |  26.52   |       1.46       |

#### INT8 vs. FP16 speedup of Swin-V2:
| Model | window_size | input_size |  FP16  |  INT8  | Speedup |
| :---: | :---------: | :--------: | :----: | :----: | :-----: |
| TINY  |      8      |    256     | 34.67  | 23.59  |  1.47   |
| SMALL |      8      |    256     | 57.81  | 38.44  |  1.50   |
| BASE  |      8      |    256     | 82.88  | 54.88  |  1.51   |
| TINY  |     16      |    256     | 37.84  | 26.58  |  1.42   |
| SMALL |     16      |    256     | 63.65  | 43.11  |  1.48   |
| BASE  |     16      |    256     | 90.40  | 60.82  |  1.49   |
| BASE  |     24      |    384     |   -    | 596.62 |    -    |
| LARGE |     16      |    256     | 169.22 | 108.34 |  1.56   |
| LARGE |     24      |    384     |   -    |   -    |    -    |

### Swin Performance on A10
#### Swin-V1 TF32
On chips with Ampere architectures (like A30, A100), user can use `export NVIDIA_TF32_OVERRIDE=1` to enforce the program run under TF32, otherwise FP32 GEMM is used by default, which is much slower.
| Batch_size | torch.jit.trace |  cpp  | speedup | trt plugin | speedup | torch op | speedup |
| :--------: | :-------------: | :---: | :-----: | :--------: | :-----: | :------: | :-----: |
|     1      |      4.97       | 1.72  |  2.89   |    1.69    |  2.94   |   1.76   |  2.82   |
|     8      |      13.20      | 9.17  |  1.44   |    9.21    |  1.43   |   9.30   |  1.42   |
|     16     |      25.80      | 17.64 |  1.46   |   17.72    |  1.45   |  17.80   |  1.45   |
|     32     |      50.41      | 33.90 |  1.49   |   34.06    |  1.48   |  34.30   |  1.47   |

#### Swin-V1 FP16
| Batch_size | torch.jit.trace |  cpp  | speedup | trt plugin | speedup | torch op | speedup |
| :--------: | :-------------: | :---: | :-----: | :--------: | :-----: | :------: | :-----: |
|     1      |      5.04       | 0.92  |  5.48   |    0.97    |  5.20   |   1.39   |  3.63   |
|     8      |      7.53       | 3.20  |  2.24   |    3.58    |  2.10   |   3.28   |  2.30   |
|     16     |      14.80      | 6.10  |  2.43   |    6.73    |  2.20   |   6.40   |  2.31   |
|     32     |      29.27      | 11.89 |  2.46   |   12.99    |  2.25   |  12.31   |  2.38   |

#### Swin-V1 INT8
| Batch_size | torch.jit.trace |  cpp  | speedup(vs FP16) | trt plugin | speedup(vs FP16) | torch op | speedup(vs FP16) |
| :--------: | :-------------: | :---: | :--------------: | :--------: | :--------------: | :------: | :--------------: |
|     1      |                 | 1.01  |       0.91       |    1.22    |       0.80       |   1.02   |       1.36       |
|     8      |                 | 2.15  |       1.49       |    2.62    |       1.36       |   2.36   |       1.39       |
|     16     |                 | 3.96  |       1.54       |    4.74    |       1.42       |   4.37   |       1.46       |
|     32     |                 | 7.40  |       1.61       |    9.09    |       1.43       |   8.49   |       1.45       |

#### INT8 vs. FP16 speedup on Swin-V1 TINY/SMALL/BASE/LARGE:
| Batch_size | TINY (FP16) | TINY (INT8) | Speedup | SMALL (FP16) | SMALL (INT8) | Speedup | BASE (FP16) | BASE (INT8) | Speedup | LARGE (FP16) | LARGE (INT8) | Speedup |
| :--------: | :---------: | :---------: | :-----: | :----------: | :----------: | :-----: | :---------: | :---------: | :-----: | :----------: | :----------: | :-----: |
|     1      |    0.91     |    1.01     |  0.91   |     1.67     |     1.90     |  0.88   |    1.86     |    2.17     |  0.86   |     2.93     |     2.74     |  1.07   |
|     8      |    3.20     |    2.15     |  1.49   |     5.38     |     3.57     |  1.51   |    7.54     |    4.62     |  1.63   |    14.56     |     7.89     |  1.85   |
|     16     |    6.10     |    3.96     |  1.54   |    10.32     |     6.41     |  1.61   |    15.10    |    8.73     |  1.73   |    27.94     |    15.25     |  1.83   |
|     32     |    11.89    |    7.40     |  1.61   |    19.90     |    11.99     |  1.66   |    28.60    |    16.94    |  1.69   |    52.56     |    29.82     |  1.76   |
|     64     |    23.20    |    15.28    |  1.52   |    39.33     |    24.98     |  1.57   |    56.91    |    35.59    |  1.60   |    106.81    |    64.29     |  1.66   |
#### Swin-V2 TF32
On chips with Ampere architectures (like A30, A100), user can use `export NVIDIA_TF32_OVERRIDE=1` to enforce the program run under TF32, otherwise FP32 GEMM is used by default, which is much slower.
| Batch_size | torch.jit.trace |  cpp  | speedup | trt plugin | speedup | torch op | speedup |
| :--------: | :-------------: | :---: | :-----: | :--------: | :-----: | :------: | :-----: |
|     1      |      6.35       | 1.94  |  3.27   |    1.93    |  3.29   |   1.96   |  3.24   |
|     8      |      16.50      | 11.20 |  1.47   |   11.34    |  1.46   |  11.26   |  1.47   |
|     16     |      32.66      | 22.13 |  1.48   |   22.34    |  1.46   |  22.43   |  1.46   |
|     32     |      63.60      | 43.35 |  1.47   |   43.89    |  1.45   |  44.00   |  1.45   |

#### Swin-V2 FP16
| Batch_size | torch.jit.trace |  cpp  | speedup | trt plugin | speedup | torch op | speedup |
| :--------: | :-------------: | :---: | :-----: | :--------: | :-----: | :------: | :-----: |
|     1      |      6.14       | 1.00  |  6.14   |    1.10    |  5.58   |   1.38   |  4.45   |
|     8      |      10.83      | 4.17  |  2.60   |    4.65    |  2.33   |   4.42   |  2.45   |
|     16     |      20.88      | 8.06  |  2.59   |    8.86    |  2.36   |   8.34   |  2.50   |
|     32     |      40.93      | 15.99 |  2.56   |   17.48    |  2.34   |  16.04   |  2.55   |

#### Swin-V2 INT8
| Batch_size | torch.jit.trace |  cpp  | speedup(vs FP16) | trt plugin | speedup(vs FP16) | torch op | speedup(vs FP16) |
| :--------: | :-------------: | :---: | :--------------: | :--------: | :--------------: | :------: | :--------------: |
|     1      |                 | 1.10  |       0.91       |    1.28    |       0.86       |   1.10   |       1.25       |
|     8      |                 | 2.76  |       1.51       |    3.38    |       1.38       |   3.04   |       1.45       |
|     16     |                 | 5.29  |       1.52       |    6.34    |       1.40       |   5.75   |       1.45       |
|     32     |                 | 10.42 |       1.54       |   12.17    |       1.44       |  11.37   |       1.41       |

#### INT8 vs. FP16 speedup of Swin-V2:
| Model | window_size | input_size | FP16  |  INT8  | Speedup |
| :---: | :---------: | :--------: | :---: | :----: | :-----: |
| TINY  |      8      |    256     | 15.99 | 10.42  |  1.54   |
| SMALL |      8      |    256     | 26.84 | 16.65  |  1.61   |
| BASE  |      8      |    256     | 37.59 | 23.29  |  1.61   |
| TINY  |     16      |    256     | 17.46 | 11.85  |  1.54   |
| SMALL |     16      |    256     | 29.34 | 19.01  |  1.54   |
| BASE  |     16      |    256     | 40.84 | 26.31  |  1.55   |
| BASE  |     24      |    384     |   -   | 225.99 |    -    |
| LARGE |     16      |    256     | 74.66 | 45.90  |  1.63   |
| LARGE |     24      |    384     |   -   |   -    |    -    |


### Swin Performance on A100
Here, `torch.jit.trace` means using tracing to convert Torch model to TorchScript model and then profile its performance. 
#### Swin-V1 TF32
On chips with Ampere architectures (like A30, A100), user can use `export NVIDIA_TF32_OVERRIDE=1` to enforce the program run under TF32, otherwise FP32 GEMM is used by default, which is much slower.
| Batch_size | torch.jit.trace |  cpp  | speedup | trt plugin | speedup | torch op | speedup |
| :--------: | :-------------: | :---: | :-----: | :--------: | :-----: | :------: | :-----: |
|     1      |      4.90       | 1.59  |  3.08   |    1.73    |  2.83   |   1.77   |  2.77   |
|     8      |      7.79       | 5.52  |  1.41   |    5.57    |  1.40   |   5.53   |  1.41   |
|     16     |      13.40      | 9.81  |  1.37   |    9.47    |  1.41   |   9.69   |  1.38   |
|     32     |      25.25      | 18.23 |  1.39   |   18.25    |  1.38   |  19.09   |  1.32   |

#### Swin-V1 FP16
| Batch_size | torch.jit.trace |  cpp  | speedup | trt plugin | speedup | torch op | speedup |
| :--------: | :-------------: | :---: | :-----: | :--------: | :-----: | :------: | :-----: |
|     1      |      5.01       | 1.23  |  4.07   |    1.47    |  3.40   |   1.65   |  3.04   |
|     8      |      5.47       | 1.94  |  2.81   |    2.79    |  1.96   |   1.92   |  2.85   |
|     16     |      8.97       | 3.17  |  2.83   |    3.11    |  2.88   |   2.87   |  3.13   |
|     32     |      16.62      | 5.46  |  3.04   |    6.24    |  2.66   |   5.73   |  2.90   |

#### Swin-V1 INT8
| Batch_size | torch.jit.trace |  cpp  | speedup(vs FP16) | trt plugin | speedup(vs FP16) | torch op | speedup(vs FP16) |
| :--------: | :-------------: | :---: | :--------------: | :--------: | :--------------: | :------: | :--------------: |
|     1      |                 | 1.07  |       1.14       |    1.45    |       1.01       |   1.20   |       1.38       |
|     8      |                 | 1.62  |       1.20       |    1.88    |       1.48       |   1.85   |       1.04       |
|     16     |                 | 2.63  |       1.21       |    2.80    |       1.11       |   2.61   |       1.10       |
|     32     |                 | 4.46  |       1.23       |    5.05    |       1.24       |   3.88   |       1.48       |

#### INT8 vs. FP16 speedup on Swin-V1 TINY/SMALL/BASE/LARGE:
| Batch_size | TINY (FP16) | TINY (INT8) | Speedup | SMALL (FP16) | SMALL (INT8) | Speedup | BASE (FP16) | BASE (INT8) | Speedup | LARGE (FP16) | LARGE (INT8) | Speedup |
| :--------: | :---------: | :---------: | :-----: | :----------: | :----------: | :-----: | :---------: | :---------: | :-----: | :----------: | :----------: | :-----: |
|     1      |    1.23     |    1.07     |  1.14   |     1.88     |     2.06     |  0.91   |    2.00     |    3.74     |  0.53   |     2.17     |     2.96     |  0.73   |
|     8      |    1.94     |    1.62     |  1.20   |     3.28     |     3.21     |  1.02   |    3.98     |    3.96     |  1.01   |     6.15     |     4.58     |  1.34   |
|     16     |    3.17     |    2.63     |  1.21   |     5.40     |     4.39     |  1.23   |    7.30     |    5.33     |  1.37   |    11.76     |     8.11     |  1.45   |
|     32     |    5.46     |    4.46     |  1.23   |     8.83     |     6.56     |  1.34   |    11.07    |    9.08     |  1.22   |    20.88     |    16.28     |  1.28   |
|     64     |    10.35    |    8.17     |  1.27   |    17.15     |    13.22     |  1.29   |    21.93    |    16.39    |  1.34   |    41.78     |    29.29     |  1.43   |

#### Swin-V2 TF32
On chips with Ampere architectures (like A30, A100), user can use `export NVIDIA_TF32_OVERRIDE=1` to enforce the program run under TF32, otherwise FP32 GEMM is used by default, which is much slower.
| Batch_size | torch.jit.trace |  cpp  | speedup | trt plugin | speedup | torch op | speedup |
| :--------: | :-------------: | :---: | :-----: | :--------: | :-----: | :------: | :-----: |
|     1      |      6.38       | 1.74  |  3.67   |    1.48    |  4.31   |   1.78   |  3.58   |
|     8      |      8.26       | 7.02  |  1.18   |    4.96    |  1.66   |   5.57   |  1.48   |
|     16     |      15.24      | 10.78 |  1.41   |    8.85    |  1.72   |  10.42   |  1.46   |
|     32     |      28.30      | 19.72 |  1.44   |   16.84    |  1.68   |  19.43   |  1.46   |

#### Swin-V2 FP16
| Batch_size | torch.jit.trace |  cpp  | speedup | trt plugin | speedup | torch op | speedup |
| :--------: | :-------------: | :---: | :-----: | :--------: | :-----: | :------: | :-----: |
|     1      |      6.33       | 1.05  |  6.03   |    1.34    |  4.72   |   1.35   |  4.69   |
|     8      |      7.64       | 2.40  |  3.18   |    2.67    |  2.86   |   2.40   |  3.18   |
|     16     |      12.91      | 3.43  |  3.76   |    4.36    |  2.96   |   4.12   |  3.13   |
|     32     |      24.84      | 7.29  |  3.40   |    8.46    |  2.94   |   7.94   |  3.12   |

#### Swin-V2 INT8
| Batch_size | torch.jit.trace |  cpp  | speedup(vs FP16) | trt plugin | speedup(vs FP16) | torch op | speedup(vs FP16) |
| :--------: | :-------------: | :---: | :--------------: | :--------: | :--------------: | :------: | :--------------: |
|     1      |                 | 1.15  |       0.91       |    1.55    |       0.86       |   1.21   |       1.11       |
|     8      |                 | 1.95  |       1.23       |    2.54    |       1.05       |   2.27   |       1.06       |
|     16     |                 | 2.98  |       1.15       |    3.81    |       1.14       |   3.41   |       1.21       |
|     32     |                 | 5.17  |       1.41       |    6.65    |       1.27       |   6.22   |       1.28       |

#### INT8 vs. FP16 speedup of Swin-V2:
| Model | window_size | input_size | FP16  |  INT8  | Speedup |
| :---: | :---------: | :--------: | :---: | :----: | :-----: |
| TINY  |      8      |    256     | 7.29  |  5.17  |  1.41   |
| SMALL |      8      |    256     | 11.55 |  9.63  |  1.20   |
| BASE  |      8      |    256     | 16.78 | 12.74  |  1.31   |
| TINY  |     16      |    256     | 8.64  |  7.52  |  1.15   |
| SMALL |     16      |    256     | 13.86 | 11.66  |  1.19   |
| BASE  |     16      |    256     | 19.27 | 14.97  |  1.29   |
| BASE  |     24      |    384     |   -   | 135.38 |    -    |
| LARGE |     16      |    256     | 31.37 | 24.20  |  1.30   |
| LARGE |     24      |    384     |   -   |   -    |    -    |