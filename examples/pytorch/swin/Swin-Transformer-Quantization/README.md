# Swin Transformer Quantization Toolkit

This folder contains the guidance for Swin Transformer Quantization Toolkit.

## Model Zoo

### Regular ImageNet-1K trained models

| name | resolution |acc@1 | acc@5 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|
| Swin-T | 224x224 | 81.2 | 95.5 | 28M | 4.5G | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)/[baidu](https://pan.baidu.com/s/156nWJy4Q28rDlrX-rRbI3w) |
| Swin-S | 224x224 | 83.2 | 96.2 | 50M | 8.7G | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth)/[baidu](https://pan.baidu.com/s/1KFjpj3Efey3LmtE1QqPeQg) |
| Swin-B | 224x224 | 83.5 | 96.5 | 88M | 15.4G | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth)/[baidu](https://pan.baidu.com/s/16bqCTEc70nC_isSsgBSaqQ) |
| Swin-B | 384x384 | 84.5 | 97.0 | 88M | 47.1G | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth)/[baidu](https://pan.baidu.com/s/1xT1cu740-ejW7htUdVLnmw) |

### ImageNet-22K pre-trained models

| name | resolution |acc@1 | acc@5 | #params | FLOPs | 22K model | 1K model |
|:---: |:---: |:---:|:---:|:---:|:---:|:---:|:---:|
| Swin-B | 224x224 | 85.2 | 97.5 | 88M | 15.4G | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth)/[baidu](https://pan.baidu.com/s/1y1Ec3UlrKSI8IMtEs-oBXA) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth)/[baidu](https://pan.baidu.com/s/1n_wNkcbRxVXit8r_KrfAVg) |
| Swin-B | 384x384 | 86.4 | 98.0 | 88M | 47.1G | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)/[baidu](https://pan.baidu.com/s/1vwJxnJcVqcLZAw9HaqiR6g) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth)/[baidu](https://pan.baidu.com/s/1caKTSdoLJYoi4WBcnmWuWg) |
| Swin-L | 224x224 | 86.3 | 97.9 | 197M | 34.5G | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth)/[baidu](https://pan.baidu.com/s/1pws3rOTFuOebBYP3h6Kx8w) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth)/[baidu](https://pan.baidu.com/s/1NkQApMWUhxBGjk1ne6VqBQ) |
| Swin-L | 384x384 | 87.3 | 98.2 | 197M | 103.9G | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth)/[baidu](https://pan.baidu.com/s/1sl7o_bJA143OD7UqSLAMoA) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth)/[baidu](https://pan.baidu.com/s/1X0FLHQyPOC6Kmv2CmgxJvA) |

Note: access code for `baidu` is `swin`.

## Usage

### Environment setup

- Initialize submodule
    ```bash
    git submodule update --init
    ``` 

- Run the container.

    You can choose pytorch version you want. Here, we list some possible images:  
    - `nvcr.io/nvidia/pytorch:21.07-py3` contains the PyTorch 1.8.0 and python 3.8

- Install additional dependencies (not included by container)
    ```bash
    pip install timm==0.4.12
    pip install termcolor==1.1.0
    ```


### Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
- To boost the slow speed when reading images from massive small files, we also support zipped ImageNet, which includes
  four files:
    - `train.zip`, `val.zip`: which store the zipped folder for train and validate splits.
    - `train_map.txt`, `val_map.txt`: which store the relative path in the corresponding zip file and ground truth
      label. Make sure the data folder looks like this:

  ```bash
  $ tree data
  data
  └── ImageNet-Zip
      ├── train_map.txt
      ├── train.zip
      ├── val_map.txt
      └── val.zip
  
  $ head -n 5 data/ImageNet-Zip/val_map.txt
  ILSVRC2012_val_00000001.JPEG	65
  ILSVRC2012_val_00000002.JPEG	970
  ILSVRC2012_val_00000003.JPEG	230
  ILSVRC2012_val_00000004.JPEG	809
  ILSVRC2012_val_00000005.JPEG	516
  
  $ head -n 5 data/ImageNet-Zip/train_map.txt
  n01440764/n01440764_10026.JPEG	0
  n01440764/n01440764_10027.JPEG	0
  n01440764/n01440764_10029.JPEG	0
  n01440764/n01440764_10040.JPEG	0
  n01440764/n01440764_10042.JPEG	0
  ```

### Calibration

To calibrate and then evaluate a calibrated `Swin Transformer` on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> \
  --master_port 12345 main.py \
  --calib \
  --cfg <config-file> \
  --resume <checkpoint> \
  --data-path <imagenet-path> \
  --num-calib-batch <batch-number> \
  --calib-batchsz <batch-size> \
  --int8-mode <mode>\
  --calib-output-path <output-path> \ 
```

For example, to calibrate the `Swin-T` with a single GPU: (For calibration, we only support using one single GPU). You can see **calib.sh** for reference.

```bash
python -m torch.distributed.launch --nproc_per_node 1 \
  --master_port 12345 main.py \
  --calib \
  --cfg SwinTransformer/configs/swin_tiny_patch4_window7_224.yaml \
  --resume swin_tiny_patch4_window7_224.pth \
  --data-path <imagenet-path> \ 
  --num-calib-batch 10 \
  --calib-batchsz 8\
  --int8-mode 1\
  --calib-output-path calib-checkpoint
```
### Difference between `--int8-mode 1` and `--int8-mode 2`
|name| resolution|Original Accuracy|PTQ(mode=1)|QAT(mode=1)|
|:---:|:---:|:---:|:---:|:---:|
|Swin-T|224x224|81.18%|80.75%(-0.43%)|81.00%(-0.18%)|
|Swin-S|224x224|83.21%|82.90%(-0.31%)|83.00%(-0.21%)|
|Swin-B|224x224|83.42%|83.10%(-0.32%)|83.42%(-0.00%)|
|Swin-B|384x384|84.47%|84.05%(-0.42%)|84.16%(-0.31%)|
|Swin-L|224x224|86.25%|83.53%(-2.72%)|86.12%(-0.13%)|
|Swin-L|384x384|87.25%|83.10%(-4.15%)|87.11%(-0.14%)|

For Swin-T/S/B, set `--int8-mode 1` suffices to get negligible accuracy loss for both PTQ/QAT. However, for Swin-L, `--int8-mode 1` cannot get a satisfactory result for PTQ accuracy. This is due to that `--int8-mode 1` means all GEMM outputs(INT32) are quantized to INT8, and in order to improve PTQ performance some GEMM output quantization have to be disabled. `--int8-mode 2` means quantization of `fc2` and `PatchMerge` outputs are disabled. The result is as follows:
|name| resolution|Original Accuracy|PTQ(mode=1)|PTQ(mode=2)|
|:---:|:---:|:---:|:---:|:---:|
|Swin-L|224x224|86.25%|83.53%(-2.72%)|85.93%(-0.32%)|
|Swin-L|384x384|87.25%|83.10%(-4.15%)|86.92%(-0.33%)|

### Evaluation a Calibrated model

To evaluate a pre-calibrated `Swin Transformer` on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> \
  --master_port 12345 main.py \
  --eval \
  --cfg <config-file> \
  --resume <calibrated-checkpoint> \
  --data-path <imagenet-path> \
  --int8-mode <mode> \
  --batch-size <batch-size>
```

For example, to evaluate the `Swin-T` with a single GPU. You can see **run.sh** for reference.

```bash
python -m torch.distributed.launch --nproc_per_node 1 \
  --master_port 12345 main.py \
  --eval \
  --cfg SwinTransformer/configs/swin_tiny_patch4_window7_224.yaml \
  --resume ./calib-checkpoint/swin_tiny_patch4_window7_224_calib.pth \
  --data-path <imagenet-path> \
  --int8-mode 1\
  --batch-size 128
```
### Quantization Aware Training (QAT)

To run QAT with `Swin Transformer`, run:
```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> \
    --master_port 12345 main.py \
    --train \
    --cfg <config-file> \
    --resume <calibrated-checkpoint> \
    --data-path <imagenet-path> \
    --quant-mode <mode> \
    --teacher <uncalibrated-checkpoint> \
    --output <qat-output-path> \
    --distill \
    --int8-mode <mode>\
    --batch-size <batch-size> \
    --num-epochs <num-of-epochs> \
    --qat-lr <learning-rate-of-QAT>
```
For example, to do QAT with `Swin Transformer` by 4 GPU on a single node for 5 epochs, run: (You can see **qat.sh** for reference.)

```bash
python -m torch.distributed.launch --nproc_per_node 4 \
    --master_port 12345 main.py \
    --train \
    --cfg SwinTransformer/configs/swin_tiny_patch4_window7_224.yaml \
    --resume ./calib-checkpoint/swin_tiny_patch4_window7_224_calib.pth \
    --data-path /data/datasets/ILSVRC2012 \
    --quant-mode ft2 \
    --teacher swin_tiny_patch4_window7_224.pth \
    --output qat-output \
    --distill \
    --int8-mode 1\
    --batch-size 128 \
    --num-epochs 5 \
    --qat-lr 1e-5
```