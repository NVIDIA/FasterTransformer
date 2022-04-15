# ViT Quantization Toolkit

This tutorial contains the guidance for ViT Quantization Toolkit.

## 1. Download Pre-trained model (Google's Official Checkpoint)

* [Available models](https://console.cloud.google.com/storage/vit_models/): ViT-B_16(**85.8M**), R50+ViT-B_16(**97.96M**), ViT-B_32(**87.5M**), ViT-L_16(**303.4M**), ViT-L_32(**305.5M**), ViT-H_14(**630.8M**)
  * imagenet21k pre-train models
    * ViT-B_16, ViT-B_32, ViT-L_16, ViT-L_32, ViT-H_14
  * imagenet21k pre-train + imagenet2012 fine-tuned models
    * ViT-B_16-224, ViT-B_16, ViT-B_32, ViT-L_16-224, ViT-L_16, ViT-L_32
  * Hybrid Model([Resnet50](https://github.com/google-research/big_transfer) + Transformer)
    * R50-ViT-B_16
```
# imagenet21k pre-train
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
# imagenet21k pre-train + imagenet2012 fine-tuning
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/{MODEL_NAME}.npz
```
## 2 Usage

### 2.1 Environment setup

- Initialize submodule
    ```bash
    git submodule update --init
    ``` 

- Run the container.

    You can choose pytorch version you want. Here, we list some possible images:  
    - `nvcr.io/nvidia/pytorch:21.07-py3` contains the PyTorch 1.8.0 and python 3.8

- Install additional dependencies (not included by container)
    ```bash
    pip install timm==0.4.12 termcolor==1.1.0 pytorch-quantization ml_collections
    ```


### 2.2 Data preparation

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

### 2.3 Calibration & Evaluation

To calibrate and then evaluate a calibrated `Vision Transformer` on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> \
  --master_port 12345 main.py \
  --calib \
  --name vit \
  --pretrained_dir <checkpoint> \
  --data-path <imagenet-path> \
  --model_type <model-type> \
  --img_size <img-size> \
  --num-calib-batch <batch-number> \
  --calib-batchsz <batch-size> \
  --quant-mode <mode> \
  --calibrator <calibrator> \
  --percentile <percentile> \
  --calib-output-path <calib-output-path>
```

For example, to calibrate the `ViT-B_16` with a single GPU: (For calibration, we only support using one single GPU). You can use **sh calib.sh** for simplicity.

```bash
export DATA_DIR=Path to the dataset
export CKPT_DIR=Path to the ViT checkpoints
python -m torch.distributed.launch --nproc_per_node 1 \
    --master_port 12345 main.py \
    --calib \
    --name vit \
    --pretrained_dir $CKPT_DIR/ViT-B_16.npz \
    --data-path $DATA_DIR \
    --model_type ViT-B_16 \
    --img_size 384 \
    --num-calib-batch 20 \
    --calib-batchsz 8 \
    --quant-mode ft2 \
    --calibrator percentile \
    --percentile 99.99 \
    --calib-output-path $CKPT_DIR
```
### Difference between `--quant-mode 1` and `--quant-mode 2`
`--quant-mode 1` indicates that all GEMMs are quantized to be INT8-in-INT32-out, while `--quant-mode 2` means quantizating all GEMMs to be INT8-in-INT8-out. This is a speed-versus-accuracy trade-off: `mode=2` is faster in CUDA implementation but its accuracy is lower.
|name| resolution|Original Accuracy|PTQ(mode=1)|PTQ(mode=2)|
|:---:|:---:|:---:|:---:|:---:|
|ViT-B_16|384x384|83.97%|82.57%(-1.40%)|81.82%(-2.15%)|

In order to narrow the accuracy gap for `mode=2`, QAT is a reasonable choice.

### 2.4 Quantization Aware Training (QAT)

To run QAT finetuning with a calibrated `Vision Transformer`, run:
```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> \
    --master_port 12345 main.py \
    --train \
    --name vit \
    --pretrained_dir <calibrated-checkpoint> \
    --data-path <imagenet-path> \
    --model_type <model-type> \
    --quant-mode <mode> \
    --img_size <img-size> \
    --distill \
    --teacher <uncalibrated-checkpoint> \
    --output <qat-output-path> \
    --quant-mode <mode>\
    --batch-size <batch-size> \
    --num-epochs <num-of-epochs> \
    --qat-lr <learning-rate-of-QAT>
```
For example, to do QAT with `ViT-B_16` by 4 GPU on a single node for 5 epochs, run: (You can see **qat.sh** for reference.)

```bash
export DATA_DIR=Path to the dataset
export CKPT_DIR=Path to the ViT checkpoints
python -m torch.distributed.launch --nproc_per_node 4 \
    --master_port 12345 main.py \
    --train \
    --name vit \
    --pretrained_dir $CKPT_DIR/ViT-B_16_calib.pth \
    --data-path $DATA_DIR \
    --model_type ViT-B_16 \
    --quant-mode ft2 \
    --img_size 384 \
    --distill \
    --teacher $CKPT_DIR/ViT-B_16.npz \
    --output qat_output \
    --quant-mode ft2\
    --batch-size 32 \
    --num-epochs 5 \
    --qat-lr 1e-4
```
### Improvement brought by QAT (under mode=2)
As shown below, the accuracy gap is narrowed down from 2.15% to 0.65%.
|name| resolution|Original Accuracy|PTQ(mode=2)|QAT(mode=2)|
|:---:|:---:|:---:|:---:|:---:|
|ViT-B_16|384x384|83.97%|81.82%(-2.15%)|83.32(-0.65%)|