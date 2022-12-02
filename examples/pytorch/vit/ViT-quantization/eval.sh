python -m torch.distributed.launch --nproc_per_node 1 \
    --master_port 12345 eval_engine.py \
    --name vit \
    --engine ../../../tensorrt/vit/ViTEngine_16_12_12_3072_768_32_384_577_1 \
    --pretrained_dir /workspace/checkpoint/ViT-B_16_ft2_99.99_81.948.pth \
    --data-path /data/datasets/ILSVRC2012/ \
    --batch-size 32
