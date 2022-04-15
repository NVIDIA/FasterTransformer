python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 12346 main.py \
    --eval \
    --cfg SwinTransformer/configs/swin_tiny_patch4_window7_224.yaml \
    --resume ./calib-checkpoint/swin_tiny_patch4_window7_224_calib.pth \
    --data-path /data/datasets/ILSVRC2012/ \
    --quant-mode ft2\
    --int8-mode 1\
    --batch-size 128
