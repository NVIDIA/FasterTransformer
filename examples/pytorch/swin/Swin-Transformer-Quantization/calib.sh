python -m torch.distributed.launch --nproc_per_node 1 \
  --master_port 12345 main.py \
  --calib \
  --cfg SwinTransformer/configs/swin_tiny_patch4_window7_224.yaml \
  --resume swin_tiny_patch4_window7_224.pth \
  --data-path /data/datasets/ILSVRC2012 \
  --num-calib-batch 10 \
  --calib-batchsz 8\
  --int8-mode 1\
  --calib-output-path calib-checkpoint
