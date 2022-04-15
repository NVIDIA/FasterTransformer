python infer_swintransformer_int8_op.py \
    --eval \
    --data-path /workspace \
    --cfg Swin-Transformer-Quantization/SwinTransformer/configs/swin_tiny_patch4_window7_224.yaml \
    --resume  Swin-Transformer-Quantization/calib-checkpoint/swin_tiny_patch4_window7_224_calib.pth \
    --th-path ../../../build/lib/libpyt_swintransformer.so \
    --int8-mode 1\
    --batch-size $1
