python infer_swintransformer_op.py \
    --eval \
    --data-path /workspace \
    --cfg Swin-Transformer-Quantization/SwinTransformer/configs/swin_tiny_patch4_window7_224.yaml \
    --resume Swin-Transformer-Quantization/swin_tiny_patch4_window7_224.pth \
    --th-path ../../../build/lib/libpyt_swintransformer.so \
    --batch-size $1
