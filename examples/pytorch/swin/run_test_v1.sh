python infer_swintransformer_op.py \
    --eval \
    --data-path /workspace \
    --cfg Swin-Transformer-Quantization/SwinTransformer/configs/swin/swin_tiny_patch4_window7_224.yaml \
    --resume Swin-Transformer-Quantization/swin_tiny_patch4_window7_224.pth \
    --th-path ../../../build/lib/libth_transformer.so \
    --version 1 \
    --batch-size $1
