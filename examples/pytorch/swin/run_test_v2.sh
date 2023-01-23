python infer_swintransformer_op.py \
    --eval \
    --data-path /workspace \
    --cfg Swin-Transformer-Quantization/SwinTransformer/configs/swinv2/swinv2_tiny_patch4_window8_256.yaml \
    --resume Swin-Transformer-Quantization/swinv2_tiny_patch4_window8_256.pth \
    --th-path ../../../build/lib/libth_transformer.so \
    --version 2 \
    --batch-size $1
