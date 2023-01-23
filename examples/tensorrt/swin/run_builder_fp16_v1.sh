python builder.py \
    --batch-size 32 \
    --cfg ../../pytorch/swin/Swin-Transformer-Quantization/SwinTransformer/configs/swin/swin_tiny_patch4_window7_224.yaml \
    --resume ../../pytorch/swin/Swin-Transformer-Quantization/swin_tiny_patch4_window7_224.pth \
    --th-path ../../../build/lib/libth_transformer.so \
    --version 1 \
    --fp16 \
    --output swin_transformer_fp16_v1.engine

