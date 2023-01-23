python builder.py \
    --batch-size 32 \
    --cfg ../../pytorch/swin/Swin-Transformer-Quantization/SwinTransformer/configs/swinv2/swinv2_tiny_patch4_window8_256.yaml \
    --resume ../../pytorch/swin/Swin-Transformer-Quantization/swinv2_tiny_patch4_window8_256.pth \
    --th-path ../../../build/lib/libth_transformer.so \
    --version 2 \
    --output swin_transformer_fp32_v2.engine

