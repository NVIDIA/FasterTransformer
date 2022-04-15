python builder_fp16.py \
    --batch-size 32 \
    --cfg ../../pytorch/swin/Swin-Transformer-Quantization/SwinTransformer/configs/swin_tiny_patch4_window7_224.yaml \
    --resume ../../pytorch/swin/Swin-Transformer-Quantization/swin_tiny_patch4_window7_224.pth \
    --th-path ../../../build/lib/libpyt_swintransformer.so \
    --output swin_transformer_fp16.engine

