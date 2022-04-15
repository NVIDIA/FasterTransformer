python builder_int8.py \
    --batch-size 32 \
    --cfg ../../pytorch/swin/Swin-Transformer-Quantization/SwinTransformer/configs/swin_tiny_patch4_window7_224.yaml \
    --resume ../../pytorch/swin/Swin-Transformer-Quantization/calib-checkpoint/swin_tiny_patch4_window7_224_calib.pth \
    --th-path ../../../build/lib/libpyt_swintransformer.so \
    --output swin_transformer_int8.engine