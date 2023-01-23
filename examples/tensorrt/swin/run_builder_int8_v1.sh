python builder_int8.py \
    --batch-size 32 \
    --cfg ../../pytorch/swin/Swin-Transformer-Quantization/SwinTransformer/configs/swin/swin_tiny_patch4_window7_224.yaml \
    --resume ../../pytorch/swin/Swin-Transformer-Quantization/calib-checkpoint/swin_tiny_patch4_window7_224_calib.pth \
    --th-path ../../../build/lib/libth_transformer.so \
    --int8-mode 1 \
    --version 1 \
    --output swin_transformer_int8_v1.engine
