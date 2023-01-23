python builder_int8.py \
    --batch-size 32 \
    --cfg ../../pytorch/swin/Swin-Transformer-Quantization/SwinTransformer/configs/swinv2/swinv2_tiny_patch4_window8_256.yaml \
    --resume ../../pytorch/swin/Swin-Transformer-Quantization/calib-checkpoint/swinv2_tiny_patch4_window8_256_calib.pth \
    --th-path ../../../build/lib/libth_transformer.so \
    --int8-mode 1 \
    --version 2 \
    --output swin_transformer_int8_v2.engine
