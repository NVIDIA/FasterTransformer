if [ $1 -eq 1 ]; then
    python infer_swintransformer_plugin_int8.py \
        --eval \
        --cfg ../../pytorch/swin/Swin-Transformer-Quantization/SwinTransformer/configs/swin/swin_tiny_patch4_window7_224.yaml \
        --resume ../../pytorch/swin/Swin-Transformer-Quantization/calib-checkpoint/swin_tiny_patch4_window7_224_calib.pth \
        --int8-mode 1 \
        --version 1 \
        --th-path ../../../build/lib/libth_transformer.so \
        --engine swin_transformer_int8_v1.engine \
        --batch-size $2
fi

if [ $1 -eq 2 ]; then
    python infer_swintransformer_plugin_int8.py \
        --eval \
        --cfg ../../pytorch/swin/Swin-Transformer-Quantization/SwinTransformer/configs/swinv2/swinv2_tiny_patch4_window8_256.yaml \
        --resume ../../pytorch/swin/Swin-Transformer-Quantization/calib-checkpoint/swinv2_tiny_patch4_window8_256_calib.pth \
        --int8-mode 1 \
        --version 2 \
        --th-path ../../../build/lib/libth_transformer.so \
        --engine swin_transformer_int8_v2.engine \
        --batch-size $2
fi