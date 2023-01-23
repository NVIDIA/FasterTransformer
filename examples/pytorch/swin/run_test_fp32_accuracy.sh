python infer_swintransformer_acc.py \
    --eval \
    --data-path $1 \
    --cfg Swin-Transformer-Quantization/SwinTransformer/configs/swinv2/swinv2_tiny_patch4_window8_256.yaml \
    --resume Swin-Transformer-Quantization/swinv2_tiny_patch4_window8_256.pth \
    --th-path ../../../build/lib/libth_transformer.so \
    --version 2 \
    --batch-size 128
