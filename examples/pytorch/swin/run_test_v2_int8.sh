python infer_swintransformer_int8_op.py \
    --profile \
    --data-path /data/datasets/ILSVRC2012 \
    --cfg Swin-Transformer-Quantization/SwinTransformer/configs/swinv2/swinv2_tiny_patch4_window8_256.yaml \
    --resume Swin-Transformer-Quantization/calib-checkpoint/swinv2_tiny_patch4_window8_256_calib.pth \
    --th-path ../../../build/lib/libth_transformer.so \
    --version 2 \
    --int8-mode 1 \
    --batch-size $1
