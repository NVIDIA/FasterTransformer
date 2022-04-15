python infer_visiontransformer_int8_op.py \
    --th-path=../../../build/lib/libpyt_vit.so \
    --calibrated_dir /workspace/checkpoint/ViT-B_16_ft1_99.999_82.846.pth \
    --img_size 384 \
    --quant-mode ft1
