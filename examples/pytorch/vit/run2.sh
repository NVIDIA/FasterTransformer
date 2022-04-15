python infer_visiontransformer_int8_op.py \
    --th-path=../../../build/lib/libpyt_vit.so \
    --calibrated_dir /workspace/checkpoint/ViT-B_16_ft2_99.99_81.948.pth \
    --img_size 384 \
    --quant-mode ft2 
