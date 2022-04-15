python infer_swintransformer_plugin.py \
    --eval \
    --cfg ../../pytorch/swin/Swin-Transformer-Quantization/SwinTransformer/configs/swin_tiny_patch4_window7_224.yaml \
    --resume ../../pytorch/swin/Swin-Transformer-Quantization/swin_tiny_patch4_window7_224.pth \
    --th-path ../../../build/lib/libpyt_swintransformer.so \
    --engine swin_transformer_fp32.engine \
    --batch-size $1 
