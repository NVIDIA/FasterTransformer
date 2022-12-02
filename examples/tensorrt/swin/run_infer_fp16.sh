if [ $1 -eq 1 ]; then
	python infer_swintransformer_plugin.py \
		--eval \
		--cfg ../../pytorch/swin/Swin-Transformer-Quantization/SwinTransformer/configs/swin/swin_tiny_patch4_window7_224.yaml \
		--resume ../../pytorch/swin/Swin-Transformer-Quantization/swin_tiny_patch4_window7_224.pth \
		--th-path ../../../build/lib/libpyt_swintransformer.so \
		--use-fp16 \
		--engine swin_transformer_fp16_v1.engine \
		--batch-size $2
fi

if [ $1 -eq 2 ]; then
	python infer_swintransformer_plugin.py \
		--eval \
		--cfg ../../pytorch/swin/Swin-Transformer-Quantization/SwinTransformer/configs/swinv2/swinv2_tiny_patch4_window8_256.yaml \
		--resume ../../pytorch/swin/Swin-Transformer-Quantization/swinv2_tiny_patch4_window8_256.pth \
		--th-path ../../../build/lib/libpyt_swintransformer.so \
		--use-fp16 \
		--engine swin_transformer_fp16_v2.engine \
		--batch-size $2
fi
