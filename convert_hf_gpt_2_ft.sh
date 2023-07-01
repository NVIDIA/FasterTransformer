python examples/pytorch/gpt/utils/huggingface_gpt_convert.py \
-i hf_models/gpt2/ \
-o ft_models/huggingface-models/c-model/gpt2 \
-i_g 1 \
-t_g 1 \
-weight_data_type fp16 \
-p 16
