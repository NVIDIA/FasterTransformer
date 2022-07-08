# Temporary simple script for launching

python3 examples/pytorch/gpt/gpt_example.py \
	--layer_num 32 \
	--head_num 12 \
        --size_per_head 128 \
        --ckpt_path "/checkpoint/1-gpu/" \
        --vocab_file "/vocabularies/gpt2-vocab.json" \
        --merges_file "/vocabularies/gpt2-merges.txt" \
	--max_seq_len 2048 \
	--data_type "bf16" \
	--sample_input_file "inputs.txt" \
	--sample_output_file "outputs.txt" \
	--lib_path "build/lib/libth_gpt.so" \
