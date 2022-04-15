#!/bin/bash

TASK="LAMBADA"
LAMBADA_PATH="../models/megatron-models/lambada_test.jsonl"
VALID_DATA=$LAMBADA_PATH

VOCAB_FILE=../models/gpt2-vocab.json
MERGE_FILE=../models/gpt2-merges.txt
CHECKPOINT=../models/megatron-models/345m/

python -m torch.distributed.run --nproc_per_node 1 ../examples/pytorch/gpt/evaluate_zeroshot_gpt.py \
               --task $TASK \
               --valid-data $VALID_DATA \
               --tokenizer-type GPT2BPETokenizer \
               --strict-lambada \
               --vocab-file $VOCAB_FILE \
               --merge-file $MERGE_FILE \
               --load $CHECKPOINT \
               --tensor-model-parallel-size 1 \
               --num-layers 24 \
               --hidden-size 1024 \
               --num-attention-heads 16 \
               --micro-batch-size 32 \
               --checkpoint-activations \
               --seq-length 1024 \
               --max-position-embeddings 1024 \
               --log-interval 10 \
               --fp16 \
               --no-load-optim \
               --no-load-rng \
               --ckpt-path '../models/megatron-models/c-model/345m/1-gpu' \
               --lib-path "lib/libth_gpt.so" \
               --beam_width 1 \
               --top_k 1 \
               --top_p 0.0

sleep 20

python -m torch.distributed.run --nproc_per_node 1 ../examples/pytorch/gpt/evaluate_zeroshot_gpt.py \
               --task $TASK \
               --valid-data $VALID_DATA \
               --tokenizer-type GPT2BPETokenizer \
               --strict-lambada \
               --vocab-file $VOCAB_FILE \
               --merge-file $MERGE_FILE \
               --load $CHECKPOINT \
               --tensor-model-parallel-size 1 \
               --num-layers 24 \
               --hidden-size 1024 \
               --num-attention-heads 16 \
               --micro-batch-size 32 \
               --checkpoint-activations \
               --seq-length 1024 \
               --max-position-embeddings 1024 \
               --log-interval 10 \
               --fp16 \
               --no-load-optim \
               --no-load-rng \
               --ckpt-path '../models/megatron-models/c-model/345m/1-gpu' \
               --lib-path "lib/libth_gpt.so" \
               --beam_width 1 \
               --top_k 0 \
               --top_p 0.5