import os
import sys
import numpy as np
import torch

# This file is used with "https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/checkpoint_util/tools/checkpoint_util.py"
# Example
# python tools/checkpoint_util.py --model-type GPT --loader megatron --saver fastertransformer \
# --input /home/scratch.bhsueh_sw/megatron_new_ckpt/357m-pipeline-2-tensor-2/  --output ./tmp  --target-tensor-parallel-size 2

def add_arguments(parser):
    group = parser.add_argument_group(title='FasterTransformer saver')

    group.add_argument('--megatron-path', type=str, default=".",
                       help='Base directory of Megatron repository')
    group.add_argument('--target-tensor-parallel-size', type=int,
                       help='Target tensor model parallel size')

def save_checkpoint(queue, args):

    sys.path.insert(0, args.megatron_path)
    from megatron.global_vars import set_global_variables, get_args

    md = queue.get()

    os.environ["WORLD_SIZE"] = f'{args.target_tensor_parallel_size}'

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--num-layers', str(md.num_layers),
                '--hidden-size', str(md.hidden_size),
                '--seq-length', str(md.seq_length),
                '--num-attention-heads', str(md.num_attention_heads),
                '--max-position-embeddings', str(md.max_position_embeddings),
                '--tokenizer-type', str(md.tokenizer_type),
                '--tensor-model-parallel-size', str(args.target_tensor_parallel_size),
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--save-interval', '1',
                '--save', args.output
                ]
    set_global_variables()

    # margs = megatron args
    margs = get_args()

    # Embeddings
    #-----------
    pos_embed = queue.get()
    full_word_embed = queue.get()

    # Tell Megatron what our full size is
    margs.padded_vocab_size = full_word_embed.shape[0]
    if margs.padded_vocab_size % args.target_tensor_parallel_size != 0:
        print("source vocab size is not evenly divisble by target tensor parallel size")
        exit(1)

    if(os.path.exists(args.output) == False):
        os.makedirs(args.output)

    with open(args.output + "/args.txt", "w") as outfile:
        outfile.write("{}\n".format(md))
     
    pos_embed.cpu().numpy().astype(np.float32).tofile(args.output + "/model.wpe.bin")
    full_word_embed.cpu().numpy().astype(np.float32).tofile(args.output + "/model.wte.bin")

    # Transformer layers
    #-------------------
    for layer in range(md.num_layers):
        # get full tensors
        input_layernorm_weight = queue.get().T
        input_layernorm_bias = queue.get().T
        full_qkv_weight = queue.get().T
        full_qkv_bias = queue.get().T
        full_dense_weight = queue.get().T
        dense_bias = queue.get().T
        post_layernorm_weight = queue.get().T
        post_layernorm_bias = queue.get().T
        full_mlp_l0_weight = queue.get().T
        full_mlp_l0_bias = queue.get().T
        full_mlp_l1_weight = queue.get().T
        mlp_l1_bias = queue.get().T

        #  Assume the version of checkpoint is 3
        ckpt_ver = 3
        if ckpt_ver == 3:
            num_splits = 3
            size_per_head = (int)(md.hidden_size / md.num_attention_heads)
            full_qkv_weight = full_qkv_weight.reshape(md.hidden_size, md.num_attention_heads ,num_splits, size_per_head)
            full_qkv_weight = full_qkv_weight.permute(0, 2, 1, 3)
            full_qkv_weight = full_qkv_weight.reshape(md.hidden_size, num_splits, md.hidden_size)
            full_qkv_bias = full_qkv_bias.reshape(md.num_attention_heads ,num_splits, size_per_head)
            full_qkv_bias = full_qkv_bias.permute(1, 0, 2)
            full_qkv_bias = full_qkv_bias.reshape(num_splits, md.hidden_size)

        # Split up the parallel tensors
        out_qkv_weight = torch.chunk(full_qkv_weight, args.target_tensor_parallel_size, dim=-1)
        out_qkv_bias = torch.chunk(full_qkv_bias, args.target_tensor_parallel_size, dim=-1)
        out_dense_weight = torch.chunk(full_dense_weight, args.target_tensor_parallel_size, dim=0)
        out_mlp_l0_weight = torch.chunk(full_mlp_l0_weight, args.target_tensor_parallel_size, dim=-1)
        out_mlp_l0_bias = torch.chunk(full_mlp_l0_bias, args.target_tensor_parallel_size, dim=-1)
        out_mlp_l1_weight = torch.chunk(full_mlp_l1_weight, args.target_tensor_parallel_size, dim=0)

        # Save model
        input_layernorm_weight.cpu().numpy().astype(np.float32).tofile(args.output + "/model.layers.{}.input_layernorm.weight.bin".format(layer))
        input_layernorm_bias.cpu().numpy().astype(np.float32).tofile(args.output + "/model.layers.{}.input_layernorm.bias.bin".format(layer))
        
        post_layernorm_weight.cpu().numpy().astype(np.float32).tofile(args.output + "/model.layers.{}.post_attention_layernorm.weight.bin".format(layer))
        post_layernorm_bias.cpu().numpy().astype(np.float32).tofile(args.output + "/model.layers.{}.post_attention_layernorm.bias.bin".format(layer))
        dense_bias.cpu().numpy().astype(np.float32).tofile(args.output + "/model.layers.{}.attention.dense.bias.bin".format(layer))
        
        mlp_l1_bias.cpu().numpy().astype(np.float32).tofile(args.output + "/model.layers.{}.mlp.dense_4h_to_h.bias.bin".format(layer))

        for tp_rank in range(args.target_tensor_parallel_size):

            out_qkv_weight[tp_rank].cpu().numpy().astype(np.float32).tofile(args.output + "/model.layers.{}.attention.query_key_value.weight.{}.bin".format(layer, tp_rank))
            out_qkv_bias[tp_rank].cpu().numpy().astype(np.float32).tofile(args.output + "/model.layers.{}.attention.query_key_value.bias.{}.bin".format(layer, tp_rank))
            out_dense_weight[tp_rank].cpu().numpy().astype(np.float32).tofile(args.output + "/model.layers.{}.attention.dense.weight.{}.bin".format(layer, tp_rank))
            out_mlp_l0_weight[tp_rank].cpu().numpy().astype(np.float32).tofile(args.output + "/model.layers.{}.mlp.dense_h_to_4h.weight.{}.bin".format(layer, tp_rank))
            out_mlp_l0_bias[tp_rank].cpu().numpy().astype(np.float32).tofile(args.output + "/model.layers.{}.mlp.dense_h_to_4h.bias.{}.bin".format(layer, tp_rank))
            out_mlp_l1_weight[tp_rank].cpu().numpy().astype(np.float32).tofile(args.output + "/model.layers.{}.mlp.dense_4h_to_h.weight.{}.bin".format(layer, tp_rank))
            
    final_layernorm_weight = queue.get().T
    final_layernorm_bias = queue.get().T

    final_layernorm_weight.cpu().numpy().astype(np.float32).tofile(args.output + "/model.final_layernorm.weight.bin")
    final_layernorm_bias.cpu().numpy().astype(np.float32).tofile(args.output + "/model.final_layernorm.bias.bin")

    del final_layernorm_weight
    del final_layernorm_bias

    print("Done!")
