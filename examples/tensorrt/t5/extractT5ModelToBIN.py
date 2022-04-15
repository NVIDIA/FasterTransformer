import os
import sys
import argparse
import numpy as np
import torch
from transformers import T5ForConditionalGeneration

modelName = 't5-small'
savePath  = './para'

def fuse_decoder_qkv(model, factor, saved_dir):
    model_dict = {}
    for name, param in model.named_parameters():
        if name.find("decoder") != -1 and name.find("SelfAttention") != -1:
            model_dict[name] = param

    for i in range(model.decoder.config.num_layers):
        shape = model_dict[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"].T.shape
        qkv = torch.cat([model_dict[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"].T,
                         model_dict[f"decoder.block.{i}.layer.0.SelfAttention.k.weight"].T,
                         model_dict[f"decoder.block.{i}.layer.0.SelfAttention.v.weight"].T], dim=-1)

        qkv = qkv.reshape([shape[0], 3, shape[1]])
        qkv = qkv.float().cpu().detach().numpy()

        split_vals = np.split(qkv, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir + "/" + f"decoder.block.{i}.layer.0.SelfAttention.qkv.weight.{j}.bin"
            split_vals[j].tofile(saved_path)

def split_and_convert_process(key, val, factor, saved_dir):
    val = val.T.detach().numpy()
    saved_key = key

    if key.find("shared.weight") != -1:
        # shared weights, only need to convert the weights of rank 0
        saved_path = saved_dir + "/" + f"{saved_key}.bin"
        val.tofile(saved_path)

        saved_path = saved_dir + "/" + f"{saved_key}_T.bin"
        val.T.tofile(saved_path)
    elif key.find("layer_norm.weight") != -1:
        # shared weights, only need to convert the weights of rank 0
        saved_path = saved_dir + "/" + f"{saved_key}.bin"
        val.tofile(saved_path)

    elif (
        key.find("SelfAttention.o.weight") != -1
        or key.find("EncDecAttention.o.weight") != -1
        or key.find("DenseReluDense.wo.weight") != -1
        ):
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir + "/" + f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path)

    elif (
        key.find("DenseReluDense.wi.weight") != -1
        or (key.find("encoder") != -1 and (
            key.find("SelfAttention.q.weight") != -1
            or key.find("SelfAttention.k.weight") != -1
            or key.find("SelfAttention.v.weight") != -1
            )
            )
        or key.find("EncDecAttention.q.weight") != -1
        or key.find("EncDecAttention.k.weight") != -1
        or key.find("EncDecAttention.v.weight") != -1
        ):
        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir + "/" + f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path)
    elif key.find("relative_attention_bias") != -1:
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir + "/" + f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path)
    elif (
        key.find("decoder") != -1 and
        (
            key.find("SelfAttention.q.weight") != -1
            or key.find("SelfAttention.k.weight") != -1
            or key.find("SelfAttention.v.weight") != -1
        )
        ):
        pass
    else:
        print(f"[ERROR] cannot find key '{key}'")

if __name__ == "__main__":
    os.system("mkdir -p para")

    t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
    for name, param in t5_model.named_parameters():
        split_and_convert_process(name, param, 1, savePath)
    fuse_decoder_qkv(t5_model, 1, savePath)

    print("extract T5 model weight finish!")

