from argparse import ArgumentParser
from io import BytesIO
from os import makedirs

import numpy as np

import torch

torch.set_printoptions(linewidth=130, sci_mode=False)
np.set_printoptions(linewidth=130, suppress=True)


def reshard(x, old_shape):
    import jax.numpy as jnp
    if len(x.shape) == 1:
        # print("epoch")
        # print(x)
        out = x[0:1]

    elif len(x.shape) == 2:
        #print(f"LN/bias {x.shape}")
        #print(x[:, :16])

        if (x[1:] == x[-1]).all():
            #print("LN")
            if (x[1:] == 0).all() or (x[1:] == 1).all():
                out = x[0:1]
            else:
                #print("shard bias")
                out = x[0:1] * 8#* x.shape[0] / old_shape[0]
        else:
            #print("bias")
            out = x.reshape(old_shape)

        #print(out[:, :16])

    elif len(x.shape) == 3:
        #print(f"weight {x.shape}")
        if x.shape[0] * x.shape[2] == old_shape[2]:
            #print("case 1")
            out = jnp.transpose(x, (1, 0, 2)).reshape(old_shape)
        elif x.shape[0] * x.shape[1] == old_shape[1]:
            #print("case 2")
            out = x.reshape(old_shape)
        else:
            raise Exception(f"unimplemented, {x.shape}, {old_shape}")
    else:
        raise Exception(f"unimplemented, {x}")
    #flattened, structure = jax.tree_flatten(out)
    #return flattened
    return out


def get_old_shape(t, dim=2):
    if len(t.shape) == 3:
        shard_shape = t.shape
        if dim == 1:
            return (shard_shape[0] * shard_shape[1], shard_shape[2])
        elif dim == 2:
            return (shard_shape[1], shard_shape[0] * shard_shape[2])
        else:
            raise ValueError(f"unsupported dim {dim}")
    if len(t.shape) == 2:
        return (t.shape[1] * t.shape[0],)
    else:
        raise ValueError(f"unsupported shape {t.shape}")


def read_shard(ckpt_dir, idx):
    out = []
    file_path = ckpt_dir + f"{idx}.npz"
    #print(f"-- {file_path}")
    with open(file_path, "rb") as f:
        buf = f.read()
        f_io = BytesIO(buf)
        deserialized = np.load(f_io)
        for i in deserialized:
            out.append(deserialized[i])
            #print(deserialized[i].shape)
    return out


def savebin(param, save_path):
    if isinstance(param, torch.Tensor):
        param = param.cpu().float().numpy()
    np.squeeze(param).astype(np.float32).tofile(save_path + ".bin")


def param2file(pt_param, layer_id, save_dir, dest_key):
    base_n = save_dir + "/model.layers." + str(layer_id) + "."
    save_path = base_n + dest_key
    savebin(pt_param, save_path)


def param2distributed(
    pt_param,
    layer_id,
    save_dir,
    dest_key,
    n_inference_gpus,
    split_axis,
):
    np_param = pt_param.cpu().float().numpy()
    base_n = save_dir + "/model.layers." + str(layer_id) + "."
    save_path = base_n + dest_key
    split_param = np.split(np_param, n_inference_gpus, axis=split_axis)
    for i, p in enumerate(split_param):
        savebin(p, save_path + f".{i}")


def save(w, save_dir, n_inference_gpus, num_layers=28):
    makedirs(save_dir, exist_ok=True)

    savebin(w['transformer.wte.weight'], save_dir + "/model.wte")
    for l in range(num_layers):
        print(f"Saving layer {l} / 28")
        base_k = "transformer.h." + str(l) + "."
        param2file(
          w[base_k + "ln_1.bias"],
          l, save_dir, "input_layernorm.bias"
        )
        param2file(
          w[base_k + "ln_1.weight"],
          l, save_dir, "input_layernorm.weight"
        )
        param2distributed(
          w[base_k + "mlp.c_fc.weight"].T,
          l, save_dir, "mlp.dense_h_to_4h.weight",
          n_inference_gpus, split_axis=-1 # split fast indx
        )
        param2distributed(
          w[base_k + "mlp.c_fc.bias"],
          l, save_dir, "mlp.dense_h_to_4h.bias",
          n_inference_gpus, split_axis=-1 # split fast indx
        )

        param2distributed(
          w[base_k + "mlp.c_proj.weight"].T,
          l, save_dir, "mlp.dense_4h_to_h.weight",
          n_inference_gpus, split_axis=0  # split slow indx
        )
        param2file(
          w[base_k + "mlp.c_proj.bias"],
          l, save_dir, "mlp.dense_4h_to_h.bias"
        )
        param2distributed(
          w[base_k + "attn.attention.out_proj.weight"].T,
          l, save_dir, "attention.dense.weight",
          n_inference_gpus, split_axis=0  # split slow indx
        )
        QKV_w = torch.stack([
          w[base_k + "attn.attention.q_proj.weight"],
          w[base_k + "attn.attention.k_proj.weight"],
          w[base_k + "attn.attention.v_proj.weight"],
        ]) # [qkv, n_heads * dim_head, latent_space]
        QKV_w = QKV_w.permute(2, 0, 1)
        param2distributed(
          QKV_w, l, save_dir, "attention.query_key_value.weight",
          n_inference_gpus, split_axis=-1 # split fast indx
        )
        # Other unneeded per-layer params:
        # attn.attention.masked_bias = torch.tensor(-1e9)
        # attn.attention.bias = torch.tril(torch.ones(1, 1, 2048, 2048))
    savebin(w['transformer.ln_f.weight'], save_dir + "/model.final_layernorm.weight")
    savebin(w['transformer.ln_f.bias'], save_dir + "/model.final_layernorm.bias")
    # lm head fast index should be hidden layer size, not vocab:
    savebin(w['lm_head.weight'], save_dir + "/model.lm_head.weight")
    savebin(w['lm_head.bias'], save_dir + "/model.lm_head.bias")


def main(ckpt_dir, num_layers=28, total_shards=8):
    import jax.numpy as jnp
    unshard = None
    transforms = [
        ("transformer.wte.bias", None, None),
        ("transformer.wte.weight", unshard, 1)
    ]

    checkpoint = {}

    layer_names = sorted(map(str, range(num_layers)))
    for layer in layer_names:
        checkpoint[
            f"transformer.h.{layer}.attn.attention.bias"
        ] = torch.tril(torch.ones(1, 1, 2048, 2048))
        checkpoint[
            f"transformer.h.{layer}.attn.attention.masked_bias"
        ] = torch.tensor(-1e9)
        transforms.extend([
            (f"transformer.h.{layer}.attn.attention.q_proj.weight", unshard, 2),
            (f"transformer.h.{layer}.attn.attention.v_proj.weight", unshard, 2),
            (f"transformer.h.{layer}.attn.attention.k_proj.weight", unshard, 2),
            (f"transformer.h.{layer}.attn.attention.out_proj.weight", unshard, 1),
            (f"transformer.h.{layer}.mlp.c_fc.bias", unshard, 1),
            (f"transformer.h.{layer}.mlp.c_fc.weight", unshard, 2),
            (f"transformer.h.{layer}.mlp.c_proj.bias", None, None),
            (f"transformer.h.{layer}.mlp.c_proj.weight", unshard, 1),
            (f"transformer.h.{layer}.ln_1.bias", None, None),
            (f"transformer.h.{layer}.ln_1.weight", None, None),
        ])
    transforms.extend([
        ("lm_head.bias", unshard, 1),
        ("lm_head.weight", unshard, 2),
        ("transformer.ln_f.bias", None, None),
        ("transformer.ln_f.weight", None, None),
    ])

    part = 0
    element = 0
    while len(transforms) > 0:
        print(f"loading shards for part {part}")
        shards = [
            read_shard(f"{ckpt_dir}shard_{i}/", part) for i in range(total_shards)
        ]
        print(f"read from checkpoint")

        unsharded = []

        for all_shards in zip(*shards):
            x = np.stack(all_shards)
            # No idea why this is V2...?
            if x.dtype == np.dtype('V2'):
                x.dtype = jnp.bfloat16
            x = x.astype(np.float32)
            unsharded.append(x)
            #print(f"unsharded: {x.shape}")

        while len(transforms) > 0 and len(unsharded) > 0:
            transform = transforms.pop(0)
            params = unsharded.pop(0)
            if transform[2] is not None:
                old_shape = (1,) + get_old_shape(params, transform[2])
            else:
                old_shape = (params.shape[1],)
            print(f"< {params.shape} to {old_shape}")
            params = reshard(params, old_shape).squeeze(0).T
            params = torch.tensor(np.array(params.copy())).half()
            if params.isnan().any() or params.isinf().any():
                raise ValueError(f"fp16 over/underflow at {part} {element}")
            checkpoint[transform[0]] = params
            print(f"> {transform[0]} {params.shape}")
            element += 1
        part += 1

    checkpoint['transformer.wte.weight'] = (
        checkpoint['transformer.wte.weight'].T + checkpoint['transformer.wte.bias']
    )
    del checkpoint['transformer.wte.bias']

    print(f"left over: {unsharded}")
    return checkpoint


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert GPT-J slim checkpoint to FasterTransformer",
    )
    parser.add_argument(
        "--output-dir", help="Folder where binary files are stored", default="j6b_ckpt/"
    )
    parser.add_argument(
        "--ckpt-dir", help="Folder containing slim GPT-J checkpoint", default="step_383500/"
    )
    parser.add_argument(
        "--n-inference-gpus", help="Number of GPUs used for inference runtime", default=1, type=int
    )
    args = parser.parse_args()

    num_layers = 28

    print("loading")
    in_path = args.ckpt_dir
    if len(in_path)>3 and in_path[-3:] == ".pt":
        checkpoint = torch.load(in_path)
    else:
        checkpoint = main(in_path, num_layers)

    print("saving")
    # load as in: https://github.com/finetuneanon/misc/blob/main/SizeTest.ipynb
    out_path = args.output_dir
    if len(out_path)>3 and out_path[-3:] == ".pt":
        torch.save(checkpoint, out_path)
    else:
        output_dir = out_path + f"/{args.n_inference_gpus}-gpu/"
        save(checkpoint, output_dir, args.n_inference_gpus, num_layers)

    print("done")
