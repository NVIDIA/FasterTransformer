# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Convert huggingface bert model. Use https://huggingface.co/bert-base-uncased as demo.
'''

import argparse
import configparser
import multiprocessing
import numpy as np
import pathlib
import torch
import os
import sys

# __root_package_path__ = pathlib.Path(__file__).parent.parent.parent.parent.parent.absolute().as_posix()
# if __root_package_path__ not in sys.path:
#     print(
#         f"[ERROR] add project root directory to your PYTHONPATH with "
#         f"'export PYTHONPATH={__root_package_path__}:${{PYTHONPATH}}'"
#     )

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../..")
sys.path.append(dir_path)
from examples.pytorch.utils import torch2np, safe_transpose, WEIGHT2DTYPE

from transformers import BertModel # transformers-4.10.0-py3

def split_and_convert_process(i, saved_dir,factor,key, args, val):

    if key.find("attention.output.dense.bias") != -1 or \
       key.find("attention.output.LayerNorm.weight") != -1 or \
       key.find("attention.output.LayerNorm.bias") != -1 or \
       key.find("output.dense.bias") != -1 or \
       key.find("output.LayerNorm.weight") != -1 or \
       key.find("output.LayerNorm.bias") != -1 :
    
        # shared weights, only need to convert the weights of rank 0
        if i == 0:
            saved_path = saved_dir + "/model." + key + ".bin"
            val.tofile(saved_path)

    elif key.find("attention.output.dense.weight") != -1 or key.find("output.dense.weight") != -1:
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = f"{saved_dir}/model.{key}.{i * factor + j}.bin"
            split_vals[j].tofile(saved_path)

    elif key.find("attention.self.query.weight") != -1 or \
         key.find("attention.self.query.bias") != -1 or \
         key.find("attention.self.key.weight") != -1 or \
         key.find("attention.self.key.bias") != -1 or \
         key.find("attention.self.value.weight") != -1 or \
         key.find("attention.self.value.bias") != -1 or \
         key.find("intermediate.dense.weight") != -1 or \
         key.find("intermediate.dense.bias") != -1:

        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir + "/model." + key + ".%d.bin" % (i * factor + j)
            split_vals[j].tofile(saved_path)

    else:
        print("[WARNING] cannot convert key '{}'".format(key))

def split_and_convert(args):
    saved_dir = args.saved_dir + "/%d-gpu/" % args.infer_tensor_para_size

    if(os.path.exists(saved_dir) == False):
        os.makedirs(saved_dir)
    ckpt_name = args.in_file

    t_gpu_num = args.training_tensor_para_size
    i_gpu_num = args.infer_tensor_para_size
    assert(i_gpu_num % t_gpu_num == 0)

    factor = (int)(i_gpu_num / t_gpu_num)

    # load position_embedding from rank 0
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertModel.from_pretrained(args.in_file).to(torch_device)
    np_weight_data_type = WEIGHT2DTYPE[args.weight_data_type]

    hf_config = vars(model.config)

    # NOTE: save parameters to config files (loaded by triton backends)
    config = configparser.ConfigParser()
    config["bert"] = {}
    try:
        config["bert"]["model_name"] = "bert" if hf_config["model_type"] == '' else hf_config["model_type"]
        config["bert"]["position_embedding_type"] = str(hf_config["position_embedding_type"])
        config["bert"]["hidden_size"] = str(hf_config["hidden_size"])
        config["bert"]["num_layer"] = str(hf_config["num_hidden_layers"])
        config["bert"]["head_num"] = str(hf_config["num_attention_heads"])
        config["bert"]["size_per_head"] = str(hf_config["hidden_size"] // hf_config["num_attention_heads"])
        config["bert"]["activation_type"] = str(hf_config["hidden_act"])
        config["bert"]["inter_size"] = str(hf_config["intermediate_size"])
        config["bert"]["max_position_embeddings"] = str(hf_config["max_position_embeddings"])
        config["bert"]["layer_norm_eps"] = str(hf_config["layer_norm_eps"])
        config["bert"]["weight_data_type"] = args.weight_data_type
        config["bert"]["tensor_para_size"] = str(args.infer_tensor_para_size)
        with open(saved_dir + "/config.ini", 'w') as configfile:
            config.write(configfile)
    except:
        print(f"Fail to save the config in config.ini.")

    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")
    pool = multiprocessing.Pool(args.processes)
    for name, param in model.named_parameters():
        if name.find("weight") == -1 and name.find("bias") == -1:
            continue
        else:
            pool.starmap(split_and_convert_process,
                        [(0, saved_dir, factor, name, args,
                            torch2np(safe_transpose(param.detach()), np_weight_data_type))], )

    pool.close()
    pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-saved_dir', '-o', type=str, help='file name of output file', required=True)
    parser.add_argument('-in_file', '-i', type=str, help='file name of input checkpoint file', required=True)
    parser.add_argument('-training_tensor_para_size', '-t_g', type=int, help='The size of tensor parallelism for training.', default=1)
    parser.add_argument('-infer_tensor_para_size', '-i_g', type=int, help='The size of tensor parallelism for inference.', required=True)
    parser.add_argument("-processes", "-p", type=int, help="How many processes to spawn for conversion (default: 4)", default=4)
    parser.add_argument("-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"])

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    split_and_convert(args)
