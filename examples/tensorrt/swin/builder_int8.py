# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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
import argparse
import ctypes
import json
import numpy as np
import os
import os.path
import re
import sys
import time
import pycuda.autoinit
import tensorrt as trt
import torch

import sys
sys.path.insert(0, "../../pytorch/swin/Swin-Transformer-Quantization")
sys.path.insert(0, "../../pytorch/swin")

from SwinTransformer.config import get_config
from SwinTransformerINT8Weight import SwinTransformerINT8Weight

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="max batch size")
    parser.add_argument('--th-path', type=str, help='path to pytorch library')
    parser.add_argument('--data-path', type=str, help='path to dataset', default=None)
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset', default=None)
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument('--eval', action='store_true', help='Perform evaluation only', default=True)
    parser.add_argument('--int8-mode', type=int, help='int8 mode', default=1, choices=[1, 2])
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel', default=-1)

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

#TensorRT Initialization
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
handle = ctypes.CDLL("../../../build/lib/libswinTransformer_plugin.so", mode=ctypes.RTLD_GLOBAL)
if not handle:
    raise RuntimeError("Fail to load plugin library")

trt.init_libnvinfer_plugins(TRT_LOGGER, "")
plg_registry = trt.get_plugin_registry()
swinTransformer_plg_creator = plg_registry.get_plugin_creator("CustomSwinTransformerINT8Plugin", "1", "")

def set_tensor_name(tensor, prefix, name):
    tensor.name = prefix + name

def set_output_name(layer, prefix, name, out_idx = 0):
    set_tensor_name(layer.get_output(out_idx), prefix, name)

def swin_transformer(network, config, args, input_img, weights_dict):
    depths = config.MODEL.SWIN.DEPTHS
    num_heads = config.MODEL.SWIN.NUM_HEADS
    if config.MODEL.SWIN.QK_SCALE is not None:
        qk_scale = config.MODEL.SWIN.QK_SCALE
    else:
        qk_scale = 1.0
    if config.MODEL.SWIN.APE:
        ape = 1
    else:
        ape = 0
    if config.MODEL.SWIN.PATCH_NORM:
        patch_norm = 1
    else:
        patch_norm = 0
    if config.MODEL.SWIN.QKV_BIAS:
        qkv_bias = 1
    else:
        qkv_bias = 0

    int8_mode = trt.PluginField("int8_mode", np.array([args.int8_mode]).astype(np.int32), trt.PluginFieldType.INT32)
    max_batch_size = trt.PluginField("max_batch_size", np.array([config.DATA.BATCH_SIZE]).astype(np.int32), trt.PluginFieldType.INT32)
    img_size = trt.PluginField("img_size", np.array([config.DATA.IMG_SIZE]).astype(np.int32), trt.PluginFieldType.INT32)
    patch_size = trt.PluginField("patch_size", np.array([config.MODEL.SWIN.PATCH_SIZE]).astype(np.int32), trt.PluginFieldType.INT32)
    in_chans = trt.PluginField("in_chans", np.array([config.MODEL.SWIN.IN_CHANS]).astype(np.int32), trt.PluginFieldType.INT32)
    embed_dim = trt.PluginField("embed_dim", np.array([config.MODEL.SWIN.EMBED_DIM]).astype(np.int32), trt.PluginFieldType.INT32)
    window_size = trt.PluginField("window_size", np.array([config.MODEL.SWIN.WINDOW_SIZE]).astype(np.int32), trt.PluginFieldType.INT32)
    ape = trt.PluginField("ape", np.array([ape]).astype(np.int32), trt.PluginFieldType.INT32)
    patch_norm = trt.PluginField("patch_norm", np.array([patch_norm]).astype(np.int32), trt.PluginFieldType.INT32)
    layer_num = trt.PluginField("layer_num", np.array([len(depths)]).astype(np.int32), trt.PluginFieldType.INT32)
    mlp_ratio = trt.PluginField("mlp_ratio", np.array([config.MODEL.SWIN.MLP_RATIO]).astype(np.float32), trt.PluginFieldType.FLOAT32)
    qkv_bias = trt.PluginField("qkv_bias", np.array([qkv_bias]).astype(np.int32), trt.PluginFieldType.INT32)
    qk_scale = trt.PluginField("qk_scale", np.array([qk_scale]).astype(np.float32), trt.PluginFieldType.FLOAT32)
    depths_f = trt.PluginField("depths", np.array(depths).astype(np.int32), trt.PluginFieldType.INT32)
    num_heads_f = trt.PluginField("num_heads", np.array(num_heads).astype(np.int32), trt.PluginFieldType.INT32)
   
    sw_weights = SwinTransformerINT8Weight(len(depths), config.MODEL.SWIN.WINDOW_SIZE, depths, num_heads, args.th_path, weights_dict)   

    for i in range(len(sw_weights.weights)):
        sw_weights.weights[i] = sw_weights.weights[i].cpu()

    part_fc = []
    weight_idx = 0
    for l in range(len(depths)):
        for b in range(depths[l]):
            part_fc.append(trt.PluginField("attention_qkv_kernel_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("attention_qkv_bias_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("attention_proj_kernel_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("attention_proj_bias_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("mlp_linear_kernel_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("mlp_linear_bias_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("mlp_linear2_kernel_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("mlp_linear2_bias_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("block_norm_gamma_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("block_norm_beta_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("block_norm2_gamma_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("block_norm2_beta_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1
            part_fc.append(trt.PluginField("block_d_amaxlist_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
            weight_idx += 1
            part_fc.append(trt.PluginField("block_h_amaxlist_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx]).astype(np.float32), trt.PluginFieldType.FLOAT32))
            weight_idx += 1
            part_fc.append(trt.PluginField("attention_relative_pos_bias_{}_{}".format(l, b), np.array(sw_weights.weights[weight_idx].cpu()).astype(np.float16), trt.PluginFieldType.FLOAT16))
            weight_idx += 1

        part_fc.append(trt.PluginField("patchMerge_norm_gamma_{}".format(l), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
        weight_idx += 1
        part_fc.append(trt.PluginField("patchMerge_norm_beta_{}".format(l), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
        weight_idx += 1
        part_fc.append(trt.PluginField("patchMerge_linear_kernel_{}".format(l), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
        weight_idx += 1
        part_fc.append(trt.PluginField("attn_mask_{}".format(l), np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
        weight_idx += 1

    part_fc.append(trt.PluginField("patchEmbed_proj_kernel", np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
    weight_idx += 1
    part_fc.append(trt.PluginField("patchEmbed_proj_bias", np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
    weight_idx += 1
    part_fc.append(trt.PluginField("patchEmbed_norm_gamma", np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
    weight_idx += 1
    part_fc.append(trt.PluginField("patchEmbed_norm_beta", np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
    weight_idx += 1
    part_fc.append(trt.PluginField("norm_gamma", np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))
    weight_idx += 1
    part_fc.append(trt.PluginField("norm_beta", np.array(sw_weights.weights[weight_idx]).astype(np.float16), trt.PluginFieldType.FLOAT16))


    pfc = trt.PluginFieldCollection([int8_mode, max_batch_size, img_size, patch_size, in_chans, embed_dim, window_size, ape, patch_norm, layer_num, mlp_ratio, qkv_bias, qk_scale, depths_f, num_heads_f] + part_fc)
    fn = swinTransformer_plg_creator.create_plugin("swin_transformer", pfc)
    inputs = [input_img]
    sw = network.add_plugin_v2(inputs, fn) 

    set_output_name(sw, "swin_transformer_", "output")
    return sw

def load_weights(inputbase, config):
    weights_dict = dict()
    try:
        tensor_dict = torch.load(inputbase,
                                 map_location='cpu')
        # tensor_dict = tensor_dict['model']
        # remove training-related variables in the checkpoint
        param_names = [key for key in sorted(tensor_dict)]

        for pn in param_names:
            if isinstance(tensor_dict[pn], np.ndarray):
                tensor = tensor_dict[pn]
            else:
                tensor = tensor_dict[pn].numpy()

            shape = tensor.shape

            ##to be compatible with SwinTransformerWeight
            if "index" in pn:
                flat_tensor = tensor.astype(dtype=np.int64)
                weights_dict[pn] = torch.tensor(flat_tensor, dtype=torch.int64).cuda()
            elif "table" in pn:
                flat_tensor = tensor.astype(dtype=np.float32)
                weights_dict[pn] = torch.tensor(flat_tensor, dtype=torch.float32).cuda()
            else:
                flat_tensor = tensor.astype(dtype=np.float32)
                weights_dict[pn] = torch.tensor(flat_tensor, dtype=torch.float32).cuda()

            shape_str = "{} ".format(len(shape)) + " ".join([str(d) for d in shape])
            #print("TensorRT name: {:}, shape: {:}".format("module."+pn, shape_str))

    except Exception as error:
        TRT_LOGGER.log(TRT_LOGGER.ERROR, str(error))

    return weights_dict


def build_engine(config, args, weights_dict):
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
        builder_config.max_workspace_size = 8 << 30
        builder_config.set_flag(trt.BuilderFlag.FP16)
        builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # Create the network
        input_img = network.add_input(name="input_img", dtype=trt.float16, shape=(-1, config.MODEL.SWIN.IN_CHANS, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE))
   
        # Specify profiles 
        ##TODO: There is a bug in TRT when opt batch is large
        profile = builder.create_optimization_profile()
        min_shape = (1, config.MODEL.SWIN.IN_CHANS, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
        max_shape = (config.DATA.BATCH_SIZE, config.MODEL.SWIN.IN_CHANS, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
        profile.set_shape("input_img", min=min_shape, opt=min_shape, max=max_shape)
        builder_config.add_optimization_profile(profile)

        #import pdb;pdb.set_trace()
        sw_output = swin_transformer(network, config, args, input_img, weights_dict) 
        sw_output.precision = trt.float16
        sw_output.set_output_type(0, trt.float16)

        output_size = weights_dict["head.bias"].shape[0]
        output = network.add_fully_connected(sw_output.get_output(0), output_size, trt.Weights(weights_dict["head.weight"].cpu().numpy().astype(np.float16).flatten()), trt.Weights(weights_dict["head.bias"].cpu().numpy().astype(np.float16).flatten()))
        network.mark_output(output.get_output(0))
        print("Before build_engine")
        engine = builder.build_engine(network, builder_config)
        print("After build_engine")
        return engine


def main():

    args, config = parse_option()

    weights_dict = load_weights(config.MODEL.RESUME, config)
    
    with build_engine(config, args, weights_dict) as engine:
        TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Serializing Engine...")
        serialized_engine = engine.serialize()
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format(args.output))
        with open(args.output, "wb") as fout:
            fout.write(serialized_engine)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")

if __name__ == "__main__":
    main()

