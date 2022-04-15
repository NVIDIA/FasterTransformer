# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import sys
sys.path.insert(0, "./Swin-Transformer-Quantization/SwinTransformer")

from config import get_config
from models import build_model
from SwinTransformerWeightTransposeQKVWeight import SwinTransformerWeightTransposeQKVWeight

#from torch._C import _nvtx

test_time = 100
warmup_time = 10

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer evaluation script', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                       'full: cache all data, '
                       'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    # easy config modification
    parser.add_argument('--th-path', type=str, help='path to pytorch library')
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(args, config):

    model = build_model(config)
    model.cuda()

    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint, strict=False)
    validate_with_random_data(args, config, model)

@torch.no_grad()
def run_swintransformernv_op(args, config, model, images, use_fp16):
    th_path = args.th_path
    depths = config.MODEL.SWIN.DEPTHS
    depths_tensor = torch.tensor(depths, dtype=torch.int)
    num_heads = config.MODEL.SWIN.NUM_HEADS
    num_heads_tensor = torch.tensor(num_heads, dtype=torch.int)
    layer_num = len(depths)
    window_size = config.MODEL.SWIN.WINDOW_SIZE
    max_batch = config.DATA.BATCH_SIZE
    img_size = config.DATA.IMG_SIZE
    patch_size = config.MODEL.SWIN.PATCH_SIZE
    in_chans = config.MODEL.SWIN.IN_CHANS
    embed_dim = config.MODEL.SWIN.EMBED_DIM
    ape = config.MODEL.SWIN.APE
    patch_norm = config.MODEL.SWIN.PATCH_NORM
    mlp_ratio = config.MODEL.SWIN.MLP_RATIO
    qkv_bias = config.MODEL.SWIN.QKV_BIAS
    if config.MODEL.SWIN.QK_SCALE is not None:
        qk_scale = config.MODEL.SWIN.QK_SCALE
    else:
        qk_scale = 1.0
    torch.classes.load_library(th_path)
    sw_weights = SwinTransformerWeightTransposeQKVWeight(layer_num, window_size, depths, num_heads, th_path, model.state_dict())
    if use_fp16:
        sw_weights.to_half()
        model.half()
    sw_weights.to_cuda()

    ##run pytorch op 
    try:
        swin_transformer = torch.classes.SwinTransformer.Class(sw_weights.weights, depths_tensor, num_heads_tensor, max_batch, img_size, patch_size, in_chans, embed_dim, window_size, ape, patch_norm, layer_num, mlp_ratio, qkv_bias, qk_scale)
    except:
        # legacy ths for 20.03 image
        swin_transformer = torch.classes.SwinTransformerClass(sw_weights.weights, depths_tensor, num_heads_tensor, max_batch, img_size, patch_size, in_chans, embed_dim, window_size, ape, patch_norm, layer_num, mlp_ratio, qkv_bias, qk_scale)
    # warm up
    for i in range(warmup_time):
        op_embedding = swin_transformer.forward(images)
        op_output = model.head(op_embedding)

    torch.cuda.synchronize()
    op_begin = time.time()
    #_nvtx.rangePushA("op")
    for i in range(test_time):
        op_embedding = swin_transformer.forward(images)
        op_output = model.head(op_embedding)
    #_nvtx.rangePop()
    torch.cuda.synchronize()
    op_end = time.time()
    op_output = op_output.cpu().numpy() 
    if use_fp16:
        print("FP16 op time : ", (op_end - op_begin)/test_time*1000.0, "ms")
    else:
        print("FP32 op time : ", (op_end - op_begin)/test_time*1000.0, "ms")

    return op_output


@torch.no_grad()
def run_torch(model, images, mark):
    # warm up
    for i in range(warmup_time):
        output = model(images)

    torch.cuda.synchronize()
    torch_start = time.time()
    #_nvtx.rangePushA("torch")
    for i in range(test_time):
        torch_output = model(images)
    #_nvtx.rangePop()
    torch.cuda.synchronize()
    torch_end = time.time()
    torch_output = torch_output.cpu().numpy()
    print(mark + " time : ", (torch_end - torch_start)/test_time*1000.0, "ms")
    return torch_output

@torch.no_grad()
def validate_with_random_data(args, config, model):
    model.eval()
    
    max_batch = config.DATA.BATCH_SIZE
    img_size = config.DATA.IMG_SIZE
    in_chans = config.MODEL.SWIN.IN_CHANS
    image = np.random.rand(1, in_chans, img_size, img_size)
    images = np.repeat(image, max_batch, axis=0)
    images_half = torch.tensor(images, dtype=torch.half)
    images_float = torch.tensor(images, dtype=torch.float)
    images_half = images_half.cuda(non_blocking=True)
    images_float = images_float.cuda(non_blocking=True)
    ##run original swin-transformer

    # run pytorch op
    FP32_op_output = run_swintransformernv_op(args, config, model, images_float, False)

    traced_module_float = torch.jit.trace(model, images_float)
    FP32_torch_traced_output = run_torch(traced_module_float, images_float, "FP32 torch trace")
    FP32_torch_output = run_torch(model, images_float, "FP32 torch")

    FP16_op_output = run_swintransformernv_op(args, config, model, images_half, True)

    traced_module_half = torch.jit.trace(model, images_half)
    FP16_torch_traced_output = run_torch(traced_module_half, images_half, "FP16 torch trace")
    FP16_torch_output = run_torch(model, images_half, "FP16 torch")



    diff = abs(FP32_torch_traced_output - FP32_op_output)
    print("FP32_torch_traced_output vs FP32_op_output , avg diff : ", diff.mean(), "max diff : ", diff.max())
    diff = abs(FP16_torch_traced_output - FP16_op_output)
    print("FP16_torch_traced_output vs FP16_op_output , avg diff : ", diff.mean(), "max diff : ", diff.max())

if __name__ == '__main__':
    args, config = parse_option()

    seed = config.SEED + int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    main(args, config)
