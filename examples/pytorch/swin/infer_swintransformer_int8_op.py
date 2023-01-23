# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from tqdm import tqdm

import sys
sys.path.insert(0, "./Swin-Transformer-Quantization")

from SwinTransformer.config import get_config
from models import build_model
from data import build_val_loader
from SwinTransformer.utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from SwinTransformerINT8Weight import SwinTransformerINT8Weight
import quant_utils

test_time = 100
warmup_time = 10

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
    parser.add_argument('--version', type=int, default=1, help='version of swin', choices=[1, 2])
    parser.add_argument('--disable_amp', type=bool, default=True, help='disable amp', )
    parser.add_argument('--fused_window_process', type=bool, default=False, help='whether use fused window process', )
    parser.add_argument('--th-path', type=str, help='path to pytorch library')
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', required=True, help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--profile', action='store_true', help='Perform profiling only, with some random data')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--calib', action='store_true', help='Perform calibration only')
    parser.add_argument('--train', action='store_true', help='Perform training only')
    parser.add_argument('--int8-mode', type=int, help='int8 mode', choices=[1, 2])
    parser.add_argument('--num-calib-batch', type=int, default=4, help='Number of batches for calibration. 0 will disable calibration.')
    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    quant_utils.add_arguments(parser)
    args, unparsed = parser.parse_known_args()
    args = quant_utils.set_args(args)
    quant_utils.set_default_quantizers(args)

    config = get_config(args)

    return args, config


def main(args, config):
    model = build_model(config)
    model.cuda()

    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint, strict=False)
    print(msg)
    del checkpoint

    if args.profile:
        quant_utils.configure_model(model, args, calib=False)
        validate_with_random_data(config, args, model)
    elif args.eval:
        dataset_val, data_loader_val = build_val_loader(config)
        quant_utils.configure_model(model, args, calib=False)
        acc1, acc5, loss = validate(config, args, data_loader_val, model)
        print(f"Accuracy of resumed network on the {len(dataset_val)} test images: {acc1:.1f}%")


@torch.no_grad()
def validate(config, args, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    if args.version == 1:
        depths = config.MODEL.SWIN.DEPTHS
        num_heads = config.MODEL.SWIN.NUM_HEADS
        window_size = config.MODEL.SWIN.WINDOW_SIZE
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
    elif args.version == 2:
        depths = config.MODEL.SWINV2.DEPTHS
        num_heads = config.MODEL.SWINV2.NUM_HEADS
        window_size = config.MODEL.SWINV2.WINDOW_SIZE
        patch_size = config.MODEL.SWINV2.PATCH_SIZE
        in_chans = config.MODEL.SWINV2.IN_CHANS
        embed_dim = config.MODEL.SWINV2.EMBED_DIM
        ape = config.MODEL.SWINV2.APE
        patch_norm = config.MODEL.SWINV2.PATCH_NORM
        mlp_ratio = config.MODEL.SWINV2.MLP_RATIO
        qkv_bias = config.MODEL.SWINV2.QKV_BIAS
        qk_scale = 1.0

    int8_mode = args.int8_mode
    version = args.version
    th_path = args.th_path
    depths_tensor = torch.tensor(depths, dtype=torch.int)
    num_heads_tensor = torch.tensor(num_heads, dtype=torch.int)
    layer_num = len(depths)
    max_batch = config.DATA.BATCH_SIZE
    img_size = config.DATA.IMG_SIZE
    sw_weights = SwinTransformerINT8Weight(layer_num, window_size, depths, num_heads, th_path, model.state_dict(), version=version)

    torch.classes.load_library(th_path)
    try:
        swin_transformer = torch.classes.SwinTransformerINT8.Class(sw_weights.weights, int8_mode, depths_tensor, num_heads_tensor, max_batch, img_size, patch_size, in_chans, embed_dim, window_size, ape, patch_norm, layer_num, mlp_ratio, qkv_bias, qk_scale, version)
    except:
        # legacy ths for 20.03 image
        swin_transformer = torch.classes.SwinTransformerINT8Class(sw_weights.weights, int8_mode, depths_tensor, num_heads_tensor, max_batch, img_size, patch_size, in_chans, embed_dim, window_size, ape, patch_norm, layer_num, mlp_ratio, qkv_bias, qk_scale, version)

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images_half = images.half()
        images_half = images_half.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        swin_tansformer_output = swin_transformer.forward(images_half)
        output = model.head(swin_tansformer_output.float())

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            print(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    print(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

@torch.no_grad()
def run_swintransformernv_op(config, args, model, images, use_fp16):
    if args.version == 1:
        depths = config.MODEL.SWIN.DEPTHS
        num_heads = config.MODEL.SWIN.NUM_HEADS
        window_size = config.MODEL.SWIN.WINDOW_SIZE
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
    elif args.version == 2:
        depths = config.MODEL.SWINV2.DEPTHS
        num_heads = config.MODEL.SWINV2.NUM_HEADS
        window_size = config.MODEL.SWINV2.WINDOW_SIZE
        patch_size = config.MODEL.SWINV2.PATCH_SIZE
        in_chans = config.MODEL.SWINV2.IN_CHANS
        embed_dim = config.MODEL.SWINV2.EMBED_DIM
        ape = config.MODEL.SWINV2.APE
        patch_norm = config.MODEL.SWINV2.PATCH_NORM
        mlp_ratio = config.MODEL.SWINV2.MLP_RATIO
        qkv_bias = config.MODEL.SWINV2.QKV_BIAS
        qk_scale = 1.0
    int8_mode = args.int8_mode
    version = args.version
    th_path = args.th_path
    depths_tensor = torch.tensor(depths, dtype=torch.int)
    num_heads_tensor = torch.tensor(num_heads, dtype=torch.int)
    layer_num = len(depths)
    max_batch = config.DATA.BATCH_SIZE
    img_size = config.DATA.IMG_SIZE
    print('window_size', window_size, img_size)
    
    torch.classes.load_library(th_path)
    sw_weights = SwinTransformerINT8Weight(layer_num, window_size, depths, num_heads, th_path, model.state_dict(), version=version)

    ##run pytorch op 
    try:
        swin_transformer = torch.classes.SwinTransformerINT8.Class(sw_weights.weights, int8_mode, depths_tensor, num_heads_tensor, max_batch, img_size, patch_size, in_chans, embed_dim, window_size, ape, patch_norm, layer_num, mlp_ratio, qkv_bias, qk_scale, version)
    except:
        # legacy ths for 20.03 image
        swin_transformer = torch.classes.SwinTransformerINT8Class(sw_weights.weights, int8_mode, depths_tensor, num_heads_tensor, max_batch, img_size, patch_size, in_chans, embed_dim, window_size, ape, patch_norm, layer_num, mlp_ratio, qkv_bias, qk_scale, version)

    # warm up
    for i in range(warmup_time):
        op_embedding = swin_transformer.forward(images)
        op_output = model.head(op_embedding.float())

    torch.cuda.synchronize()
    op_begin = time.time()
    
    for i in range(test_time):
        op_embedding = swin_transformer.forward(images)
    
    torch.cuda.synchronize()
    op_end = time.time()
    op_output = op_output.cpu().numpy() 
    if use_fp16:
        print("INT8 op time : ", (op_end - op_begin)/test_time*1000.0, "ms")
    else:
        print("INT8 op time : ", (op_end - op_begin)/test_time*1000.0, "ms")

    return op_output


@torch.no_grad()
def validate_with_random_data(config, args, model):
    model.eval()


    max_batch = config.DATA.BATCH_SIZE
    img_size = config.DATA.IMG_SIZE
    in_chans = config.MODEL.SWIN.IN_CHANS
    image = np.random.rand(1, in_chans, img_size, img_size)
    images = np.repeat(image, max_batch, axis=0)
    print(images.shape)
    images_half = torch.tensor(images, dtype=torch.half)
    images_float = torch.tensor(images, dtype=torch.float)
    images_half = images_half.cuda(non_blocking=True)
    images_float = images_float.cuda(non_blocking=True)

    test_time = 100
    warmup_time = 10
    INT8_op_output = run_swintransformernv_op(config, args, model, images_half, True)
    INT8_torch_output = model(images_float)
    INT8_torch_output = INT8_torch_output.cpu().numpy()

    diff = abs(INT8_torch_output - INT8_op_output)
    assert diff.mean() < 0.1, "[ERROR] SWIN INT8 Op TEST FAIL !"
    print("INT8_torch_output vs INT8_op_output , avg diff : ", diff.mean((1)), "max diff : ", diff.max((1)))


if __name__ == '__main__':
    args, config = parse_option()

    seed = config.SEED 
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    main(args, config)
