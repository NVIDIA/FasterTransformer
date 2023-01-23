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

from torch._C import _nvtx

import sys
sys.path.insert(0, "./Swin-Transformer-Quantization")

# from config_modified_int8 import get_config
from SwinTransformer.config import get_config
from SwinTransformer.models import build_model
from data import build_val_loader
from SwinTransformer.optimizer import build_optimizer
from SwinTransformer.logger import create_logger
from SwinTransformer.utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

from SwinTransformerWeightTransposeQKVWeight import SwinTransformerWeightTransposeQKVWeight

def saveToTxt(x, name, clean=False):
    if clean :
        with open("tmp2/"+name, 'w+') as fout:
            xx = x.reshape([-1])
            for i in xx:
                fout.write("{}\n".format(i))
    else:
        with open("tmp2/"+name, 'a+') as fout:
            shape = x.shape
            fout.write("{}\n".format(len(shape)))
            fout.write(" ".join([str(s) for s in shape])+"\n")
            xx = x.reshape([-1])
            for i in xx:
                fout.write("{}\n".format(i))


try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


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
    parser.add_argument('--version', type=int, default=1, help='version of swin', )
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
    parser.add_argument('--resume', help='resume from checkpoint')
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
    parser.add_argument('--fp16', action='store_true', help='Using FP16 precision instead of FP32')
    parser.add_argument('--num-calib-batch', type=int, default=4, help='Number of batches for calibration. 0 will disable calibration.')
    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def main(args, config):
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()

    optimizer = build_optimizer(config, model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = None
    if config.MODEL.RESUME:
        # max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint, strict=False)
        logger.info(msg)
        del checkpoint

    dataset_val, data_loader_val = build_val_loader(config)
    acc1, acc5, loss = validate(config, args, data_loader_val, model)
    logger.info(f"Accuracy of resumed network on the {len(dataset_val)} test images: {acc1:.1f}%")
    return


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
        
        
    version = args.version    
    th_path = args.th_path
    depths_tensor = torch.tensor(depths, dtype=torch.int)
    num_heads_tensor = torch.tensor(num_heads, dtype=torch.int)
    layer_num = len(depths)
    max_batch = config.DATA.BATCH_SIZE
    img_size = config.DATA.IMG_SIZE
    torch.classes.load_library(th_path)
    sw_weights = SwinTransformerWeightTransposeQKVWeight(layer_num, window_size, depths, num_heads, th_path, model.state_dict(), version)
    if args.fp16:
        model.half()
        sw_weights.to_half()
    else:
        sw_weights.to_float32()
    sw_weights.to_cuda()

    ##run pytorch op
    try:
        swin_transformer = torch.classes.SwinTransformer.Class(sw_weights.weights, depths_tensor, num_heads_tensor, max_batch, img_size, patch_size, in_chans, embed_dim, window_size, ape, patch_norm, layer_num, mlp_ratio, qkv_bias, qk_scale, version)
    except:
        # legacy ths for 20.03 image
        swin_transformer = torch.classes.SwinTransformerClass(sw_weights.weights, depths_tensor, num_heads_tensor, max_batch, img_size, patch_size, in_chans, embed_dim, window_size, ape, patch_norm, layer_num, mlp_ratio, qkv_bias, qk_scale, version)

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        if args.fp16:
            images = images.half()
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        # output_th = model(images)
        swin_tansformer_output = swin_transformer.forward(images)
        output = model.head(swin_tansformer_output)
        # diff = output - output_th
        # print(diff.mean(), diff.max(), diff.min())

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
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    seed = config.SEED 
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # if dist.get_rank() == 0:
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    main(args, config)
