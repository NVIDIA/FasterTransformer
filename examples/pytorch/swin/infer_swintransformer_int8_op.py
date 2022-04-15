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

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from tqdm import tqdm

from torch._C import _nvtx

import sys
sys.path.insert(0, "./Swin-Transformer-Quantization")

# from config_modified_int8 import get_config
from SwinTransformer.config import get_config
from models import build_model
from data import build_val_loader
from SwinTransformer.optimizer import build_optimizer
from SwinTransformer.logger import create_logger
from SwinTransformer.utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from SwinTransformerINT8Weight import SwinTransformerINT8Weight
import quant_utils

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
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
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
    if args.quant_mode is not None:
        args = quant_utils.set_args(args)
    quant_utils.set_default_quantizers(args)

    config = get_config(args)

    return args, config


def main(args, config):
    
    

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    # logger.info(str(model))
    # quant_utils.print_quant_summary(model)

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    '''
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    '''
    lr_scheduler = None
    if config.MODEL.RESUME:
        # max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        msg = model_without_ddp.load_state_dict(checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint, strict=False)
        logger.info(msg)
        del checkpoint

    if args.profile:
        quant_utils.configure_model(model, args, calib=False)
        validate_with_random_data(config, args, model_without_ddp)
        return

    if args.eval:
        dataset_val, data_loader_val = build_val_loader(config)
        quant_utils.configure_model(model, args, calib=False)
        # validate_with_random_data(config, model_without_ddp)
        acc1, acc5, loss = validate(config, args, data_loader_val, model_without_ddp)
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
    int8_mode = args.int8_mode
    if config.MODEL.SWIN.QK_SCALE is not None:
        qk_scale = config.MODEL.SWIN.QK_SCALE
    else:
        qk_scale = 1.0
    sw_weights = SwinTransformerINT8Weight(layer_num, window_size, depths, num_heads, th_path, model.state_dict())
    # sw_weights.to_half()
    # sw_weights.to_cuda()
    torch.classes.load_library(th_path)
    try:
        swin_transformer = torch.classes.SwinTransformerINT8.Class(sw_weights.weights, int8_mode, depths_tensor, num_heads_tensor, max_batch, img_size, patch_size, in_chans, embed_dim, window_size, ape, patch_norm, layer_num, mlp_ratio, qkv_bias, qk_scale)
    except:
        # legacy ths for 20.03 image
        swin_transformer = torch.classes.SwinTransformerINT8Class(sw_weights.weights, int8_mode, depths_tensor, num_heads_tensor, max_batch, img_size, patch_size, in_chans, embed_dim, window_size, ape, patch_norm, layer_num, mlp_ratio, qkv_bias, qk_scale)

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images_half = torch.tensor(images, dtype=torch.half)
        images_half = images_half.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        # output_th = model(images)
        swin_tansformer_output = swin_transformer.forward(images_half)
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

@torch.no_grad()
def run_swintransformernv_op(config, args, model, images, use_fp16):
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
    int8_mode = args.int8_mode
    if config.MODEL.SWIN.QK_SCALE is not None:
        qk_scale = config.MODEL.SWIN.QK_SCALE
    else:
        qk_scale = 1.0
    torch.classes.load_library(th_path)
    sw_weights = SwinTransformerINT8Weight(layer_num, window_size, depths, num_heads, th_path, model.state_dict())
    # if use_fp16:
    #     sw_weights.to_half()
    # sw_weights.to_cuda()

    ##run original swin-transformer
    test_time = 100
    warmup_time = 10

    ##run pytorch op 
    try:
        swin_transformer = torch.classes.SwinTransformerINT8.Class(sw_weights.weights, int8_mode, depths_tensor, num_heads_tensor, max_batch, img_size, patch_size, in_chans, embed_dim, window_size, ape, patch_norm, layer_num, mlp_ratio, qkv_bias, qk_scale)
    except:
        # legacy ths for 20.03 image
        swin_transformer = torch.classes.SwinTransformerINT8Class(sw_weights.weights, int8_mode, depths_tensor, num_heads_tensor, max_batch, img_size, patch_size, in_chans, embed_dim, window_size, ape, patch_norm, layer_num, mlp_ratio, qkv_bias, qk_scale)

    # warm up
    for i in range(warmup_time):
        op_embedding = swin_transformer.forward(images)
        op_output = model.head(op_embedding)

    torch.cuda.synchronize()
    op_begin = time.time()
    
    for i in range(test_time):
        # print('Before {}: {} GB'.format(i, torch.cuda.max_memory_allocated()/1024/1024/1024))
        # print('Before res{}: {} GB'.format(i, torch.cuda.max_memory_reserved()/1024/1024/1024))
        _nvtx.rangePushA("op {}".format(i))
        op_embedding = swin_transformer.forward(images)
        _nvtx.rangePop()
        # print('After  {}: {} GB'.format(i, torch.cuda.max_memory_allocated()/1024/1024/1024))
        # print('After  res{}: {} GB'.format(i, torch.cuda.max_memory_reserved()/1024/1024/1024))
        # torch.cuda.empty_cache()
        # op_output = model.head(op_embedding)
    
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
    ##run original swin-transformer
    test_time = 100
    warmup_time = 10
    '''
    # warm up
    for i in range(warmup_time):
        output = model(images_float)

    torch_end = time.time()
    for i in range(test_time):
        FP32_torch_output = model(images_float)

    FP32_torch_output = FP32_torch_output.cpu().numpy()
    print("FP32 input torch time : ", (time.time() - torch_end)/test_time*1000.0, "ms")
    '''
    # warm up
    # for i in range(warmup_time):
    #     output = model(images_half)

    # torch.cuda.synchronize()
    # torch_start = time.time()
    # for i in range(test_time):
    INT8_torch_output = model(images_half)

    # torch.cuda.synchronize()
    # torch_end = time.time()
    INT8_torch_output = INT8_torch_output.cpu().numpy()
    # print("FP16 input torch time : ", (torch_end - torch_start)/test_time*1000.0, "ms")

    '''
    diff = abs(FP32_torch_output - FP16_torch_output)
    print("FP32_torch_output vs FP16_torch_output , avg diff : ", diff.mean(), "max diff : ", diff.max())
    '''

    ## run pytorch op
    INT8_op_output = run_swintransformernv_op(config, args, model, images_half, True)
    # diff_int8 = abs(INT8_op_output[0, :] - INT8_op_output[1, :])
    # print("diff between instance 0 and 1:", diff_int8.mean())
    diff = abs(INT8_torch_output - INT8_op_output)
    print("INT8_torch_output vs INT8_op_output , avg diff : ", diff.mean((1)), "max diff : ", diff.max((1)))



@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


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
    # torch.cuda.set_device(config.LOCAL_RANK)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # torch.distributed.barrier()

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

    # print config
    # logger.info(config.dump())

    main(args, config)
