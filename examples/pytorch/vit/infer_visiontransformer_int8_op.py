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
import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import accuracy, AverageMeter

from VisionTransformerINT8WeightLoader import ViTINT8WeightLoader

import sys
sys.path.insert(0, "./ViT-quantization/ViT-pytorch")


# from config import get_config
# from models import build_model
from models.modeling import CONFIGS

sys.path.insert(0, "./ViT-quantization")
from vit_int8 import VisionTransformerINT8
import quant_utils
from config import get_config
from data import build_val_loader

test_time = 100
warmup_time = 10

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    print(config)

    num_classes = 1000 if args.dataset == "ImageNet" else 100

    model = VisionTransformerINT8(config, args.img_size, zero_head=False, num_classes=num_classes)
    model.load_state_dict(torch.load(args.calibrated_dir))

    quant_utils.configure_model(model, args, calib=False)
    model.to(args.device)

    return config, model

def parse_option():
    parser = argparse.ArgumentParser('ViT evaluation script', add_help=False)

    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                             "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--img_size", default=384, type=int, 
                        help="Resolution size")
    parser.add_argument("--dataset", choices=["ImageNet"], default="ImageNet",
                        help="Which downstream task.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--calibrated_dir", type=str, default="checkpoint/ViT-B_16_calib.pth",
                        help="Where to search for calibrated ViT models.")
    parser.add_argument("--data-path", type=str, default="/workspace/imagenet",
                        help="Root directory for datasets.")

    # easy config modification
    parser.add_argument('--th-path', type=str, help='path to pytorch library', required=True)
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--int8-mode', type=int, choices=[1,2,3], default=2,
                        help="Which int8 mode to use, choices=[1,2,3], default=2")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    parser.add_argument("--validate", action="store_true", help='If true, validate on ImageNet, else just profile')

    quant_utils.add_arguments(parser)
    args, unparsed = parser.parse_known_args()
    if args.quant_mode is not None:
        args = quant_utils.set_args(args)
    quant_utils.set_default_quantizers(args)

    config = get_config(args)

    if args.quant_mode == 'ft1':
        args.int8_mode = 1
    elif args.quant_mode == 'ft2':
        args.int8_mode = 2
    else:
        raise NotImplementedError("For ViT-INT8, we only support ft1/ft2 as quant_mode")

    return args, config


def main(args, data_config):

    model_cfg, model = setup(args)

    if args.validate:
        validate(args, data_config, model_cfg, model)
    else:
        validate_with_random_data(args, model_cfg, model)


@torch.no_grad()
def validate(args, data_config, config, model):
    dataset, data_loader = build_val_loader(data_config)
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    th_path = args.th_path
    patch_size = config.patches.size[0]
    num_heads = config.transformer.num_heads
    layer_num = config.transformer.num_layers
    inter_size = config.transformer.mlp_dim
    embed_dim = config.hidden_size
    max_batch = args.batch_size
    img_size = args.img_size
    int8_mode = args.int8_mode
    in_chans = 3
    model.half()
    
    vit_weights = ViTINT8WeightLoader(layer_num, args.img_size, patch_size, model.state_dict())
    vit_weights.to_int8(args.th_path)
    vit_weights.to_cuda()

    weights = vit_weights.listed_weights()

    torch.classes.load_library(th_path)
    try:
        vit = torch.classes.VisionTransformerINT8.Class(weights, 
                                                        max_batch, 
                                                        img_size, 
                                                        patch_size, 
                                                        in_chans,
                                                        embed_dim, 
                                                        num_heads,
                                                        inter_size,
                                                        layer_num,
                                                        int8_mode
                                                        )
    except:
        # legacy ths for 20.03 image
        vit = torch.classes.VisionTransformerINT8Class(weights,
                                                        max_batch, 
                                                        img_size, 
                                                        patch_size, 
                                                        in_chans,
                                                        embed_dim, 
                                                        num_heads,
                                                        inter_size,
                                                        layer_num,
                                                        int8_mode
                                                        )

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images_half = torch.tensor(images, dtype=torch.half)
        images_half = images_half.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        op_tmp = vit.forward(images_half)
        # op_tmp,_ = model.transformer(images_half)
        output = model.head(op_tmp[:, 0])

        # output_th, _ = model(images_half)
        # diff = abs(output - output_th).cpu().numpy()
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

        if idx % data_config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            print(
                f'Test: [{idx:4}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    print(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

@torch.no_grad()
def run_vitnv_op(args, config, model, images):
    th_path = args.th_path
    patch_size = config.patches.size[0]
    num_heads = config.transformer.num_heads
    layer_num = config.transformer.num_layers
    inter_size = config.transformer.mlp_dim
    embed_dim = config.hidden_size
    max_batch = args.batch_size
    img_size = args.img_size
    int8_mode = args.int8_mode
    with_cls_token = config.classifier == 'token'
    in_chans = 3
    model.half()
    torch.classes.load_library(th_path)
    vit_weights = ViTINT8WeightLoader(layer_num, args.img_size, patch_size, model.state_dict(), config.classifier)
    vit_weights.to_int8(args.th_path)
    vit_weights.to_cuda()

    weights = vit_weights.listed_weights()

    ##run pytorch op 
    try:
        vit = torch.classes.VisionTransformerINT8.Class(weights, 
                                                        max_batch, 
                                                        img_size, 
                                                        patch_size, 
                                                        in_chans,
                                                        embed_dim, 
                                                        num_heads,
                                                        inter_size,
                                                        layer_num,
                                                        int8_mode,
                                                        with_cls_token
                                                        )
    except:
        # legacy ths for 20.03 image
        vit = torch.classes.VisionTransformerINT8Class(weights,
                                                        max_batch, 
                                                        img_size, 
                                                        patch_size, 
                                                        in_chans,
                                                        embed_dim, 
                                                        num_heads,
                                                        inter_size,
                                                        layer_num,
                                                        int8_mode,
                                                        with_cls_token
                                                        )
    # warm up
    for i in range(warmup_time):
        op_tmp = vit.forward(images)
        # op_output = model.head(op_tmp[:,0])

    torch.cuda.synchronize()
    op_begin = time.time()
    #_nvtx.rangePushA("op")
    for i in range(test_time):
        op_tmp = vit.forward(images)
        op_output = model.head(op_tmp[:,0])
    #_nvtx.rangePop()
    torch.cuda.synchronize()
    op_end = time.time()
    op_output = op_output.cpu().numpy() 
    print("INT8 op time : ", (op_end - op_begin)/test_time*1000.0, "ms")

    return op_output


@torch.no_grad()
def run_torch(model, images, mark):
    # warm up
    # for i in range(warmup_time):
    #     output = model(images)

    # torch.cuda.synchronize()
    # torch_start = time.time()
    #_nvtx.rangePushA("torch")
    # for i in range(test_time):
    torch_output = model(images)
    #_nvtx.rangePop()
    # torch.cuda.synchronize()
    # torch_end = time.time()
    torch_output = torch_output[0].cpu().numpy()
    # print(mark + " time : ", (torch_end - torch_start)/test_time*1000.0, "ms")

    return torch_output

@torch.no_grad()
def validate_with_random_data(args, model_cfg, model):
    model.eval()
    
    max_batch = args.batch_size
    img_size = args.img_size
    in_chans = 3
    image = np.random.rand(1, in_chans, img_size, img_size)
    images = np.repeat(image, max_batch, axis=0)
    images_half = torch.tensor(images, dtype=torch.half).cuda(non_blocking=True)
    ##run original swin-transformer

    # traced_module_float = torch.jit.trace(model, images_float)
    # FP32_torch_traced_output = run_torch(traced_module_float, images_float, "FP32 torch trace")
    model.half()
    INT8_torch_output = run_torch(model, images_half, "INT8 torch")
    print(INT8_torch_output.shape)

    # run pytorch op
    INT8_op_output = run_vitnv_op(args, model_cfg, model, images_half)
    print(INT8_op_output.shape)

    # diff = abs(FP16_torch_traced_output - FP16_op_output)
    diff = abs(INT8_torch_output - INT8_op_output)
    print("INT8_torch_output vs INT8_op_output , avg diff : ", diff.mean((1)), "max diff : ", diff.max((1)))

if __name__ == '__main__':
    args, data_config = parse_option()

    seed = args.seed #+ int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=datetime.timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    main(args, data_config)
