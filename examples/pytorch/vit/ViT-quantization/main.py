# coding=utf-8
# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
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
from __future__ import absolute_import, division, print_function
from email.policy import strict

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

import sys 
sys.path.insert(0, "./ViT-pytorch")
from models.modeling import CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.dist_util import get_world_size

from data import build_loader
from config import get_config
import quant_utils
from vit_int8 import VisionTransformerINT8

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Knowledge_Distillation_Loss(torch.nn.Module):
    def __init__(self, scale, T = 3):
        super(Knowledge_Distillation_Loss, self).__init__()
        self.KLdiv = torch.nn.KLDivLoss()
        self.T = T
        self.scale = scale

    def get_knowledge_distillation_loss(self, output_student, output_teacher):
        loss_kl = self.KLdiv(torch.nn.functional.log_softmax(output_student / self.T, dim=1), torch.nn.functional.softmax(output_teacher / self.T, dim=1))

        loss = loss_kl
        return self.scale * loss

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to %s", model_checkpoint)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 10 if args.dataset == "cifar10" else 1000

    model = VisionTransformerINT8(config, args.img_size, zero_head=False, num_classes=num_classes)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


@torch.no_grad()
def valid(args, config, model, test_loader):
    # Validation!
    eval_losses = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(test_loader):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch

        logits, _ = model(x)

        eval_loss = loss_fct(logits, y)
        acc1, acc5 = accuracy(logits, y, topk=(1, 5))

        eval_losses.update(eval_loss.item(), y.size(0))
        acc1_meter.update(acc1.item(), y.size(0))
        acc5_meter.update(acc5.item(), y.size(0))

        
        if step % config.PRINT_FREQ == 0:
            logger.info(
                f'Test: [{step}/{len(test_loader)}]\t'
                f'Loss {eval_losses.val:.4f} ({eval_losses.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    return acc1_meter.avg

def calib(args, config, model):
    """ Calibrate the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    
    dataset_train, dataset_val, train_loader, test_loader = build_loader(config, args)
    # Calibration
    quant_utils.configure_model(model, args, calib=True)
    model.eval()
    quant_utils.enable_calibration(model)
    # Run forward passes on a sample of the training set
    for step, (samples, targets) in enumerate(tqdm(train_loader, desc='Calibration', total=args.num_calib_batch)):
        if step > args.num_calib_batch:
            break
        samples = samples.to(args.device)
        outputs = model(samples)
    quant_utils.finish_calibration(model, args)

    # model.load_state_dict(torch.load('checkpoint/{}_{}_{}.pth'.format(args.model_type, args.quant_mode, args.percentile)))

    quant_utils.configure_model(model, args, calib=False)
    if args.local_rank in [-1, 0]:
        accuracy = valid(args, config, model, test_loader)
    logger.info("Test Accuracy: \t%f" %accuracy)

    output_model_path = os.path.join(args.calib_output_path, '{}_calib.pth'.format(args.model_type))
    if not os.path.exists(args.calib_output_path):
        os.mkdir(args.calib_output_path)
    torch.save(model.state_dict(), output_model_path)
    logger.info(f'Model is saved to {output_model_path}')

def train(args, config):
    num_classes = 1000
    model_config = CONFIGS[args.model_type]
    model = VisionTransformerINT8(model_config, args.img_size, zero_head=False, num_classes=num_classes)
    model_ckpt = torch.load(args.pretrained_dir, map_location="cpu")
    model.load_state_dict(model_ckpt["model"] if "model" in model_ckpt else model_ckpt, strict=False)
    model.cuda()
    model.train()
    
    teacher = None
    dis_loss = None
    if args.distill:
        teacher = VisionTransformerINT8(model_config, args.img_size, zero_head=False, num_classes=num_classes)
        teacher.load_from(np.load(args.teacher))
        dis_loss = Knowledge_Distillation_Loss(scale=args.distillation_loss_scale).cuda()
        teacher.cuda()
        teacher.eval()
        quant_utils.set_quantizer_by_name(teacher, [''], _disabled=True)

    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    # train_loader, test_loader = get_loader(args)
    dataset_train, dataset_val, train_loader, test_loader = build_loader(config, args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.qat_lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    
    print('args.qat_lr: %.6f' % (args.qat_lr))
    print('optimizer.lr: %.6f' % optimizer.state_dict()['param_groups'][0]['lr'])
    t_total = args.num_steps
    # if args.decay_type == "cosine":
    #     scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # else:
    #     scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    
    print('查看optimizer.param_groups结构:')
    i_list=[i for i in optimizer.param_groups[0].keys()]
    print(i_list)
    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization epochs = %d", args.num_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    quant_utils.configure_model(model, args, calib=False)
    for epoch_i in range(args.num_epochs):
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            outputs, loss = model(x, y)

            if teacher:
                with torch.no_grad():
                    teacher_outputs, _ = teacher(x)
                loss_t = dis_loss.get_knowledge_distillation_loss(outputs, teacher_outputs)
                loss = loss + loss_t

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # scheduler.step()
                optimizer.step()
                lr = optimizer.param_groups[0]['lr']
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "EPOCH [%d/%d] (%d / %d Steps) (loss=%2.5f) (lr=%.7f)" % 
                        (epoch_i, args.num_epochs, global_step, len(epoch_iterator), losses.val, lr)
                )
        
        if args.local_rank in [-1, 0]:
            accuracy = valid(args, config, model, test_loader)
            if best_acc < accuracy:
                save_model(args, model)
                best_acc = accuracy
            model.train()

                # if global_step % t_total == 0:
                #     break
        losses.reset()
        # if global_step % t_total == 0:
        #     break

    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")

def parse_option():
    parser = argparse.ArgumentParser()
    # Required parameters
    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
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
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--calib', action='store_true', help='Perform calibration only')
    parser.add_argument('--train', action='store_true', help='Perform training only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--num-calib-batch', type=int, default=10, help='Number of batches for calibration. 0 will disable calibration.')
    parser.add_argument('--calib-batchsz', type=int, default=8, help='Batch size when doing calibration')
    parser.add_argument('--calib-output-path', type=str, default='calib-checkpoint', help='Output directory to save calibrated model')

    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs to run QAT fintuning.")
    parser.add_argument("--qat-lr", type=float, default=1e-6, help="learning rate for QAT.")
    parser.add_argument("--distill", action='store_true', help='Using distillation')
    parser.add_argument("--teacher", type=str, help='teacher model path')
    parser.add_argument('--distillation_loss_scale', type=float, default=10000., help="scale applied to distillation component of loss")

    # distributed training
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=384, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=2000, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    quant_utils.add_arguments(parser)
    args, unparsed = parser.parse_known_args()
    if args.quant_mode is not None:
        args = quant_utils.set_args(args)
    quant_utils.set_default_quantizers(args)

    config = get_config(args)

    return args, config


def main():
    
    args, config = parse_option()
    # print(config.dump())

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Calibration
    if args.calib:
        args, model = setup(args)
        calib(args, config, model)
    
    # Quantization-Aware Training
    if args.train:
        # args, model = setup(args)
        train(args, config)


if __name__ == "__main__":
    main()
