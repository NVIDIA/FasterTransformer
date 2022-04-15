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

import sys
sys.path.insert(0, "./ViT-quantization/ViT-pytorch")

# from config import get_config
# from models import build_model
from models.modeling import VisionTransformer, CONFIGS
from VisionTransformerWeightLoader import ViTWeightLoader

#from torch._C import _nvtx

test_time = 100
warmup_time = 10

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    print(config)

    model = VisionTransformer(config, args.img_size, zero_head=False, num_classes=1000)
    model.load_from(np.load(args.pretrained_dir))
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
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")

    # easy config modification
    parser.add_argument('--th-path', type=str, help='path to pytorch library', required=True)
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args, unparsed = parser.parse_known_args()


    return args


def main(args):

    config, model = setup(args)

    validate_with_random_data(args, config, model)

@torch.no_grad()
def run_vitnv_op(args, config, model, images, use_fp16):
    th_path = args.th_path
    patch_size = config.patches.size[0]
    num_heads = config.transformer.num_heads
    layer_num = config.transformer.num_layers
    inter_size = config.transformer.mlp_dim
    embed_dim = config.hidden_size
    max_batch = args.batch_size
    img_size = args.img_size
    with_cls_token = int(config.classifier == 'token')
    in_chans = 3
    torch.classes.load_library(th_path)
    vit_weights = ViTWeightLoader(layer_num, args.img_size, patch_size, args.pretrained_dir, config.classifier)
    if use_fp16:
        vit_weights.to_half()
        model.half()
    vit_weights.to_cuda()

    ##run pytorch op 
    try:
        vit = torch.classes.VisionTransformer.Class(vit_weights.weights, 
                                                                 max_batch, 
                                                                 img_size, 
                                                                 patch_size, 
                                                                 in_chans,
                                                                 embed_dim, 
                                                                 num_heads,
                                                                 inter_size,
                                                                 layer_num,
                                                                 with_cls_token
                                                                 )
    except:
        # legacy ths for 20.03 image
        vit = torch.classes.VisionTransformerClass(vit_weights.weights,
                                                                 max_batch, 
                                                                 img_size, 
                                                                 patch_size, 
                                                                 in_chans,
                                                                 embed_dim, 
                                                                 num_heads,
                                                                 inter_size,
                                                                 layer_num,
                                                                 with_cls_token
                                                                 )
    # warm up
    for i in range(warmup_time):
        op_tmp = vit.forward(images)
        op_output = model.head(op_tmp[:,0])

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
    torch_output = torch_output[0].cpu().numpy()
    print(mark + " time : ", (torch_end - torch_start)/test_time*1000.0, "ms")

    return torch_output

@torch.no_grad()
def validate_with_random_data(args, config, model):
    model.eval()
    
    max_batch = args.batch_size
    img_size = args.img_size
    in_chans = 3
    image = np.random.rand(1, in_chans, img_size, img_size)
    images = np.repeat(image, max_batch, axis=0)
    images_half = torch.tensor(images, dtype=torch.half)
    images_float = torch.tensor(images, dtype=torch.float)
    images_half = images_half.cuda(non_blocking=True)
    images_float = images_float.cuda(non_blocking=True)

    # run pytorch op
    FP32_op_output = run_vitnv_op(args, config, model, images_float, False)

    # traced_module_float = torch.jit.trace(model, images_float)
    # FP32_torch_traced_output = run_torch(traced_module_float, images_float, "FP32 torch trace")
    FP32_torch_output = run_torch(model, images_float, "FP32 torch")

    FP16_op_output = run_vitnv_op(args, config, model, images_half, True)

    # traced_module_half = torch.jit.trace(model, images_half)
    # FP16_torch_traced_output = run_torch(traced_module_half, images_half, "FP16 torch trace")
    FP16_torch_output = run_torch(model, images_half, "FP16 torch")

    # diff = abs(FP32_torch_traced_output - FP32_op_output)
    diff = abs(FP32_torch_output - FP32_op_output)
    print("FP32_torch_traced_output vs FP32_op_output , avg diff : ", diff.mean(), "max diff : ", diff.max())
    # diff = abs(FP16_torch_traced_output - FP16_op_output)
    diff = abs(FP16_torch_output - FP16_op_output)
    print("FP16_torch_traced_output vs FP16_op_output , avg diff : ", diff.mean(), "max diff : ", diff.max())

if __name__ == '__main__':
    args = parse_option()

    seed = args.seed + int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    main(args)
