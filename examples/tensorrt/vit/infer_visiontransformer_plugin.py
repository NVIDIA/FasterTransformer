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

import tensorrt as trt
import ctypes

import sys
sys.path.insert(0, "../../pytorch/vit/ViT-quantization/ViT-pytorch")

from models.modeling import VisionTransformer, CONFIGS
from plugin_loader import ViTPluginLoader


test_time = 100
warmup_time = 10

def setup_torch(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    print(config)

    model = VisionTransformer(config, args.img_size, zero_head=False, num_classes=1000)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    return config, model

def setup_trt(args, config):
    p_loader = ViTPluginLoader(args.plugin_path)
    p_loader.load_model_config(config, args)
    engine = p_loader.build_network(args.pretrained_dir)
    return engine, p_loader

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
    parser.add_argument('--plugin_path', type=str, default="../../../build/lib/libvit_plugin.so", help='path to plugin lib')
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--fp16', action='store_true', 
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args, unparsed = parser.parse_known_args()


    return args


def main(args):

    config, model = setup_torch(args)
    engine, p_loader = setup_trt(args, config)

    validate_with_random_data(p_loader, model, engine)

@torch.no_grad()
def run_trt_plugin(plugin_loader:ViTPluginLoader, images, engine):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    seq_len = plugin_loader.seq_len_
    embed_dim = plugin_loader.embed_dim_
    max_batch = plugin_loader.max_batch_
    img_size = plugin_loader.img_size_
    in_chans = plugin_loader.in_chans_
    is_fp16 = plugin_loader.is_fp16_

    dtype_trt = trt.float16 if is_fp16 else trt.float32
    dtype_np = np.float16 if is_fp16 else np.float32
    dtype_torch = torch.float16 if is_fp16 else torch.float32

    with engine.create_execution_context() as context:

        context.active_optimization_profile = 0

        stream = torch.cuda.Stream()

        context.set_binding_shape(0, (max_batch, in_chans, img_size, img_size))
        output_shape = tuple(context.get_binding_shape(1))
        print(output_shape)

        # Copy input h2d
        d_inputs = [images]
        d_output = torch.empty(output_shape, dtype=torch.float32).cuda()

        # warm up
        for i in range(warmup_time):
            context.execute_async_v2([d_inp.data_ptr() for d_inp in d_inputs] + [d_output.data_ptr()], stream.cuda_stream)

        #ignore the last fc layer
        torch.cuda.synchronize()
        op_end = time.time()
        for i in range(test_time):
            context.execute_async_v2([d_inp.data_ptr() for d_inp in d_inputs] + [d_output.data_ptr()], stream.cuda_stream)
        stream.synchronize()

        torch.cuda.synchronize()
        print("plugin time : ", (time.time() - op_end)/test_time*1000.0, "ms")

        return d_output.cpu().numpy()


@torch.no_grad()
def run_torch(model, images, mark):
    # warm up
    for i in range(warmup_time):
        output = model(images)

    torch.cuda.synchronize()
    torch_start = time.time()
    for i in range(test_time):
        torch_output = model.transformer(images)
    
    torch.cuda.synchronize()
    torch_end = time.time()
    embed = model.transformer.embeddings(images)
    np.save('embed_torch.npy',embed.cpu().numpy())
    torch_output = torch_output[0].cpu().numpy()
    np.save('torch_out.npy', torch_output)
    print(mark + " time : ", (torch_end - torch_start)/test_time*1000.0, "ms")

    return torch_output

@torch.no_grad()
def validate_with_random_data(plugin_loader:ViTPluginLoader, model, engine):
    model.eval()
    if plugin_loader.is_fp16_:
        model.half()

    dtype_torch = torch.float16 if plugin_loader.is_fp16_ else torch.float

    max_batch = plugin_loader.max_batch_
    img_size = plugin_loader.img_size_ 
    in_chans = plugin_loader.in_chans_
    image = np.random.rand(1, in_chans, img_size, img_size)
    images = np.repeat(image, max_batch, axis=0)
    images_tensor = torch.tensor(images, dtype=dtype_torch)
    images_tensor = images_tensor.cuda(non_blocking=True)

    plugin_output = run_trt_plugin(plugin_loader, images_tensor, engine)

    torch_output = run_torch(model, images_tensor, "torch")
    print(torch_output.shape)
    print(plugin_output.shape)

    diff = abs(torch_output - plugin_output.reshape(torch_output.shape))
    print("torch_output vs plugin_output , avg diff : ", diff.mean(), "max diff : ", diff.max())


if __name__ == '__main__':
    args = parse_option()

    seed = args.seed + int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    main(args)
