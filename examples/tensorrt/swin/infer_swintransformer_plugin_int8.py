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

import ctypes
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import sys
#sys.path.insert(0, "../third_party/Swin-Transformer")
sys.path.insert(0, "../../pytorch/swin/Swin-Transformer-Quantization")
sys.path.insert(0, "../../pytorch/swin")

from SwinTransformer.config import get_config
from models import build_model
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
    parser.add_argument('--engine', type=str, help='path to TRT engine')
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
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--int8-mode', type=int, help='int8 mode', choices=[1, 2])
    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    quant_utils.add_arguments(parser)
    args, unparsed = parser.parse_known_args()
    if args.quant_mode is not None:
        args = quant_utils.set_args(args)
    quant_utils.set_default_quantizers(args)

    config = get_config(args)

    return args, config


def main(config, args):
    model = build_model(config)
    model.cuda()

    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint, strict=False)
    quant_utils.configure_model(model, args, calib=False)
    validate_with_random_data(config, args, model)

@torch.no_grad()
def run_swintransformer_plugin(args, config, model, image):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    # Import necessary plugins for BERT TensorRT
    ctypes.CDLL("../../../build/lib/libswinTransformer_plugin.so", mode=ctypes.RTLD_GLOBAL)

    depths = config.MODEL.SWIN.DEPTHS
    layer_num = len(depths)
    max_batch = config.DATA.BATCH_SIZE
    img_size = config.DATA.IMG_SIZE
    in_chans = config.MODEL.SWIN.IN_CHANS
    embed_dim = config.MODEL.SWIN.EMBED_DIM

    output_size = model.head.bias.shape[0]
    print("output_size ", output_size)

    with open(args.engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime, \
            runtime.deserialize_cuda_engine(f.read()) as engine, \
            engine.create_execution_context() as context:

        context.active_optimization_profile = 0

        input_nbytes0 = max_batch * in_chans * img_size * img_size * trt.float16.itemsize
        stream = cuda.Stream()

        d_inputs = [cuda.mem_alloc(input_nbytes0)]
        output_shape = (max_batch * output_size)
        h_output = cuda.pagelocked_empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(h_output.nbytes)

        #import pdb;pdb.set_trace()
        context.set_binding_shape(0, (max_batch, in_chans, img_size, img_size))

        image = image.astype(np.float16)

        # Copy input h2d
        h_input_embeds = cuda.register_host_memory(np.ascontiguousarray(image).ravel())
        cuda.memcpy_htod_async(d_inputs[0], h_input_embeds, stream)

        # warm up
        for i in range(warmup_time):
            context.execute_async_v2(bindings=[int(d_inp) for d_inp in d_inputs] + [int(d_output)], stream_handle=stream.handle)
        
        #ignore the last fc layer
        op_end = time.time()
        for i in range(test_time):
            context.execute_async_v2(bindings=[int(d_inp) for d_inp in d_inputs] + [int(d_output)], stream_handle=stream.handle)
        stream.synchronize()

        print("plugin time : ", (time.time() - op_end)/test_time*1000.0, "ms")

        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        return h_output

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

    ## run pytorch plugin
    plugin_output = run_swintransformer_plugin(args, config, model, images)

    # warm up
    model.half()
    # traced_module = torch.jit.trace(model, images_half)
    # torch_traced_output = run_torch(traced_module, images_half, "torch trace")
    torch_output = run_torch(model, images_half, "torch")

    diff = abs(torch_output - plugin_output.reshape(max_batch, -1))
    print("torch_output vs plugin_output , avg diff : ", diff.mean((1)), "max diff : ", diff.max((1)))

if __name__ == '__main__':
    args, config = parse_option()

    seed = config.SEED + int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    main(config, args)
