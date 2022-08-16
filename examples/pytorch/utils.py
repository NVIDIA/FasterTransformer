# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
import torch
import typing


def print_memory_usage(info=""):
    t = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    r = torch.cuda.memory_reserved(0) / 1024 ** 2
    a = torch.cuda.memory_allocated(0) / 1024 ** 2
    f = r - a  # free inside reserved
    print(f"[INFO][{info}] total_memory: {t}, reversed: {r}, allocated: {a}")


def torch2np(tensor: torch.Tensor, np_data_type: typing.Optional[np.dtype] = None):
    tensor = tensor.cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)

    data = tensor.numpy()
    if np_data_type is not None:
        data = data.astype(np_data_type)

    return data


def safe_transpose(tensor):
    if tensor.dim() == 1:
        return tensor
    if tensor.dim() == 2:
        return tensor.T
    raise ValueError("Tensor has more than 2 dimensions, unable to safely transpose.")


WEIGHT2DTYPE = {
    "fp32": np.float32,
    "fp16": np.float16,
}


def cpu_map_location(storage, loc):
    return storage.cpu()


def gpu_map_location(storage, loc):
    if loc.startswith("cuda"):
        training_gpu_idx = int(loc.split(":")[1])
        inference_gpu_idx = training_gpu_idx % torch.cuda.device_count()
        return storage.cuda(inference_gpu_idx)
    elif loc.startswith("cpu"):
        return storage.cpu()
    else:
        raise NotImplementedError(f"Not handled {loc}")