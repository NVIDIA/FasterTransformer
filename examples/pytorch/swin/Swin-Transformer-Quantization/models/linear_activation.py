# --------------------------------------------------------
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

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import sys
import math
from pytorch_quantization.nn import QuantLinear, TensorQuantizer

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

#used only for triton inference
def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

# used specifically for training since torch.nn.functional.gelu breaks ONNX export
def bias_gelu_training(bias, y):
    x = bias + y
    return torch.nn.functional.gelu(x) # Breaks ONNX export

def bias_tanh(bias, y):
    x = bias + y
    return torch.tanh(x)

def swish(x):
    return x * torch.sigmoid(x)

def bias_noact(bias, y):
    return bias + y

#torch.nn.functional.gelu(x) # Breaks ONNX export
ACT2FN = {"gelu": gelu, "bias_gelu": bias_gelu, "bias_tanh": bias_tanh, "relu": torch.nn.functional.relu, "swish": swish,
          "bias_noact": bias_noact}

class QuantizedConv2d(nn.Module):
    r"""Conv2d with INT8 quantization
    """
    __constants__ = ['bias']

    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True):
        super(QuantizedConv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = Parameter(torch.Tensor(out_channel, in_channel, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channel, 1, 1))
        else:
            self.register_parameter('bias', None)
        
        self._input_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
        self._weight_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_weight)
        self._aftergemm_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_normal_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        input = self._input_quantizer(input)
        weight = self._weight_quantizer(self.weight)
        output = self._aftergemm_quantizer(F.conv2d(input, weight, bias=None, stride=self.stride))
        if self.bias is None:
            return output
        else:
            return self.bias + output

class LinearActivation(nn.Module):
    r"""Fused Linear and Activation Module.
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, act='gelu', bias=True, do_quant=True):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_fn = nn.Identity()
        self.biased_act_fn = None
        
        if isinstance(act, str) or (sys.version_info[0] == 2 and isinstance(act, unicode)):
            if bias and not 'bias' in act:
                act = 'bias_' + act
                self.biased_act_fn = ACT2FN[act]
            else:
                self.act_fn = ACT2FN[act]
        else:
            self.act_fn = act
        
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.do_quant = do_quant
        if do_quant:
            self._input_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
            self._weight_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_weight)
            self._aftergemm_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_normal_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        if self.do_quant:
            input = self._input_quantizer(input)
            weight = self._weight_quantizer(self.weight)
        else:
            weight = self.weight
        
        if not self.bias is None:
            if self.do_quant:
                return self.biased_act_fn(self.bias, self._aftergemm_quantizer(F.linear(input, weight, None)))
            else:
                return self.biased_act_fn(self.bias, F.linear(input, weight, None))
        else:
            if self.do_quant:
                return self.act_fn(self._aftergemm_quantizer(F.linear(input, weight, None)))
            else:
                return self.act_fn(F.linear(input, weight, None))
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )