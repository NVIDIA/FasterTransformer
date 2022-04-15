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

import sys
import argparse
import re
import numpy as np
import torch

ACTIVATION_AMAX_NUM = 72
INT8O_KERNEL_NUM = 5
INT8O_GEMM_NUM = 7
TRT_FUSED_MHA_AMAX_NUM = 3
SCALE_RESERVE_NUM = 8

def extract_amaxlist(init_dict, depths, ths_path='../lib/libpyt_swintransformer.so', verbose=True):
    # print("Quantizing checkpoint ...")
    torch.classes.load_library(ths_path)
    weight_quantize = torch.ops.fastertransformer.swin_weight_quantize

    layer_num = len(depths)

    amaxTotalNum = ACTIVATION_AMAX_NUM + INT8O_KERNEL_NUM + INT8O_GEMM_NUM + 1 +  TRT_FUSED_MHA_AMAX_NUM + SCALE_RESERVE_NUM

    kernel_name_list = ["attn.qkv",
                        "attn.proj",
                        "mlp.fc1",
                        "mlp.fc2"]

    amax_name_list = ["attn.qkv._input_quantizer",
                      "attn.qkv._aftergemm_quantizer",
                      "attn.proj._input_quantizer",
                      "attn.proj._aftergemm_quantizer",
                      "attn.matmul_q_input_quantizer",
                      "attn.matmul_k_input_quantizer",
                      "attn.matmul_v_input_quantizer",
                      "attn.matmul_a_input_quantizer",
                      "attn.softmax_input_quantizer",
                      "mlp.fc1._input_quantizer",
                      "mlp.fc1._aftergemm_quantizer",
                      "mlp.fc2._input_quantizer",
                      "mlp.fc2._aftergemm_quantizer",
                      "add1_residual_input_quantizer",
                      "add2_residual_input_quantizer"
                      ]

    int8O_gemm_weight_amax_list = [0 for i in range(INT8O_GEMM_NUM)]
    int8O_gemm_weight_list = ["attn.qkv",
                              "attn.proj",
                              "mlp.fc1",
                              "mlp.fc2",
                              "attn.matmul_k_input_quantizer",
                              "attn.matmul_v_input_quantizer"]

    int8O_gemm_input_amax_list = [0 for i in range(INT8O_GEMM_NUM)]
    int8O_gemm_input_list = ["attn.qkv._input_quantizer",
                             "attn.proj._input_quantizer",
                             "mlp.fc1._input_quantizer",
                             "mlp.fc2._input_quantizer",
                             "attn.matmul_q_input_quantizer",
                             "attn.matmul_a_input_quantizer"]
    
    int8O_gemm_output_amax_list = [0 for i in range(INT8O_GEMM_NUM)]
    int8O_gemm_output_list = ["attn.qkv._aftergemm_quantizer",
                              "attn.proj._aftergemm_quantizer",
                              "mlp.fc1._aftergemm_quantizer",
                              "mlp.fc2._aftergemm_quantizer",
                              "attn.softmax_input_quantizer",
                              "attn.proj._input_quantizer"]

    downsample_input = "downsample.reduction._input_quantizer"
    downsample_weight = "downsample.reduction._weight_quantizer"
    downsample_out = "downsample.reduction._aftergemm_quantizer"

    factor = 1000000.0
    for i in range(layer_num):
        for depth in range(depths[i]):
            amaxList = np.zeros([amaxTotalNum]).astype(np.float32)
            amax_id = 0
            for amax_name in amax_name_list:
                quant_max = init_dict["layers.{}.blocks.{}.{}._amax".format(i, depth, amax_name)].item()
                amax = abs(quant_max)#round(abs(quant_max)*factor)/factor
                if amax_name in int8O_gemm_input_list:
                    int8O_gemm_input_amax_list[int8O_gemm_input_list.index(amax_name)] = amax
                if amax_name in int8O_gemm_output_list:
                    int8O_gemm_output_amax_list[int8O_gemm_output_list.index(amax_name)] = amax
                if amax_name in int8O_gemm_weight_list:
                    int8O_gemm_weight_amax_list[int8O_gemm_weight_list.index(amax_name)] = amax      
                amaxList[amax_id] = amax
                amax_id += 1
                amaxList[amax_id] = amax/127.0
                amax_id += 1
                amaxList[amax_id] = amax/127.0/127.0
                amax_id += 1
                amaxList[amax_id] = 127.0/amax
                amax_id += 1
                # if verbose:
                #     print(i, amax_name)
                #     print('quant_max:', quant_max)
                #     print('amax:', amax)
            if i != layer_num - 1:
                amax = init_dict["layers.{}.{}._amax".format(i, downsample_input)].item()
                amaxList[amax_id] = amax
                amax_id += 1
                amaxList[amax_id] = amax/127.0
                amax_id += 1
                amaxList[amax_id] = amax/127.0/127.0
                amax_id += 1
                amaxList[amax_id] = 127.0/amax
                amax_id += 1
                amax = init_dict["layers.{}.{}._amax".format(i, downsample_out)].item()
                amaxList[amax_id] = amax
                amax_id += 1
                amaxList[amax_id] = amax/127.0
                amax_id += 1
                amaxList[amax_id] = amax/127.0/127.0
                amax_id += 1
                amaxList[amax_id] = 127.0/amax
                amax_id += 1
            else:
                amax_id += 8
            if verbose:
                print("done process layer_{} block_{} activation amax".format(i, depth))

            #kernel amax starts from ACTIVATION_AMAX_NUM
            assert amax_id == 68
            amax_id = ACTIVATION_AMAX_NUM
            for kernel_id, kernel_name in enumerate(kernel_name_list):
                kernel = init_dict["layers.{}.blocks.{}.{}.weight".format(i, depth, kernel_name)].transpose(-1, -2).contiguous()
                quant_max2 = init_dict["layers.{}.blocks.{}.{}._weight_quantizer._amax".format(i, depth, kernel_name)]
                amax2 = abs(quant_max2)
                # if (amax2.dim() == 0):
                #     quant_max_processed = torch.full((kernel.size(1),), amax2.item(), dtype=amax2.dtype, device=amax2.device)
                # else:
                #     quant_max_processed = amax2.view(-1)
                kernel_processed = weight_quantize(kernel, amax2.cuda())
                init_dict["layers.{}.blocks.{}.{}.weight".format(i, depth, kernel_name)] = kernel_processed
                if kernel_name in int8O_gemm_weight_list:
                    int8O_gemm_weight_amax_list[int8O_gemm_weight_list.index(kernel_name)] = amax2.item()
                amaxList[amax_id] = amax2
                amax_id += 1
                # if verbose:
                #     print(i, kernel_name)
                #     print('kernel:', kernel)
                #     print('quant_max2:', quant_max2)
                #     print('quant_max_processed_:', quant_max_processed)
            if i != layer_num - 1:
                amaxList[amax_id] = init_dict["layers.{}.downsample.reduction._weight_quantizer._amax".format(i)].item()
            amax_id += 1

            assert amax_id == ACTIVATION_AMAX_NUM + INT8O_KERNEL_NUM
            #for int8O gemm deQuant
            for j in range(INT8O_GEMM_NUM - 1):
                amaxList[amax_id] = (int8O_gemm_input_amax_list[j]*int8O_gemm_weight_amax_list[j])/(127.0*int8O_gemm_output_amax_list[j])
                
                # print('layernum:', i, 'j:', j, ' gemm_int8IO_scale:',amaxList[amax_id])
                # print(int8O_gemm_input_amax_list[j], int8O_gemm_weight_amax_list[j], int8O_gemm_output_amax_list[j])
                amax_id += 1
            
            if i != layer_num - 1:
                patchMerge_i = init_dict["layers.{}.{}._amax".format(i, downsample_input)].item()
                patchMerge_w = init_dict["layers.{}.{}._amax".format(i, downsample_weight)].item()
                patchMerge_o = init_dict["layers.{}.{}._amax".format(i, downsample_out)].item()
                amaxList[amax_id] = (patchMerge_i * patchMerge_w) / (127 * patchMerge_o)
            amax_id += 1
            assert amax_id == ACTIVATION_AMAX_NUM + INT8O_KERNEL_NUM + INT8O_GEMM_NUM
            
            amax_id += 1
            #for trt fused MHA amax 
            #### QKV_addBias_amax
            # amaxList[amax_id] = np.maximum(np.maximum(amaxList[16],amaxList[20]), amaxList[24])
            # amax_id += 1
            # #### softmax amax
            # amaxList[amax_id] = amaxList[28]
            # amax_id += 1
            # #### bmm2 amax
            # amaxList[amax_id] = amaxList[8]
            # amax_id += 1
            qkvMax = np.maximum(np.maximum(amaxList[16],amaxList[20]), amaxList[24])
            amaxList[amax_id] = amaxList[16] * amaxList[20] / (127.0 * 127.0)
            amax_id += 1
            amaxList[amax_id] = 127.0 / amaxList[28]
            amax_id += 1
            amaxList[amax_id] = amaxList[24] * amaxList[28] / (127.0 * amaxList[8])
            amax_id += 1            

            init_dict["layers.{}.blocks.{}.amaxList".format(i, depth)] = torch.tensor(amaxList, dtype=torch.float32)
            if verbose:
                print("done process layer_{} block_{} kernel weight".format(i, depth))

        if i != layer_num - 1:
            kernel = init_dict["layers.{}.downsample.reduction.weight".format(i)]
            quant_max2 = init_dict["layers.{}.downsample.reduction._weight_quantizer._amax".format(i)]
            amax2 = abs(quant_max2)

            kernel = kernel.transpose(-1, -2).contiguous()
            kernel_processed = weight_quantize(kernel, amax2.cuda())
            
            init_dict["layers.{}.downsample.reduction.weight".format(i)] = kernel_processed


    # print("Quantizing checkpoint done.")
    return init_dict


if __name__ == '__main__':
    weights = torch.load('pytorch_model.bin')
    extract_amaxlist(weights, [2, 2, 6, 2])