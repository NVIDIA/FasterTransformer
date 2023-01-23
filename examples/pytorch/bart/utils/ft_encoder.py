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

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

class FTBartEncoderWeight(object):
    def __init__(
            self,
            config,
            tensor_para_size,
            pipeline_para_size,
            *,
            bart_with_bias=True,
            mbart=False,
            use_gated_activation=False,
            position_embedding_type=1,
            weight_data_type
    ):
        self.num_layer = config.encoder_layers
        self.config = config
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.bart_with_bias = bart_with_bias
        self.mbart = mbart
        self.use_gated_activation = use_gated_activation
        self.real_weights_num = 24  # assume all weights are allocated
        self.position_embedding_type = position_embedding_type
        self.weight_data_type = weight_data_type
        self.w = []
        self.use_mpi = dist.is_mpi_available()

        if self.use_mpi:
            try:
                dist.init_process_group(backend='mpi')
            except:
                print("[INFO] WARNING: Exception occurred in dist.init_process_group(backend = 'mpi'). Maybe the process group has been initialized somewhere else.")
        else:
            print("[INFO] MPI is not available in this PyTorch build.")
            assert tensor_para_size == 1, "[FATAL] MPI is required for tensor_para_size > 1."
            assert pipeline_para_size == 1, "[FATAL] MPI is required for pipeline_para_size > 1."

        self.rank = dist.get_rank() if self.use_mpi else 0
        self.device_count = torch.cuda.device_count()
        self.device = self.rank % self.device_count
        torch.cuda.set_device(self.device)

        world_size = dist.get_world_size() if self.use_mpi else 1
        assert world_size == tensor_para_size * \
            pipeline_para_size, "[ERROR] world_size != tensor_para_size * pipeline_para_size"
        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.pipeline_para_rank = self.rank // self.tensor_para_size

    def load_from_model(self, model):  
        '''Only applies to HuggingFace models. 
        Weight loading order: PyTorch tensor order should conform to src/fastertransformer/th_op/BartEncoderOp.h:FasterTransformerBartEncoder. For per-layer weights, the tensor is a stack of the weight across all layers.
        '''
        start_layer = self.pipeline_para_rank * self.num_layer // self.pipeline_para_size
        end_layer = (self.pipeline_para_rank + 1) * self.num_layer // self.pipeline_para_size

        np_weight_dtype = self.weight_data_type
        torch_weight_dtype = {np.float32: torch.float32, np.float16: torch.float16}[np_weight_dtype]

        encoder_weight_dict = {}
        for name, param in model.named_parameters():
            # HF BART/mBART model's weight names are prepended with "model.", remove for consistency
            name = name.replace("model.", "")
            
            if param.dim() == 2:
                param_t = param.transpose(1, 0) # PyTorch --> FT weight loading needs transpose
            elif param.dim() == 1:
                param_t = param
            else:
                assert False, f"The dimension of param {name} should be 1 or 2"
            if name.find("encoder.layers") != -1 or name.find("encoder.layernorm_embedding") != -1 or name.find("encoder.layer_norm") != -1:
                encoder_weight_dict[name] = param_t
            if name.find("encoder.embed_positions") != -1:
                encoder_weight_dict[name] = param # positional embedding table should NOT be transposed

        # [0]
        t = torch.stack([encoder_weight_dict["encoder.layers.{}.self_attn_layer_norm.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        # [1]
        t = torch.stack([encoder_weight_dict["encoder.layers.{}.self_attn.q_proj.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous())
        # [2]
        t = torch.stack([encoder_weight_dict["encoder.layers.{}.self_attn.k_proj.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous())
        # [3]
        t = torch.stack([encoder_weight_dict["encoder.layers.{}.self_attn.v_proj.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous())
        # [4]
        t = torch.stack([encoder_weight_dict["encoder.layers.{}.self_attn.out_proj.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous())
        # [5]
        t = torch.stack([encoder_weight_dict["encoder.layers.{}.final_layer_norm.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        # [6]
        t = torch.stack([encoder_weight_dict["encoder.layers.{}.fc1.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous())
        # [7] add empty weight for gated activation for now (BART/mBART model by default don't use gated activation)
        self.w.append(torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda())
        # [8]
        t = torch.stack([encoder_weight_dict["encoder.layers.{}.fc2.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous())
        # [9] (1) positional embedding table should NOT be transposed, [max position embeddings, hidden size] (2) need to apply offset of 2 for absolute position embeddings in BART/mBART
        t = encoder_weight_dict["encoder.embed_positions.weight"][2:, :].contiguous().cuda()
        self.w.append(t)
        # [10] input embedding table should NOT be transposed, [vocab, hidden size]. Directly obtained from raw weight is untransposed
        t = model.get_input_embeddings().weight.contiguous().cuda() 
        # input word embedding may be scaled (mBART), instead of customize this in FT, it's better to modify the embedding loading part in PyT
        embedding_scale = np.sqrt(model.config.d_model) if model.config.scale_embedding else 1.0
        t = t * embedding_scale
        self.w.append(t)
        # [11] LayerNorm after embedding & before transformer block, special in BART/mBART
        t = encoder_weight_dict["encoder.layernorm_embedding.weight"].contiguous().cuda()
        self.w.append(t)
        # [12] LayerNorm after transformer block, special in mBART
        if self.mbart:
            t = encoder_weight_dict["encoder.layer_norm.weight"].contiguous().cuda()
        else:
            t = torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda()
        self.w.append(t)

        if self.bart_with_bias:
            # [13]
            t = torch.stack([encoder_weight_dict["encoder.layers.{}.self_attn_layer_norm.bias".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            # [14]
            t = torch.stack([encoder_weight_dict["encoder.layers.{}.self_attn.q_proj.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            self.w.append(t)
            # [15]
            t = torch.stack([encoder_weight_dict["encoder.layers.{}.self_attn.k_proj.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            self.w.append(t)
            # [16]
            t = torch.stack([encoder_weight_dict["encoder.layers.{}.self_attn.v_proj.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            self.w.append(t)
            # [17]
            t = torch.stack([encoder_weight_dict["encoder.layers.{}.self_attn.out_proj.bias".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            # [18]
            t = torch.stack([encoder_weight_dict["encoder.layers.{}.final_layer_norm.bias".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            # [19]
            t = torch.stack([encoder_weight_dict["encoder.layers.{}.fc1.bias".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            self.w.append(t)
            # [20] add empty bias for gated activation for now (BART/mBART model by default don't use gated activation)
            self.w.append(torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda())
            # [21]
            t = torch.stack([encoder_weight_dict["encoder.layers.{}.fc2.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            # [22]
            t = encoder_weight_dict["encoder.layernorm_embedding.bias"].contiguous().cuda()
            self.w.append(t)
            # [23] LayerNorm after transformer block, special in mBART
            if self.mbart:
                t = encoder_weight_dict["encoder.layer_norm.bias"].contiguous().cuda()
            else:
                t = torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda()
            self.w.append(t)
        else:
            # TODO: pass None Type to Torch Op
            for i in range(11):
                self.w.append(torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda())

    def to_cuda(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].cuda()

    def to_float(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].float()

    def to_half(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].half()

    def to_single(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].float()

    def to_bfloat16(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].bfloat16()

class FTBartEncoder(nn.Module):
    def __init__(self, encoder_weight_list, lib_path, head_num, head_size, inter_size, d_model, is_remove_padding,
                 num_layer, num_bucket=32, max_distance=128, sparse=False, q_scaling=1.0, tensor_para_size=1, pipeline_para_size=1,
                 bart_with_bias=True, mbart=False, position_embedding_type=1, activation_type="gelu", layernorm_type="post_layernorm"):
        super().__init__()

        self.use_mpi = dist.is_mpi_available()

        if self.use_mpi:
            try:
                dist.init_process_group(backend='mpi')
            except:
                print("[INFO] WARNING: Exception occurred in dist.init_process_group(backend = 'mpi'). Maybe the process group has been initialized somewhere else.")
        else:
            print("[INFO] MPI is not available in this PyTorch build.")
            assert tensor_para_size == 1, "[FATAL] MPI is required for tensor_para_size > 1."
            assert pipeline_para_size == 1, "[FATAL] MPI is required for pipeline_para_size > 1."

        torch.classes.load_library(lib_path)
        try:
            self.encoder = torch.classes.FasterTransformer.BartEncoder(*encoder_weight_list, head_num, head_size, inter_size, d_model,
                                                                     is_remove_padding, num_layer, num_bucket, max_distance, sparse, q_scaling, tensor_para_size, pipeline_para_size,
                                                                     bart_with_bias, mbart, position_embedding_type, activation_type, layernorm_type)
        except:
            self.encoder = torch.classes.FasterTransformerBartEncoder(*encoder_weight_list, head_num, head_size, inter_size, d_model,
                                                                    is_remove_padding, num_layer, num_bucket, max_distance, sparse, q_scaling, tensor_para_size, pipeline_para_size,
                                                                    bart_with_bias, mbart, position_embedding_type, activation_type, layernorm_type)

    def forward(self, input, seq_len, inputs_embeds=None):
        output = self.encoder.forward(input, seq_len, inputs_embeds)
        return output