# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

class FTT5DecodingWeight(object):
    def __init__(self, config, tensor_para_size, pipeline_para_size):
        self.config = config
        self.num_layer = config.num_layers
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.w = []
        
        try:
            dist.init_process_group(backend='mpi')
        except:
            print("[INFO] WARNING: Have initalize the process group")
        
        self.rank = dist.get_rank()
        self.device_count = torch.cuda.device_count()
        self.device = self.rank % self.device_count
        torch.cuda.set_device(self.device)
        
        world_size = dist.get_world_size()
        assert world_size == tensor_para_size * pipeline_para_size, "[ERROR] world_size != tensor_para_size * pipeline_para_size"
        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.pipeline_para_rank = self.rank // self.tensor_para_size
        
    def load_from_model(self, model):
        start_layer = self.pipeline_para_rank * self.num_layer //  self.pipeline_para_size
        end_layer = (self.pipeline_para_rank + 1) * self.num_layer //  self.pipeline_para_size
        
        weight_dict = {}
        qkv_tmp = []
        for name, param in model.named_parameters():
            param_t = param.T
            if name.find("decoder.block") != -1:
                if name.find(".SelfAttention.q.weight") != -1 or name.find(".SelfAttention.k.weight") != -1 or name.find(".SelfAttention.v.weight") != -1:
                    qkv_tmp.append(param_t)
                    if len(qkv_tmp) == 3:
                        qkv = torch.cat(qkv_tmp, dim=-1)
                        weight_dict[name.replace("v.weight", "qkv.weight")] = qkv
                        qkv_tmp = []
                else:
                    weight_dict[name] = param_t
            elif name.find("decoder") != -1:
                weight_dict[name] = param_t

        # load by torch model directly
        t = torch.stack([weight_dict["decoder.block.{}.layer.0.layer_norm.weight".format(i)] for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.0.SelfAttention.qkv.weight".format(i)] for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.reshape([t.shape[0], t.shape[1], 3, t.shape[2] // 3])
        t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.0.SelfAttention.o.weight".format(i)] for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.1.layer_norm.weight".format(i)] for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.1.EncDecAttention.q.weight".format(i)] for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.1.EncDecAttention.k.weight".format(i)] for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.1.EncDecAttention.v.weight".format(i)] for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.1.EncDecAttention.o.weight".format(i)] for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.2.layer_norm.weight".format(i)] for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.2.DenseReluDense.wi.weight".format(i)] for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.2.DenseReluDense.wo.weight".format(i)] for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        t = weight_dict["decoder.final_layer_norm.weight"].contiguous().cuda()
        self.w.append(t)
        t = model.get_output_embeddings().weight.contiguous().cuda()
        self.w.append(t)
        t = weight_dict["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"].contiguous().cuda()
        t = t.split(t.shape[0] // self.tensor_para_size, dim=0)[self.tensor_para_rank].contiguous()
        self.w.append(t)
    
    def load_from_bin(self, ckpt_path):
        start_layer = self.pipeline_para_rank * self.num_layer //  self.pipeline_para_size
        end_layer = (self.pipeline_para_rank + 1) * self.num_layer //  self.pipeline_para_size
        
        # load by binary files
        t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.0.layer_norm.weight.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.0.SelfAttention.qkv.weight.{self.tensor_para_rank}.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.0.SelfAttention.o.weight.{self.tensor_para_rank}.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.layer_norm.weight.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.EncDecAttention.q.weight.{self.tensor_para_rank}.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.EncDecAttention.k.weight.{self.tensor_para_rank}.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.EncDecAttention.v.weight.{self.tensor_para_rank}.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.EncDecAttention.o.weight.{self.tensor_para_rank}.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.2.layer_norm.weight.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.2.DenseReluDense.wi.weight.{self.tensor_para_rank}.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.2.DenseReluDense.wo.weight.{self.tensor_para_rank}.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.final_layer_norm.weight.bin", dtype=np.single)).contiguous().cuda()
        self.w.append(t)
        t = torch.from_numpy(np.fromfile(f"{ckpt_path}/shared.weight.bin", dtype=np.single).reshape([self.config.d_model, self.config.vocab_size])).T.contiguous().cuda()
        self.w.append(t)
        t = torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight.{self.tensor_para_rank}.bin", dtype=np.single)).contiguous().cuda()
        self.w.append(t)
        
    def to_cuda(self):
        for i in range(len(self.w)):
            self.w[i] = self.w[i].cuda()

    def to_half(self):
        for i in range(len(self.w)):
            self.w[i] = self.w[i].half()

class FTT5Decoding(nn.Module):
    def __init__(self, decoding_weight_list, lib_path, head_num, head_size, inter_size,
                 mem_d_model, d_model, num_layer, start_id, end_id, vocab_size, num_bucket=32,
                 max_distance=128, beam_search_diversity_rate=0.0, top_k=1, top_p=0.0,
                 temperature=1.0, len_penalty=1.0, repetition_penalty=1.0,
                 tensor_para_size=1, pipeline_para_size=1):
        super().__init__()
        try:
            dist.init_process_group(backend='mpi')
        except:
            print("[INFO] WARNING: Have initalize the process group")

        torch.classes.load_library(lib_path)
        try:
            self.decoding = torch.classes.FasterTransformer.T5Decoding(head_num, head_size, inter_size, mem_d_model, d_model, num_layer,
                                                                       vocab_size, num_bucket, max_distance, start_id, end_id, 
                                                                       beam_search_diversity_rate, top_k, top_p, temperature,
                                                                       len_penalty, repetition_penalty,
                                                                       tensor_para_size, pipeline_para_size, 
                                                                       *decoding_weight_list)
        except:
            self.decoding = torch.classes.FasterTransformerT5Decoding(head_num, head_size, inter_size, mem_d_model, d_model, num_layer,
                                                                      vocab_size, num_bucket, max_distance, start_id, end_id, 
                                                                      beam_search_diversity_rate, top_k, top_p, temperature,
                                                                      len_penalty, repetition_penalty,
                                                                      tensor_para_size, pipeline_para_size, 
                                                                      *decoding_weight_list)

    def forward(self, beam_width, max_seq_len, mem_hidden_states, mem_seq_len):
        outputs, _, sequence_length = self.decoding.forward(beam_width, max_seq_len, mem_hidden_states, mem_seq_len)
        outputs = outputs.reshape([-1, beam_width, max_seq_len])
        return outputs, sequence_length.reshape([-1, beam_width])
    
class FTT5(nn.Module):
    def __init__(self, encoder, decoding):
        super().__init__()
        self.encoder = encoder
        self.decoding = decoding
        
    def forward(self, input_token, beam_size, max_seq_len):
        input_ids = input_token.input_ids.to("cuda").type(torch.int32)
        mem_seq_len = torch.sum(input_token.attention_mask, dim=1).type(torch.int32).to("cuda")
        
        ft_encoder_outputs = self.encoder.forward(input_ids, mem_seq_len)
        ft_decoding_outputs, ft_decoding_seq_lens = self.decoding.forward(beam_size, max_seq_len, ft_encoder_outputs, mem_seq_len)
        return ft_decoding_outputs.cpu().numpy(), ft_decoding_seq_lens.cpu().numpy()