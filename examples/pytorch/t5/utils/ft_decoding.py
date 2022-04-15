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
import torch.distributed as dist
import numpy as np

class FTT5DecodingWeight(object):
    def __init__(self, config, tensor_para_size, pipeline_para_size, t5_with_bias = False, position_embedding_type = 0):
        self.config = config
        self.num_layer = config.num_layers
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.t5_with_bias = t5_with_bias
        self.position_embedding_type = position_embedding_type
        self.real_weights_num = 27 if t5_with_bias else 14
        self.w = []
        self.use_mpi = dist.is_mpi_available()

        if self.use_mpi:
            try:
                dist.init_process_group(backend='mpi')
            except:
                print("[INFO] WARNING: Exception occurred in dist.init_process_group(backend='mpi'). Maybe the process group has been initialized somewhere else.")
        else:
            print("[INFO] MPI is not available in this PyTorch build.")
            assert tensor_para_size == 1, "[FATAL] MPI is required for tensor_para_size > 1."
            assert pipeline_para_size == 1, "[FATAL] MPI is required for pipeline_para_size > 1."

        self.rank = dist.get_rank() if self.use_mpi else 0
        self.device_count = torch.cuda.device_count()
        self.device = self.rank % self.device_count
        torch.cuda.set_device(self.device)

        world_size = dist.get_world_size() if self.use_mpi else 1
        assert world_size == tensor_para_size * pipeline_para_size, "[ERROR] world_size != tensor_para_size * pipeline_para_size"
        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.pipeline_para_rank = self.rank // self.tensor_para_size

    def load_from_model(self, model):
        start_layer = self.pipeline_para_rank * self.num_layer //  self.pipeline_para_size
        end_layer = (self.pipeline_para_rank + 1) * self.num_layer //  self.pipeline_para_size
        
        weight_dict = {}
        qkv_tmp = []
        for name, param in model.named_parameters():
            if param.dim() == 2:
                param_t = param.transpose(1, 0)
            elif param.dim() == 1:
                param_t = param
            else:
                assert False, f"The dimension of param {name} should be 2"
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

        #TODO: pass None Type to Torch Op
        for i in range(13):
            self.w.append(torch.empty((1,1), dtype=torch.float32).contiguous().cuda())
    
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
        t = torch.from_numpy(np.fromfile(f"{ckpt_path}/shared.weight_T.bin", dtype=np.single).reshape([self.config.d_model, self.config.vocab_size])).contiguous().cuda()
        self.w.append(t)
        t = None
        if (self.position_embedding_type == 0):
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight.{self.tensor_para_rank}.bin", dtype=np.single)).contiguous().cuda()
        else:
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/shared.ape.bin", dtype=np.single)).contiguous().cuda()
        self.w.append(t)
        
        # add 13 additional bias if it is t5 megatron structure
        if self.t5_with_bias:
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.0.layer_norm.bias.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.0.SelfAttention.qkv.bias.{self.tensor_para_rank}.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.0.SelfAttention.o.bias.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.layer_norm.bias.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.EncDecAttention.q.bias.{self.tensor_para_rank}.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.EncDecAttention.k.bias.{self.tensor_para_rank}.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.EncDecAttention.v.bias.{self.tensor_para_rank}.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.EncDecAttention.o.bias.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.2.layer_norm.bias.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.2.DenseReluDense.wi.bias.{self.tensor_para_rank}.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.2.DenseReluDense.wo.bias.bin", dtype=np.single)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.final_layer_norm.bias.bin", dtype=np.single)).contiguous().cuda()
            self.w.append(t)
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/shared.bias.bin", dtype=np.single)).contiguous().cuda()
            self.w.append(t)
        else:
            #TODO: pass None Type to Torch Op
            for i in range(13):
                self.w.append(torch.empty((1,1), dtype=torch.float32).contiguous().cuda())
            
    def to_cuda(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].cuda()

    def to_half(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].half()

class FTT5Decoding(nn.Module):
    def __init__(self, decoding_weight_list, lib_path, head_num, head_size, inter_size,
                 mem_d_model, d_model, num_layer, start_id, end_id, vocab_size, q_scaling = 1.0, num_bucket=32,
                 max_distance=128, tensor_para_size=1, pipeline_para_size=1, t5_with_bias=False, position_embedding_type=0):
        super().__init__()

        self.use_mpi = dist.is_mpi_available()

        if self.use_mpi:
            try:
                dist.init_process_group(backend='mpi')
            except:
                print("[INFO] WARNING: Exception occurred in dist.init_process_group(backend='mpi'). Maybe the process group has been initialized somewhere else.")
        else:
            print("[INFO] MPI is not available in this PyTorch build.")
            assert tensor_para_size == 1, "[FATAL] MPI is required for tensor_para_size > 1."
            assert pipeline_para_size == 1, "[FATAL] MPI is required for pipeline_para_size > 1."

        torch.classes.load_library(lib_path)
        try:
            self.decoding = torch.classes.FasterTransformer.T5Decoding(head_num, head_size, inter_size, mem_d_model, d_model, num_layer,
                                                                       vocab_size, num_bucket, max_distance, q_scaling, start_id, end_id,
                                                                       tensor_para_size, pipeline_para_size, t5_with_bias, position_embedding_type,
                                                                       *decoding_weight_list)
        except:
            self.decoding = torch.classes.FasterTransformerT5Decoding(head_num, head_size, inter_size, mem_d_model, d_model, num_layer,
                                                                      vocab_size, num_bucket, max_distance, q_scaling, start_id, end_id,
                                                                      tensor_para_size, pipeline_para_size, t5_with_bias, position_embedding_type,
                                                                      *decoding_weight_list)

    def forward(self, beam_width, max_seq_len, top_k, top_p,
                beam_search_diversity_rate, temperature,
                len_penalty, repetition_penalty, random_seed,
                mem_hidden_states, mem_seq_len,
                is_return_output_log_probs, is_return_cum_log_probs):
        # TODO (bhsueh) Not found an method to put a None Type into op forward function
        # So, the top_k and top_p must be some values now.
        results = self.decoding.forward(beam_width, max_seq_len,
                                        top_k, top_p, beam_search_diversity_rate,
                                        temperature, len_penalty, repetition_penalty,
                                        random_seed, mem_hidden_states, mem_seq_len,
                                        is_return_output_log_probs, is_return_cum_log_probs)
        results
        return results
    
class FTT5(nn.Module):
    def __init__(self, encoder, decoding):
        super().__init__()
        self.encoder = encoder
        self.decoding = decoding
        
    def forward(self, input_token, beam_size, max_seq_len,
                top_k, top_p, beam_search_diversity_rate,
                temperature=1.0, len_penalty=1.0, repetition_penalty=1.0, random_seed=0,
                is_return_output_log_probs=False, is_return_cum_log_probs=False):
        input_ids = input_token.input_ids.to("cuda").type(torch.int32)
        mem_seq_len = 0
        if hasattr(input_token, "attention_mask") :
            mem_seq_len = torch.sum(input_token.attention_mask, dim=1).type(torch.int32).to("cuda")
        else :
            mem_seq_len = input_token.seq_len.type(torch.int32).to("cuda")

        ft_encoder_outputs = self.encoder.forward(input_ids, mem_seq_len)
        results = self.decoding.forward(beam_size,
                                        max_seq_len,
                                        top_k,
                                        top_p,
                                        beam_search_diversity_rate,
                                        temperature,
                                        len_penalty,
                                        repetition_penalty,
                                        random_seed,
                                        is_return_output_log_probs,
                                        is_return_cum_log_probs,
                                        ft_encoder_outputs,
                                        mem_seq_len)
        ft_decoding_outputs = results.pop(0).reshape([-1, beam_size, max_seq_len])
        ft_decoding_seq_lens = results.pop(0).reshape([-1, beam_size])
        if is_return_output_log_probs:
            ft_output_log_probs = results.pop(0)
        if is_return_cum_log_probs:
            ft_cum_log_probs = results.pop(0)

        return ft_decoding_outputs.cpu().numpy(), ft_decoding_seq_lens.cpu().numpy()