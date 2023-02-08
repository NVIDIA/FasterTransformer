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

from tracemalloc import start
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import os


class FTT5EncoderWeight(object):
    def __init__(
            self,
            config,
            tensor_para_size,
            pipeline_para_size,
            *,
            t5_with_bias=False,
            use_gated_activation=False,
            t5_with_moe=False,
            position_embedding_type=0,
            weight_data_type
    ):
        self.num_layer = config.num_layers
        self.config = config
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.t5_with_bias = t5_with_bias
        self.use_gated_activation = use_gated_activation
        self.real_weights_num = 23 # assume all weights are allocated
        self.t5_with_moe = t5_with_moe
        self.position_embedding_type = position_embedding_type
        self.weight_data_type = weight_data_type
        self.adapter_inter_size = config.adapter_inter_size if hasattr(config, "adapter_inter_size") else 0
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

    def load_from_model(self, model):  # assume this only applies to huggingface models
        start_layer = self.pipeline_para_rank * self.num_layer // self.pipeline_para_size
        end_layer = (self.pipeline_para_rank + 1) * self.num_layer // self.pipeline_para_size

        np_weight_dtype = self.weight_data_type
        torch_weight_dtype = {np.float32: torch.float32, np.float16: torch.float16}[np_weight_dtype]

        encoder_weight_dict = {}
        for name, param in model.named_parameters():
            if param.dim() == 2:
                param_t = param.transpose(1, 0)
            elif param.dim() == 1:
                param_t = param
            else:
                assert False, f"The dimension of param {name} should be 2"
            if name.find("encoder.block") != -1 or name.find("encoder.final_layer_norm.weight") != -1:
                encoder_weight_dict[name] = param_t

        t = torch.stack([encoder_weight_dict["encoder.block.{}.layer.0.layer_norm.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.stack([encoder_weight_dict["encoder.block.{}.layer.0.SelfAttention.q.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous())
        t = torch.stack([encoder_weight_dict["encoder.block.{}.layer.0.SelfAttention.k.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous())
        t = torch.stack([encoder_weight_dict["encoder.block.{}.layer.0.SelfAttention.v.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous())
        t = torch.stack([encoder_weight_dict["encoder.block.{}.layer.0.SelfAttention.o.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous())
        t = torch.stack([encoder_weight_dict["encoder.block.{}.layer.1.layer_norm.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        if self.use_gated_activation:
            t = torch.stack([encoder_weight_dict["encoder.block.{}.layer.1.DenseReluDense.wi_0.weight".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous())
            t = torch.stack([encoder_weight_dict["encoder.block.{}.layer.1.DenseReluDense.wi_1.weight".format(i)]
                             for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous())
        else:
            t = torch.stack([encoder_weight_dict["encoder.block.{}.layer.1.DenseReluDense.wi.weight".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous())
            # add empty wi2
            self.w.append(torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda())
        t = torch.stack([encoder_weight_dict["encoder.block.{}.layer.1.DenseReluDense.wo.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous())
        t = encoder_weight_dict["encoder.final_layer_norm.weight"].contiguous().cuda()
        self.w.append(t)
        t = encoder_weight_dict["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"].contiguous().cuda()
        self.w.append(t.split(t.shape[0] // self.tensor_para_size, dim=0)[self.tensor_para_rank].contiguous())
        t = model.get_input_embeddings().weight.contiguous().cuda()
        self.w.append(t)

        #TODO: pass None Type to Torch Op
        for i in range(19):
            self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
        
    def load_from_bin(self, ckpt_path, model_type): # assume this only applies to megatron models and megatron-deepspedd models
        start_layer = self.pipeline_para_rank * self.num_layer //  self.pipeline_para_size
        end_layer = (self.pipeline_para_rank + 1) * self.num_layer //  self.pipeline_para_size

        np_weight_dtype = self.weight_data_type
        torch_weight_dtype = {np.float32: torch.float32, np.float16: torch.float16}[np_weight_dtype]

        # load by binary files
        if model_type == "Megatron":
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.0.layer_norm.weight.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.0.SelfAttention.q.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.0.SelfAttention.k.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.0.SelfAttention.v.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.0.SelfAttention.o.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.1.layer_norm.weight.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.1.DenseReluDense.wi.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            if self.use_gated_activation:
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.1.DenseReluDense.wi2.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
            else:
                self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.1.DenseReluDense.wo.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.final_layer_norm.weight.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)
            t = None
            if (self.position_embedding_type == 0):   
                t = torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)).contiguous().cuda()
            else:
                t = torch.from_numpy(np.fromfile(f"{ckpt_path}/shared.ape.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/shared.weight_T.bin", dtype=np_weight_dtype).reshape([self.config.d_model, self.config.vocab_size])).contiguous().cuda()
            self.w.append(t)
            
            # add 10 additional bias if it is t5 megatron structure
            if self.t5_with_bias:
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.0.layer_norm.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.0.SelfAttention.q.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.0.SelfAttention.k.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.0.SelfAttention.v.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.0.SelfAttention.o.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.1.layer_norm.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.1.DenseReluDense.wi.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                if self.use_gated_activation:
                    t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.1.DenseReluDense.wi2.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                    self.w.append(t)
                else:
                    self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.1.DenseReluDense.wo.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.final_layer_norm.bias.bin", dtype=np_weight_dtype)).contiguous().cuda()
                self.w.append(t)

            else:
                #TODO: pass None Type to Torch Op
                for i in range(10):
                    self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
            # add empty moe gate weight
            self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())

            if self.adapter_inter_size > 0:
                ckpt_path_block = f"{ckpt_path}/encoder.block"
                for adapter in ["after_attention_adapter", "after_ffn_adapter"]:
                    for in_out in ["wi", "wo"]:
                        t = torch.stack([torch.from_numpy(np.fromfile(
                            f"{ckpt_path_block}.{i}.{adapter}.DenseSiluDense.{in_out}.weight.{self.tensor_para_rank}.bin",
                            dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                        self.w.append(t)
                    for weight_bias in ["weight", "bias"]:
                        t = torch.stack([torch.from_numpy(np.fromfile(
                            f"{ckpt_path_block}.{i}.{adapter}.layer_norm.{weight_bias}.bin",
                            dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                        self.w.append(t)
            else:
                for i in range(8):
                    self.w.append(torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda())

        else: # Megatron-DeepSpeed, no tensor parallelism currently
            #TODO: add tensor parallelism in the conversion script
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.input_layernorm.weight.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.attention.query.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.attention.key.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.attention.value.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.attention.dense.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.post_attention_layernorm.weight.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)

            # =========== process normal and moe dense layer =================
            t_list = []
            for i in range(start_layer, end_layer):
                if (os.path.isfile(f"{ckpt_path}/encoder.layers.{i}.mlp.dense_h_to_4h.weight.{self.tensor_para_rank}.bin")):
                    t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.mlp.dense_h_to_4h.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
                else:
                    t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.mlp.deepspeed_moe.experts.deepspeed_experts.dense_h_to_4h.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
            self.w.append(torch.cat(t_list, 0).contiguous().cuda())
            # ================================================================

            # We don't have use_gated_activation in Megatron-DeepSpeed currently, so here weight placeholder is always empty
            # If we have it in the future, the binary file name should be modified according to the actual name.
            if self.use_gated_activation:
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.1.DenseReluDense.wi2.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
            else:
                self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())

            # =========== process normal and moe dense layer =================
            t_list = []
            for i in range(start_layer, end_layer):
                if (os.path.isfile(f"{ckpt_path}/encoder.layers.{i}.mlp.dense_4h_to_h.weight.{self.tensor_para_rank}.bin")):
                    t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.mlp.dense_4h_to_h.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
                else:
                    t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.mlp.deepspeed_moe.experts.deepspeed_experts.dense_4h_to_h.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
            self.w.append(torch.cat(t_list, 0).contiguous().cuda())
            # ================================================================

            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.final_layernorm.weight.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)

            # assume absolute position
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/position_embeddings.weight.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/word_embeddings.weight.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)

            if self.t5_with_bias:
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.input_layernorm.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.attention.query.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.attention.key.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.attention.value.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.attention.dense.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.post_attention_layernorm.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)

                # =========== process normal and moe dense layer =================
                t_list = []
                for i in range(start_layer, end_layer):
                    if (os.path.isfile(f"{ckpt_path}/encoder.layers.{i}.mlp.dense_h_to_4h.bias.{self.tensor_para_rank}.bin")):
                        t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.mlp.dense_h_to_4h.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
                    else:
                        t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.mlp.deepspeed_moe.experts.deepspeed_experts.dense_h_to_4h.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
                self.w.append(torch.cat(t_list, 0).contiguous().cuda())
                # ================================================================

                # We don't have use_gated_activation in Megatron-DeepSpeed currently, so here weight placeholder is always empty
                # If we have it in the future, the binary file name should be modified according to the actual name.
                if self.use_gated_activation:
                    t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.block.{i}.layer.1.DenseReluDense.wi2.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                    self.w.append(t)
                else:
                    self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
               
                # =========== process normal and moe dense layer =================
                t_list = []
                for i in range(start_layer, end_layer):
                    if (os.path.isfile(f"{ckpt_path}/encoder.layers.{i}.mlp.dense_4h_to_h.bias.bin")):
                        t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.mlp.dense_4h_to_h.bias.bin", dtype=np_weight_dtype)))
                    else:
                        t_list.append(torch.zeros_like(torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.mlp.deepspeed_moe.experts.deepspeed_experts.dense_4h_to_h.bias.bin", dtype=np_weight_dtype))))
                
                self.w.append(torch.cat(t_list, 0).contiguous().cuda())
                # ================================================================

                t = torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.final_layernorm.bias.bin", dtype=np_weight_dtype)).contiguous().cuda()
                self.w.append(t)
            else:
                for i in range(10):
                    self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
            
            if self.t5_with_moe:
                gate_list = []
                for i in range(start_layer, end_layer):
                    if (os.path.isfile(f"{ckpt_path}/encoder.layers.{i}.mlp.deepspeed_moe.gate.wg.weight.bin")):
                        gate_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/encoder.layers.{i}.mlp.deepspeed_moe.gate.wg.weight.bin", dtype=np_weight_dtype)))
                self.w.append(torch.stack(gate_list, 0).contiguous().cuda())
            else:
                self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())

            # adapters are not supported in Megatron-DeepSpeed currently, so here weight placeholder is always empty
            for i in range(8):
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


class FTT5Encoder(nn.Module):
    def __init__(self, encoder_weight_list, lib_path, head_num, head_size, inter_size, d_model, is_remove_padding,
                 num_layer, num_bucket=32, num_expert=0, moe_layer_index=[], max_distance=128, sparse=False,
                 q_scaling=1.0, tensor_para_size=1, pipeline_para_size=1, t5_with_bias=False,
                 position_embedding_type=0, moe_k=0, activation_type="relu", adapter_inter_size=0,
                 adapter_norm_position="pre"):
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
            self.encoder = torch.classes.FasterTransformer.T5Encoder(*encoder_weight_list, moe_layer_index, head_num,
                                                                     head_size, inter_size, d_model,
                                                                     is_remove_padding, num_layer, num_bucket,
                                                                     num_expert, max_distance, sparse, q_scaling,
                                                                     tensor_para_size, pipeline_para_size,
                                                                     t5_with_bias, position_embedding_type, moe_k,
                                                                     activation_type, adapter_inter_size,
                                                                     adapter_norm_position)
        except:
            self.encoder = torch.classes.FasterTransformerT5Encoder(*encoder_weight_list, moe_layer_index, head_num,
                                                                    head_size, inter_size, d_model,
                                                                    is_remove_padding, num_layer, num_bucket,
                                                                    num_expert, max_distance, sparse, q_scaling,
                                                                    tensor_para_size, pipeline_para_size,
                                                                    t5_with_bias, position_embedding_type, moe_k,
                                                                    activation_type, adapter_inter_size,
                                                                    adapter_norm_position)

    def forward(self, input, seq_len, inputs_embeds=None):
        output = self.encoder.forward(input, seq_len, inputs_embeds)
        return output