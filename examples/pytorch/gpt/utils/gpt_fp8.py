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

from __future__ import print_function

import argparse
import dataclasses
import json
import os
import pathlib
import typing

import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist


class GPTFp8Weights(object):
    def __init__(self, head_num, size_per_head, layer_num, vocab_size, max_seq_len, tensor_para_size, pipeline_para_size,
                 has_post_decoder_layernorm=True, int8_mode=0, fp8_mode=0, weights_data_type=typing.Union[str, np.float32]):
        assert(head_num % tensor_para_size == 0)

        if int8_mode != 0:
            self.weight_transpose_calibrate_quantize = torch.ops.fastertransformer.weight_transpose_calibrate_quantize

        self.head_num = head_num
        self.size_per_head = size_per_head
        self.layer_num = layer_num
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.layers_per_device = layer_num // pipeline_para_size

        self.has_post_decoder_layernorm = has_post_decoder_layernorm

        local_head_num = head_num // tensor_para_size
        global_head_num = head_num
        local_hidden_units = local_head_num * size_per_head
        global_hidden_units = global_head_num * size_per_head
        local_inter_size = local_hidden_units * 4

        self.local_head_num = local_head_num
        self.global_head_num = global_head_num
        self.local_hidden_units = local_hidden_units
        self.global_hidden_units = global_hidden_units
        self.local_inter_size = local_inter_size

        self.int8_mode = int8_mode

        if isinstance(weights_data_type, str):
            try:
                weights_data_type = {
                    "fp16": np.float16,
                    "fp32": np.float32,
                    "float16": np.float16,
                    "float32": np.float32,
                }[weights_data_type]
            except KeyError:
                raise ValueError(f"Don't know how to interpret weights_data_type: {weights_data_type}")

        assert weights_data_type in [np.float32, np.float16]
        self.weights_data_type = weights_data_type

        self.w = []
        self.int8_w = []
        self.scale = []
        # Transformer blocks
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)   # self_layernorm_gamma
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)   # self_layernorm_beta
        self.w.extend([torch.zeros(global_hidden_units, local_hidden_units * 3)] * layer_num)   # self_kernel
        self.w.extend([torch.zeros(local_hidden_units * 3)] * layer_num)   # self_bias
        self.w.extend([torch.zeros(local_hidden_units, global_hidden_units)] * layer_num)   # self_output_kernel
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)   # self_output_bias
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)   # ffn_layernorm_gamma
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)   # ffn_layernorm_beta
        self.w.extend([torch.zeros(global_hidden_units, local_inter_size)] * layer_num)   # ffn_kernel1
        self.w.extend([torch.zeros(local_inter_size)] * layer_num)   # ffn_bias1
        self.w.extend([torch.zeros(local_inter_size, global_hidden_units)] * layer_num)   # ffn_kernel2
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)   # ffn_bias2
        # After Transformer blocks
        if self.has_post_decoder_layernorm:
            self.w.append(torch.zeros(global_hidden_units))   # layernorm_gamma
            self.w.append(torch.zeros(global_hidden_units))   # layernorm_beta
        self.w.append(torch.zeros(max_seq_len, global_hidden_units))   # position_encoding_table
        self.w.append(torch.zeros(vocab_size, global_hidden_units))   # embedding_table
        self.w.append(torch.zeros(vocab_size, global_hidden_units))   # embedding_kernel

        # Initialization
        self._map(lambda w: torch.nn.init.normal_(w, mean=0., std=1.))


        if (self.int8_mode != 0):
            self.int8_w.extend([torch.zeros(global_hidden_units, local_hidden_units * 3, dtype=torch.int8)] * layer_num)   # self_int8_kernel
            self.scale.extend([torch.zeros(local_hidden_units * 3, dtype=torch.float)] * layer_num)   # self_scale
            self.int8_w.extend([torch.zeros(local_hidden_units, global_hidden_units, dtype=torch.int8)] * layer_num)   # self_output_int8_kernel
            self.scale.extend([torch.zeros(global_hidden_units, dtype=torch.float)] * layer_num)   # self_output_scale
            self.int8_w.extend([torch.zeros(global_hidden_units, local_inter_size, dtype=torch.int8)] * layer_num)   # ffn_int8_kernel1
            self.scale.extend([torch.zeros(local_inter_size, dtype=torch.float)] * layer_num)   # ffn_scale1
            self.int8_w.extend([torch.zeros(local_inter_size, global_hidden_units, dtype=torch.int8)] * layer_num)   # ffn_int8_kernel2
            self.scale.extend([torch.zeros(global_hidden_units, dtype=torch.float)] * layer_num)   # ffn_scale2



    def __getitem__(self, idx):
        return self.w[idx]

    def __setitem__(self, idx, val):
        self.w[idx] = val

    def __len__(self):
        return len(self.w)

    def _map(self, func):
        for i in range(len(self.w)):
            if isinstance(self.w[i], list):
                for j in range(len(self.w[i])):
                    self.w[i][j] = func(self.w[i][j])
            else:
                self.w[i] = func(self.w[i])

    def _map_int8(self, func):
        for i in range(len(self.int8_w)):
            if isinstance(self.int8_w[i], list):
                for j in range(len(self.int8_w[i])):
                    self.int8_w[i][j] = func(self.int8_w[i][j])

            else:
                self.int8_w[i] = func(self.int8_w[i])
        for i in range(len(self.scale)):
            if isinstance(self.scale[i], list):
                for j in range(len(self.scale[i])):
                    self.scale[i][j] = func(self.scale[i][j])

            else:
                self.scale[i] = func(self.scale[i])

    def load(self, ckpt_path, tensor_para_rank, pipeline_para_rank):
        if not os.path.exists(ckpt_path):
            return False
        w = []

        # Load
        def is_load(i): return i >= self.layers_per_device * \
            pipeline_para_rank and i < self.layers_per_device * (pipeline_para_rank + 1)
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.input_layernorm.weight.bin".format(i),
                                               dtype=self.weights_data_type)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.input_layernorm.bias.bin".format(i),
                                               dtype=self.weights_data_type)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.query_key_value.weight.{}.bin".format(i,
                                                                                                                             tensor_para_rank), dtype=self.weights_data_type)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.query_key_value.bias.{}.bin".format(i,
                                                                                                                           tensor_para_rank), dtype=self.weights_data_type)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.dense.weight.{}.bin".format(i,
                                                                                                                   tensor_para_rank), dtype=self.weights_data_type)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.dense.bias.bin".format(i),
                                               dtype=self.weights_data_type)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.post_attention_layernorm.weight.bin".format(i),
                                               dtype=self.weights_data_type)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.post_attention_layernorm.bias.bin".format(i),
                                               dtype=self.weights_data_type)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_h_to_4h.weight.{}.bin".format(i,
                                                                                                                     tensor_para_rank), dtype=self.weights_data_type)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_h_to_4h.bias.{}.bin".format(i,
                                                                                                                   tensor_para_rank), dtype=self.weights_data_type)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_4h_to_h.weight.{}.bin".format(i,
                                                                                                                     tensor_para_rank), dtype=self.weights_data_type)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_4h_to_h.bias.bin".format(i),
                                               dtype=self.weights_data_type)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])

        if self.has_post_decoder_layernorm:
            w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.final_layernorm.weight.bin", dtype=self.weights_data_type)))
            w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.final_layernorm.bias.bin", dtype=self.weights_data_type)))

        wpe = torch.from_numpy(np.fromfile(ckpt_path + "/model.wpe.bin", dtype=self.weights_data_type)
                               ).reshape(-1, self.global_hidden_units)
        assert self.max_seq_len <= wpe.size(0), (
            f"max_seq_len ({self.max_seq_len} must not exceed "
            f"the value of maximum sequence length during training ({wpe.size(0)})."
        )
        w.append(wpe)
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.wte.bin", dtype=self.weights_data_type)))
        if os.path.isfile(ckpt_path + "/model.lm_head.weight.bin"):
            w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.lm_head.weight.bin", dtype=self.weights_data_type)))
        else:
            w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.wte.bin", dtype=self.weights_data_type)))

        # Reshape
        try:
            for i in range(len(w)):
                if w[i].nelement() > 0:
                    self.w[i] = w[i].reshape(self.w[i].shape)

        except RuntimeError:
            raise RuntimeError(
                f"head_num, size_per_head, vocab_size, and max_seq_len must be the same as the ones during training "
                f"(idx: {i} expected shape: {self.w[i].shape} got shape: {w[i].shape})."
            )

        #transpose calibrate quantize the kernel
        layer_num = self.layer_num
        if self.int8_mode != 0:
            for i in range(layer_num):
                self.int8_w[i + 0*layer_num], self.scale[i + 0*layer_num] = self.weight_transpose_calibrate_quantize(self.w[2*layer_num + i])
                self.int8_w[i + 1*layer_num], self.scale[i + 1*layer_num] = self.weight_transpose_calibrate_quantize(self.w[4*layer_num + i])
                self.int8_w[i + 2*layer_num], self.scale[i + 2*layer_num] = self.weight_transpose_calibrate_quantize(self.w[8*layer_num + i])
                self.int8_w[i + 3*layer_num], self.scale[i + 3*layer_num] = self.weight_transpose_calibrate_quantize(self.w[10*layer_num + i])

        return True


class GPTFp8(nn.Module):
    def __init__(self,
                 head_num, size_per_head,
                 vocab_size, start_id, end_id, layer_num,
                 max_seq_len,
                 tensor_para_size, pipeline_para_size,
                 lib_path,
                 ckpt_path,
                 layernorm_eps = 1e-6, layernorm_type = "pre_layernorm", # gpt_variant_params
                 activation_type = "Gelu", has_post_decoder_layernorm = True, # gpt variant params
				 int8_mode = 0,
                 fp8_mode = 1,
                 weights_data_type: np.dtype = np.float32):
        super().__init__()
        self.head_num = head_num
        self.size_per_head = size_per_head
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.start_id = start_id
        self.end_id = end_id
        self.layer_num = layer_num
        # gpt_variant_params
        self.layernorm_eps = layernorm_eps
        self.layernorm_type = layernorm_type
        self.activation_type = activation_type
        self.has_post_decoder_layernorm = has_post_decoder_layernorm
        # multi-gpu params
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.use_sparse_gemm = False
        self.int8_mode = int8_mode
        self.fp8_mode = fp8_mode
        self.weights_data_type = weights_data_type
        self.ckpt_path = ckpt_path

        assert torch.cuda.is_available(), "CUDA is required for this model."

        assert head_num % tensor_para_size == 0, "head_num must be a multiple of tensor_para_size."
        assert layer_num % pipeline_para_size == 0, "layer_num must be a multiple of pipeline_para_size."

        # Load the C++ model into Pytorch model.
        torch.classes.load_library(os.path.abspath(lib_path))

        # Prepare for tensor/pipeline parallel
        try:
            dist.init_process_group(backend='mpi')
        except:
            print("[INFO] WARNING: Have initialize the process group")
        self.rank = dist.get_rank()
        self.device_count = torch.cuda.device_count()
        self.device = self.rank % self.device_count
        torch.cuda.set_device(self.device)

        world_size = dist.get_world_size()
        assert world_size == tensor_para_size * pipeline_para_size, "tensor_para_size * pipeline_para_size must be equal to world_size."

        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.pipeline_para_rank = self.rank // self.tensor_para_size

        self.model = torch.classes.FasterTransformer.GptFp8Op(self.head_num, self.size_per_head, 4 * self.head_num * self.size_per_head,
                                                           self.layer_num, self.vocab_size, self.max_seq_len, self.start_id, self.end_id,
                                                           self.tensor_para_size, self.pipeline_para_size,
                                                           self.layernorm_eps, self.layernorm_type, self.activation_type, self.ckpt_path,
                                                           self.has_post_decoder_layernorm, [])

    def forward(self,
                start_ids,
                start_lengths,
                output_len,
                beam_width=1,
                top_k=1,
                top_p=0.0,
                beam_search_diversity_rate=0.0,
                temperature=1.0,
                len_penalty=1.0,
                repetition_penalty=1.0,
                random_seed=0,
                return_output_length=False,
                return_cum_log_probs=0):
        input_len = start_ids.size(1)
        assert input_len > 0, "input len must be larger than zero. For an unconditional case, use start_id as the first token."

        # Inputs to device
        start_ids = start_ids.cuda(self.device)
        start_lengths = start_lengths.cuda(self.device)
        # outputs: output_ids, output_lengths, output_cum_log_probs (optional)
        outputs = self.model.forward(start_ids,
                                     start_lengths,
                                     output_len,
                                     beam_width, # optional, can be None
                                     top_k, # optional, can be None
                                     top_p, # optional, can be None
                                     beam_search_diversity_rate, # optional, can be None
                                     temperature, # optional, can be None
                                     len_penalty, # optional, can be None
                                     repetition_penalty, # optional, can be None
                                     random_seed, # optional, can be None
                                     return_cum_log_probs) # optional, can be None
        if return_cum_log_probs == 0:
            output_ids, output_lengths = outputs
        else:
            output_ids, output_lengths, output_cum_log_probs = outputs
        if return_output_length:
            if return_cum_log_probs > 0:
                return output_ids, output_lengths, output_cum_log_probs
            else:
                return output_ids, output_lengths
        else:
            return output_ids

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor


@dataclasses.dataclass
class GptInitModelParameters:
    head_num: int
    size_per_head: int
    layer_num: int
    max_seq_len: int
    tensor_para_size: int
    vocab_size: int
    start_id: int
    end_id: int
    pipeline_para_size: int
    weights_data_type: str
    data_type: str
    int8_mode: int
    sparse: int

    def gpt_init_kwargs(self):
        do_not_include = ["data_type", "sparse"]
        return {k: v for k, v in dataclasses.asdict(self).items() if k not in do_not_include}

    @classmethod
    def from_args(cls, args, config_reader):
        model_name = args.model_name

        return cls(
            head_num=config_reader.getint(model_name, "head_num"),
            size_per_head=config_reader.getint(model_name, "size_per_head"),
            layer_num=config_reader.getint(model_name, "num_layer"),
            max_seq_len=config_reader.getint(model_name, "max_pos_seq_len"),
            tensor_para_size=config_reader.getint(model_name, "tensor_para_size"),
            vocab_size=config_reader.getint(model_name, "vocab_size"),
            start_id=config_reader.getint(model_name, "start_id"),
            end_id=config_reader.getint(model_name, "end_id"),
            weights_data_type=config_reader.get(model_name, "weight_data_type"),
            pipeline_para_size=(
                args.pipeline_para_size or config_reader.getint("ft_instance_hyperparameter", "pipeline_para_size")
            ),
            int8_mode=(
                args.int8_mode
                if args.int8_mode is not None
                else config_reader.getint("ft_instance_hyperparameter", "int8_mode")
            ),
            data_type=(args.data_type or config_reader.get("ft_instance_hyperparameter", "data_type")),
            sparse=int(args.sparse or False),
        )

    @classmethod
    def update_argparser(cls, parser):
        parser.add_argument("--model-name", type=str, default="gpt", help="Model name from config.ini file")
        parser.add_argument("--pipeline-para-size", type=int, help="size of pipeline parallelism")
        parser.add_argument("--data-type", type=str, help="data type", choices=["fp32", "bf16", "fp16"])
        parser.add_argument(
            "--sparse",
            type=int,
            choices=[0, 1],
            help="Set sparse matrix multiplication. (Need SM 8.0 or 8.6 and SPARSITY_SUPPORT=ON)",
        )
        parser.add_argument("--int8-mode", type=int, choices=[0, 1], help="Set int8 mode")


@dataclasses.dataclass
class GptRuntimeModelParameters:
    beam_width: int
    top_k: int
    top_p: float
    beam_search_diversity_rate: float
    temperature: float
    len_penalty: float
    repetition_penalty: float

    def gpt_forward_kwargs(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_args(cls, args, config_reader):
        return cls(
            beam_width=args.beam_width or config_reader.getint("ft_instance_hyperparameter", "beam_width"),
            top_k=args.sampling_top_k or config_reader.getint("ft_instance_hyperparameter", "top_k"),
            top_p=args.sampling_top_p or config_reader.getfloat("ft_instance_hyperparameter", "top_p"),
            beam_search_diversity_rate=(
                args.beam_search_diversity_rate
                or config_reader.getfloat("ft_instance_hyperparameter", "beam_search_diversity_rate")
            ),
            temperature=args.temperature or config_reader.getfloat("ft_instance_hyperparameter", "temperature"),
            len_penalty=args.len_penalty or config_reader.getfloat("ft_instance_hyperparameter", "len_penalty"),
            repetition_penalty=(
                args.repetition_penalty or config_reader.getfloat("ft_instance_hyperparameter", "repetition_penalty")
            ),
        )

    @classmethod
    def update_argparser(cls, parser):
        parser.add_argument("--beam-width", type=int, help="beam width")
        parser.add_argument("--sampling-top-k", type=int, help="Candidate (k) value of top k sampling in decoding")
        parser.add_argument("--sampling-top-p", type=float, help="Probability (p) value of top p sampling in decoding.")
        parser.add_argument("--temperature", type=float, help="temperature")
        parser.add_argument("--len-penalty", type=float, help="len_penalty")
        parser.add_argument("--repetition-penalty", type=float, help="repetition penalty")
        parser.add_argument("--beam-search-diversity-rate", type=float, help="beam_search_diversity_rate")


DEFAULT_START_TAG = "<|endoftext|>"
DEFAULT_END_TAG = "<|endoftext|>"
OPENAI_GPT2_START_ID = 50256
OPENAI_GPT2_END_ID = 50256


@dataclasses.dataclass
class GptModelConfig:
    model_name: str
    tensor_para_size: int
    head_num: int
    size_per_head: int
    inter_size: int
    num_layer: int
    max_pos_seq_len: int
    weight_data_type: str
    vocab_size: int
    start_id: int
    end_id: int

    @classmethod
    def from_nemo_package(
        cls,
        *,
        args: argparse.Namespace,
        nemo_model_config: typing.Dict[str, typing.Any],
        vocab_path: typing.Optional[pathlib.Path] = None,
        vocab_size: int,
    ):

        if vocab_path:
            vocab_path = pathlib.Path(vocab_path)
            with vocab_path.open("r") as vocab_file:
                vocab = json.load(vocab_file)
            start_id, end_id = vocab[DEFAULT_START_TAG], vocab[DEFAULT_END_TAG]
        else:
            start_id, end_id = OPENAI_GPT2_START_ID, OPENAI_GPT2_END_ID

        return cls(
            model_name="gpt",
            tensor_para_size=args.infer_gpu_num,
            head_num=nemo_model_config["num_attention_heads"],
            size_per_head=nemo_model_config["hidden_size"] // nemo_model_config["num_attention_heads"],
            inter_size=nemo_model_config["ffn_hidden_size"],
            num_layer=nemo_model_config["num_layers"],
            max_pos_seq_len=nemo_model_config["max_position_embeddings"],
            weight_data_type=args.weight_data_type,
            vocab_size=vocab_size,
            start_id=start_id,
            end_id=end_id,
        )
