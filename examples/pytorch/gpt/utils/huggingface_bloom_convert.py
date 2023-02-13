# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

""" Convert HuggingFace pretrained checkpoint into FT format.

"""

import argparse
import configparser
import logging
import multiprocessing
import re
import time

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn
from transformers import AutoModel, BloomConfig, PretrainedConfig

PathLike = Union[str, Path]

DATATYPE_MAP = dict(
    fp32=torch.float32,
    fp16=torch.float16
)

_args = None


logger = logging.getLogger()  # get the root logger.


def set_logger(verbose=False):
    logging.basicConfig(
        # do not print logging level to make it print-like.
        format='%(message)s',
        level=logging.DEBUG if verbose else logging.INFO)


def get_args():
    global _args
    if _args is not None:
        return _args

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-dir', type=str, metavar='DIR', required=True,
        help='A checkpoint directory of a huggingface pretrained model.')
    parser.add_argument(
        '-o', '--output-dir', type=str, metavar='DIR', required=True,
        help='A directory where converted binary files for FT will be saved.')
    parser.add_argument(
        '-tp', '--tensor-para-size', type=int, metavar='N', default=1,
        help='The tensor parallel size for inference.')
    parser.add_argument(
        '-dt', '--data-type', type=str, metavar='STR', default='fp32',
        choices=list(DATATYPE_MAP),
        help='A data type of converted weights.')
    parser.add_argument(
        '-p', '--processes', type=int, metavar='N', default=1,
        help='The number of parallel processes to use for conversion.')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable verbose logging')
    _args = parser.parse_args()

    set_logger(_args.verbose)

    logger.info('\n======================= Arguments =======================')
    for k, v in vars(_args).items():
        logger.info(f' - {k.ljust(20, ".")}: {v}')
    logger.info('=========================================================')

    return _args


parameter_prefix_map = {
    r'^h.': 'layers.',
}

# pattern and replacement map.
parameter_rename_map = {
    # CasualLM weights
    'word_embeddings.weight': 'wte',
    'word_embeddings_layernorm.weight': 'pre_decoder_layernorm.weight',
    'word_embeddings_layernorm.bias': 'pre_decoder_layernorm.bias',
    'ln_f.weight': 'final_layernorm.weight',
    'ln_f.bias': 'final_layernorm.bias',
    # Layer weights
    'self_attention.dense.weight': 'attention.dense.weight',
    'self_attention.dense.bias': 'attention.dense.bias',
    'self_attention.query_key_value.weight': 'attention.query_key_value.weight',
    'self_attention.query_key_value.bias': 'attention.query_key_value.bias',
}

parameter_to_split = [
    # tuple of (name, index to split)
    ('attention.query_key_value.weight', -1),
    ('attention.query_key_value.bias', -1),
    ('attention.dense.weight', 0),
    ('mlp.dense_h_to_4h.weight', -1),
    ('mlp.dense_h_to_4h.bias', -1),
    ('mlp.dense_4h_to_h.weight', 0)
]


def safe_transpose(param: torch.nn.Parameter):
    return param.T if len(param.shape) == 2 else param


def convert_parameter_name(name: str):
    # A parameter in BloomForCausalLM has an additional prefix 'transformer.'.
    if name.startswith('transformer.'):
        name = name[len('transformer.'):]
    # Regularize the weight prefix.
    for pattern, rename in parameter_prefix_map.items():
        if re.match(pattern, name):
            name = re.sub(pattern, rename, name)
            break
    # Rename weight names.
    name_suffix = re.sub(r'layers.\d+.', '', name)
    if name_suffix in parameter_rename_map:
        name = name.replace(name_suffix, parameter_rename_map[name_suffix])
    # An GPT weight of FT has a prefix "model.".
    return 'model.' + name


def is_split_param(name: str):
    for phrase, _ in parameter_to_split:
        if phrase in name:
            return True
    return False


def axis_to_split(name: str):
    for phrase, axis in parameter_to_split:
        if phrase in name:
            return axis
    raise ValueError(f'Got a unexpected parameter name to split {name}')


# Exception handling.

def reorder_qkv_weight_or_bias(model_config: PretrainedConfig,
                               name: str,
                               param: torch.nn.Parameter):
    """ Reorder the qkv weight to use at FT.

    Note that the shape of the fused QKV weights in HF is different from the
    shape that FT requires.
       HF: (hidden_size, num_heads x 3 x head_dim)
       FT: (hidden_size, 3 x num_heads x head_dim)
    This is unlike to the other models in HF e.g. GPT where they have the
    same shape with FT, i.e., (hidden_size, 3 x num_heads x head_dim). Also,
    to split across attention heads in tensor parallel, we reshape the qkv
        weight: (hidden, 3, num_heads x head_dim).
        bias  : (3, num_heads x head_dim).

    # Args.
        model_config: PretrainedConfig, a model configuration.
        name: str, a parameter name.
        param: torch.nn.Parameter, a fused QKV weight or bias. of shape
            (..., num_heads * 3 * head_dim).
    # Returns.
        torch.nn.Parameter, a reordered fused QKV weight of size
            (..., 3, num_heads * head_dim).
    """

    if 'query_key_value' not in name:
        # Nothing to do for the non-eligible parameters.
        return param

    num_heads = model_config.n_head
    head_dim = model_config.hidden_size // model_config.n_head

    # (..., 3 x hidden) view as (..., num_heads, 3, head_dim)
    param = param.view(-1, num_heads, 3, head_dim)
    # permute to (..., 3, num_heads, head_dim)
    param = param.permute(0, 2, 1, 3)
    # final shape: weight=(hidden, 3, hidden) or bias=(3, hidden)
    if 'query_key_value.bias' in name:
        return param.reshape(3, num_heads * head_dim)
    return param.reshape(-1, 3, num_heads * head_dim)


def handle_exceptions(model_config: PretrainedConfig,
                      param_name: str,
                      param: torch.nn.Parameter):
    if 'query_key_value' in param_name:
        param = reorder_qkv_weight_or_bias(model_config, param_name, param)
    elif 'wte' in param_name:
        # The input word embedding shouldn't be transposed.
        param = param.T
    return param


def convert_and_save_parameter(param_name: str,
                               param,
                               tensor_para_size: Optional[int],
                               save_dir: PathLike):
    """ Convert and save to FT parameter

    Split a param into tensor_para_size if needed, and save each split param at
    {save_dir}/{param_name}.bin or {save_dir}/{param_name}.{tp_idx}.bin in case
    of a split param.

    # Args.
        model_config: PretrainedConfig, a model configuration.
        name: str, parameter name.
        param: torch.nn.Parameter, a model parameter to convert.
        tensor_para_size: int, tensor parallel size.
        save_dir: str or Path, a base directory of binary files.
    """

    save_dir = Path(save_dir)

    if not is_split_param(param_name):
        save_path = save_dir / f'{param_name}.bin'
        param.tofile(save_path)
        logger.debug(
            f' - {param_name.ljust(48, ".")}: shape {str(param.shape):16s}   '
            f'| saved at {str(save_path)}')
        return

    axis = axis_to_split(param_name)
    split_params = np.split(param, tensor_para_size, axis=axis)
    for tp_idx, split_param in zip(range(tensor_para_size), split_params):
        save_path = save_dir / f'{param_name}.{tp_idx}.bin'
        split_param.tofile(save_path)
        logger.debug(
            f' - {param_name.ljust(48, ".")}: shape {str(split_param.shape):16s} s '
            f'| saved at {str(save_path)} ({tp_idx}/{tensor_para_size})')


def save_bloom_config(model_config: BloomConfig, save_dir: PathLike):
    """ Save Bloom model configuration.

    Args:
        model_config: HF pretrained model configuration.
        save_dir: a directory to save the config file.
    """

    args = get_args()
    save_dir = Path(save_dir)
    save_dir.parent.mkdir(exist_ok=True, parents=True)

    config = configparser.ConfigParser()

    # FT's layernorm type string.
    if model_config.apply_residual_connection_post_layernorm:
        model_variant = 'bloom-post'
        layernorm_type = 'post_layernorm'
    else:
        model_variant = 'bloom-pre'
        layernorm_type = 'pre_layernorm'

    # We use the section name `gpt` since FT runs BLOOM model through the GPT
    # module, which requires the section name `gpt` to retrieve the weight
    # data type.
    config['gpt'] = dict(
        model_name=model_config.name_or_path,
        num_layer=model_config.n_layer,
        head_num=model_config.n_head,
        # inter_size is fixed in bloom model by 4 * hidden_size and a model
        # config does not include the intermediate dimension of FFN.
        inter_size=4 * model_config.hidden_size,
        size_per_head=model_config.hidden_size // model_config.n_head,
        vocab_size=model_config.vocab_size,
        tensor_para_size=args.tensor_para_size,
        weight_data_type=args.data_type,
        # GPT variant params
        model_variant=model_variant,
        layernorm_eps=model_config.layer_norm_epsilon,
        layernorm_type=layernorm_type,
        activation_type='Gelu',
        has_positional_encoding=False,
        has_pre_decoder_layernorm=True,
        has_post_decoder_layernorm=True,
        use_attention_linear_bias=True,
        # Special token ids.
        start_id=model_config.bos_token_id,
        end_id=model_config.eos_token_id,
    )

    with (save_dir / 'config.ini').open('w') as f:
        config.write(f, space_around_delimiters=False)


def main():
    args = get_args()
    tp_size = args.tensor_para_size

    dtype = DATATYPE_MAP[args.data_type]
    model = AutoModel.from_pretrained(args.input_dir).cpu().type(dtype)
    assert isinstance(model, torch.nn.Module)

    save_dir = Path(args.output_dir) / f'{tp_size}-gpu'
    save_dir.mkdir(exist_ok=True, parents=True)
    save_bloom_config(model.config, save_dir)

    start_time = time.time()
    logger.info(f'Start the checkpoint conversion: '
                f'{len(list(model.parameters()))} params')
    if args.processes > 1:
        pool = multiprocessing.Pool(args.processes)
        star_args = []
        for name, param in model.named_parameters():
            # Preprocess
            param_name = convert_parameter_name(name)
            param = safe_transpose(param)
            param = handle_exceptions(model.config, param_name, param)
            star_args.append((param_name, param.detach().cpu().numpy(), tp_size, save_dir))
        pool.starmap_async(convert_and_save_parameter, star_args)
        pool.close()
        pool.join()
    else:
        for name, param in model.named_parameters():
            # Preprocess
            param_name = convert_parameter_name(name)
            param = safe_transpose(param)
            param = handle_exceptions(model.config, param_name, param)
            convert_and_save_parameter(param_name, param.detach().cpu().numpy(), tp_size, save_dir)
    elapsed_time = time.time() - start_time
    logger.info(f'Checkpoint conversion (HF >> FT) has done '
                f'(elapsed time: {elapsed_time:.2f} sec)')


if __name__ == '__main__':
    main()
