# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


def add_special_tokens_to_tokenizer(tokenizer):

    # Need to add cls, sep, mask tokens to the tokenizer if they don't exist.
    # If cls, sep and mask are not attributes of the tokenizer, add it.
    if not hasattr(tokenizer, 'cls_token'):
        tokenizer.add_special_tokens({'cls_token': '<cls>'})
    if not hasattr(tokenizer.tokenizer, 'sep_id'):
        tokenizer.add_special_tokens({'sep_token': '<sep>'})
    if not hasattr(tokenizer.tokenizer, 'mask_id'):
        tokenizer.add_special_tokens({'mask_token': '<mask>'})

    # bos, eos, pad and unk may be present in the provided spm .model file, if they are, use it.
    if not hasattr(tokenizer, 'pad_token'):
        if hasattr(tokenizer.tokenizer, 'pad_id') and tokenizer.tokenizer.pad_id() > 0:
            tokenizer.pad_token = tokenizer.tokenizer.id_to_piece(tokenizer.tokenizer.pad_id())
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
    else:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

    if not hasattr(tokenizer, 'bos_token'):
        if hasattr(tokenizer.tokenizer, 'bos_id') and tokenizer.tokenizer.bos_id() > 0:
            tokenizer.bos_token = tokenizer.tokenizer.id_to_piece(tokenizer.tokenizer.bos_id())
        else:
            tokenizer.add_special_tokens({'bos_token': '<bos>'})
    else:
        tokenizer.add_special_tokens({'bos_token': '<s>'})

    if not hasattr(tokenizer, 'eos_token'):
        if hasattr(tokenizer.tokenizer, 'eos_id') and tokenizer.tokenizer.eos_id() > 0:
            tokenizer.eos_token = tokenizer.tokenizer.id_to_piece(tokenizer.tokenizer.eos_id())
        else:
            tokenizer.add_special_tokens({'eos_token': '<eos>'})
    else:
        tokenizer.add_special_tokens({'eos_token': '</s>'})
