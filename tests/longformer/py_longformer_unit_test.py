# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import unittest
import os
import sys

import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../..")

from examples.pytorch.longformer.longformer_qa import parse_from_config, build_ft_longformer, prepare_input, decode_output

class TestLongformerPytorchQA(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)
        self.passage_texts = [
            "Jim Henson was a nice puppet",
            "Tom went to the swamphack yesterday.",

            "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; "
            "Spanish: Selva Amazónica, Amazonía or usually Amazonia; "
            "French: Forêt amazonienne; Dutch: Amazoneregenwoud), "
            "also known in English as Amazonia or the Amazon Jungle, "
            "is a moist broadleaf forest that covers most of the Amazon basin of South America. "
            "This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), "
            "of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. "
            "This region includes territory belonging to nine nations. "
            "The majority of the forest is contained within Brazil, with 60% of the rainforest, "
            "followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, "
            "Ecuador, Bolivia, Guyana, Suriname and French Guiana. "
            "States or departments in four nations contain 'Amazonas' in their names. "
            "The Amazon represents over half of the planet's remaining rainforests, "
            "and comprises the largest and most biodiverse tract of tropical rainforest in the world, "
            "with an estimated 390 billion individual trees divided into 16,000 species."
        ]
        self.questions = [
            "Who was Jim Henson?",
            "When did Tom go to the swamphack?",
            "Which name is also used to describe the Amazon rainforest in English?"
        ]

        self.answers = [
            "puppet",
            "yesterday",
            "Jungle"
        ]

        self.model_dir = "examples/pytorch/longformer/longformer-large-4096-finetuned-triviaqa"
        self.ft_longformer_lib = os.path.join('build', 'lib', 'libth_longformer.so')

    def run_all_qa(self, seq_len, batch_size, ft_longformer, fp16):
        for idx in range(len(self.passage_texts)):
            passage_text = self.passage_texts[idx]
            question = self.questions[idx]
            answer = self.answers[idx]

            input_ids_b, local_attn_mask_b, global_attn_mask_b, input_ids, actual_seq_len = prepare_input(
                question, passage_text, seq_len, batch_size, self.model_dir, fp16)

            with torch.no_grad():
                outputs = ft_longformer(input_ids_b,
                                        attention_mask=local_attn_mask_b,
                                        global_attention_mask=global_attn_mask_b)
            answer_predict = decode_output(outputs, self.model_dir, input_ids, actual_seq_len)
            self.assertTrue(answer_predict.strip() == answer)

    def test_fp32_with_qa_answer(self):
        seq_len = 1024
        batch_size = 1
        max_global_token_num = 128

        (layer_num, _, head_num, size_per_head,
         intermediate_size, local_attn_window_size, attn_scaler) = parse_from_config(self.model_dir)

        ft_longformer = build_ft_longformer(self.model_dir, layer_num, head_num, size_per_head,
                                            intermediate_size, local_attn_window_size,
                                            max_global_token_num, batch_size, seq_len,
                                            attn_scaler, self.ft_longformer_lib, fp16=False)

        self.run_all_qa(seq_len, batch_size, ft_longformer, False)

    def test_fp32_with_qa_answer_2(self):
        seq_len = 2048
        batch_size = 5
        max_global_token_num = 96

        (layer_num, _, head_num, size_per_head,
         intermediate_size, local_attn_window_size, attn_scaler) = parse_from_config(self.model_dir)

        ft_longformer = build_ft_longformer(self.model_dir, layer_num, head_num, size_per_head,
                                            intermediate_size, local_attn_window_size,
                                            max_global_token_num, batch_size, seq_len,
                                            attn_scaler, self.ft_longformer_lib, fp16=False)

        self.run_all_qa(seq_len, batch_size, ft_longformer, False)

    def test_fp32_with_qa_answer_3(self):
        seq_len = 4096
        batch_size = 3
        max_global_token_num = 512

        (layer_num, _, head_num, size_per_head,
         intermediate_size, local_attn_window_size, attn_scaler) = parse_from_config(self.model_dir)

        ft_longformer = build_ft_longformer(self.model_dir, layer_num, head_num, size_per_head,
                                            intermediate_size, local_attn_window_size,
                                            max_global_token_num, batch_size, seq_len,
                                            attn_scaler, self.ft_longformer_lib, fp16=False)

        self.run_all_qa(seq_len, batch_size, ft_longformer, False)

    def test_fp16_with_qa_answer(self):
        seq_len = 1024
        batch_size = 1
        max_global_token_num = 128

        (layer_num, _, head_num, size_per_head,
         intermediate_size, local_attn_window_size, attn_scaler) = parse_from_config(self.model_dir)

        ft_longformer = build_ft_longformer(self.model_dir, layer_num, head_num, size_per_head,
                                            intermediate_size, local_attn_window_size,
                                            max_global_token_num, batch_size, seq_len,
                                            attn_scaler, self.ft_longformer_lib, fp16=True)

        self.run_all_qa(seq_len, batch_size, ft_longformer, True)

    def test_fp16_with_qa_answer_2(self):
        seq_len = 1536
        batch_size = 4
        max_global_token_num = 64

        (layer_num, _, head_num, size_per_head,
         intermediate_size, local_attn_window_size, attn_scaler) = parse_from_config(self.model_dir)

        ft_longformer = build_ft_longformer(self.model_dir, layer_num, head_num, size_per_head,
                                            intermediate_size, local_attn_window_size,
                                            max_global_token_num, batch_size, seq_len,
                                            attn_scaler, self.ft_longformer_lib, fp16=True)

        self.run_all_qa(seq_len, batch_size, ft_longformer, True)

    def test_fp16_with_qa_answer_3(self):
        seq_len = 4096
        batch_size = 8
        max_global_token_num = 256

        (layer_num, _, head_num, size_per_head,
         intermediate_size, local_attn_window_size, attn_scaler) = parse_from_config(self.model_dir)

        ft_longformer = build_ft_longformer(self.model_dir, layer_num, head_num, size_per_head,
                                            intermediate_size, local_attn_window_size,
                                            max_global_token_num, batch_size, seq_len,
                                            attn_scaler, self.ft_longformer_lib, fp16=True)

        self.run_all_qa(seq_len, batch_size, ft_longformer, True)


if __name__ == "__main__":
    unittest.main()
