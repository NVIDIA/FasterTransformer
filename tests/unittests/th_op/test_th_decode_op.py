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

import itertools
import unittest
from pathlib import Path

import torch


@unittest.skipUnless(Path('lib/libth_transformer.so').exists(),
                     'lib/libth_transformer.so does not exist.')
class TestDecodeOp(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        torch.classes.load_library('lib/libth_transformer.so')

    def setUp(self) -> None:
        self.vocab_size = 1024
        self.vocab_size_padded = 1024
        self.tensor_para_size = 1
        self.pipeline_para_size = 1
        self.end_id = 0  # eos token id.
        self.decode_op = None
        self.decode_op = torch.classes.FasterTransformer.DynamicDecodeOp(
            self.vocab_size,
            self.vocab_size_padded,
            self.tensor_para_size,
            self.pipeline_para_size,
            torch.float)

    def tearDown(self):
        del self.decode_op

    def initialize_input_token_ids(self,
                                   batch_size,
                                   beam_width,
                                   max_input_length,
                                   use_random_input_tokens=True):
        device = torch.cuda.current_device()
        if use_random_input_tokens:
            input_token_ids = torch.randint(
                1, self.vocab_size, (batch_size, max_input_length),
                dtype=torch.int, device=device)
        else:
            input_token_ids = torch.ones(
                (batch_size, max_input_length), dtype=torch.int, device=device)
        input_token_ids += self.end_id
        input_token_ids.remainder_(self.vocab_size)
        input_lengths = torch.randint(
            1, max_input_length + 1, (batch_size,), dtype=torch.int, device=device)
        input_lengths[torch.randint(0, batch_size, (1,))] = max_input_length

        # mask by end_id.
        step_indices = torch.arange(0, max_input_length, device=device)\
            .unsqueeze(0).tile(batch_size, 1)
        input_token_ids[input_lengths.unsqueeze(1) <= step_indices] = self.end_id

        # Tiling.
        input_token_ids = input_token_ids.repeat(1, beam_width)\
            .view(batch_size * beam_width, -1)
        input_lengths = input_lengths.view(-1, 1).repeat(1, beam_width).view(-1)

        return input_token_ids, input_lengths

    @staticmethod
    def safe_to_vec(batch_size, value, dtype, is_cpu=False):
        if value is None:
            return None
        device = torch.device('cpu') if is_cpu else torch.cuda.current_device()
        return value * torch.ones(batch_size, dtype=dtype, device=device)

    def run_decode(self,
                   batch_size=4,
                   beam_width=1,
                   max_input_length=4,
                   gen_length=3,
                   ite=0,
                   local_batch_size=None,
                   top_k=None,
                   top_p=None,
                   temperature=None,
                   repetition_penalty=None,
                   presence_penalty=None,
                   min_length=None,
                   len_penalty=None,
                   beam_search_diversity_rate=None,
                   random_seed=None,
                   top_p_decay=None,
                   top_p_min=None,
                   top_p_reset_ids=None,
                   logit_fn=None,
                   use_random_input_tokens=True):
        ite = 0
        local_batch_size = batch_size
        max_seq_length = max_input_length + gen_length

        device = torch.cuda.current_device()

        eos_token_ids = self.safe_to_vec(batch_size, self.end_id, torch.int, False)
        top_ks = self.safe_to_vec(batch_size, top_k, torch.int, True)
        top_ps = self.safe_to_vec(batch_size, top_p, torch.float, True)
        temperatures = self.safe_to_vec(batch_size, temperature, torch.float, True)
        repetition_penalties = self.safe_to_vec(batch_size, repetition_penalty, torch.float, True)
        presence_penalties = self.safe_to_vec(batch_size, presence_penalty, torch.float, True)
        min_lengths = self.safe_to_vec(batch_size, min_length, torch.int, True)
        len_penalties = self.safe_to_vec(batch_size, len_penalty, torch.float, True)
        beam_search_diversity_rates = self.safe_to_vec(batch_size, beam_search_diversity_rate, torch.float, True)
        random_seeds = self.safe_to_vec(batch_size, random_seed, torch.int64, True)
        top_p_decays = self.safe_to_vec(batch_size, top_p_decay, torch.float, False)
        top_p_mins = self.safe_to_vec(batch_size, top_p_min, torch.float, False)
        top_p_reset_ids = self.safe_to_vec(batch_size, top_p_reset_ids, torch.int, False)

        embedding_bias = None
        sequence_limit_lengths = None  # limit step
        stop_words_list = None
        bad_words_list = None

        if beam_width > 1:
            parent_ids = torch.zeros(
                (max_seq_length, batch_size * beam_width),
                dtype=torch.int32, device=device)
            # src/tgt cache indirections.
            cache_indirection = torch.zeros(
                (2, batch_size, beam_width, max_seq_length),
                dtype=torch.int32, device=device)
            cum_log_probs = torch.zeros(batch_size * beam_width, device=device)
            output_log_probs = None
        else:
            parent_ids = None
            cache_indirection = None
            src_cache_indirection = None
            tgt_cache_indirection = None
            cum_log_probs = None
            output_log_probs = None

        input_token_ids, input_lengths = self.initialize_input_token_ids(
            batch_size, beam_width, max_input_length, use_random_input_tokens)

        self.decode_op.setup(batch_size,
                             beam_width,
                             top_ks,
                             top_ps,
                             temperatures,
                             repetition_penalties,
                             presence_penalties,
                             min_lengths,
                             len_penalties,
                             beam_search_diversity_rates,
                             random_seeds,
                             top_p_decays,
                             top_p_mins,
                             top_p_reset_ids)

        finished = torch.zeros_like(input_lengths).bool()
        sequence_lengths = (max_input_length - 1) * torch.ones_like(input_lengths)

        # Contiguous buffer for each decode_op step, will be transposed.
        output_token_ids = torch.zeros(
            (max_seq_length, batch_size * beam_width),
            dtype=torch.int32, device=device)
        output_token_ids[:max_input_length, ...] = input_token_ids.T

        for step in range(max_input_length, max_seq_length):
            if cache_indirection is not None:
                bidx = range(ite * local_batch_size,
                             min((ite + 1) * local_batch_size, batch_size))
                src_indir_idx = (step - max_input_length) % 2
                tgt_indir_idx = 1 - src_indir_idx
                src_cache_indirection = cache_indirection[src_indir_idx, bidx, ...]
                tgt_cache_indirection = cache_indirection[tgt_indir_idx, bidx, ...]

            if logit_fn is None:
                logits = torch.randn(
                    (batch_size, beam_width, self.vocab_size_padded), device=device)
            else:
                logits = logit_fn(batch_size, beam_width, device)

            should_stop = self.decode_op.forward(
                logits,
                step,
                max_input_length,
                ite,
                local_batch_size,
                eos_token_ids,
                top_ks,
                top_ps,
                temperatures,
                repetition_penalties,
                presence_penalties,
                min_lengths,
                len_penalties,
                beam_search_diversity_rates,
                top_p_decays,
                top_p_mins,
                top_p_reset_ids,
                embedding_bias,
                input_lengths,
                sequence_limit_lengths,
                stop_words_list,
                bad_words_list,
                src_cache_indirection,
                output_token_ids.view(-1, batch_size, beam_width),
                finished,
                sequence_lengths,
                cum_log_probs,
                output_log_probs,
                parent_ids,
                tgt_cache_indirection)

            if should_stop:
                break
        # Output sequence length is seqlen + 1 since
        output_sequence_lengths = sequence_lengths + 1

        return dict(
            output_token_ids=output_token_ids.T,
            output_sequence_lengths=output_sequence_lengths
        )

    def test_min_length_correctness_at_sampling(self):
        methods = [dict(top_k=1, top_p=0.0), dict(top_k=0, top_p=0.8)]
        testcases = [
            dict(batch_size=4, max_input_length=4, min_length=2, gen_length=4),
            dict(batch_size=4, max_input_length=4, min_length=32, gen_length=64),
            # batch exceeds 1024
            dict(batch_size=2048, max_input_length=6, min_length=4, gen_length=8),
        ]

        def logit_fn(batch_size, beam_width, device):
            logits = torch.randn(
                (batch_size, beam_width, self.vocab_size_padded), device=device)
            # Make the eos token be the most probable.
            logits[..., self.end_id] = logits.max(dim=-1)[0] + 1
            return logits

        for tc, method in itertools.product(testcases, methods):
            tc.update(method)
            with self.subTest(tc):
                output_dict = self.run_decode(beam_width=1, logit_fn=logit_fn, **tc)
                output_seq_lengths = output_dict['output_sequence_lengths']
                min_sequence_length = tc['max_input_length'] + tc['min_length']
                self.assertTrue(
                    (output_seq_lengths >= min_sequence_length).all(),
                    f'failed indices {torch.where(output_seq_lengths < min_sequence_length)[0]}, '
                    f'values {output_seq_lengths[torch.where(output_seq_lengths < min_sequence_length)[0]]}')

    def test_min_length_correctness_at_beamsearch(self):
        testcases = [
            # Online Beamsearch
            dict(batch_size=4, beam_width=2, max_input_length=4, min_length=2, gen_length=4),
            # Beamsearch
            dict(batch_size=4, beam_width=16, max_input_length=4, min_length=2, gen_length=4),
            # batch * beam exceeds 1024
            dict(batch_size=1024, beam_width=2, max_input_length=4, min_length=4, gen_length=8),
            dict(batch_size=128, beam_width=16, max_input_length=4, min_length=4, gen_length=8),
            # large beam_width
            dict(batch_size=4, beam_width=60, max_input_length=4, min_length=4, gen_length=8),
        ]

        def logit_fn(batch_size, beam_width, device):
            logits = torch.randn(
                (batch_size, beam_width, self.vocab_size_padded), device=device)
            # Make the eos token be the most probable.
            logits[..., self.end_id] = logits.max(dim=-1)[0] + 1
            return logits

        for tc in testcases:
            with self.subTest(tc):
                output_dict = self.run_decode(logit_fn=logit_fn, **tc)
                output_seq_lengths = output_dict['output_sequence_lengths']
                min_sequence_length = tc['max_input_length'] + tc['min_length']
                self.assertTrue(
                    (output_seq_lengths >= min_sequence_length).all(),
                    f'failed indices {torch.where(output_seq_lengths < min_sequence_length)[0]}, '
                    f'values {output_seq_lengths[torch.where(output_seq_lengths < min_sequence_length)[0]]}')

    def test_repetition_penalty_correctness(self):
        methods = [dict(top_k=1, top_p=0.0), dict(beam_width=2)]
        testcases = [
            dict(batch_size=4, max_input_length=4, repetition_penalty=2),
            dict(batch_size=2048, max_input_length=4, repetition_penalty=2),
            dict(batch_size=4, max_input_length=4, presence_penalty=0.5),
            dict(batch_size=4, max_input_length=4, presence_penalty=1.0),
            dict(batch_size=2048, max_input_length=4, presence_penalty=0.5),
        ]
        def logit_fn(batch_size, beam_width, device):
            logits = torch.zeros(
                (batch_size, beam_width, self.vocab_size_padded), device=device)
            # The token (vocab_size - 1) is the most probable unless penalized.
            # After penalized, the expected output token ids will be
            #   [v-1, v-2, v-3, v-1, v-1, v-1, ...].
            logits[..., self.vocab_size - 1] = 2
            logits[..., self.vocab_size - 2] = 1.8
            logits[..., self.vocab_size - 3] = 1.6
            return logits

        for tc, method in itertools.product(testcases, methods):
            tc.update(method)
            gen_length = 5
            beam_width = tc.get('beam_width', 1)
            expected_toekn_ids = (self.vocab_size - 1) * torch.ones(
                (tc['batch_size'] * beam_width, gen_length), dtype=torch.int32)
            expected_toekn_ids[:, 1] = self.vocab_size - 2
            expected_toekn_ids[:, 2] = self.vocab_size - 3
            with self.subTest(tc):
                output_dict = self.run_decode(
                    gen_length=gen_length,
                    use_random_input_tokens=False, logit_fn=logit_fn, **tc)
                output_token_ids = output_dict['output_token_ids'][:, tc['max_input_length']:]
                self.assertTrue(
                    (expected_toekn_ids.to(output_token_ids.device) == output_token_ids).all())


if __name__ == '__main__':
    unittest.main()
