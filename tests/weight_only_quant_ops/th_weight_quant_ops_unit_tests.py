# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import unittest

class TestQuantize(unittest.TestCase):

    def setUp(self) -> None:
        torch.classes.load_library("lib/libth_weight_only_quant_ops.so")
        self.add_bias_and_interleave_int4s = torch.ops.fastertransformer._add_bias_and_interleave_int4s
        self.add_bias_and_interleave_int8s = torch.ops.fastertransformer._add_bias_and_interleave_int8s
        self.unpack_packed_int4s = torch.ops.fastertransformer.unpack_int4_packed_tensor_to_int8
        self.pack_int4s = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4

        self.quant_interleave = torch.ops.fastertransformer._permute_B_rows_for_mixed_gemm

        self.subbyte_transpose = torch.ops.fastertransformer._subbyte_transpose

    def reference_interleave(self, tensor, quant_type):
        assert quant_type == torch.int8 or quant_type == torch.quint4x2
        tile_rows = 16 if quant_type == torch.int8 else 32

        permutation_map = []
        if tile_rows == 16:
          permutation_map = [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15]
        elif tile_rows == 32:
          permutation_map = [0,1,8,9,16,17,24,25,2,3,10,11,18,19,26,27,4,5,12,13,20,21,28,29,6,7,14,15,22,23,30,31]
        else:
          assert False, "Unsuppoered tile cols"

        permutation_map = torch.tensor(permutation_map)

        temp = tensor.reshape([-1, tile_rows, tensor.shape[-1]])
        temp = temp[:, permutation_map, :]
        return temp.reshape(tensor.shape)

    def interleave_tester(self, quant_type, arch):
        assert quant_type == torch.int8 or quant_type == torch.quint4x2
        experts  = [1, 4, 8]
        m_shapes = [128, 256, 1024]
        n_shapes = [128, 512, 1024, 4096]

        lower_bound = -128 if quant_type == torch.int8 else -8
        upper_bound = 127 if quant_type == torch.int8 else 7

        ref_impl = self.reference_interleave

        for expert in experts:
            for m_shape in m_shapes:
                for n_shape in n_shapes:
                    tensor = torch.randint(lower_bound, upper_bound, [expert, m_shape, n_shape], dtype=torch.int8)
                    ref_result = ref_impl(tensor.reshape([expert * m_shape, n_shape]), quant_type).reshape([expert, m_shape, n_shape])

                    if quant_type == torch.quint4x2:
                        tensor = self.pack_int4s(tensor)

                    act_result = self.quant_interleave(tensor, quant_type, arch)

                    if quant_type == torch.quint4x2:
                        act_result = self.unpack_packed_int4s(act_result)

                    torch.testing.assert_close(act_result, ref_result, rtol=0, atol=0)

    def test_volta_int4_interleave(self):
        tensor = torch.randint(-8, 7, [12, 128, 128], dtype=torch.int8)
        self.assertRaises(RuntimeError, self.quant_interleave, tensor, torch.quint4x2, 70)

    def test_volta_int8_interleave(self):
        tensor = torch.randint(-128, 127, [12, 128, 128], dtype=torch.int8)
        self.assertRaises(RuntimeError, self.quant_interleave, tensor, torch.int8, 70)

    def test_turing_int4_interleave(self):
        self.interleave_tester(torch.quint4x2, 75)

    def test_turing_int8_interleave(self):
        self.interleave_tester(torch.int8, 75)

    def test_ampere_80_int4_interleave(self):
        self.interleave_tester(torch.quint4x2, 80)

    def test_ampere_80_int8_interleave(self):
        self.interleave_tester(torch.int8, 80)
    
    def test_ampere_86_int4_interleave(self):
        self.interleave_tester(torch.quint4x2, 86)

    def test_ampere_86_int8_interleave(self):
        self.interleave_tester(torch.int8, 86)

    def test_add_bias_interleave_int4(self):
        # packed repr for -7 to 8
        packed_int4s = torch.tensor([[-104, -70, -36, -2, 16, 50, 84, 118]], dtype=torch.int8)
        actual_processed_int4s = self.add_bias_and_interleave_int4s(packed_int4s)
        # Packed repr for preprocessed cuda input (computed by hand)
        expected_processed_int4 = torch.tensor([[32, 100, 49, 117, -88, -20, -71, -3]], dtype=torch.int8)
        torch.testing.assert_close(actual_processed_int4s, expected_processed_int4, rtol=0, atol=0)

    def test_add_bias_interleave_int8(self):
        int8s = torch.tensor([[-104, -70, -36, 127, 16, 50, 84, 118]], dtype=torch.int8)
        actual_processed_int8s = self.add_bias_and_interleave_int8s(int8s)
        # Packed repr for preprocessed cuda input (computed by hand)
        tmp = torch.tensor([[-104, -36, -70, 127, 16, 84, 50, 118]], dtype=torch.int32) + 128
        expected_processed_int8 = tmp.to(torch.int8)
        torch.testing.assert_close(actual_processed_int8s, expected_processed_int8, rtol=0, atol=0)

    def transpose_test_helper(self, quant_type):
        assert quant_type == torch.int8 or quant_type == torch.quint4x2
        experts  = [1, 4, 8]
        m_shapes = [128, 256, 1024]
        n_shapes = [128, 4096]

        lower_bound = -128 if quant_type == torch.int8 else -8
        upper_bound = 127 if quant_type == torch.int8 else 7

        for expert in experts:
            for m_shape in m_shapes:
                for n_shape in n_shapes:
                    tensor = torch.randint(lower_bound, upper_bound, [expert, m_shape, n_shape], dtype=torch.int8)

                    # We want to move the data, but not change the shape. The actual impl just changes to col major.
                    ref_result = tensor.permute([0, 2, 1]).reshape([expert, m_shape, n_shape])

                    if quant_type == torch.quint4x2:
                        tensor = self.pack_int4s(tensor)

                    act_result = self.subbyte_transpose(tensor, quant_type)

                    if quant_type == torch.quint4x2:
                        act_result = self.unpack_packed_int4s(act_result)

                    torch.testing.assert_close(act_result, ref_result, rtol=0, atol=0)

    def test_transpose_int4(self):
      self.transpose_test_helper(torch.quint4x2)

    def test_transpose_int8(self):
      self.transpose_test_helper(torch.int8)

    def test_unpack(self):
        packed_int4s = torch.tensor([[-104, -70, -36, -2, 16, 50, 84, 118]], dtype=torch.int8)

        unpacked_int4s_as_int8 = self.unpack_packed_int4s(packed_int4s)
        expected = torch.arange(-8, 8, dtype=torch.int8).reshape([1, 16])
        torch.testing.assert_close(unpacked_int4s_as_int8, expected, rtol=0, atol=0)
        
    def test_pack(self):
        unpacked_i4s = torch.arange(-8, 8, dtype=torch.int8).reshape([1, 16])
        packed_i4s = self.pack_int4s(unpacked_i4s)
        expected = torch.tensor([[-104, -70, -36, -2, 16, 50, 84, 118]], dtype=torch.int8)
        torch.testing.assert_close(packed_i4s, expected, rtol=0, atol=0)
    
    def test_pack_unpack_identity(self):
        initial_vals = torch.randint(-8, 7, [128, 128], dtype=torch.int8)
        expected = self.unpack_packed_int4s(self.pack_int4s(initial_vals))
        torch.testing.assert_close(initial_vals, expected, rtol=0, atol=0)

if __name__ == '__main__':
    unittest.main()
