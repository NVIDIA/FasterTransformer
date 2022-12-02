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
import torch.nn.functional as F
import numpy as np
import unittest

def random_cuda_tensor(shape, dtype, mean=0, std=1):
    return torch.empty(shape, dtype=dtype, device="cuda").normal_(mean, std)

def basic_moe_fc(activations, expert_for_row, weights, scales, biases):
  if weights.dtype == torch.int8:
      weights = torch.multiply(weights, scales.unsqueeze(1))
      weights = weights.to(activations.dtype)
  elif weights.dtype != torch.bfloat16 and weights.dtype != torch.float16 and weights.dtype != torch.float32:
      raise ValueError("Invalid data type for weights")
      

  res = torch.zeros(size=[activations.shape[0], weights.shape[-1]], dtype=activations.dtype, device='cuda')
  for row in range(activations.shape[0]):
      row_expert = expert_for_row[row]
      torch.matmul(activations[row], weights[row_expert], out=res[row : row + 1, :])
      res[row] += biases[row_expert]

  return res

def apply_act(inp, act_str):
  if act_str == "identity":
    return inp
  elif act_str == "silu":
    return torch.nn.SiLU()(inp)
  elif act_str == "relu":
    return torch.nn.ReLU()(inp)
  elif act_str == "gelu":
    return torch.nn.GELU(approximate="tanh")(inp)
  else:
    assert False, "Unsupported activation"

class TestMoeSoftmax(unittest.TestCase):
    def setUp(self) -> None:
        torch.classes.load_library("lib/libmoe_unit_ops.so")
        self.gating_softmax = torch.ops.moe_unit_ops.gating_softmax
        self.hidden = 1024
        torch.manual_seed(5258732)

    def gating_softmax_test_helper(self, dtype, rtol=1e-05, atol=1e-08):
        batch_sizes = [128, 111, 75, 64, 44, 32, 23, 16, 5, 1]
        seq_lens = [511, 250, 127, 64, 5, 1]
        ks = range(4, 0, -1)
        experts = [1024, 700, 512, 256, 200, 128, 64, 32, 18, 16, 10, 8, 5, 4, 2]

        # Reference impl
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for num_experts in experts:
                  for k in ks:
                    if k > num_experts:
                      continue
                    
                    # Some indices will mismatch due to FP arithmetic differences. We will fail if more than 1/500th do not match
                    allowed_idx_mismatches = max(5, batch_size * seq_len * k // 500)
                    inp = random_cuda_tensor([batch_size, seq_len, num_experts], dtype)
                    
                    # Run ref in FP32 to keep softmax result in fp32 when doing top-k
                    gates = F.softmax(inp, dim=-1)
                    ref_vals, ref_idxs = torch.topk(gates, k, dim=-1)
                    ref_vals = ref_vals.to(dtype)
                    ref_rows = torch.arange(0, k*batch_size*seq_len, device="cuda")

                    # optimized impl
                    act_vals, act_idxs, act_rows = self.gating_softmax(inp.to(dtype), k)

                    val_err_msg = "Val failed on batch_size={}, seq_len={}, experts={}, k={}".format(batch_size, seq_len, num_experts, k)
                    idx_err_msg = "Idx failed on batch_size={}, seq_len={}, experts={}, k={}".format(batch_size, seq_len, num_experts, k)
                    row_err_msg = "Row failed on batch_size={}, seq_len={}, experts={}, k={}".format(batch_size, seq_len, num_experts, k)

                    torch.testing.assert_close(act_vals, ref_vals, rtol=rtol, atol=atol, msg=val_err_msg, check_dtype=False)
                    row_differences = torch.where(act_idxs != ref_idxs)[:-1]
                    sorted_ref_idxs_where_mismatched = torch.sort(ref_idxs[row_differences])[0]
                    sorted_act_idxs_where_mismatched = torch.sort(act_idxs[row_differences])[0]

                    values_equal = (ref_vals[row_differences] == act_vals[row_differences])
                    sorted_idxs_equal = (sorted_ref_idxs_where_mismatched == sorted_act_idxs_where_mismatched)

                    # These are not real mismatches because the output values are the same, but indices were reported in a different order.
                    false_mismatches = torch.all(torch.logical_and(values_equal, sorted_idxs_equal), dim=-1)
                    num_false_mismatches = torch.sum(false_mismatches)

                    mismatches = torch.count_nonzero(act_idxs != ref_idxs) - num_false_mismatches
                    np.testing.assert_array_less(mismatches.cpu().numpy(), allowed_idx_mismatches + 1, err_msg=idx_err_msg)
                    torch.testing.assert_close(act_rows.permute([2, 0,1]).reshape(-1), ref_rows, rtol=0, atol=0, msg=row_err_msg, check_dtype=False)

    def test_fp32_gating_softmax(self):
        self.gating_softmax_test_helper(torch.float32, rtol=1e-05, atol=1e-08)
    
    def test_fp16_gating_softmax(self):
        # Allow tolerance for fp16 since our implementation keeps fp32 after the softmax while torch does not.
        self.gating_softmax_test_helper(torch.float16, rtol=1e-03, atol=1e-05)

    def test_bf16_gating_softmax(self):
        self.gating_softmax_test_helper(torch.bfloat16, rtol=1e-03, atol=0.005)

class TestGroupedGemmBias(unittest.TestCase):
    def setUp(self) -> None:
        torch.classes.load_library("lib/libth_weight_only_quant_ops.so")
        torch.classes.load_library("lib/libmoe_unit_ops.so")
        self.grouped_gemm_bias = torch.ops.moe_unit_ops.grouped_gemm_bias
        self.unpack_packed_int4s = torch.ops.fastertransformer.unpack_int4_packed_tensor_to_int8
        self.pack_int4s = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
        self.preprocess_weights_for_mixed_gemm = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm

        self.symmetric_quantizer = torch.ops.fastertransformer._symmetric_quantize_last_axis_of_batched_matrix
        self.add_bias_and_interleave_int4s = torch.ops.fastertransformer._add_bias_and_interleave_int4s
        self.add_bias_and_interleave_int8s = torch.ops.fastertransformer._add_bias_and_interleave_int8s

        torch.manual_seed(734876213)
        self.np_rng = np.random.default_rng(seed=82587632419)

    def ref_moe_fc(self, activations, expert_for_row, weights, scales, biases, activation_str):
      gemm_out = basic_moe_fc(activations, expert_for_row, weights, scales, biases)
      return apply_act(gemm_out, activation_str)

    def custom_moe_fc(self, torch_activations, torch_experts_for_rows, torch_weights, torch_scales, torch_biases, activation_str):
        num_experts = torch_weights.size(0)
        row_permutation = torch.argsort(torch_experts_for_rows)
        torch_rows_per_expert = torch.zeros(size=[num_experts], dtype=torch.int32)
        for expert in torch_experts_for_rows:
            torch_rows_per_expert[expert] += 1
        
        permutated_activations = torch_activations[row_permutation]
        torch_rows_per_expert = torch_rows_per_expert.to('cuda')
        res = self.grouped_gemm_bias(permutated_activations, torch_weights, torch_scales, torch_biases, torch_rows_per_expert, activation_str)
        res[row_permutation] = res[torch.arange(res.shape[0])]
        return res

    def setup_experts_for_row(self, num_rows, num_experts, active_experts):
        # We use numpy here as torch does not have a way to express random choice as elegantly to my knowledge.
        experts_arr = np.arange(num_experts)
        selected_experts = self.np_rng.choice(experts_arr, size=[active_experts], replace=False)

        # Ensure assign at least 1 row to an active expert
        expert_for_rows = self.np_rng.choice(selected_experts, size=[num_rows - active_experts], replace=True)
        return torch.tensor(np.concatenate([expert_for_rows, selected_experts]))

    def dequantize_test_helper(self, weight_type, quant_type):
      assert quant_type == torch.int8 or quant_type == torch.quint4x2

      lower_bound = -128 if quant_type == torch.int8 else -8
      upper_bound = 127 if quant_type == torch.int8 else 7

      m, n, k = 128, 128, 128
      weights = torch.randint(lower_bound, upper_bound, [k, n], dtype=torch.int8, device="cpu")

      packed_weight = self.pack_int4s(weights) if quant_type == torch.quint4x2 else weights
      cuda_weights = self.preprocess_weights_for_mixed_gemm(packed_weight, quant_type).to("cuda")
      weights = weights.to("cuda")

      act = torch.eye(m, dtype=weight_type, device="cuda")
      bias = torch.zeros([n], dtype=weight_type, device='cuda')
      torch_weight_scales = torch.ones_like(bias)
      experts_for_rows = self.setup_experts_for_row(m, 1, 1)

      actual = self.custom_moe_fc(act, experts_for_rows, cuda_weights, torch_weight_scales, bias, "identity")
      torch.testing.assert_close(actual, weights, atol=0.0, rtol=0.0, check_dtype=False)

    def test_fp16_int8_dequantize(self):
      self.dequantize_test_helper(torch.float16, torch.int8)

    def test_bf16_int8_dequantize(self):
      self.dequantize_test_helper(torch.bfloat16, torch.int8)

    def test_fp16_int4_dequantize(self):
      self.dequantize_test_helper(torch.float16, torch.quint4x2)

    def test_bf16_int4_dequantize(self):
      self.dequantize_test_helper(torch.bfloat16, torch.quint4x2)

    def moe_fc1_test_helper(self, compute_type, weight_dtype, rtol, atol, activation_str):
        torch.cuda.empty_cache() # Empty the cache here so a bad ordering does not cause OOM.
        rows = list(range(40, 0, -1))
        experts = [32, 128]
        active_experts = list(range(32, 0, -1))
        hidden_sizes = torch.tensor([1024])
        inter_sizes = 4 * hidden_sizes

        quantize = weight_dtype == torch.int8 or weight_dtype == torch.quint4x2

        for num_experts in experts:
            for hidden_size in hidden_sizes:
                for inter_size in inter_sizes:
                    torch_weights = random_cuda_tensor((num_experts, hidden_size, inter_size), dtype=compute_type, mean=0, std=0.002)
                    torch_biases = torch.randn(size=(num_experts, inter_size), device="cuda", dtype=compute_type)
                    torch_weight_scales = torch.ones_like(torch_biases, dtype=torch_weights.dtype, device="cuda")
                    cpu_weights = torch_weights.cpu()

                    if quantize:
                        ref_torch_weights, act_torch_weights, torch_weight_scales = self.symmetric_quantizer(cpu_weights, weight_dtype)
                        ref_torch_weights = self.unpack_packed_int4s(ref_torch_weights) if weight_dtype == torch.quint4x2 else ref_torch_weights
                        ref_torch_weights = ref_torch_weights.to("cuda")
                        act_torch_weights = act_torch_weights.to("cuda")
                        torch_weight_scales = torch_weight_scales.to("cuda")


                    for num_rows in rows:
                        torch_activations = torch.randn(size=(num_rows, hidden_size), dtype=compute_type, device="cuda")
                        # torch_activations = torch.ones_like(torch_activations)
                        # act_torch_weights = torch.ones_like(act_torch_weights) + 128
                        # ref_torch_weights = torch.ones_like(ref_torch_weights)
                        # torch_biases = torch.zeros_like(torch_biases)
                        # torch_weight_scales = torch.ones_like(torch_weight_scales)

                        for num_active_experts in active_experts:
                            clipped_active_experts = min(num_rows, min(num_active_experts, num_experts))
                            torch_experts_for_rows = self.setup_experts_for_row(num_rows, num_experts, clipped_active_experts)

                            ref_wt = ref_torch_weights if quantize else torch_weights
                            reference = self.ref_moe_fc(torch_activations, torch_experts_for_rows, ref_wt, torch_weight_scales, torch_biases, activation_str)

                            act_wt = act_torch_weights if quantize else torch_weights
                            act_scales = torch_weight_scales if quantize else torch.empty(0, device="cuda", dtype=compute_type)
                            actual = self.custom_moe_fc(torch_activations, torch_experts_for_rows, act_wt, act_scales, torch_biases, activation_str)

                            msg = "FC1 Failed on rows={}, experts={}, active_experts={}, hidden_size={}, inter_size={}" \
                                    .format(num_rows, num_experts, clipped_active_experts, hidden_size, inter_size)

                            torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol, msg=msg, check_dtype=False)
                            
    def test_fp32_moe_fc(self):
        self.moe_fc1_test_helper(torch.float32, torch.float32, rtol=1e-04, atol=1e-04, activation_str="identity")

    def test_fp32_moe_fc_gelu(self):
        self.moe_fc1_test_helper(torch.float32, torch.float32, rtol=1e-04, atol=1e-04, activation_str="gelu")

    def test_fp32_moe_fc_relu(self):
        self.moe_fc1_test_helper(torch.float32, torch.float32, rtol=1e-04, atol=1e-04, activation_str="relu")
  
    def test_fp32_moe_fc_silu(self):
        self.moe_fc1_test_helper(torch.float32, torch.float32, rtol=1e-04, atol=1e-04, activation_str="silu")

    def test_fp16_moe_fc(self):
        self.moe_fc1_test_helper(torch.float16, torch.float16, rtol=0.01, atol=0.005, activation_str="identity")

    def test_fp16_moe_fc_gelu(self):
        self.moe_fc1_test_helper(torch.float16, torch.float16, rtol=0.01, atol=0.005, activation_str="gelu")

    def test_fp16_moe_fc_relu(self):
        self.moe_fc1_test_helper(torch.float16, torch.float16, rtol=0.01, atol=0.005, activation_str="relu")

    def test_fp16_moe_fc_silu(self):
        self.moe_fc1_test_helper(torch.float16, torch.float16, rtol=0.01, atol=0.005, activation_str="silu")

    def test_int8_fp16_moe_fc(self):
        self.moe_fc1_test_helper(torch.float16, torch.int8, rtol=0.01, atol=0.005, activation_str="identity")

    def test_int4_fp16_moe_fc_gelu(self):
        self.moe_fc1_test_helper(torch.float16, torch.quint4x2, rtol=0.01, atol=0.005, activation_str="gelu")

    def test_bf16_moe_fc_relu(self):
        self.moe_fc1_test_helper(torch.bfloat16, torch.bfloat16, rtol=0.01, atol=0.005, activation_str="relu")
    
    def test_int8_bf16_moe_fc_silu(self):
        self.moe_fc1_test_helper(torch.bfloat16, torch.int8, rtol=0.01, atol=0.005, activation_str="silu")

    def test_int4_bf16_moe_fc(self):
        self.moe_fc1_test_helper(torch.bfloat16, torch.quint4x2, rtol=0.01, atol=0.005, activation_str="identity")

class TestMoe(unittest.TestCase):

  def setUp(self) -> None:
    torch.classes.load_library("lib/libth_weight_only_quant_ops.so")
    torch.classes.load_library("lib/libmoe_unit_ops.so")

    self.run_moe_fc = torch.ops.moe_unit_ops.run_moe_fc
    self.preprocess_weights_for_mixed_gemm = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm
    self.unpack_packed_int4s = torch.ops.fastertransformer.unpack_int4_packed_tensor_to_int8

    self.symmetric_quantizer = torch.ops.fastertransformer._symmetric_quantize_last_axis_of_batched_matrix

    torch.manual_seed(734876213)
  
  def generate_inputs(self, num_rows, active_rows, hidden_size, num_experts, dtype, quant_type):
    inputs = dict()

    inputs["input_activations"] = random_cuda_tensor([num_rows, hidden_size], dtype, mean=0, std=0.002)
    inputs["gating_output"] = random_cuda_tensor([num_rows, num_experts], dtype)

    inputs["skip_layer"] = random_cuda_tensor([num_rows, hidden_size], dtype)

    num_finished_sentences = num_rows - active_rows
    finished_sentences = torch.randint(0, num_rows, [num_finished_sentences], device="cuda")
    inputs["finished"] = torch.zeros([num_rows], dtype=torch.bool, device="cuda")
    inputs["finished"][finished_sentences] = True

    return inputs
  
  def generate_weights(self, hidden_size, inter_size, num_experts, dtype, quant_type):
    weights = dict()
    quantize = quant_type == torch.int8 or quant_type == torch.quint4x2

    weights["fc1_expert_weights_for_ref"] = random_cuda_tensor([num_experts, hidden_size, inter_size], dtype, mean=0, std=0.002)
    weights["fc1_expert_weights_for_ft"] = weights["fc1_expert_weights_for_ref"]
    weights["fc1_scales"] = torch.ones(size=[num_experts, inter_size], dtype=dtype, device="cuda")
    weights["fc1_expert_biases"] = random_cuda_tensor([num_experts, inter_size], dtype, mean=0, std=0.002)

    weights["fc2_expert_weights_for_ref"] = random_cuda_tensor([num_experts, inter_size, hidden_size], dtype, mean=0, std=0.002)
    weights["fc2_expert_weights_for_ft"] = weights["fc2_expert_weights_for_ref"]
    weights["fc2_scales"] = torch.ones(size=[num_experts, hidden_size], dtype=dtype, device="cuda")
    weights["fc2_expert_biases"] = random_cuda_tensor([num_experts, hidden_size], dtype, mean=0, std=0.002)

    if quantize:
        ref_torch_weights_fc1, act_torch_weights_fc1, torch_weight_scales_fc1 = self.symmetric_quantizer(weights["fc1_expert_weights_for_ft"].cpu(), quant_type)
        ref_torch_weights_fc2, act_torch_weights_fc2, torch_weight_scales_fc2 = self.symmetric_quantizer(weights["fc2_expert_weights_for_ft"].cpu(), quant_type)

        if quant_type == torch.quint4x2:
          ref_torch_weights_fc1 = self.unpack_packed_int4s(ref_torch_weights_fc1)
          ref_torch_weights_fc2 = self.unpack_packed_int4s(ref_torch_weights_fc2)


        weights["fc1_expert_weights_for_ref"] = ref_torch_weights_fc1.to("cuda")
        weights["fc1_expert_weights_for_ft"] = act_torch_weights_fc1.to("cuda")
        weights["fc1_scales"] = torch_weight_scales_fc1.to("cuda")

        weights["fc2_expert_weights_for_ref"] = ref_torch_weights_fc2.to("cuda")
        weights["fc2_expert_weights_for_ft"] = act_torch_weights_fc2.to("cuda")
        weights["fc2_scales"] = torch_weight_scales_fc2.to("cuda")
  
    return weights
  
  def run_ft_moe(self, input_dict, active_rows, k, activation_str):
    moe_output = self.run_moe_fc(input_dict["input_activations"], input_dict["gating_output"], \
                    input_dict["fc1_expert_weights_for_ft"], input_dict["fc1_scales"], input_dict["fc1_expert_biases"], \
                    activation_str, \
                    input_dict["fc2_expert_weights_for_ft"], input_dict["fc2_scales"], input_dict["fc2_expert_biases"], \
                    input_dict["skip_layer"], input_dict["finished"], active_rows, k)
    
    return moe_output
  
  def run_ref_moe(self, input_dict, k, activation_str):
    gates = F.softmax(input_dict["gating_output"].to(torch.float32), dim=-1).to(input_dict["gating_output"].dtype)
    expert_scales, experts_for_row = torch.topk(gates, k, dim=-1)

    output = torch.zeros_like(input_dict["input_activations"])
    output += input_dict["skip_layer"]

    for k_idx in range(k):
      current_expert_scales = expert_scales[:, k_idx].unsqueeze(1)
      current_experts_for_row = experts_for_row[:, k_idx]

      moe_fc_1_result = basic_moe_fc(input_dict["input_activations"], current_experts_for_row, 
                                     input_dict["fc1_expert_weights_for_ref"], input_dict["fc1_scales"], input_dict["fc1_expert_biases"])
      moe_fc_1_result = apply_act(moe_fc_1_result, activation_str)

      moe_fc_2_result = basic_moe_fc(moe_fc_1_result, current_experts_for_row, 
                                     input_dict["fc2_expert_weights_for_ref"], input_dict["fc2_scales"], input_dict["fc2_expert_biases"])
      
      output = output + current_expert_scales * moe_fc_2_result
    
    return output

  def moe_test_helper(self, dtype, quant_type, rtol, atol, activation_str="gelu", experts_list=[32], hidden_sizes=[1024], inter_sizes=[4096]):
    torch.cuda.empty_cache() # Empty the cache here so a bad ordering does not cause OOM.
    rows = [40, 1000]
    ks = range(1, 9)

    for hidden_size in hidden_sizes:
      for inter_size in inter_sizes:
        for experts in experts_list:
          weights = self.generate_weights(hidden_size, inter_size, experts, dtype, quant_type)
          for row in rows:
            for active_rows in [1, row // 2, row]:
              for k in ks:
                if k > experts:
                  continue
                input_dict = self.generate_inputs(row, active_rows, hidden_size, experts, dtype, quant_type)
                input_dict.update(weights)            
                rows_to_check = torch.logical_not(input_dict["finished"])

                # Only take unfinished rows. We can write anything to the output of rows that already complete.
                act_output = self.run_ft_moe(input_dict, row, k, activation_str)[rows_to_check]
                ref_output = self.run_ref_moe(input_dict, k, activation_str)[rows_to_check]

                msg = "Moe Failed on rows={}, active_rows={}, experts={}, k={}, hidden_size={}, inter_size={}" \
                        .format(row, active_rows, experts, k, hidden_size, inter_size)
                torch.testing.assert_close(act_output, ref_output, rtol=rtol, atol=atol, msg=msg, check_dtype=False)
  
  def test_moe_fp32_relu(self):
    self.moe_test_helper(torch.float32, torch.float32, rtol=1e-3, atol=1e-6, \
                         activation_str="relu", \
                         experts_list=[64, 32, 16, 8, 4, 2], hidden_sizes=[2048, 1024], \
                         inter_sizes=[4096])

  def test_moe_fp16_gelu(self):
    self.moe_test_helper(torch.float16, torch.float16, rtol=1e-3, atol=0.005, \
                         activation_str="gelu", \
                         experts_list=[128, 30, 7, 5, 3], hidden_sizes=[2048, 1024], \
                         inter_sizes=[4096])

  # We limit the configs in the quantization code-path only because quantizing is quite slow
  # which makes testing very slow when using FP32/FP16 configs.
  def test_moe_fp16_int8_gelu(self):
    self.moe_test_helper(torch.float16, torch.int8, rtol=1e-3, atol=1e-3, \
                         activation_str="gelu", \
                         experts_list=[135], hidden_sizes=[2048], \
                         inter_sizes=[4096])

  def test_moe_fp16_int4_silu(self):
    self.moe_test_helper(torch.float16, torch.quint4x2, rtol=1e-3, atol=1e-3, \
                         activation_str="silu", \
                         experts_list=[196], hidden_sizes=[1024], \
                         inter_sizes=[8192])

  def test_moe_bf16_gelu(self):
    self.moe_test_helper(torch.bfloat16, torch.bfloat16, rtol=1e-2, atol=0.005, \
                         activation_str="gelu", \
                         experts_list=[64, 32], hidden_sizes=[1024], \
                         inter_sizes=[4096])

  # We limit the configs in the quantization code-path only because quantizing is quite slow
  # which makes testing very slow when using FP32/FP16 configs.
  def test_moe_bf16_int8_relu(self):
    self.moe_test_helper(torch.bfloat16, torch.int8, rtol=1e-2, atol=0.005, \
                         activation_str="relu", \
                         experts_list=[48], hidden_sizes=[1024], \
                         inter_sizes=[4096])

  def test_moe_bf16_int4_identity(self):
    self.moe_test_helper(torch.bfloat16, torch.quint4x2, rtol=1e-2, atol=0.005, \
                         activation_str="identity", \
                         experts_list=[256, 63], hidden_sizes=[1024], \
                         inter_sizes=[8192])

if __name__ == '__main__':
    unittest.main()