# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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



#!/bin/bash
# GPU-related setting
gpu_id=0
gemm_file=../gemm/gemm.fp32.1080ti
#gemm_file=../gemm/gemm.fp16.1080ti

# Meta perameters
batch_size=8
seq_len=128
warm_up_ite=50
profile_ite=100

#Input
json_file=./xlnet_config.json

# Parameters for Fasterxlnet
run_mode=2 #FP32_TIME_TEST=2
#run_mode=0 #FP16_TIME_TEST=0


for seq_len in 32 #64 128
do
    for batch_size  in 1 8
    do
        # Run tensorflow xla to collect baseline
        cm="python3 ../python/runProfile.py -s $seq_len -b $batch_size -w $warm_up_ite -t $profile_ite -j $json_file"
        echo COMMAND: $cm
        TF_XLA_FLAGS="--tf_xla_auto_jit=2" CUDA_VISIBLE_DEVICES=$gpu_id $cm
        cm="../runTest -m $run_mode -g $gpu_id -e $gemm_file -s $seq_len -b $batch_size  -w $warm_up_ite -t $profile_ite -j $json_file"
        echo COMMAND: $cm
        $cm
    done
done


