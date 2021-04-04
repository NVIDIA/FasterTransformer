# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

batch_size=8

for precision in fp16
do
  for seq_len in 1024 2048
  do
    logdir="gpt-log-${precision}-seq${seq_len}"
    mkdir -p ${logdir}

    for num_layer in 1 2 4 32 96
    do
      for gpu_num in 1 2 4 8
      do
        prefix=${logdir}/${gpu_num}gpu-${num_layer}layers-${precision}-bs${batch_size}
        python ../sample/pytorch/utils/generate_gpt_config.py --batch_size ${batch_size} \
                                                            --max_seq_len ${seq_len} \
                                                            --start_len 1 \
                                                            --num_layer ${num_layer} \
                                                            --head_number 96 \
                                                            --size_per_head 128 \
                                                            --vocab_size 50257 \
                                                            --data_type ${precision} \
                                                            --sampling_topk 0 \
                                                            --sampling_topp 0.9 \
                                                            --tensor_para_size ${gpu_num} \
                                                            --local_batch_size ${batch_size}
        
        nsys profile --stat true -w true -o$prefix -f true mpirun -n ${gpu_num} --allow-run-as-root ./bin/gpt_sample .tmp.config.ini 2>&1 | tee ${prefix}.log
        rm .tmp.config.ini

        nsys stats --force-overwrite -f csv -o . ${prefix}.qdrep
      done
    done
  done
done