# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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

# apt-get update
# apt-get install bc

pip install transformers==2.5.1

export NVIDIA_TF32_OVERRIDE=0

for precision in fp16 fp32;
do

if [ "$precision" = "fp16" ]; then
    echo "Using fp16."
    precision_num=1
    precision_larger="FP16"
else
    echo "Using fp32"
    precision_num=0
    precision_larger="FP32"
fi

logdir="bert-base-log-${precision}"
if [ ! -f ${logdir} ] ; then
    mkdir ${logdir} -p
fi
all_log="${logdir}/all-log.log"
echo -e "| Batch_size | Seq_len | Precision | TorchScript <br/> Latency (ms) | FT <br/> Latency (ms) | EFF-FT <br/> Latency (ms) | FT <br/> Speedup | EFF-FT <br/> Speedup | " > $all_log
echo -e "|:----------:|:-------:|:---------:|:------------------------------:|:---------------------:|:-------------------------:|:----------------:|:--------------------:| " >> $all_log

cat /proc/cpuinfo > ${logdir}/cpuinfo.txt
nvidia-smi > ${logdir}/gpuinfo.txt

for batch_size in 1 8 32 ;
do
for seq_len in 32 128 384 ;
do
if [ -f "gemm_config.in" ] ; then
        rm gemm_config.in
    fi
    ../build/bin/bert_gemm ${batch_size} ${seq_len} 12 64 ${precision_num} 0

    tmp_log_ths=${logdir}/batchsize-${batch_size}-seq-${seq_len}-${precision}-ths-log.log
    if [ "$precision" = "fp16" ]; then
        python ../examples/pytorch/bert/bert_example.py ${batch_size} 12 ${seq_len} 12 64 --fp16 --time 2>&1 | tee $tmp_log_ths
    else
        python ../examples/pytorch/bert/bert_example.py ${batch_size} 12 ${seq_len} 12 64 --time 2>&1 | tee $tmp_log_ths
    fi
    ths_time=`tail -n 3 ${tmp_log_ths} | head -n 1 | awk '{print $5}'`
    ft_time=`tail -n 2 ${tmp_log_ths} | head -n 1 | awk '{print $5}'`
    eff_ft_time=`tail -n 1 ${tmp_log_ths} | head -n 1 | awk '{print $5}'`

    ft_speedup=$(echo "scale=2; $ths_time / $ft_time" | bc)
    eff_ft_speedup=$(echo "scale=2; $ths_time / $eff_ft_time" | bc)
    echo ' ' | awk -v batch_size=$batch_size -v seq_len=$seq_len -v ft_time=$ft_time -v ths_time=$ths_time \
                        -v eff_ft_time=$eff_ft_time -v ft_speedup=$ft_speedup -v eff_ft_speedup=$eff_ft_speedup \
                        -v precision_larger=$precision_larger \
                        '{print "| " batch_size " | " seq_len " | " precision_larger " | " ths_time " | " \
                        ft_time " | " eff_ft_time " | " ft_speedup " | " eff_ft_speedup " | "  }' >> $all_log
done
done
done