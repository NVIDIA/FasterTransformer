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

if [ $FT_REPO_PATH ];then
	echo "FT_REPO_PATH = $FT_REPO_PATH"
else
	echo "FT_REPO_PATH IS NOT EXISTS"
    exit
fi

export NVIDIA_TF32_OVERRIDE=0

for precision in fp16 fp32;
do

if [ "$precision" = "fp16" ]; then
    echo "Using fp16."
    precision_num=1
else
    echo "Using fp32"
    precision_num=0
fi

logdir="bert-base-log-${precision}"
if [ ! -f ${logdir} ] ; then
    mkdir ${logdir} -p
fi

all_log="${logdir}/all-log.log"
echo -e "| Batch_size | Seq_len | Precision | TF <br/> Latency (ms) | FT <br/> Latency (ms) | EFF-FT <br/> Latency (ms) | FT <br/> Speedup | EFF-FT <br/> Speedup | " > $all_log
echo -e "|:----------:|:-------:|:---------:|:---------------------:|:---------------------:|:-------------------------:|:----------------:|:--------------------:| " >> $all_log

cat /proc/cpuinfo > ${logdir}/cpuinfo.txt
nvidia-smi > ${logdir}/gpuinfo.txt

for batch_size in 1 8 32 ;
do
for seq_len in 32 128 384 ;
do
    if [ -f "gemm_config.in" ] ; then
        rm gemm_config.in
    fi
    ${FT_REPO_PATH}/build/bin/bert_gemm ${batch_size} ${seq_len} 12 64 ${precision_num} 0

    tmp_log_tf=${logdir}/batchsize-${batch_size}-seq-${seq_len}-${precision}-tf-log.log
    python ${FT_REPO_PATH}/examples/tensorflow/bert/bert_example.py -batch ${batch_size} -s ${seq_len} -time 1 -d ${precision} 2>&1 | tee $tmp_log_tf
    
    tf_time=`tail -n 3 ${tmp_log_tf} | head -n 1 | awk '{print $11}'`
    ft_time=`tail -n 2 ${tmp_log_tf} | head -n 1 | awk '{print $11}'`
    eff_ft_time=`tail -n 1 ${tmp_log_tf} | head -n 1 | awk '{print $11}'`
    precision_type=`tail -n 1 ${tmp_log_tf} | head -n 1 | awk '{print $7}'`

    ft_speedup=$(echo "scale=2; $tf_time / $ft_time" | bc)
    eff_ft_speedup=$(echo "scale=2; $tf_time / $eff_ft_time" | bc)

    echo "" | awk -v tf_time=$tf_time -v ft_time=$ft_time \
                        -v eff_ft_time=$eff_ft_time -v ft_speedup=$ft_speedup -v eff_ft_speedup=$eff_ft_speedup \
                        -v batch_size=$batch_size -v seq_len=$seq_len -v precision_type=$precision_type \
                        '{printf "| %3d | %3d | %s | %5.2f | %5.2f | %5.2f | %4.2f | %4.2f | \n", batch_size, seq_len,
                        precision_type, tf_time, ft_time, eff_ft_time, ft_speedup, eff_ft_speedup }' >> $all_log
done
done
cat ${all_log}
done
