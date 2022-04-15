# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

MODEL='base'
layer_num=12
head_num=12
head_size=64

while test $# -gt 0
do
    case "$1" in
        base) MODEL='base'
            ;;
        large) MODEL='large'
            ;;
        *) echo "Invalid argument $1...exiting"
            exit 0
            ;;
    esac
    shift
done

if [ "${MODEL}" = 'large' ]; then
    layer_num=24
    head_num=16
fi

logdir="bert-${MODEL}-log-int8-2-sp"
mkdir ${logdir}
all_log="${logdir}/all-log.log"
echo -e "| <batch_size, seq_len> | Dense (ms) | Sparse (ms) | Dense EFF (ms) | Sparse EFF (ms) | " > $all_log
echo -e "|:---------------------:|:------:|:------:|:------:|:------:| " >> $all_log

for batch_size in 1 8 32 ;
do
for seq_len in 32 128 384 ;
do
    ./bin/bert_gemm ${batch_size} ${seq_len} ${head_num} ${head_size} 1 2
    sleep 2s

    tmp_log_pt=${logdir}/batchsize-${batch_size}-seq-${seq_len}-log.log
    tmp_log_pt_sp=${logdir}/batchsize-${batch_size}-seq-${seq_len}-sp-log.log

    python ../examples/pytorch/bert/bert_example.py ${batch_size} ${layer_num} ${seq_len} ${head_num} ${head_size} --fp16 --int8_mode 2 --time 2>&1 | tee $tmp_log_pt
    sleep 5s
    python ../examples/pytorch/bert/bert_example.py ${batch_size} ${layer_num} ${seq_len} ${head_num} ${head_size} --fp16 --int8_mode 2 --sparse --time 2>&1 | tee $tmp_log_pt_sp
    sleep 5s

    ft_o_time=`tail -n 2 ${tmp_log_pt} | head -n 1 | awk '{print $5}'`
    sp_o_time=`tail -n 2 ${tmp_log_pt_sp} | head -n 1 | awk '{print $5}'`
    fteff_o_time=`tail -n 1 ${tmp_log_pt} | head -n 1 | awk '{print $5}'`
    speff_o_time=`tail -n 1 ${tmp_log_pt_sp} | head -n 1 | awk '{print $5}'`

    echo ' ' | awk -v batch_size=$batch_size -v seq_len=$seq_len \
                        -v ft_o_time=$ft_o_time -v sp_o_time=$sp_o_time \
                        -v fteff_o_time=$fteff_o_time -v speff_o_time=$speff_o_time \
                        '{print "| <" batch_size ", " seq_len "> | " ft_o_time " | " sp_o_time " | " fteff_o_time " | " speff_o_time " | "}' >> $all_log

done
done