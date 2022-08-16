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

# $1: TP size
# $2: PP size

export NVIDIA_TF32_OVERRIDE=0
tensor_para_size=$1
pipeline_para_size=$2
total_gpu_count=$(echo "scale=2; ${tensor_para_size} * ${pipeline_para_size} " | bc)

vocab_size=51200

logdir="gpt-TP${tensor_para_size}-PP${pipeline_para_size}-log"
if [ ! -f ${logdir} ]; then
    mkdir ${logdir} -p
fi

all_log="${logdir}/all-log.log"

echo -e "| model size | Batch Size | Input length | Output length | Decode value | Precision | FT latency (ms) |" > $all_log
echo -e "|:----------:|:----------:|:------------:|:-------------:|:------------:|:---------:|:---------------:|" >> $all_log

cat /proc/cpuinfo > ${logdir}/cpuinfo.txt
nvidia-smi > ${logdir}/gpuinfo.txt

for model_size in "345m" "5b";
do
    if [ "$model_size" = "345m" ]; then
        head_num=16
        size_per_head=64
        inter_size=$(echo "scale=2; $head_num * ${size_per_head} * 4 " | bc)
        num_layer=24
    elif [ "$model_size" = "5b" ]; then
        head_num=32
        size_per_head=128
        inter_size=$(echo "scale=2; $head_num * ${size_per_head} * 4 " | bc)
        num_layer=24
    fi

for decode_type in "beamsearch" "sampling";
do

    if [ "$decode_type" = "beamsearch" ]; then
        decode_values=(4)
    elif [ "$decode_type" = "sampling" ]; then
        decode_values=(4 0.5)
    fi

for request_batch_size in 1 4 16;
do
for input_length in 60;
do
for request_output_len in 80;
do
for decode_value in ${decode_values[@]};
do

if [ "$decode_type" = "beamsearch" ]; then
    beam_width=$decode_value
    topk=0
    topp=0.0
elif [ "$decode_type" = "sampling" ]; then
    beam_width=1
    if [[ $decode_value == +([[:digit:]]) ]]; then
        topk=$decode_value
        topp=0.0
    else
        topk=0
        topp=$decode_value
    fi
fi

tmp_log=${logdir}/batchsize-${request_batch_size}-decode_value-${decode_value}-${input_length}-${request_output_len}-${decode_type}-${decode_value}.log

python ../examples/pytorch/gpt/utils/generate_start_ids.py --max_batch_size ${request_batch_size} --max_input_length ${input_length}
./bin/gpt_gemm ${request_batch_size} ${beam_width} ${input_length} ${head_num} ${size_per_head} ${inter_size} ${vocab_size} 1 ${tensor_para_size}
python ../examples/pytorch/gpt/utils/generate_gpt_config.py \
                                        --max_seq_len 1024 \
                                        --beam_width ${beam_width} \
                                        --head_num ${head_num} \
                                        --size_per_head ${size_per_head} \
                                        --inter_size ${inter_size} \
                                        --num_layer ${num_layer} \
                                        -v 51200 \
                                        -d fp16 \
                                        -topk ${topk} \
                                        -topp ${topp} \
                                        --tensor_para_size ${tensor_para_size} \
                                        --pipeline_para_size ${pipeline_para_size} \
                                        -request_batch_size ${request_batch_size} \
                                        --request_output_len ${request_output_len}
mpirun -n ${total_gpu_count} --allow-run-as-root ./bin/multi_gpu_gpt_example .tmp.config.ini 2>&1 | tee ${tmp_log}
ft_latency=`tail -n 1 ${tmp_log} | head -n 1 | awk '{print $17}'`
echo "" | awk -v ft_latency=$ft_latency \
            -v batch_size=$request_batch_size \
            -v input_length=${input_length} -v request_output_len="$request_output_len" \
            -v model_size=${model_size} -v decode_value="$decode_value" -v decode_type="$decode_type" \
            '{printf "| %5s | %3d | %4d | %4d | %10s %5s | FP16 | %7.2f |\n", model_size, batch_size, input_length, request_output_len,
              decode_type, decode_value, ft_latency}' >> $all_log

rm .tmp.config.ini

done # decode_values
done # request_output_len
done # input_length
done # batch_size
done # decode_type
done # model_size