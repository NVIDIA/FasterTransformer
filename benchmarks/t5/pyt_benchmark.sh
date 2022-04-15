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

for model_size in "t5-base";
do
if [ "$model_size" = "t5-small" ]; then
    encoder_head_num=8
    encoder_size_per_head=64
    encoder_d_model=512
    encoder_num_layer=6
    encoder_inter_size=2048

    decoder_head_num=8
    decoder_size_per_head=64
    decoder_d_model=512
    decoder_num_layer=6
    decoder_inter_size=2048
    decoder_vocab_size=32128
elif [ "$model_size" = "t5-base" ]; then
    encoder_head_num=12
    encoder_size_per_head=64
    encoder_d_model=768
    encoder_num_layer=12
    encoder_inter_size=3072

    decoder_head_num=12
    decoder_size_per_head=64
    decoder_d_model=768
    decoder_num_layer=12
    decoder_inter_size=3072
    decoder_vocab_size=32128
elif [ "$model_size" = "t5-3b" ]; then
    encoder_head_num=32
    encoder_size_per_head=128
    encoder_d_model=1024
    encoder_num_layer=24
    encoder_inter_size=16384

    decoder_head_num=32
    decoder_size_per_head=128
    decoder_d_model=1024
    decoder_num_layer=24
    decoder_inter_size=16384
    decoder_vocab_size=32128
else
    echo "[ERROR] no model_size $model_size"
fi

for precision in fp32;
do

if [ "$precision" = "fp16" ]; then
    echo "Using fp16."
    precision_num=1
    precision_large="FP16"
else
    echo "Using fp32"
    precision_num=0
    precision_large="FP32"
fi

for decode_type in "beamsearch" "sampling";
do

if [ "$decode_type" = "beamsearch" ]; then
    decode_values=(4 32)
elif [ "$decode_type" = "sampling" ]; then
    decode_values=(4 0.5)
fi

logdir="${model_size}-${decode_type}-${precision}-log"
if [ ! -f ${logdir} ] ; then
    mkdir ${logdir} -p
fi

all_log="${logdir}/all-log.log"
echo -e "| Batch Size | ${decode_type} | Precision | Huggingface <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoding <br/> Speedup |" > $all_log
echo -e "|:----------:|:--------------:|:---------:|:----------------------------------------:|:----------------------------------------:|:-------------------------:|" >> $all_log

cat /proc/cpuinfo > ${logdir}/cpuinfo.txt
nvidia-smi > ${logdir}/gpuinfo.txt

for batch_size in 1 8 32 128;
do
for decode_value in ${decode_values[@]};
do
    if [ "$decode_type" = "beamsearch" ]; then
        beam_width=$decode_value
        topk=0
        topp=0.0
        test_time="01"
    elif [ "$decode_type" = "sampling" ]; then
        beam_width=1
        if [[ $decode_value == +([[:digit:]]) ]]; then
            topk=$decode_value
            topp=0.0
        else
            topk=0
            topp=$decode_value
        fi
        test_time="23"
    fi

    if [ -f "gemm_config.in" ] ; then
        rm gemm_config.in
    fi

    tmp_log_th=${logdir}/batchsize-${batch_size}-beamwidth-${beam_width}-seq-128-${precision}-${decode_type}-${decode_value}-th-log.log
    ./bin/t5_gemm ${batch_size} ${beam_width} 128 \
                  ${encoder_d_model} ${encoder_head_num} ${encoder_size_per_head} ${encoder_inter_size} \
                  ${decoder_d_model} ${decoder_head_num} ${decoder_size_per_head} ${decoder_inter_size} \
                  ${decoder_vocab_size} ${precision_num} > ${logdir}/batchsize-${batch_size}-beamwidth-${beam_width}-seq-128-${precision}-th-log.gemm.log

    python ${FT_REPO_PATH}/examples/pytorch/t5/translate_example.py \
            --batch_size ${batch_size} \
            --beam_width ${beam_width} \
            --max_seq_len 128 \
            --data_type ${precision} \
            --beam_search_diversity_rate 0.0 \
            --model ${model_size} \
            --sampling_topk ${topk} \
            --sampling_topp ${topp} \
            --test_time ${test_time} 2>&1 | tee ${tmp_log_th}
    ft_decoding_throughput=`tail -n 1 ${tmp_log_th} | awk '{print $16}'`
    th_throughput=`tail -n 2 ${tmp_log_th} | head -n 1 | awk '{print $16}'`
    ft_decoding_speedup=$(echo "scale=2; $ft_decoding_throughput / $th_throughput " | bc)
    
    echo "" | awk -v th_throughput=$th_throughput \
                        -v ft_decoding_throughput=$ft_decoding_throughput \
                        -v ft_decoding_speedup=$ft_decoding_speedup -v batch_size=$batch_size -v decode_value="$decode_value" \
                        -v precision_large=$precision_large \
                        '{printf "| %3d | %4s | %s | %5d | %5d | %5.2f |\n", batch_size, decode_value,
                        precision_large, th_throughput, ft_decoding_throughput, 
                        ft_decoding_speedup }' >> $all_log

done # decode_value
done # batch_size
done # decode_type
done # for precision
done # for model_size