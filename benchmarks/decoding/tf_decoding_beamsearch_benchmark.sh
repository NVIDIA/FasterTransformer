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
    precision_large="FP16"
    if [ ! -d "${FT_REPO_PATH}/translation/ckpt_fp16/" ] ; then
        echo "[ERROR] Cannot find the checkpoint at ${FT_REPO_PATH}/translation/ckpt_fp16/"
        exit
    fi
else
    echo "Using fp32"
    precision_num=0
    precision_large="FP32"
    if [ ! -d "${FT_REPO_PATH}/translation/ckpt/" ] ; then
        echo "[ERROR] Cannot find the checkpoint at ${FT_REPO_PATH}/translation/ckpt/"
        exit
    fi
fi

logdir="decoding-beamsearch-log-${precision}"
if [ ! -f ${logdir} ] ; then
    mkdir ${logdir} -p
fi

all_log="${logdir}/all-log.log"
echo -e "| Batch Size | Beam Width | Precision | TF <br/> Throughput (token/sec) | FT Decoder <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoder <br/> Speedup | FT Decoding <br/> Speedup | " > $all_log
echo -e "|:----------:|:----------:|:---------:|:-------------------------------:|:---------------------------------------:|:----------------------------------------:|:------------------------:|:-------------------------:| " >> $all_log

cat /proc/cpuinfo > ${logdir}/cpuinfo.txt
nvidia-smi > ${logdir}/gpuinfo.txt

for batch_size in 1 8 128 ;
do
for beam_width in 4 32 ;
do
    if [ -f "gemm_config.in" ] ; then
        rm gemm_config.in
    fi
    if [ -f "gemm_config.in" ] ; then
        rm gemm_config.in
    fi
    tmp_log_tf=${logdir}/batchsize-${batch_size}-beamwidth-${beam_width}-seq-128-${precision}-tf-log.log
    ./bin/bert_gemm ${batch_size} 128 8 64 ${precision_num} 0
    ./bin/decoding_gemm ${batch_size} ${beam_width} 8 64 2048 32001 128 512 ${precision_num} 1
    python ${FT_REPO_PATH}/examples/tensorflow/decoding/translate_example.py \
            --batch_size ${batch_size} \
            --beam_width ${beam_width} \
            --max_seq_len 128 \
            --data_type ${precision} \
            --beam_search_diversity_rate 0.0 \
            --sampling_topk 1 \
            --sampling_topp 0.00 \
            --test_time 012 2>&1 | tee ${tmp_log_tf}
    ft_decoding_throughput=`tail -n 1 ${tmp_log_tf} | awk '{print $16}'`
    ft_decoder_throughput=`tail -n 2 ${tmp_log_tf} | head -n 1 | awk '{print $16}'`
    tf_throughput=`tail -n 3 ${tmp_log_tf} | head -n 1 | awk '{print $16}'`
    ft_decoder_speedup=$(echo "scale=2; $ft_decoder_throughput / $tf_throughput " | bc)
    ft_decoding_speedup=$(echo "scale=2; $ft_decoding_throughput / $tf_throughput " | bc)
    
    echo "" | awk -v tf_throughput=$tf_throughput -v ft_decoder_throughput=$ft_decoder_throughput \
                        -v ft_decoding_throughput=$ft_decoding_throughput -v ft_decoder_speedup=$ft_decoder_speedup \
                        -v ft_decoding_speedup=$ft_decoding_speedup -v batch_size=$batch_size -v beam_width=$beam_width \
                        -v precision_large=$precision_large \
                        '{printf "| %3d | %3d | %s | %5d | %5d | %5d | %4.2f | %4.2f | \n", batch_size, beam_width,
                        precision_large, tf_throughput, ft_decoder_throughput, ft_decoding_throughput, 
                        ft_decoder_speedup, ft_decoding_speedup }' >> $all_log
done # for beam_width
done # for batch_size
done # for precision