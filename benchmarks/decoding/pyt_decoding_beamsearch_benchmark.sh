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
pip install opennmt-py==1.1.1

export NVIDIA_TF32_OVERRIDE=0

for precision in fp16 fp32;
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

logdir="decoding-log-${precision}"
if [ ! -f ${logdir} ] ; then
    mkdir ${logdir} -p
fi

all_log="${logdir}/all-log.log"
echo -e "| Batch Size | Beam Width | Precision | PyTorch <br/> Throughput (token/sec) | FT Decoder <br/> Throughput (token/sec) | FT Decoding <br/> Throughput (token/sec) | FT Decoder <br/> Speedup | FT Decoding <br/> Speedup | " > $all_log
echo -e "|:----------:|:----------:|:---------:|:------------------------------------:|:---------------------------------------:|:----------------------------------------:|:------------------------:|:-------------------------:| " >> $all_log

cat /proc/cpuinfo > ${logdir}/cpuinfo.txt
nvidia-smi > ${logdir}/gpuinfo.txt

for batch_size in 1 8 128 ;
do
for beam_size in 1 4 32 ;
do
    if [ -f "gemm_config.in" ] ; then
        rm gemm_config.in
    fi

    ./bin/bert_gemm ${batch_size} 128 8 64 ${precision_num} 0
    ./bin/decoding_gemm ${batch_size} ${beam_size} 8 64 2048 31538 128 512 ${precision_num} 1
    sleep 60

    py_log=${logdir}/beamsize-${beam_size}-batchsize-${batch_size}-seq-128-${precision}-ths-log.log
    ft_decoder_log=${logdir}/beamsize-${beam_size}-batchsize-${batch_size}-seq-128-${precision}-ft-decoder-log.log
    ft_decoding_log=${logdir}/beamsize-${beam_size}-batchsize-${batch_size}-seq-128-${precision}-ft-decoding-log.log

    python ../examples/pytorch/decoding/translate_example.py --batch_size ${batch_size} --beam_size ${beam_size} \
                                                             --max_seq_len 128 \
                                                             --model_type torch_decoding \
                                                             --data_type ${precision} \
                                                             --output_file output.txt 2>&1 | tee ${py_log}
    sleep 60
    python ../examples/pytorch/decoding/translate_example.py --batch_size ${batch_size} --beam_size ${beam_size} \
                                                             --max_seq_len 128 \
                                                             --model_type torch_decoding_with_decoder_ext \
                                                             --data_type ${precision} \
                                                             --output_file output.txt 2>&1 | tee ${ft_decoder_log}
    sleep 60
    python ../examples/pytorch/decoding/translate_example.py --batch_size ${batch_size} --beam_size ${beam_size} \
                                                             --max_seq_len 128 \
                                                             --model_type decoding_ext \
                                                             --data_type ${precision} \
                                                             --output_file output.txt 2>&1 | tee ${ft_decoding_log}
    sleep 60
    ft_decoding_throughput=`tail -n 1 ${ft_decoding_log} | awk '{print $4}'`
    ft_decoder_throughput=`tail -n 1 ${ft_decoder_log} | awk '{print $4}'`
    py_throughput=`tail -n 1 ${py_log} | awk '{print $4}'`
    ft_decoder_speedup=$(echo "scale=2; $ft_decoder_throughput / $py_throughput " | bc)
    ft_decoding_speedup=$(echo "scale=2; $ft_decoding_throughput / $py_throughput " | bc)

    echo "" | awk -v py_throughput=$py_throughput -v ft_decoder_throughput=$ft_decoder_throughput \
                        -v ft_decoding_throughput=$ft_decoding_throughput -v ft_decoder_speedup=$ft_decoder_speedup \
                        -v ft_decoding_speedup=$ft_decoding_speedup -v batch_size=$batch_size -v beam_size=$beam_size \
                        -v precision_large=$precision_large \
                        '{printf "| %3d | %3d | %s | %5d | %5d | %5d | %4.2f | %4.2f | \n", batch_size, beam_size,
                        precision_large, py_throughput, ft_decoder_throughput, ft_decoding_throughput,
                        ft_decoder_speedup, ft_decoding_speedup }' >> $all_log

done
done
done