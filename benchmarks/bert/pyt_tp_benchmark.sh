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

# apt-get update
# apt-get install bc
set -x
export NVIDIA_TF32_OVERRIDE=0

MODEL_LAYER=32
HEAD_NUM=32
SIZE_PER_HEAD=128
HIDDEN_SIZE=$(echo "${HEAD_NUM} * ${SIZE_PER_HEAD}" | bc)
INTER_SIZE=$(echo "${HIDDEN_SIZE} * 4" | bc)
for precision in fp16;
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

logdir="bert-6B-log-${precision}-triton"
if [ ! -f ${logdir} ] ; then
    mkdir ${logdir} -p
fi
all_log="${logdir}/all-log.log"
echo -e "| Batch_size | Seq_len | Precision | TP1, PP1 <br/> Latency (ms) | TP2, PP1 <br/> Latency (ms) | TP4, PP1 <br/> Latency (ms) | TP1, PP2 <br/> Latency (ms) | TP1, PP4 <br/> Latency (ms) | " > $all_log
echo -e "|:----------:|:-------:|:---------:|:---------------------------:|:---------------------------:|:---------------------------:|:---------------------------:|:---------------------------:| " >> $all_log

cat /proc/cpuinfo > ${logdir}/cpuinfo.txt
nvidia-smi > ${logdir}/gpuinfo.txt

echo "[bert]
        model_name = bert
        position_embedding_type = absolute
        hidden_size = ${HIDDEN_SIZE}
        num_layer = ${MODEL_LAYER}
        head_num = ${HEAD_NUM}
        size_per_head = ${SIZE_PER_HEAD}
        activation_type = gelu
        inter_size = ${INTER_SIZE}
        max_position_embeddings = 512
        layer_norm_eps = 1e-12
        weight_data_type = fp32
        tensor_para_size = 1" > config.ini

for batch_size in 1 4 32 128 ;
do
for seq_len in 32 128 384 1024 ;
do
    # tp 1, pp 1
    python ../examples/pytorch/bert/utils/update_bert_config.py \
        --model-dir ./ \
        --config-ini-path config.ini \
        --pipeline-para-size 1 \
        --tensor-para-size 1 \
        --data-type fp16 \
        --request-batch-size ${batch_size} \
        --request-seq-len ${seq_len}
    tmp_log=${logdir}/batchsize-${batch_size}-seq-${seq_len}-${precision}-tp-1-pp-1.log
    CUDA_VISIBLE_DEVICES=4 ./bin/bert_triton_example config.ini 2>&1 | tee ${tmp_log}
    ft_tp1_pp1_time=`tail -n 1 ${tmp_log} | head -n 1 | awk '{print $7}'`
    sleep 5

    # tp 2, pp 1
    python ../examples/pytorch/bert/utils/update_bert_config.py \
        --model-dir ./ \
        --config-ini-path config.ini \
        --pipeline-para-size 1 \
        --tensor-para-size 2 \
        --data-type fp16 \
        --request-batch-size ${batch_size} \
        --request-seq-len ${seq_len}
    tmp_log=${logdir}/batchsize-${batch_size}-seq-${seq_len}-${precision}-tp-2-pp-1.log
    CUDA_VISIBLE_DEVICES=4,5 ./bin/bert_triton_example config.ini 2>&1 | tee ${tmp_log}
    ft_tp2_pp1_time=`tail -n 1 ${tmp_log} | head -n 1 | awk '{print $7}'`
    sleep 5

    # tp 4, pp 1
    python ../examples/pytorch/bert/utils/update_bert_config.py \
        --model-dir ./ \
        --config-ini-path config.ini \
        --pipeline-para-size 1 \
        --tensor-para-size 4 \
        --data-type fp16 \
        --request-batch-size ${batch_size} \
        --request-seq-len ${seq_len}
    tmp_log=${logdir}/batchsize-${batch_size}-seq-${seq_len}-${precision}-tp-4-pp-1.log
    CUDA_VISIBLE_DEVICES=4,5,6,7 ./bin/bert_triton_example config.ini 2>&1 | tee ${tmp_log}
    ft_tp4_pp1_time=`tail -n 1 ${tmp_log} | head -n 1 | awk '{print $7}'`
    sleep 5

    # tp 1, pp 2
    python ../examples/pytorch/bert/utils/update_bert_config.py \
        --model-dir ./ \
        --config-ini-path config.ini \
        --pipeline-para-size 2 \
        --tensor-para-size 1 \
        --data-type fp16 \
        --request-batch-size ${batch_size} \
        --request-seq-len ${seq_len}
    tmp_log=${logdir}/batchsize-${batch_size}-seq-${seq_len}-${precision}-tp-1-pp-2.log
    CUDA_VISIBLE_DEVICES=4,5 ./bin/bert_triton_example config.ini 2>&1 | tee ${tmp_log}
    ft_tp1_pp2_time=`tail -n 1 ${tmp_log} | head -n 1 | awk '{print $7}'`
    sleep 5

    echo "| ${batch_size} | ${seq_len} | fp16 | ${ft_tp1_pp1_time} | ${ft_tp2_pp1_time} | ${ft_tp4_pp1_time} | ${ft_tp1_pp2_time} | " >> ${all_log}
done
done
done