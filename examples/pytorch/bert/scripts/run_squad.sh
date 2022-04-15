#! /bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

HEAD_NUM=16
HEAD_SIZE=64
BATCH_SIZE=8
SEQ_LEN=384
SPARSE=false
PAD=false

MAIN_PATH=$PWD

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --mtype)
    MODEL_TYPE="$2"; shift 2;;
  --dtype)
    DATA_TYPE="$2"; shift 2;;
  --path)
    MODEL_PATH="$2"; shift 2;;
  --head_num)
    HEAD_NUM="$2"; shift 2;;
  --head_size)
    HEAD_SIZE="$2"; shift 2;;
  --bs)
    BATCH_SIZE="$2"; shift 2;;
  --seqlen)
    SEQ_LEN="$2"; shift 2;;
  --sparse)
    SPARSE="$2"; shift 2;;
  --remove_padding)
    PAD="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

if [ "$MODEL_TYPE" != "ori" ] && [ "$MODEL_TYPE" != "ths" ] && [ "$MODEL_TYPE" != "thsext" ]; then
    echo "wrong model type, need be one of [ori, ths, thsext]"
    exit 1
fi
if [ "$DATA_TYPE" != "fp32" ] && [ "$DATA_TYPE" != "fp16" ] && [ "$DATA_TYPE" != "int8_1" ] && [ "$DATA_TYPE" != "int8_2" ] && [ "$DATA_TYPE" != "int8_3" ]; then
    echo "wrong data type, need be one of [fp32, fp16, int8_1, int8_2, int8_3]"
    exit 1
fi

if [ "$DATA_TYPE" == "fp32" ] || [ "$DATA_TYPE" == "fp16" ]; then
    if [ "$DATA_TYPE" == "fp32" ]; then
        FP16_MODE=0
    else
        FP16_MODE=1
    fi
    INT8_MODE=0
    if [ "$MODEL_PATH" == "" ]; then
        MODEL_PATH=$MAIN_PATH/pytorch/bert_squad/models/bert-large-uncased-whole-word-masking-finetuned-squad
        mkdir -p $MODEL_PATH
        cd $MAIN_PATH/pytorch/bert_squad/models/bert-large-uncased-whole-word-masking-finetuned-squad
        if [ ! -f "config.json" ]; then
            wget -c https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json
            mv bert-large-uncased-whole-word-masking-finetuned-squad-config.json config.json
        fi
        if [ ! -f "pytorch_model.bin" ]; then
            wget -c https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin
            mv bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin pytorch_model.bin
        fi
        if [ ! -f "vocab.txt" ]; then
            wget -c https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt
            mv bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt vocab.txt
        fi
        cd $MAIN_PATH
        HEAD_NUM=16
        HEAD_SIZE=64
    fi
else
    FP16_MODE=1
    if [ "$DATA_TYPE" == "int8_1" ]; then
        INT8_MODE=1
    elif [ "$DATA_TYPE" == "int8_2" ]; then
        INT8_MODE=2
    elif [ "$DATA_TYPE" == "int8_3" ]; then
        INT8_MODE=3
    else
        echo "wrong data type"; exit;
    fi
    if [ "$MODEL_PATH" == "" ]; then echo "--path not provided for int8 mode"; exit; fi
fi


mkdir -p $MAIN_PATH/pytorch/bert_squad/squad_data
mkdir -p $MAIN_PATH/pytorch/bert_squad/output
cd $MAIN_PATH/pytorch/bert_squad/squad_data
wget -c https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

if [ "$MODEL_TYPE" == "thsext" ]; then
    $MAIN_PATH/bin/bert_gemm ${BATCH_SIZE} ${SEQ_LEN} ${HEAD_NUM} ${HEAD_SIZE} ${FP16_MODE} ${INT8_MODE}
fi

SPCMD=""
if [ "$SPARSE" = "true" ] ; then
   SPCMD="--sparse"
elif [ "$SPARSE" = "false" ] ; then
   SPCMD=""
else
   echo "Unknown <sparse> argument"
   exit 1
fi

PADCMD=""
if [ "$PAD" = "true" ] ; then
   PADCMD="--remove_padding"
elif [ "$PAD" = "false" ] ; then
   PADCMD=""
else
   echo "Unknown <remove_padding> argument"
   exit 1
fi

CMD="python $MAIN_PATH/../examples/pytorch/bert/run_squad.py"
CMD+=" --model_name_or_path $MODEL_PATH"
CMD+=" --do_eval"
CMD+=" --do_lower_case"
CMD+=" --predict_file $MAIN_PATH/pytorch/bert_squad/squad_data/dev-v1.1.json"
CMD+=" --output_dir $MAIN_PATH/pytorch/bert_squad/output/"
CMD+=" --cache_dir $MAIN_PATH/pytorch/bert_squad/models/"
CMD+=" --max_seq_length ${SEQ_LEN}"
CMD+=" --per_gpu_eval_batch_size ${BATCH_SIZE}"
CMD+=" --model_type $MODEL_TYPE"
CMD+=" --data_type $DATA_TYPE"
CMD+=" --int8_mode $INT8_MODE"
CMD+=" $SPCMD"
CMD+=" $PADCMD"

cd $MAIN_PATH
$CMD
