#! /bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

helpFunction()
{
    echo "Usage: bash verifyCorrectness_FP32.sh -d data_dir -m model_dir -s npz_dir -e gemm_file -g gpu_id -f is_use_fp16"
    echo -e "\t -d The directory of input data. Default: ./data/STS-B "
    echo -e "\t -n The data name. Default: sts-b "
    echo -e "\t -m The directory of the xlnet models. Default: ./data/xlnet_cased_L-12_H-768_A-12"
    echo -e "\t -s The directory which stores the generated npz files. Default: ./data"
    echo -e "\t -e The gemm file for selecting cublas functions. Default: ./gemm.in"
    echo -e "\t -g Specify which GPU to use. Default: 0 "
    echo -e "\t -f Specify use float16 or not. 1 means run in float16 mode. Default: 0 "
    exit 1
}
############SET PERAMETERS######################
gpu_id=0
data_dir=./data/STS-B
data_name=sts-b
model_dir=./data/xlnet_cased_L-12_H-768_A-12
npz_dir=./data
use_float16=0 ### Different from FP32 version
gemm_file=./gemm.in


while getopts "d:n:m:s:e:g:f:h" opt
do
    case "$opt" in
        d ) data_dir="$OPTARG" ;;
        n ) data_name="$OPTARG" ;;
        m ) model_dir="$OPTARG" ;;
        s ) npz_dir="$OPTARG" ;;
        e ) gemm_file="$OPTARG" ;;
        g ) gpu_id="$OPTARG" ;;
        f ) use_float16="$OPTARG" ;;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
    esac
done

# Print helpFunction in case parameters are empty
if [ -z "$data_dir" ] || [ -z "$model_dir" ] || [ -z "$npz_dir" ]
then
    echo "Some or all of the parameters are empty";
    helpFunction
fi

echo ""
echo "Get Parameters: data_dir=${data_dir} data_name=${data_name} model_dir=${model_dir} npz_dir=${npz_dir} gemm_file=${gemm_file} gpu_id=${gpu_id} use_float16=${use_float16}"
echo ""
# Model
spiece_model_file=${model_dir}/spiece.model
ckpt_file=${model_dir}/xlnet_model.ckpt
json_file=${model_dir}/xlnet_config.json
ckpt_file_16=${model_dir}/xlnet_model_fp16.ckpt

echo "Model files: ${spiece_model_file} ${ckpt_file} ${json_file}"
echo ""

# Meta perameters
batch_size=8
num_layers=1
seq_len=128
head_number=12
size_per_head=64
num_token=32000
index=0
epislon=0.00

# Npz 
if [[ ! -e $npz_dir ]]; then
    mkdir $npz_dir
fi

data_file=${npz_dir}/data.npz
model_file=${npz_dir}/model.npz
output_file=${npz_dir}/output.npz

echo "Npz files: ${data_file} ${model_file} ${output_file}"
echo ""

############RUN COMMAND######################
# Convert input data format to npz
cm="python ./convertInput.py -t $data_name -b $batch_size -d $data_dir  -l $seq_len -s $spiece_model_file -o $data_file -u 0"
echo ""
echo COMMAND: $cm
$cm

# Convert model weight format to npz
cm="python ./convertModel.py -i $ckpt_file -o $model_file"
echo ""
echo COMMAND: $cm
$cm

# Run tensorflow to generate verification results in npz
if [ ${use_float16} = 0 ]
then
    echo use_float16=${use_float16}
    cm="python ./runData.py -i $data_file -o $output_file -j $json_file -m $ckpt_file -b $batch_size -l $seq_len -f $use_float16 -n $index"
    echo ""
    echo COMMAND: $cm
    $cm
else
    echo use_float16=${use_float16}
    cm="python ../../tensorflow/ckpt_type_convert.py --init_checkpoint=$ckpt_file --fp16_checkpoint=$ckpt_file_16"
    echo COMMAND: $cm
    $cm
    cm="python ./runData.py -i $data_file -o $output_file -j $json_file -m $ckpt_file_16 -b $batch_size -l $seq_len -f $use_float16 -n $index"
    echo COMMAND: $cm
    $cm
fi

# Run FasterXLNet to verify its correctness
cm="../../../build/bin/xlnet_correctness_example ${batch_size} ${num_layers} ${seq_len} ${head_number} ${size_per_head} ${num_token} ${data_file} ${model_file} ${output_file} ${use_float16} "
echo ""
echo COMMAND: $cm
$cm
