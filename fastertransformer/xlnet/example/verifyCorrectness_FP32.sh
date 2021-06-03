#! /bin/bash
############SET PERAMETERS######################
# GPU-related Setting 
gemm_file=../gemm/gemm.fp32.1080ti ### Different from FP16 version
gpu_id=0

# Input Data and Model 
data_dir=../data/STS-B
data_name=sts-b
spiece_model_file=../data/xlnet_cased_L-12_H-768_A-12/spiece.model
ckpt_file=../data/xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt
json_file=../data/xlnet_cased_L-12_H-768_A-12/xlnet_config.json
ckpt_file_16=../data/xlnet_model_fp16.ckpt


# Meta perameters
batch_size=8
seq_len=128
use_float16=0 ### Different from FP32 version
index=0
epislon=0.00

# Output
output_data_dir=../data
if [[ ! -e $output_data_dir ]]; then
    mkdir $output_data_dir
fi


# Input
data_file=${output_data_dir}/data.npz
model_file=${output_data_dir}/model.npz
output_file=${output_data_dir}/output.npz

############RUN COMMAND######################

# Convert input data format to npz
cm="python ../python/convertInput.py -t $data_name -b $batch_size -d $data_dir  -l $seq_len -s $spiece_model_file -o $data_file -u 0"
echo COMMAND: $cm
$cm

# Convert model weight format to npz
cm="python ../python/convertModel.py -i $ckpt_file -o $model_file"
echo COMMAND: $cm
$cm

# Run tensorflow to generate verification results in npz
cm="python ../python/runData.py -i $data_file -o $output_file -j $json_file -m $ckpt_file -b $batch_size -l $seq_len -f $use_float16 -n $index"
echo COMMAND: $cm
$cm

# Run FasterXLNet to verify its correctness
run_mode=3  ## FP16_TIME_TEST=0, FP16_CORRECTNESS_TEST=1,FP32_TIME_TEST=2,FP32_CORRECTNESS_TEST=3
cm="../runTest -m $run_mode -b $batch_size -s $seq_len -q $epislone -g $gpu_id -e $gemm_file -j $json_file -i $data_file -p $model_file -r $output_file " 
echo COMMAND: $cm
$cm
