# PyTorch BERT Quantization/Sparsity Example

Based on `https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT`

Original README [here](README_orig.md)

This directory contains examples for BERT PTQ/QAT and sparsity related training.

## Setup

Please follow the original README to do some inital setup.

setup steps:
```bash
export ROOT_DIR=</path/to/this_repo_root_dir>
export DATA_DIR=</path/to/data_dir>
export MODEL_DIR=</path/to/model/checkpoint>

git clone https://github.com/NVIDIA/TensorRT.git
cd TensorRT
git checkout release/7.2
cd tools/pytorch-quantization
pip install .
```

download SQuAD data:
```bash
cd $DATA_DIR
bash $ROOT_DIR/data/squad/squad_download.sh
```

download pre-trained checkpoint, config file, and vocab file (bert-base-uncased):
```bash
cd $MODEL_DIR
mkdir bert-base-uncased
wget https://s3.amazonaws.com/models.huggingface.co/bert/google/bert_uncased_L-12_H-768_A-12/pytorch_model.bin -O bert-base-uncased/pytorch_model.bin
wget https://s3.amazonaws.com/models.huggingface.co/bert/google/bert_uncased_L-12_H-768_A-12/config.json -O bert-base-uncased/config.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/google/bert_uncased_L-12_H-768_A-12/vocab.txt -O bert-base-uncased/vocab.txt

cd $ROOT_DIR
```

## Sparsity
Add `--sparse` flag can do sparse training for Ampere structured sparsity easily.

Usually, only down-stream sparse finetuning is not enough for accuracy recovery, so pre-training is needed.
Please follow `README_orig.md` for pre-training.
One recommended recipe for sparse pre-training:
- do dense pre-training stage 1 (128 seqlen)
- do dense pre-training stage 2 (512 seqlen)
- do sparse pre-training stage 2 (same hyperparameter with dense stage 2)

When do sparse pre-training, add `--sparse` flag and use `--dense_checkpoint` for initialization. Training more steps for sparse stage usually gains better accuracy.

Note: sparse fine-tuning and QAT can be combined.


## Post Training Quantization

Firstly, finetune for a float dense model:

```bash
python -m torch.distributed.launch --nproc_per_node=8 run_squad.py \
  --init_checkpoint=$MODEL_DIR/bert-base-uncased/pytorch_model.bin \
  --do_train \
  --train_file=$DATA_DIR/v1.1/train-v1.1.json \
  --train_batch_size=4 \
  --learning_rate=3e-5 \
  --num_train_epochs=2 \
  --do_predict \
  --predict_file=$DATA_DIR/v1.1/dev-v1.1.json \
  --eval_script=$DATA_DIR/v1.1/evaluate-v1.1.py \
  --do_eval \
  --do_lower_case \
  --bert_model=bert-base-uncased \
  --max_seq_length=384 \
  --doc_stride=128 \
  --vocab_file=$MODEL_DIR/bert-base-uncased/vocab.txt \
  --config_file=$MODEL_DIR/bert-base-uncased/config.json \
  --json-summary=$MODEL_DIR/dllogger.json \
  --output_dir=$MODEL_DIR/bert-base-uncased-finetuned \
  --max_steps=-1 \
  --fp16 \
  --quant-disable
```

The results would be like:

```bash
{"exact_match": 82.63, "f1": 89.53}
```

Then do PTQ, `quant_mode` is unified with int8_mode in FasterTransformer, can be one of `ft1` or `ft2` or `ft3`.

```bash
python run_squad.py \
  --init_checkpoint=$MODEL_DIR/bert-base-uncased-finetuned/pytorch_model.bin \
  --do_calib \
  --train_file=$DATA_DIR/v1.1/train-v1.1.json \
  --train_batch_size=16 \
  --num-calib-batch=16 \
  --do_predict \
  --predict_file=$DATA_DIR/v1.1/dev-v1.1.json \
  --eval_script=$DATA_DIR/v1.1/evaluate-v1.1.py \
  --do_eval \
  --do_lower_case \
  --bert_model=bert-base-uncased \
  --max_seq_length=384 \
  --doc_stride=128 \
  --vocab_file=$MODEL_DIR/bert-base-uncased/vocab.txt \
  --config_file=$MODEL_DIR/bert-base-uncased/config.json \
  --json-summary=$MODEL_DIR/dllogger.json \
  --output_dir=$MODEL_DIR/bert-base-uncased-PTQ-mode-2 \
  --max_steps=-1 \
  --fp16 \
  --calibrator percentile \
  --percentile 99.999 \
  --quant_mode ft2
```

The results would be like:

```bash
{"exact_match": 81.92, "f1": 89.09}     # for mode 1
{"exact_match": 80.36, "f1": 88.09}     # for mode 2
```


## Quantization Aware Fine-tuning

If PTQ does not yield an acceptable result you can finetune with quantization to recover accuracy.
We recommend to calibrate the pretrained model and finetune to avoid overfitting:

`quant_mode` is unified with int8_mode in FasterTransformer, can be one of `ft1` or `ft2` or `ft3`.

```bash
python run_squad.py \
  --init_checkpoint=$MODEL_DIR/bert-base-uncased/pytorch_model.bin \
  --do_calib \
  --train_file=$DATA_DIR/v1.1/train-v1.1.json \
  --train_batch_size=16 \
  --num-calib-batch=16 \
  --do_lower_case \
  --bert_model=bert-base-uncased \
  --max_seq_length=384 \
  --doc_stride=128 \
  --vocab_file=$MODEL_DIR/bert-base-uncased/vocab.txt \
  --config_file=$MODEL_DIR/bert-base-uncased/config.json \
  --json-summary=$MODEL_DIR/dllogger.json \
  --output_dir=$MODEL_DIR/bert-base-uncased-calib-mode-2 \
  --max_steps=-1 \
  --fp16 \
  --calibrator percentile \
  --percentile 99.99 \
  --quant_mode ft2

python -m torch.distributed.launch --nproc_per_node=8 run_squad.py \
  --init_checkpoint=$MODEL_DIR/bert-base-uncased-calib-mode-2/pytorch_model.bin \
  --do_train \
  --train_file=$DATA_DIR/v1.1/train-v1.1.json \
  --train_batch_size=4 \
  --learning_rate=3e-5 \
  --num_train_epochs=2 \
  --do_predict \
  --predict_file=$DATA_DIR/v1.1/dev-v1.1.json \
  --eval_script=$DATA_DIR/v1.1/evaluate-v1.1.py \
  --do_eval \
  --do_lower_case \
  --bert_model=bert-base-uncased \
  --max_seq_length=384 \
  --doc_stride=128 \
  --vocab_file=$MODEL_DIR/bert-base-uncased/vocab.txt \
  --config_file=$MODEL_DIR/bert-base-uncased/config.json \
  --json-summary=$MODEL_DIR/dllogger.json \
  --output_dir=$MODEL_DIR/bert-base-uncased-QAT-mode-2 \
  --max_steps=-1 \
  --fp16 \
  --quant_mode ft2
```

The results would be like:

```bash
{"exact_match": 82.17, "f1": 89.37}     # for mode 1
{"exact_match": 82.02, "f1": 89.30}     # for mode 2
```

The results of quantization may differ if different seeds are provided.


## Quantization Aware Fine-tuning with Knowledge-distillation

Knowledge-distillation can get better results, we usually starts from a PTQ checkpoint.

```bash
python run_squad.py \
  --init_checkpoint=$MODEL_DIR/bert-base-uncased-finetuned/pytorch_model.bin \
  --do_calib \
  --train_file=$DATA_DIR/v1.1/train-v1.1.json \
  --train_batch_size=16 \
  --num-calib-batch=16 \
  --do_lower_case \
  --bert_model=bert-base-uncased \
  --max_seq_length=384 \
  --doc_stride=128 \
  --vocab_file=$MODEL_DIR/bert-base-uncased/vocab.txt \
  --config_file=$MODEL_DIR/bert-base-uncased/config.json \
  --json-summary=$MODEL_DIR/dllogger.json \
  --output_dir=$MODEL_DIR/bert-base-uncased-PTQ-mode-2-for-KD \
  --max_steps=-1 \
  --fp16 \
  --calibrator percentile \
  --percentile 99.99 \
  --quant_mode ft2

python -m torch.distributed.launch --nproc_per_node=8 run_squad.py \
  --init_checkpoint=$MODEL_DIR/bert-base-uncased-PTQ-mode-2-for-KD/pytorch_model.bin \
  --do_train \
  --train_file=$DATA_DIR/v1.1/train-v1.1.json \
  --train_batch_size=4 \
  --learning_rate=3e-5 \
  --num_train_epochs=10 \
  --do_predict \
  --predict_file=$DATA_DIR/v1.1/dev-v1.1.json \
  --eval_script=$DATA_DIR/v1.1/evaluate-v1.1.py \
  --do_eval \
  --do_lower_case \
  --bert_model=bert-base-uncased \
  --max_seq_length=384 \
  --doc_stride=128 \
  --vocab_file=$MODEL_DIR/bert-base-uncased/vocab.txt \
  --config_file=$MODEL_DIR/bert-base-uncased/config.json \
  --json-summary=$MODEL_DIR/dllogger.json \
  --output_dir=$MODEL_DIR/bert-base-uncased-QAT-mode-2 \
  --max_steps=-1 \
  --fp16 \
  --quant_mode ft2 \
  --distillation \
  --teacher=$MODEL_DIR/bert-base-uncased-finetuned/pytorch_model.bin
```

The results would be like:

```bash
{"exact_match": 83.67, "f1": 90.37}
```
