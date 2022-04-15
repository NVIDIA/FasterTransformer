# TensorFlow BERT Quantization Example

Based on [link](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)

Original README: [link](README_orig.md)

This directory contains examples for BERT PTQ/QAT related training.

Hardware settings:
 * 4 x Tesla V100-SXM2-16GB (with mclk 877MHz, pclk 1530MHz)

## Setup

The docker `nvcr.io/nvidia/tensorflow:20.03-tf1-py3` is used for test (TensorFlow 1.15.2)

setup steps:
```
pip install ft-tensorflow-quantization/
export TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=0
```

## Download pretrained bert checkpoint and SQuAD dataset

Download pretrained bert checkpoint.

```bash
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip -O uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip -d squad_model
```

Download SQuAD dataset

```bash
mkdir squad_data
wget -P squad_data https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget -P squad_data https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

## Post Training Quantization

### Finetune a high precision model with:

```bash
mpirun -np 4 -H localhost:4 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib \
    python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/bert_model.ckpt \
    --output_dir=squad_model/finetuned_base \
    --do_train=True \
    --do_predict=True \
    --if_quant=False \
    --train_batch_size=8 \
    --learning_rate=1e-5 \
    --num_train_epochs=2.0 \
    --save_checkpoints_steps 1000 \
    --horovod

python ../../sample/tensorflow/tensorflow_bert/squad_evaluate_v1_1.py squad_data/dev-v1.1.json squad_model/finetuned_base/predictions.json
```

The results would be like:

```bash
{"exact_match": 82.44, "f1": 89.57}
```

### PTQ by calibrating:

`quant_mode` is unified with int8_mode in FasterTransformer, can be one of `ft1` or `ft2` or `ft3`.

```bash
python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/finetuned_base/model.ckpt-5474 \
    --output_dir=squad_model/PTQ_mode_2 \
    --do_train=False \
    --do_predict=True \
    --do_calib=True \
    --if_quant=True \
    --train_batch_size=16 \
    --calib_batch=16 \
    --calib_method=percentile \
    --percentile=99.999 \
    --quant_mode=ft2

python ../../sample/tensorflow/tensorflow_bert/squad_evaluate_v1_1.py squad_data/dev-v1.1.json squad_model/PTQ_mode_2/predictions.json
```

The results would be like:

```bash
{"exact_match": 81.67, "f1": 88.94}     # for mode 1
{"exact_match": 80.44, "f1": 88.30}     # for mode 2
```


## Quantization Aware Fine-tuning

If PTQ does not yield an acceptable result you can finetune with quantization to recover accuracy.
We recommend to calibrate the pretrained model and finetune to avoid overfitting:

`quant_mode` is unified with int8_mode in FasterTransformer, can be one of `ft1` or `ft2` or `ft3`.

```bash
python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/bert_model.ckpt \
    --output_dir=squad_model/QAT_calibrated_mode_2 \
    --do_train=False \
    --do_calib=True \
    --train_batch_size=16 \
    --calib_batch=16 \
    --calib_method=percentile \
    --percentile=99.99 \
    --quant_mode=ft2


mpirun -np 4 -H localhost:4 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib \
    python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/QAT_calibrated_mode_2/model.ckpt-calibrated \
    --output_dir=squad_model/QAT_mode_2 \
    --do_train=True \
    --do_predict=True \
    --if_quant=True \
    --train_batch_size=8 \
    --learning_rate=1e-5 \
    --num_train_epochs=2.0 \
    --save_checkpoints_steps 1000 \
    --quant_mode=ft2 \
    --horovod

python ../../sample/tensorflow/tensorflow_bert/squad_evaluate_v1_1.py squad_data/dev-v1.1.json squad_model/QAT_mode_2/predictions.json
```

The results would be like:

```bash
{"exact_match": 82.11, "f1": 89.39}     # for mode 1
{"exact_match": 81.74, "f1": 89.12}     # for mode 2
```


The results of quantization may differ if different seeds are provided.


## Quantization Aware Fine-tuning with Knowledge-distillation

Knowledge-distillation can get better results, we usually starts from a PTQ checkpoint.

```bash
python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/finetuned_base/model.ckpt-5474 \
    --output_dir=squad_model/PTQ_mode_2_for_KD \
    --do_train=False \
    --do_predict=False \
    --do_calib=True \
    --if_quant=True \
    --train_batch_size=16 \
    --calib_batch=16 \
    --calib_method=percentile \
    --percentile=99.99 \
    --quant_mode=ft2

mpirun -np 4 -H localhost:4 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib \
    python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/PTQ_mode_2_for_KD/model.ckpt-calibrated \
    --output_dir=squad_model/QAT_KD_mode_2 \
    --do_train=True \
    --do_predict=True \
    --if_quant=True \
    --train_batch_size=8 \
    --learning_rate=2e-5 \
    --num_train_epochs=10.0 \
    --save_checkpoints_steps 1000 \
    --quant_mode=ft2 \
    --horovod \
    --distillation=True \
    --teacher=squad_model/finetuned_base/model.ckpt-5474

python ../../sample/tensorflow/tensorflow_bert/squad_evaluate_v1_1.py squad_data/dev-v1.1.json squad_model/QAT_KD_mode_2/predictions.json
```

The results would be like:

```bash
{"exact_match": 84.06, "f1": 90.63}     # for mode 1
{"exact_match": 84.02, "f1": 90.56}     # for mode 2
```
