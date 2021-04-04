TensorFlow op unit tests
===

# Encoder

1. FP32/FP16 check

```bash
python tensorflow/unit_test/encoder_unit_test.py # bert encoder test
python tensorflow/unit_test/open_encoder_unit_test.py # opennmt encoder test
```

2. FP32/FP16 on squad

Need to prepare the squad model. Details are in `docs/encoder_guide.md`. After preparing the model, please run

```bash
python squad_unit_test.py
```

3. int8 

For squad_int8_unit_test.py, we can run like this:

```bash
python tensorflow/unit_test/squad_int8_unit_test.py --output_dir ${output_dir} --model_file ${init_checkpoint} --int8_mode ${int8_mode}
python tensorflow/unit_test/squad_int8_unit_test.py --output_dir squad_int8_unit_test_1_384 --model_file squad_model/v4.0_prerelease_ckpt/QAT_mode_2/model.ckpt-10949 --int8_mode 2
```

Note that the ckpt is put int GOS server.

# Decoder/Decoding

1. fp32/fp16 of decoder/decoding

Need to download ckpt by `sample/tensorflow/utils/translation/download_model_data.sh` and convert the ckpt to fp16 by 

```bash
python tensorflow/tensorflow_bert/ckpt_type_convert.py --init_checkpoint=translation/ckpt/model.ckpt-500000 --fp16_checkpoint=translation/ckpt/fp16_model.ckpt-500000
```

Then run unit test by

```bash
python tensorflow/unit_test/decoding_unit_test.py
```