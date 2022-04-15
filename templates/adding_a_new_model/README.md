Contribute rule
===

FasterTransformer welcomes the community contributions. This documents describe how to add a new model, adding a new feature or optimize some kernels.

# How to add a new model

If a contributor has a transformer-based, and FasterTransformer still does not support, he/she can follow this guide to add the model in FasterTransformer. Here, we use Longformer as an example.

1. Check the model architecture with supporting model. For example, some components of Longformer and BERT are same (like FFN), then he/she can reuse these components directly.
2. Create the `longformer` folder in `src/fastertransformer/models/`.
3. Add CUDA codes to implement the different component. For example, the attention layers of Longformer and BERT are different. The attention layer of longformer should be put in `src/fastertransformer/layers`. The file name can be `LongformerAttentionLayer`.
   1. Note that if the model architecture are similar but different, don't modify the current model to fit the new model. For example, the differences between `Encoder.cc` and `Bert.cc` are the positions of layer normalization. We should reuse the attention layer, feed forward network layer and layer normalization kernel to create a new class `Encoder`, but not modify the `Bert` class to fit the Encoder.
4. Combine and organize all components of Longformer, and add the codes of full workflow in `src/fastertransformer/models/longformer`. The file name can be `Longformer`.
5. Add an example code to show how to use and verify the correctness. A simple example like `tensorflow/bert/bert_example.py` is ok. An task example like `tensorflow/bert/run_squad_wrap.py` is better. The example code can be cpp, TensorFlow or PyTorch (should put in `examples/cpp/longformer`, `examples/tensorflow/longformer` and `examples.pytorch/longformer` respectively). The requirement is that other users can use this example code to check the correctness.
6. Add an guide to explain how to use your codes, and show the benchmark in docs.
7. Submit a pull request and start the review.

# How to add a new feature?

# How to optimize some kernels?

Assume we have a new layer normalization kernel, which provides better performance than current layer normalization kernel, called `inovkeLayerNorm`.

1. Adding into existing file if there is sutiable file (like `src/fastertransformer/kernels/layernorm_kernels.cu`). Otherwise, create a new file in `src/fastertransformer/kernels/`. The function can be `invokeLayerNormV2` (simplest method to distinguish with current kernel) or `invokeLayerNormWithoutBlockReduction`, where `BlockReduction` is a method to accelerate the kernel and we can distinguish it from current one.
2. Provide a benchmark on some cases. For example

  * BERT performance on A100 and TensorFlow

  | Batch_size | Seq_len | Precision | FT old layernorm <br/> Latency (ms) | FT new layernorm <br/> Latency (ms) | Speedup |
  |:----------:|:-------:|:---------:|:-----------------------------------:|:-----------------------------------:|:-------:|
  |   1 |  32 | FP16 |  2.57  |  1.87  | 1.30 |
  |   1 | 128 | FP16 |  5.37  |  4.70  | 2.10 |
  |   1 | 384 | FP16 |  7.39  |  6.61  | 0.81 |
  |   8 |  32 | FP16 |  5.26  |  4.59  | 1.13 |
  |   8 | 128 | FP16 | 13.29  | 12.54  | 1.89 |
  |   8 | 384 | FP16 | 38.07  | 36.66  | 1.71 |
  |  32 |  32 | FP16 | 13.78  | 13.24  | 1.79 |
  |  32 | 128 | FP16 | 45.90  | 45.02  | 1.86 |
  |  32 | 384 | FP16 | 150.26 | 143.41 | 1.78 |

  Contributor only needs to shows the performance on some cases. We will review and test on other framework/GPUs if the modification makes sense.

3. Submit a pull request and start the review.

# Coding style
- Follow the .clang-format as much as possible.
- Naming
  1. Filenames
     * Uppercase Camel case for the file which contains only one class. For example, `BertLayer.cc` only contains the `BertLayer` class.  
     * Other files are lowercase with `_`, like `cuda_utils.h`.
  2. function
     * lower Camel-Case, like `invokeLayerNorm`.
  3. variables
     *  lowercase with `_`, like `batch_size`.