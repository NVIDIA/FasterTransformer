# T5Plugin in FasterTransformer
+ Original Faster Transformer: [link](https://github.com/NVIDIA/FasterTransformer)
+ This project aims to wapper the encoder and decoding parts of the faster transformer as TensoRT plugins respectively. The most of teh code file are from orignial Faster TRansformer project.
  
## Envionment
+ **nvcr.io/nvidia/pytorch:21.02-py3** (including CUDA 11.2.0, cudnn 8.1.0.77, 1.8.0a0+52ea372, TensorRTTensorRT 7.2.2.3+cuda11.1.0.024)
+ Now the code in the repository are compatible for TensorRT7, maybe need several edition before using in TensorRT8.

---
## T5EncoderPlugin in T5Plugin
| Classification  | Tensor/Parameter Shape                 | Data Type      | Description                        |
| :-------------: | :------------------------------------- | :------------- | :--------------------------------- |
|  input tensor   |                                        |                |                                    |
|       [0]       | [nBatchSize,nSequenceLength]           | int32          | input token after tokenization     |
|       [1]       | [nBatchSize]                           | int32          | real sequence length of each input |
| input parameter |                                        |                |                                    |
|       [0]       | []                                     | int32          | max_batch_size                     |
|       [1]       | []                                     | int32          | max_seq_len                        |
|       [2]       | []                                     | int32          | beam_width (not use but necessary) |
|       [3]       | []                                     | int32          | head_num                           |
|       [4]       | []                                     | int32          | size_per_head                      |
|       [5]       | []                                     | int32          | inter_size                         |
|       [6]       | []                                     | int32          | d_model                            |
|       [7]       | []                                     | int32          | num_layer                          |
|       [8]       | []                                     | int32          | num_bucket                         |
|       [9]       | []                                     | int32          | max_distance                       |
|      [10]       | []                                     | int32          | sm                                 |
|      [11]       | []                                     | float32        | q_scaling                          |
|  output tensor  |                                        |                |                                    |
|       [0]       | [nBatchSize,nSequenceLength,nModelDim] | foat32/float16 | encoder output                     |

---
## T5DecodingPlugin in T5Plugin
| Classification  | Tensor/Parameter Shape                  | Data Type       | Description                         |
| :-------------: | :-------------------------------------- | :-------------- | :---------------------------------- |
|  input tensor   |                                         |                 |                                     |
|       [0]       | [nBatchSize,nSequenceLength,nModelDim]  | foat32/float16  | encoder output                      |
|       [1]       | [nBatchSize]                            | int32           | real sequence length of each input  |
| input parameter |                                         |                 |                                     |
|       [0]       | []                                      | int32           | max_batch_size                      |
|       [1]       | []                                      | int32           | max_seq_len                         |
|       [2]       | []                                      | int32           | mem_max_seq_len                     |
|       [3]       | []                                      | int32           | beam_width                          |
|       [4]       | []                                      | int32           | head_num                            |
|       [5]       | []                                      | int32           | size_per_head                       |
|       [6]       | []                                      | int32           | inter_size                          |
|       [7]       | []                                      | int32           | d_model                             |
|       [8]       | []                                      | int32           | num_layer                           |
|       [9]       | []                                      | int32           | vocab_size                          |
|      [10]       | []                                      | int32           | num_bucket                          |
|      [11]       | []                                      | int32           | max_distance                        |
|      [12]       | []                                      | int32           | start_id                            |
|      [13]       | []                                      | int32           | end_id                              |
|      [14]       | []                                      | float32         | beam_search_diversity_rate          |
|      [15]       | []                                      | int32           | top_k                               |
|      [16]       | []                                      | float32         | top_p                               |
|      [17]       | []                                      | float32         | temperature                         |
|      [18]       | []                                      | float32         | len_penalty                         |
|      [19]       | []                                      | float32         | repetition_penalty                  |
|  output tensor  |                                         |                 |                                     |
|       [0]       | [nBatchSize,nBeamSize,nSequenceLength]  | float32/float16 | decoding output                     |
|       [1]       | [nBatchSize,nBeamSize]                  | float32/float16 | decoding parent output (useless)    |
|       [2]       | [nBatchSize,nBeamSize]                  | float32/float16 | real sequence length of each output |

---
## Performance test result
+ [**Here**](https://nvidia-my.sharepoint.com/:x:/p/wili/EV9nBHbLFG1HuGAKt626mJgBwz-k3FVQtg3FnJ3GQdfCCw?e=1OeT0Q)

