## Summary

* before transformer

`model['model']['language_model']['embedding']['word_embeddings']`
`model['model']['language_model']['embedding']['position_embeddings']`

* transformer block

`model['model']['language_model']['encoder']['layers.0.input_layernorm.weight']`
`model['model']['language_model']['encoder']['layers.0.input_layernorm.bias']`
`model['model']['language_model']['encoder']['layers.0.self_attention.query_key_value.weight']`
`model['model']['language_model']['encoder']['layers.0.self_attention.query_key_value.bias']`
`model['model']['language_model']['encoder']['layers.0.self_attention.dense.weight']`
`model['model']['language_model']['encoder']['layers.0.self_attention.dense.bias']`
`model['model']['language_model']['encoder']['layers.0.post_attention_layernorm.weight']`
`model['model']['language_model']['encoder']['layers.0.post_attention_layernorm.bias']`
`model['model']['language_model']['encoder']['layers.0.mlp.dense_h_to_4h.weight']`
`model['model']['language_model']['encoder']['layers.0.mlp.dense_h_to_4h.bias']`
`model['model']['language_model']['encoder']['layers.0.mlp.dense_4h_to_h.weight']`
`model['model']['language_model']['encoder']['layers.0.mlp.dense_4h_to_h.bias']`

* after transformer

`model['model']['language_model']['encoder']['final_layernorm.weight']`
`model['model']['language_model']['encoder']['final_layernorm.bias']`
`model['model']['word_embeddings_for_head']['weight']`

In different pipeline paralle rank, they use `0` as start number of layer id.

## Details

* get arguments:

* `model['checkpoint_version']`
* `model['args']`
* `model['args'].pipeline_model_parallel_size`
* `model['args'].tensor_model_parallel_size`

* `mp_rank_00_000`

```bash
 key: args
 key: checkpoint_version
 key: iteration
 key: model
  key: language_model
   key: embedding
    key: word_embeddings
     key: weight
      shape: torch.Size([3200, 20480])
    key: position_embeddings
     key: weight
      shape: torch.Size([2048, 20480])
   key: encoder
    key: layers.0.input_layernorm.weight
     shape: torch.Size([20480])
    key: layers.0.input_layernorm.bias
     shape: torch.Size([20480])
    key: layers.0.self_attention.query_key_value.weight
     shape: torch.Size([3840, 20480])
    key: layers.0.self_attention.query_key_value.bias
     shape: torch.Size([3840])
    key: layers.0.self_attention.dense.weight
     shape: torch.Size([20480, 1280])
    key: layers.0.self_attention.dense.bias
     shape: torch.Size([20480])
    key: layers.0.post_attention_layernorm.weight
     shape: torch.Size([20480])
    key: layers.0.post_attention_layernorm.bias
     shape: torch.Size([20480])
    key: layers.0.mlp.dense_h_to_4h.weight
     shape: torch.Size([5120, 20480])
    key: layers.0.mlp.dense_h_to_4h.bias
     shape: torch.Size([5120])
    key: layers.0.mlp.dense_4h_to_h.weight
     shape: torch.Size([20480, 5120])
    key: layers.0.mlp.dense_4h_to_h.bias
     shape: torch.Size([20480])
    key: layers.1.input_layernorm.weight
     shape: torch.Size([20480])
    key: layers.1.input_layernorm.bias
     shape: torch.Size([20480])
    key: layers.1.self_attention.query_key_value.weight
     shape: torch.Size([3840, 20480])
    key: layers.1.self_attention.query_key_value.bias
     shape: torch.Size([3840])
    key: layers.1.self_attention.dense.weight
     shape: torch.Size([20480, 1280])
    key: layers.1.self_attention.dense.bias
     shape: torch.Size([20480])
    key: layers.1.post_attention_layernorm.weight
     shape: torch.Size([20480])
    key: layers.1.post_attention_layernorm.bias
     shape: torch.Size([20480])
    key: layers.1.mlp.dense_h_to_4h.weight
     shape: torch.Size([5120, 20480])
    key: layers.1.mlp.dense_h_to_4h.bias
     shape: torch.Size([5120])
    key: layers.1.mlp.dense_4h_to_h.weight
     shape: torch.Size([20480, 5120])
    key: layers.1.mlp.dense_4h_to_h.bias
     shape: torch.Size([20480])
    key: layers.33.input_layernorm.weight
     shape: torch.Size([20480])
    key: layers.33.input_layernorm.bias
     shape: torch.Size([20480])
    key: layers.33.self_attention.query_key_value.weight
     shape: torch.Size([3840, 20480])
    key: layers.33.self_attention.query_key_value.bias
     shape: torch.Size([3840])
    key: layers.33.self_attention.dense.weight
     shape: torch.Size([20480, 1280])
    key: layers.33.self_attention.dense.bias
     shape: torch.Size([20480])
    key: layers.33.post_attention_layernorm.weight
     shape: torch.Size([20480])
    key: layers.33.post_attention_layernorm.bias
     shape: torch.Size([20480])
    key: layers.33.mlp.dense_h_to_4h.weight
     shape: torch.Size([5120, 20480])
    key: layers.33.mlp.dense_h_to_4h.bias
     shape: torch.Size([5120])
    key: layers.33.mlp.dense_4h_to_h.weight
     shape: torch.Size([20480, 5120])
    key: layers.33.mlp.dense_4h_to_h.bias
     shape: torch.Size([20480])
    key: layers.34.input_layernorm.weight
     shape: torch.Size([20480])
    key: layers.34.input_layernorm.bias
     shape: torch.Size([20480])
    key: layers.34.self_attention.query_key_value.weight
     shape: torch.Size([3840, 20480])
    key: layers.34.self_attention.query_key_value.bias
     shape: torch.Size([3840])
    key: layers.34.self_attention.dense.weight
     shape: torch.Size([20480, 1280])
    key: layers.34.self_attention.dense.bias
     shape: torch.Size([20480])
    key: layers.34.post_attention_layernorm.weight
     shape: torch.Size([20480])
    key: layers.34.post_attention_layernorm.bias
     shape: torch.Size([20480])
    key: layers.34.mlp.dense_h_to_4h.weight
     shape: torch.Size([5120, 20480])
    key: layers.34.mlp.dense_h_to_4h.bias
     shape: torch.Size([5120])
    key: layers.34.mlp.dense_4h_to_h.weight
     shape: torch.Size([20480, 5120])
    key: layers.34.mlp.dense_4h_to_h.bias
     shape: torch.Size([20480])
```

* `mp_rank_12_001`

```bash
 key: args
 key: checkpoint_version
 key: iteration
 key: model
  key: language_model
   key: encoder
    key: layers.0.input_layernorm.weight
     shape: torch.Size([20480])
    key: layers.0.input_layernorm.bias
     shape: torch.Size([20480])
    key: layers.0.self_attention.query_key_value.weight
     shape: torch.Size([3840, 20480])
    key: layers.0.self_attention.query_key_value.bias
     shape: torch.Size([3840])
    key: layers.0.self_attention.dense.weight
     shape: torch.Size([20480, 1280])
    key: layers.0.self_attention.dense.bias
     shape: torch.Size([20480])
    key: layers.0.post_attention_layernorm.weight
     shape: torch.Size([20480])
    key: layers.0.post_attention_layernorm.bias
     shape: torch.Size([20480])
    key: layers.0.mlp.dense_h_to_4h.weight
     shape: torch.Size([5120, 20480])
    key: layers.0.mlp.dense_h_to_4h.bias
     shape: torch.Size([5120])
    key: layers.0.mlp.dense_4h_to_h.weight
     shape: torch.Size([20480, 5120])
    key: layers.0.mlp.dense_4h_to_h.bias
     shape: torch.Size([20480])
    key: layers.1.input_layernorm.weight
     shape: torch.Size([20480])
    key: layers.1.input_layernorm.bias
     shape: torch.Size([20480])
    key: layers.1.self_attention.query_key_value.weight
     shape: torch.Size([3840, 20480])
    key: layers.1.self_attention.query_key_value.bias
     shape: torch.Size([3840])
    key: layers.1.self_attention.dense.weight
     shape: torch.Size([20480, 1280])
    key: layers.1.self_attention.dense.bias
     shape: torch.Size([20480])
    key: layers.1.post_attention_layernorm.weight
     shape: torch.Size([20480])
    key: layers.1.post_attention_layernorm.bias
     shape: torch.Size([20480])
    key: layers.1.mlp.dense_h_to_4h.weight
     shape: torch.Size([5120, 20480])
    key: layers.1.mlp.dense_h_to_4h.bias
     shape: torch.Size([5120])
    key: layers.1.mlp.dense_4h_to_h.weight
     shape: torch.Size([20480, 5120])
    key: layers.1.mlp.dense_4h_to_h.bias
     shape: torch.Size([20480])
    key: layers.33.input_layernorm.weight
     shape: torch.Size([20480])
    key: layers.33.input_layernorm.bias
     shape: torch.Size([20480])
    key: layers.33.self_attention.query_key_value.weight
     shape: torch.Size([3840, 20480])
    key: layers.33.self_attention.query_key_value.bias
     shape: torch.Size([3840])
    key: layers.33.self_attention.dense.weight
     shape: torch.Size([20480, 1280])
    key: layers.33.self_attention.dense.bias
     shape: torch.Size([20480])
    key: layers.33.post_attention_layernorm.weight
     shape: torch.Size([20480])
    key: layers.33.post_attention_layernorm.bias
     shape: torch.Size([20480])
    key: layers.33.mlp.dense_h_to_4h.weight
     shape: torch.Size([5120, 20480])
    key: layers.33.mlp.dense_h_to_4h.bias
     shape: torch.Size([5120])
    key: layers.33.mlp.dense_4h_to_h.weight
     shape: torch.Size([20480, 5120])
    key: layers.33.mlp.dense_4h_to_h.bias
     shape: torch.Size([20480])
    key: layers.34.input_layernorm.weight
     shape: torch.Size([20480])
    key: layers.34.input_layernorm.bias
     shape: torch.Size([20480])
    key: layers.34.self_attention.query_key_value.weight
     shape: torch.Size([3840, 20480])
    key: layers.34.self_attention.query_key_value.bias
     shape: torch.Size([3840])
    key: layers.34.self_attention.dense.weight
     shape: torch.Size([20480, 1280])
    key: layers.34.self_attention.dense.bias
     shape: torch.Size([20480])
    key: layers.34.post_attention_layernorm.weight
     shape: torch.Size([20480])
    key: layers.34.post_attention_layernorm.bias
     shape: torch.Size([20480])
    key: layers.34.mlp.dense_h_to_4h.weight
     shape: torch.Size([5120, 20480])
    key: layers.34.mlp.dense_h_to_4h.bias
     shape: torch.Size([5120])
    key: layers.34.mlp.dense_4h_to_h.weight
     shape: torch.Size([20480, 5120])
    key: layers.34.mlp.dense_4h_to_h.bias
     shape: torch.Size([20480])
```

* mp_rank_00_002

```bash
 key: args
 key: checkpoint_version
 key: iteration
 key: model
  key: language_model
   key: encoder
    key: layers.0.input_layernorm.weight
     shape: torch.Size([20480])
    key: layers.0.input_layernorm.bias
     shape: torch.Size([20480])
    key: layers.0.self_attention.query_key_value.weight
     shape: torch.Size([3840, 20480])
    key: layers.0.self_attention.query_key_value.bias
     shape: torch.Size([3840])
    key: layers.0.self_attention.dense.weight
     shape: torch.Size([20480, 1280])
    key: layers.0.self_attention.dense.bias
     shape: torch.Size([20480])
    key: layers.0.post_attention_layernorm.weight
     shape: torch.Size([20480])
    key: layers.0.post_attention_layernorm.bias
     shape: torch.Size([20480])
    key: layers.0.mlp.dense_h_to_4h.weight
     shape: torch.Size([5120, 20480])
    key: layers.0.mlp.dense_h_to_4h.bias
     shape: torch.Size([5120])
    key: layers.0.mlp.dense_4h_to_h.weight
     shape: torch.Size([20480, 5120])
    key: layers.0.mlp.dense_4h_to_h.bias
     shape: torch.Size([20480])
    key: layers.1.input_layernorm.weight
     shape: torch.Size([20480])
    key: layers.1.input_layernorm.bias
     shape: torch.Size([20480])
    key: layers.1.self_attention.query_key_value.weight
     shape: torch.Size([3840, 20480])
    key: layers.1.self_attention.query_key_value.bias
     shape: torch.Size([3840])
    key: layers.1.self_attention.dense.weight
     shape: torch.Size([20480, 1280])
    key: layers.1.self_attention.dense.bias
     shape: torch.Size([20480])
    key: layers.1.post_attention_layernorm.weight
     shape: torch.Size([20480])
    key: layers.1.post_attention_layernorm.bias
     shape: torch.Size([20480])
    key: layers.1.mlp.dense_h_to_4h.weight
     shape: torch.Size([5120, 20480])
    key: layers.1.mlp.dense_h_to_4h.bias
     shape: torch.Size([5120])
    key: layers.1.mlp.dense_4h_to_h.weight
     shape: torch.Size([20480, 5120])
    key: layers.1.mlp.dense_4h_to_h.bias
     shape: torch.Size([20480])
    key: layers.33.input_layernorm.weight
     shape: torch.Size([20480])
    key: layers.33.input_layernorm.bias
     shape: torch.Size([20480])
    key: layers.33.self_attention.query_key_value.weight
     shape: torch.Size([3840, 20480])
    key: layers.33.self_attention.query_key_value.bias
     shape: torch.Size([3840])
    key: layers.33.self_attention.dense.weight
     shape: torch.Size([20480, 1280])
    key: layers.33.self_attention.dense.bias
     shape: torch.Size([20480])
    key: layers.33.post_attention_layernorm.weight
     shape: torch.Size([20480])
    key: layers.33.post_attention_layernorm.bias
     shape: torch.Size([20480])
    key: layers.33.mlp.dense_h_to_4h.weight
     shape: torch.Size([5120, 20480])
    key: layers.33.mlp.dense_h_to_4h.bias
     shape: torch.Size([5120])
    key: layers.33.mlp.dense_4h_to_h.weight
     shape: torch.Size([20480, 5120])
    key: layers.33.mlp.dense_4h_to_h.bias
     shape: torch.Size([20480])
    key: layers.34.input_layernorm.weight
     shape: torch.Size([20480])
    key: layers.34.input_layernorm.bias
     shape: torch.Size([20480])
    key: layers.34.self_attention.query_key_value.weight
     shape: torch.Size([3840, 20480])
    key: layers.34.self_attention.query_key_value.bias
     shape: torch.Size([3840])
    key: layers.34.self_attention.dense.weight
     shape: torch.Size([20480, 1280])
    key: layers.34.self_attention.dense.bias
     shape: torch.Size([20480])
    key: layers.34.post_attention_layernorm.weight
     shape: torch.Size([20480])
    key: layers.34.post_attention_layernorm.bias
     shape: torch.Size([20480])
    key: layers.34.mlp.dense_h_to_4h.weight
     shape: torch.Size([5120, 20480])
    key: layers.34.mlp.dense_h_to_4h.bias
     shape: torch.Size([5120])
    key: layers.34.mlp.dense_4h_to_h.weight
     shape: torch.Size([20480, 5120])
    key: layers.34.mlp.dense_4h_to_h.bias
     shape: torch.Size([20480])
    key: final_layernorm.weight
     shape: torch.Size([20480])
    key: final_layernorm.bias
     shape: torch.Size([20480])
  key: word_embeddings_for_head
   key: weight
    shape: torch.Size([3200, 20480])
```