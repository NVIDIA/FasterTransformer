## Summary

* before transformer

`model['model']['language_model']['embedding']['word_embeddings']`
`model['model']['language_model']['embedding']['position_embeddings']`

* transformer block

`model['model']['language_model']['transformer']['layers.0.input_layernorm.weight']`
`model['model']['language_model']['transformer']['layers.0.input_layernorm.bias']`
`model['model']['language_model']['transformer']['layers.0.attention.query_key_value.weight']`
`model['model']['language_model']['transformer']['layers.0.attention.query_key_value.bias']`
`model['model']['language_model']['transformer']['layers.0.attention.dense.weight']`
`model['model']['language_model']['transformer']['layers.0.attention.dense.bias']`
`model['model']['language_model']['transformer']['layers.0.post_attention_layernorm.weight']`
`model['model']['language_model']['transformer']['layers.0.post_attention_layernorm.bias']`
`model['model']['language_model']['transformer']['layers.0.mlp.dense_h_to_4h.weight']`
`model['model']['language_model']['transformer']['layers.0.mlp.dense_h_to_4h.bias']`
`model['model']['language_model']['transformer']['layers.0.mlp.dense_4h_to_h.weight']`
`model['model']['language_model']['transformer']['layers.0.mlp.dense_4h_to_h.bias']`

* after transformer

`model['model']['language_model']['transformer']['final_layernorm.weight']`
`model['model']['language_model']['transformer']['final_layernorm.bias']`


* `mp_rank_00_000`

```bash
 key: iteration
 key: model
  key: language_model
   key: embedding
    key: word_embeddings
     key: weight
      shape: torch.Size([50304, 1024])
    key: position_embeddings
     key: weight
      shape: torch.Size([1024, 1024])
   key: transformer
    key: layers.0.input_layernorm.weight
     shape: torch.Size([1024])
    key: layers.0.input_layernorm.bias
     shape: torch.Size([1024])
    key: layers.0.attention.query_key_value.weight
     shape: torch.Size([3072, 1024])
    key: layers.0.attention.query_key_value.bias
     shape: torch.Size([3072])
    key: layers.0.attention.dense.weight
     shape: torch.Size([1024, 1024])
    key: layers.0.attention.dense.bias
     shape: torch.Size([1024])
    key: layers.0.post_attention_layernorm.weight
     shape: torch.Size([1024])
    key: layers.0.post_attention_layernorm.bias
     shape: torch.Size([1024])
    key: layers.0.mlp.dense_h_to_4h.weight
     shape: torch.Size([4096, 1024])
    key: layers.0.mlp.dense_h_to_4h.bias
     shape: torch.Size([4096])
    key: layers.0.mlp.dense_4h_to_h.weight
     shape: torch.Size([1024, 4096])
    key: layers.0.mlp.dense_4h_to_h.bias
     shape: torch.Size([1024])
    key: layers.1.input_layernorm.weight
     shape: torch.Size([1024])
    key: layers.1.input_layernorm.bias
     shape: torch.Size([1024])
    key: layers.1.attention.query_key_value.weight
     shape: torch.Size([3072, 1024])
    key: layers.1.attention.query_key_value.bias
     shape: torch.Size([3072])
    key: layers.1.attention.dense.weight
     shape: torch.Size([1024, 1024])
    key: layers.1.attention.dense.bias
     shape: torch.Size([1024])
    key: layers.1.post_attention_layernorm.weight
     shape: torch.Size([1024])
    key: layers.1.post_attention_layernorm.bias
     shape: torch.Size([1024])
    key: layers.1.mlp.dense_h_to_4h.weight
     shape: torch.Size([4096, 1024])
    key: layers.1.mlp.dense_h_to_4h.bias
     shape: torch.Size([4096])
    key: layers.1.mlp.dense_4h_to_h.weight
     shape: torch.Size([1024, 4096])
    key: layers.1.mlp.dense_4h_to_h.bias
     shape: torch.Size([1024])
    key: layers.22.input_layernorm.weight
     shape: torch.Size([1024])
    key: layers.22.input_layernorm.bias
     shape: torch.Size([1024])
    key: layers.22.attention.query_key_value.weight
     shape: torch.Size([3072, 1024])
    key: layers.22.attention.query_key_value.bias
     shape: torch.Size([3072])
    key: layers.22.attention.dense.weight
     shape: torch.Size([1024, 1024])
    key: layers.22.attention.dense.bias
     shape: torch.Size([1024])
    key: layers.22.post_attention_layernorm.weight
     shape: torch.Size([1024])
    key: layers.22.post_attention_layernorm.bias
     shape: torch.Size([1024])
    key: layers.22.mlp.dense_h_to_4h.weight
     shape: torch.Size([4096, 1024])
    key: layers.22.mlp.dense_h_to_4h.bias
     shape: torch.Size([4096])
    key: layers.22.mlp.dense_4h_to_h.weight
     shape: torch.Size([1024, 4096])
    key: layers.22.mlp.dense_4h_to_h.bias
     shape: torch.Size([1024])
    key: layers.23.input_layernorm.weight
     shape: torch.Size([1024])
    key: layers.23.input_layernorm.bias
     shape: torch.Size([1024])
    key: layers.23.attention.query_key_value.weight
     shape: torch.Size([3072, 1024])
    key: layers.23.attention.query_key_value.bias
     shape: torch.Size([3072])
    key: layers.23.attention.dense.weight
     shape: torch.Size([1024, 1024])
    key: layers.23.attention.dense.bias
     shape: torch.Size([1024])
    key: layers.23.post_attention_layernorm.weight
     shape: torch.Size([1024])
    key: layers.23.post_attention_layernorm.bias
     shape: torch.Size([1024])
    key: layers.23.mlp.dense_h_to_4h.weight
     shape: torch.Size([4096, 1024])
    key: layers.23.mlp.dense_h_to_4h.bias
     shape: torch.Size([4096])
    key: layers.23.mlp.dense_4h_to_h.weight
     shape: torch.Size([1024, 4096])
    key: layers.23.mlp.dense_4h_to_h.bias
     shape: torch.Size([1024])
    key: final_layernorm.weight
     shape: torch.Size([1024])
    key: final_layernorm.bias
     shape: torch.Size([1024])
```