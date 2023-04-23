import transformers
import torch

from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('/data/llama-7b-hf')
prompt = "Hey"
inputs = tokenizer(prompt, return_tensors='pt')
print(inputs)

model = LlamaForCausalLM.from_pretrained("/data/llama-7b-hf")
generated_ids = model.generate(inputs.input_ids, max_length=10)
print(generated_ids)
output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)
