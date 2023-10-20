import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('/data/llama-7b-hf')

prompt = "Hey, are you consciours? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors='pt')
model = LlamaForCausalLM.from_pretrained("/data/llama-7b-hf")
hf_config = vars(model.config)
print(hf_config)
generated_ids = model.forward(inputs.input_ids, output_hidden_states=True)
print(generated_ids)

tokens = [0,18637,29892,526,366,1136,455,2470,29973,1815,366,5193,304,592,29973,18637,29892,526,366,1136,455,2470,29973,1815,366,5193,304,592,29973,18637,29892,526,366,1136,455,2470,29973,1815,366,5193,304,592,29973,18637,29892,526,366]
print(tokenizer.decode(tokens))
