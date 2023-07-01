# run GPT2Model
# from gpt2 import GPT2Tokenizer, GPT2Model
# tokenizer = GPT2Tokenizer.from_pretrained('hf_models/gpt2')
# model = GPT2Model.from_pretrained('hf_models/gpt2')
# text = "Hello, my dog is cute"
# inputs = tokenizer(text, return_tensors='pt')
# print(inputs)
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state


# run GPT2LMHeadModel
from transformers import AutoTokenizer
from gpt2 import GPT2LMHeadModel
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('hf_models/gpt2')
model = GPT2LMHeadModel.from_pretrained('hf_models/gpt2')

model.eval()
# model.half()

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
print(inputs["input_ids"])
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
logits = outputs.logits
np.save('hf_logits',logits.detach().cpu().numpy())

output_ids = model.generate(inputs["input_ids"],max_new_tokens=32)
print(output_ids)
# output_ids = model.generate(inputs["input_ids"])
outs = tokenizer.decode(output_ids[0])
print(outs)