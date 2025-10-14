import torch
from transformers import GPT2LMHeadModel, GPT2Config
from tokenizers import Tokenizer

tok = Tokenizer.from_file("local_tokenizer.json")
config = GPT2Config.from_pretrained("./my_local_llm")
model = GPT2LMHeadModel.from_pretrained("./my_local_llm")

prompt = "Artificial intelligence"
input_ids = torch.tensor(tok.encode(prompt).ids).unsqueeze(0)

output = model.generate(input_ids, max_new_tokens=30)
decoded = tok.decode(output[0].tolist())
print("\n--- OUTPUT ---\n")
print(decoded)


