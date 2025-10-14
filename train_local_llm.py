import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# ======== 1️⃣ Train tokenizer locally ==========
print("Training tokenizer...")
tokenizer = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(vocab_size=2000, special_tokens=["<pad>", "<s>", "</s>", "<unk>"])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.train(["data.txt"])
tokenizer.save("local_tokenizer.json")

# ======== 2️⃣ Define model config ==========
tok = Tokenizer.from_file("local_tokenizer.json")
vocab_size = tok.get_vocab_size()

config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=128,
    n_embd=128,
    n_layer=2,
    n_head=2
)
model = GPT2LMHeadModel(config)

# ======== 3️⃣ Prepare data ==========
print("Preparing data...")
with open("data.txt") as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

encoded = [torch.tensor(tok.encode(line).ids[:128]) for line in lines]

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()

# ======== 4️⃣ Train model ==========
print("Training...")
for epoch in range(5):
    total_loss = 0
    for seq in encoded:
        seq = seq.unsqueeze(0)
        out = model(seq, labels=seq)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: avg loss = {total_loss/len(encoded):.4f}")

# ======== 5️⃣ Save ==========
print("Saving model...")
model.save_pretrained("./my_local_llm")
print("Done! Model saved in ./my_local_llm and tokenizer in local_tokenizer.json")


