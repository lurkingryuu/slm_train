from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
training_args = TrainingArguments(output_dir="./my-llm", per_device_train_batch_size=2, num_train_epochs=1)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_datasets["train"])
trainer.train()

