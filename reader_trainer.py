from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import datasets


model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

dataset = datasets.load_dataset("wikitext", "wikitext-2-v1", split="train")

print(dataset)

