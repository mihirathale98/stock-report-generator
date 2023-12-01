import datasets
from transformers import AutoTokenizer, AutoModel
from transformers import DataCollatorForTokenClassification
from ner_dataset import NERDataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer


finer_dataset = datasets.load_dataset("nlpaueb/finer-139")

finer_tag_names = finer_dataset["train"].features["ner_tags"].feature.names

id2label = {i: label for i, label in enumerate(finer_tag_names)}
label2id = {v: k for k, v in id2label.items()}

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

train_dataset = NERDataset(finer_dataset["train"], tokenizer)
val_dataset = NERDataset(finer_dataset["validation"], tokenizer)
test_dataset = NERDataset(finer_dataset["test"], tokenizer)

train_dataloader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator
)
val_dataloader = DataLoader(
    val_dataset, batch_size=8, shuffle=True, collate_fn=data_collator
)
test_dataloader = DataLoader(
    test_dataset, batch_size=8, shuffle=True, collate_fn=data_collator
)

model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(finer_tag_names), id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="dbert-finer-ner",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="epoch",
    logging_steps=1000,
    eval_steps = 1000,
    gradient_accumulation_steps=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

