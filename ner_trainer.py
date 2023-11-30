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

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
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
    output_dir="bert-finer-ner",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataloader=train_dataloader,
    eval_dataloader=val_dataloader,
    tokenizer=tokenizer,
)

trainer.train()

