import datasets
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader

finer_dataset = datasets.load_dataset("nlpaueb/finer-139")

finer_tag_names = finer_dataset["train"].features["ner_tags"].feature.names

finer_dataset['train'] = finer_dataset['train'].shuffle(seed=42).select(range(200000))
finer_dataset['validation'] = finer_dataset['validation'].shuffle(seed=42).select(range(30000))
finer_dataset['test'] = finer_dataset['test'].shuffle(seed=42).select(range(30000))


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["tokens"]
        old_labels = item["ner_tags"]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, is_split_into_words=True)
        new_labels = []
        word_ids = inputs.word_ids(0)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(old_labels[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        inputs['labels'] = label_ids
        return {'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': inputs['labels']}


tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

train_dataset = NERDataset(finer_dataset["train"], tokenizer)
val_dataset = NERDataset(finer_dataset["validation"], tokenizer)
test_dataset = NERDataset(finer_dataset["test"], tokenizer)

train_dataloader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator
)
val_dataloader = DataLoader(
    val_dataset, batch_size=8, shuffle=True,
)
test_dataloader = DataLoader(
    test_dataset, batch_size=8, shuffle=True,
)

id2label = {i: label for i, label in enumerate(finer_tag_names)}
label2id = {v: k for k, v in id2label.items()}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForTokenClassification.from_pretrained(
    "distilroberta-base", num_labels=len(finer_tag_names), id2label=id2label, label2id=label2id
)

model.to(device)

model.train()

for param in model.bert.parameters():
    param.requires_grad = False

training_args = TrainingArguments(
    output_dir="distrob-finer-ner",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="epoch",
    logging_steps=1000,
    eval_steps=1000,
    gradient_accumulation_steps=16
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

with open("log_history_ner.txt", 'w') as f:
    f.write(trainer.state.log_history)

trainer.model.save_pretrained("finer_ner_model")
trainer.model.save_state_dict("finer_ner_model_st_dict")
