import datasets
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader

"""
Fine-tune the BERT model for NER using the Finer dataset from HuggingFace Datasets library and save the model for inference later on.
"""

# Load the dataset
finer_dataset = datasets.load_dataset("nlpaueb/finer-139")


# Get the tag names
finer_tag_names = finer_dataset["train"].features["ner_tags"].feature.names

# Select a subset of the data
finer_dataset['train'] = finer_dataset['train'].shuffle(seed=42).select(range(200000))
finer_dataset['validation'] = finer_dataset['validation'].shuffle(seed=42).select(range(30000))
finer_dataset['test'] = finer_dataset['test'].shuffle(seed=42).select(range(30000))


class NERDataset(torch.utils.data.Dataset):
    """
    Custom dataset for NER
    
    """
    def __init__(self, dataset, tokenizer):
        """
        Initialize the dataset
        
        Args:
            dataset (datasets.Dataset): Dataset from HuggingFace Datasets library
            tokenizer (transformers.AutoTokenizer): Tokenizer from HuggingFace Transformers library
            
        """
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Return the length of the dataset
        
        Returns:
            len (int): Length of the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get an item from the dataset
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            item (dict): Dictionary containing the input_ids, attention_mask and labels
            
        """
        # Get the item
        item = self.dataset[idx]

        # Get the text and labels
        text = item["tokens"]

        # Get the labels
        old_labels = item["ner_tags"]

        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, is_split_into_words=True)
        new_labels = []

        # Get the labels for each word
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


# Initialize the tokenizer and the data collator
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

# Initialize the datasets
train_dataset = NERDataset(finer_dataset["train"], tokenizer)
val_dataset = NERDataset(finer_dataset["validation"], tokenizer)
test_dataset = NERDataset(finer_dataset["test"], tokenizer)

# Initialize the dataloaders
train_dataloader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator
)
val_dataloader = DataLoader(
    val_dataset, batch_size=8, shuffle=True,
)
test_dataloader = DataLoader(
    test_dataset, batch_size=8, shuffle=True,
)

# Get the id to label map
id2label = {i: label for i, label in enumerate(finer_tag_names)}
label2id = {v: k for k, v in id2label.items()}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the model
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased", num_labels=len(finer_tag_names), id2label=id2label, label2id=label2id
)

model.to(device)

# Set the model to train mode
model.train()

# Freeze the parameters for faster training
for param in model.bert.parameters():
    param.requires_grad = False

# Initialize the training arguments
training_args = TrainingArguments(
    output_dir="bert-finer-ner",
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

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the log history
with open("log_history_ner.txt", 'w') as f:
    f.write(trainer.state.log_history)

# Save the model
trainer.model.save_pretrained("finer_ner_model")
trainer.model.save_state_dict("finer_ner_model_st_dict")
