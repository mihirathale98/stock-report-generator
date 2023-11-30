import datasets
import torch
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader


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
        inputs = self.tokenizer(text, return_tensors="pt", is_split_into_words=True)
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
