import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

class EncoderModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)

    def encode_batch(self, text_batch):
        return self.model(
            **self.tokenizer(text_batch, return_tensors="pt", max_length=self.model.config.max_position_embeddings,
                             truncation=True, padding=True).to(self.device))

    def encode_text(self, text_list, batch_size=32):
        encodings = np.zeros((len(text_list), 768))
        for i in tqdm(range(0, len(text_list), batch_size), desc="Encoding passages"):
            text_batch = text_list[i:i + batch_size]
            encodings_batch = self.encode_batch(text_batch)
            encodings[i:i + batch_size] = encodings_batch.last_hidden_state[:, 0, :].detach().numpy()
        return encodings