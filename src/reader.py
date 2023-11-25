import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ReaderModel:
    def __init__(self, model_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)

    def generate_response(self, prompt, max_length=100):
        input_ids = self.tokenizer(prompt, max_length=self.model.config.max_position_embeddings, truncation=True,
                                   padding=True, return_tensors='pt')
        output = self.model.generate(input_ids, max_length=max_length, num_beams=5, early_stopping=True)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response
