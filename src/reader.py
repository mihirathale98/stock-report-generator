import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ReaderModel:
    def __init__(self, model_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)

    def generate_response(self, prompt,model_max_length=512, max_length=100):
        input_ids = self.tokenizer(prompt, max_length=model_max_length, truncation=True,
                                   padding=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            output = self.model.generate(**input_ids, max_length=max_length, num_beams=5, early_stopping=True)
        out = output[0][len(self.tokenizer.tokenize(prompt)) + 1:]
        response = self.tokenizer.decode(out, skip_special_tokens=True)
        return response
