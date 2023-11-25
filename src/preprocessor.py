import re
import os
from transformers import AutoTokenizer
import nltk
import json

nltk.download('punkt')
from nltk.tokenize import sent_tokenize


# Pre processor cleans text and splits the call into paragraphs
class Preprocessor:
    def __init__(self, model_name):
        self.paragraphs = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.id2para_map = {}

    def split_into_sentences(self, text):
        return sent_tokenize(text)

    def split_into_paragraphs(self, text, max_tokens=512):
        sentences = self.split_into_sentences(text)
        self.paragraphs = []
        while len(sentences) > 0:
            para = ""
            sentence = sentences.pop(0)
            while len(self.tokenizer.tokenize(" ".join([para, sentence]))) < 508 and len(sentences) > 0:
                para = " ".join([para, sentence])
                sentence = sentences.pop(0)
            self.paragraphs.append(para)
        self.save_id2para_map()
        return self.paragraphs

    def save_id2para_map(self):
        if os.path.exists('../id2para_map.json'):
            with open('id2para_map.json', 'r') as f:
                self.id2para_map = json.load(f)
            for i, para in enumerate(self.paragraphs):
                self.id2para_map[i] = para
        else:
            self.id2para_map = {i: para for i, para in enumerate(self.paragraphs)}
        with open('../id2para_map.json', 'w') as f:
            json.dump(self.id2para_map, f)

    def clean_text(self, text):
        pass
