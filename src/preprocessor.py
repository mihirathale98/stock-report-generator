from transformers import AutoTokenizer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


# Pre processor cleans text and splits the call into paragraphs
class Preprocessor:
    """Preprocessor class for text preprocessing"""
    def __init__(self, model_name):
        self.paragraphs = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.id2para_map = {}

    def split_into_sentences(self, text):
        return sent_tokenize(text)

    def split_into_paragraphs(self, text, max_tokens=512):
        """Split the text into paragraphs with max length of 512 tokens"""
        sentences = self.split_into_sentences(text)
        self.paragraphs = []
        while len(sentences) > 0:
            para = ""
            sentence = sentences.pop(0)
            while len(self.tokenizer.tokenize(" ".join([para, sentence]))) < 508 and len(sentences) > 0:
                para = " ".join([para, sentence])
                sentence = sentences.pop(0)
            self.paragraphs.append(para)
        return self.paragraphs

    def clean_text(self, text):
        # transcripts are already cleaned, hence we haven't applied additional cleaning steps here
        # we can add cleaning steps according to the data sources
        pass
