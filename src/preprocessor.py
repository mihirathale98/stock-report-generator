from transformers import AutoTokenizer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

"""
Pre processor class for text preprocessing and splitting into paragraphs of max length 512 tokens for LLM training
"""


# Pre processor cleans text and splits the call into paragraphs
class Preprocessor:
    """Preprocessor class for text preprocessing"""
    def __init__(self, model_name):
        """
        Initialize the preprocessor

        Args:
            model_name (str): Model name from HuggingFace

        """
        self.paragraphs = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.id2para_map = {}

    def split_into_sentences(self, text):
        """
        Split the text into sentences
        
        Args:
            text (str): Text to split into sentences
            
        Returns:
            sentences (list): List of sentences in the text
        """
        return sent_tokenize(text)

    def split_into_paragraphs(self, text, max_tokens=512):
        """
        Split the text into paragraphs with max length of 512 tokens
        
        Args:
            text (str): Text to split into paragraphs
            max_tokens (int): Maximum number of tokens in a paragraph
            
        Returns:
            paragraphs (list): List of paragraphs in the text
            
        """
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
        """
        Clean the text
        
        Args:
            text (str): Text to clean
            
        Returns:
            text (str): Cleaned text
            
        """
        # transcripts are already cleaned, hence we haven't applied additional cleaning steps here
        # we can add cleaning steps according to the data sources
        pass
