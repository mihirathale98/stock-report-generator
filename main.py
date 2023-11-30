import json
from src.indexer import FAISSIndexer
from src.encoder import EncoderModel
from src.preprocessor import Preprocessor
from src.reader import ReaderModel
import torch

print(torch.cuda.is_available())

earnings_call_files = ['AMC Q4 2020.txt']

model = EncoderModel('distilbert-base-uncased')
preprocessor = Preprocessor('distilbert-base-uncased')
indexer = FAISSIndexer(768)
reader = ReaderModel('TinyLlama/TinyLlama-1.1B-Chat-v0.6')


def load_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()


def build_index(file_paths):
    for file_path in file_paths:
        text = load_file(file_path)
        paragraphs = preprocessor.split_into_paragraphs(text)
        encodings = model.encode_text(paragraphs)
        indexer.create_index(encodings)


def get_answer(query):
    query_encoding = model.encode_text([query])
    relevant_paragraphs = retrieve_paragraphs(query_encoding)
    context = ""
    for pid in relevant_paragraphs[1][0]:
        context += ID_2_MAP[str(pid)] + "\n"
    answer = reader.generate_response(context)
    return answer


def retrieve_paragraphs(query_embedding, k=5):
    return indexer.search(query_embedding, k=k)


def load_map(path):
    with open(path, 'r') as f:
        ID_2_MAP = json.load(f)
    return ID_2_MAP


if __name__ == '__main__':
    build_index(earnings_call_files)
    ID_2_MAP = load_map('id2para_map.json')

    print(get_answer("How does the call sentiment look like?"))
