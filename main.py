from src.indexer import FAISSIndexer
from src.encoder import EncoderModel
from src.preprocessor import Preprocessor

print("Loading model...")
model = EncoderModel('distilbert-base-uncased')
preprocessor = Preprocessor('distilbert-base-uncased')
print("Initializing index...")
indexer = FAISSIndexer(768)


def load_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()


def main():
    text = load_file('AMC Q4 2020.txt')
    print("Preprocessing text...")
    paragraphs = preprocessor.split_into_paragraphs(text)
    print("Encoding paragraphs...")
    encodings = model.encode_text(paragraphs)
    print("Indexing...")
    indexer.create_index(encodings)
    print("Index Ready...")
    retriever_results = indexer.search(model.encode_text('What is the growth of the stock in the given quarter?'), k=5)

    print(retriever_results)

if __name__ == '__main__':
    main()
