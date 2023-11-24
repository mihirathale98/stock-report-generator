import faiss
class FAISSIndexer:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)

    def create_index(self, encodings):
        self.index.add(encodings)

    def search(self, query_vector, k=5):
        return self.index.search(query_vector, k)
