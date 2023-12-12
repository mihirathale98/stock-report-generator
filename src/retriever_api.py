import faiss
import json
import numpy as np
from encoder import EncoderModel
from fastapi import FastAPI
import uvicorn

"""
API to retrieve paragraphs from the index for a given query using FAISS index and HuggingFace DistilBERT model for encoding the paragraphs into vectors 
and the query into a vector and then using the FAISS index to retrieve the top 5 paragraphs for the given query from the index
"""

app = FastAPI()

# Initialize the index
index = faiss.IndexFlatIP(768)

# Initialize the encoder
model = EncoderModel('distilbert-base-uncased')

# Load the id2para_map
with open('id2para_map.json', 'r') as f:
    id2para_map = json.load(f)

# Load the encodings into the index
encodings = np.load('encodings.npy')
index.add(encodings)

@app.post("/retrieve")
def retrieve_from_index(request: dict):
    """
    Retrieve paragraphs from the index
    
    Args:
        request (dict): Request containing the query
        
    Returns:
        context (str): Context containing the top 5 paragraphs from the index for the given query
        
    """
    paragraphs = index.search(model.encode_text([request['query']]).astype(np.float32), 5)
    print(paragraphs)
    context = ""
    for pid in paragraphs[1][0]:
        context += id2para_map[f"{pid}"] + "\n"
    return {'context': context}


if __name__ == '__main__':
    uvicorn.run("indexer:app", port=8002, workers=1)
