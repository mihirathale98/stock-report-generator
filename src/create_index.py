import faiss
import os
import json
import numpy as np
from encoder import EncoderModel
from preprocessor import Preprocessor
import torch

'''
Create the index for the paragraphs

1. Load all the paragraphs
2. Encode all the paragraphs
3. Create the index
4. Save the index and the id2para_map

'''


# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_file(file_path):
    '''
    Loads a file and returns a string
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        text (str): Text in the file
        
    '''
    with open('../earnings_call_transcripts/' + file_path, 'r') as f:
        return f.read()

# Get all the filenames
earnings_call_files = os.listdir('../earnings_call_transcripts')

# Initialize the model and the preprocessor
model = EncoderModel('distilbert-base-uncased')
preprocessor = Preprocessor('distilbert-base-uncased')

# Load all the files
all_paragraphs = []
for file_path in earnings_call_files:
    text = load_file(file_path)
    paragraphs = preprocessor.split_into_paragraphs(text)
    all_paragraphs.extend(paragraphs)

# Save the id2para_map
id2para_map = {int(i): para for i, para in enumerate(all_paragraphs)}
with open('id2para_map.json', 'w') as f:
    json.dump(id2para_map, f)

# Encode the paragraphs
encodings = model.encode_text(all_paragraphs)
encodings = encodings.astype(np.float32)

# Save the encodings
np.save('encodings.npy', encodings)