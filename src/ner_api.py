import json
from fastapi import FastAPI
from transformers import pipeline
import uvicorn

'''
API to get named entity recognition predictions for a given text

'''

def load_pipeline(model_name):
    '''
    Load the named entity recognition pipeline
    
    Args:
        model_name (str): Model name from HuggingFace
        
    Returns:
        pipe (transformers.Pipeline): Named entity recognition pipeline
        
    '''
    pipe = pipeline("ner", model=model_name)
    return pipe


def get_predictions(text):
    '''
    Get predictions for a given text
    
    Args:
        text (str): Text to get predictions for
        
    Returns:
        words (list): List of words in the text
        
    '''
    words = []
    preds = pipe(text)
    for pred in preds:
        words.append(pred['word'])
    return words

# Load the pipeline
pipe = load_pipeline("dslim/bert-base-NER")

app = FastAPI()

@app.post("/ner")
async def ner(request: dict):
    '''
    Get named entity recognition predictions for a given text
    
    Args:
        request (dict): Request containing the text
        
    Returns:
        ner_predictions (list): List of named entities
        
    '''
    text = request['text']
    preds = get_predictions(text)
    return {'ner_predictions': preds}

'''
Run the API with uvicorn

'''
if __name__ == '__main__':
    uvicorn.run("ner_api:app", port=8003, workers=1)
