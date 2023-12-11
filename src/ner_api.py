import json
from fastapi import FastAPI
from transformers import pipeline
import uvicorn


def load_pipeline(model_name):
    """Load the named entity recognition pipeline"""
    pipe = pipeline("ner", model=model_name)
    return pipe


def get_predictions(text):
    """Get predictions for a given text"""
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
    text = request['text']
    preds = get_predictions(text)
    return {'ner_predictions': preds}

if __name__ == '__main__':
    uvicorn.run("ner_api:app", port=8003, workers=1)
