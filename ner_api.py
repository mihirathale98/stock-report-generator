import json
import nltk
from fastapi import FastAPI
from transformers import pipeline

import uvicorn
from pyngrok import ngrok

app = FastAPI()


def load_pipeline(model_name):
    pipe = pipeline("ner", model=model_name)
    return pipe


def get_predictions(text):
    return pipe(text)


pipe = load_pipeline("dslim/bert-base-NER")
@app.post("/ner")
async def ner(request: dict):
    text = request['text']
    return get_predictions(pipe, text)