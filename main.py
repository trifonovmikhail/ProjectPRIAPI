from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

def load_model():
    tokenizer = AutoTokenizer.from_pretrained('s-nlp/russian_toxicity_classifier')
    model = AutoModelForSequenceClassification.from_pretrained('s-nlp/russian_toxicity_classifier')
    return tokenizer, model

def classify_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    return scores

app = FastAPI()

tokenizer, model = load_model()

class TextRequest(BaseModel):
    text: str

@app.post("/predict/")
def classify(request: TextRequest):
    try:
        text = request.text
        scores = classify_text(text, tokenizer, model)
        response = {"Нейтральность": float(scores[0]), "Токсичность": float(scores[1])}
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Классификатор токсичности текста"}