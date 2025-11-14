from fastapi import FastAPI, Form
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("models/final")
model = AutoModelForSequenceClassification.from_pretrained("models/final")
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

@app.post("/predict")
def predict(text: str = Form(...)):
    result = pipe(text)[0]
    return {"label": result["label"], "score": result["score"]}
