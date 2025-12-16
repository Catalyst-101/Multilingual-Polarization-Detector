from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATHS = {
    "en": "models/eng_model",
    "zh": "models/zho_model",
    "ur": "models/urd_model",
    "ar": "models/arb_model",
    "tr": "models/tur_model",
}

DEFAULT_LANG_KEY = "en"

tokenizers = {}
models = {}


def _load_resources(lang_key: str):
    
    if lang_key in models:
        return

    path = MODEL_PATHS.get(lang_key, MODEL_PATHS[DEFAULT_LANG_KEY])
    
    if not Path(path).exists():
        path = MODEL_PATHS[DEFAULT_LANG_KEY]

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(DEVICE)
    model.eval()

    tokenizers[lang_key] = tokenizer
    models[lang_key] = model


class RequestData(BaseModel):
    text: str
    language: str


@app.post("/predict")
def predict(data: RequestData):
    text = data.text
    lang = data.language.lower()

    lang_key = lang if lang in MODEL_PATHS else DEFAULT_LANG_KEY
    _load_resources(lang_key)

    tokenizer = tokenizers[lang_key]
    model = models[lang_key]

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return {
        "label": pred,
        "confidence": round(confidence, 3)
    }
