from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="Multilingual Polarization Classifier")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_REPOS = {
    "en": "Catalyst-101/Polarization-Classification-English",
    "zh": "Catalyst-101/Polarization-Classification-Chinese",
    "ur": "Catalyst-101/Polarization-Classification-Urdu",
    "ar": "Catalyst-101/Polarization-Classification-Arabic",
    "tr": "Catalyst-101/Polarization-Classification-Turkish",
}

DEFAULT_LANG = "en"

tokenizers = {}
models = {}

def load_model(lang: str):
    if lang in models:
        return

    repo_id = MODEL_REPOS.get(lang, MODEL_REPOS[DEFAULT_LANG])

    print(f"🔹 Loading model [{lang}] from HF: {repo_id}")

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)

    model.to(DEVICE)
    model.eval()

    tokenizers[lang] = tokenizer
    models[lang] = model

@app.on_event("startup")
def preload_models():
    print("Preloading all models...")
    for lang in MODEL_REPOS.keys():
        try:
            load_model(lang)
        except Exception as e:
            print(f"Failed to load model [{lang}]: {e}")
    print("Model preloading completed.")

class RequestData(BaseModel):
    text: str
    language: str

@app.post("/predict")
def predict(data: RequestData):
    text = data.text.strip()
    lang = data.language.lower()

    lang_key = lang if lang in MODEL_REPOS else DEFAULT_LANG

    tokenizer = tokenizers[lang_key]
    model = models[lang_key]

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return {
        "language": lang_key,
        "label": pred,
        "confidence": round(confidence, 4)
    }

@app.get("/")
def health():
    return {
        "status": "running",
        "device": DEVICE,
        "loaded_models": list(models.keys())
    }