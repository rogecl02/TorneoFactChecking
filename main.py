from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
app = FastAPI()

LABELS = {
    0: "SUPPORTED",
    1: "REFUTED"
}

MODEL_PATH = "outputs/mvp-fact-checking/model_quantized"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

config = AutoConfig.from_pretrained(MODEL_PATH)
config.num_labels = 2

model = AutoModelForSequenceClassification.from_config(config)

model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

state_dict = torch.load(
    f"{MODEL_PATH}/model.pt",
    map_location=torch.device("cpu"),
    weights_only=False
)

model.load_state_dict(state_dict)
model.eval()

class InputData(BaseModel):
    claim: str
    context: str

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    enc = tokenizer(
        data.claim,
        data.context,
        max_length=96,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False 
        )

    with torch.no_grad():
        outputs = model(**enc)
        pred = outputs.logits.argmax(-1).item()

    return {
        "predicted_label": LABELS[pred]
    }