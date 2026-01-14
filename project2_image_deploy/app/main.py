import io, torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from src.model import SimpleCNN

app = FastAPI(title="CIFAR-10 Classifier")

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)

import os
MODEL_PATH = "models/cifar10_cnn.pt"
if not os.path.exists(MODEL_PATH):
    if os.path.exists("../models/cifar10_cnn.pt"):
        MODEL_PATH = "../models/cifar10_cnn.pt"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print(f"Warning: {MODEL_PATH} not found. Model will use random weights.")

model.eval()

tfm = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(prob, dim=0)

    return {"label": CLASSES[int(idx)], "confidence": float(conf)}
