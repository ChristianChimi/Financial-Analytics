from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
import joblib
import numpy as np
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Modello e scaler
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(2, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = SimpleModel()
model.load_state_dict(torch.load("mlp_model.pth"))
model.eval()

scaler = joblib.load("scaler.pkl")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(request: Request, nasdaq: float = Form(...), sp500: float = Form(...)):
    input_scaled = scaler.transform([[nasdaq, sp500]])
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    prediction = model(input_tensor).item()
    return templates.TemplateResponse("form.html", {
        "request": request,
        "prediction": round(prediction, 2)
    })

