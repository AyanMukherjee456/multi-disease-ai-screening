from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

# ==========================================
# IMPORT INFERENCE FUNCTIONS
# ==========================================

from src.copd.copd_inference import predict_copd
from src.diabetes.diabetes_inference import predict_diabetes
from src.heart.heart_inference import predict_heart
from src.kidney.kidney_inference import predict_kidney

# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="Multi Disease Prediction API",
    version="1.0"
)

# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# GENERIC REQUEST MODEL
# ==========================================

class PredictionRequest(BaseModel):
    data: Dict[str, Any]

# ==========================================
# COPD
# ==========================================

@app.post("/predict/copd")
def copd_prediction(request: PredictionRequest):

    result = predict_copd(request.data)

    return result

# ==========================================
# DIABETES
# ==========================================

@app.post("/predict/diabetes")
def diabetes_prediction(request: PredictionRequest):

    result = predict_diabetes(request.data)

    return result

# ==========================================
# HEART
# ==========================================

@app.post("/predict/heart")
def heart_prediction(request: PredictionRequest):

    result = predict_heart(request.data)

    return result

# ==========================================
# KIDNEY
# ==========================================

@app.post("/predict/kidney")
def kidney_prediction(request: PredictionRequest):

    result = predict_kidney(request.data)

    return result

# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Multi Disease Prediction API Running"
    }