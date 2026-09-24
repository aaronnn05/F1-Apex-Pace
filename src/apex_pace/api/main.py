from contextlib import asynccontextmanager
from pathlib import Path

import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "models" / "xgboost_tuned.json"

ml_model = xgb.XGBRegressor()

@asynccontextmanager
async def lifespan(app: FastAPI):       # lifespan defines what should happen when the application starts
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    ml_model.load_model(str(MODEL_PATH))
    yield       # marks boundary between startup and shutdown
    # E.g. you could have code here that runs during shutdown

app = FastAPI(
    title="ApexPace API",
    description="F1 lap-time prediction API",
    version="1.0.0",
    lifespan=lifespan
)

class LapPredictionRequest(BaseModel):
    prev_lap_time: float = Field(..., ge=60.0, le=150.0)
    rolling_3lap_mean: float = Field(..., ge=60.0, le=150.0)
    compound_code: int = Field(..., ge=1, le=5)
    is_fresh_tyre: int = Field(..., ge=0, le=1)
    TyreLife: int = Field(..., ge=1, le=60)

class PredictionResponse(BaseModel):
    predicted_lap_time_seconds: float
    model_version: str

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": MODEL_PATH.name
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_lap_time(payload: LapPredictionRequest):
    features = [
    [payload.prev_lap_time, payload.rolling_3lap_mean, payload.compound_code, payload.is_fresh_tyre, payload.TyreLife]
    ]

    prediction = ml_model.predict(features)[0]

    return PredictionResponse(
        predicted_lap_time_seconds=round(float(prediction), 3),
        model_version="xgboost_tuned_v1"
        )

