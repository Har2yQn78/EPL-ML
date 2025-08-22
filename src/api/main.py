from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI(
    title="Football Match Predictor API",
    description="An API to serve predictions from our V4.1 XGBoost Model"
)


MODEL_PATH = "models/v4_model.joblib"
model = joblib.load(MODEL_PATH)

class MatchFeatures(BaseModel):
    pi_prob_h: float = Field(..., example=0.45)
    pi_prob_d: float = Field(..., example=0.25)
    pi_prob_a: float = Field(..., example=0.30)
    market_prob_h: float = Field(..., example=0.50)
    market_prob_d: float = Field(..., example=0.28)
    market_prob_a: float = Field(..., example=0.22)
    form_xg_for_home: float = Field(..., example=1.8)
    form_xg_against_home: float = Field(..., example=1.1)
    form_xg_for_away: float = Field(..., example=1.4)
    form_xg_against_away: float = Field(..., example=1.5)

    class Config:
        json_schema_extra = {
            "example": {
                "pi_prob_h": 0.45, "pi_prob_d": 0.25, "pi_prob_a": 0.30,
                "market_prob_h": 0.50, "market_prob_d": 0.28, "market_prob_a": 0.22,
                "form_xg_for_home": 1.8, "form_xg_against_home": 1.1,
                "form_xg_for_away": 1.4, "form_xg_against_away": 1.5
            }
        }


@app.post("/predict")
def predict_match(features: MatchFeatures):
    input_df = pd.DataFrame([features.model_dump()])
    pred_proba = model.predict_proba(input_df)[0]

    response = {
        'prediction_probabilities': {
            'draw': float(pred_proba[0]),
            'home_win': float(pred_proba[1]),
            'away_win': float(pred_proba[2])
        }
    }

    return response