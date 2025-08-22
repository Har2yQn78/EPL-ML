from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel, Field
import numpy as np
from scipy.stats import poisson

app = FastAPI(
    title="Football Match Predictor API",
    description="An API to serve predictions from our V4.1 Models"
)


print("Loading all production models...")
wdl_model = joblib.load("models/v4_model.joblib")
home_goals_model = joblib.load("models/v4_home_goals_model.joblib")
away_goals_model = joblib.load("models/v4_away_goals_model.joblib")
print("Models loaded successfully.")


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


@app.post("/predict_outcome")
def predict_outcome(features: MatchFeatures):
    input_df = pd.DataFrame([features.model_dump()])
    pred_proba = wdl_model.predict_proba(input_df)[0]

    response = {
        'prediction_probabilities': {
            'draw': float(pred_proba[0]),
            'home_win': float(pred_proba[1]),
            'away_win': float(pred_proba[2])
        }
    }
    return response


@app.post("/predict_score")
def predict_score(features: MatchFeatures):
    input_df = pd.DataFrame([features.model_dump()])

    expected_home_goals = float(home_goals_model.predict(input_df)[0])
    expected_away_goals = float(away_goals_model.predict(input_df)[0])

    max_goals = 7
    home_goal_probs = poisson.pmf(range(max_goals + 1), expected_home_goals)
    away_goal_probs = poisson.pmf(range(max_goals + 1), expected_away_goals)
    score_grid = np.outer(home_goal_probs, away_goal_probs)

    score_grid_json = {
        f"{h}-{a}": float(score_grid[h, a])
        for h in range(max_goals + 1)
        for a in range(max_goals + 1)
    }

    response = {
        'predicted_expected_goals': {
            'home': round(expected_home_goals, 2),
            'away': round(expected_away_goals, 2)
        },
        'correct_score_probabilities': score_grid_json
    }
    return response