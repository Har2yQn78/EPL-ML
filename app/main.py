from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date
import pandas as pd
import joblib
import os
import numpy as np
from typing import Dict
from .utils import prepare_features_and_predict

MODEL_DIR = "model"
DATASET_PATH = "dataset/EPL-2015-2025.csv"
MODEL_PIPELINE_PATH = os.path.join(MODEL_DIR, 'full_prediction_pipeline.joblib')
IMPUTER_PATH = os.path.join(MODEL_DIR, 'imputer.joblib')
ROLLING_WINDOW_SIZE = 7

EXPECTED_FEATURES = [
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_GoalsScored_Home',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_GoalsConceded_Home',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_ShotsOnTarget_Home',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_Corners_Home',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_YellowCards_Home',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_Points_Home',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_GoalsScored_Away',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_GoalsConceded_Away',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_ShotsOnTarget_Away',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_Corners_Away',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_YellowCards_Away',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_Points_Away',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_GoalsScored_Overall',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_GoalsConceded_Overall',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_ShotsOnTarget_Overall',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_ShotsOnTarget_Overall',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_Points_Overall',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_GoalsScored_Overall',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_GoalsConceded_Overall',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_ShotsOnTarget_Overall',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_Points_Overall'
]

loaded_pipeline = None
loaded_imputer = None
historical_df_sorted = None
all_teams_in_history = None
app = FastAPI(
    title="EPL Match Prediction API",
    description="Predicts English Premier League match outcomes using a trained model.",
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    match_date: date
@app.on_event("startup")
def load_assets():
    global loaded_pipeline, loaded_imputer, historical_df_sorted, all_teams_in_history

    print("Loading model and historical data...")

    if not os.path.exists(MODEL_PIPELINE_PATH):
        print(f"Error: Model pipeline not found at {MODEL_PIPELINE_PATH}")
        loaded_pipeline = None
    else:
        try:
            loaded_pipeline = joblib.load(MODEL_PIPELINE_PATH)
            print(f"Model pipeline loaded successfully from {MODEL_PIPELINE_PATH}")
        except Exception as e:
            print(f"Error loading model pipeline: {e}")
            loaded_pipeline = None

    if not os.path.exists(IMPUTER_PATH):
         print(f"Error: Imputer not found at {IMPUTER_PATH}")
         loaded_imputer = None
    else:
        try:
            loaded_imputer = joblib.load(IMPUTER_PATH)
            print(f"Imputer loaded successfully from {IMPUTER_PATH}")
        except Exception as e:
            print(f"Error loading imputer: {e}")
            loaded_imputer = None

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Historical dataset not found at {DATASET_PATH}")
        historical_df_sorted = None
    else:
        try:
            historical_df = pd.read_csv(DATASET_PATH)
            historical_df['MatchDate'] = pd.to_datetime(historical_df['MatchDate'])
            historical_df_sorted = historical_df.sort_values(by='MatchDate').reset_index(drop=True)
            all_teams_in_history = pd.concat([historical_df_sorted['HomeTeam'], historical_df_sorted['AwayTeam']]).unique().tolist()
            print(f"Historical data loaded and sorted successfully from {DATASET_PATH}")
            print(f"Loaded history contains data up to {historical_df_sorted['MatchDate'].max().date()}")

        except Exception as e:
            print(f"Error loading or processing historical data: {e}")
            historical_df_sorted = None
            all_teams_in_history = None

    if loaded_pipeline and hasattr(loaded_pipeline.named_steps['classifier'], 'n_features_in_'):
         if loaded_pipeline.named_steps['classifier'].n_features_in_ != len(EXPECTED_FEATURES):
              print(f"WARNING: Model trained on {loaded_pipeline.named_steps['classifier'].n_features_in_} features, but EXPECTED_FEATURES list has {len(EXPECTED_FEATURES)}.")
              print("Prediction may fail due to feature mismatch.")

@app.get("/")
async def read_root():
    return {"message": "EPL Match Prediction API is running. Use /predict to get predictions."}

@app.post("/predict")
async def predict_match(request: PredictionRequest) -> Dict[str, float]:
    if loaded_pipeline is None or loaded_imputer is None or historical_df_sorted is None or not EXPECTED_FEATURES:
        raise HTTPException(status_code=500, detail="Model or data not loaded. Please check server logs.")

    match_date_ts = pd.Timestamp(request.match_date)

    if historical_df_sorted['MatchDate'].max() < match_date_ts:
         print(f"Predicting for future date: {request.match_date}. Latest historical data is {historical_df_sorted['MatchDate'].max().date()}.")
    elif historical_df_sorted['MatchDate'].min() > match_date_ts:
         print(f"Warning: Predicting for date {request.match_date} which is before earliest history {historical_df_sorted['MatchDate'].min().date()}. Prediction may be unreliable.")

    if request.home_team not in all_teams_in_history or request.away_team not in all_teams_in_history:
         raise HTTPException(status_code=400, detail=f"One or both teams ('{request.home_team}', '{request.away_team}') not found in historical data.")

    predicted_probabilities = prepare_features_and_predict(
        home_team=request.home_team,
        away_team=request.away_team,
        match_date=match_date_ts,
        historical_df_sorted=historical_df_sorted,
        model_pipeline=loaded_pipeline,
        imputer=loaded_imputer,
        expected_features=EXPECTED_FEATURES,
        window_size=ROLLING_WINDOW_SIZE
    )

    if predicted_probabilities is None:
        raise HTTPException(status_code=500, detail="Failed to generate prediction features or probabilities.")

    return {k: float(v) for k, v in predicted_probabilities.items()}

EXPECTED_FEATURES = [
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_GoalsScored_Home',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_GoalsConceded_Home',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_ShotsOnTarget_Home',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_Corners_Home',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_YellowCards_Home',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_Points_Home',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_GoalsScored_Away',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_GoalsConceded_Away',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_ShotsOnTarget_Away',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_Corners_Away',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_YellowCards_Away',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_Points_Away',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_GoalsScored_Overall',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_GoalsConceded_Overall',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_ShotsOnTarget_Overall',
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_Points_Overall',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_GoalsScored_Overall',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_GoalsConceded_Overall',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_ShotsOnTarget_Overall',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_Points_Overall'
]