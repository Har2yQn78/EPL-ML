from dotenv import load_dotenv
load_dotenv()

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date
import pandas as pd
import joblib
import numpy as np
from typing import Dict, List, Union, Any
from sklearn.pipeline import Pipeline

try:
    from .utils import prepare_features_and_predict
except ImportError:
    def prepare_features_and_predict(*args, **kwargs):
        raise NotImplementedError("Utility function 'prepare_features_and_predict' not loaded. Make sure utils.py exists.")

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
  f'HomeAvgLast{ROLLING_WINDOW_SIZE}_Points_Overall',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_GoalsScored_Overall',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_GoalsConceded_Overall',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_ShotsOnTarget_Overall',
  f'AwayAvgLast{ROLLING_WINDOW_SIZE}_Points_Overall'
]

loaded_pipeline: Union[Pipeline, None] = None
loaded_imputer: Union[Any, None] = None
historical_df_sorted: Union[pd.DataFrame, None] = None
all_teams_in_history: Union[List[str], None] = None

app = FastAPI(
    title="EPL Match Prediction API",
    description="Predicts English Premier League match outcomes using a trained model.",
    version="1.0.0",
)

allowed_origins_str = os.getenv("ALLOWED_ORIGINS")

origins = []
if allowed_origins_str:
    origins = [origin.strip() for origin in allowed_origins_str.split(',') if origin.strip()]
    print(f"CORS configured with allowed origins from environment variable: {origins}")
else:
    print("WARNING: ALLOWED_ORIGINS environment variable is not set. CORS will not allow any cross-origin requests.")
    print("Set ALLOWED_ORIGINS in your .env file (local) or Render environment settings (deployment).")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        all_teams_in_history = None
    else:
        try:
            historical_df = pd.read_csv(DATASET_PATH)
            if 'MatchDate' in historical_df.columns:
                historical_df['MatchDate'] = pd.to_datetime(historical_df['MatchDate'])
                historical_df_sorted = historical_df.sort_values(by='MatchDate').reset_index(drop=True)

                if 'HomeTeam' in historical_df_sorted.columns and 'AwayTeam' in historical_df_sorted.columns:
                     all_teams_in_history = pd.concat([historical_df_sorted['HomeTeam'], historical_df_sorted['AwayTeam']]).unique().tolist()
                else:
                     print("Warning: 'HomeTeam' or 'AwayTeam' column not found in historical data. Cannot validate teams.")
                     all_teams_in_history = []

                print(f"Historical data loaded and sorted successfully from {DATASET_PATH}")
                if historical_df_sorted is not None and not historical_df_sorted.empty:
                     print(f"Loaded history contains data from {historical_df_sorted['MatchDate'].min().date()} up to {historical_df_sorted['MatchDate'].max().date()}")
                else:
                     print("Historical data loaded, but it appears empty.")

            else:
                print("Error: 'MatchDate' column not found in historical data. Cannot process dates.")
                historical_df_sorted = None
                all_teams_in_history = None

        except Exception as e:
            print(f"Error loading or processing historical data: {e}")
            historical_df_sorted = None
            all_teams_in_history = None

    if loaded_pipeline is not None and hasattr(loaded_pipeline, 'steps'):
         classifier_step = loaded_pipeline.steps[-1][1] if loaded_pipeline.steps else None
         if hasattr(classifier_step, 'n_features_in_'):
            print(f"Classifier step expects {classifier_step.n_features_in_} features based on n_features_in_.")
            if classifier_step and hasattr(loaded_pipeline.named_steps['classifier'], 'n_features_in_') and loaded_pipeline.named_steps['classifier'].n_features_in_ != len(EXPECTED_FEATURES):
               print("WARNING: Feature count mismatch between loaded model and EXPECTED_FEATURES list.")
               print(f"Model expects {loaded_pipeline.named_steps['classifier'].n_features_in_}, EXPECTED_FEATURES has {len(EXPECTED_FEATURES)}.")
            elif not classifier_step:
                 print("Warning: Could not identify classifier step in the pipeline for feature count check.")
         else:
            print("Warning: Final pipeline step does not have 'n_features_in_' attribute for feature count check.")
    elif loaded_pipeline is not None:
         print("Warning: Loaded pipeline does not have a standard 'steps' attribute for feature count check.")
    else:
         print("Model pipeline not loaded, skipping feature count check.")


@app.get("/")
async def read_root():
    return {"message": "EPL Match Prediction API is running. Use /predict to get predictions."}

@app.post("/predict")
async def predict_match(request: PredictionRequest) -> Dict[str, float]:
    if loaded_pipeline is None or loaded_imputer is None or historical_df_sorted is None or historical_df_sorted.empty or all_teams_in_history is None or not EXPECTED_FEATURES:
        print("Server readiness check failed. Assets not loaded correctly.")
        raise HTTPException(status_code=500, detail="Server not ready: Model or data not loaded correctly during startup. Please check server logs.")

    match_date_ts = pd.Timestamp(request.match_date)

    latest_history_date = historical_df_sorted['MatchDate'].max()
    earliest_history_date = historical_df_sorted['MatchDate'].min()

    if match_date_ts > latest_history_date:
         print(f"Predicting for future date: {request.match_date} which is after latest history ({latest_history_date.date()}).")
    elif match_date_ts < earliest_history_date:
         print(f"Warning: Predicting for date {request.match_date} which is before earliest history ({earliest_history_date.date()}). Prediction may be unreliable due to limited prior data.")
    else:
        print(f"Predicting for date {request.match_date} within historical range ({earliest_history_date.date()} to {latest_history_date.date()}).")

    if request.home_team not in all_teams_in_history or request.away_team not in all_teams_in_history:
         invalid_teams = [team for team in [request.home_team, request.away_team] if team not in all_teams_in_history]
         invalid_teams_joined = ', '.join(f"'{t}'" for t in invalid_teams)
         detail_message = f"One or both teams ({invalid_teams_joined}) not found in historical data."
         raise HTTPException(status_code=400, detail=detail_message)


    if request.home_team == request.away_team:
         raise HTTPException(status_code=400, detail="Home and Away teams cannot be the same.")

    try:
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
    except Exception as e:
        print(f"Error during feature preparation or prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: Failed to generate prediction features or probabilities. Details: {e}")


    if not isinstance(predicted_probabilities, dict) or not all(key in predicted_probabilities for key in ['H', 'D', 'A']):
         print(f"Warning: Prediction result in unexpected format: {predicted_probabilities}")
         raise HTTPException(status_code=500, detail="Internal Server Error: Prediction result in unexpected format (Expected {{'H', 'D', 'A'}}).")

    final_result: Dict[str, float] = {
        'H': float(predicted_probabilities.get('H', 0.0)),
        'D': float(predicted_probabilities.get('D', 0.0)),
        'A': float(predicted_probabilities.get('A', 0.0)),
    }

    return final_result