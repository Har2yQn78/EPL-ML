from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Import the core prediction logic and schemas
from predictor import core
from schemas import MatchInput, PredictionOutput

app = FastAPI(
    title="EPL Outcome Predictor API",
    description="API to predict English Premier League match outcomes using a trained model."
)

# --- Startup Event ---
# Load model, imputer, and data when the FastAPI application starts
@app.on_event("startup")
async def startup_event():
    core.load_assets()

# --- Root Endpoint (Optional) ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the EPL Outcome Predictor API. Go to /docs for interactive API documentation."}

# --- Prediction Endpoint ---
@app.post("/predict", response_model=PredictionOutput)
async def predict_epl_match(match_input: MatchInput):
    """
    Predicts the outcome probabilities for a given EPL match.

    Requires Home Team, Away Team, and Match Date (YYYY-MM-DD).
    """
    # Check if assets were loaded successfully
    if core.loaded_pipeline is None or core.loaded_imputer is None or core.historical_df is None:
         raise HTTPException(status_code=503, detail="Model assets or historical data not loaded. Server is not ready.")

    # Perform prediction
    result = core.predict_match(
        home_team=match_input.home_team,
        away_team=match_input.away_team,
        match_date_str=match_input.match_date
    )

    # Handle cases where prediction failed (e.g., teams not found, date too early)
    if result is None:
        # The predict_match function already prints a specific error,
        # but we raise a generic 400 error here for the API response.
         raise HTTPException(status_code=400, detail="Prediction could not be processed. Check team names and date format.")

    predicted_outcome, probabilities = result

    return PredictionOutput(
        predicted_outcome=predicted_outcome,
        probabilities=probabilities
    )

# You can add other endpoints if needed, e.g., /teams to list available teams
# or /features to list features used by the model.