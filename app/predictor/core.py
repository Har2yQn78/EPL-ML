import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime

# Import constants and utility function
from .utils import get_rolling_stats_refined, EXPECTED_FEATURES, ROLLING_WINDOW

# Define paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model_assets', 'full_prediction_pipeline.joblib')
IMPUTER_PATH = os.path.join(BASE_DIR, 'model_assets', 'imputer.joblib')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'EPL-2015-2025.csv')

# Global variables to hold loaded assets and data
loaded_pipeline = None
loaded_imputer = None
historical_df = None
all_teams_in_data = None

def load_assets():
    """Loads the model pipeline, imputer, and historical data."""
    global loaded_pipeline, loaded_imputer, historical_df, all_teams_in_data
    print("Loading model assets and historical data...")
    try:
        loaded_pipeline = joblib.load(MODEL_PATH)
        loaded_imputer = joblib.load(IMPUTER_PATH)
        historical_df = pd.read_csv(DATA_PATH)
        historical_df['MatchDate'] = pd.to_datetime(historical_df['MatchDate'])
        # Sort data once on load
        historical_df = historical_df.sort_values(by='MatchDate').reset_index(drop=True)
        all_teams_in_data = pd.concat([historical_df['HomeTeam'], historical_df['AwayTeam']]).unique().tolist()

        print("Assets and data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading assets: {e}. Make sure 'model_assets' and 'data' directories exist and contain the correct files.")
        # Set variables to None to indicate failure
        loaded_pipeline = None
        loaded_imputer = None
        historical_df = None
        all_teams_in_data = None
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        loaded_pipeline = None
        loaded_imputer = None
        historical_df = None
        all_teams_in_data = None


def predict_match(home_team: str, away_team: str, match_date_str: str):
    """
    Predicts the outcome probabilities for a given EPL match.

    Args:
        home_team (str): Name of the home team.
        away_team (str): Name of the away team.
        match_date_str (str): Date of the match in 'YYYY-MM-DD' format.

    Returns:
        tuple: (predicted_outcome: str, probabilities: dict) or None if prediction fails.
    """
    if loaded_pipeline is None or loaded_imputer is None or historical_df is None:
        print("Prediction failed: Model assets or historical data not loaded.")
        return None

    try:
        match_date = pd.to_datetime(match_date_str)
    except ValueError:
        print(f"Error: Invalid match_date format: {match_date_str}. Use YYYY-MM-DD.")
        return None

    # Basic check if teams are in historical data
    if home_team not in all_teams_in_data or away_team not in all_teams_in_data:
         print(f"Error: One or both teams ('{home_team}', '{away_team}') not found in historical data.")
         return None

    # Check if the match date is too early for rolling stats (less than ROLLING_WINDOW matches per team)
    # This is implicitly handled by get_rolling_stats_refined returning NaNs,
    # which are then imputed. But we might want a more explicit error/warning
    # if predicting a match right at the start of a new season before any history is built.
    # For simplicity, we'll rely on imputation for early matches as in the notebook.

    # Calculate features for the new match
    home_stats = get_rolling_stats_refined(home_team, match_date, historical_df, ROLLING_WINDOW, location_filter='Home')
    home_stats_overall = get_rolling_stats_refined(home_team, match_date, historical_df, ROLLING_WINDOW, location_filter=None)

    away_stats = get_rolling_stats_refined(away_team, match_date, historical_df, ROLLING_WINDOW, location_filter='Away')
    away_stats_overall = get_rolling_stats_refined(away_team, match_date, historical_df, ROLLING_WINDOW, location_filter=None)

    # The combined features weren't used in the final notebook model (Cell 4 X.columns)
    # If you add them to EXPECTED_FEATURES, uncomment and calculate them here:
    # home_attack_away_defense = home_stats.get('GoalsScored', np.nan) - away_stats.get('GoalsConceded', np.nan)
    # away_attack_home_defense = away_stats.get('GoalsScored', np.nan) - home_stats.get('GoalsConceded', np.nan)


    # Create the feature dictionary based on the expected features
    match_features_dict = {
        f'HomeAvgLast{ROLLING_WINDOW}_GoalsScored_Home': home_stats.get('GoalsScored', np.nan),
        f'HomeAvgLast{ROLLING_WINDOW}_GoalsConceded_Home': home_stats.get('GoalsConceded', np.nan),
        f'HomeAvgLast{ROLLING_WINDOW}_ShotsOnTarget_Home': home_stats.get('ShotsOnTarget', np.nan),
        f'HomeAvgLast{ROLLING_WINDOW}_Corners_Home': home_stats.get('Corners', np.nan),
        f'HomeAvgLast{ROLLING_WINDOW}_YellowCards_Home': home_stats.get('YellowCards', np.nan),
        f'HomeAvgLast{ROLLING_WINDOW}_Points_Home': home_stats.get('Points', np.nan),

        f'AwayAvgLast{ROLLING_WINDOW}_GoalsScored_Away': away_stats.get('GoalsScored', np.nan),
        f'AwayAvgLast{ROLLING_WINDOW}_GoalsConceded_Away': away_stats.get('GoalsConceded', np.nan),
        f'AwayAvgLast{ROLLING_WINDOW}_ShotsOnTarget_Away': away_stats.get('ShotsOnTarget', np.nan),
        f'AwayAvgLast{ROLLING_WINDOW}_Corners_Away': away_stats.get('Corners', np.nan),
        f'AwayAvgLast{ROLLING_WINDOW}_YellowCards_Away': away_stats.get('YellowCards', np.nan),
        f'AwayAvgLast{ROLLING_WINDOW}_Points_Away': away_stats.get('Points', np.nan),

        f'HomeAvgLast{ROLLING_WINDOW}_GoalsScored_Overall': home_stats_overall.get('GoalsScored', np.nan),
        f'HomeAvgLast{ROLLING_WINDOW}_GoalsConceded_Overall': home_stats_overall.get('GoalsConceded', np.nan),
        f'HomeAvgLast{ROLLING_WINDOW}_ShotsOnTarget_Overall': home_stats_overall.get('ShotsOnTarget', np.nan),
        f'HomeAvgLast{ROLLING_WINDOW}_Points_Overall': home_stats_overall.get('Points', np.nan),

        f'AwayAvgLast{ROLLING_WINDOW}_GoalsScored_Overall': away_stats_overall.get('GoalsScored', np.nan),
        f'AwayAvgLast{ROLLING_WINDOW}_GoalsConceded_Overall': away_stats_overall.get('GoalsConceded', np.nan),
        f'AwayAvgLast{ROLLING_WINDOW}_ShotsOnTarget_Overall': away_stats_overall.get('ShotsOnTarget', np.nan),
        f'AwayAvgLast{ROLLING_WINDOW}_Points_Overall': away_stats_overall.get('Points', np.nan),

        # If combined features were used, include them here
        # f'HomeAttack_vs_AwayDefense_{ROLLING_WINDOW}Avg': home_attack_away_defense,
        # f'AwayAttack_vs_HomeDefense_{ROLLING_WINDOW}Avg': away_attack_home_defense,
    }

    # Create DataFrame for prediction, ensuring column order matches training data
    X_new = pd.DataFrame([match_features_dict])
    X_new = X_new.reindex(columns=EXPECTED_FEATURES)

    # Impute missing values using the fitted imputer
    X_new_imputed_array = loaded_imputer.transform(X_new)
    X_new_imputed = pd.DataFrame(X_new_imputed_array, columns=X_new.columns, index=X_new.index)

    # Predict probabilities using the loaded pipeline (which includes scaling)
    prediction_proba_array = loaded_pipeline.predict_proba(X_new_imputed)
    class_labels = loaded_pipeline.classes_ # Get class labels in order (A, D, H)

    # Format probabilities into a dictionary
    prediction_proba_dict = dict(zip(class_labels, prediction_proba_array[0].tolist())) # Convert numpy float to Python float

    # Determine the most likely outcome
    predicted_outcome = max(prediction_proba_dict, key=prediction_proba_dict.get)

    return predicted_outcome, prediction_proba_dict