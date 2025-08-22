import pandas as pd
import xgboost as xgb
import os
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from src.core.database import engine


def train_goal_models():
    print("Step 4: Training Goal Prediction Models")
    print("Loading data from 'features' table...")
    df_features = pd.read_sql("SELECT * FROM features", engine)

    features = [
        'pi_prob_h', 'pi_prob_d', 'pi_prob_a',
        'market_prob_h', 'market_prob_d', 'market_prob_a',
        'form_xg_for_home', 'form_xg_against_home',
        'form_xg_for_away', 'form_xg_against_away'
    ]
    X = df_features[features]
    y_home = df_features['goals_home']
    y_away = df_features['goals_away']

    best_params = {'gamma': 0.1, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200}

    print("\nPerforming a final performance check on a test split...")
    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
        X, y_home, y_away, test_size=0.2, random_state=42
    )

    home_model_eval = xgb.XGBRegressor(objective='reg:squarederror', seed=42, **best_params)
    home_model_eval.fit(X_train, y_home_train)
    y_home_pred = home_model_eval.predict(X_test)
    mae_home = mean_absolute_error(y_home_test, y_home_pred)
    rmse_home = np.sqrt(mean_squared_error(y_home_test, y_home_pred))

    away_model_eval = xgb.XGBRegressor(objective='reg:squarederror', seed=42, **best_params)
    away_model_eval.fit(X_train, y_away_train)
    y_away_pred = away_model_eval.predict(X_test)
    mae_away = mean_absolute_error(y_away_test, y_away_pred)
    rmse_away = np.sqrt(mean_squared_error(y_away_test, y_away_pred))

    print("\n--- Final Goal Model Performance (on Test Set) ---")
    print(f"Home Goals -> MAE: {mae_home:.4f}, RMSE: {rmse_home:.4f}")
    print(f"Away Goals -> MAE: {mae_away:.4f}, RMSE: {rmse_away:.4f}")

    print("\nRetraining models on all available data for deployment...")
    final_home_model = xgb.XGBRegressor(objective='reg:squarederror', seed=42, **best_params)
    final_home_model.fit(X, y_home)

    final_away_model = xgb.XGBRegressor(objective='reg:squarederror', seed=42, **best_params)
    final_away_model.fit(X, y_away)

    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    home_model_path = os.path.join(output_dir, "v4_home_goals_model.joblib")
    joblib.dump(final_home_model, home_model_path)
    away_model_path = os.path.join(output_dir, "v4_away_goals_model.joblib")
    joblib.dump(final_away_model, away_model_path)
    print(f"\nFinal goal prediction models saved successfully.")


if __name__ == "__main__":
    train_goal_models()