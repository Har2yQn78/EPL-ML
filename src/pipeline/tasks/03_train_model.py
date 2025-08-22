import pandas as pd
import xgboost as xgb
import os
import joblib
from sklearn.metrics import log_loss, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from src.core.database import engine


def train_model():
    print("Step 3: Training Final Model")

    print("Loading data from 'features' table...")
    try:
        df_features = pd.read_sql("SELECT * FROM features", engine)
    except Exception as e:
        print(f"Error: Could not load data from database. {e}")
        return

    def get_result(row):
        if row['goals_home'] > row['goals_away']:
            return 1  # Home Win
        elif row['goals_home'] < row['goals_away']:
            return 2  # Away Win
        else:
            return 0  # Draw

    df_features['result'] = df_features.apply(get_result, axis=1)

    features = [
        'pi_prob_h', 'pi_prob_d', 'pi_prob_a',
        'market_prob_h', 'market_prob_d', 'market_prob_a',
        'form_xg_for_home', 'form_xg_against_home',
        'form_xg_for_away', 'form_xg_against_away'
    ]
    X = df_features[features]
    y = df_features['result']

    best_params = {'gamma': 0.2, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200}

    print("\nPerforming a final performance check on a test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    eval_model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, seed=42, **best_params)
    eval_model.fit(X_train, y_train)

    y_pred_proba = eval_model.predict_proba(X_test)
    loss = log_loss(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, eval_model.predict(X_test))
    print(f"\n--- Final Model Performance (on Test Set) ---")
    print(f"Log Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.2%}")

    print(f"\nRetraining final model on all {len(X)} available matches for deployment...")
    final_production_model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, seed=42, **best_params)
    final_production_model.fit(X, y)

    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "v4_model.joblib")
    joblib.dump(final_production_model, model_path)
    print(f"Final production model saved to {model_path}")


if __name__ == "__main__":
    train_model()