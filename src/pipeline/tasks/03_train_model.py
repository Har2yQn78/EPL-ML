import pandas as pd
import xgboost as xgb
import os
import joblib
from sklearn.metrics import log_loss, accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train_model():
    print("Step 3: Training Final Model")
    input_path = "data/processed/final_feature_dataset.parquet"
    df_features = pd.read_parquet(input_path)

    def get_result(row):
        if row['goals_home'] > row['goals_away']:
            return 1
        elif row['goals_home'] < row['goals_away']:
            return 2
        else:
            return 0

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    eval_model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, seed=42, **best_params)
    eval_model.fit(X_train, y_train)
    y_pred_proba = eval_model.predict_proba(X_test)
    loss = log_loss(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, eval_model.predict(X_test))
    print(f"\n--- Final Model Performance Check ---")
    print(f"Log Loss: {loss:.4f} / Accuracy: {accuracy:.2%}")

    print("\nRetraining model on all available data for deployment...")
    final_production_model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, seed=42, **best_params)
    final_production_model.fit(X, y)

    # Save the final production model
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "v4_model.joblib")
    joblib.dump(final_production_model, model_path)
    print(f"Final production model saved to {model_path}")


if __name__ == "__main__":
    train_model()