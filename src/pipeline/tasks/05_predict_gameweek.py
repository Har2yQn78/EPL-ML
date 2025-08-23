import os
import joblib
import requests
import pandas as pd
from datetime import date, timedelta, datetime, timezone

from penaltyblog.ratings import PiRatingSystem
from src.core.database import engine


def get_upcoming_fixtures(days_ahead: int = 7):
    print("Fetching upcoming fixtures from Premier League API (Pulselive)...")
    url = "https://footballapi.pulselive.com/football/fixtures"
    params = {
        "comps": "1",
        "statuses": "U",
        "page": "0",
        "pageSize": "200",
        "sort": "asc",
    }
    headers = {
        "Origin": "https://www.premierleague.com",
        "Referer": "https://www.premierleague.com/",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()

        content = data.get("content", [])
        if not content:
            print("No upcoming fixtures returned by the API.")
            return []

        today_utc = datetime.now(timezone.utc).date()
        end_date_utc = today_utc + timedelta(days=days_ahead)

        fixtures = []
        for f in content:
            kickoff_label = None
            kickoff_dt_utc = None

            kickoff = f.get("kickoff") or {}
            if "label" in kickoff and kickoff["label"]:
                kickoff_label = kickoff["label"]
                try:
                    kickoff_dt_utc = datetime.fromisoformat(
                        kickoff_label.replace("Z", "+00:00")
                    )
                except Exception:
                    kickoff_dt_utc = None

            if kickoff_dt_utc is None and "millis" in kickoff:
                try:
                    kickoff_dt_utc = datetime.fromtimestamp(
                        kickoff["millis"] / 1000.0, tz=timezone.utc
                    )
                except Exception:
                    pass

            if kickoff_dt_utc is None:
                continue

            kickoff_date_utc = kickoff_dt_utc.date()
            if not (today_utc <= kickoff_date_utc <= end_date_utc):
                continue

            teams = f.get("teams") or []
            if len(teams) < 2:
                continue

            home_team_name = None
            away_team_name = None

            for t in teams:
                team_info = t.get("team") or {}
                name = team_info.get("name")
                is_home = t.get("isHome")
                if is_home is True:
                    home_team_name = name
                elif is_home is False:
                    away_team_name = name

            if home_team_name is None or away_team_name is None:
                home_team_name = home_team_name or (teams[0].get("team") or {}).get("name")
                away_team_name = away_team_name or (teams[1].get("team") or {}).get("name")

            if not home_team_name or not away_team_name:
                continue

            fixtures.append({
                "date": kickoff_date_utc,
                "team_home": home_team_name,
                "team_away": away_team_name,
            })

        if not fixtures:
            print(f"No fixtures found in the next {days_ahead} days.")
            return []

        df_out = pd.DataFrame(fixtures).sort_values("date")
        out_path_csv = "next_gameweek_pl_api.csv"
        df_out.to_csv(out_path_csv, index=False)
        print(f"Saved upcoming fixtures to '{out_path_csv}'")
        return df_out.to_dict("records")

    except Exception as e:
        print(f"Error fetching fixtures from PL API: {e}")
        return []


def get_prematch_features(df_history, home_team, away_team):
    pi_ratings = PiRatingSystem()
    for _, row in df_history.iterrows():
        pi_ratings.update_ratings(
            row["team_home"],
            row["team_away"],
            row["goals_home"] - row["goals_away"],
        )
    pi_probs = pi_ratings.calculate_match_probabilities(home_team, away_team)

    team_history = []
    for _, row in df_history.iterrows():
        match_date = pd.to_datetime(row["date"]).date()
        team_history.append({
            "date": match_date,
            "team": row["team_home"],
            "xg_for": row["xg_home"],
            "xg_against": row["xg_away"],
        })
        team_history.append({
            "date": match_date,
            "team": row["team_away"],
            "xg_for": row["xg_away"],
            "xg_against": row["xg_home"],
        })

    df_team_history = pd.DataFrame(team_history).sort_values("date")

    df_team_history = df_team_history.set_index("date")
    form_stats = (
        df_team_history
        .groupby("team")[["xg_for", "xg_against"]]
        .rolling(window=10, min_periods=1)
        .mean()
        .rename(columns=lambda x: f"form_{x}")
    )
    form_stats = form_stats.reset_index()
    home_form = form_stats[form_stats["team"] == home_team].tail(1)
    away_form = form_stats[form_stats["team"] == away_team].tail(1)

    features = {
        "pi_prob_h": pi_probs["home_win"],
        "pi_prob_d": pi_probs["draw"],
        "pi_prob_a": pi_probs["away_win"],
        "market_prob_h": 0.0,
        "market_prob_d": 0.0,
        "market_prob_a": 0.0,
        "form_xg_for_home": home_form["form_xg_for"].iloc[0] if not home_form.empty else 1.0,
        "form_xg_against_home": home_form["form_xg_against"].iloc[0] if not home_form.empty else 1.0,
        "form_xg_for_away": away_form["form_xg_for"].iloc[0] if not away_form.empty else 1.0,
        "form_xg_against_away": away_form["form_xg_against"].iloc[0] if not away_form.empty else 1.0,
    }
    return features


def run_predictions():
    print("--- Running Weekly Prediction Pipeline ---")
    df_history = pd.read_sql("SELECT * FROM matches ORDER BY date", engine)
    model_path = os.getenv("MODEL_PATH", "models/v4_model.joblib")
    model = joblib.load(model_path)

    upcoming_fixtures = get_upcoming_fixtures(days_ahead=7)
    if not upcoming_fixtures:
        print("No upcoming fixtures found for the next 7 days.")
        return

    print("\n--- Upcoming Gameweek Predictions ---")
    predictions = []
    for fixture in upcoming_fixtures:
        home_team = fixture["team_home"]
        away_team = fixture["team_away"]

        features = get_prematch_features(df_history, home_team, away_team)
        df_features = pd.DataFrame([features])

        pred_proba = model.predict_proba(df_features)[0]
        prediction = {
            "match": f"{home_team} vs {away_team}",
            "home_win_prob": f"{pred_proba[1]:.2%}",
            "draw_prob": f"{pred_proba[0]:.2%}",
            "away_win_prob": f"{pred_proba[2]:.2%}",
        }
        predictions.append(prediction)

    df_predictions = pd.DataFrame(predictions)
    print(df_predictions.to_string(index=False))


if __name__ == "__main__":
    run_predictions()
