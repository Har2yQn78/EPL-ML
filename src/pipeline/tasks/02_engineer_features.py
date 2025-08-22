import pandas as pd
import penaltyblog as pb
from penaltyblog.ratings import PiRatingSystem
import os
from tqdm.auto import tqdm
from src.core.database import engine


def engineer_features():
    print("Step 2: Engineering Pre-Match Features")

    print("Loading data from 'matches' table...")
    try:
        df_master = pd.read_sql("SELECT * FROM matches ORDER BY date", engine)
    except Exception as e:
        print(f"Error: Could not load data from database. {e}")
        return

    print("Calculating Pi Ratings and Market Probabilities...")
    pi_ratings = PiRatingSystem()
    pi_prob_h, pi_prob_d, pi_prob_a = [], [], []

    for _, row in tqdm(df_master.iterrows(), total=df_master.shape[0], desc="Calculating Pi Ratings"):
        pi_probs = pi_ratings.calculate_match_probabilities(row['team_home'], row['team_away'])
        pi_prob_h.append(pi_probs['home_win'])
        pi_prob_d.append(pi_probs['draw'])
        pi_prob_a.append(pi_probs['away_win'])
        pi_ratings.update_ratings(row['team_home'], row['team_away'], row['goals_home'] - row['goals_away'])

    df_master['pi_prob_h'] = pi_prob_h
    df_master['pi_prob_d'] = pi_prob_d
    df_master['pi_prob_a'] = pi_prob_a

    df_master[['market_prob_h', 'market_prob_d', 'market_prob_a']] = df_master.apply(
        lambda row: pd.Series(pb.implied.power([row["psh"], row["psd"], row["psa"]])["implied_probabilities"]), axis=1
    )

    print("Calculating rolling averages for team form...")
    team_history = []
    for _, row in df_master.iterrows():
        team_history.append(
            {'date': row['date'], 'team': row['team_home'], 'xg_for': row['xg_home'], 'xg_against': row['xg_away']})
        team_history.append(
            {'date': row['date'], 'team': row['team_away'], 'xg_for': row['xg_away'], 'xg_against': row['xg_home']})
    df_history = pd.DataFrame(team_history)

    stats_for_form = ['xg_for', 'xg_against']
    form_stats = df_history.groupby('team')[stats_for_form].rolling(window=10, min_periods=1).mean().shift(1)
    form_stats = form_stats.reset_index(level=0, drop=True)  # Fix the index
    df_form = df_history[['date', 'team']].join(form_stats.rename(columns=lambda x: f'form_{x}'))

    df_final_features = pd.merge(df_master, df_form.rename(columns={'team': 'team_home'}), on=['date', 'team_home'],
                                 how='left')
    df_final_features = pd.merge(df_final_features, df_form.rename(columns={'team': 'team_away'}),
                                 on=['date', 'team_away'], how='left', suffixes=('_home', '_away'))

    df_final_features.dropna(inplace=True)

    print(f"\nFeature engineering complete. Final dataset has {len(df_final_features)} matches.")
    print("Saving data to 'features' table in PostgreSQL...")

    df_final_features.to_sql(
        'features',
        con=engine,
        if_exists='replace',
        index=False
    )
    print("Data saved successfully.")


if __name__ == "__main__":
    engineer_features()