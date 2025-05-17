import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
def get_rolling_stats_refined(team, match_date, historical_df_sorted, window_size, location_filter=None):
    team_matches = historical_df_sorted[historical_df_sorted['MatchDate'] < match_date].copy()
    team_matches_filtered = team_matches[(team_matches['HomeTeam'] == team) | (team_matches['AwayTeam'] == team)]

    if location_filter == 'Home':
        team_matches_filtered = team_matches_filtered[team_matches_filtered['HomeTeam'] == team]
    elif location_filter == 'Away':
        team_matches_filtered = team_matches_filtered[team_matches_filtered['AwayTeam'] == team]
    recent_matches = team_matches_filtered.sort_values(by='MatchDate').tail(window_size)

    if recent_matches.empty:
        stats = {'Points': np.nan, 'GoalsScored': np.nan, 'GoalsConceded': np.nan}
        for col in ['Shots', 'ShotsOnTarget', 'Corners', 'Fouls', 'YellowCards', 'RedCards']:
             stats[col] = np.nan
        return stats


    stats = {}
    valid_matches_results = recent_matches.dropna(subset=['FullTimeResult', 'HomeTeam', 'AwayTeam'])
    if not valid_matches_results.empty:
         points = valid_matches_results.apply(lambda row:\
             3 if (row['HomeTeam'] == team and row['FullTimeResult'] == 'H') or \
                  (row['AwayTeam'] == team and row['FullTimeResult'] == 'A') else \
             1 if row['FullTimeResult'] == 'D' else \
             0, axis=1)
         stats['Points'] = points.mean()
    else:
         stats['Points'] = np.nan

    valid_matches_goals = recent_matches.dropna(subset=['FullTimeHomeGoals', 'FullTimeAwayGoals'])
    if not valid_matches_goals.empty:
        stats['GoalsScored'] = valid_matches_goals.apply(lambda row: row['FullTimeHomeGoals'] if row['HomeTeam'] == team else row['FullTimeAwayGoals'], axis=1).mean()
        stats['GoalsConceded'] = valid_matches_goals.apply(lambda row: row['FullTimeAwayGoals'] if row['HomeTeam'] == team else row['FullTimeHomeGoals'], axis=1).mean()
    else:
        stats['GoalsScored'] = np.nan
        stats['GoalsConceded'] = np.nan

    cols_to_calc = ['ShotsOnTarget', 'Corners', 'YellowCards']

    for col in cols_to_calc:
        home_col = f'Home{col}'
        away_col = f'Away{col}'
        if home_col in recent_matches.columns and away_col in recent_matches.columns:
             valid_matches_stat = recent_matches.dropna(subset=[home_col, away_col])
             if not valid_matches_stat.empty:
                stats[col] = valid_matches_stat.apply(lambda row: row[home_col] if row['HomeTeam'] == team else row[away_col], axis=1).mean()
             else:
                stats[col] = np.nan
        else:
             stats[col] = np.nan

    return stats

def prepare_features_and_predict(home_team, away_team, match_date, historical_df_sorted,
                                 model_pipeline, imputer, expected_features, window_size):

    all_teams = pd.concat([historical_df_sorted['HomeTeam'], historical_df_sorted['AwayTeam']]).unique()
    if home_team not in all_teams or away_team not in all_teams:
        print(f"Teams {home_team} or {away_team} not found in historical data.")
        return None
    home_team_home_stats = get_rolling_stats_refined(home_team, match_date, historical_df_sorted, window_size, location_filter='Home')
    home_team_overall_stats = get_rolling_stats_refined(home_team, match_date, historical_df_sorted, window_size, location_filter=None)
    away_team_away_stats = get_rolling_stats_refined(away_team, match_date, historical_df_sorted, window_size, location_filter='Away')
    away_team_overall_stats = get_rolling_stats_refined(away_team, match_date, historical_df_sorted, window_size, location_filter=None)
    match_features_dict = {
        f'HomeAvgLast{window_size}_GoalsScored_Home': home_team_home_stats.get('GoalsScored', np.nan),
        f'HomeAvgLast{window_size}_GoalsConceded_Home': home_team_home_stats.get('GoalsConceded', np.nan),
        f'HomeAvgLast{window_size}_ShotsOnTarget_Home': home_team_home_stats.get('ShotsOnTarget', np.nan),
        f'HomeAvgLast{window_size}_Corners_Home': home_team_home_stats.get('Corners', np.nan),
        f'HomeAvgLast{window_size}_YellowCards_Home': home_team_home_stats.get('YellowCards', np.nan),
        f'HomeAvgLast{window_size}_Points_Home': home_team_home_stats.get('Points', np.nan),

        f'AwayAvgLast{window_size}_GoalsScored_Away': away_team_away_stats.get('GoalsScored', np.nan),
        f'AwayAvgLast{window_size}_GoalsConceded_Away': away_team_away_stats.get('GoalsConceded', np.nan),
        f'AwayAvgLast{window_size}_ShotsOnTarget_Away': away_team_away_stats.get('ShotsOnTarget', np.nan),
        f'AwayAvgLast{window_size}_Corners_Away': away_team_away_stats.get('Corners', np.nan),
        f'AwayAvgLast{window_size}_YellowCards_Away': away_team_away_stats.get('YellowCards', np.nan),
        f'AwayAvgLast{window_size}_Points_Away': away_team_away_stats.get('Points', np.nan),

        f'HomeAvgLast{window_size}_GoalsScored_Overall': home_team_overall_stats.get('GoalsScored', np.nan),
        f'HomeAvgLast{window_size}_GoalsConceded_Overall': home_team_overall_stats.get('GoalsConceded', np.nan),
        f'HomeAvgLast{window_size}_ShotsOnTarget_Overall': home_team_overall_stats.get('ShotsOnTarget', np.nan),
        f'HomeAvgLast{window_size}_Points_Overall': home_team_overall_stats.get('Points', np.nan),

        f'AwayAvgLast{window_size}_GoalsScored_Overall': away_team_overall_stats.get('GoalsScored', np.nan),
        f'AwayAvgLast{window_size}_GoalsConceded_Overall': away_team_overall_stats.get('GoalsConceded', np.nan),
        f'AwayAvgLast{window_size}_ShotsOnTarget_Overall': away_team_overall_stats.get('ShotsOnTarget', np.nan),
        f'AwayAvgLast{window_size}_Points_Overall': away_team_overall_stats.get('Points', np.nan),
    }

    X_new = pd.DataFrame([match_features_dict])
    X_new = X_new.reindex(columns=expected_features)
    X_new_imputed_array = imputer.transform(X_new)
    X_new_imputed = pd.DataFrame(X_new_imputed_array, columns=X_new.columns, index=X_new.index)

    try:
        prediction_proba_array = model_pipeline.predict_proba(X_new_imputed)
        class_labels = model_pipeline.classes_
        prediction_proba_dict = dict(zip(class_labels, prediction_proba_array[0]))
        return prediction_proba_dict
    except Exception as e:
        print(f"Prediction failed during model inference: {e}")
        return None