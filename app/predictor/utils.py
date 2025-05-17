import pandas as pd
import numpy as np

ROLLING_WINDOW = 7
TARGET = 'FullTimeResult'
def get_rolling_stats_refined(team, match_date, df_sorted, window_size, location_filter=None):
    required_cols = ['MatchDate', 'HomeTeam', 'AwayTeam', 'FullTimeResult',
                     'FullTimeHomeGoals', 'FullTimeAwayGoals',
                     'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget',
                     'HomeCorners', 'AwayCorners', 'HomeFouls', 'AwayFouls',
                     'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards']

    if not all(col in df_sorted.columns for col in required_cols):
         pass

    team_matches = df_sorted[df_sorted['MatchDate'] < match_date].copy()
    team_matches_filtered = team_matches[(team_matches['HomeTeam'] == team) | (team_matches['AwayTeam'] == team)]

    if location_filter == 'Home':
        team_matches_filtered = team_matches_filtered[team_matches_filtered['HomeTeam'] == team]
    elif location_filter == 'Away':
        team_matches_filtered = team_matches_filtered[team_matches_filtered['AwayTeam'] == team]
    recent_matches = team_matches_filtered.sort_values(by='MatchDate').tail(window_size)

    if recent_matches.empty:
        stats = {
            'Points': np.nan,
            'GoalsScored': np.nan,
            'GoalsConceded': np.nan,
            'Shots': np.nan,
            'ShotsOnTarget': np.nan,
            'Corners': np.nan,
            'Fouls': np.nan,
            'YellowCards': np.nan,
            'RedCards': np.nan
        }
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

    for col in ['Shots', 'ShotsOnTarget', 'Corners', 'Fouls', 'YellowCards', 'RedCards']:
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

EXPECTED_FEATURES = [
    f'HomeAvgLast{ROLLING_WINDOW}_GoalsScored_Home',
    f'HomeAvgLast{ROLLING_WINDOW}_GoalsConceded_Home',
    f'HomeAvgLast{ROLLING_WINDOW}_ShotsOnTarget_Home',
    f'HomeAvgLast{ROLLING_WINDOW}_Corners_Home',
    f'HomeAvgLast{ROLLING_WINDOW}_YellowCards_Home',
    f'HomeAvgLast{ROLLING_WINDOW}_Points_Home',
    f'AwayAvgLast{ROLLING_WINDOW}_GoalsScored_Away',
    f'AwayAvgLast{ROLLING_WINDOW}_GoalsConceded_Away',
    f'AwayAvgLast{ROLLING_WINDOW}_ShotsOnTarget_Away',
    f'AwayAvgLast{ROLLING_WINDOW}_Corners_Away',
    f'AwayAvgLast{ROLLING_WINDOW}_YellowCards_Away',
    f'AwayAvgLast{ROLLING_WINDOW}_Points_Away',
    f'HomeAvgLast{ROLLING_WINDOW}_GoalsScored_Overall',
    f'HomeAvgLast{ROLLING_WINDOW}_GoalsConceded_Overall',
    f'HomeAvgLast{ROLLING_WINDOW}_ShotsOnTarget_Overall',
    f'HomeAvgLast{ROLLING_WINDOW}_Points_Overall',
    f'AwayAvgLast{ROLLING_WINDOW}_GoalsScored_Overall',
    f'AwayAvgLast{ROLLING_WINDOW}_GoalsConceded_Overall',
    f'AwayAvgLast{ROLLING_WINDOW}_ShotsOnTarget_Overall',
    f'AwayAvgLast{ROLLING_WINDOW}_Points_Overall',
]