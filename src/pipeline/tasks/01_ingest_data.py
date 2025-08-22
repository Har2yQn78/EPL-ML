import pandas as pd
import penaltyblog as pb
from penaltyblog.scrapers import FootballData
from understat import Understat
import asyncio
import aiohttp
import os


async def fetch_understat_data(seasons):
    print("Fetching tactical data from Understat...")
    all_understat_matches = []
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        for season in seasons:
            print(f"...fetching {season} season...")
            results = await understat.get_league_results("EPL", season)
            df_season = pd.DataFrame(results)
            all_understat_matches.append(df_season)
    return pd.concat(all_understat_matches, ignore_index=True)


def fetch_odds_data(seasons_fd):
    print("\nFetching odds data from football-data.co.uk...")
    all_odds_df = []
    for season in seasons_fd:
        print(f"...fetching {season} season...")
        df_odds_season = FootballData("ENG Premier League", season).get_fixtures()
        all_odds_df.append(df_odds_season)
    return pd.concat(all_odds_df, ignore_index=True)


async def create_master_match_list():
    seasons_understat = [2022, 2023, 2024]
    seasons_fd = ["2022-2023", "2023-2024", "2024-2025"]
    df_understat = await fetch_understat_data(seasons_understat)
    df_odds = fetch_odds_data(seasons_fd)
    df_understat['team_home_understat'] = df_understat['h'].apply(lambda x: x['title'])
    df_understat['team_away_understat'] = df_understat['a'].apply(lambda x: x['title'])
    df_understat.rename(columns={'datetime': 'date'}, inplace=True)
    df_understat['date'] = pd.to_datetime(df_understat['date']).dt.date
    odds_cols = ["date", "team_home", "team_away", "goals_home", "goals_away", "psh", "psd", "psa"]
    df_odds = df_odds[odds_cols].dropna(subset=['psh', 'psd', 'psa']).copy()
    df_odds['date'] = pd.to_datetime(df_odds['date']).dt.date

    print("\nStandardizing team names and merging data sources...")
    name_mapping_understat = {
        'Manchester City': 'Man City', 'Manchester United': 'Man United',
        'West Bromwich Albion': 'West Brom', 'West Ham United': 'West Ham',
        'Stoke City': 'Stoke', 'Swansea City': 'Swansea', 'Leicester City': 'Leicester',
        'Norwich City': 'Norwich', 'AFC Bournemouth': 'Bournemouth', 'Tottenham Hotspur': 'Tottenham',
        'Wolverhampton Wanderers': 'Wolves', 'Sheffield United': 'Sheff Utd',
        'Nottingham Forest': 'Nott\'m Forest', 'Brighton and Hove Albion': 'Brighton',
        'Leeds United': 'Leeds'
    }
    df_understat.replace({'team_home_understat': name_mapping_understat, 'team_away_understat': name_mapping_understat},
                         inplace=True)

    df_master = pd.merge(
        df_odds, df_understat,
        left_on=['date', 'team_home'], right_on=['date', 'team_home_understat'],
        how='inner'
    )
    df_master = df_master[df_master['team_away'] == df_master['team_away_understat']]
    df_master.sort_values(by='date', inplace=True)
    df_master.reset_index(drop=True, inplace=True)

    print(f"\nMaster match list with {len(df_master)} games created.")

    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "master_match_list.parquet")
    df_master.to_parquet(output_path)
    print(f"Data saved successfully to {output_path}")


if __name__ == "__main__":
    asyncio.run(create_master_match_list())