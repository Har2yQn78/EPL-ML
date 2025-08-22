import pandas as pd
from penaltyblog.scrapers import FootballData
from understat import Understat
import asyncio
import aiohttp
import os


async def fetch_understat_data(seasons):
    print("Fetching tactical data from Understat...")
    all_matches = []
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        for season in seasons:
            print(f"...fetching Understat {season} season...")
            results = await understat.get_league_results("EPL", season)
            all_matches.append(pd.DataFrame(results))
    return pd.concat(all_matches, ignore_index=True)


def fetch_odds_data(seasons_fd):
    print("\nFetching odds data from football-data.co.uk...")
    all_odds = []
    for season in seasons_fd:
        print(f"...fetching Odds {season} season...")
        all_odds.append(FootballData("ENG Premier League", season).get_fixtures())
    return pd.concat(all_odds, ignore_index=True)


async def create_master_match_list():
    seasons_understat = [2022, 2023, 2024]
    seasons_fd = ["2022-2023", "2023-2024", "2024-2025"]

    df_understat = await fetch_understat_data(seasons_understat)
    df_odds = fetch_odds_data(seasons_fd)

    df_understat['team_home'] = df_understat['h'].apply(lambda x: x['title'])
    df_understat['team_away'] = df_understat['a'].apply(lambda x: x['title'])
    df_understat['xg_home'] = df_understat['xG'].apply(lambda x: x['h'])
    df_understat['xg_away'] = df_understat['xG'].apply(lambda x: x['a'])
    df_understat.rename(columns={'datetime': 'date'}, inplace=True)
    df_understat['date'] = pd.to_datetime(df_understat['date']).dt.date
    understat_cols = ['date', 'team_home', 'team_away', 'xg_home', 'xg_away']
    df_understat = df_understat[understat_cols]

    # odds data
    odds_cols = ["date", "team_home", "team_away", "goals_home", "goals_away", "psh", "psd", "psa"]
    df_odds = df_odds[odds_cols].dropna(subset=['psh', 'psd', 'psa']).copy()
    df_odds['date'] = pd.to_datetime(df_odds['date']).dt.date

    print("\nStandardizing team names and merging data sources...")
    name_mapping = {
        'Manchester City': 'Man City', 'Manchester United': 'Man United', 'Tottenham Hotspur': 'Tottenham',
        'Wolverhampton Wanderers': 'Wolves', 'Sheffield United': 'Sheff Utd', 'Nottingham Forest': "Nott'm Forest",
        'Brighton and Hove Albion': 'Brighton', 'Leeds United': 'Leeds', 'West Ham United': 'West Ham',
        'AFC Bournemouth': 'Bournemouth', 'Luton Town': 'Luton', 'Leicester City': 'Leicester'
    }
    df_understat.replace({'team_home': name_mapping, 'team_away': name_mapping}, inplace=True)

    df_master = pd.merge(df_odds, df_understat, on=['date', 'team_home', 'team_away'], how='inner')
    df_master.sort_values(by='date', inplace=True)
    df_master.reset_index(drop=True, inplace=True)

    # Save the output
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "master_match_list.parquet")
    df_master.to_parquet(output_path)
    print(f"\nData saved successfully to {output_path}")


if __name__ == "__main__":
    asyncio.run(create_master_match_list())