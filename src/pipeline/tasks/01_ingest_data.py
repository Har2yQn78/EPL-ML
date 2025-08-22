import pandas as pd
from penaltyblog.scrapers import FootballData
from understat import Understat
import asyncio
import aiohttp
import os
from sqlalchemy import text, inspect
from src.core.database import engine


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


async def run_ingestion():
    print("Starting data ingestion...")

    latest_date = None
    table_exists = inspect(engine).has_table('matches')

    if table_exists:
        with engine.connect() as connection:
            latest_date_result = connection.execute(text("SELECT MAX(date) FROM matches;")).scalar()
            if latest_date_result:
                latest_date = pd.to_datetime(latest_date_result)
                print(f"Database contains data up to: {latest_date.date()}")

    if latest_date:
        print("Performing incremental update...")
        seasons_understat = [2025]
        seasons_fd = ["2025-2026"]
    else:
        print("No existing data found. Performing initial bulk load...")
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

    odds_cols = ["date", "team_home", "team_away", "goals_home", "goals_away", "psh", "psd", "psa"]
    df_odds = df_odds[odds_cols].dropna(subset=['psh', 'psd', 'psa']).copy()
    df_odds['date'] = pd.to_datetime(df_odds['date']).dt.date

    print("\nStandardizing and merging...")
    name_mapping = {
        'Manchester City': 'Man City', 'Manchester United': 'Man United', 'Tottenham Hotspur': 'Tottenham',
        'Wolverhampton Wanderers': 'Wolves', 'Sheffield United': 'Sheff Utd', 'Nottingham Forest': "Nott'm Forest",
        'Brighton and Hove Albion': 'Brighton', 'Leeds United': 'Leeds', 'West Ham United': 'West Ham',
        'AFC Bournemouth': 'Bournemouth', 'Luton Town': 'Luton', 'Leicester City': 'Leicester'
    }
    df_understat.replace({'team_home': name_mapping, 'team_away': name_mapping}, inplace=True)

    df_master = pd.merge(df_odds, df_understat, on=['date', 'team_home', 'team_away'], how='inner')

    if latest_date:
        df_master['date_dt'] = pd.to_datetime(df_master['date'])
        df_master = df_master[df_master['date_dt'] > latest_date].drop(columns=['date_dt'])
        print(f"Found {len(df_master)} new matches to add.")
        if df_master.empty:
            print("No new matches found. Pipeline finished.")
            return

    df_master.sort_values(by='date', inplace=True)
    df_master.reset_index(drop=True, inplace=True)

    # Save to Database
    write_method = 'append' if latest_date else 'replace'
    print(f"Saving {len(df_master)} rows to PostgreSQL using '{write_method}' method...")

    df_master.to_sql('matches', con=engine, if_exists=write_method, index=False)
    print("Data saved successfully.")


if __name__ == "__main__":
    asyncio.run(run_ingestion())