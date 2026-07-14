"""Fetch and save league game logs from the NBA API."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

# Last 10 completed regular seasons (update the final entry each offseason).
LAST_10_SEASONS = [
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

DEFAULT_SEASON_TYPE = "Regular Season"
API_PAUSE_SECONDS = 0.6  # be polite to stats.nba.com


def season_type_slug(season_type: str) -> str:
    return season_type.lower().replace(" ", "_")


def fetch_league_games(
    season: str,
    season_type: str = DEFAULT_SEASON_TYPE,
) -> pd.DataFrame:
    """Return one row per team per game for a single season."""
    response = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
    )
    return response.get_data_frames()[0]


def save_league_games(
    df: pd.DataFrame,
    season: str,
    output_dir: Path,
    season_type: str = DEFAULT_SEASON_TYPE,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"league_games_{season}_{season_type_slug(season_type)}.csv"
    df.to_csv(path, index=False)
    return path


def fetch_and_save_season(
    season: str,
    output_dir: Path,
    season_type: str = DEFAULT_SEASON_TYPE,
) -> Path:
    df = fetch_league_games(season, season_type=season_type)
    path = save_league_games(df, season, output_dir, season_type=season_type)
    games = df["GAME_ID"].nunique()
    print(
        f"  {season}: {len(df):,} rows, {games:,} games → {path.name}"
    )
    return path


def fetch_and_save_seasons(
    seasons: list[str],
    output_dir: Path,
    season_type: str = DEFAULT_SEASON_TYPE,
) -> list[Path]:
    paths: list[Path] = []
    for i, season in enumerate(seasons):
        paths.append(fetch_and_save_season(season, output_dir, season_type))
        if i < len(seasons) - 1:
            time.sleep(API_PAUSE_SECONDS)
    return paths
