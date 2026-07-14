"""Fetch and save per-game advanced team ratings from the NBA API."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import teamgamelogs

from src.fetch_games import (
    API_PAUSE_SECONDS,
    DEFAULT_SEASON_TYPE,
    LAST_10_SEASONS,
    season_type_slug,
)

ADVANCED_MEASURE_TYPE = "Advanced"

# Columns we care about for modeling (+ ids for joining)
RATING_COLUMNS = [
    "SEASON_YEAR",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "E_OFF_RATING",
    "E_DEF_RATING",
    "E_NET_RATING",
    "PACE",
    "PIE",
]


def fetch_team_ratings(
    season: str,
    season_type: str = DEFAULT_SEASON_TYPE,
) -> pd.DataFrame:
    """Return one row per team per game with advanced rating stats."""
    response = teamgamelogs.TeamGameLogs(
        season_nullable=season,
        season_type_nullable=season_type,
        measure_type_player_game_logs_nullable=ADVANCED_MEASURE_TYPE,
    )
    df = response.get_data_frames()[0]
    keep = [c for c in RATING_COLUMNS if c in df.columns]
    return df[keep]


def save_team_ratings(
    df: pd.DataFrame,
    season: str,
    output_dir: Path,
    season_type: str = DEFAULT_SEASON_TYPE,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"league_ratings_{season}_{season_type_slug(season_type)}.csv"
    df.to_csv(path, index=False)
    return path


def fetch_and_save_season(
    season: str,
    output_dir: Path,
    season_type: str = DEFAULT_SEASON_TYPE,
) -> Path:
    df = fetch_team_ratings(season, season_type=season_type)
    path = save_team_ratings(df, season, output_dir, season_type=season_type)
    games = df["GAME_ID"].nunique()
    print(f"  {season}: {len(df):,} rows, {games:,} games → {path.name}")
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


__all__ = [
    "ADVANCED_MEASURE_TYPE",
    "LAST_10_SEASONS",
    "fetch_and_save_season",
    "fetch_and_save_seasons",
    "fetch_team_ratings",
]
