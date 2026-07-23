"""Build one-row-per-game modeling datasets from raw CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.fetch_games import DEFAULT_SEASON_TYPE, LAST_10_SEASONS, season_type_slug
from src.ratings import add_pregame_rolling_ratings
from src.rest_days import add_rest_days

PLAYOFF_SEASON_TYPE = "Playoffs"

PREGAME_RATING_COLS = [
    "pregame_off_rating",
    "pregame_def_rating",
    "pregame_net_rating",
]

HOME_PREFIX = "home_"
AWAY_PREFIX = "away_"


def _raw_paths(raw_dir: Path, season: str, season_type: str) -> tuple[Path, Path]:
    slug = season_type_slug(season_type)
    games_path = raw_dir / f"league_games_{season}_{slug}.csv"
    ratings_path = raw_dir / f"league_ratings_{season}_{slug}.csv"
    return games_path, ratings_path


def _is_home_matchup(matchup: str) -> bool:
    return " vs. " in matchup


def _enrich_team_games(games: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Merge rest days and pre-game ratings onto each team-game row."""
    ratings = add_pregame_rolling_ratings(ratings)
    games = add_rest_days(games)

    rating_features = ["GAME_ID", "TEAM_ID", *PREGAME_RATING_COLS]
    merged = games.merge(
        ratings[rating_features],
        on=["GAME_ID", "TEAM_ID"],
        how="inner",
    )
    merged["GAME_DATE"] = pd.to_datetime(merged["GAME_DATE"])
    return merged


def _to_game_level(team_games: pd.DataFrame, season: str) -> pd.DataFrame:
    """Convert two team-game rows into one modeling row per game."""
    home = team_games[team_games["MATCHUP"].map(_is_home_matchup)].copy()
    away = team_games[~team_games["MATCHUP"].map(_is_home_matchup)].copy()

    home = home.rename(
        columns={
            "TEAM_ABBREVIATION": "home_team",
            "rest_days": "home_rest_days",
            "is_back_to_back": "home_is_back_to_back",
            **{col: f"{HOME_PREFIX}{col}" for col in PREGAME_RATING_COLS},
        }
    )
    away = away.rename(
        columns={
            "TEAM_ABBREVIATION": "away_team",
            "rest_days": "away_rest_days",
            "is_back_to_back": "away_is_back_to_back",
            **{col: f"{AWAY_PREFIX}{col}" for col in PREGAME_RATING_COLS},
        }
    )

    home_cols = [
        "GAME_ID",
        "GAME_DATE",
        "home_team",
        "home_rest_days",
        "home_is_back_to_back",
        *[f"{HOME_PREFIX}{col}" for col in PREGAME_RATING_COLS],
        "WL",
    ]
    away_cols = [
        "GAME_ID",
        "away_team",
        "away_rest_days",
        "away_is_back_to_back",
        *[f"{AWAY_PREFIX}{col}" for col in PREGAME_RATING_COLS],
    ]

    games = home[home_cols].merge(away[away_cols], on="GAME_ID", how="inner")
    games["season"] = season
    games["home_win"] = (games["WL"] == "W").astype(int)
    games = games.drop(columns=["WL"])

    feature_cols = [
        "home_rest_days",
        "away_rest_days",
        *[f"{HOME_PREFIX}{col}" for col in PREGAME_RATING_COLS],
        *[f"{AWAY_PREFIX}{col}" for col in PREGAME_RATING_COLS],
    ]
    games = games.dropna(subset=feature_cols)

    col_order = [
        "season",
        "GAME_ID",
        "GAME_DATE",
        "home_team",
        "away_team",
        "home_win",
        *feature_cols,
        "home_is_back_to_back",
        "away_is_back_to_back",
    ]
    return games[col_order].sort_values("GAME_DATE").reset_index(drop=True)


def build_playoff_season_dataset(
    season: str,
    raw_dir: Path,
) -> pd.DataFrame:
    """
    Build playoff games using regular-season history for pre-game features.

    Playoff game 1 still needs a team's regular-season stats for ratings/rest.
    """
    reg_games_path, reg_ratings_path = _raw_paths(raw_dir, season, DEFAULT_SEASON_TYPE)
    po_games_path, po_ratings_path = _raw_paths(raw_dir, season, PLAYOFF_SEASON_TYPE)

    for path in (reg_games_path, reg_ratings_path, po_games_path, po_ratings_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    reg_games = pd.read_csv(reg_games_path)
    po_games = pd.read_csv(po_games_path)
    reg_ratings = pd.read_csv(reg_ratings_path)
    po_ratings = pd.read_csv(po_ratings_path)

    all_games = pd.concat([reg_games, po_games], ignore_index=True)
    all_ratings = pd.concat([reg_ratings, po_ratings], ignore_index=True)
    enriched = _enrich_team_games(all_games, all_ratings)

    playoff_game_ids = po_games["GAME_ID"].unique()
    playoff_team_games = enriched[enriched["GAME_ID"].isin(playoff_game_ids)]
    return _to_game_level(playoff_team_games, season)


def build_season_dataset(
    season: str,
    raw_dir: Path,
    season_type: str = DEFAULT_SEASON_TYPE,
) -> pd.DataFrame:
    if season_type == PLAYOFF_SEASON_TYPE:
        return build_playoff_season_dataset(season, raw_dir)

    games_path, ratings_path = _raw_paths(raw_dir, season, season_type)
    if not games_path.exists():
        raise FileNotFoundError(f"Missing games file: {games_path}")
    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing ratings file: {ratings_path}")

    games = pd.read_csv(games_path)
    ratings = pd.read_csv(ratings_path)
    team_games = _enrich_team_games(games, ratings)
    return _to_game_level(team_games, season)


def build_multi_season_dataset(
    seasons: list[str],
    raw_dir: Path,
    season_type: str = DEFAULT_SEASON_TYPE,
) -> pd.DataFrame:
    frames = [build_season_dataset(season, raw_dir, season_type) for season in seasons]
    return pd.concat(frames, ignore_index=True)


def save_dataset(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


__all__ = [
    "LAST_10_SEASONS",
    "PLAYOFF_SEASON_TYPE",
    "build_multi_season_dataset",
    "build_playoff_season_dataset",
    "build_season_dataset",
    "save_dataset",
]
