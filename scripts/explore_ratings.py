"""
Step 4: Explore offensive / defensive ratings

Run:
    source .venv/bin/activate
    python scripts/explore_ratings.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fetch_ratings import fetch_team_ratings  # noqa: E402
from src.ratings import add_pregame_rolling_ratings  # noqa: E402

SEASON = "2023-24"
TEAM_ABBREV = "LAL"


def main() -> None:
    print("=== Fetching advanced team game logs ===")
    print('Endpoint: TeamGameLogs with MeasureType="Advanced"\n')

    df = fetch_team_ratings(SEASON)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}\n")

    sample = df[df["TEAM_ABBREVIATION"] == TEAM_ABBREV].copy()
    sample["GAME_DATE"] = pd.to_datetime(sample["GAME_DATE"])
    sample = sample.sort_values("GAME_DATE")

    print(f"=== {TEAM_ABBREV} first 5 games (in-game ratings) ===")
    print(
        sample.head(5)[
            ["GAME_DATE", "MATCHUP", "WL", "OFF_RATING", "DEF_RATING", "NET_RATING"]
        ].to_string(index=False)
    )
    print()

    print("=== Pre-game rolling ratings (what we'll use for prediction) ===")
    print("Each value is the team's average from *prior* games only.\n")
    with_pregame = add_pregame_rolling_ratings(df)
    lal = with_pregame[with_pregame["TEAM_ABBREVIATION"] == TEAM_ABBREV].copy()
    lal = lal.sort_values("GAME_DATE")
    print(
        lal.head(5)[
            [
                "GAME_DATE",
                "OFF_RATING",
                "pregame_off_rating",
                "DEF_RATING",
                "pregame_def_rating",
            ]
        ].to_string(index=False)
    )
    print()
    print("Note: first game of the season has NaN pregame ratings (no history yet).")
    print("We will handle that when building the full dataset in Step 6.")


if __name__ == "__main__":
    main()
