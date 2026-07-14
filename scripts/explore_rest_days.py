"""
Step 5: Explore rest days between games

Run:
    source .venv/bin/activate
    python scripts/explore_rest_days.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rest_days import add_rest_days  # noqa: E402

SEASON = "2023-24"
TEAM_ABBREV = "LAL"
GAMES_CSV = PROJECT_ROOT / "data" / "raw" / f"league_games_{SEASON}_regular_season.csv"


def main() -> None:
    if not GAMES_CSV.exists():
        print(f"Missing {GAMES_CSV.name}. Run: python scripts/fetch_games.py --season {SEASON}")
        return

    df = pd.read_csv(GAMES_CSV)
    with_rest = add_rest_days(df)

    team = with_rest[with_rest["TEAM_ABBREVIATION"] == TEAM_ABBREV].sort_values("GAME_DATE")

    print("=== Rest days formula ===")
    print("rest_days = (current_game_date - previous_game_date).days - 1\n")

    print(f"=== {TEAM_ABBREV} first 5 games ===")
    print(
        team.head(5)[["GAME_DATE", "MATCHUP", "WL", "rest_days", "is_back_to_back"]].to_string(
            index=False
        )
    )
    print()

    b2b = team[team["is_back_to_back"]]
    print(f"=== {TEAM_ABBREV} back-to-backs in {SEASON}: {len(b2b)} games ===")
    print(
        b2b.head(5)[["GAME_DATE", "MATCHUP", "WL", "rest_days"]].to_string(index=False)
    )
    print()

    print("=== League-wide rest day distribution ===")
    counts = with_rest["rest_days"].dropna().value_counts().sort_index()
    for days, n in counts.head(8).items():
        label = "back-to-back" if days == 0 else f"{int(days)} day(s) rest"
        print(f"  {int(days):2d} rest days: {n:4d} team-games  ({label})")


if __name__ == "__main__":
    main()
