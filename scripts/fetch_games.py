"""
Step 3: Fetch historical game data

Examples:
    # One season (good first run)
    python scripts/fetch_games.py --season 2023-24

    # Last 10 regular seasons
    python scripts/fetch_games.py --last 10
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fetch_games import (  # noqa: E402
    DEFAULT_SEASON_TYPE,
    LAST_10_SEASONS,
    fetch_and_save_season,
    fetch_and_save_seasons,
)

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LeagueGameLog CSVs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--season", help="Single season, e.g. 2023-24")
    group.add_argument(
        "--last",
        type=int,
        metavar="N",
        help="Fetch the last N seasons from our predefined list",
    )
    parser.add_argument(
        "--season-type",
        default=DEFAULT_SEASON_TYPE,
        help='NBA season type (default: "Regular Season")',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Saving to: {RAW_DATA_DIR}\n")

    if args.season:
        fetch_and_save_season(args.season, RAW_DATA_DIR, args.season_type)
    else:
        seasons = LAST_10_SEASONS[-args.last :]
        print(f"Fetching {len(seasons)} seasons: {', '.join(seasons)}\n")
        fetch_and_save_seasons(seasons, RAW_DATA_DIR, args.season_type)

    print("\nDone.")


if __name__ == "__main__":
    main()
