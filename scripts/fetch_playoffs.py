"""
Step 10 helper: fetch playoff game logs and ratings for the last N seasons.

Run:
    source .venv/bin/activate
    python scripts/fetch_playoffs.py --last 10
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fetch_games import DEFAULT_SEASON_TYPE, LAST_10_SEASONS, fetch_and_save_seasons  # noqa: E402
from src.fetch_ratings import fetch_and_save_seasons as fetch_rating_seasons  # noqa: E402

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PLAYOFFS = "Playoffs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download playoff games and ratings")
    parser.add_argument(
        "--last",
        type=int,
        default=10,
        metavar="N",
        help="Fetch the last N seasons (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seasons = LAST_10_SEASONS[-args.last :]
    print(f"Fetching playoff data for: {', '.join(seasons)}\n")

    print("Games:")
    fetch_and_save_seasons(seasons, RAW_DATA_DIR, PLAYOFFS)
    print("\nRatings:")
    fetch_rating_seasons(seasons, RAW_DATA_DIR, PLAYOFFS)
    print("\nDone.")


if __name__ == "__main__":
    main()
