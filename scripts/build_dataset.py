"""
Step 6: Build modeling dataset (one row per game)

Examples:
    python scripts/build_dataset.py --season 2023-24
    python scripts/build_dataset.py --last 10
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.build_dataset import (  # noqa: E402
    LAST_10_SEASONS,
    build_multi_season_dataset,
    build_season_dataset,
    save_dataset,
)
from src.fetch_games import DEFAULT_SEASON_TYPE, season_type_slug  # noqa: E402

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge games, ratings, and rest days into one row per game"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--season", help="Single season, e.g. 2023-24")
    group.add_argument(
        "--last",
        type=int,
        metavar="N",
        help="Build dataset for the last N seasons",
    )
    parser.add_argument(
        "--season-type",
        default=DEFAULT_SEASON_TYPE,
        help='NBA season type (default: "Regular Season")',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slug = season_type_slug(args.season_type)

    if args.season:
        df = build_season_dataset(args.season, RAW_DATA_DIR, args.season_type)
        output_path = PROCESSED_DATA_DIR / f"games_{args.season}_{slug}.csv"
    else:
        seasons = LAST_10_SEASONS[-args.last :]
        print(f"Building for seasons: {', '.join(seasons)}\n")
        df = build_multi_season_dataset(seasons, RAW_DATA_DIR, args.season_type)
        output_path = PROCESSED_DATA_DIR / f"games_last{args.last}_{slug}.csv"

    save_dataset(df, output_path)

    print(f"Rows: {len(df):,} games")
    print(f"Home win rate: {df['home_win'].mean():.1%}")
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
