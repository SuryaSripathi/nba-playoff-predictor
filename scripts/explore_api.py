"""
Step 2: Hello, NBA API

Run:
    source .venv/bin/activate
    python scripts/explore_api.py

This script fetches one team's game log and a slice of league-wide games,
then prints shapes, columns, and sample rows so we know what we're working with.
"""

from nba_api.stats.endpoints import leaguegamelog, teamgamelog
from nba_api.stats.static import teams

# --- Config (change these to explore) ---
TEAM_ABBREV = "LAL"
SEASON = "2023-24"
SEASON_TYPE = "Regular Season"  # later: "Playoffs"


def main() -> None:
    # 1) Look up a team ID from a human-readable abbreviation
    team = next(t for t in teams.get_teams() if t["abbreviation"] == TEAM_ABBREV)
    print("=== Team lookup ===")
    print(f"{team['full_name']} → id={team['id']}, abbrev={team['abbreviation']}\n")

    # 2) One team, one season: TeamGameLog
    team_log = teamgamelog.TeamGameLog(
        team_id=team["id"],
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
    )
    team_df = team_log.get_data_frames()[0]

    print("=== TeamGameLog (one team, one season) ===")
    print(f"Shape: {team_df.shape[0]} rows × {team_df.shape[1]} columns")
    print(f"Columns: {list(team_df.columns)}")
    print("\nFirst 3 games (most recent first):")
    print(team_df.head(3).to_string(index=False))
    print()

    # 3) All teams, one season: LeagueGameLog (2 rows per game)
    league_log = leaguegamelog.LeagueGameLog(
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
    )
    league_df = league_log.get_data_frames()[0]

    print("=== LeagueGameLog (every team-game in the season) ===")
    print(f"Shape: {league_df.shape[0]} rows × {league_df.shape[1]} columns")
    print(f"Unique games: {league_df['GAME_ID'].nunique()}")
    print(f"Columns: {list(league_df.columns)}")
    print("\nOne game (both teams share the same GAME_ID):")
    sample_game_id = league_df.iloc[0]["GAME_ID"]
    print(league_df[league_df["GAME_ID"] == sample_game_id].to_string(index=False))
    print()

    # 4) Columns we'll care about later
    print("=== Columns useful for our predictor (later steps) ===")
    useful = {
        "GAME_ID": "unique game key — join home/away rows",
        "GAME_DATE": "compute rest days between games",
        "TEAM_ID / TEAM_ABBREVIATION": "identify each side",
        "MATCHUP": "home vs away (contains 'vs.' or '@')",
        "WL": "win/loss label for that team",
        "PTS / PLUS_MINUS": "scoring; margin hints at strength",
    }
    for col, why in useful.items():
        print(f"  • {col}: {why}")
    print()
    print("Not in this endpoint (we'll add in Steps 4–5):")
    print("  • OFF_RATING / DEF_RATING → from advanced box scores or team metrics")
    print("  • REST_DAYS → derived from GAME_DATE per team")


if __name__ == "__main__":
    main()
