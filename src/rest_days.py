"""Compute rest days between games for each team."""

from __future__ import annotations

import pandas as pd


def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rest_days: full days off since the team's previous game.

    Examples:
        played yesterday → rest_days = 0 (back-to-back)
        played 2 days ago → rest_days = 1 (one day off in between)

    First game of the season for each team gets NaN.
    """
    out = df.copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"])
    out = out.sort_values(["TEAM_ID", "GAME_DATE"])

    prior_date = out.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    out["rest_days"] = (out["GAME_DATE"] - prior_date).dt.days - 1
    out["is_back_to_back"] = out["rest_days"] == 0

    return out
