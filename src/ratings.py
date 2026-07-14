"""Compute team rating features for modeling."""

from __future__ import annotations

import pandas as pd

RATING_COLS = ["OFF_RATING", "DEF_RATING", "NET_RATING"]


def add_pregame_rolling_ratings(
    df: pd.DataFrame,
    window: int | None = None,
) -> pd.DataFrame:
    """
    Add season-to-date average ratings *before* each game.

    Uses only prior games for the same team (shift(1)) so we don't leak
    the current game's outcome into the features.
    """
    out = df.copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"])
    out = out.sort_values(["TEAM_ID", "GAME_DATE"])

    grouped = out.groupby("TEAM_ID", group_keys=False)

    for col in RATING_COLS:
        prior = grouped[col].shift(1)
        if window is None:
            out[f"pregame_{col.lower()}"] = prior.groupby(out["TEAM_ID"]).expanding().mean().reset_index(level=0, drop=True)
        else:
            out[f"pregame_{col.lower()}"] = prior.groupby(out["TEAM_ID"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)

    return out
