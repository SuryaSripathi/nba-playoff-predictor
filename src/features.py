"""Feature columns shared across modeling scripts."""

FEATURE_COLS = [
    "home_rest_days",
    "away_rest_days",
    "home_pregame_off_rating",
    "home_pregame_def_rating",
    "home_pregame_net_rating",
    "away_pregame_off_rating",
    "away_pregame_def_rating",
    "away_pregame_net_rating",
]

DERIVED_FEATURE_COLS = [
    "rest_advantage",
    "net_rating_diff",
]

ITERATION_FEATURE_COLS = FEATURE_COLS + DERIVED_FEATURE_COLS

TARGET_COL = "home_win"


def add_derived_features(df):
    """Add simple diff features for model iteration."""
    out = df.copy()
    out["rest_advantage"] = out["home_rest_days"] - out["away_rest_days"]
    out["net_rating_diff"] = (
        out["home_pregame_net_rating"] - out["away_pregame_net_rating"]
    )
    return out
