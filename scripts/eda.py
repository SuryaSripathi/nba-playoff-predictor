"""
Step 7: Exploratory data analysis with Seaborn

Run:
    source .venv/bin/activate
    python scripts/eda.py

Reads the 10-season processed dataset and saves plots to reports/figures/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import FEATURE_COLS  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "games_last10_regular_season.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATASET_PATH.name}. Run: python scripts/build_dataset.py --last 10"
        )
    df = pd.read_csv(DATASET_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("=== Dataset overview ===")
    print(f"Games: {len(df):,}")
    print(f"Seasons: {df['season'].nunique()} ({df['season'].min()} → {df['season'].max()})")
    print(f"Home win rate: {df['home_win'].mean():.1%}")
    print(f"Missing values: {df[FEATURE_COLS + ['home_win']].isna().sum().sum()}")
    print()

    print("=== Feature ranges (sanity check) ===")
    print(df[FEATURE_COLS].describe().round(1).to_string())
    print()

    b2b = df.groupby("home_is_back_to_back")["home_win"].mean()
    print("=== Home win rate by home back-to-back ===")
    print(f"  Not B2B: {b2b.get(False, 0):.1%}")
    print(f"  B2B:     {b2b.get(True, 0):.1%}")
    print()

    rest_adv = df.copy()
    rest_adv["rest_advantage"] = rest_adv["home_rest_days"] - rest_adv["away_rest_days"]
    rest_bins = pd.cut(rest_adv["rest_advantage"], bins=[-10, -1, 0, 1, 10], labels=["away +2+", "away +1", "even", "home +1+"])
    print("=== Home win rate by rest advantage ===")
    print(rest_adv.groupby(rest_bins, observed=True)["home_win"].mean().round(3).to_string())
    print()


def plot_home_win_rate(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    rates = df["home_win"].value_counts(normalize=True).sort_index()
    sns.barplot(
        x=["Away win", "Home win"],
        y=[rates.get(0, 0), rates.get(1, 0)],
        hue=["Away win", "Home win"],
        legend=False,
        ax=ax,
        palette=["#e76f51", "#2a9d8f"],
    )
    ax.set_ylabel("Proportion of games")
    ax.set_title("Home vs away win rate (10 seasons)")
    ax.set_ylim(0, 0.7)
    for i, v in enumerate([rates.get(0, 0), rates.get(1, 0)]):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_home_win_rate.png", dpi=150)
    plt.close(fig)


def plot_net_rating_distributions(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    long = df.melt(
        value_vars=["home_pregame_net_rating", "away_pregame_net_rating"],
        var_name="side",
        value_name="pregame_net_rating",
    )
    long["side"] = long["side"].map(
        {
            "home_pregame_net_rating": "Home team",
            "away_pregame_net_rating": "Away team",
        }
    )
    sns.kdeplot(data=long, x="pregame_net_rating", hue="side", fill=True, alpha=0.35, ax=ax)
    ax.set_xlabel("Pre-game net rating")
    ax.set_title("Distribution of pre-game net ratings")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_net_rating_distribution.png", dpi=150)
    plt.close(fig)


def plot_rest_days(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    rest_long = df.melt(
        value_vars=["home_rest_days", "away_rest_days"],
        var_name="side",
        value_name="rest_days",
    )
    rest_long["side"] = rest_long["side"].map(
        {"home_rest_days": "Home", "away_rest_days": "Away"}
    )
    sns.countplot(data=rest_long, x="rest_days", hue="side", ax=axes[0])
    axes[0].set_title("Rest days distribution")
    axes[0].set_xlabel("Rest days")

    b2b = (
        df.groupby(["home_is_back_to_back", "away_is_back_to_back"])["home_win"]
        .mean()
        .reset_index()
    )
    b2b["scenario"] = b2b.apply(
        lambda r: f"Home B2B={r['home_is_back_to_back']}, Away B2B={r['away_is_back_to_back']}",
        axis=1,
    )
    sns.barplot(data=b2b, x="scenario", y="home_win", hue="scenario", legend=False, ax=axes[1], palette="muted")
    axes[1].set_title("Home win rate by B2B scenario")
    axes[1].set_ylabel("Home win rate")
    axes[1].set_ylim(0, 0.8)
    axes[1].tick_params(axis="x", rotation=20)
    for i, row in b2b.iterrows():
        axes[1].text(i, row["home_win"] + 0.02, f"{row['home_win']:.1%}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_rest_days_and_b2b.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[FEATURE_COLS + ["home_win"]].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        ax=ax,
    )
    ax.set_title("Feature correlation with home_win")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_net_rating_vs_win(df: pd.DataFrame) -> None:
    df = df.copy()
    df["rating_diff"] = df["home_pregame_net_rating"] - df["away_pregame_net_rating"]
    df["rating_edge"] = pd.cut(
        df["rating_diff"],
        bins=[-50, -5, 0, 5, 50],
        labels=["Away +5+", "Away 0–5", "Home 0–5", "Home +5+"],
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    edge_rates = df.groupby("rating_edge", observed=True)["home_win"].mean().reset_index()
    sns.barplot(data=edge_rates, x="rating_edge", y="home_win", hue="rating_edge", legend=False, palette="viridis", ax=ax)
    ax.set_ylabel("Home win rate")
    ax.set_xlabel("Pre-game net rating edge (home − away)")
    ax.set_title("Home win rate by team strength edge")
    ax.set_ylim(0, 0.85)
    for i, row in edge_rates.iterrows():
        ax.text(i, row["home_win"] + 0.02, f"{row['home_win']:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_win_rate_by_rating_edge.png", dpi=150)
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    print_summary(df)

    print("Saving plots...")
    plot_home_win_rate(df)
    plot_net_rating_distributions(df)
    plot_rest_days(df)
    plot_correlation_heatmap(df)
    plot_net_rating_vs_win(df)

    print(f"\nDone. Figures saved to: {FIGURES_DIR}/")
    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  • {path.name}")


if __name__ == "__main__":
    main()
