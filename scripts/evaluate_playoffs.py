"""
Step 10: Compare model performance on regular season vs playoffs

Run:
    source .venv/bin/activate
    python scripts/fetch_playoffs.py --last 10
    python scripts/build_dataset.py --last 10 --season-type Playoffs
    python scripts/evaluate_playoffs.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import compute_metrics, plot_calibration_curve, plot_roc_curve  # noqa: E402
from src.features import FEATURE_COLS, add_derived_features  # noqa: E402
from src.train import TEST_SEASONS, load_dataset, make_model  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REGULAR_DATASET = PROJECT_ROOT / "data" / "processed" / "games_last10_regular_season.csv"
PLAYOFF_DATASET = PROJECT_ROOT / "data" / "processed" / "games_last10_playoffs.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on regular season, compare test performance on playoffs"
    )
    parser.add_argument(
        "--regular-dataset",
        type=Path,
        default=REGULAR_DATASET,
    )
    parser.add_argument(
        "--playoff-dataset",
        type=Path,
        default=PLAYOFF_DATASET,
    )
    return parser.parse_args()


def fit_and_predict(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model = make_model()
    X_train = df_train[FEATURE_COLS]
    y_train = df_train["home_win"]
    X_test = df_test[FEATURE_COLS]
    y_test = df_test["home_win"]

    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)
    y_prob = pd.Series(model.predict_proba(X_test)[:, 1], index=y_test.index)

    from src.train import TrainResult

    return TrainResult(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        split_name="",
        feature_cols=FEATURE_COLS,
    )


def metrics_row(name: str, result, games: int, home_win_rate: float) -> dict:
    m = compute_metrics(result)
    return {
        "dataset": name,
        "games": games,
        "home_win_rate": home_win_rate,
        "accuracy": m.accuracy,
        "roc_auc": m.roc_auc,
        "log_loss": m.log_loss,
        "brier_score": m.brier_score,
    }


def plot_comparison(rows: list[dict], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    long = df.melt(
        id_vars=["dataset"],
        value_vars=["accuracy", "roc_auc"],
        var_name="metric",
        value_name="score",
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=long, x="metric", y="score", hue="dataset", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Regular season vs playoffs (same test seasons)")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.playoff_dataset.exists():
        raise FileNotFoundError(
            "Missing playoff dataset. Run:\n"
            "  python scripts/fetch_playoffs.py --last 10\n"
            "  python scripts/build_dataset.py --last 10 --season-type Playoffs"
        )

    regular = add_derived_features(load_dataset(args.regular_dataset))
    playoffs = add_derived_features(load_dataset(args.playoff_dataset))

    train_reg = regular[~regular["season"].isin(TEST_SEASONS)]
    test_reg = regular[regular["season"].isin(TEST_SEASONS)]
    test_po = playoffs[playoffs["season"].isin(TEST_SEASONS)]

    reg_result = fit_and_predict(train_reg, test_reg)
    po_result = fit_and_predict(train_reg, test_po)

    reg_result.split_name = f"regular season test ({', '.join(sorted(TEST_SEASONS))})"
    po_result.split_name = f"playoffs test ({', '.join(sorted(TEST_SEASONS))})"

    rows = [
        metrics_row(
            "Regular season",
            reg_result,
            len(test_reg),
            test_reg["home_win"].mean(),
        ),
        metrics_row(
            "Playoffs",
            po_result,
            len(test_po),
            test_po["home_win"].mean(),
        ),
    ]
    comparison = pd.DataFrame(rows)

    print("=== Train on regular season (2015-16 → 2022-23) ===")
    print(f"Train games: {len(train_reg):,}\n")
    print("=== Test on 2023-24 + 2024-25 ===")
    print(comparison.round(3).to_string(index=False))
    print()
    print("Notes:")
    print("  • Playoff features use regular-season history for game 1 ratings/rest.")
    print("  • Lower playoff accuracy often means playoffs are harder to predict.")
    print("  • Home win rate usually rises in playoffs (home court matters more).")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_comparison(rows, FIGURES_DIR / "10_regular_vs_playoffs.png")
    plot_roc_curve(reg_result, FIGURES_DIR / "10_roc_regular.png", "Regular season")
    plot_roc_curve(po_result, FIGURES_DIR / "10_roc_playoffs.png", "Playoffs")
    plot_calibration_curve(
        po_result, FIGURES_DIR / "10_calibration_playoffs.png", "Playoffs"
    )

    print(f"\nPlots saved to: {FIGURES_DIR}/")
    print("  • 10_regular_vs_playoffs.png")
    print("  • 10_roc_regular.png")
    print("  • 10_roc_playoffs.png")
    print("  • 10_calibration_playoffs.png")


if __name__ == "__main__":
    main()
