"""
Step 8: Train baseline logistic regression

Run:
    source .venv/bin/activate
    python scripts/train_model.py
    python scripts/train_model.py --split time
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import DEFAULT_DATASET, format_results, load_dataset, train_and_evaluate  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline logistic regression")
    parser.add_argument(
        "--split",
        choices=["random", "time"],
        default="random",
        help="random = 80/20 stratified; time = train on 2015-2022, test on 2023-25",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_DATASET,
        help="Path to processed games CSV",
    )
    return parser.parse_args()


def save_confusion_matrix(y_test, y_pred) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred away", "Pred home"],
        yticklabels=["Actual away", "Actual home"],
        ax=ax,
    )
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "06_confusion_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_dataset(args.dataset)
    result = train_and_evaluate(df, split=args.split)

    print(format_results(result))
    save_confusion_matrix(result.y_test, result.y_pred)
    print(f"\nConfusion matrix plot → {FIGURES_DIR / '06_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
