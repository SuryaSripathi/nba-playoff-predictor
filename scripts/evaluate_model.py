"""
Step 9: Evaluate model quality and iterate

Run:
    source .venv/bin/activate
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --split random
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import (  # noqa: E402
    compare_models,
    compute_metrics,
    format_metrics,
    plot_calibration_curve,
    plot_feature_coefficients,
    plot_roc_curve,
)
from src.features import FEATURE_COLS, ITERATION_FEATURE_COLS, add_derived_features  # noqa: E402
from src.train import DEFAULT_DATASET, load_dataset, train_and_evaluate  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and compare logistic regression models")
    parser.add_argument(
        "--split",
        choices=["random", "time"],
        default="time",
        help="Train/test split strategy (default: time)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_DATASET,
        help="Path to processed games CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = add_derived_features(load_dataset(args.dataset))

    baseline = train_and_evaluate(df, split=args.split, feature_cols=FEATURE_COLS)
    improved = train_and_evaluate(df, split=args.split, feature_cols=ITERATION_FEATURE_COLS)

    baseline_metrics = compute_metrics(baseline)
    improved_metrics = compute_metrics(improved)

    print(format_metrics(baseline_metrics, baseline, "Baseline model (8 features)"))
    print()
    print(format_metrics(improved_metrics, improved, "Iteration model (+ rest/rating diffs)"))
    print()
    print("=== Model comparison ===")
    comparison = compare_models(
        [
            ("baseline", baseline),
            ("iteration", improved),
        ]
    )
    print(comparison.round(3).to_string(index=False))
    print()
    print("Lower log loss / Brier = better probability estimates.")
    print("ROC AUC measures ranking ability (can the model separate home wins from losses?).")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc_curve(baseline, FIGURES_DIR / "07_roc_curve.png", "Baseline")
    plot_calibration_curve(
        baseline, FIGURES_DIR / "08_calibration_curve.png", "Baseline"
    )
    plot_feature_coefficients(
        improved,
        FIGURES_DIR / "09_feature_coefficients.png",
        "Iteration model coefficients",
    )

    print(f"\nPlots saved to: {FIGURES_DIR}/")
    print("  • 07_roc_curve.png")
    print("  • 08_calibration_curve.png")
    print("  • 09_feature_coefficients.png")


if __name__ == "__main__":
    main()
