"""Extended model evaluation metrics and plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from src.train import TrainResult, baseline_home_win_accuracy, feature_importance


@dataclass
class EvalMetrics:
    accuracy: float
    baseline_accuracy: float
    roc_auc: float
    log_loss: float
    brier_score: float

    def as_frame(self, model_name: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "model": model_name,
                    "accuracy": self.accuracy,
                    "baseline_accuracy": self.baseline_accuracy,
                    "roc_auc": self.roc_auc,
                    "log_loss": self.log_loss,
                    "brier_score": self.brier_score,
                }
            ]
        )


def compute_metrics(result: TrainResult) -> EvalMetrics:
    y_test = result.y_test
    y_prob = result.y_prob
    y_pred = result.y_pred

    return EvalMetrics(
        accuracy=accuracy_score(y_test, y_pred),
        baseline_accuracy=baseline_home_win_accuracy(y_test),
        roc_auc=roc_auc_score(y_test, y_prob),
        log_loss=log_loss(y_test, y_prob),
        brier_score=brier_score_loss(y_test, y_prob),
    )


def format_metrics(metrics: EvalMetrics, result: TrainResult, model_name: str) -> str:
    lines = [
        f"=== {model_name} ===",
        f"Split: {result.split_name}",
        f"Accuracy:  {metrics.accuracy:.1%}  (baseline: {metrics.baseline_accuracy:.1%})",
        f"ROC AUC:   {metrics.roc_auc:.3f}",
        f"Log loss:  {metrics.log_loss:.3f}  (lower is better)",
        f"Brier:     {metrics.brier_score:.3f}  (lower is better; measures calibration)",
        "",
        "Top coefficients:",
        feature_importance(result).head(5).to_string(index=False),
    ]
    return "\n".join(lines)


def plot_roc_curve(result: TrainResult, output_path: Path, label: str) -> None:
    fpr, tpr, _ = roc_curve(result.y_test, result.y_prob)
    auc = roc_auc_score(result.y_test, result.y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_calibration_curve(result: TrainResult, output_path: Path, label: str) -> None:
    prob_true, prob_pred = calibration_curve(
        result.y_test,
        result.y_prob,
        n_bins=10,
        strategy="quantile",
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", label=label)
    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Actual home win rate")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_coefficients(result: TrainResult, output_path: Path, title: str) -> None:
    coefs = feature_importance(result)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=coefs,
        y="feature",
        x="coefficient",
        hue="feature",
        legend=False,
        palette="coolwarm",
        ax=ax,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Coefficient (standardized features)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compare_models(results: list[tuple[str, TrainResult]]) -> pd.DataFrame:
    frames = []
    for name, result in results:
        frames.append(compute_metrics(result).as_frame(name))
    return pd.concat(frames, ignore_index=True)
