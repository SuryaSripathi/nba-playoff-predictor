"""Train and evaluate a baseline logistic regression model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLS, TARGET_COL

DEFAULT_DATASET = Path("data/processed/games_last10_regular_season.csv")
TEST_SEASONS = {"2023-24", "2024-25"}


@dataclass
class TrainResult:
    model: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: pd.Series
    y_prob: pd.Series
    split_name: str
    feature_cols: list[str]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/build_dataset.py --last 10"
        )
    return pd.read_csv(path)


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )


def random_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    X = df[feature_cols]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, "random 80/20 split"


def time_split(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    """Train on older seasons, test on the most recent ones."""
    train_df = df[~df["season"].isin(TEST_SEASONS)]
    test_df = df[df["season"].isin(TEST_SEASONS)]
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]
    seasons = ", ".join(sorted(TEST_SEASONS))
    return X_train, X_test, y_train, y_test, f"time split (test = {seasons})"


def train_and_evaluate(
    df: pd.DataFrame,
    split: str = "random",
    feature_cols: list[str] | None = None,
) -> TrainResult:
    feature_cols = feature_cols or FEATURE_COLS
    if split == "time":
        X_train, X_test, y_train, y_test, split_name = time_split(df, feature_cols)
    else:
        X_train, X_test, y_train, y_test, split_name = random_split(df, feature_cols)

    model = make_model()
    model.fit(X_train, y_train)

    y_pred = pd.Series(model.predict(X_test), index=y_test.index, name="y_pred")
    y_prob = pd.Series(model.predict_proba(X_test)[:, 1], index=y_test.index, name="y_prob")

    return TrainResult(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        split_name=split_name,
        feature_cols=feature_cols,
    )


def baseline_home_win_accuracy(y_test: pd.Series) -> float:
    """Always predict home team wins."""
    return accuracy_score(y_test, pd.Series(1, index=y_test.index))


def feature_importance(result: TrainResult) -> pd.DataFrame:
    clf: LogisticRegression = result.model.named_steps["clf"]
    return (
        pd.DataFrame(
            {
                "feature": result.feature_cols,
                "coefficient": clf.coef_[0],
            }
        )
        .assign(abs_coeff=lambda d: d["coefficient"].abs())
        .sort_values("abs_coeff", ascending=False)
        .drop(columns="abs_coeff")
        .reset_index(drop=True)
    )


def format_results(result: TrainResult) -> str:
    acc = accuracy_score(result.y_test, result.y_pred)
    base = baseline_home_win_accuracy(result.y_test)
    cm = confusion_matrix(result.y_test, result.y_pred)
    report = classification_report(result.y_test, result.y_pred, digits=3)
    coefs = feature_importance(result).to_string(index=False)

    lines = [
        f"Split: {result.split_name}",
        f"Train games: {len(result.y_train):,}  |  Test games: {len(result.y_test):,}",
        "",
        f"Model accuracy:     {acc:.1%}",
        f"Baseline (always home wins): {base:.1%}",
        "",
        "Confusion matrix [[TN, FP], [FN, TP]]:",
        str(cm),
        "",
        "Classification report:",
        report,
        "Feature coefficients (standardized inputs):",
        coefs,
    ]
    return "\n".join(lines)
