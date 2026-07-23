"""
train_models.py

Benchmark multiple Machine Learning algorithms for
Alert Filtering Classifier (AFC)

Pipeline

Parquet
    ↓
Load Dataset
    ↓
Preprocessing
    ↓
Benchmark Multiple Models
    ↓
Select Best Model
    ↓
Save Model
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm

from pandas.api.types import is_numeric_dtype

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, clone


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("AFC")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ============================================================
# Constants
# ============================================================

LABEL_COLUMN = "Label"

EXPECTED_FEATURE_COUNT = 22
ATTACK_LABEL = "ATTACK"
BENIGN_LABEL = "BENIGN"

DROP_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
}


# ============================================================
# Config
# ============================================================

@dataclass(slots=True)
class TrainConfig:
    data_dir: Path
    model_dir: Path

    test_size: float = 0.2
    random_state: int = 42

    sample_frac: float = 0.2

    max_rows_per_file: int | None = None


# ============================================================
# Utilities
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dataframe column names.
    """

    df = df.copy()

    df.columns = [
        str(col)
        .strip()
        .replace("\n", " ")
        for col in df.columns
    ]

    return df


def iter_parquet_files(data_dir: Path) -> list[Path]:

    parquet_files = sorted(
        data_dir.glob("*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {data_dir}"
        )

    return parquet_files


# ============================================================
# Read Dataset
# ============================================================

def read_one_parquet(
    path: Path,
    config: TrainConfig,
) -> pd.DataFrame:

    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except ImportError as exc:
        raise RuntimeError(
            f"Unable to read {path.name}. Install `pyarrow` or `fastparquet` for parquet support. "
            "Run `pip install pyarrow` or `pip install fastparquet` in the current environment."
        ) from exc

    df = normalize_columns(df)

    if config.sample_frac < 1.0:

        df = df.sample(
            frac=config.sample_frac,
            random_state=config.random_state,
        )

    if (
        config.max_rows_per_file is not None
        and
        len(df) > config.max_rows_per_file
    ):

        df = df.sample(
            n=config.max_rows_per_file,
            random_state=config.random_state,
        )

    logger.info(
        "%s -> %,d rows",
        path.name,
        len(df),
    )

    return df


def load_dataset(
    config: TrainConfig,
) -> pd.DataFrame:

    frames = []

    for file in tqdm(
        iter_parquet_files(config.data_dir),
        desc="Loading Parquet",
    ):

        frames.append(
            read_one_parquet(
                file,
                config,
            )
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    data = normalize_columns(data)

    if LABEL_COLUMN not in data.columns:

        raise KeyError(
            "Label column not found."
        )

    logger.info(
        "Dataset Shape: %s",
        data.shape,
    )

    return data


# ============================================================
# Prepare Features
# ============================================================

def prepare_dataset(
    data: pd.DataFrame,
):

    y = data[LABEL_COLUMN].astype(str).str.strip()
    y = y.where(y.str.upper().eq(BENIGN_LABEL), ATTACK_LABEL)

    X = data.drop(
        columns=[LABEL_COLUMN]
    )

    X = X.drop(
        columns=[
            c
            for c in DROP_COLUMNS
            if c in X.columns
        ],
        errors="ignore",
    )

    for col in X.columns:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    feature_names = list(X.columns)

    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} Lightweight Ontology features, got {len(feature_names)}."
        )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                feature_names,
            )
        ],
        remainder="drop",
    )

    logger.info(
        "Features: %d",
        len(feature_names),
    )

    return (
        X,
        y,
        preprocess,
        feature_names,
    )


# ============================================================
# Train Test Split
# ============================================================

def split_dataset(
    X,
    y,
    config: TrainConfig,
):

    stratify = (
        y
        if y.value_counts().min() >= 2
        else None
    )

    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )
# ============================================================
# Machine Learning Models
# ============================================================

MODELS = {
    "Gaussian NB": GaussianNB(),

    
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced",
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        random_state=42,
    ),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        random_state=42,
    ),

  "Logistic Regression": LogisticRegression(
        solver="saga",
        max_iter=500,
        random_state=42,
    ),


    "Linear SVM": LinearSVC(
        random_state=42,
        dual=False,
        max_iter=3000,
    ),
}
# ============================================================
# Build Pipeline
# ============================================================

def build_pipeline(
    preprocess,
    estimator,
):

    return Pipeline(
        steps=[
            (
                "preprocess",
                preprocess,
            ),
            (
                "classifier",
                estimator,
            ),
        ]
    )
# ============================================================
# Evaluation
# ============================================================

def evaluate_model(

    model,

    X_test,

    y_test,

):

    start = time.perf_counter()

    y_pred = model.predict(X_test)

    predict_time = (
        time.perf_counter() - start
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=ATTACK_LABEL,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=ATTACK_LABEL,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=ATTACK_LABEL,
        zero_division=0,
    )

    roc_auc = np.nan

    if hasattr(model, "predict_proba"):

        try:

            proba = model.predict_proba(
                X_test
            )

            if len(model.classes_) == 2:

                attack_index = list(model.classes_).index(ATTACK_LABEL)
                roc_auc = roc_auc_score(
                    y_test.eq(ATTACK_LABEL).astype(int),
                    proba[:, attack_index],
                )

        except Exception:

            pass

    return {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC_AUC": roc_auc,

        "Prediction_Time": predict_time,

    }
# ============================================================
# Train One Model
# ============================================================

def train_one_model(

    name,

    estimator,

    preprocess,

    X_train,

    X_test,

    y_train,

    y_test,

):

    logger.info(
        "Training %s",
        name,
    )

    pipeline = build_pipeline(

        preprocess,

        clone(estimator),

    )

    start = time.perf_counter()

    pipeline.fit(

        X_train,

        y_train,

    )

    train_time = (
        time.perf_counter() - start
    )

    metrics = evaluate_model(

        pipeline,

        X_test,

        y_test,

    )

    metrics["Train_Time"] = train_time

    metrics["Model"] = name

    logger.info(

        "%s | F1 = %.4f",

        name,

        metrics["F1"],

    )

    return pipeline, metrics
# ============================================================
# Feature Importance
# ============================================================

def extract_feature_importance(

    pipeline,

    feature_names,

):

    estimator = pipeline.named_steps[
        "classifier"
    ]

    if not hasattr(
        estimator,
        "feature_importances_",
    ):

        return None

    importance = pd.DataFrame(

        {

            "Feature": feature_names,

            "Importance": estimator.feature_importances_,

        }

    )

    return importance.sort_values(

        "Importance",

        ascending=False,

    )
# ============================================================
# Benchmark All Models
# ============================================================

def benchmark_models(
    preprocess,
    feature_names,
    X_train,
    X_test,
    y_train,
    y_test,
    config: TrainConfig,
):
    """
    Train all models and select the best one based on F1-score.
    """

    results = []

    trained_models = {}

    best_model = None
    best_name = None
    best_score = -1.0

    importance_df = None

    logger.info("=" * 60)
    logger.info("Benchmarking %d models...", len(MODELS))
    logger.info("=" * 60)

    for name, estimator in tqdm(
        MODELS.items(),
        desc="Training Models",
    ):

        pipeline, metrics = train_one_model(
            name=name,
            estimator=estimator,
            preprocess=preprocess,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        trained_models[name] = pipeline

        results.append(metrics)

        if metrics["F1"] > best_score:

            best_score = metrics["F1"]

            best_model = pipeline

            best_name = name

            importance_df = extract_feature_importance(
                pipeline,
                feature_names,
            )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1",
        ascending=False,
    ).reset_index(drop=True)

    logger.info("")
    logger.info("Benchmark Results")
    logger.info("")
    logger.info(results_df)

    return (
        results_df,
        best_model,
        best_name,
        importance_df,
    )
# ============================================================
# Save Results
# ============================================================

def save_results(
    results_df: pd.DataFrame,
    config: TrainConfig,
):

    config.model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        config.model_dir
        / "results.csv"
    )

    results_df.to_csv(
        output,
        index=False,
    )

    logger.info(
        "Saved %s",
        output,
    )
# ============================================================
# Save Best Model
# ============================================================

def save_best_model(
    model,
    model_name: str,
    feature_names,
    config: TrainConfig,
):

    model_path = (
        config.model_dir
        / "best_model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    metadata = {

        "best_model": model_name,

        "feature_names": feature_names,

        "n_features": len(feature_names),

        "classes": [str(label) for label in model.classes_],

        "positive_class": ATTACK_LABEL,

    }

    metadata_path = (
        config.model_dir
        / "metadata.json"
    )

    metadata_path.write_text(

        json.dumps(
            metadata,
            indent=4,
        ),

        encoding="utf-8",

    )

    logger.info(
        "Best model: %s",
        model_name,
    )

    logger.info(
        "Saved %s",
        model_path,
    )
# ============================================================
# Save Feature Importance
# ============================================================

def save_feature_importance(
    importance_df,
    config: TrainConfig,
):

    if importance_df is None:

        logger.info(
            "Model does not support feature importance."
        )

        return

    output = (
        config.model_dir
        / "feature_importance.csv"
    )

    importance_df.to_csv(
        output,
        index=False,
    )

    logger.info(
        "Saved %s",
        output,
    )
    # ============================================================
# Summary
# ============================================================

def print_summary(
    results_df,
):

    print()

    print("=" * 70)

    print("Benchmark Summary")

    print("=" * 70)

    print(
        results_df[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "Train_Time",
            ]
        ]
    )

    print("=" * 70)
# ============================================================
# Visualization
# ============================================================

import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
)


def save_performance_plot(
    results_df: pd.DataFrame,
    config: TrainConfig,
) -> None:
    """
    Save F1-score comparison.
    """

    plt.figure(figsize=(10, 6))

    plt.bar(
        results_df["Model"],
        results_df["F1"],
    )

    plt.xticks(rotation=30, ha="right")

    plt.ylabel("F1-score")

    plt.tight_layout()

    output = config.model_dir / "model_comparison.png"

    plt.savefig(
        output,
        dpi=300,
    )

    plt.close()

    logger.info(
        "Saved %s",
        output,
    )
# ============================================================
# Classification Report
# ============================================================

def save_classification_report(
    model,
    X_test,
    y_test,
    config: TrainConfig,
):

    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        digits=6,
    )

    output = (
        config.model_dir
        / "classification_report.txt"
    )

    output.write_text(
        report,
        encoding="utf-8",
    )

    logger.info(
        "Saved %s",
        output,
    )
# ============================================================
# Confusion Matrix
# ============================================================

def save_confusion_matrix(
    model,
    X_test,
    y_test,
    config: TrainConfig,
):

    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        xticks_rotation=45,
    )

    disp.figure_.tight_layout()

    output = (
        config.model_dir
        / "confusion_matrix.png"
    )

    disp.figure_.savefig(
        output,
        dpi=300,
    )

    plt.close(
        disp.figure_,
    )

    logger.info(
        "Saved %s",
        output,
    )
# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "data/parquet_lightweight"
        ),
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "models"
        ),
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
    )

    return parser.parse_args()
# ============================================================
# Main
# ============================================================

def main():

    configure_logging()

    args = parse_args()

    config = TrainConfig(

        data_dir=args.data_dir,

        model_dir=args.model_dir,

        test_size=args.test_size,

        sample_frac=args.sample_frac,

        max_rows_per_file=args.max_rows_per_file,

    )

    logger.info("Loading dataset...")

    data = load_dataset(config)

    X, y, preprocess, feature_names = prepare_dataset(
        data
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_dataset(
        X,
        y,
        config,
    )

    (
        results_df,
        best_model,
        best_name,
        importance_df,
    ) = benchmark_models(

        preprocess,

        feature_names,

        X_train,

        X_test,

        y_train,

        y_test,

        config,

    )

    save_results(
        results_df,
        config,
    )

    save_best_model(

        best_model,

        best_name,

        feature_names,

        config,

    )

    save_feature_importance(

        importance_df,

        config,

    )

    save_classification_report(

        best_model,

        X_test,

        y_test,

        config,

    )

    save_confusion_matrix(

        best_model,

        X_test,

        y_test,

        config,

    )

    save_performance_plot(

        results_df,

        config,

    )

    print_summary(
        results_df,
    )

    logger.info("Finished.")
# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()
