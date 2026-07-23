from __future__ import annotations

import argparse
import json
import logging
import time

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype

from tqdm.auto import tqdm

import logging

logger = logging.getLogger(__name__)
# ==========================================================
# CONFIGURATION
# ==========================================================

LABEL_COLUMN = "Label"

EXPECTED_FEATURE_COUNT = 22

DROP_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
}

ONTOLOGY_FEATURES = [
    "Destination Port",
    "Protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "SYN Flag Count",
    "ACK Flag Count",
    "FIN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "URG Flag Count",
    "Average Packet Size",
    "Packet Length Mean",
    "Packet Length Std",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Active Mean",
    "Idle Mean",
    LABEL_COLUMN,
]


@dataclass(slots=True)
class PreprocessConfig:
    input_dir: Path
    full_output_dir: Path
    lightweight_output_dir: Path
    compression: str = "snappy"


# ==========================================================
# LOGGING
# ==========================================================

def configure_logging() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ==========================================================
# UTILITIES
# ==========================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa tên cột.
    """

    df = df.copy()

    df.columns = [
        str(col)
        .strip()
        .replace("\n", " ")
        for col in df.columns
    ]

    return df


def get_csv_files(directory: Path) -> list[Path]:

    directory.mkdir(exist_ok=True)

    csv_files = sorted(directory.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {directory}"
        )

    return csv_files


# ==========================================================
# CLEANING
# ==========================================================

def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loại bỏ cột trùng tên.

    Không dùng df.T.duplicated()
    vì rất tốn RAM với CICIDS.
    """

    return df.loc[:, ~df.columns.duplicated()]


def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:

    keep = []

    for col in df.columns:

        if df[col].isna().all():
            continue

        if (
            df[col]
            .astype(str)
            .str.strip()
            .eq("")
            .all()
        ):
            continue

        keep.append(col)

    return df[keep]


def convert_numeric(df: pd.DataFrame) -> pd.DataFrame:

    numeric_cols = [
        c
        for c in df.columns
        if c != LABEL_COLUMN
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    for col in df.columns:

        if col == LABEL_COLUMN:
            continue

        if df[col].isna().all():

            logger.warning(
                "Drop column %s because all values are NaN",
                col,
            )

            df = df.drop(columns=col)

            continue

        if df[col].isna().any():

            median = df[col].median()

            df[col] = df[col].fillna(median)

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    before_rows = len(df)

    df = normalize_columns(df)

    df = df.drop_duplicates(ignore_index=True)

    df = remove_duplicate_columns(df)

    df = remove_empty_columns(df)

    df = df.drop(
        columns=[
            c
            for c in DROP_COLUMNS
            if c in df.columns
        ],
        errors="ignore",
    )

    if LABEL_COLUMN not in df.columns:

        raise KeyError(
            "Label column not found."
        )

    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN]
        .astype(str)
        .str.strip()
    )

    df = df[
        df[LABEL_COLUMN] != ""
    ].copy()

    df = convert_numeric(df)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = fill_missing_values(df)

    logger.info(
        "Rows: %d -> %d | Columns: %d",
        before_rows,
        len(df),
        df.shape[1],
    )

    return df
# ==========================================================
# VALIDATION
# ==========================================================

def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Kiểm tra DataFrame sau khi làm sạch.
    """

    if LABEL_COLUMN not in df.columns:
        raise ValueError("Missing Label column.")

    duplicated = df.columns[df.columns.duplicated()]

    if len(duplicated):

        raise ValueError(
            f"Duplicate columns detected: {duplicated.tolist()}"
        )

    if np.isinf(
        df.select_dtypes(include=np.number)
    ).values.any():

        raise ValueError(
            "Infinite values still exist."
        )

    object_columns = [

        c

        for c in df.columns

        if c != LABEL_COLUMN

        and df[c].dtype == object

    ]

    if object_columns:

        logger.warning(
            "Object columns remaining: %s",
            object_columns,
        )


# ==========================================================
# ONTOLOGY FEATURE SELECTION
# ==========================================================

def select_ontology_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Chỉ giữ ontology feature.

    Không raise lỗi nếu thiếu feature.
    """

    available = [

        feature

        for feature in ONTOLOGY_FEATURES

        if feature in df.columns

    ]

    missing = sorted(

        set(ONTOLOGY_FEATURES)

        - set(available)

    )

    if missing:

        raise ValueError(
            "Cannot create the 22-feature Lightweight Ontology dataset. "
            f"Missing: {', '.join(missing)}"
        )

    df = df[available].copy()

    feature_count = len(df.columns) - 1

    if feature_count != EXPECTED_FEATURE_COUNT:

        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} ontology features, got {feature_count}."
        )

    validate_dataframe(df)

    return df


# ==========================================================
# EXPORT PARQUET
# ==========================================================

def export_parquet(
    df: pd.DataFrame,
    output_path: Path,
    compression: str,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(

        output_path,

        engine="pyarrow",

        compression=compression,

        index=False,

    )

    logger.info(

        "Saved %s (%d rows, %d columns)",

        output_path.name,

        len(df),

        df.shape[1],

    )


# ==========================================================
# FEATURE STATISTICS
# ==========================================================

def feature_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for col in df.columns:

        if col == LABEL_COLUMN:
            continue

        series = df[col]

        rows.append(
            {
                "Feature": col,
                "Missing": int(series.isna().sum()),
                "Mean": float(series.mean()),
                "Std": float(series.std()),
                "Min": float(series.min()),
                "Max": float(series.max()),
            }
        )

    return pd.DataFrame(rows)


# ==========================================================
# METADATA
# ==========================================================

def write_metadata(
    output_dir: Path,
    feature_names: list[str],
    removed_columns: set[str],
    numeric_columns: set[str],
) -> None:

    metadata = {

        "dataset": "CICIDS2017",

        "version": "Lightweight",

        "feature_names": feature_names,

        "ontology_features": feature_names,

        "removed_columns": sorted(
            removed_columns
        ),

        "numeric_columns": sorted(
            numeric_columns
        ),

        "label_column": LABEL_COLUMN,

        "total_features": len(feature_names),

    }

    metadata_path = output_dir / "metadata.json"

    metadata_path.write_text(

        json.dumps(

            metadata,

            indent=4,

            ensure_ascii=False,

        ),

        encoding="utf-8",

    )

    logger.info(

        "Metadata written -> %s",

        metadata_path,

    )


# ==========================================================
# SAVE FEATURE STATISTICS
# ==========================================================

def export_statistics(
    stats: pd.DataFrame,
    output_dir: Path,
) -> None:

    stats.to_csv(

        output_dir / "feature_statistics.csv",

        index=False,

    )

    logger.info(

        "Feature statistics saved."

    )
    # ==========================================================
# PROCESS SINGLE DATASET
# ==========================================================

def process_dataset(
    csv_path: Path,
    config: PreprocessConfig,
) -> tuple[set[str], set[str], pd.DataFrame]:

    logger.info("Processing %s", csv_path.name)

    start = time.perf_counter()

    df = pd.read_csv(
        csv_path,
        low_memory=False,
        on_bad_lines="skip",
        skipinitialspace=True,
        encoding="utf-8",
        encoding_errors="replace",
    )

    original_columns = set(df.columns)

    # -----------------------------
    # CLEAN
    # -----------------------------

    df = clean_dataframe(df)

    validate_dataframe(df)

    # -----------------------------
    # FULL PARQUET
    # -----------------------------

    full_output = (
        config.full_output_dir
        / f"{csv_path.stem}.parquet"
    )

    export_parquet(
        df,
        full_output,
        config.compression,
    )

    # -----------------------------
    # LIGHTWEIGHT
    # -----------------------------

    lightweight_df = select_ontology_features(df)

    light_output = (
        config.lightweight_output_dir
        / f"{csv_path.stem}.parquet"
    )

    export_parquet(
        lightweight_df,
        light_output,
        config.compression,
    )

    removed_columns = (
        original_columns
        - set(df.columns)
    )

    numeric_columns = {

        c

        for c in lightweight_df.columns

        if c != LABEL_COLUMN

        and is_numeric_dtype(
            lightweight_df[c]
        )

    }

    elapsed = time.perf_counter() - start

    logger.info(
        "%s finished in %.2f seconds",
        csv_path.name,
        elapsed,
    )

    return (
        removed_columns,
        numeric_columns,
        lightweight_df,
    )


# ==========================================================
# PROCESS ALL DATASETS
# ==========================================================

def process_all_datasets(
    config: PreprocessConfig,
) -> None:

    csv_files = get_csv_files(
        config.input_dir
    )

    config.full_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config.lightweight_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_removed = set()

    total_numeric = set()

    statistics = []

    total_rows = 0

    logger.info(
        "Found %d CSV files.",
        len(csv_files),
    )

    for csv_file in tqdm(
        csv_files,
        desc="Processing",
        unit="file",
    ):

        removed, numeric, light_df = process_dataset(
            csv_file,
            config,
        )

        total_removed.update(
            removed
        )

        total_numeric.update(
            numeric
        )

        statistics.append(
            feature_statistics(light_df)
        )

        total_rows += len(light_df)

    # -----------------------------
    # Statistics
    # -----------------------------

    if statistics:

        stats = pd.concat(
            statistics,
            ignore_index=True,
        )

        export_statistics(
            stats,
            config.lightweight_output_dir,
        )

    # -----------------------------
    # Metadata
    # -----------------------------

    write_metadata(
        output_dir=config.lightweight_output_dir,
        feature_names=[
            c
            for c in ONTOLOGY_FEATURES
            if c != LABEL_COLUMN
        ],
        removed_columns=total_removed,
        numeric_columns=total_numeric,
    )

    logger.info("===================================")
    logger.info("Processing completed")
    logger.info("Files processed : %d", len(csv_files))
    logger.info("Rows processed  : %d", total_rows)
    logger.info(
        "Output (full)  : %s",
        config.full_output_dir,
    )
    logger.info(
        "Output (light) : %s",
        config.lightweight_output_dir,
    )
    logger.info("===================================")


# ==========================================================
# CLI
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Preprocess CICIDS2017"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/csv"),
    )

    parser.add_argument(
        "--full-output-dir",
        type=Path,
        default=Path("data/parquet"),
    )

    parser.add_argument(
        "--light-output-dir",
        type=Path,
        default=Path("data/parquet_lightweight"),
    )

    parser.add_argument(
        "--compression",
        default="snappy",
        choices=[
            "snappy",
            "gzip",
            "brotli",
            "zstd",
        ],
    )

    return parser.parse_args()


# ==========================================================
# MAIN
# ==========================================================

def main():

    configure_logging()

    args = parse_args()

    config = PreprocessConfig(

        input_dir=args.input_dir,

        full_output_dir=args.full_output_dir,

        lightweight_output_dir=args.light_output_dir,

        compression=args.compression,

    )

    process_all_datasets(config)


if __name__ == "__main__":

    main()
