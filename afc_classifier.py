"""
Alert Filtering Classifier (AFC)
================================

Script này huấn luyện một bộ lọc cảnh báo từ các file Parquet trong thư mục data/parquet.

Ý tưởng chính:
- Mỗi dòng trong Parquet là một network flow.
- Cột Label cho biết flow là BENIGN hay một kiểu tấn công.
- AFC học cách tách traffic bình thường khỏi traffic đáng cảnh báo.

Mặc định script train bài toán nhị phân:
- BENIGN  -> 0 / bình thường
- ATTACK  -> 1 / cảnh báo cần giữ lại

Nếu cần train đa lớp, dùng thêm tham số: --mode multiclass
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Các cột này có thể làm model "học vẹt" theo IP, timestamp hoặc flow id.
# Trong bài toán AFC thực tế, ta thường muốn model học hành vi traffic
# hơn là nhớ máy nào nối với máy nào trong tập train.
DROP_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
}

EXPECTED_FEATURE_COUNT = 22


@dataclass(frozen=True)
class TrainConfig:
    """Gom các tham số train lại thành một object để code dễ đọc hơn."""

    data_dir: Path
    model_dir: Path
    mode: str
    max_rows_per_file: int | None
    sample_frac: float
    test_size: float
    random_state: int
    n_estimators: int


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa tên cột.

    Bộ CICIDS có file header dạng "Flow ID, Source IP, ..." và có file
    dạng "Flow ID,Source IP,...". Nếu không strip khoảng trắng, pandas sẽ coi
    " Label" và "Label" là hai tên cột khác nhau.
    """

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def normalize_label(value: object, mode: str) -> str:
    """
    Làm sạch nhãn và chuyển về dạng model cần học.

    Một số file WebAttack có ký tự lỗi encoding; thay vì để label bị vỡ ngẫu
    nhiên, ta strip và gom nhãn lỗi vào ATTACK nếu train nhị phân.
    """

    label = str(value).strip()
    if mode == "binary":
        return "BENIGN" if label.upper() == "BENIGN" else "ATTACK"
    return label


def iter_parquet_files(data_dir: Path) -> Iterable[Path]:
    """Trả về danh sách Parquet theo thứ tự ổn định để kết quả dễ lặp lại."""

    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file .parquet nào trong {data_dir}")
    return files


def read_one_parquet(path: Path, config: TrainConfig) -> pd.DataFrame:
    """
    Đọc một file Parquet.

    Với máy cá nhân, train toàn bộ CICIDS có thể rất nặng. Vì vậy:
    - --max-rows-per-file giới hạn số dòng lấy từ mỗi file.
    - --sample-frac lấy ngẫu nhiên một phần mỗi file.
    Hai tham số này giúp code chạy được nhanh khi làm demo/báo cáo.
    """

    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except ImportError as exc:
        raise RuntimeError(
            f"Không thể đọc file {path.name}. Cần cài đặt `pyarrow` hoặc `fastparquet` để hỗ trợ Parquet. "
            "Chạy `pip install pyarrow` hoặc `pip install fastparquet` trong môi trường ảo hiện tại."
        ) from exc

    df = normalize_columns(df)

    if config.sample_frac < 1.0:
        df = df.sample(frac=config.sample_frac, random_state=config.random_state)

    if config.max_rows_per_file is not None and len(df) > config.max_rows_per_file:
        df = df.sample(n=config.max_rows_per_file, random_state=config.random_state)

    print(f"Đã đọc {len(df):>7,} dòng từ {path.name}")
    return df


def load_dataset(config: TrainConfig) -> pd.DataFrame:
    """Đọc tất cả Parquet và gộp thành một DataFrame duy nhất."""

    frames = [read_one_parquet(path, config) for path in iter_parquet_files(config.data_dir)]
    data = pd.concat(frames, ignore_index=True)
    data = normalize_columns(data)

    if "Label" not in data.columns:
        raise KeyError("Không tìm thấy cột Label trong dữ liệu")

    data["Label"] = data["Label"].map(lambda value: normalize_label(value, config.mode))
    return data


def split_features_and_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Tách X/y và ép các feature về số.

    Lý do ép numeric:
    - RandomForest chỉ làm việc với số.
    - Các cột IP/timestamp/id đã được drop ở trên.
    - Nếu còn cột text nào sót lại, to_numeric sẽ biến thành NaN và imputer xử lý.
    """

    y = data["Label"].astype(str)
    x = data.drop(columns=["Label"])
    x = x.drop(columns=[col for col in DROP_COLUMNS if col in x.columns], errors="ignore")

    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    # Các cột Flow Bytes/s, Flow Packets/s thỉnh thoảng có inf khi duration = 0.
    # Imputer không xử lý inf, nên đổi inf thành NaN trước.
    x = x.replace([np.inf, -np.inf], np.nan)

    if x.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} Lightweight Ontology features, got {x.shape[1]}. "
            "Use data/parquet_lightweight or verify the preprocessing output."
        )

    return x, y


def build_model(
    feature_names: list[str],
    n_estimators: int,
    random_state: int,
) -> Pipeline:
    """
    Tạo pipeline tiền xử lý + mô hình.

    RandomForest phù hợp cho baseline AFC vì:
    - Chịu được feature phi tuyến.
    - Không đòi hỏi scale feature.
    - Cho biết feature importance để giải thích kết quả.
    """

    numeric_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocess, feature_names),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", classifier),
        ]
    )


def save_training_artifacts(
    model: Pipeline,
    model_dir: Path,
    feature_names: list[str],
    labels: list[str],
    config: TrainConfig,
) -> None:
    """Lưu model và metadata để predict sau này đúng feature."""

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "afc_model.joblib")

    metadata = {
        "mode": config.mode,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "labels": labels,
        "dropped_columns": sorted(DROP_COLUMNS),
        "random_state": config.random_state,
        "n_estimators": config.n_estimators,
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def train(config: TrainConfig) -> None:
    """Hàm chính cho lệnh train."""

    data = load_dataset(config)
    x, y = split_features_and_target(data)

    print("\nPhân bố nhãn sau khi tiền xử lý:")
    print(y.value_counts().to_string())

    # Stratify giúp tập test giữ tỉ lệ nhãn gần giống tập train.
    # Nếu có nhãn quá hiếm (chỉ 1 dòng), train_test_split sẽ không stratify được.
    stratify_target = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_target,
    )

    model = build_model(
        feature_names=list(x.columns),
        n_estimators=config.n_estimators,
        random_state=config.random_state,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("\nBáo cáo đánh giá AFC:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())))

    save_training_artifacts(
        model=model,
        model_dir=config.model_dir,
        feature_names=list(x.columns),
        labels=sorted(y.unique()),
        config=config,
    )
    print(f"\nĐã lưu model vào: {config.model_dir / 'afc_model.joblib'}")


def load_model_and_metadata(model_dir: Path) -> tuple[Pipeline, dict]:
    """Nạp model và metadata đã tạo từ lệnh train."""

    model_path = model_dir / "afc_model.joblib"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Chưa thấy model. Hãy chạy train trước, ví dụ: python afc_classifier.py train"
        )

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def prepare_predict_features(input_parquet: Path, metadata: dict) -> pd.DataFrame:
    """Đọc Parquet mới và sắp xếp cột đúng như lúc train."""

    data = pd.read_parquet(input_parquet, engine="pyarrow")
    data = normalize_columns(data)

    if "Label" in data.columns:
        data = data.drop(columns=["Label"])

    data = data.drop(columns=[col for col in DROP_COLUMNS if col in data.columns], errors="ignore")

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)

    # Nếu Parquet mới thiếu cột nào đó, thêm cột NaN để imputer xử lý.
    # Nếu Parquet có cột lạ, bỏ qua cột lạ để model nhận đúng schema đã train.
    feature_names = metadata["feature_names"]
    for col in feature_names:
        if col not in data.columns:
            data[col] = np.nan
    return data[feature_names]


def predict(model_dir: Path, input_parquet: Path, output_parquet: Path) -> None:
    """Hàm chính cho lệnh predict."""

    model, metadata = load_model_and_metadata(model_dir)
    x = prepare_predict_features(input_parquet, metadata)

    predictions = model.predict(x)
    result = pd.DataFrame({"afc_prediction": predictions})

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        classes = list(model.classes_)
        if "ATTACK" in classes:
            attack_index = classes.index("ATTACK")
            result["attack_probability"] = probabilities[:, attack_index]

    result.to_parquet(output_parquet, index=False, engine="pyarrow", compression="snappy")
    print(f"Đã ghi kết quả predict vào: {output_parquet}")


def parse_args() -> argparse.Namespace:
    """Khai báo CLI để có thể train/predict bằng terminal."""

    parser = argparse.ArgumentParser(description="Alert Filtering Classifier từ dữ liệu Parquet")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Huấn luyện AFC")
    train_parser.add_argument("--data-dir", type=Path, default=Path("data/parquet_lightweight"))
    train_parser.add_argument("--model-dir", type=Path, default=Path("models"))
    train_parser.add_argument("--mode", choices=["binary", "multiclass"], default="binary")
    train_parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=75_000,
        help="Số dòng tối đa mỗi file; đặt <= 0 để đọc toàn bộ",
    )
    train_parser.add_argument("--sample-frac", type=float, default=1.0)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--n-estimators", type=int, default=200)

    predict_parser = subparsers.add_parser("predict", help="Dự đoán Parquet mới bằng model AFC")
    predict_parser.add_argument("--model-dir", type=Path, default=Path("models"))
    predict_parser.add_argument("--input-parquet", type=Path, required=True)
    predict_parser.add_argument("--output-parquet", type=Path, default=Path("afc_predictions.parquet"))

    return parser.parse_args()


def main() -> None:
    """Điểm vào của chương trình."""

    args = parse_args()

    if args.command == "train":
        config = TrainConfig(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            mode=args.mode,
            max_rows_per_file=None
            if args.max_rows_per_file <= 0
            else args.max_rows_per_file,
            sample_frac=args.sample_frac,
            test_size=args.test_size,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
        )
        train(config)
    elif args.command == "predict":
        predict(
            model_dir=args.model_dir,
            input_parquet=args.input_parquet,
            output_parquet=args.output_parquet,
        )


if __name__ == "__main__":
    main()
