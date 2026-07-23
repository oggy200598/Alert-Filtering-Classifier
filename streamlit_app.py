"""Streamlit interface for the 22-feature Alert Filtering Classifier."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "best_model.joblib"
METADATA_PATH = PROJECT_DIR / "models" / "metadata.json"
EXPECTED_FEATURE_COUNT = 22


st.set_page_config(
    page_title="Alert Filtering Classifier",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    """Load the model and its feature schema once per Streamlit session."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Không tìm thấy best_model.joblib hoặc metadata.json trong thư mục models.")

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    feature_names = metadata.get("feature_names", [])

    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Model phải dùng {EXPECTED_FEATURE_COUNT} đặc trưng, nhưng metadata có {len(feature_names)}."
        )

    return model, metadata


def prepare_features(data: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Match uploaded CSV columns to the exact feature schema used during training."""
    frame = data.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.drop(columns=["Label"], errors="ignore")

    missing = [column for column in feature_names if column not in frame.columns]
    for column in feature_names:
        if column not in frame.columns:
            frame[column] = np.nan
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame[feature_names], missing


def decode_label(prediction: object, metadata: dict) -> str:
    """Decode numeric notebook labels (0/1) or preserve string model labels."""
    if isinstance(prediction, (int, np.integer)):
        classes = metadata.get("classes", [])
        if 0 <= int(prediction) < len(classes):
            return str(classes[int(prediction)])
    return str(prediction)


def attack_probability(model, features: pd.DataFrame, metadata: dict) -> np.ndarray | None:
    """Return the probability of ATTACK using the model's actual class order."""
    if not hasattr(model, "predict_proba"):
        return None

    classes = metadata.get("classes", [])
    class_mapping = metadata.get("class_mapping", {})
    attack_code = class_mapping.get("ATTACK", classes.index("ATTACK") if "ATTACK" in classes else "ATTACK")
    model_classes = list(model.classes_)

    if attack_code not in model_classes:
        return None

    return model.predict_proba(features)[:, model_classes.index(attack_code)]


st.title("🛡️ Alert Filtering Classifier")
st.write("Tải CSV có 22 đặc trưng Lightweight Ontology để phân loại BENIGN hoặc ATTACK.")

try:
    model, metadata = load_artifacts()
except Exception as exc:
    st.error(f"Không thể tải model: {exc}")
    st.stop()

feature_names = metadata["feature_names"]
uploaded_file = st.file_uploader("Chọn tệp CSV", type=["csv"])

if uploaded_file is not None:
    try:
        uploaded_data = pd.read_csv(uploaded_file)
        features, missing_columns = prepare_features(uploaded_data, feature_names)
    except Exception as exc:
        st.error(f"Không thể đọc hoặc chuẩn hóa CSV: {exc}")
        st.stop()

    st.subheader("Dữ liệu đầu vào")
    st.dataframe(uploaded_data.head())

    if missing_columns:
        st.warning(
            "CSV thiếu các cột sau; chúng sẽ được điền NaN để pipeline xử lý: "
            + ", ".join(missing_columns)
        )

    if st.button("Dự đoán", type="primary"):
        try:
            raw_predictions = model.predict(features)
            result = uploaded_data.copy()
            result["Prediction"] = [decode_label(value, metadata) for value in raw_predictions]

            probabilities = attack_probability(model, features, metadata)
            if probabilities is not None:
                result["Attack Probability"] = probabilities

            benign_count = int((result["Prediction"] == "BENIGN").sum())
            attack_count = int((result["Prediction"] == "ATTACK").sum())
            col1, col2 = st.columns(2)
            col1.metric("BENIGN", benign_count)
            col2.metric("ATTACK", attack_count)

            st.subheader("Kết quả")
            st.dataframe(result, use_container_width=True)
            st.download_button(
                "📥 Tải kết quả CSV",
                result.to_csv(index=False).encode("utf-8-sig"),
                "afc_predictions.csv",
                "text/csv",
            )
        except Exception as exc:
            st.error(f"Dự đoán thất bại: {exc}")
