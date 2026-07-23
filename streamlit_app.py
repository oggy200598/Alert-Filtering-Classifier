"""Streamlit interface for the 22-feature Alert Filtering Classifier (Multi-file Big Data, Parquet & Visualizations Support)."""

from __future__ import annotations

import json
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix


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


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read CSV or Parquet files efficiently."""
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension == ".parquet":
        return pd.read_parquet(uploaded_file)
    elif file_extension == ".csv":
        return pd.read_csv(uploaded_file)
    else:
        raise ValueError("Định dạng file không được hỗ trợ. Vui lòng chọn file CSV hoặc Parquet.")


def prepare_features(data: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Match uploaded dataframe columns to the exact feature schema used during training."""
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


# --- MAIN INTERFACE ---

st.title("🛡️ Alert Filtering Classifier")
st.write("Tải các tập dữ liệu (**CSV** hoặc **Parquet**) có 22 đặc trưng Lightweight Ontology để phân loại BENIGN hoặc ATTACK.")

try:
    model, metadata = load_artifacts()
except Exception as exc:
    st.error(f"Không thể tải model: {exc}")
    st.stop()

feature_names = metadata["feature_names"]

uploaded_files = st.file_uploader(
    "Chọn các tệp dữ liệu (Hỗ trợ chọn nhiều file CSV & Parquet)", 
    type=["csv", "parquet"],
    accept_multiple_files=True
)

if uploaded_files:
    try:
        with st.spinner("Đang tải và gộp dữ liệu từ các file..."):
            data_frames = []
            for file in uploaded_files:
                df = read_uploaded_file(file)
                data_frames.append(df)
            
            uploaded_data = pd.concat(data_frames, ignore_index=True)
            features, missing_columns = prepare_features(uploaded_data, feature_names)

        st.success(f"Đã tải thành công {len(uploaded_files)} tệp với tổng cộng {len(uploaded_data):,} dòng dữ liệu!")
    except Exception as exc:
        st.error(f"Không thể đọc hoặc chuẩn hóa tệp: {exc}")
        st.stop()

    st.subheader("Dữ liệu đầu vào (5 dòng đầu)")
    st.dataframe(uploaded_data.head(), width="stretch")

    if missing_columns:
        st.warning(
            "Dữ liệu thiếu các cột sau; chúng sẽ được điền NaN để pipeline xử lý: "
            + ", ".join(missing_columns)
        )

    if st.button("Dự đoán", type="primary"):
        try:
            with st.spinner("Đang chạy mô hình dự đoán..."):
                raw_predictions = model.predict(features)
                result = uploaded_data.copy()
                result["Prediction"] = [decode_label(value, metadata) for value in raw_predictions]

                probabilities = attack_probability(model, features, metadata)
                if probabilities is not None:
                    result["Attack Probability"] = probabilities

            benign_count = int((result["Prediction"] == "BENIGN").sum())
            attack_count = int((result["Prediction"] == "ATTACK").sum())
            
            col1, col2 = st.columns(2)
            col1.metric("BENIGN (Lành tính)", benign_count)
            col2.metric("ATTACK (Tấn công)", attack_count)

            # --- BIỂU ĐỒ TRỰC QUAN KẾT QUẢ ---
            st.markdown("---")
            st.subheader("📈 Phân tích trực quan kết quả dự đoán")
            
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig_pie = px.pie(
                    names=["BENIGN", "ATTACK"],
                    values=[benign_count, attack_count],
                    title="Tỷ lệ phân loại cảnh báo",
                    hole=0.4,
                    color=["BENIGN", "ATTACK"],
                    color_discrete_map={"BENIGN": "#2ecc71", "ATTACK": "#e74c3c"}
                )
                st.plotly_chart(fig_pie, width="stretch")

            with chart_col2:
                if "Attack Probability" in result.columns:
                    fig_hist = px.histogram(
                        result, 
                        x="Attack Probability", 
                        nbins=30,
                        title="Phân phối Xác suất Tấn công (Attack Probability)",
                        labels={"Attack Probability": "Xác suất ATTACK"},
                        color_discrete_sequence=["#3498db"]
                    )
                    st.plotly_chart(fig_hist, width="stretch")

            # --- ĐÁNH GIÁ ĐỘ CHÍNH XÁC (Nếu file đầu vào có cột 'Label') ---
            if "Label" in uploaded_data.columns:
                st.markdown("---")
                st.subheader("📊 Đánh giá mô hình & Ma trận nhầm lẫn (Confusion Matrix)")

                y_true = uploaded_data["Label"].astype(str).str.strip()
                y_pred = result["Prediction"].astype(str).str.strip()

                acc = accuracy_score(y_true, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="weighted", zero_division=0
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{acc * 100:.2f}%")
                c2.metric("Precision", f"{precision * 100:.2f}%")
                c3.metric("Recall", f"{recall * 100:.2f}%")
                c4.metric("F1-Score", f"{f1 * 100:.2f}%")

                eval_col1, eval_col2 = st.columns(2)

                with eval_col1:
                    labels_cm = sorted(list(set(y_true) | set(y_pred)))
                    cm = confusion_matrix(y_true, y_pred, labels=labels_cm)

                    fig_cm, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_cm, yticklabels=labels_cm, ax=ax)
                    ax.set_xlabel("Dự đoán (Predicted)")
                    ax.set_ylabel("Thực tế (Actual)")
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig_cm)

                with eval_col2:
                    with st.expander("🔍 Báo cáo chi tiết (Classification Report)", expanded=True):
                        report_dict = classification_report(y_true, y_pred, output_dict=True)
                        st.dataframe(pd.DataFrame(report_dict).transpose(), width="stretch")

                result["Is_Correct"] = y_true == y_pred
                incorrect_preds = result[result["Is_Correct"] == False]

                if not incorrect_preds.empty:
                    st.warning(f"⚠️ Phát hiện **{len(incorrect_preds):,}** dòng bị dự đoán **SAI** so với nhãn thực tế.")
                    with st.expander("❌ Xem danh sách các mẫu dự đoán SAI"):
                        st.dataframe(incorrect_preds, width="stretch")
                else:
                    st.success("🎉 Tất cả mẫu dữ liệu đều được dự đoán ĐÚNG 100%!")

            # --- FEATURE IMPORTANCE ---
            if hasattr(model, "feature_importances_"):
                st.markdown("---")
                st.subheader("⭐ Độ quan trọng của các đặc trưng (Feature Importance)")
                
                fi_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=True)

                fig_fi = px.bar(
                    fi_df, 
                    x="Importance", 
                    y="Feature", 
                    orientation="h",
                    title="Top đặc trưng ảnh hưởng nhiều nhất đến quyết định của Mô hình",
                    color="Importance",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_fi, width="stretch")

            st.subheader("Xem trước kết quả (100 dòng đầu)")
            st.dataframe(result.head(100), width="stretch")

            # --- Xuất file kết quả ---
            st.subheader("📥 Tải về kết quả gộp")
            export_format = st.radio("Chọn định dạng file tải về:", ["Parquet (Tối ưu file lớn)", "CSV"], horizontal=True)

            if export_format == "Parquet (Tối ưu file lớn)":
                buffer = io.BytesIO()
                result.to_parquet(buffer, index=False)
                st.download_button(
                    label="📥 Tải kết quả Parquet",
                    data=buffer.getvalue(),
                    file_name="afc_predictions_combined.parquet",
                    mime="application/octet-stream",
                )
            else:
                st.download_button(
                    label="📥 Tải kết quả CSV",
                    data=result.to_csv(index=False).encode("utf-8-sig"),
                    file_name="afc_predictions_combined.csv",
                    mime="text/csv",
                )

        except Exception as exc:
            st.error(f"Dự đoán thất bại: {exc}")