import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    PROJECT_TITLE,
    DATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    LOG_LEVEL,
    ensure_directories,
)
from hf_api import generate_ai_response
from predict import load_model_and_preprocessor, prepare_input
from train import train_pipeline
from utils import configure_logging


ensure_directories()
configure_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)


def load_local_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        logger.warning("Dataset not found at %s", DATA_PATH)
        return pd.DataFrame()
    try:
        return pd.read_csv(DATA_PATH)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to read dataset: %s", exc)
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def get_or_train_model() -> Dict[str, Any]:
    if MODEL_PATH.exists() and PREPROCESSOR_PATH.exists():
        model, preprocessor, meta = load_model_and_preprocessor()
        return {
            "model": model,
            "preprocessor": preprocessor,
            "meta": meta,
        }

    result = train_pipeline()

    if "model" not in result:
        best_name = result["best_model_name"]
        best_model = result["results"][best_name]["model"]
        import joblib
        joblib.dump(best_model, MODEL_PATH)

    model, preprocessor, meta = load_model_and_preprocessor()
    return {
        "model": model,
        "preprocessor": preprocessor,
        "meta": meta,
    }


def set_page_config() -> None:
    st.set_page_config(
        page_title=PROJECT_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    css_path = Path(__file__).resolve().parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def sidebar_navigation() -> str:
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Dashboard",
            "Ticket Classifier",
            "AI Response Generator",
            "Model Analytics",
            "Admin Panel",
        ],
    )
    return page


def render_dashboard(model_bundle: Dict[str, Any], df: pd.DataFrame) -> None:
    st.markdown("<div class='section-header'>Overview</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Rows</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='metric-value'>{len(df) if not df.empty else 0}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='metric-subtitle'>Total records</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        acc = st.session_state.get("last_accuracy", None)
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Model Accuracy</div>", unsafe_allow_html=True)
        if acc is not None:
            st.markdown(
                f"<div class='metric-value'>{acc:.3f}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='metric-value'>N/A</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-subtitle'>Training accuracy</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Features</div>", unsafe_allow_html=True)
        if not df.empty:
            st.markdown(
                f"<div class='metric-value'>{df.shape[1]}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='metric-value'>0</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-subtitle'>Columns after preprocessing</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    if not df.empty:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown(
                "<div class='section-header'>Sample Records</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(df.head(50))
        with col_right:
            st.markdown(
                "<div class='section-header'>Target Distribution</div>",
                unsafe_allow_html=True,
            )
            model_meta = model_bundle.get("meta", {})
            schema = model_meta.get("schema", {})
            target_col = schema.get("target_col", None)
            if target_col and target_col in df.columns:
                fig = px.histogram(df, x=target_col, nbins=20)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Target column not available in dataset.")
    else:
        st.info("Dataset not available. Please upload a dataset from the Admin Panel.")


def render_ticket_classifier(model_bundle: Dict[str, Any]) -> None:
    st.markdown(
        "<div class='section-header'>Ticket Classifier</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Provide a description and the system will classify it using the trained model.",
        unsafe_allow_html=False,
    )

    ticket_text = st.text_area("Ticket text", height=180)

    if st.button("Classify Ticket"):
        if not ticket_text.strip():
            st.warning("Please enter ticket text.")
            return

        with st.spinner("Classifying ticket..."):
            model = model_bundle["model"]
            preprocessor = model_bundle["preprocessor"]
            meta = model_bundle["meta"]

            df = pd.DataFrame([{"text": ticket_text}])
            df = prepare_input(df, meta)
            X = preprocessor.transform(df)
            preds = model.predict(X)

            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X)
                confidence = float(np.max(probas))
            else:
                confidence = 1.0

            target_encoder = meta.get("target_encoder")
            if target_encoder is not None:
                label = target_encoder.inverse_transform(preds)[0]
            else:
                label = preds[0]

        st.success("Ticket classified successfully.")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicted Category", value=str(label))
        with col2:
            st.metric(label="Confidence", value=f"{confidence:.3f}")


def render_ai_response_generator() -> None:
    st.markdown(
        "<div class='section-header'>AI Response Generator</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Use the LLM to generate a professional support response.",
        unsafe_allow_html=False,
    )

    ticket_text = st.text_area("Ticket text", height=220, key="ai_ticket_text")

    if st.button("Generate Response"):
        if not ticket_text.strip():
            st.warning("Please enter ticket text.")
            return

        with st.spinner("Generating AI response..."):
            response = generate_ai_response(ticket_text)

        st.markdown("**AI Response**")
        st.write(response)


def render_model_analytics(model_bundle: Dict[str, Any]) -> None:
    st.markdown(
        "<div class='section-header'>Model Analytics</div>",
        unsafe_allow_html=True,
    )

    from train import train_models, build_preprocessing_pipeline as _  # noqa: F401

    df = load_local_data()
    if df.empty:
        st.info("Dataset not available for analytics.")
        return

    st.markdown("Confusion matrix and model comparison based on training data.", unsafe_allow_html=False)

    try:
        from train import load_data  # noqa
    except Exception:
        pass

    model = model_bundle["model"]
    preprocessor = model_bundle["preprocessor"]
    meta = model_bundle["meta"]

    schema = meta.get("schema", {})
    target_col = schema.get("target_col")
    if not target_col or target_col not in df.columns:
        st.warning("Target column missing for analytics.")
        return

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
    import matplotlib.pyplot as plt

    processed_df = df.drop(columns=schema.get("dropped_cols", []), errors="ignore")

    # Drop rows with NaN in target to align with training-time behavior
    before_rows = len(processed_df)
    processed_df = processed_df[processed_df[target_col].notna()].copy()
    after_rows = len(processed_df)
    if after_rows < before_rows:
        st.info(
            f"Dropped {before_rows - after_rows} rows with missing values in target column "
            f"'{target_col}' for analytics."
        )
    if processed_df.empty:
        st.warning("No valid rows available for analytics after dropping NaN targets.")
        return

    y_true = processed_df[target_col]
    X_raw = processed_df.drop(columns=[target_col], errors="ignore")
    X = preprocessor.transform(X_raw)

    preds = model.predict(X)
    target_encoder = meta.get("target_encoder")

    # Recreate the same target encoding/binning logic used during training
    if target_encoder is not None:
        y_true_enc = target_encoder.transform(y_true.astype(str))
    else:
        # Mirror build_preprocessing_pipeline behaviour for numeric targets
        if not pd.api.types.is_numeric_dtype(y_true):
            y_true_enc = y_true
        else:
            if y_true.nunique() > 20:
                # High-cardinality numeric target was binned into quantiles for training
                y_true_enc = pd.qcut(
                    y_true.rank(method="first"),
                    q=4,
                    labels=False,
                )
            else:
                y_true_enc = y_true.values

    acc = accuracy_score(y_true_enc, preds)
    st.session_state["last_accuracy"] = acc

    cm = confusion_matrix(y_true_enc, preds)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, colorbar=False)
        st.pyplot(fig)

    with col2:
        st.metric("Accuracy", f"{acc:.3f}")

    if hasattr(model, "feature_importances_"):
        st.markdown("### Feature Importance")
        importances = model.feature_importances_
        fig_imp = px.bar(
            x=list(range(len(importances))),
            y=importances,
            labels={"x": "Feature Index", "y": "Importance"},
        )
        st.plotly_chart(fig_imp, use_container_width=True)


def render_admin_panel() -> None:
    st.markdown(
        "<div class='section-header'>Admin Panel</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "Upload a new CSV dataset and retrain the model from the latest data.",
        unsafe_allow_html=False,
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df_new = pd.read_csv(uploaded_file)
        df_new.to_csv(DATA_PATH, index=False)
        st.success("New dataset uploaded successfully.")

    if st.button("Retrain Model"):
        with st.spinner("Retraining model, please wait..."):
            result = train_pipeline()
            if "best_model_name" in result:
                best_name = result["best_model_name"]
                st.session_state["last_accuracy"] = result["results"][best_name]["accuracy"]
        st.success("Model retrained successfully.")


def main() -> None:
    set_page_config()
    st.title(PROJECT_TITLE)
    st.caption("End-to-end ML + LLM system for daily stock market forecasting and ticket intelligence.")

    page = sidebar_navigation()

    model_bundle = {}
    df = load_local_data()

    if page in {"Dashboard", "Ticket Classifier", "Model Analytics"}:
        with st.spinner("Loading model and preprocessing pipeline..."):
            try:
                model_bundle = get_or_train_model()
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Failed to load or train model: {exc}")

    if page == "Dashboard":
        render_dashboard(model_bundle, df)
    elif page == "Ticket Classifier":
        if model_bundle:
            render_ticket_classifier(model_bundle)
        else:
            st.warning("Model not available. Please check Admin Panel to train one.")
    elif page == "AI Response Generator":
        render_ai_response_generator()
    elif page == "Model Analytics":
        if model_bundle:
            render_model_analytics(model_bundle)
        else:
            st.warning("Model not available. Please check Admin Panel to train one.")
    elif page == "Admin Panel":
        render_admin_panel()


if __name__ == "__main__":
    main()
