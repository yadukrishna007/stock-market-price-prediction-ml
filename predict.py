import logging
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from config import MODEL_PATH, PREPROCESSOR_PATH, LOG_LEVEL, ensure_directories
from utils import configure_logging


logger = logging.getLogger(__name__)


def load_model_and_preprocessor() -> Tuple[Any, Any, Dict[str, Any]]:
    ensure_directories()
    configure_logging(LOG_LEVEL)

    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "Model or preprocessor not found. Please run training first."
        )

    model = joblib.load(MODEL_PATH)
    pre_meta = joblib.load(PREPROCESSOR_PATH)
    preprocessor = pre_meta["preprocessor"]
    meta = pre_meta["meta"]

    logger.info("Loaded model and preprocessing artifacts.")
    return model, preprocessor, meta


def prepare_input(df: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    schema = meta["schema"]
    drop_cols = schema.get("dropped_cols", [])

    df_processed = df.drop(columns=drop_cols, errors="ignore").copy()
    return df_processed


def predict_single(record: Dict[str, Any]) -> Dict[str, Any]:
    model, preprocessor, meta = load_model_and_preprocessor()

    input_df = pd.DataFrame([record])
    input_df = prepare_input(input_df, meta)

    X = preprocessor.transform(input_df)
    preds = model.predict(X)

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        confidences = np.max(probas, axis=1)
        proba_per_class = probas[0].tolist()
    else:
        confidences = np.array([1.0])
        proba_per_class = []

    target_encoder = meta.get("target_encoder")
    if target_encoder is not None:
        label = target_encoder.inverse_transform(preds)[0]
    else:
        label = preds[0]

    return {
        "label": label,
        "confidence": float(confidences[0]),
        "proba_per_class": proba_per_class,
    }


def classify_text_ticket(ticket_text: str) -> Dict[str, Any]:
    record = {"text": ticket_text}
    return predict_single(record)


if __name__ == "__main__":
    try:
        example = {}
        result = predict_single(example)
        print(result)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
