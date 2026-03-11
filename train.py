import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import (
    DATA_PATH,
    TEXT_COLUMN,
    TARGET_COLUMN,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    LOG_LEVEL,
    ensure_directories,
)
from utils import configure_logging, build_preprocessing_pipeline


logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.error("Dataset not found at %s", path)
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    logger.info("Loaded dataset with shape %s", df.shape)
    return df


def train_models(X, y) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        logger.info("Training model: %s", name)
        model.fit(X, y)
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        cls_report = classification_report(y, preds, output_dict=False)
        cm = confusion_matrix(y, preds)

        feature_importances = None
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_

        results[name] = {
            "model": model,
            "accuracy": acc,
            "classification_report": cls_report,
            "confusion_matrix": cm,
            "feature_importances": feature_importances,
        }

        logger.info("Model %s accuracy: %.4f", name, acc)

    return results


def select_best_model(results: Dict[str, Dict[str, Any]]) -> str:
    best_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
    logger.info("Best model selected: %s (accuracy=%.4f)", best_name, results[best_name]["accuracy"])
    return best_name


def save_artifacts(
    best_model_name: str,
    results: Dict[str, Dict[str, Any]],
    preprocessor,
    meta: Dict[str, Any],
) -> None:
    best_model = results[best_model_name]["model"]
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump({"preprocessor": preprocessor, "meta": meta}, PREPROCESSOR_PATH)
    logger.info("Saved model to %s and preprocessor to %s", MODEL_PATH, PREPROCESSOR_PATH)


def train_pipeline() -> Dict[str, Any]:
    ensure_directories()
    configure_logging(LOG_LEVEL)

    if MODEL_PATH.exists() and PREPROCESSOR_PATH.exists():
        logger.info("Existing model and preprocessor detected, skipping training.")
        model = joblib.load(MODEL_PATH)
        pre_meta = joblib.load(PREPROCESSOR_PATH)
        return {
            "model": model,
            "preprocessor": pre_meta["preprocessor"],
            "meta": pre_meta["meta"],
        }

    df = load_data(DATA_PATH)

    X, y, preprocessor, meta = build_preprocessing_pipeline(
        df=df,
        text_col=TEXT_COLUMN,
        target_col=TARGET_COLUMN,
    )

    results = train_models(X, y)
    best_name = select_best_model(results)
    save_artifacts(best_name, results, preprocessor, meta)

    return {
        "best_model_name": best_name,
        "results": results,
        "preprocessor": preprocessor,
        "meta": meta,
    }


if __name__ == "__main__":
    train_pipeline()
