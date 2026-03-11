import logging
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import FunctionTransformer


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _ensure_text_series(x):
    """
    Ensure the input to text preprocessing steps is a 1D pandas Series of strings.
    ColumnTransformer often passes a 2D DataFrame (n_samples, 1) to the
    text pipeline, which does not have a .str accessor. This helper
    normalises the input so downstream lambdas can safely use .str.
    """
    if isinstance(x, pd.Series):
        s = x
    elif isinstance(x, pd.DataFrame):
        # Take the single text column
        s = x.iloc[:, 0]
    else:
        # Fallback for numpy arrays or other array-like inputs
        s = pd.Series(x)
    return s.astype(str)


def _lowercase_text(x: Any) -> pd.Series:
    """Top-level function so it can be pickled."""
    s = _ensure_text_series(x)
    return s.str.lower()


def _remove_punct_text(x: Any) -> pd.Series:
    """Top-level function so it can be pickled."""
    s = _ensure_text_series(x)
    return s.str.replace(r"[^\w\s]", " ", regex=True)


def detect_schema(df: pd.DataFrame, text_col: str, target_col: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    logger.info("Detecting dataset schema dynamically.")

    schema = {
        "text_col": None,
        "target_col": None,
        "numeric_cols": [],
        "categorical_cols": [],
        "id_like_cols": [],
        "high_null_cols": [],
        "constant_cols": [],
        "dropped_cols": [],
    }

    if text_col in df.columns:
        schema["text_col"] = text_col
    else:
        logger.warning("Configured text column '%s' not found. Auto-detecting.", text_col)
        text_candidates = [
            c
            for c in df.columns
            if df[c].dtype == "object" and c != target_col
        ]
        schema["text_col"] = text_candidates[0] if text_candidates else None

    if target_col in df.columns:
        schema["target_col"] = target_col
    else:
        logger.error("Configured target column '%s' not found in dataset.", target_col)
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    n_rows = len(df)

    for col in df.columns:
        if col == schema["target_col"]:
            continue

        unique_ratio = df[col].nunique(dropna=True) / max(n_rows, 1)
        null_ratio = df[col].isna().mean()

        if unique_ratio > 0.95:
            schema["id_like_cols"].append(col)
        if null_ratio > 0.5:
            schema["high_null_cols"].append(col)
        if df[col].nunique(dropna=True) <= 1:
            schema["constant_cols"].append(col)

    drop_cols = set(schema["id_like_cols"] + schema["high_null_cols"] + schema["constant_cols"])
    if schema["text_col"] in drop_cols:
        drop_cols.remove(schema["text_col"])
    if schema["target_col"] in drop_cols:
        drop_cols.remove(schema["target_col"])

    schema["dropped_cols"] = sorted(list(drop_cols))

    logger.info("Dropping columns: %s", schema["dropped_cols"])
    logger.info("ID-like columns: %s", schema["id_like_cols"])
    logger.info("High-null columns (>50%%): %s", schema["high_null_cols"])
    logger.info("Constant columns: %s", schema["constant_cols"])

    df_processed = df.drop(columns=schema["dropped_cols"], errors="ignore").copy()

    for col in df_processed.columns:
        if col in (schema["target_col"], schema["text_col"]):
            continue
        if pd.api.types.is_numeric_dtype(df_processed[col]):
            schema["numeric_cols"].append(col)
        else:
            schema["categorical_cols"].append(col)

    logger.info("Numeric columns: %s", schema["numeric_cols"])
    logger.info("Categorical columns: %s", schema["categorical_cols"])

    return schema, df_processed


def _build_text_preprocessor(df: pd.DataFrame, text_col: str) -> Pipeline:
    has_upper = df[text_col].fillna("").str.contains(r"[A-Z]").any()
    has_punct = df[text_col].fillna("").str.contains(r"[^\w\s]").any()
    n_rows = len(df)

    pre_steps: List[Tuple[str, BaseEstimator]] = []

    if has_upper:
        pre_steps.append(
            (
                "lowercase",
                FunctionTransformer(_lowercase_text, validate=False),
            )
        )
    if has_punct:
        pre_steps.append(
            (
                "remove_punct",
                FunctionTransformer(_remove_punct_text, validate=False),
            )
        )
    if n_rows > 500:
        from sklearn.feature_extraction.text import TfidfVectorizer

        pre_steps.append(
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=5000,
                ),
            )
        )

    if not pre_steps:
        from sklearn.feature_extraction.text import TfidfVectorizer

        pre_steps.append(
            ("tfidf", TfidfVectorizer(max_features=5000)),
        )

    logger.info(
        "Text preprocessing configured with steps: %s",
        [name for name, _ in pre_steps],
    )

    return Pipeline(pre_steps)


def build_preprocessing_pipeline(
    df: pd.DataFrame, text_col: str, target_col: str
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer, dict]:
    logger.info("Building preprocessing pipeline.")

    schema, df_processed = detect_schema(df, text_col, target_col)

    # Drop rows where the target is missing before any encoding
    target_name = schema["target_col"]
    if target_name not in df_processed.columns:
        raise ValueError(f"Target column '{target_name}' missing after preprocessing.")

    before_rows = len(df_processed)
    df_processed = df_processed[df_processed[target_name].notna()].copy()
    after_rows = len(df_processed)
    if after_rows < before_rows:
        logger.info(
            "Dropped %d rows with NaN in target column '%s'.",
            before_rows - after_rows,
            target_name,
        )
    if after_rows == 0:
        raise ValueError(
            f"All rows have NaN in target column '{target_name}'. "
            "Cannot train a model."
        )

    y_raw = df_processed[target_name]
    X_raw = df_processed.drop(columns=[target_name], errors="ignore")

    target_encoder = None
    if not pd.api.types.is_numeric_dtype(y_raw):
        logger.info("Target column is non-numeric. Applying LabelEncoder.")
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw.astype(str))
    else:
        if y_raw.nunique() > 20:
            logger.info(
                "Numeric target with high cardinality detected. Binning into quantiles for classification."
            )
            y = pd.qcut(y_raw.rank(method="first"), q=4, labels=False)
        else:
            y = y_raw.values

    transformers = []

    if schema["text_col"] and schema["text_col"] in X_raw.columns:
        text_pipeline = _build_text_preprocessor(X_raw, schema["text_col"])
        transformers.append(("text", text_pipeline, [schema["text_col"]]))
    else:
        logger.info("No text column detected or available for preprocessing.")

    if schema["numeric_cols"]:
        num_imputer = SimpleImputer(strategy="median")
        transformers.append(("numeric", num_imputer, schema["numeric_cols"]))

    if schema["categorical_cols"]:
        low_card, high_card = [], []
        for col in schema["categorical_cols"]:
            if X_raw[col].nunique(dropna=True) <= 20:
                low_card.append(col)
            else:
                high_card.append(col)

        if low_card:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            cat_pipeline = Pipeline([("imputer", cat_imputer), ("ohe", ohe)])
            transformers.append(("categorical_low", cat_pipeline, low_card))

        if high_card:
            logger.info(
                "High cardinality categorical columns detected (label encoded separately): %s",
                high_card,
            )
            for col in high_card:
                le = LabelEncoder()
                X_raw[col] = le.fit_transform(X_raw[col].astype(str).fillna("NA"))

            num_imputer_high = SimpleImputer(strategy="median")
            transformers.append(
                ("categorical_high", num_imputer_high, high_card)
            )

    if not transformers:
        logger.error("No valid feature columns available for preprocessing.")
        raise ValueError("No valid feature columns available for preprocessing.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    logger.info("Fitting preprocessing pipeline.")
    X = preprocessor.fit_transform(X_raw)

    meta = {
        "schema": schema,
        "target_encoder": target_encoder,
    }

    return X, np.asarray(y), preprocessor, meta
