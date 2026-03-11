import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PROJECT_TITLE = "Daily Market Data Prediction (Stock Market Forecasting)"

TEXT_COLUMN = (
    "nifty50_high, nifty50_low, nifty50_open, nifty50_volume\n"
    "sp500_close, sp500_high, sp500_low, sp500_open, sp500_volume\n"
    "usd_inr_close, usd_inr_high, usd_inr_low, usd_inr_open, usd_inr_volume\n"
    "gold_close, gold_high, gold_low, gold_open, gold_volume\n"
    "brent_close, brent_high, brent_low, brent_open, brent_volume"
)

TARGET_COLUMN = "nifty50_close"

DATA_PATH = BASE_DIR / "data" / "dataset.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO")


def ensure_directories() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "assets").mkdir(parents=True, exist_ok=True)


ensure_directories()
