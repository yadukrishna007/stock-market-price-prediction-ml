## Daily Market Data Prediction & Ticket Intelligence

This project is a small end‑to‑end ML + LLM system built with **Streamlit**. It:

- **Trains a classifier** on historical market data (Nifty50, S&P 500, USD/INR, Gold, Brent).
- **Classifies text “tickets”** into classes learned from the data.
- **Generates LLM‑powered responses** for support tickets.
- Provides **analytics**, including a confusion matrix and feature importance.

Everything runs locally in a virtual environment and is driven by a `dataset.csv` file.

---

## 1. Project structure

- `app.py` – Streamlit web app (dashboard, classifier, analytics, admin panel).
- `train.py` – Training pipeline; builds preprocessing + multiple models, picks the best and saves it.
- `predict.py` – Loads the saved model and preprocessor and exposes prediction helpers.
- `utils.py` – Schema detection and preprocessing pipeline (numeric, categorical, text).
- `hf_api.py` – Simple wrapper around a Hugging Face LLM API for AI responses.
- `config.py` – Paths and configuration (project title, target column, data path, etc.).
- `data/dataset.csv` – Market data used for training and analytics.
- `models/model.pkl` & `models/preprocessor.pkl` – Saved model and preprocessing artifacts (created after training).
- `assets/style.css` – Custom CSS to make the UI look nicer.
- `requirements.txt` – Python dependencies.

---

## 2. Prerequisites

- **Python 3.10+** (3.13 is fine, as you’re already using it).
- **Internet access** if you want the AI Response Generator (Hugging Face API) to work.
- A terminal (PowerShell on Windows is fine).

Optional but recommended:

- A **virtual environment** (already set up as `venv/` in this project).

---

## 3. Setup & installation

From the project root:

```powershell
cd "C:\Users\yaduk\Desktop\ml class\stock market"

# (1) Create venv if it doesn't exist
python -m venv venv

# (2) Activate venv
.\venv\Scripts\Activate.ps1

# (3) Install dependencies
pip install -r requirements.txt

# (4) Install extra deps used in analytics
pip install matplotlib
```

If you change the code later and get environment‑related errors, deactivate and reactivate the venv and rerun the installs.

---

## 4. Data – `data/dataset.csv`

The app expects a CSV at:

```text
data/dataset.csv
```

with columns like:

- `date`
- `nifty50_close` (used as **TARGET_COLUMN**)
- `nifty50_high`, `nifty50_low`, `nifty50_open`, `nifty50_volume`
- `sp500_*`, `usd_inr_*`, `gold_*`, `brent_*`

Notes:

- Rows with **missing target** (`nifty50_close` = NaN) are **dropped automatically** during training and analytics.
- You can replace this file with your own data as long as:
  - It includes the target column configured in `config.py` (`TARGET_COLUMN = "nifty50_close"` by default).
  - The file name and path stay the same, or you update `DATA_PATH` in `config.py`.

You can also upload a new CSV from the **Admin Panel** inside the app; it will overwrite `data/dataset.csv`.

---

## 5. Running the Streamlit app

Always run the app with **Streamlit**, not with `python app.py`.

From the project root, with venv activated:

```powershell
cd "C:\Users\yaduk\Desktop\ml class\stock market"
.\venv\Scripts\Activate.ps1

streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) – open it in your browser.

---

## 6. Pages & features

### 6.1 Dashboard

- Shows:
  - **Total rows** in the dataset.
  - **Last training accuracy** (from Model Analytics / training).
  - **Number of features** after preprocessing.
- Also displays:
  - A **sample of the data**.
  - A **histogram of the target** column (e.g. `nifty50_close`).

### 6.2 Ticket Classifier

- Text box to enter a **ticket description**.
- On “Classify Ticket”:
  - The text is wrapped into a `DataFrame`, preprocessed with the saved preprocessor.
  - The trained model predicts a class.
  - If the model supports `predict_proba`, the app computes a **confidence score**.
- Output:
  - Predicted **category label**.
  - Model **confidence**.

### 6.3 AI Response Generator

- Lets you paste a ticket description and click “Generate Response”.
- Uses `hf_api.generate_ai_response(...)` to call a Hugging Face LLM and return a **draft support reply**.
- You may need to:
  - Configure an HF API key (typically via environment variable) inside `hf_api.py`.

### 6.4 Model Analytics

- Uses the **current dataset** and the **saved model + preprocessor** to compute:
  - **Confusion matrix**.
  - **Accuracy**.
  - **Feature importance** (when supported by the model, e.g. RandomForest / XGBoost).
- Behavior:
  - Drops columns marked as “dropped” in the learned schema.
  - Drops rows with missing target values.
  - Applies the **same label encoding / binning** as during training, so predictions and ground truth are comparable.

### 6.5 Admin Panel

- **Upload a new CSV**:
  - Replaces `data/dataset.csv`.
- **Retrain Model**:
  - Runs the full training pipeline:
    - Loads `data/dataset.csv`.
    - Detects schema (numeric, categorical, text).
    - Builds preprocessing pipeline.
    - Trains several models (Logistic Regression, Random Forest, XGBoost).
    - Selects the **best model by accuracy**.
    - Saves `models/model.pkl` and `models/preprocessor.pkl`.
  - Updates `st.session_state["last_accuracy"]` so the dashboard can display training accuracy.

---

## 7. Training from the command line (optional)

You can also train without the UI:

```powershell
cd "C:\Users\yaduk\Desktop\ml class\stock market"
.\venv\Scripts\Activate.ps1

python train.py
```

This will:

- Load `data/dataset.csv`.
- Build the preprocessing pipeline.
- Train multiple models and select the best.
- Save artifacts under `models/`.

If you ever get errors about corrupted `model.pkl` or `preprocessor.pkl` (for example after code changes), you can safely delete them and rerun training:

```powershell
Remove-Item -Force ".\models\model.pkl" -ErrorAction SilentlyContinue
Remove-Item -Force ".\models\preprocessor.pkl" -ErrorAction SilentlyContinue

python train.py
```

---

## 8. Common issues & fixes

- **“Thread 'MainThread': missing ScriptRunContext / Session state does not function when running a script without `streamlit run`”**  
  You ran `python app.py`.  
  → Use `streamlit run app.py` instead.

- **“Dataset not found at ...\data\dataset.csv”**  
  The CSV is missing.  
  → Put your CSV at `data/dataset.csv` or upload it from the Admin Panel.

- **“Input y contains NaN” or “Input y_true contains NaN”**  
  NaNs in the target column.  
  → The code already drops these rows for training and analytics. Make sure your dataset has at least some non‑NaN target values.

- **“Classification metrics can't handle a mix of continuous and multiclass targets”**  
  Happens if target processing in analytics and training don’t match.  
  → The current code mirrors the training logic (label encoding / binning) inside `render_model_analytics`, so this should already be handled.

- **“No module named 'matplotlib'”**  
  Analytics page imports `matplotlib`.  
  → Run `pip install matplotlib` inside your venv.

---

## 9. Customizing the project

- **Change the target**: edit `TARGET_COLUMN` in `config.py` and ensure your CSV has that column.
- **Change model types or hyperparameters**: edit `train.py` (`train_models` function).
- **Change preprocessing logic**: edit `utils.py` (`detect_schema` and `build_preprocessing_pipeline`).
- **Change look & feel**: update `assets/style.css` or add more Streamlit components in `app.py`.

This README should give you everything you need to:

1. Set up the environment.  
2. Run the Streamlit app.  
3. Understand what each page does and how the data flows.  
4. Debug the most common issues quickly.

