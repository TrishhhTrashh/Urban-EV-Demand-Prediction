"""
train_model.py
Trains XGBoost and Linear Regression models on the synthetic EV dataset.
Saves trained models + scaler to the models/ directory.
Run: python train_model.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  xgboost not installed – skipping XGBoost model.")

# Allow running from project root
sys.path.insert(0, os.path.dirname(__file__))
from utils.features import engineer_features, FEATURE_COLS, TARGET_COL

DATA_PATH  = "data/ev_charging_data.csv"
MODEL_DIR  = "models"
TRAIN_FRAC = 0.8


# ─── helpers ──────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r     = rmse(y_test, preds)
    print(f"  {name:<20}  MAE={mae:.2f}  RMSE={r:.2f}")
    return {"mae": round(mae, 4), "rmse": round(r, 4)}


def save(obj, filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, filename), "wb") as f:
        pickle.dump(obj, f)
    print(f"  💾  Saved → models/{filename}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Load & generate data if missing
    if not os.path.exists(DATA_PATH):
        print("📦  Dataset not found. Generating…")
        from data.generate_data import generate_ev_data
        generate_ev_data()

    df = pd.read_csv(DATA_PATH)
    print(f"📂  Loaded {len(df):,} rows from {DATA_PATH}")

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Train / test split (time-based)
    split_idx = int(len(df) * TRAIN_FRAC)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test,  y_test  = test[FEATURE_COLS],  test[TARGET_COL]

    # 4. Scale for Linear Regression
    scaler   = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("\n📊  Training results:")
    metrics = {}

    # 5a. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    metrics["LinearRegression"] = evaluate("Linear Regression", lr, X_test_s, y_test)
    save(lr,     "linear_regression.pkl")
    save(scaler, "scaler.pkl")

    # 5b. XGBoost
    if XGBOOST_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbosity=0,
        )
        xgb.fit(X_train, y_train)
        metrics["XGBoost"] = evaluate("XGBoost", xgb, X_test, y_test)
        save(xgb, "xgboost.pkl")

    # 6. Save metrics & test slice for the app
    with open(os.path.join(MODEL_DIR, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    # Save a small test slice so the app can plot historical vs predicted
    test_sample = test.copy()
    if XGBOOST_AVAILABLE:
        test_sample["predicted_xgb"] = xgb.predict(test[FEATURE_COLS])
    test_sample["predicted_lr"] = lr.predict(scaler.transform(test[FEATURE_COLS]))
    test_sample.to_csv(os.path.join(MODEL_DIR, "test_predictions.csv"), index=False)

    print("\n✅  Training complete. All artefacts saved to models/")


if __name__ == "__main__":
    main()
