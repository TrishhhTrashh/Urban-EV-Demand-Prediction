"""
app.py  ─  Urban EV Charging Demand Predictor
Run: streamlit run app.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from utils.features import FEATURE_COLS, TARGET_COL

# ─── CONFIG ─────────────────────────────────────────────────────
st.set_page_config(page_title="Urban EV Demand Predictor", page_icon="⚡", layout="wide")

MODEL_DIR = "models"
DATA_PATH = "data/ev_charging_data.csv"

# ─── LOADERS ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    def _load(name):
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    return {
        "lr": _load("linear_regression.pkl"),
        "xgb": _load("xgboost.pkl"),
        "scaler": _load("scaler.pkl"),
        "metrics": _load("metrics.pkl"),
    }


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def predict_future(art, base_row, n_hours, model_key):
    results = []
    last_ts = base_row["timestamp"]
    last_demand = base_row[TARGET_COL]

    lag_buffer = [last_demand] * 24

    for h in range(1, n_hours + 1):
        next_ts = last_ts + pd.Timedelta(hours=h)

        row = pd.DataFrame([{
            "hour": next_ts.hour,
            "day_of_week": next_ts.dayofweek,
            "month": next_ts.month,
            "temperature": base_row["temperature"],
            "traffic_index": base_row["traffic_index"],
            "demand_lag1": lag_buffer[-1],
            "demand_lag24": lag_buffer[-24],
            "demand_rolling_mean_3h": np.mean(lag_buffer[-3:])
        }])

        X = row[FEATURE_COLS]

        if model_key == "xgb" and art["xgb"]:
            pred = float(art["xgb"].predict(X)[0])
        else:
            pred = float(art["lr"].predict(art["scaler"].transform(X))[0])

        lag_buffer.append(pred)
        results.append({"timestamp": next_ts, "predicted_demand_kwh": pred})

    return pd.DataFrame(results)


# ─── MAIN ───────────────────────────────────────────────────────
def main():
    st.title("⚡ Urban EV Charging Demand Predictor")

    art = load_artifacts()
    data = load_data()

    if data is None:
        st.error("Run data generator first.")
        return

    zones = sorted(data["zone_id"].unique())
    selected_zone = st.sidebar.selectbox("Zone", zones)

    model_choice = st.sidebar.selectbox("Model", ["XGBoost", "Linear Regression"])
    model_key = "xgb" if model_choice == "XGBoost" else "lr"

    hours = st.sidebar.slider("Prediction Hours", 1, 48, 12)

    zone_df = data[data["zone_id"] == selected_zone].sort_values("timestamp")
    last_row = zone_df.iloc[-1]

    future_df = predict_future(art, last_row, hours, model_key)

    avg_pred = future_df["predicted_demand_kwh"].mean()
    peak_pred = future_df["predicted_demand_kwh"].max()

    # ─── FIXED METRICS ─────────────────────────────
    metrics = art.get("metrics", {})

    model_key_map = {
        "Linear Regression": "LinearRegression",
        "XGBoost": "XGBoost"
    }

    model_metrics = metrics.get(model_key_map.get(model_choice, ""), {})

    mae = model_metrics.get("mae", "—")
    rmse = model_metrics.get("rmse", "—")

    # ─── DISPLAY ──────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Avg Demand", f"{avg_pred:.2f}")
    col2.metric("Peak Demand", f"{peak_pred:.2f}")
    col3.metric("MAE", f"{mae:.2f}" if mae != "—" else "—")
    col4.metric("RMSE", f"{rmse:.2f}" if rmse != "—" else "—")

    # ─── GRAPH ────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=zone_df["timestamp"],
        y=zone_df[TARGET_COL],
        name="Historical"
    ))

    fig.add_trace(go.Scatter(
        x=future_df["timestamp"],
        y=future_df["predicted_demand_kwh"],
        name="Predicted"
    ))

    fig.add_shape(
        type="line",
        x0=last_row["timestamp"],
        x1=last_row["timestamp"],
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="green", dash="dash")
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
