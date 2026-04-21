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

from utils.features import engineer_features, FEATURE_COLS, TARGET_COL

# ─── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban EV Demand Predictor",
    page_icon="⚡",
    layout="wide",
)

# ─── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.stApp { background: #0d1117; color: #e6edf3; }

section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
}
.metric-card .label { font-size: 0.78rem; color: #8b949e; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
.metric-card .value { font-family: 'Space Mono', monospace; font-size: 1.9rem; color: #58a6ff; font-weight: 700; }
.metric-card .sub   { font-size: 0.75rem; color: #8b949e; margin-top: 4px; }

.pred-highlight {
    background: linear-gradient(135deg, #1f2d3d 0%, #0d1f2d 100%);
    border: 1px solid #58a6ff44;
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
    margin-bottom: 1.5rem;
}
.pred-highlight .zone-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.1em; }
.pred-highlight .pred-value { font-family: 'Space Mono', monospace; font-size: 3rem; color: #58a6ff; font-weight: 700; line-height: 1.1; }
.pred-highlight .pred-unit  { font-size: 1rem; color: #8b949e; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8b949e;
    border-bottom: 1px solid #30363d;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

div[data-testid="stSelectbox"] > label,
div[data-testid="stSlider"] > label     { color: #8b949e !important; font-size: 0.82rem; }

div[data-testid="stAlert"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─── helpers ──────────────────────────────────────────────────────────────────

MODEL_DIR  = "models"
DATA_PATH  = "data/ev_charging_data.csv"

@st.cache_resource
def load_artifacts():
    """Load models, scaler, metrics, and test predictions."""
    def _load(name):
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    artefacts = {
        "lr":      _load("linear_regression.pkl"),
        "xgb":     _load("xgboost.pkl"),
        "scaler":  _load("scaler.pkl"),
        "metrics": _load("metrics.pkl"),
    }
    pred_path = os.path.join(MODEL_DIR, "test_predictions.csv")
    artefacts["test_df"] = pd.read_csv(pred_path) if os.path.exists(pred_path) else None
    return artefacts


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def models_ready(art):
    return art["lr"] is not None and art["scaler"] is not None


def predict_future(art, zone, base_row, n_hours, model_key):
    """
    Iteratively predict n_hours into the future from the last known record.
    Returns list of (timestamp, demand) tuples.
    """
    results = []
    last_demand = base_row[TARGET_COL]
    last_ts     = base_row["timestamp"]

    # Rolling history for lag features
    lag_buffer = [last_demand] * 24

    for h in range(1, n_hours + 1):
        next_ts = last_ts + pd.Timedelta(hours=h)

        row = pd.DataFrame([{
            "hour":               next_ts.hour,
            "day_of_week":        next_ts.dayofweek,
            "month":              next_ts.month,
            "temperature":        base_row["temperature"] + np.random.normal(0, 1),
            "traffic_index":      base_row["traffic_index"] * (0.9 + 0.2 * np.random.random()),
            "demand_lag1":        lag_buffer[-1],
            "demand_lag24":       lag_buffer[-24] if len(lag_buffer) >= 24 else last_demand,
            "demand_rolling_mean_3h": np.mean(lag_buffer[-3:]),
        }])

        X = row[FEATURE_COLS]

        if model_key == "xgb" and art["xgb"]:
            pred = float(art["xgb"].predict(X)[0])
        else:
            pred = float(art["lr"].predict(art["scaler"].transform(X))[0])

        pred = max(0, pred)
        lag_buffer.append(pred)
        results.append({"timestamp": next_ts, "predicted_demand_kwh": round(pred, 2)})

    return pd.DataFrame(results)


# ─── layout ───────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("## ⚡ Urban EV Charging Demand Predictor")
    st.markdown("<p style='color:#8b949e; margin-top:-8px;'>Machine-learning forecast for city zone planners</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Load data & models
    art  = load_artifacts()
    data = load_data()

    if data is None:
        st.error("Dataset not found. Run `python data/generate_data.py` first.")
        st.stop()

    if not models_ready(art):
        st.warning("⚠️ Models not trained yet. Run `python train_model.py` first.")
        if st.button("🚀 Train models now"):
            with st.spinner("Training models…"):
                import subprocess
                subprocess.run([sys.executable, "train_model.py"], check=True)
            st.success("Training complete! Refresh the page.")
        st.stop()

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("")

        zones        = sorted(data["zone_id"].unique().tolist())
        selected_zone = st.selectbox("🗺️  Select Zone", zones)

        model_choice  = st.selectbox(
            "🤖  Model",
            ["XGBoost", "Linear Regression"] if art["xgb"] else ["Linear Regression"]
        )
        model_key = "xgb" if model_choice == "XGBoost" else "lr"

        n_hours = st.slider("⏱️  Hours to predict ahead", min_value=1, max_value=48, value=12, step=1)
        history_hours = st.slider("📜  Historical hours to display", 24, 168, 72, step=24)

        st.markdown("---")
        st.markdown("<p style='font-size:0.75rem; color:#8b949e;'>Urban EV Charging MVP<br>For city planning demo use only.</p>", unsafe_allow_html=True)

    # ── Slice data for selected zone ──────────────────────────────────────────
    zone_df  = data[data["zone_id"] == selected_zone].copy()
    zone_df  = zone_df.sort_values("timestamp")
    hist_df  = zone_df.tail(history_hours)
    last_row = zone_df.iloc[-1]

    # ── Run prediction ────────────────────────────────────────────────────────
    future_df = predict_future(art, selected_zone, last_row, n_hours, model_key)
    avg_pred  = future_df["predicted_demand_kwh"].mean()
    peak_pred = future_df["predicted_demand_kwh"].max()

    # ─── Top metrics row ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Avg Predicted Demand</div>
            <div class="value">{avg_pred:.0f}</div>
            <div class="sub">kWh / hour</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Peak Predicted</div>
            <div class="value">{peak_pred:.0f}</div>
            <div class="sub">kWh / hour</div>
        </div>""", unsafe_allow_html=True)

    metrics = art.get("metrics", {})
    model_key_map = {
    "Linear Regression": "LinearRegression",
    "XGBoost": "XGBoost"
}

model_metrics = metrics.get(model_key_map[model_choice], {})
    mae  = model_metrics.get("mae",  "—")
    rmse = model_metrics.get("rmse", "—")

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Model MAE</div>
            <div class="value">{mae if mae == '—' else f'{mae:.1f}'}</div>
            <div class="sub">Mean Abs Error (kWh)</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Model RMSE</div>
            <div class="value">{rmse if rmse == '—' else f'{rmse:.1f}'}</div>
            <div class="sub">Root Mean Sq Error</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Main chart ───────────────────────────────────────────────────────────
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_df["timestamp"],
        y=hist_df[TARGET_COL],
        mode="lines",
        name="Historical Demand",
        line=dict(color="#8b949e", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(139,148,158,0.07)",
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=future_df["timestamp"],
        y=future_df["predicted_demand_kwh"],
        mode="lines+markers",
        name=f"Predicted ({model_choice})",
        line=dict(color="#58a6ff", width=2.5, dash="dot"),
        marker=dict(size=5, color="#58a6ff"),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
    ))

    # Divider at forecast start
    fig.add_annotation(
    x=last_row["timestamp"],
    y=1,
    xref="x",
    yref="paper",
    text="Now",
    showarrow=False,
    font=dict(color="#3fb950", size=11),
)

    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="DM Sans"),
        title=dict(text=f"Demand Forecast — {selected_zone}", font=dict(color="#e6edf3", size=16, family="Space Mono")),
        xaxis=dict(showgrid=True, gridcolor="#21262d", zeroline=False, tickfont=dict(color="#8b949e")),
        yaxis=dict(showgrid=True, gridcolor="#21262d", zeroline=False, title="Energy Demand (kWh)", tickfont=dict(color="#8b949e")),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=50, b=0),
        height=420,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ─── Bottom panels ────────────────────────────────────────────────────────
    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("<div class='section-title'>📋 Hourly Forecast Breakdown</div>", unsafe_allow_html=True)
        display_df = future_df.copy()
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        display_df.columns = ["Timestamp", "Predicted Demand (kWh)"]
        st.dataframe(display_df, use_container_width=True, height=280)

    with right:
        st.markdown("<div class='section-title'>📊 Model Accuracy Summary</div>", unsafe_allow_html=True)

        for mname, mvals in metrics.items():
            badge_color = "#3fb950" if mname == model_choice else "#8b949e"
            st.markdown(f"""
            <div style='background:#161b22; border:1px solid #30363d; border-radius:8px;
                        padding:14px 18px; margin-bottom:10px;'>
              <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span style='font-family:Space Mono,monospace; font-size:0.82rem; color:{badge_color};'>{mname}</span>
                {'<span style="font-size:0.65rem; background:#3fb95022; color:#3fb950; padding:2px 8px; border-radius:4px; border:1px solid #3fb95044;">active</span>' if mname == model_choice else ''}
              </div>
              <div style='margin-top:8px; display:flex; gap:24px;'>
                <div><span style='font-size:0.72rem; color:#8b949e;'>MAE</span><br>
                     <span style='font-family:Space Mono,monospace; font-size:1.1rem; color:#e6edf3;'>{mvals['mae']:.2f}</span></div>
                <div><span style='font-size:0.72rem; color:#8b949e;'>RMSE</span><br>
                     <span style='font-family:Space Mono,monospace; font-size:1.1rem; color:#e6edf3;'>{mvals['rmse']:.2f}</span></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='margin-top:8px; padding:12px; background:#161b22; border:1px solid #30363d;
                    border-radius:8px; font-size:0.75rem; color:#8b949e; line-height:1.6;'>
          <strong style='color:#e6edf3;'>Dataset:</strong> 90 days × 4 zones × 24h<br>
          <strong style='color:#e6edf3;'>Features:</strong> Hour, day, month, temperature,
          traffic index, lag-1h, lag-24h, rolling 3h mean<br>
          <strong style='color:#e6edf3;'>Split:</strong> 80% train / 20% test (time-ordered)
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
