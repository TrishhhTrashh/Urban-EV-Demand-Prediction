# ⚡ Urban EV Charging Demand Predictor

A lightweight ML-powered MVP web app for predicting hourly EV charging demand
across city zones — built for demonstration to city planners.

---

## 📁 Project Structure

```
ev_charging_predictor/
├── app.py                    # Streamlit web app
├── train_model.py            # Model training script
├── requirements.txt
├── data/
│   ├── generate_data.py      # Synthetic dataset generator
│   └── ev_charging_data.csv  # Generated automatically
├── models/                   # Saved models (generated after training)
│   ├── linear_regression.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   ├── metrics.pkl
│   └── test_predictions.csv
└── utils/
    └── features.py           # Shared feature engineering helpers
```

---

## 🚀 Quick Start

### 1. Clone / navigate to the project folder

```bash
cd ev_charging_predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the synthetic dataset

```bash
python data/generate_data.py
```

This creates `data/ev_charging_data.csv` — 90 days × 4 zones × 24 hours = **8,640 rows**.

### 5. Train the models

```bash
python train_model.py
```

Trains **Linear Regression** and **XGBoost** and saves artefacts to `models/`.
You'll see MAE and RMSE printed to the terminal.

### 6. Launch the Streamlit app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

---

## 🖥️ App Features

| Feature | Details |
|---|---|
| Zone selector | Zone_A, Zone_B, Zone_C, Zone_D |
| Model selector | XGBoost or Linear Regression |
| Forecast horizon | 1–48 hours ahead (slider) |
| Historical window | 24–168 hours of history |
| Interactive chart | Historical demand + forecast overlay |
| Metrics panel | MAE & RMSE per model |
| Hourly table | Timestamped predicted kWh breakdown |

---

## 📊 Dataset Schema

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Hourly UTC timestamp |
| `zone_id` | string | City zone identifier |
| `temperature` | float | Ambient temperature (°C) |
| `traffic_index` | float | Relative traffic intensity (0–100) |
| `historical_energy_demand_kwh` | float | EV charging energy consumed (kWh) |

---

## 🤖 Model Details

**Features used:**
- Time features: `hour`, `day_of_week`, `month`
- Context: `temperature`, `traffic_index`
- Lag features: `demand_lag1` (1 hour ago), `demand_lag24` (24 hours ago)
- Rolling: `demand_rolling_mean_3h`

**Models:**
- `LinearRegression` — fast baseline, scaled with StandardScaler
- `XGBRegressor` — gradient-boosted trees, typically lower error

**Evaluation:** 80/20 time-based train/test split, metrics = MAE + RMSE on held-out test set.

---

## ⚙️ Customisation Tips

- **Add zones:** Edit `zone_profiles` in `data/generate_data.py`
- **More history:** Increase `days=90` in `generate_ev_data()`
- **Tune XGBoost:** Adjust `n_estimators`, `max_depth`, `learning_rate` in `train_model.py`
- **Add LSTM:** Replace the XGBoost block with a `keras.Sequential` LSTM; use the same `FEATURE_COLS`

---

*MVP built for city planning demonstration — not for production use.*
