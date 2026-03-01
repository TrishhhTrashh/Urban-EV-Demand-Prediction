"""
utils/features.py
Shared feature-engineering helpers used by training and the Streamlit app.
"""

import pandas as pd
import numpy as np


FEATURE_COLS = [
    "hour", "day_of_week", "month",
    "temperature", "traffic_index",
    "demand_lag1", "demand_lag24",
    "demand_rolling_mean_3h",
]

TARGET_COL = "energy_demand_kwh"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time, lag, and rolling features. Drops rows with NaN lags."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)

    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month

    df["demand_lag1"]  = df.groupby("zone_id")[TARGET_COL].shift(1)
    df["demand_lag24"] = df.groupby("zone_id")[TARGET_COL].shift(24)
    df["demand_rolling_mean_3h"] = (
        df.groupby("zone_id")[TARGET_COL]
        .transform(lambda x: x.shift(1).rolling(3).mean())
    )

    df.dropna(subset=FEATURE_COLS, inplace=True)
    return df
