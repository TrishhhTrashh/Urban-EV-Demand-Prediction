"""
generate_data.py
Generates synthetic hourly EV charging demand data for multiple city zones.
"""

import numpy as np
import pandas as pd
import os

def generate_ev_data(
    zones=["Zone_A", "Zone_B", "Zone_C", "Zone_D"],
    days=90,
    seed=42,
    output_path="data/ev_charging_data.csv"
):
    np.random.seed(seed)
    records = []

    start_date = pd.Timestamp("2024-01-01")
    hours = pd.date_range(start=start_date, periods=days * 24, freq="h")

    zone_profiles = {
        "Zone_A": {"base": 120, "peak_hour": 18, "temp_sensitivity": 0.8},
        "Zone_B": {"base": 80,  "peak_hour": 8,  "temp_sensitivity": 0.5},
        "Zone_C": {"base": 150, "peak_hour": 17, "temp_sensitivity": 1.0},
        "Zone_D": {"base": 60,  "peak_hour": 12, "temp_sensitivity": 0.6},
    }

    for zone in zones:
        profile = zone_profiles[zone]

        for ts in hours:
            hour        = ts.hour
            day_of_week = ts.dayofweek   # 0=Mon … 6=Sun
            month       = ts.month

            # Temperature: seasonal sine wave + noise
            seasonal_temp = 15 + 12 * np.sin((month - 3) * np.pi / 6)
            temperature   = seasonal_temp + np.random.normal(0, 3)

            # Traffic: peaks at morning/evening rush, lower on weekends
            traffic_base   = 30 + 60 * np.exp(-0.5 * ((hour - profile["peak_hour"]) / 3) ** 2)
            weekend_factor = 0.6 if day_of_week >= 5 else 1.0
            traffic_index  = max(0, traffic_base * weekend_factor + np.random.normal(0, 5))

            # Demand: base + hourly + traffic + temp effects + noise
            hour_factor  = 1.0 + 0.5 * np.sin((hour - 6) * np.pi / 12)
            temp_effect  = profile["temp_sensitivity"] * max(0, temperature - 20)
            demand       = (
                profile["base"] * hour_factor
                + 0.4 * traffic_index
                + temp_effect
                + np.random.normal(0, 10)
            )
            demand = max(0, demand)

            records.append({
                "timestamp":                  ts,
                "zone_id":                    zone,
                "temperature":                round(temperature, 2),
                "traffic_index":              round(traffic_index, 2),
                "historical_energy_demand_kwh": round(demand, 2),
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅  Dataset saved → {output_path}  ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    generate_ev_data()
