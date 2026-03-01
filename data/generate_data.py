import os
import numpy as np
import pandas as pd

def generate_ev_data():
    os.makedirs("data", exist_ok=True)

    np.random.seed(42)

    hours = 1000
    zones = 3   # simulate 3 city zones

    rows = []

    for zone_id in range(1, zones + 1):
        timestamps = pd.date_range(start="2024-01-01", periods=hours, freq="h")
        temperature = np.random.normal(25, 5, hours)
        traffic_index = np.random.uniform(0.5, 1.5, hours)

        demand = (
            50
            + 5 * zone_id
            + 2 * temperature
            + 30 * traffic_index
            + np.random.normal(0, 10, hours)
        )

        df_zone = pd.DataFrame({
            "zone_id": zone_id,
            "timestamp": timestamps,
            "temperature": temperature,
            "traffic_index": traffic_index,
            "energy_demand_kwh": demand
        })

        rows.append(df_zone)

    final_df = pd.concat(rows)
    final_df.to_csv("data/ev_charging_data.csv", index=False)

    print("✅ Synthetic dataset with zones generated.")