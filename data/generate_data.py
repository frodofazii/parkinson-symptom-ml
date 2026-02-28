import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

data = pd.DataFrame({
    "tremor_intensity": np.random.normal(0.5, 0.15, n),
    "movement_speed": np.random.normal(1.0, 0.2, n),
    "tap_accuracy": np.random.normal(0.7, 0.1, n),
    "sleep_quality": np.random.normal(0.6, 0.2, n),
})

data["symptom_severity"] = (
    0.4 * data["tremor_intensity"]
    - 0.3 * data["movement_speed"]
    + 0.2 * data["tap_accuracy"]
    + np.random.normal(0, 0.05, n)
) > 0.3

data["symptom_severity"] = data["symptom_severity"].astype(int)
data.to_csv("synthetic_sensor_data.csv", index=False)