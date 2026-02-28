import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_split(path):
    df = pd.read_csv(path)
    X = df.drop("symptom_severity", axis=1)
    y = df["symptom_severity"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )