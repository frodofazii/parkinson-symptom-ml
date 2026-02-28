# src/train.py
# RandomForest training with fixed random seed for reproducibility

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocess import load_and_split


def train_model():
    # Load and split the dataset
    X_train, X_test, y_train, y_test = load_and_split(
        "data/synthetic_sensor_data.csv"
    )

    # Initialize regression model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    preds = model.predict(X_test)

    # Evaluate using regression metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Print results
    print("Model Evaluation Results:")
    print("-------------------------")
    print(f"MAE  (Mean Absolute Error): {mae:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"R2 Score (Goodness of Fit): {r2:.4f}")