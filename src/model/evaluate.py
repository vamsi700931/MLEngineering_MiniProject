from pathlib import Path
import json

import joblib
import mlflow
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset_10000.parquet"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "nyc_taxi_eta_model.joblib"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "models"
    / "evaluation_metrics.json"
)

FEATURE_COLUMNS = [
    "vendor_id",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "store_and_fwd_flag",
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
    "distance_km",
]

TARGET_COLUMN = "trip_duration"


def main() -> None:
    print("Starting model evaluation...")

    # ---------------------------------------------------------
    # MLflow configuration
    # ---------------------------------------------------------
    mlflow.set_tracking_uri(
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    mlflow.set_experiment("NYC Taxi ETA Prediction")

    # ---------------------------------------------------------
    # Load training dataset
    # ---------------------------------------------------------
    df = pd.read_parquet(TRAINING_FILE)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # ---------------------------------------------------------
    # Recreate the same test split used during training
    # ---------------------------------------------------------
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # ---------------------------------------------------------
    # Load trained model
    # ---------------------------------------------------------
    artifact = joblib.load(MODEL_FILE)

    preprocessor = artifact["preprocessor"]
    model = artifact["model"]

    # ---------------------------------------------------------
    # Transform test data and make predictions
    # ---------------------------------------------------------
    X_test_processed = preprocessor.transform(X_test)

    y_pred = model.predict(X_test_processed)

    # ---------------------------------------------------------
    # Calculate evaluation metrics
    # ---------------------------------------------------------
    mae = mean_absolute_error(y_test, y_pred)

    rmse = mean_squared_error(
        y_test,
        y_pred,
    ) ** 0.5

    r2 = r2_score(y_test, y_pred)

    # ---------------------------------------------------------
    # Prepare metrics
    # ---------------------------------------------------------
    metrics = {
        "model": "RandomForestRegressor",
        "records": len(df),
        "test_records": len(y_test),
        "mae_seconds": round(float(mae), 2),
        "rmse_seconds": round(float(rmse), 2),
        "r2": round(float(r2), 4),
    }

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------
    print("\nModel evaluation results:")
    print(f"MAE:  {mae:.2f} seconds")
    print(f"RMSE: {rmse:.2f} seconds")
    print(f"R²:   {r2:.4f}")

    # ---------------------------------------------------------
    # Save evaluation metrics as JSON
    # ---------------------------------------------------------
    METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(METRICS_FILE, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nEvaluation metrics saved to: {METRICS_FILE}")

    # ---------------------------------------------------------
    # Log evaluation metrics to MLflow
    # ---------------------------------------------------------
    with mlflow.start_run(run_name="random_forest_evaluation"):

        mlflow.log_param(
            "model_type",
            "RandomForestRegressor",
        )

        mlflow.log_param(
            "test_records",
            len(y_test),
        )

        mlflow.log_metric(
            "mae_seconds",
            float(mae),
        )

        mlflow.log_metric(
            "rmse_seconds",
            float(rmse),
        )

        mlflow.log_metric(
            "r2",
            float(r2),
        )

        mlflow.log_artifact(
            str(METRICS_FILE),
            artifact_path="evaluation",
        )

        run_id = mlflow.active_run().info.run_id

        print(f"MLflow evaluation Run ID: {run_id}")
        print("Evaluation metrics logged to MLflow.")

    print("Model evaluation completed successfully.")


if __name__ == "__main__":
    main()