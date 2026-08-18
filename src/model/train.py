from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset_10000.parquet"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "nyc_taxi_eta_model.joblib"


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
    print("Starting model training pipeline...")

    # ---------------------------------------------------------
    # MLflow experiment configuration
    # ---------------------------------------------------------
    mlflow.set_tracking_uri(
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    mlflow.set_experiment("NYC Taxi ETA Prediction")

    # ---------------------------------------------------------
    # Load training dataset
    # ---------------------------------------------------------
    df = pd.read_parquet(TRAINING_FILE)

    print(f"Training records: {len(df):,}")

    # ---------------------------------------------------------
    # Separate features and target
    # ---------------------------------------------------------
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # ---------------------------------------------------------
    # Feature types
    # ---------------------------------------------------------
    categorical_features = ["store_and_fwd_flag"]

    numerical_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in categorical_features
    ]

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numerical",
                "passthrough",
                numerical_features,
            ),
        ]
    )

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"Training set: {len(X_train):,}")
    print(f"Test set: {len(X_test):,}")

    # ---------------------------------------------------------
    # Start MLflow run
    # ---------------------------------------------------------
    with mlflow.start_run(run_name="random_forest_eta"):

        # Log dataset/training parameters
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("training_records", len(df))
        mlflow.log_param("train_records", len(X_train))
        mlflow.log_param("test_records", len(X_test))
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        # Log model parameters
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 20)
        mlflow.log_param("min_samples_leaf", 2)
        mlflow.log_param("n_jobs", -1)

        # Log feature information
        mlflow.log_param(
            "feature_count",
            len(FEATURE_COLUMNS),
        )

        # -----------------------------------------------------
        # Transform features
        # -----------------------------------------------------
        X_train_processed = preprocessor.fit_transform(X_train)

        # -----------------------------------------------------
        # Train model
        # -----------------------------------------------------
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=20,
            min_samples_leaf = 2,
            n_jobs=-1,
        )

        model.fit(X_train_processed, y_train)

        # -----------------------------------------------------
        # Save preprocessing and model together
        # -----------------------------------------------------
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model_artifact = {
            "preprocessor": preprocessor,
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        }

        joblib.dump(model_artifact, MODEL_FILE)

        # -----------------------------------------------------
        # Log model artifact to MLflow
        # -----------------------------------------------------
        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
        )

        # Log the existing joblib model as an artifact
        mlflow.log_artifact(
            str(MODEL_FILE),
            artifact_path="model_artifact",
        )

        # -----------------------------------------------------
        # Run information
        # -----------------------------------------------------
        run_id = mlflow.active_run().info.run_id

        print(f"MLflow Run ID: {run_id}")
        print(f"Model saved to: {MODEL_FILE}")
        print("MLflow tracking completed successfully.")

    print("Model training pipeline completed successfully.")


if __name__ == "__main__":
    main()