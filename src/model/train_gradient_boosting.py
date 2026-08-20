from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
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

MODEL_FILE = (
    MODEL_DIR
    / "nyc_taxi_eta_gradient_boosting.joblib"
)

METRICS_FILE = (
    MODEL_DIR
    / "gradient_boosting_metrics.json"
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
    print("Starting tuned Gradient Boosting training pipeline...")

    # ---------------------------------------------------------
    # MLflow configuration
    # ---------------------------------------------------------
    mlflow.set_tracking_uri(
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    mlflow.set_experiment(
        "NYC Taxi ETA Prediction"
    )

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
    categorical_features = [
        "store_and_fwd_flag"
    ]

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
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
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
    # Tuned Gradient Boosting hyperparameters
    # ---------------------------------------------------------
    n_estimators = 300
    learning_rate = 0.03
    max_depth = 4
    min_samples_leaf = 3
    random_state = 42

    # ---------------------------------------------------------
    # Start MLflow run
    # ---------------------------------------------------------
    with mlflow.start_run(
        run_name="gradient_boosting_tuned_eta"
    ):

        # -----------------------------------------------------
        # Log experiment parameters
        # -----------------------------------------------------
        mlflow.log_param(
            "model_type",
            "GradientBoostingRegressor",
        )

        mlflow.log_param(
            "experiment_type",
            "tuned_gradient_boosting",
        )

        mlflow.log_param(
            "training_records",
            len(df),
        )

        mlflow.log_param(
            "train_records",
            len(X_train),
        )

        mlflow.log_param(
            "test_records",
            len(X_test),
        )

        mlflow.log_param(
            "test_size",
            0.2,
        )

        mlflow.log_param(
            "random_state",
            random_state,
        )

        # -----------------------------------------------------
        # Log Gradient Boosting hyperparameters
        # -----------------------------------------------------
        mlflow.log_param(
            "n_estimators",
            n_estimators,
        )

        mlflow.log_param(
            "learning_rate",
            learning_rate,
        )

        mlflow.log_param(
            "max_depth",
            max_depth,
        )

        mlflow.log_param(
            "min_samples_leaf",
            min_samples_leaf,
        )

        mlflow.log_param(
            "feature_count",
            len(FEATURE_COLUMNS),
        )

        # -----------------------------------------------------
        # Transform features
        # -----------------------------------------------------
        X_train_processed = preprocessor.fit_transform(
            X_train
        )

        X_test_processed = preprocessor.transform(
            X_test
        )

        # -----------------------------------------------------
        # Train tuned Gradient Boosting model
        # -----------------------------------------------------
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

        model.fit(
            X_train_processed,
            y_train,
        )

        # -----------------------------------------------------
        # Evaluate model
        # -----------------------------------------------------
        y_pred = model.predict(
            X_test_processed
        )

        mae = mean_absolute_error(
            y_test,
            y_pred,
        )

        rmse = mean_squared_error(
            y_test,
            y_pred,
        ) ** 0.5

        r2 = r2_score(
            y_test,
            y_pred,
        )

        # -----------------------------------------------------
        # Log metrics to MLflow
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # Save model artifact
        # -----------------------------------------------------
        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_artifact = {
            "preprocessor": preprocessor,
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "model_type": "GradientBoostingRegressor",
        }

        joblib.dump(
            model_artifact,
            MODEL_FILE,
        )

        # -----------------------------------------------------
        # Log model to MLflow
        # -----------------------------------------------------
        mlflow.sklearn.log_model(
            model,
            name="gradient_boosting_tuned_model",
        )

        # -----------------------------------------------------
        # Log complete joblib artifact
        # -----------------------------------------------------
        mlflow.log_artifact(
            str(MODEL_FILE),
            artifact_path="model_artifact",
        )

        # -----------------------------------------------------
        # Save metrics locally
        # -----------------------------------------------------
        metrics = {
            "model": "GradientBoostingRegressor",
            "experiment": "tuned_gradient_boosting",
            "records": len(df),
            "test_records": len(y_test),
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "mae_seconds": round(
                float(mae),
                2,
            ),
            "rmse_seconds": round(
                float(rmse),
                2,
            ),
            "r2": round(
                float(r2),
                4,
            ),
        }

        with open(
            METRICS_FILE,
            "w",
        ) as file:
            json.dump(
                metrics,
                file,
                indent=4,
            )

        # -----------------------------------------------------
        # Display results
        # -----------------------------------------------------
        run_id = (
            mlflow.active_run()
            .info
            .run_id
        )

        print("\nTuned Gradient Boosting evaluation results:")
        print(
            f"MAE:  {mae:.2f} seconds"
        )
        print(
            f"RMSE: {rmse:.2f} seconds"
        )
        print(
            f"R²:   {r2:.4f}"
        )

        print(
            f"\nMLflow Run ID: {run_id}"
        )

        print(
            f"Model saved to: {MODEL_FILE}"
        )

        print(
            f"Metrics saved to: {METRICS_FILE}"
        )

        print(
            "MLflow tracking completed successfully."
        )

    print(
        "Tuned Gradient Boosting training pipeline "
        "completed successfully."
    )


if __name__ == "__main__":
    main()