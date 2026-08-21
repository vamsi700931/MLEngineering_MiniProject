from pathlib import Path
import json
import shutil

import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset_10000.parquet"
)

DRIFT_DECISION_FILE = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "drift_decision.json"
)

MODEL_COMPARISON_FILE = (
    PROJECT_ROOT
    / "models"
    / "model_comparison.json"
)

CURRENT_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "nyc_taxi_eta_gradient_boosting_tuned.joblib"
)

CANDIDATE_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "nyc_taxi_eta_retrained_candidate.joblib"
)

RETRAINING_RESULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "retraining_result.json"
)


# ------------------------------------------------------------------
# Features and target
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Retraining configuration
# ------------------------------------------------------------------

TEST_SIZE = 0.20
RANDOM_STATE = 42

# For the project demo, this allows us to demonstrate the
# retraining workflow even when the current drift is moderate.
FORCE_RETRAIN = False


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def load_current_model_metrics() -> dict:
    """
    Read the currently selected model metrics from the
    model comparison artifact.
    """

    if not MODEL_COMPARISON_FILE.exists():
        raise FileNotFoundError(
            f"Model comparison file not found: "
            f"{MODEL_COMPARISON_FILE}"
        )

    with open(
        MODEL_COMPARISON_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        comparison = json.load(file)

    selected_model = comparison.get("selected_model")

    if not selected_model:
        raise ValueError(
            "No selected model found in model_comparison.json."
        )

    return selected_model


def train_candidate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """
    Train the candidate Gradient Boosting model.
    """

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Calculate regression evaluation metrics.
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    return {
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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print("Starting controlled model retraining workflow...")

    # --------------------------------------------------------------
    # Validate required files
    # --------------------------------------------------------------

    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: "
            f"{TRAINING_FILE}"
        )

    if not DRIFT_DECISION_FILE.exists():
        raise FileNotFoundError(
            f"Drift decision file not found: "
            f"{DRIFT_DECISION_FILE}"
        )

    if not MODEL_COMPARISON_FILE.exists():
        raise FileNotFoundError(
            f"Model comparison file not found: "
            f"{MODEL_COMPARISON_FILE}"
        )

    # --------------------------------------------------------------
    # Load drift decision
    # --------------------------------------------------------------

    with open(
        DRIFT_DECISION_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        drift_decision = json.load(file)

    overall_status = drift_decision.get(
        "overall_status",
        "UNKNOWN",
    )

    retraining_recommended = drift_decision.get(
        "retraining_recommended",
        False,
    )

    print(
        f"Current drift status: {overall_status}"
    )

    print(
        f"Retraining recommended by monitoring: "
        f"{retraining_recommended}"
    )

    print(
        f"Explicit retraining trigger: "
        f"{FORCE_RETRAIN}"
    )

    # --------------------------------------------------------------
    # Decide whether retraining should happen
    # --------------------------------------------------------------

    should_retrain = (
        retraining_recommended
        or FORCE_RETRAIN
    )

    if not should_retrain:

        result = {
            "retraining_triggered": False,
            "trigger_reason": (
                "Drift is below the retraining threshold."
            ),
            "drift_status": overall_status,
            "retraining_recommended": False,
            "candidate_model_trained": False,
            "candidate_promoted": False,
            "action": "KEEP_CURRENT_MODEL",
        }

        RETRAINING_RESULT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            RETRAINING_RESULT_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                indent=4,
            )

        print("\nRetraining decision:")
        print("Retraining triggered: False")
        print(
            "Reason: Drift is below the "
            "retraining threshold."
        )
        print("Action: KEEP_CURRENT_MODEL")

        print(
            f"\nResult saved to: "
            f"{RETRAINING_RESULT_FILE}"
        )

        print(
            "Controlled retraining workflow completed."
        )

        return

    # --------------------------------------------------------------
    # Load training data
    # --------------------------------------------------------------

    df = pd.read_parquet(
        TRAINING_FILE
    )

    print(
        f"\nTraining records: {len(df):,}"
    )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # --------------------------------------------------------------
    # Handle categorical feature
    # --------------------------------------------------------------

    X = pd.get_dummies(
        X,
        columns=["store_and_fwd_flag"],
        dtype=float,
    )

    # --------------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(
        f"Training set: {len(X_train):,}"
    )

    print(
        f"Test set: {len(X_test):,}"
    )

    # --------------------------------------------------------------
    # Train candidate
    # --------------------------------------------------------------

    print("\nTraining candidate model...")

    candidate_model = train_candidate_model(
        X_train,
        y_train,
    )

    # --------------------------------------------------------------
    # Evaluate candidate
    # --------------------------------------------------------------

    candidate_metrics = evaluate_model(
        candidate_model,
        X_test,
        y_test,
    )

    print("\nCandidate model evaluation:")
    print(
        f"MAE:  "
        f"{candidate_metrics['mae_seconds']:.2f} seconds"
    )

    print(
        f"RMSE: "
        f"{candidate_metrics['rmse_seconds']:.2f} seconds"
    )

    print(
        f"R²:   "
        f"{candidate_metrics['r2']:.4f}"
    )

    # --------------------------------------------------------------
    # Load current model metrics
    # --------------------------------------------------------------

    current_model_metrics = (
        load_current_model_metrics()
    )

    current_rmse = float(
        current_model_metrics["rmse_seconds"]
    )

    candidate_rmse = float(
        candidate_metrics["rmse_seconds"]
    )

    print("\nModel comparison:")
    print(
        f"Current model RMSE:   "
        f"{current_rmse:.2f} seconds"
    )

    print(
        f"Candidate model RMSE: "
        f"{candidate_rmse:.2f} seconds"
    )

    # --------------------------------------------------------------
    # Promote only if candidate is better
    # --------------------------------------------------------------

    candidate_better = (
        candidate_rmse < current_rmse
    )

    candidate_promoted = False

    if candidate_better:

        joblib.dump(
            candidate_model,
            CANDIDATE_MODEL_FILE,
        )

        candidate_promoted = True

        action = "PROMOTE_RETRAINED_MODEL"

        print(
            "\nCandidate model performed better."
        )

        print(
            "Action: PROMOTE_RETRAINED_MODEL"
        )

        print(
            f"Candidate model saved to: "
            f"{CANDIDATE_MODEL_FILE}"
        )

    else:

        action = "KEEP_CURRENT_MODEL"

        print(
            "\nCandidate model did not improve "
            "the current model."
        )

        print(
            "Action: KEEP_CURRENT_MODEL"
        )

    # --------------------------------------------------------------
    # Save retraining result
    # --------------------------------------------------------------

    result = {
        "retraining_triggered": True,
        "trigger_reason": (
            "Significant drift detected or "
            "explicit retraining trigger enabled."
        ),
        "drift_status": overall_status,
        "candidate_model": {
            "model": "GradientBoostingRegressor",
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_samples_leaf": 3,
            **candidate_metrics,
        },
        "current_model": {
            "model": current_model_metrics.get(
                "model"
            ),
            "experiment": current_model_metrics.get(
                "experiment"
            ),
            "rmse_seconds": current_rmse,
            "mae_seconds": current_model_metrics.get(
                "mae_seconds"
            ),
            "r2": current_model_metrics.get(
                "r2"
            ),
        },
        "candidate_better": candidate_better,
        "candidate_promoted": candidate_promoted,
        "action": action,
    }

    RETRAINING_RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RETRAINING_RESULT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
        )

    print(
        f"\nRetraining result saved to: "
        f"{RETRAINING_RESULT_FILE}"
    )

    print(
        "Controlled retraining workflow "
        "completed successfully."
    )


if __name__ == "__main__":
    main()