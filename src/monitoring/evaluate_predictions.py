from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_LOG_FILE = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "prediction_log.jsonl"
)

TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset_10000.parquet"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "monitoring_metrics.json"
)


def load_prediction_log() -> pd.DataFrame:
    """Load prediction records from the JSONL log."""

    if not PREDICTION_LOG_FILE.exists():
        raise FileNotFoundError(
            f"Prediction log not found: {PREDICTION_LOG_FILE}"
        )

    records = []

    with open(
        PREDICTION_LOG_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return pd.DataFrame(records)


def load_actual_durations() -> pd.DataFrame:
    """
    Load actual trip durations from the existing processed dataset.

    The existing NYC taxi dataset provides the actual
    trip_duration values that we use for monitoring.
    """

    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAINING_FILE}"
        )

    df = pd.read_parquet(TRAINING_FILE)

    return df[["trip_duration"]].copy()


def main() -> None:
    print("Starting prediction monitoring...")

    # ----------------------------------------------------------
    # Load prediction records
    # ----------------------------------------------------------

    predictions = load_prediction_log()

    print(
        f"Prediction records found: {len(predictions):,}"
    )

    # ----------------------------------------------------------
    # Keep predictions that have actual durations
    # ----------------------------------------------------------

    predictions_with_actuals = predictions[
        predictions["actual_duration_seconds"].notna()
    ].copy()

    # ----------------------------------------------------------
    # If actual durations are not yet available, demonstrate
    # monitoring using the existing dataset.
    # ----------------------------------------------------------

    if predictions_with_actuals.empty:

        print(
            "No actual durations found in prediction log."
        )

        print(
            "Using existing NYC taxi trip durations "
            "to demonstrate monitoring."
        )

        actuals = load_actual_durations()

        number_of_records = min(
            len(predictions),
            len(actuals),
        )

        if number_of_records == 0:
            print(
                "No records available for monitoring."
            )
            return

        predictions_with_actuals = predictions.iloc[
            :number_of_records
        ].copy()

        predictions_with_actuals[
            "actual_duration_seconds"
        ] = actuals.iloc[
            :number_of_records
        ]["trip_duration"].to_numpy()

    # ----------------------------------------------------------
    # Calculate prediction errors
    # ----------------------------------------------------------

    y_pred = predictions_with_actuals[
        "predicted_duration_seconds"
    ]

    y_actual = predictions_with_actuals[
        "actual_duration_seconds"
    ]

    mae = mean_absolute_error(
        y_actual,
        y_pred,
    )

    rmse = mean_squared_error(
        y_actual,
        y_pred,
    ) ** 0.5

    predictions_with_actuals[
        "absolute_error_seconds"
    ] = (
        y_actual - y_pred
    ).abs()

    predictions_with_actuals[
        "error_seconds"
    ] = (
        y_actual - y_pred
    )

    # ----------------------------------------------------------
    # Monitoring metrics
    # ----------------------------------------------------------

    metrics = {
        "monitoring_records": len(
            predictions_with_actuals
        ),
        "mae_seconds": round(
            float(mae),
            2,
        ),
        "rmse_seconds": round(
            float(rmse),
            2,
        ),
        "mean_absolute_error_seconds": round(
            float(
                predictions_with_actuals[
                    "absolute_error_seconds"
                ].mean()
            ),
            2,
        ),
    }

    # ----------------------------------------------------------
    # Save monitoring metrics
    # ----------------------------------------------------------

    METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # ----------------------------------------------------------
    # Display results
    # ----------------------------------------------------------

    print("\nMonitoring results:")
    print(
        f"Records monitored: {metrics['monitoring_records']:,}"
    )
    print(
        f"MAE:  {metrics['mae_seconds']:.2f} seconds"
    )
    print(
        f"RMSE: {metrics['rmse_seconds']:.2f} seconds"
    )

    print(
        f"\nMonitoring metrics saved to: {METRICS_FILE}"
    )

    print(
        "Prediction monitoring completed successfully."
    )


if __name__ == "__main__":
    main()