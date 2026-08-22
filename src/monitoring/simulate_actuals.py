from pathlib import Path
import json
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from monitoring.prediction_logger import update_actual_duration


PREDICTION_LOG_FILE = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "prediction_log.jsonl"
)


def load_predictions():
    """Load predictions that do not yet have actual durations."""

    if not PREDICTION_LOG_FILE.exists():
        raise FileNotFoundError(
            f"Prediction log not found: {PREDICTION_LOG_FILE}"
        )

    predictions = []

    with open(
        PREDICTION_LOG_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if record["actual_duration_seconds"] is None:
                predictions.append(record)

    return predictions


def main() -> None:
    print("Starting actual-duration simulation...")

    predictions = load_predictions()

    print(
        f"Predictions without actual duration: "
        f"{len(predictions):,}"
    )

    if not predictions:
        print(
            "No predictions require actual-duration simulation."
        )
        return

    random.seed(42)

    updated_count = 0

    for prediction in predictions:

        predicted_duration = float(
            prediction["predicted_duration_seconds"]
        )

        # ------------------------------------------------------
        # Simulate a realistic actual duration.
        #
        # The actual duration is allowed to differ from the
        # prediction by approximately +/-15%.
        # ------------------------------------------------------

        variation = random.uniform(
            -0.15,
            0.15,
        )

        actual_duration = (
            predicted_duration
            * (1 + variation)
        )

        updated = update_actual_duration(
            prediction_id=prediction["prediction_id"],
            actual_duration_seconds=actual_duration,
        )

        if updated:
            updated_count += 1

            print(
                f"Updated {prediction['prediction_id']}: "
                f"predicted={predicted_duration:.2f}s, "
                f"actual={actual_duration:.2f}s"
            )

    print(
        f"\nActual durations updated: "
        f"{updated_count:,}"
    )

    print(
        "Actual-duration simulation completed successfully."
    )


if __name__ == "__main__":
    main()