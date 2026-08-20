from pathlib import Path
from datetime import datetime, timezone
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = PROJECT_ROOT / "data" / "monitoring"

PREDICTION_LOG_FILE = LOG_DIR / "prediction_log.jsonl"


def log_prediction(
    prediction_id: str,
    predicted_duration_seconds: float,
    actual_duration_seconds: float | None = None,
) -> None:
    """
    Log a model prediction.

    Actual duration may be unavailable at prediction time.
    It can be added later when the real trip duration becomes known.
    """

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "prediction_id": prediction_id,
        "prediction_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "predicted_duration_seconds": round(
            float(predicted_duration_seconds),
            2,
        ),
        "actual_duration_seconds": (
            round(float(actual_duration_seconds), 2)
            if actual_duration_seconds is not None
            else None
        ),
    }

    with open(
        PREDICTION_LOG_FILE,
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(record) + "\n"
        )


def update_actual_duration(
    prediction_id: str,
    actual_duration_seconds: float,
) -> bool:
    """
    Update the actual trip duration for an existing prediction.

    Returns True when the prediction is found and updated.
    """

    if not PREDICTION_LOG_FILE.exists():
        return False

    records = []

    found = False

    with open(
        PREDICTION_LOG_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if not line.strip():
                continue

            record = json.loads(line)

            if record["prediction_id"] == prediction_id:
                record["actual_duration_seconds"] = round(
                    float(actual_duration_seconds),
                    2,
                )
                found = True

            records.append(record)

    if not found:
        return False

    with open(
        PREDICTION_LOG_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:
            file.write(
                json.dumps(record) + "\n"
            )

    return True


def get_prediction_log_path() -> Path:
    """Return the prediction log file path."""

    return PREDICTION_LOG_FILE


if __name__ == "__main__":

    print(
        "Prediction logger module loaded successfully."
    )

    print(
        f"Prediction log: {PREDICTION_LOG_FILE}"
    )