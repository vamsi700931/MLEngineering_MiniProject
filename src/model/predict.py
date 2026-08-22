from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "nyc_taxi_eta_gradient_boosting_tuned.joblib"
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


def load_model():
    """Load the trained model artifact."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    artifact = joblib.load(MODEL_FILE)

    return artifact


def predict_eta(
    vendor_id,
    passenger_count,
    pickup_longitude,
    pickup_latitude,
    dropoff_longitude,
    dropoff_latitude,
    store_and_fwd_flag,
    pickup_hour,
    pickup_day_of_week,
    is_weekend,
    distance_km,
):
    """
    Predict taxi trip duration in seconds.
    """

    artifact = load_model()

    preprocessor = artifact["preprocessor"]
    model = artifact["model"]

    # Create input record using exactly the same
    # feature names used during model training.
    input_data = pd.DataFrame(
        [
            {
                "vendor_id": vendor_id,
                "passenger_count": passenger_count,
                "pickup_longitude": pickup_longitude,
                "pickup_latitude": pickup_latitude,
                "dropoff_longitude": dropoff_longitude,
                "dropoff_latitude": dropoff_latitude,
                "store_and_fwd_flag": store_and_fwd_flag,
                "pickup_hour": pickup_hour,
                "pickup_day_of_week": pickup_day_of_week,
                "is_weekend": is_weekend,
                "distance_km": distance_km,
            }
        ]
    )

    # Ensure the feature order is exactly the same
    # as the order used during training.
    input_data = input_data[FEATURE_COLUMNS]

    # Apply the same preprocessing used during training.
    input_processed = preprocessor.transform(input_data)

    # Generate prediction.
    prediction = model.predict(input_processed)

    return float(prediction[0])


def main() -> None:
    print("Starting local ETA prediction...")

    predicted_duration = predict_eta(
        vendor_id=2,
        passenger_count=1,
        pickup_longitude=-73.986771,
        pickup_latitude=40.736771,
        dropoff_longitude=-73.983032,
        dropoff_latitude=40.722736,
        store_and_fwd_flag="N",
        pickup_hour=18,
        pickup_day_of_week=2,
        is_weekend=0,
        distance_km=1.59,
    )

    predicted_minutes = predicted_duration / 60

    print("\nPrediction result:")
    print(
        f"Predicted trip duration: "
        f"{predicted_duration:.2f} seconds"
    )
    print(
        f"Predicted ETA: "
        f"{predicted_minutes:.2f} minutes"
    )

    print("\nLocal ETA prediction completed successfully.")


if __name__ == "__main__":
    main()