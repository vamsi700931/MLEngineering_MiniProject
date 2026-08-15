from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nyc_taxi_trip_duration_10000.csv"
)

CSV_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nyc_taxi_features_10000.csv"
)

PARQUET_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nyc_taxi_features_10000.parquet"
)


def load_dataset() -> pd.DataFrame:
    """Load the ingested project dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    return pd.read_csv(INPUT_FILE)


def calculate_haversine_distance(
    pickup_latitude: pd.Series,
    pickup_longitude: pd.Series,
    dropoff_latitude: pd.Series,
    dropoff_longitude: pd.Series,
) -> pd.Series:
    """
    Calculate great-circle distance between pickup and drop-off
    coordinates using the Haversine formula.

    Returns distance in kilometers.
    """

    earth_radius_km = 6371.0

    latitude_1 = np.radians(pickup_latitude)
    latitude_2 = np.radians(dropoff_latitude)

    delta_latitude = np.radians(
        dropoff_latitude - pickup_latitude
    )

    delta_longitude = np.radians(
        dropoff_longitude - pickup_longitude
    )

    a = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(latitude_1)
        * np.cos(latitude_2)
        * np.sin(delta_longitude / 2) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    return earth_radius_km * c


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build features required for ETA prediction."""

    df = df.copy()

    # ---------------------------------------------------------
    # Timestamp conversion
    # ---------------------------------------------------------
    # Convert pickup_datetime to a real pandas datetime.
    # This is required by Feast as the event timestamp.
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"],
        errors="raise",
    )

    # Convert drop-off timestamp as well for validation/
    # consistency, although it is not retained as a model feature.
    df["dropoff_datetime"] = pd.to_datetime(
        df["dropoff_datetime"],
        errors="raise",
    )

    # ---------------------------------------------------------
    # Time-based features
    # ---------------------------------------------------------
    df["pickup_hour"] = (
        df["pickup_datetime"].dt.hour
    )

    df["pickup_day_of_week"] = (
        df["pickup_datetime"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["pickup_day_of_week"] >= 5
    ).astype(int)

    # ---------------------------------------------------------
    # Distance feature
    # ---------------------------------------------------------
    df["distance_km"] = calculate_haversine_distance(
        pickup_latitude=df["pickup_latitude"],
        pickup_longitude=df["pickup_longitude"],
        dropoff_latitude=df["dropoff_latitude"],
        dropoff_longitude=df["dropoff_longitude"],
    )

    # ---------------------------------------------------------
    # Select final feature dataset
    # ---------------------------------------------------------
    # pickup_datetime is intentionally retained because Feast
    # uses it as the event timestamp.
    #
    # trip_duration is retained as the prediction target.
    #
    # dropoff_datetime is not retained because it would create
    # target leakage for ETA prediction.
    selected_columns = [
        "id",
        "pickup_datetime",
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
        "trip_duration",
    ]

    return df[selected_columns]


def save_features(
    df: pd.DataFrame,
) -> None:
    """Save engineered features as CSV and Parquet."""

    CSV_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save CSV
    df.to_csv(
        CSV_OUTPUT_FILE,
        index=False,
    )

    # Save Parquet
    df.to_parquet(
        PARQUET_OUTPUT_FILE,
        index=False,
    )

    print(
        f"CSV feature dataset saved to: "
        f"{CSV_OUTPUT_FILE}"
    )

    print(
        f"Parquet feature dataset saved to: "
        f"{PARQUET_OUTPUT_FILE}"
    )


def main() -> None:
    """Run the feature building pipeline."""

    print("Starting feature building pipeline...")

    df = load_dataset()

    print(
        f"Input records: {len(df):,}"
    )

    feature_df = build_features(df)

    print(
        f"Output records: {len(feature_df):,}"
    )

    print(
        f"Output features: {len(feature_df.columns):,}"
    )

    save_features(feature_df)

    print(
        "Feature building pipeline completed successfully."
    )


if __name__ == "__main__":
    main()