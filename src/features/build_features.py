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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nyc_taxi_features_10000.csv"
)


def load_dataset() -> pd.DataFrame:
    """Load the validated project dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    return pd.read_csv(INPUT_FILE)


def calculate_haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2,
) -> np.ndarray:
    """Calculate straight-line distance between two coordinates."""

    earth_radius_km = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a),
    )

    return earth_radius_km * c


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the required ETA prediction features."""

    df = df.copy()

    pickup_datetime = pd.to_datetime(
        df["pickup_datetime"],
        errors="raise",
    )

    # Required time-based features
    df["pickup_hour"] = pickup_datetime.dt.hour

    df["pickup_day_of_week"] = (
        pickup_datetime.dt.dayofweek
    )

    df["is_weekend"] = (
            pickup_datetime.dt.dayofweek >= 5
    ).astype(int)

    # Required distance feature
    df["distance_km"] = calculate_haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    return df


def select_model_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Select model features and target."""

    columns = [
        "id",
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

    return df[columns].copy()


def save_features(df: pd.DataFrame) -> None:
    """Save the engineered feature dataset."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Feature dataset saved to: {OUTPUT_FILE}"
    )


def run_feature_pipeline() -> None:
    """Run the feature building pipeline."""

    print("Starting feature building pipeline...")

    df = load_dataset()

    print(
        f"Input records: {len(df):,}"
    )

    feature_df = build_features(df)

    feature_df = select_model_columns(
        feature_df
    )

    print(
        f"Output records: {len(feature_df):,}"
    )

    print(
        f"Output features: {len(feature_df.columns)}"
    )

    save_features(feature_df)

    print(
        "Feature building pipeline completed successfully."
    )


if __name__ == "__main__":
    run_feature_pipeline()