from pathlib import Path

import pandas as pd
from feast import FeatureStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_REPO = (
    PROJECT_ROOT
    / "feature_store"
    / "feature_repo"
)

SOURCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nyc_taxi_features_10000.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_features_10000.parquet"
)


def load_entity_dataframe() -> pd.DataFrame:
    """Load entity keys and event timestamps."""

    df = pd.read_parquet(SOURCE_FILE)

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"],
        errors="raise",
    )

    return df[
        [
            "id",
            "pickup_datetime",
        ]
    ].rename(
        columns={
            "pickup_datetime": "event_timestamp",
        }
    )


def get_training_features() -> pd.DataFrame:
    """Retrieve historical features from Feast."""

    entity_df = load_entity_dataframe()

    store = FeatureStore(
        repo_path=str(FEATURE_REPO)
    )

    feature_refs = [
        "taxi_trip_features:vendor_id",
        "taxi_trip_features:passenger_count",
        "taxi_trip_features:pickup_longitude",
        "taxi_trip_features:pickup_latitude",
        "taxi_trip_features:dropoff_longitude",
        "taxi_trip_features:dropoff_latitude",
        "taxi_trip_features:store_and_fwd_flag",
        "taxi_trip_features:pickup_hour",
        "taxi_trip_features:pickup_day_of_week",
        "taxi_trip_features:is_weekend",
        "taxi_trip_features:distance_km",
    ]

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    ).to_df()

    return training_df


def main() -> None:
    """Generate training features from Feast."""

    print("Retrieving training features from Feast...")

    training_df = get_training_features()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Training records: {len(training_df):,}"
    )

    print(
        f"Training columns: {len(training_df.columns)}"
    )

    print(
        f"Training dataset saved to: {OUTPUT_FILE}"
    )

    print("\nTraining dataset preview:")
    print(
        training_df.head().to_string(index=False)
    )


if __name__ == "__main__":
    main()