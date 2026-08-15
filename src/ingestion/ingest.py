from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILE = PROJECT_ROOT / "data" / "raw" / "source" / "NYC.csv"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nyc_taxi_trip_duration_10000.csv"
)

N_RECORDS = 10_000

REQUIRED_COLUMNS = [
    "id",
    "vendor_id",
    "pickup_datetime",
    "dropoff_datetime",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "store_and_fwd_flag",
    "trip_duration",
]


def load_source_data(source_file: Path) -> pd.DataFrame:
    """Load the source NYC taxi dataset."""

    if not source_file.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {source_file}"
        )

    print(f"Loading source dataset: {source_file}")

    return pd.read_csv(source_file)


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that all required columns are present."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def select_latest_records(
    df: pd.DataFrame,
    n_records: int = N_RECORDS,
) -> pd.DataFrame:
    """Select the latest records based on pickup datetime."""

    df = df.copy()

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"],
        errors="coerce",
    )

    invalid_timestamps = df["pickup_datetime"].isna().sum()

    if invalid_timestamps > 0:
        raise ValueError(
            f"Found {invalid_timestamps} invalid pickup timestamps."
        )

    df = df.sort_values(
        by="pickup_datetime",
        ascending=False,
    )

    selected_df = df.head(n_records).copy()

    # Store selected records chronologically.
    selected_df = selected_df.sort_values(
        by="pickup_datetime",
        ascending=True,
    )

    return selected_df


def save_dataset(
    df: pd.DataFrame,
    output_file: Path,
) -> None:
    """Save the project dataset."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print(f"Saved dataset to: {output_file}")


def run_ingestion() -> None:
    """Run the complete data ingestion pipeline."""

    print("Starting data ingestion pipeline...")

    df = load_source_data(SOURCE_FILE)

    print(f"Source records: {len(df):,}")

    validate_schema(df)

    print("Schema validation: PASSED")

    project_df = select_latest_records(df)

    print(f"Selected records: {len(project_df):,}")

    save_dataset(project_df, OUTPUT_FILE)

    print("Data ingestion pipeline completed successfully.")


if __name__ == "__main__":
    run_ingestion()