from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILE = PROJECT_ROOT / "data" / "raw" / "source" / "NYC.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "nyc_taxi_trip_duration_10000.csv"

# Dataset configuration
N_RECORDS = 10_000
DATETIME_COLUMN = "pickup_datetime"


def create_project_dataset():
    """Create the project dataset using the latest pickup records."""

    print(f"Reading source dataset: {SOURCE_FILE}")

    df = pd.read_csv(SOURCE_FILE)

    print(f"Source records: {len(df):,}")

    # Convert pickup timestamp to datetime
    df[DATETIME_COLUMN] = pd.to_datetime(
        df[DATETIME_COLUMN],
        errors="coerce"
    )

    # Check for invalid timestamps
    invalid_timestamps = df[DATETIME_COLUMN].isna().sum()

    if invalid_timestamps > 0:
        raise ValueError(
            f"Found {invalid_timestamps} invalid pickup timestamps."
        )

    # Sort newest pickup records first
    df = df.sort_values(
        by=DATETIME_COLUMN,
        ascending=False
    )

    # Select latest N records
    project_df = df.head(N_RECORDS).copy()

    # Sort chronologically for easier downstream processing
    project_df = project_df.sort_values(
        by=DATETIME_COLUMN,
        ascending=True
    )

    # Save selected dataset
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    project_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nProject dataset created successfully.")
    print(f"Output file : {OUTPUT_FILE}")
    print(f"Records     : {len(project_df):,}")
    print(
        f"Date range  : "
        f"{project_df[DATETIME_COLUMN].min()} "
        f"to "
        f"{project_df[DATETIME_COLUMN].max()}"
    )


if __name__ == "__main__":
    create_project_dataset()