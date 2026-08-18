from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_features_10000.parquet"
)

TARGET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nyc_taxi_features_10000.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset_10000.parquet"
)


def main() -> None:
    print("Preparing training dataset...")

    features_df = pd.read_parquet(FEATURE_FILE)
    target_df = pd.read_parquet(TARGET_FILE)

    print(f"Feature records: {len(features_df):,}")
    print(f"Target records: {len(target_df):,}")

    # Keep only the target column and entity ID.
    target_df = target_df[["id", "trip_duration"]]

    # Merge Feast features with the target.
    training_df = features_df.merge(
        target_df,
        on="id",
        how="inner",
        validate="one_to_one",
    )

    # Ensure the target exists and contains no missing values.
    if training_df["trip_duration"].isna().any():
        raise ValueError("Missing values found in target column.")

    print(f"Training records: {len(training_df):,}")
    print(f"Training columns: {len(training_df.columns)}")

    training_df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Training dataset saved to: {OUTPUT_FILE}")

    print("\nTraining dataset columns:")
    print(training_df.columns.tolist())

    print("\nTraining dataset preview:")
    print(training_df.head())


if __name__ == "__main__":
    main()