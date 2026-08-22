from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset_10000.parquet"
)

MONITORING_DIR = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
)

DRIFT_DATA_FILE = (
    MONITORING_DIR
    / "simulated_production_data.parquet"
)


def main() -> None:
    print("Starting production data drift simulation...")

    # ---------------------------------------------------------
    # Load reference/training data
    # ---------------------------------------------------------

    df = pd.read_parquet(TRAINING_FILE)

    print(
        f"Reference records: {len(df):,}"
    )

    # ---------------------------------------------------------
    # Create simulated production data
    # ---------------------------------------------------------

    production_df = df.copy()

    # ---------------------------------------------------------
    # Simulate a distribution shift in taxi trip distance.
    #
    # Production trips are assumed to become longer than the
    # historical training distribution.
    # ---------------------------------------------------------

    production_df["distance_km"] = (
        production_df["distance_km"] * 1.40
    )

    # Keep distances non-negative.
    production_df["distance_km"] = (
        production_df["distance_km"].clip(lower=0)
    )

    # ---------------------------------------------------------
    # Save simulated production data
    # ---------------------------------------------------------

    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    production_df.to_parquet(
        DRIFT_DATA_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Display comparison
    # ---------------------------------------------------------

    reference_mean = df["distance_km"].mean()
    production_mean = production_df["distance_km"].mean()

    reference_median = df["distance_km"].median()
    production_median = production_df["distance_km"].median()

    print("\nDistance distribution comparison:")

    print(
        f"Reference mean distance:   "
        f"{reference_mean:.2f} km"
    )

    print(
        f"Production mean distance:  "
        f"{production_mean:.2f} km"
    )

    print(
        f"Reference median distance: "
        f"{reference_median:.2f} km"
    )

    print(
        f"Production median distance:"
        f" {production_median:.2f} km"
    )

    print(
        f"\nSimulated production data saved to:"
        f" {DRIFT_DATA_FILE}"
    )

    print(
        "Production data drift simulation "
        "completed successfully."
    )


if __name__ == "__main__":
    main()