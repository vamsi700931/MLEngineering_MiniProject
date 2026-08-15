from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nyc_taxi_trip_duration_10000.csv"
)

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


def load_dataset() -> pd.DataFrame:
    """Load the project dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    return pd.read_csv(DATA_FILE)


def validate_schema(df: pd.DataFrame) -> None:
    """Validate the dataset schema."""

    actual_columns = set(df.columns)
    required_columns = set(REQUIRED_COLUMNS)

    missing_columns = required_columns - actual_columns
    unexpected_columns = actual_columns - required_columns

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if unexpected_columns:
        print(
            f"Warning: unexpected columns found: "
            f"{sorted(unexpected_columns)}"
        )

    print("Schema validation: PASSED")


def validate_missing_values(df: pd.DataFrame) -> None:
    """Report missing values in each column."""

    missing_counts = df.isna().sum()
    missing_percentages = (missing_counts / len(df)) * 100

    missing_report = pd.DataFrame(
        {
            "missing_count": missing_counts,
            "missing_percentage": missing_percentages.round(2),
        }
    )

    print("\nMissing-value report:")
    print(missing_report)

    total_missing = int(missing_counts.sum())

    if total_missing == 0:
        print("\nMissing-value validation: PASSED")
    else:
        print(
            f"\nMissing-value validation: "
            f"{total_missing} missing values found"
        )


def validate_timestamps(df: pd.DataFrame) -> None:
    """Validate pickup and drop-off timestamps."""

    pickup = pd.to_datetime(
        df["pickup_datetime"],
        errors="coerce",
    )

    dropoff = pd.to_datetime(
        df["dropoff_datetime"],
        errors="coerce",
    )

    invalid_pickup = pickup.isna().sum()
    invalid_dropoff = dropoff.isna().sum()

    invalid_order = (dropoff < pickup).sum()

    print("\nTimestamp validation:")

    print(f"Invalid pickup timestamps: {invalid_pickup}")
    print(f"Invalid drop-off timestamps: {invalid_dropoff}")
    print(f"Drop-off before pickup: {invalid_order}")

    if (
        invalid_pickup == 0
        and invalid_dropoff == 0
        and invalid_order == 0
    ):
        print("Timestamp validation: PASSED")
    else:
        raise ValueError(
            "Timestamp validation failed."
        )


def validate_coordinates(df: pd.DataFrame) -> None:
    """Validate latitude and longitude ranges."""

    latitude_columns = [
        "pickup_latitude",
        "dropoff_latitude",
    ]

    longitude_columns = [
        "pickup_longitude",
        "dropoff_longitude",
    ]

    invalid_latitude = 0
    invalid_longitude = 0

    for column in latitude_columns:
        invalid_latitude += (
            ~df[column].between(-90, 90)
        ).sum()

    for column in longitude_columns:
        invalid_longitude += (
            ~df[column].between(-180, 180)
        ).sum()

    print("\nCoordinate validation:")

    print(
        f"Invalid latitude values: {invalid_latitude}"
    )

    print(
        f"Invalid longitude values: {invalid_longitude}"
    )

    if invalid_latitude == 0 and invalid_longitude == 0:
        print("Coordinate validation: PASSED")
    else:
        raise ValueError(
            "Coordinate validation failed."
        )
def report_coordinate_distribution(df: pd.DataFrame) -> None:
    """Report geographic coordinate distribution."""

    coordinate_columns = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]

    print("\nCoordinate distribution:")

    distribution = df[coordinate_columns].describe(
        percentiles=[0.01, 0.50, 0.99]
    ).T

    print(
        distribution[
            [
                "min",
                "1%",
                "50%",
                "99%",
                "max",
            ]
        ]
    )
def report_trip_duration_distribution(
    df: pd.DataFrame,
) -> None:
    """Report trip-duration distribution and IQR outliers."""

    duration = df["trip_duration"]

    print("\nTrip-duration distribution:")

    print(f"Minimum: {duration.min():.2f} seconds")
    print(f"1st percentile: {duration.quantile(0.01):.2f} seconds")
    print(f"25th percentile: {duration.quantile(0.25):.2f} seconds")
    print(f"Median: {duration.median():.2f} seconds")
    print(f"75th percentile: {duration.quantile(0.75):.2f} seconds")
    print(f"99th percentile: {duration.quantile(0.99):.2f} seconds")
    print(f"Maximum: {duration.max():.2f} seconds")
    print(f"Mean: {duration.mean():.2f} seconds")
    print(f"Standard deviation: {duration.std():.2f} seconds")

    q1 = duration.quantile(0.25)
    q3 = duration.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = (
        (duration < lower_bound)
        | (duration > upper_bound)
    )

    print("\nIQR outlier analysis:")
    print(f"Q1: {q1:.2f} seconds")
    print(f"Q3: {q3:.2f} seconds")
    print(f"IQR: {iqr:.2f} seconds")
    print(f"Lower bound: {lower_bound:.2f} seconds")
    print(f"Upper bound: {upper_bound:.2f} seconds")
    print(f"Outlier records: {outliers.sum():,}")
    print(
        f"Outlier percentage: "
        f"{(outliers.mean() * 100):.2f}%"
    )

def report_extreme_trip_durations(
    df: pd.DataFrame,
    top_n: int = 10,
) -> None:
    """Report the longest trips for manual inspection."""

    columns = [
        "id",
        "pickup_datetime",
        "dropoff_datetime",
        "passenger_count",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "trip_duration",
    ]

    longest_trips = (
        df[columns]
        .sort_values(
            by="trip_duration",
            ascending=False,
        )
        .head(top_n)
    )

    print(
        f"\nTop {top_n} longest trips:"
    )

    print(
        longest_trips.to_string(index=False)
    )

def validate_passenger_count(df: pd.DataFrame) -> None:
    """Validate passenger count."""

    invalid_passengers = (
        df["passenger_count"] <= 0
    ).sum()

    print("\nPassenger-count validation:")

    print(
        f"Invalid passenger counts: {invalid_passengers}"
    )

    if invalid_passengers == 0:
        print("Passenger-count validation: PASSED")
    else:
        raise ValueError(
            "Passenger-count validation failed."
        )


def validate_trip_duration(df: pd.DataFrame) -> None:
    """Validate trip duration target."""

    invalid_duration = (
        df["trip_duration"] <= 0
    ).sum()

    print("\nTrip-duration validation:")

    print(
        f"Invalid trip durations: {invalid_duration}"
    )

    if invalid_duration == 0:
        print("Trip-duration validation: PASSED")
    else:
        raise ValueError(
            "Trip-duration validation failed."
        )


if __name__ == "__main__":
    df = load_dataset()

    print(f"Dataset records: {len(df):,}")
    print(f"Dataset columns: {len(df.columns)}")

    validate_schema(df)
    validate_missing_values(df)
    validate_timestamps(df)
    validate_coordinates(df)
    validate_passenger_count(df)
    validate_trip_duration(df)
    report_coordinate_distribution(df)
    report_trip_duration_distribution(df)
    report_extreme_trip_durations(df)