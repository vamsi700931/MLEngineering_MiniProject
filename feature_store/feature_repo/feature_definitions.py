from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nyc_taxi_features_10000.parquet"
)


taxi_trip = Entity(
    name="taxi_trip",
    join_keys=["id"],
    value_type=ValueType.STRING,
    description="NYC taxi trip identifier",
)


taxi_features_source = FileSource(
    name="nyc_taxi_features_source",
    path=str(FEATURE_DATASET),
    timestamp_field="pickup_datetime",
)


taxi_trip_features = FeatureView(
    name="taxi_trip_features",
    entities=[taxi_trip],
    ttl=None,
    schema=[
        Field(name="vendor_id", dtype=Int64),
        Field(name="passenger_count", dtype=Int64),
        Field(name="pickup_longitude", dtype=Float32),
        Field(name="pickup_latitude", dtype=Float32),
        Field(name="dropoff_longitude", dtype=Float32),
        Field(name="dropoff_latitude", dtype=Float32),
        Field(name="store_and_fwd_flag", dtype=String),
        Field(name="pickup_hour", dtype=Int64),
        Field(name="pickup_day_of_week", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
        Field(name="distance_km", dtype=Float32),
    ],
    source=taxi_features_source,
)