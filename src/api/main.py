from pathlib import Path
import sys
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ------------------------------------------------------------------
# Model and monitoring imports
# ------------------------------------------------------------------

from model.predict import predict_eta
from monitoring.prediction_logger import log_prediction


# ------------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------------

app = FastAPI(
    title="NYC Taxi ETA Prediction API",
    description="REST API for predicting NYC taxi trip duration.",
    version="1.0.0",
)


# ------------------------------------------------------------------
# Request schema
# ------------------------------------------------------------------

class PredictionRequest(BaseModel):
    vendor_id: int = Field(..., ge=1)

    passenger_count: int = Field(
        ...,
        ge=1,
        le=10,
    )

    pickup_longitude: float = Field(
        ...,
        ge=-180,
        le=180,
    )

    pickup_latitude: float = Field(
        ...,
        ge=-90,
        le=90,
    )

    dropoff_longitude: float = Field(
        ...,
        ge=-180,
        le=180,
    )

    dropoff_latitude: float = Field(
        ...,
        ge=-90,
        le=90,
    )

    store_and_fwd_flag: str = Field(
        ...,
        pattern="^[YN]$",
    )

    pickup_hour: int = Field(
        ...,
        ge=0,
        le=23,
    )

    pickup_day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
    )

    is_weekend: int = Field(
        ...,
        ge=0,
        le=1,
    )

    distance_km: float = Field(
        ...,
        ge=0,
    )


# ------------------------------------------------------------------
# Response schema
# ------------------------------------------------------------------

class PredictionResponse(BaseModel):
    predicted_trip_duration_seconds: float
    predicted_eta_minutes: float


# ------------------------------------------------------------------
# Health endpoint
# ------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "NYC Taxi ETA Prediction API",
    }


# ------------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    try:
        # ----------------------------------------------------------
        # Generate prediction
        # ----------------------------------------------------------

        predicted_duration = predict_eta(
            vendor_id=request.vendor_id,
            passenger_count=request.passenger_count,
            pickup_longitude=request.pickup_longitude,
            pickup_latitude=request.pickup_latitude,
            dropoff_longitude=request.dropoff_longitude,
            dropoff_latitude=request.dropoff_latitude,
            store_and_fwd_flag=request.store_and_fwd_flag,
            pickup_hour=request.pickup_hour,
            pickup_day_of_week=request.pickup_day_of_week,
            is_weekend=request.is_weekend,
            distance_km=request.distance_km,
        )

        predicted_minutes = predicted_duration / 60

        # ----------------------------------------------------------
        # Create prediction ID
        # ----------------------------------------------------------

        prediction_id = f"pred_{uuid.uuid4().hex[:12]}"

        # ----------------------------------------------------------
        # Log prediction
        #
        # Actual duration is not available at prediction time.
        # It will be added later when the real trip duration
        # becomes available.
        # ----------------------------------------------------------

        log_prediction(
            prediction_id=prediction_id,
            predicted_duration_seconds=predicted_duration,
            actual_duration_seconds=None,
        )

        # ----------------------------------------------------------
        # Return API response
        # ----------------------------------------------------------

        return PredictionResponse(
            predicted_trip_duration_seconds=round(
                predicted_duration,
                2,
            ),
            predicted_eta_minutes=round(
                predicted_minutes,
                2,
            ),
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )