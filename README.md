# MLEngineering Mini Project — NYC Taxi ETA Prediction

## 1. Project Overview

This project implements an end-to-end **Machine Learning Engineering / MLOps pipeline** for predicting NYC taxi trip duration and exposing the selected model through a REST API.

The project follows **Flavor A — Delivery / Ride ETA Prediction** from the Machine Learning Engineering mini-project brief.

The implemented lifecycle covers:

- Data ingestion
- Data validation
- Feature engineering
- Dataset versioning with DVC
- Model experimentation
- MLflow experiment tracking
- Model comparison and selection
- Model packaging
- FastAPI model serving
- Prediction logging
- Prediction-performance monitoring
- Data-drift simulation and detection
- Drift-based retraining decision
- Controlled candidate-model retraining
- Candidate-vs-current model evaluation
- API latency measurement

---

## 2. Problem Statement

A ride-hailing platform needs to estimate taxi trip duration based on trip characteristics such as:

- Vendor
- Passenger count
- Pickup/drop-off coordinates
- Pickup time
- Day of week
- Weekend indicator
- Trip distance
- Store-and-forward flag

The selected model predicts **trip duration in seconds**, which is also exposed as an ETA in minutes through the API.

The project uses a 10,000-record subset of the NYC Taxi Trip Duration dataset.

---

# 3. Project Architecture

The project follows the following end-to-end MLOps architecture:

```text
                         NYC TAXI RAW DATA
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Data Ingestion        │
                    │ & Validation          │
                    │                       │
                    │ • Schema              │
                    │ • Missing values      │
                    │ • GPS validation      │
                    │ • Timestamp validation│
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Feature Engineering   │
                    │                       │
                    │ • Distance            │
                    │ • Time features       │
                    │ • Weekend indicator   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ DVC Versioned Data    │
                    │                       │
                    │ Dataset & features    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   MODEL TRAINING &    │
                    │   EXPERIMENTATION     │
                    └───────────┬───────────┘
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
              Random Forest       Gradient Boosting
                     │                     │
                     └──────────┬──────────┘
                                ▼
                         ┌─────────────┐
                         │   MLflow    │
                         │ Experiments │
                         └──────┬──────┘
                                │
                                ▼
                       Model Comparison
                                │
                                ▼
                     ┌────────────────────┐
                     │ Selected Best Model│
                     └─────────┬──────────┘
                               │
                               ▼
                         ┌───────────┐
                         │  FastAPI  │
                         │ /predict  │
                         │ /health   │
                         └─────┬─────┘
                               │
                               ▼
                          Prediction
                               │
                               ▼
                    Prediction Logging
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       Performance Monitoring        Data Drift Detection
                 │                           │
                 │                    PSI / Feature Drift
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                     Retraining Decision
                               │
                    ┌──────────┴──────────┐
                    │                     │
              No Retraining          Retraining
                    │                     │
                    │                     ▼
                    │             Candidate Training
                    │                     │
                    │                     ▼
                    │             Candidate Evaluation
                    │                     │
                    │                     ▼
                    │              Model Comparison
                    │                     │
                    │                     ▼
                    │               Model Promotion
                    │                     │
                    └─────────────┬───────┘
                                  ▼
                           Production Model
                                  │
                                  └──────────► FastAPI
```

### Architecture explanation

The system starts with the raw NYC Taxi dataset. Data ingestion and validation establish data quality before feature engineering creates the model-ready dataset.

DVC is used to version the important data artifacts. Multiple models are then trained and evaluated, with MLflow used for experiment tracking. The selected model is packaged and exposed through FastAPI.

Once the model is serving predictions, prediction results are logged for monitoring. Production-like data drift is simulated and measured using Population Stability Index (PSI). A decision layer determines whether drift should only be monitored or whether retraining should be considered.

When retraining is triggered, a candidate model is trained and evaluated against the current model. The candidate is promoted only when it performs better according to the configured evaluation criterion.

---

# 4. Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Experiment tracking | MLflow |
| Data/model versioning | DVC + Git |
| Model serialization | Joblib |
| API | FastAPI |
| API server | Uvicorn |
| Data format | CSV, Parquet, JSON/JSONL |
| Monitoring | Custom Python monitoring scripts |
| Drift detection | PSI |
| Environment | Python virtual environment |

---

# 5. Project Structure

The current project structure is:

```text
MLEngineering_MiniProject/
│
├── data/
│   ├── interim/
│   │
│   ├── monitoring/
│   │   ├── api_latency_metrics.json
│   │   ├── drift_decision.json
│   │   ├── drift_metrics.json
│   │   ├── monitoring_metrics.json
│   │   ├── prediction_log.jsonl
│   │   ├── retraining_result.json
│   │   └── simulated_production_data.parquet
│   │
│   ├── processed/
│   │   ├── nyc_taxi_features_10000.csv
│   │   ├── nyc_taxi_features_10000.csv.dvc
│   │   ├── nyc_taxi_features_10000.parquet
│   │   ├── training_dataset_10000.parquet
│   │   ├── training_features_10000.parquet
│   │   └── training_features_10000.parquet.dvc
│   │
│   └── raw/
│       ├── nyc_taxi_trip_duration_10000.csv
│       ├── nyc_taxi_trip_duration_10000.csv.dvc
│       └── source/
│           └── NYC.csv
│
├── models/
│   ├── evaluation_metrics.json
│   ├── gradient_boosting_metrics.json
│   ├── model_comparison.json
│   ├── nyc_taxi_eta_gradient_boosting_tuned.joblib
│   ├── nyc_taxi_eta_gradient_boosting_tuned.joblib.dvc
│   ├── nyc_taxi_eta_model.joblib
│   ├── nyc_taxi_eta_model.joblib.dvc
│   └── nyc_taxi_eta_retrained_candidate.joblib
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── test_api_latency.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │   └── get_training_data.py
│   │
│   ├── ingestion/
│   │   ├── create_project_dataset.py
│   │   └── ingest.py
│   │
│   ├── model/
│   │   ├── compare_models.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   ├── prepare_training_data.py
│   │   ├── train.py
│   │   └── train_gradient_boosting.py
│   │
│   ├── monitoring/
│   │   ├── detect_drift.py
│   │   ├── drift_decision.py
│   │   ├── evaluate_predictions.py
│   │   ├── prediction_logger.py
│   │   ├── retrain_model.py
│   │   ├── simulate_actuals.py
│   │   └── simulate_drift.py
│   │
│   └── validation/
│       └── validate.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 6. M2 — Data Engineering & Versioning

## 6.1 Data ingestion

The raw NYC Taxi data is ingested and a project-specific 10,000-record dataset is created.

Relevant scripts:

```text
src/ingestion/ingest.py
src/ingestion/create_project_dataset.py
```

The project keeps raw, processed, and monitoring data in separate directories.

---

## 6.2 Data validation

Validation is implemented in:

```text
src/validation/validate.py
```

The validation pipeline checks:

- Required schema
- Missing values
- Pickup timestamps
- Drop-off timestamps
- Drop-off occurring before pickup
- Missing GPS coordinates
- Latitude range
- Longitude range
- Passenger count
- Trip duration

The final validation run produced:

```text
Dataset records: 10,000
Dataset columns: 11

Schema validation: PASSED
Missing-value validation: PASSED

Invalid pickup timestamps: 0
Invalid drop-off timestamps: 0
Drop-off before pickup: 0

Missing pickup latitude:       0
Missing pickup longitude:      0
Missing drop-off latitude:     0
Missing drop-off longitude:    0
Invalid latitude values:       0
Invalid longitude values:      0

GPS validation: PASSED
Passenger-count validation: PASSED
Trip-duration validation: PASSED
```

### Trip-duration distribution

```text
Minimum:       2 seconds
Median:        692 seconds
Mean:          1005.21 seconds
99th percentile: 3755.22 seconds
Maximum:       86387 seconds
```

An IQR-based analysis identified 595 records (5.95%) as statistical outliers. These were retained for the project rather than automatically deleted, since extreme trip durations can represent genuine long-duration trips and the model should be evaluated against the available target distribution.

---

# 7. Feature Engineering

Feature engineering is implemented under:

```text
src/features/
```

The pipeline creates model-ready features including:

- Trip distance
- Pickup hour
- Day of week
- Weekend indicator
- Encoded categorical values required by the model

The resulting training data is stored in Parquet format.

---

# 8. DVC — Data and Model Versioning

DVC is used alongside Git to track important data and model artifacts.

Examples include:

```text
data/raw/nyc_taxi_trip_duration_10000.csv.dvc
data/processed/nyc_taxi_features_10000.csv.dvc
data/processed/training_features_10000.parquet.dvc
models/nyc_taxi_eta_model.joblib.dvc
models/nyc_taxi_eta_gradient_boosting_tuned.joblib.dvc
```

Git tracks pipeline code and DVC metadata, while DVC tracks the corresponding large data/model artifacts.

---

# 9. M3 — Model Experimentation

Two model approaches were implemented and compared:

1. Random Forest
2. Gradient Boosting

Relevant files include:

```text
src/model/train.py
src/model/evaluate.py
src/model/train_gradient_boosting.py
src/model/compare_models.py
```

The project also includes a tuned Gradient Boosting model.

Experiment tracking and model comparison were implemented as part of the M3 workflow.

---

# 10. Model Selection

The model comparison workflow produces:

```text
models/model_comparison.json
```

The selected model is packaged as a Joblib artifact.

The project also contains:

```text
models/nyc_taxi_eta_model.joblib
models/nyc_taxi_eta_gradient_boosting_tuned.joblib
```

The model-selection process is based on evaluation metrics rather than simply selecting a model by name.

---

# 11. M4 — Model Packaging & API Deployment

The production API is implemented in:

```text
src/api/main.py
```

The API uses FastAPI and exposes:

### Health endpoint

```text
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "service": "NYC Taxi ETA Prediction API"
}
```

### Prediction endpoint

```text
POST /predict
```

The API validates input using Pydantic and returns:

```json
{
  "predicted_trip_duration_seconds": 895.85,
  "predicted_eta_minutes": 14.93
}
```

The API performs validation for:

- Vendor ID
- Passenger count
- Latitude/longitude ranges
- Store-and-forward flag
- Pickup hour
- Day of week
- Weekend indicator
- Distance

---

# 12. Running the API

Activate the virtual environment and run:

```powershell
python -m uvicorn src.api.main:app --reload
```

The API becomes available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

# 13. Sample Prediction Payload

Example request body:

```json
{
  "vendor_id": 1,
  "passenger_count": 1,
  "pickup_longitude": -73.9857,
  "pickup_latitude": 40.7484,
  "dropoff_longitude": -73.9851,
  "dropoff_latitude": 40.7580,
  "store_and_fwd_flag": "N",
  "pickup_hour": 18,
  "pickup_day_of_week": 2,
  "is_weekend": 0,
  "distance_km": 1.5
}
```

The API returns predicted trip duration and ETA in minutes.

---

# 14. API Latency Measurement

Basic API latency measurement is implemented in:

```text
src/api/test_api_latency.py
```

The test executed 50 prediction requests.

Observed results:

```text
Total requests:       50
Successful requests:  50
Failed requests:      0

Average latency:      100.63 ms
Median latency:       31.96 ms
Minimum latency:      21.71 ms
Maximum latency:      3024.55 ms
```

Results are saved to:

```text
data/monitoring/api_latency_metrics.json
```

The maximum latency is substantially higher than the median because the test includes normal local-service variability such as application startup/reload or occasional request overhead. The result demonstrates basic latency awareness rather than a formal load/performance benchmark.

---

# 15. M5 — Prediction Logging

Prediction logging is implemented in:

```text
src/monitoring/prediction_logger.py
```

Predictions are stored in:

```text
data/monitoring/prediction_log.jsonl
```

Each record contains:

- Prediction ID
- Prediction timestamp
- Predicted duration
- Actual duration when available

Example:

```json
{
  "prediction_id": "pred_451eca1d71f5",
  "prediction_timestamp": "2026-08-20T18:08:45.159996+00:00",
  "predicted_duration_seconds": 895.85,
  "actual_duration_seconds": 933.32
}
```

---

# 16. Prediction Performance Monitoring

Prediction monitoring is implemented in:

```text
src/monitoring/evaluate_predictions.py
```

Actual durations are simulated for demonstration purposes using:

```text
src/monitoring/simulate_actuals.py
```

After actual durations were available, the monitoring results were:

```text
Records monitored: 4
MAE:  90.45 seconds
RMSE: 100.38 seconds
```

The metrics are stored in:

```text
data/monitoring/monitoring_metrics.json
```

This demonstrates the monitoring workflow:

```text
Prediction
    ↓
Prediction Log
    ↓
Actual Outcome
    ↓
MAE / RMSE
    ↓
Monitoring Metrics
```

---

# 17. Data Drift Simulation

Production-like drift is simulated in:

```text
src/monitoring/simulate_drift.py
```

The simulation creates a production dataset with a shifted distance distribution.

Observed distribution:

```text
Reference mean distance:   3.42 km
Production mean distance:  4.79 km

Reference median distance: 2.13 km
Production median distance: 2.99 km
```

The simulated production dataset is stored in:

```text
data/monitoring/simulated_production_data.parquet
```

---

# 18. Drift Detection

Drift detection is implemented in:

```text
src/monitoring/detect_drift.py
```

The project uses Population Stability Index (PSI).

Configured interpretation:

```text
PSI < 0.10       → No significant drift
0.10 – 0.25      → Moderate drift
PSI > 0.25       → Significant drift
```

Observed result:

```text
Feature: distance_km
PSI:     0.1622
Status:  moderate_drift
```

Therefore:

```text
Overall drift detected: True
```

The result is stored in:

```text
data/monitoring/drift_metrics.json
```

---

# 19. Drift Decision

The drift decision layer is implemented in:

```text
src/monitoring/drift_decision.py
```

For the simulated result:

```text
Status: MODERATE_DRIFT
Action: MONITOR_AND_EVALUATE
Retraining recommended: False
```

This demonstrates that the system does not automatically retrain the model for every detected distribution change.

The decision is stored in:

```text
data/monitoring/drift_decision.json
```

---

# 20. Controlled Retraining Workflow

Controlled retraining is implemented in:

```text
src/monitoring/retrain_model.py
```

The workflow supports two triggers:

1. Monitoring-based retraining recommendation
2. Explicit retraining trigger for demonstration/testing

The retraining process:

```text
Drift Decision
      │
      ▼
Retraining Trigger
      │
      ▼
Train Candidate Model
      │
      ▼
Evaluate Candidate
      │
      ▼
Compare with Current Model
      │
      ├── Candidate worse
      │       ↓
      │   Keep current model
      │
      └── Candidate better
              ↓
        Promote candidate
```

A controlled retraining demonstration produced:

```text
Candidate model evaluation:
MAE:  504.48 seconds
RMSE: 2589.45 seconds
R²:   -0.6384

Current model RMSE:   2590.00 seconds
Candidate model RMSE: 2589.45 seconds

Candidate model performed better.
Action: PROMOTE_RETRAINED_MODEL
```

The candidate artifact is stored as:

```text
models/nyc_taxi_eta_retrained_candidate.joblib
```

The retraining decision/result is stored in:

```text
data/monitoring/retraining_result.json
```

The retraining workflow intentionally compares the candidate with the current model before promotion rather than blindly replacing the production model.

---

# 21. Important Retraining Design Note

Retraining with the same historical training data does **not** represent a real production retraining cycle by itself.

The retraining workflow in this project is primarily a controlled demonstration of the engineering mechanism:

- detect a trigger
- train a candidate
- evaluate it
- compare it against the current model
- promote only when it performs better

In a real production system, retraining would normally incorporate newly collected production data, newly labeled outcomes, updated feature distributions, or a refreshed training window.

The project uses a controlled retraining demonstration because the available mini-project dataset is static and historical.

---

# 22. Monitoring Architecture

The production monitoring flow is:

```text
                    FastAPI Prediction
                           │
                           ▼
                  Prediction Log
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       Actual Outcomes            Production Data
              │                         │
              ▼                         ▼
      Performance Metrics        PSI Drift Detection
              │                         │
              └────────────┬────────────┘
                           ▼
                   Monitoring Decision
                           │
                           ▼
                    Retraining Logic
                           │
                           ▼
                    Candidate Model
                           │
                           ▼
                 Candidate Evaluation
                           │
                           ▼
                    Model Promotion
```

---

# 23. Reproducibility

The project uses:

- Git for source-code version control
- DVC for data/model artifact versioning
- MLflow for experiment tracking
- Joblib for model serialization
- Requirements file for Python dependencies
- Incremental Git commits reflecting project progress

The repository history contains separate commits for major project stages including:

- Model experiment comparison
- Local prediction pipeline
- FastAPI serving
- Prediction logging
- Prediction monitoring
- Drift detection
- Drift decision logic
- Controlled retraining
- API latency measurement

---

# 24. Installation

Create and activate a Python virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

# 25. End-to-End Execution

The project can be demonstrated in the following order.

### Step 1 — Validate data

```powershell
python src/validation/validate.py
```

### Step 2 — Build features

```powershell
python src/features/build_features.py
```

### Step 3 — Prepare training data

```powershell
python src/model/prepare_training_data.py
```

### Step 4 — Run model experiments

Run the Random Forest and Gradient Boosting training/evaluation workflows available under:

```text
src/model/
```

### Step 5 — Compare models

```powershell
python src/model/compare_models.py
```

### Step 6 — Start API

```powershell
python -m uvicorn src.api.main:app --reload
```

### Step 7 — Test API

Open:

```text
http://127.0.0.1:8000/docs
```

### Step 8 — Measure API latency

With the API running:

```powershell
python src/api/test_api_latency.py
```

### Step 9 — Monitor prediction performance

```powershell
python src/monitoring/evaluate_predictions.py
```

### Step 10 — Simulate production drift

```powershell
python src/monitoring/simulate_drift.py
```

### Step 11 — Detect drift

```powershell
python src/monitoring/detect_drift.py
```

### Step 12 — Evaluate drift decision

```powershell
python src/monitoring/drift_decision.py
```

### Step 13 — Run controlled retraining

```powershell
python src/monitoring/retrain_model.py
```

---

# 26. Monitoring Artifacts

The following monitoring artifacts are generated:

```text
data/monitoring/
├── api_latency_metrics.json
├── drift_decision.json
├── drift_metrics.json
├── monitoring_metrics.json
├── prediction_log.jsonl
├── retraining_result.json
└── simulated_production_data.parquet
```

These provide evidence for the M5 monitoring and retraining workflow.

---

# 27. Project Results Summary

| Area | Result |
|---|---|
| Dataset | 10,000 NYC taxi records |
| Schema validation | Passed |
| Missing-value validation | Passed |
| Timestamp validation | Passed |
| GPS validation | Passed |
| Passenger validation | Passed |
| Trip-duration validation | Passed |
| Prediction monitoring records | 4 |
| Prediction MAE | 90.45 seconds |
| Prediction RMSE | 100.38 seconds |
| Drift feature | `distance_km` |
| Drift PSI | 0.1622 |
| Drift status | Moderate drift |
| Retraining decision | Monitor and evaluate |
| Candidate retraining test | Successful |
| Candidate promotion test | Candidate performed better |
| API requests tested | 50 |
| API success rate | 100% |
| Average API latency | 100.63 ms |
| Median API latency | 31.96 ms |

---

# 28. Limitations

### Static historical dataset

The project uses a fixed 10,000-record historical dataset. It does not represent a continuously arriving production data stream.

### Simulated production outcomes

Actual trip durations used for the monitoring demonstration are simulated because a live production system is not available.

### Simulated drift

The production dataset used for drift detection is intentionally modified to demonstrate the monitoring workflow.

### Controlled retraining

The retraining demonstration uses the available historical dataset. In a production environment, the training data would normally be refreshed with newly collected and labeled observations.

### Latency measurement

The API latency test is a basic local measurement and is not a full load-testing or distributed performance benchmark.

---

# 29. Key MLOps Design Decisions

### Why multiple models?

Training more than one model allows the project to demonstrate objective model comparison rather than assuming a single algorithm is optimal.

### Why MLflow?

MLflow provides experiment tracking and supports reproducibility through logged model runs, parameters and metrics.

### Why DVC?

DVC provides versioning for data and model artifacts that should not be managed directly as ordinary source-code files.

### Why PSI?

PSI provides a simple quantitative method for detecting changes in feature distributions and is suitable for demonstrating data-drift monitoring.

### Why not retrain for moderate drift?

Moderate drift is treated as a signal requiring monitoring and evaluation rather than an automatic replacement of the model. This reduces unnecessary retraining.

### Why compare a candidate before promotion?

A drift event does not guarantee that a newly trained model will perform better. Candidate evaluation protects the current model from being replaced by an inferior model.

---

# 30. Final MLOps Lifecycle

The completed project can be summarized as:

```text
RAW DATA
   │
   ▼
INGESTION
   │
   ▼
VALIDATION
   │
   ▼
FEATURE ENGINEERING
   │
   ▼
DVC VERSIONING
   │
   ▼
MODEL EXPERIMENTATION
   │
   ├──────────────► Random Forest
   │
   └──────────────► Gradient Boosting
                         │
                         ▼
                      MLflow
                         │
                         ▼
                 MODEL COMPARISON
                         │
                         ▼
                  SELECTED MODEL
                         │
                         ▼
                      FASTAPI
                         │
                         ▼
                    PREDICTIONS
                         │
                         ▼
                 PREDICTION LOGGING
                         │
               ┌─────────┴─────────┐
               ▼                   ▼
        PERFORMANCE             DATA DRIFT
         MONITORING             DETECTION
               │                   │
               └─────────┬─────────┘
                         ▼
                RETRAINING DECISION
                         │
                         ▼
                CANDIDATE TRAINING
                         │
                         ▼
               CANDIDATE EVALUATION
                         │
                         ▼
                  MODEL PROMOTION
                         │
                         └──────────► FASTAPI
```

---

# 31. Conclusion

This project demonstrates an end-to-end Machine Learning Engineering workflow rather than only a model-training exercise.

The implemented system covers the major stages of the MLOps lifecycle:

**data → validation → features → versioning → experimentation → model selection → deployment → monitoring → drift detection → retraining → model promotion.**

The architecture is designed so that model predictions can be served through an API, observed through monitoring metrics, evaluated for data drift, and subjected to a controlled retraining process when required.

This provides a production-oriented foundation for an ETA prediction service while remaining reproducible and demonstrable within the scope of the mini-project.
