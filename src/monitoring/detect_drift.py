from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset_10000.parquet"
)

PRODUCTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "simulated_production_data.parquet"
)

MONITORING_DIR = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
)

DRIFT_METRICS_FILE = (
    MONITORING_DIR
    / "drift_metrics.json"
)

FEATURES_TO_MONITOR = [
    "distance_km",
]


def calculate_psi(
    reference: pd.Series,
    production: pd.Series,
    bins: int = 10,
) -> float:
    """
    Calculate Population Stability Index (PSI).

    PSI measures how much a production distribution has
    shifted compared with a reference distribution.
    """

    reference = reference.dropna()
    production = production.dropna()

    # Create bins using the reference distribution.
    breakpoints = np.percentile(
        reference,
        np.linspace(0, 100, bins + 1),
    )

    # Remove duplicate bin edges.
    breakpoints = np.unique(breakpoints)

    # If there are not enough unique values, PSI cannot
    # be calculated reliably using these bins.
    if len(breakpoints) < 3:
        return 0.0

    # Extend the first and last boundaries so that all
    # production values are included.
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    reference_buckets = pd.cut(
        reference,
        bins=breakpoints,
        include_lowest=True,
    )

    production_buckets = pd.cut(
        production,
        bins=breakpoints,
        include_lowest=True,
    )

    reference_distribution = (
        reference_buckets
        .value_counts(normalize=True, sort=False)
    )

    production_distribution = (
        production_buckets
        .value_counts(normalize=True, sort=False)
    )

    # Replace zero proportions to avoid log(0).
    epsilon = 0.0001

    reference_distribution = (
        reference_distribution
        .clip(lower=epsilon)
    )

    production_distribution = (
        production_distribution
        .clip(lower=epsilon)
    )

    psi = (
        (
            production_distribution
            - reference_distribution
        )
        * np.log(
            production_distribution
            / reference_distribution
        )
    ).sum()

    return float(psi)


def classify_drift(psi: float) -> str:
    """
    Classify drift based on PSI value.
    """

    if psi < 0.10:
        return "no_significant_drift"

    if psi < 0.25:
        return "moderate_drift"

    return "significant_drift"


def main() -> None:
    print("Starting drift detection...")

    # ---------------------------------------------------------
    # Validate input files
    # ---------------------------------------------------------

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {REFERENCE_FILE}"
        )

    if not PRODUCTION_FILE.exists():
        raise FileNotFoundError(
            f"Production dataset not found: {PRODUCTION_FILE}"
        )

    # ---------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------

    reference_df = pd.read_parquet(
        REFERENCE_FILE
    )

    production_df = pd.read_parquet(
        PRODUCTION_FILE
    )

    print(
        f"Reference records:  {len(reference_df):,}"
    )

    print(
        f"Production records: {len(production_df):,}"
    )

    # ---------------------------------------------------------
    # Calculate PSI for monitored features
    # ---------------------------------------------------------

    results = []

    for feature in FEATURES_TO_MONITOR:

        if feature not in reference_df.columns:
            raise ValueError(
                f"Feature '{feature}' not found "
                "in reference dataset."
            )

        if feature not in production_df.columns:
            raise ValueError(
                f"Feature '{feature}' not found "
                "in production dataset."
            )

        reference_values = reference_df[feature]
        production_values = production_df[feature]

        psi = calculate_psi(
            reference_values,
            production_values,
        )

        drift_status = classify_drift(psi)

        result = {
            "feature": feature,
            "psi": round(psi, 4),
            "drift_status": drift_status,
            "reference_mean": round(
                float(reference_values.mean()),
                4,
            ),
            "production_mean": round(
                float(production_values.mean()),
                4,
            ),
            "reference_median": round(
                float(reference_values.median()),
                4,
            ),
            "production_median": round(
                float(production_values.median()),
                4,
            ),
        }

        results.append(result)

    # ---------------------------------------------------------
    # Overall drift decision
    # ---------------------------------------------------------

    significant_drift_features = [
        result["feature"]
        for result in results
        if result["drift_status"] == "significant_drift"
    ]

    moderate_drift_features = [
        result["feature"]
        for result in results
        if result["drift_status"] == "moderate_drift"
    ]

    overall_drift_detected = (
        len(significant_drift_features) > 0
        or len(moderate_drift_features) > 0
    )

    monitoring_result = {
        "reference_dataset": str(
            REFERENCE_FILE.relative_to(PROJECT_ROOT)
        ),
        "production_dataset": str(
            PRODUCTION_FILE.relative_to(PROJECT_ROOT)
        ),
        "psi_thresholds": {
            "no_significant_drift": "< 0.10",
            "moderate_drift": "0.10 - 0.25",
            "significant_drift": "> 0.25",
        },
        "overall_drift_detected": (
            overall_drift_detected
        ),
        "significant_drift_features": (
            significant_drift_features
        ),
        "moderate_drift_features": (
            moderate_drift_features
        ),
        "features": results,
    }

    # ---------------------------------------------------------
    # Save drift metrics
    # ---------------------------------------------------------

    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        DRIFT_METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            monitoring_result,
            file,
            indent=4,
        )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("\nDrift detection results:")
    print(
        f"{'Feature':<25}"
        f"{'PSI':>10}"
        f"{'Status':>25}"
    )

    print("-" * 60)

    for result in results:

        print(
            f"{result['feature']:<25}"
            f"{result['psi']:>10.4f}"
            f"{result['drift_status']:>25}"
        )

    print(
        f"\nOverall drift detected: "
        f"{overall_drift_detected}"
    )

    print(
        f"Drift metrics saved to: "
        f"{DRIFT_METRICS_FILE}"
    )

    print(
        "Drift detection completed successfully."
    )


if __name__ == "__main__":
    main()