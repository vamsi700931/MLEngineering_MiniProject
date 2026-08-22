from pathlib import Path
import json


# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DRIFT_METRICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "drift_metrics.json"
)

DRIFT_DECISION_FILE = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "drift_decision.json"
)


# ------------------------------------------------------------------
# Drift thresholds
# ------------------------------------------------------------------

NO_DRIFT_THRESHOLD = 0.10
SIGNIFICANT_DRIFT_THRESHOLD = 0.25


# ------------------------------------------------------------------
# Determine action
# ------------------------------------------------------------------

def determine_action(psi: float) -> tuple[str, str, bool]:
    """
    Determine the operational action based on PSI.

    Returns:
        drift_status
        recommended_action
        retraining_recommended
    """

    if psi < NO_DRIFT_THRESHOLD:
        return (
            "no_significant_drift",
            "NO_ACTION",
            False,
        )

    if psi < SIGNIFICANT_DRIFT_THRESHOLD:
        return (
            "moderate_drift",
            "MONITOR_AND_EVALUATE",
            False,
        )

    return (
        "significant_drift",
        "RETRAINING_RECOMMENDED",
        True,
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print("Starting drift decision analysis...")

    # --------------------------------------------------------------
    # Validate input
    # --------------------------------------------------------------

    if not DRIFT_METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Drift metrics file not found: "
            f"{DRIFT_METRICS_FILE}"
        )

    # --------------------------------------------------------------
    # Load drift metrics
    # --------------------------------------------------------------

    with open(
        DRIFT_METRICS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        drift_metrics = json.load(file)

    features = drift_metrics.get("features", [])

    if not features:
        raise ValueError(
            "No feature drift results found in "
            "drift_metrics.json."
        )

    # --------------------------------------------------------------
    # Evaluate each feature
    # --------------------------------------------------------------

    feature_decisions = []

    for feature_result in features:

        feature = feature_result["feature"]
        psi = float(feature_result["psi"])

        drift_status, action, retraining = (
            determine_action(psi)
        )

        feature_decisions.append(
            {
                "feature": feature,
                "psi": round(psi, 4),
                "drift_status": drift_status,
                "recommended_action": action,
                "retraining_recommended": retraining,
            }
        )

    # --------------------------------------------------------------
    # Determine overall action
    # --------------------------------------------------------------

    retraining_features = [
        result["feature"]
        for result in feature_decisions
        if result["retraining_recommended"]
    ]

    moderate_drift_features = [
        result["feature"]
        for result in feature_decisions
        if result["drift_status"] == "moderate_drift"
    ]

    if retraining_features:
        overall_status = "SIGNIFICANT_DRIFT"
        overall_action = "RETRAINING_RECOMMENDED"
        retraining_recommended = True

    elif moderate_drift_features:
        overall_status = "MODERATE_DRIFT"
        overall_action = "MONITOR_AND_EVALUATE"
        retraining_recommended = False

    else:
        overall_status = "NO_SIGNIFICANT_DRIFT"
        overall_action = "NO_ACTION"
        retraining_recommended = False

    # --------------------------------------------------------------
    # Build decision result
    # --------------------------------------------------------------

    decision_result = {
        "decision_policy": {
            "no_drift_threshold": NO_DRIFT_THRESHOLD,
            "significant_drift_threshold": (
                SIGNIFICANT_DRIFT_THRESHOLD
            ),
        },
        "overall_status": overall_status,
        "overall_action": overall_action,
        "retraining_recommended": retraining_recommended,
        "retraining_features": retraining_features,
        "moderate_drift_features": (
            moderate_drift_features
        ),
        "features": feature_decisions,
    }

    # --------------------------------------------------------------
    # Save decision
    # --------------------------------------------------------------

    DRIFT_DECISION_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        DRIFT_DECISION_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            decision_result,
            file,
            indent=4,
        )

    # --------------------------------------------------------------
    # Display decision
    # --------------------------------------------------------------

    print("\nDrift decision results:")

    print(
        f"{'Feature':<25}"
        f"{'PSI':>10}"
        f"{'Status':>25}"
        f"{'Action':>30}"
    )

    print("-" * 90)

    for result in feature_decisions:

        print(
            f"{result['feature']:<25}"
            f"{result['psi']:>10.4f}"
            f"{result['drift_status']:>25}"
            f"{result['recommended_action']:>30}"
        )

    print("\nOverall decision:")
    print(f"Status: {overall_status}")
    print(f"Action: {overall_action}")
    print(
        f"Retraining recommended: "
        f"{retraining_recommended}"
    )

    print(
        f"\nDecision saved to: "
        f"{DRIFT_DECISION_FILE}"
    )

    print(
        "Drift decision analysis completed successfully."
    )


if __name__ == "__main__":
    main()