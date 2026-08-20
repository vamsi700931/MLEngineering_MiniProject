from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

COMPARISON_FILE = (
    MODEL_DIR / "model_comparison.json"
)


def main() -> None:
    print("Starting model comparison...")

    # ---------------------------------------------------------
    # Model results from MLflow experiments
    # ---------------------------------------------------------

    results = [
        {
            "model": "RandomForestRegressor",
            "experiment": "random_forest_eta",
            "mlflow_run_id": (
                "cc20de33f30847f2a463ccc5153a5146"
            ),
            "mae_seconds": 499.46,
            "rmse_seconds": 3083.49,
            "r2": -1.3232,
        },
        {
            "model": "GradientBoostingRegressor",
            "experiment": "gradient_boosting_eta",
            "mlflow_run_id": (
                "bd54c1a84d3d4364bf6da28e107ab508"
            ),
            "mae_seconds": 538.70,
            "rmse_seconds": 2692.97,
            "r2": -0.7720,
        },
        {
            "model": "GradientBoostingRegressor",
            "experiment": "tuned_gradient_boosting",
            "mlflow_run_id": (
                "5795a945309f4c909ce00b332e6e874d"
            ),
            "mae_seconds": 504.40,
            "rmse_seconds": 2590.00,
            "r2": -0.6391,
        },
    ]

    # ---------------------------------------------------------
    # Select best model
    #
    # RMSE is used as the primary selection metric because
    # large ETA prediction errors are particularly important.
    # ---------------------------------------------------------

    best_model = min(
        results,
        key=lambda result: result["rmse_seconds"],
    )

    # ---------------------------------------------------------
    # Create comparison output
    # ---------------------------------------------------------

    comparison = {
        "selection_metric": "rmse_seconds",
        "selection_direction": "lower_is_better",
        "models": results,
        "selected_model": best_model,
        "selection_reason": (
            "Selected based on the lowest RMSE among "
            "the evaluated models."
        ),
    }

    # ---------------------------------------------------------
    # Save comparison
    # ---------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        COMPARISON_FILE,
        "w",
    ) as file:
        json.dump(
            comparison,
            file,
            indent=4,
        )

    # ---------------------------------------------------------
    # Display comparison
    # ---------------------------------------------------------

    print("\nModel comparison:")
    print(
        f"{'Model':<30}"
        f"{'MAE':>12}"
        f"{'RMSE':>12}"
        f"{'R²':>12}"
    )

    print("-" * 66)

    for result in results:
        print(
            f"{result['experiment']:<30}"
            f"{result['mae_seconds']:>12.2f}"
            f"{result['rmse_seconds']:>12.2f}"
            f"{result['r2']:>12.4f}"
        )

    print("\nSelected model:")
    print(
        f"Model: "
        f"{best_model['model']}"
    )

    print(
        f"Experiment: "
        f"{best_model['experiment']}"
    )

    print(
        f"RMSE: "
        f"{best_model['rmse_seconds']:.2f} seconds"
    )

    print(
        f"MAE: "
        f"{best_model['mae_seconds']:.2f} seconds"
    )

    print(
        f"R²: "
        f"{best_model['r2']:.4f}"
    )

    print(
        "\nSelection reason: "
        "lowest RMSE among evaluated models."
    )

    print(
        f"\nModel comparison saved to: "
        f"{COMPARISON_FILE}"
    )

    print(
        "Model comparison completed successfully."
    )


if __name__ == "__main__":
    main()