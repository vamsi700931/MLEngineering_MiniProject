import json
import statistics
import time
from pathlib import Path

import requests


# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MONITORING_DIR = PROJECT_ROOT / "data" / "monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = MONITORING_DIR / "api_latency_metrics.json"


# ------------------------------------------------------------------
# API configuration
# ------------------------------------------------------------------

API_URL = "http://127.0.0.1:8000/predict"

NUMBER_OF_REQUESTS = 50


# ------------------------------------------------------------------
# Sample prediction payload
# ------------------------------------------------------------------

PAYLOAD = {
    "vendor_id": 1,
    "passenger_count": 1,
    "pickup_longitude": -73.985,
    "pickup_latitude": 40.748,
    "dropoff_longitude": -73.978,
    "dropoff_latitude": 40.752,
    "store_and_fwd_flag": "N",
    "pickup_hour": 18,
    "pickup_day_of_week": 3,
    "is_weekend": 0,
    "distance_km": 1.25,
}


# ------------------------------------------------------------------
# Latency measurement
# ------------------------------------------------------------------

def measure_latency():
    print("Starting API latency measurement...")
    print(f"API endpoint: {API_URL}")
    print(f"Requests: {NUMBER_OF_REQUESTS}")
    print()

    latencies = []
    successful_requests = 0
    failed_requests = 0

    for request_number in range(1, NUMBER_OF_REQUESTS + 1):

        start_time = time.perf_counter()

        try:
            response = requests.post(
                API_URL,
                json=PAYLOAD,
                timeout=10,
            )

            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000

            if response.status_code == 200:
                successful_requests += 1
                latencies.append(latency_ms)

            else:
                failed_requests += 1

                print(
                    f"Request {request_number}: "
                    f"FAILED - HTTP {response.status_code}"
                )

        except requests.RequestException as exc:
            failed_requests += 1

            print(
                f"Request {request_number}: "
                f"FAILED - {exc}"
            )

    if not latencies:
        raise RuntimeError(
            "No successful API requests were recorded."
        )

    average_latency = statistics.mean(latencies)
    minimum_latency = min(latencies)
    maximum_latency = max(latencies)
    median_latency = statistics.median(latencies)

    results = {
        "api_endpoint": API_URL,
        "total_requests": NUMBER_OF_REQUESTS,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "latency_unit": "milliseconds",
        "average_latency_ms": round(average_latency, 2),
        "median_latency_ms": round(median_latency, 2),
        "minimum_latency_ms": round(minimum_latency, 2),
        "maximum_latency_ms": round(maximum_latency, 2),
    }

    return results


# ------------------------------------------------------------------
# Save results
# ------------------------------------------------------------------

def save_results(results):
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    results = measure_latency()

    print()
    print("API latency results:")
    print("----------------------------------------")
    print(
        f"Total requests:       "
        f"{results['total_requests']}"
    )
    print(
        f"Successful requests:  "
        f"{results['successful_requests']}"
    )
    print(
        f"Failed requests:      "
        f"{results['failed_requests']}"
    )
    print(
        f"Average latency:      "
        f"{results['average_latency_ms']} ms"
    )
    print(
        f"Median latency:       "
        f"{results['median_latency_ms']} ms"
    )
    print(
        f"Minimum latency:      "
        f"{results['minimum_latency_ms']} ms"
    )
    print(
        f"Maximum latency:      "
        f"{results['maximum_latency_ms']} ms"
    )

    save_results(results)

    print()
    print(
        f"Latency metrics saved to: "
        f"{OUTPUT_FILE}"
    )
    print("API latency measurement completed successfully.")


if __name__ == "__main__":
    main()