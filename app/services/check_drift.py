import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import PREDICTION_LOG_PATH
from app.services.ml_experiments import load_synthetic_records, normalize_features
from app.services.synthetic_dataset import FEATURE_COLUMNS


def _read_logged_features(path: Path) -> list[list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction log does not exist: {path}")
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            try:
                record = json.loads(line)
                features = record["features"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid prediction log entry at line {line_number}."
                ) from error
            if not isinstance(record, dict) or not isinstance(features, dict):
                raise ValueError(
                    f"Invalid prediction log entry at line {line_number}."
                )
            rows.append(normalize_features(features, row_number=line_number))
    if not rows:
        raise ValueError("The prediction log contains no entries.")
    return rows


def compare_input_distributions(
    logged_features: list[list[str]],
    baseline_features: list[list[str]],
    threshold: float = 0.1,
) -> dict[str, Any]:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be greater than 0 and at most 1.")
    if not logged_features or not baseline_features:
        raise ValueError("Both logged and baseline data must contain records.")

    comparisons: list[dict[str, Any]] = []
    drift_detected = False
    for column_index, column in enumerate(FEATURE_COLUMNS):
        baseline_counts = Counter(row[column_index] for row in baseline_features)
        logged_counts = Counter(row[column_index] for row in logged_features)
        categories = sorted(set(baseline_counts) | set(logged_counts))
        category_differences = [
            {
                "value": category,
                "baseline_share": baseline_counts[category] / len(baseline_features),
                "logged_share": logged_counts[category] / len(logged_features),
            }
            for category in categories
        ]
        maximum_difference = max(
            abs(item["logged_share"] - item["baseline_share"])
            for item in category_differences
        )
        detected = maximum_difference >= threshold
        drift_detected = drift_detected or detected
        comparisons.append(
            {
                "feature": column,
                "maximum_absolute_share_difference": maximum_difference,
                "threshold": threshold,
                "drift_signal": detected,
                "category_shares": category_differences,
            }
        )
    return {
        "status": "data_drift_signal" if drift_detected else "no_threshold_exceeded",
        "baseline_records": len(baseline_features),
        "logged_records": len(logged_features),
        "features": comparisons,
        "limitation": (
            "This compares feature distributions only. It is not a model performance "
            "or accuracy check and cannot assess outcomes without verified labels."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare prediction-input distributions with the synthetic training set."
    )
    parser.add_argument("--log-path", type=Path, default=PREDICTION_LOG_PATH)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()
    baseline_records = load_synthetic_records()
    baseline_features = [
        normalize_features(record, row_number=index)
        for index, record in enumerate(baseline_records, start=2)
    ]
    result = compare_input_distributions(
        _read_logged_features(args.log_path),
        baseline_features,
        threshold=args.threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
