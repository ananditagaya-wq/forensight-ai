import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from app.services.ml_experiments import (
    grouped_stratified_split,
    load_synthetic_records,
    normalize_row,
)
from app.services.synthetic_dataset import (
    DATASET_SOURCE,
    FEATURE_COLUMNS,
    OUTCOME_LABELS,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "model.joblib"
TRACKING_DB_PATH = PROJECT_ROOT / "mlflow.db"
COMPARISON_PATH = PROJECT_ROOT / "reports" / "experiment_comparison.csv"
LOGGER = logging.getLogger(__name__)


def build_pipeline(
    algorithm: str,
    random_state: int,
    *,
    lr_c: float = 1.0,
    tree_max_depth: int = 4,
    tree_min_samples_leaf: int = 5,
) -> Pipeline:
    if algorithm == "Logistic Regression":
        estimator = LogisticRegression(
            C=lr_c,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
    elif algorithm == "Decision Tree":
        estimator = DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=tree_max_depth,
            min_samples_leaf=tree_min_samples_leaf,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return Pipeline(
        steps=[
            ("categorical_encoder", OneHotEncoder(handle_unknown="ignore")),
            ("classifier", estimator),
        ]
    )


def evaluate_pipeline(
    pipeline: Pipeline,
    train_features: list[list[str]],
    train_outcomes: list[str],
    test_features: list[list[str]],
    test_outcomes: list[str],
) -> dict[str, Any]:
    pipeline.fit(train_features, train_outcomes)
    predictions = [str(value) for value in pipeline.predict(test_features)]
    class_report = classification_report(
        test_outcomes,
        predictions,
        labels=OUTCOME_LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_outcomes, predictions, labels=OUTCOME_LABELS)
    return {
        "accuracy": float(accuracy_score(test_outcomes, predictions)),
        "macro_precision": float(
            precision_score(
                test_outcomes,
                predictions,
                labels=OUTCOME_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                test_outcomes,
                predictions,
                labels=OUTCOME_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                test_outcomes,
                predictions,
                labels=OUTCOME_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "class_metrics": [
            {
                "label": label,
                "precision": float(class_report[label]["precision"]),
                "recall": float(class_report[label]["recall"]),
                "f1": float(class_report[label]["f1-score"]),
                "support": int(class_report[label]["support"]),
            }
            for label in OUTCOME_LABELS
        ],
        "confusion_matrix": {
            "actual_classes": list(OUTCOME_LABELS),
            "predicted_classes": list(OUTCOME_LABELS),
            "counts": matrix.tolist(),
        },
    }


def _tracking_uri(database_path: Path) -> str:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database_path.resolve().as_posix()}"


def run_experiments(
    *,
    lr_c_values: list[float],
    tree_depths: list[int],
    tree_min_samples_leaf: int,
    random_state: int,
    experiment_name: str,
    tracking_db: Path,
    model_path: Path,
    comparison_path: Path,
) -> list[dict[str, Any]]:
    if not lr_c_values or any(value <= 0 for value in lr_c_values):
        raise ValueError("Provide one or more positive Logistic Regression C values.")
    if not tree_depths or any(value < 1 for value in tree_depths):
        raise ValueError("Provide one or more positive Decision Tree depths.")
    if tree_min_samples_leaf < 1:
        raise ValueError("tree_min_samples_leaf must be at least one.")

    records = load_synthetic_records()
    normalized = [
        normalize_row(record, row_number)
        for row_number, record in enumerate(records, start=2)
    ]
    features = [row[0] for row in normalized]
    outcomes = [row[1] for row in normalized]
    train_indices, test_indices = grouped_stratified_split(
        features, outcomes, random_state=random_state
    )
    train_features = [features[index] for index in train_indices]
    test_features = [features[index] for index in test_indices]
    train_outcomes = [outcomes[index] for index in train_indices]
    test_outcomes = [outcomes[index] for index in test_indices]

    mlflow.set_tracking_uri(_tracking_uri(tracking_db))
    artifact_root = tracking_db.parent / "mlartifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_root.resolve().as_uri(),
        )
    mlflow.set_experiment(experiment_name)
    configurations = [
        (
            "Logistic Regression",
            {"C": c_value, "max_iter": 1000, "class_weight": "balanced"},
            {"lr_c": c_value},
        )
        for c_value in lr_c_values
    ] + [
        (
            "Decision Tree",
            {
                "max_depth": depth,
                "min_samples_leaf": tree_min_samples_leaf,
                "class_weight": "balanced",
            },
            {
                "tree_max_depth": depth,
                "tree_min_samples_leaf": tree_min_samples_leaf,
            },
        )
        for depth in tree_depths
    ]

    results: list[dict[str, Any]] = []
    for run_number, (algorithm, hyperparameters, pipeline_options) in enumerate(
        configurations, start=1
    ):
        run_name = f"{run_number:02d}-{algorithm.lower().replace(' ', '-')}"
        pipeline = build_pipeline(algorithm, random_state, **pipeline_options)
        metrics = evaluate_pipeline(
            pipeline,
            train_features,
            train_outcomes,
            test_features,
            test_outcomes,
        )
        with mlflow.start_run(run_name=run_name) as active_run:
            mlflow.set_tags(
                {
                    "algorithm": algorithm,
                    "dataset_type": "synthetic",
                    "split_strategy": "stratified-grouped-by-feature-profile",
                }
            )
            mlflow.log_params(
                {
                    "algorithm": algorithm,
                    **hyperparameters,
                    "random_state": random_state,
                    "split_folds": 4,
                    "training_records": len(train_features),
                    "testing_records": len(test_features),
                    "feature_count": len(FEATURE_COLUMNS),
                    "dataset_source": DATASET_SOURCE,
                }
            )
            mlflow.log_metrics(
                {
                    "accuracy": metrics["accuracy"],
                    "macro_precision": metrics["macro_precision"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_f1": metrics["macro_f1"],
                }
            )
            for class_metric in metrics["class_metrics"]:
                label_key = class_metric["label"].removeprefix("synthetic_")
                mlflow.log_metrics(
                    {
                        f"{label_key}_precision": class_metric["precision"],
                        f"{label_key}_recall": class_metric["recall"],
                        f"{label_key}_f1": class_metric["f1"],
                    }
                )
            mlflow.log_dict(metrics["confusion_matrix"], "confusion_matrix.json")
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format="cloudpickle",
            )
            results.append(
                {
                    "run_name": run_name,
                    "run_id": active_run.info.run_id,
                    "algorithm": algorithm,
                    "hyperparameters": json.dumps(hyperparameters, sort_keys=True),
                    "training_records": len(train_features),
                    "testing_records": len(test_features),
                    **metrics,
                    "pipeline": pipeline,
                }
            )

    best = max(
        results,
        key=lambda result: (
            result["macro_f1"],
            result["accuracy"],
            result["run_name"],
        ),
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best["pipeline"],
            "algorithm": best["algorithm"],
            "selected_run_id": best["run_id"],
            "feature_columns": list(FEATURE_COLUMNS),
            "outcome_labels": list(OUTCOME_LABELS),
            "dataset_source": DATASET_SOURCE,
            "evaluation": {
                key: value
                for key, value in best.items()
                if key
                in {
                    "accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "class_metrics",
                    "confusion_matrix",
                    "training_records",
                    "testing_records",
                }
            },
            "evaluation_note": (
                "Experimental metrics on synthetic data. The held-out split was "
                "used to compare configurations; this is not external validation."
            ),
            "experiment_comparison": [
                {
                    key: result[key]
                    for key in (
                        "run_name",
                        "run_id",
                        "algorithm",
                        "hyperparameters",
                        "accuracy",
                        "macro_precision",
                        "macro_recall",
                        "macro_f1",
                        "training_records",
                        "testing_records",
                        "class_metrics",
                        "confusion_matrix",
                    )
                }
                | {"selected": result["run_id"] == best["run_id"]}
                for result in results
            ],
        },
        model_path,
    )
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "run_name",
        "run_id",
        "algorithm",
        "hyperparameters",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "training_records",
        "testing_records",
    )
    with comparison_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {key: result[key] for key in columns} for result in results
        )

    LOGGER.info(
        "Selected %s (run %s) by macro F1 %.4f, with accuracy %.4f.",
        best["algorithm"],
        best["run_id"],
        best["macro_f1"],
        best["accuracy"],
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare synthetic-data Logistic Regression and Decision Tree models."
    )
    parser.add_argument("--lr-c-values", nargs="+", type=float, default=[0.1, 1.0])
    parser.add_argument("--tree-depths", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--tree-min-samples-leaf", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiment-name", default="forensight-csl236-final")
    parser.add_argument("--tracking-db", type=Path, default=TRACKING_DB_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--comparison-path", type=Path, default=COMPARISON_PATH)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    results = run_experiments(
        lr_c_values=args.lr_c_values,
        tree_depths=args.tree_depths,
        tree_min_samples_leaf=args.tree_min_samples_leaf,
        random_state=args.random_state,
        experiment_name=args.experiment_name,
        tracking_db=args.tracking_db,
        model_path=args.model_path,
        comparison_path=args.comparison_path,
    )
    for result in results:
        print(
            f"{result['run_name']}: {result['algorithm']} "
            f"accuracy={result['accuracy']:.4f} "
            f"macro_precision={result['macro_precision']:.4f} "
            f"macro_recall={result['macro_recall']:.4f} "
            f"macro_f1={result['macro_f1']:.4f}"
        )
    print(f"Saved model: {args.model_path}")
    print(f"Comparison CSV: {args.comparison_path}")
    print(f"Experiment tracking database: {args.tracking_db}")


if __name__ == "__main__":
    main()
