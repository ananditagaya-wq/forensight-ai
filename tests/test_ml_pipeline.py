import csv

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from app.services.check_drift import compare_input_distributions
from app.services.ml_experiments import (
    DatasetValidationError,
    grouped_stratified_split,
    load_synthetic_records,
    normalize_features,
    normalize_row,
)
from app.services.model_runtime import load_model_artifact, predict_with_artifact
from app.services.prediction_logging import append_prediction_log
from app.services.synthetic_dataset import (
    FEATURE_COLUMNS,
    OUTCOME_LABELS,
    OUTCOME_COLUMN,
)
from train import build_pipeline, evaluate_pipeline, run_experiments


def _normalized_data():
    records = load_synthetic_records()
    normalized = [
        normalize_row(record, row_number)
        for row_number, record in enumerate(records, start=2)
    ]
    return records, [row[0] for row in normalized], [row[1] for row in normalized]


def test_dataset_contract_and_preprocessing_reject_unknown_values():
    records, features, outcomes = _normalized_data()

    assert len(records) == 1500
    assert all(len(row) == len(FEATURE_COLUMNS) == 8 for row in features)
    assert set(outcomes) == set(OUTCOME_LABELS)
    assert all(record[OUTCOME_COLUMN] in OUTCOME_LABELS for record in records)

    with pytest.raises(DatasetValidationError, match="unsupported value"):
        normalize_features(
            {**records[0], "evidence_type": "Unknown evidence"}
        )
    with pytest.raises(DatasetValidationError, match="missing value"):
        normalize_features(
            {key: value for key, value in records[0].items() if key != "storage_condition"}
        )


def test_grouped_stratified_split_is_reproducible_and_separates_repeated_profiles():
    _, features, outcomes = _normalized_data()
    train_a, test_a = grouped_stratified_split(features, outcomes, random_state=42)
    train_b, test_b = grouped_stratified_split(features, outcomes, random_state=42)

    assert train_a == train_b
    assert test_a == test_b
    assert set(train_a).isdisjoint(test_a)
    assert set(train_a) | set(test_a) == set(range(len(features)))
    assert {tuple(features[index]) for index in train_a}.isdisjoint(
        {tuple(features[index]) for index in test_a}
    )
    assert {outcomes[index] for index in train_a} == set(OUTCOME_LABELS)
    assert {outcomes[index] for index in test_a} == set(OUTCOME_LABELS)


@pytest.mark.parametrize(
    ("algorithm", "expected_type", "options"),
    [
        ("Logistic Regression", LogisticRegression, {"lr_c": 0.5}),
        ("Decision Tree", DecisionTreeClassifier, {"tree_max_depth": 4}),
    ],
)
def test_both_pipelines_train_predict_and_report_held_out_metrics(
    algorithm, expected_type, options
):
    _, features, outcomes = _normalized_data()
    train_indices, test_indices = grouped_stratified_split(
        features, outcomes, random_state=42
    )
    train_features = [features[index] for index in train_indices]
    train_outcomes = [outcomes[index] for index in train_indices]
    test_features = [features[index] for index in test_indices]
    test_outcomes = [outcomes[index] for index in test_indices]
    pipeline = build_pipeline(algorithm, random_state=42, **options)

    metrics = evaluate_pipeline(
        pipeline,
        train_features,
        train_outcomes,
        test_features,
        test_outcomes,
    )

    assert isinstance(pipeline.named_steps["classifier"], expected_type)
    assert set(pipeline.named_steps["categorical_encoder"].categories_[0]) == {
        "biological",
        "trace",
        "digital",
        "firearms",
        "document",
        "other",
    }
    assert len(pipeline.predict(test_features)) == len(test_indices)
    repeat_pipeline = build_pipeline(algorithm, random_state=42, **options)
    repeat_metrics = evaluate_pipeline(
        repeat_pipeline,
        train_features,
        train_outcomes,
        test_features,
        test_outcomes,
    )
    assert list(pipeline.predict(test_features)) == list(
        repeat_pipeline.predict(test_features)
    )
    assert metrics["macro_f1"] == pytest.approx(repeat_metrics["macro_f1"])
    probabilities = pipeline.predict_proba([features[0]])[0]
    assert sum(probabilities) == pytest.approx(1.0)
    assert set(pipeline.classes_) == set(OUTCOME_LABELS)
    assert all(
        0 <= metrics[key] <= 1
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    )
    assert sum(map(sum, metrics["confusion_matrix"]["counts"])) == len(test_indices)
    assert {item["label"] for item in metrics["class_metrics"]} == set(OUTCOME_LABELS)


def test_mlflow_runs_comparison_and_selected_joblib_artifact(tmp_path):
    tracking_db = tmp_path / "tracking.db"
    model_path = tmp_path / "model" / "model.joblib"
    comparison_path = tmp_path / "comparison.csv"

    results = run_experiments(
        lr_c_values=[0.5],
        tree_depths=[3, 5],
        tree_min_samples_leaf=5,
        random_state=42,
        experiment_name="test-forensight-experiments",
        tracking_db=tracking_db,
        model_path=model_path,
        comparison_path=comparison_path,
    )

    assert len(results) == 3
    assert {result["algorithm"] for result in results} == {
        "Logistic Regression",
        "Decision Tree",
    }
    assert tracking_db.is_file()
    assert comparison_path.is_file()
    with comparison_path.open(encoding="utf-8", newline="") as comparison_file:
        rows = list(csv.DictReader(comparison_file))
    assert len(rows) == 3
    assert {row["algorithm"] for row in rows} == {
        "Logistic Regression",
        "Decision Tree",
    }

    artifact = load_model_artifact(model_path)
    assert artifact is not None
    assert artifact["algorithm"] in {"Logistic Regression", "Decision Tree"}
    prediction = predict_with_artifact(artifact, load_synthetic_records()[0])
    assert prediction["prediction"] in OUTCOME_LABELS
    assert sum(prediction["class_probabilities"].values()) == pytest.approx(1.0)
    assert artifact["selected_run_id"] in {
        result["run_id"] for result in results
    }


def test_missing_model_artifact_is_reported_without_crashing(tmp_path):
    assert load_model_artifact(tmp_path / "not-created.joblib") is None


def test_prediction_log_failure_is_non_fatal_and_drift_is_input_only(tmp_path):
    features = {
        "evidence_type": "Trace",
        "collection_conditions": "controlled",
        "ppe_compliant": True,
        "tool_condition": "clean",
        "packaging_condition": "intact",
        "storage_condition": "controlled",
        "separately_packaged": True,
        "chain_of_custody_documented": True,
    }
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file", encoding="utf-8")
    assert not append_prediction_log(
        blocked_parent / "predictions.jsonl", features, OUTCOME_LABELS[0]
    )

    baseline = normalize_features(features)
    unchanged = compare_input_distributions([baseline], [baseline])
    assert unchanged["status"] == "no_threshold_exceeded"
    assert "cannot assess outcomes" in unchanged["limitation"]
