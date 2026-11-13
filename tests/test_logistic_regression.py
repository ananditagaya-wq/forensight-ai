import csv

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.services.logistic_regression import (
    DatasetValidationError,
    FEATURE_COLUMNS,
    LogisticRegressionExperiment,
    OUTCOME_COLUMN,
    OUTCOME_LABELS,
    get_model_evaluation,
)
from app.services.synthetic_dataset import (
    DATASET_PATH,
    DATASET_SOURCE,
    RECORD_COUNT,
    generate_synthetic_records,
    write_synthetic_dataset,
)


def test_synthetic_dataset_is_reproducible_and_uses_assessment_features():
    first = generate_synthetic_records()
    second = generate_synthetic_records()

    assert len(first) == RECORD_COUNT == 1500
    assert first == second
    assert all(set(record) == {*FEATURE_COLUMNS, OUTCOME_COLUMN} for record in first)
    assert {record[OUTCOME_COLUMN] for record in first} == set(OUTCOME_LABELS)
    assert {record["evidence_type"] for record in first} == {
        "Biological",
        "Trace",
        "Digital",
        "Firearms",
        "Document",
        "Other",
    }


def test_synthetic_labels_include_probabilistic_noise():
    outcomes_by_features = {}
    for record in generate_synthetic_records():
        features = tuple(record[column] for column in FEATURE_COLUMNS)
        outcomes_by_features.setdefault(features, set()).add(record[OUTCOME_COLUMN])

    assert any(
        len(outcomes) == len(OUTCOME_LABELS)
        for outcomes in outcomes_by_features.values()
    )


def test_csv_writer_preserves_the_reproducible_dataset(tmp_path):
    path = write_synthetic_dataset(tmp_path / "synthetic.csv")

    with path.open(encoding="utf-8", newline="") as dataset_file:
        records = list(csv.DictReader(dataset_file))

    expected = [
        {key: str(value) for key, value in record.items()}
        for record in generate_synthetic_records()
    ]
    assert len(records) == RECORD_COUNT
    assert set(records[0]) == {*FEATURE_COLUMNS, OUTCOME_COLUMN}
    assert records == expected


def test_logistic_regression_encodes_features_and_predicts_probabilities():
    records = generate_synthetic_records()
    experiment = LogisticRegressionExperiment()

    evaluation = experiment.train_and_evaluate(records)

    assert evaluation["status"] == "experimental"
    assert evaluation["dataset_type"] == "synthetic"
    assert evaluation["dataset_source"] == DATASET_SOURCE
    assert evaluation["model_status"] == "Experimental — synthetic data"
    assert evaluation["training_records"] == 1125
    assert evaluation["testing_records"] == 375
    assert isinstance(
        experiment.pipeline.named_steps["classifier"], LogisticRegression
    )
    encoder = experiment.pipeline.named_steps["categorical_encoder"]
    assert len(encoder.categories_) == len(FEATURE_COLUMNS)
    assert len(encoder.categories_[0]) == 6
    matrix = evaluation["confusion_matrix"]
    assert matrix["actual_classes"] == matrix["predicted_classes"]
    assert sum(map(sum, matrix["counts"])) == evaluation["testing_records"]
    assert {metric["label"] for metric in evaluation["class_metrics"]} == set(
        OUTCOME_LABELS
    )
    assert "not forensic validation" in evaluation["evaluation_scope"]

    prediction = experiment.predict(records[0])
    assert prediction["experimental_prediction"] in OUTCOME_LABELS
    assert set(prediction["class_probabilities"]) == set(matrix["predicted_classes"])
    assert sum(prediction["class_probabilities"].values()) == pytest.approx(1.0)


def test_holdout_split_is_disjoint_complete_and_stratified():
    records = generate_synthetic_records()
    experiment = LogisticRegressionExperiment()
    experiment.train_and_evaluate(records)

    train_indices = set(experiment.training_indices)
    test_indices = set(experiment.testing_indices)
    assert train_indices.isdisjoint(test_indices)
    assert train_indices | test_indices == set(range(len(records)))
    assert len(test_indices) == 375


def test_reported_metrics_match_held_out_predictions():
    experiment = LogisticRegressionExperiment()
    evaluation = experiment.train_and_evaluate(generate_synthetic_records())
    actual = experiment._test_outcomes
    predicted = experiment._test_predictions
    labels = evaluation["confusion_matrix"]["actual_classes"]

    assert evaluation["accuracy"] == pytest.approx(accuracy_score(actual, predicted))
    assert evaluation["precision_macro"] == pytest.approx(
        precision_score(
            actual, predicted, labels=labels, average="macro", zero_division=0
        )
    )
    assert evaluation["recall_macro"] == pytest.approx(
        recall_score(actual, predicted, labels=labels, average="macro", zero_division=0)
    )
    assert evaluation["macro_f1"] == pytest.approx(
        f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)
    )
    assert evaluation["confusion_matrix"]["counts"] == confusion_matrix(
        actual, predicted, labels=labels
    ).tolist()


def test_dashboard_evaluation_uses_the_bundled_synthetic_dataset():
    evaluation = get_model_evaluation()

    with DATASET_PATH.open(encoding="utf-8", newline="") as dataset_file:
        bundled_records = list(csv.DictReader(dataset_file))
    expected_records = [
        {key: str(value) for key, value in record.items()}
        for record in generate_synthetic_records()
    ]

    assert evaluation["status"] == "experimental"
    assert evaluation["dataset_type"] == "synthetic"
    assert bundled_records == expected_records
    assert evaluation["training_records"] + evaluation["testing_records"] == RECORD_COUNT
    assert all(
        0 <= evaluation[key] <= 1
        for key in ("accuracy", "precision_macro", "recall_macro", "macro_f1")
    )
    assert "synthetic" in evaluation["evaluation_scope"].lower()


def test_training_rejects_small_or_single_class_datasets():
    records = generate_synthetic_records(40)
    records_by_class = {
        label: [record for record in records if record[OUTCOME_COLUMN] == label]
        for label in OUTCOME_LABELS
    }
    small_records = [records_by_class[label][0] for label in OUTCOME_LABELS]
    with pytest.raises(DatasetValidationError, match="At least eight records"):
        LogisticRegressionExperiment().train_and_evaluate(small_records)

    one_class_records = [
        {**record, OUTCOME_COLUMN: OUTCOME_LABELS[0]} for record in records[:20]
    ]
    with pytest.raises(DatasetValidationError, match="both documented synthetic"):
        LogisticRegressionExperiment().train_and_evaluate(one_class_records)
