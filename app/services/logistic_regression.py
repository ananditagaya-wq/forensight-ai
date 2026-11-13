from typing import Mapping, Sequence

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

from app.services.ml_experiments import (
    DatasetValidationError,
    grouped_stratified_split,
    load_synthetic_records,
    normalize_features,
    normalize_row,
)
from app.services.synthetic_dataset import (
    DATASET_SOURCE,
    FEATURE_COLUMNS,
    OUTCOME_COLUMN,
    OUTCOME_LABELS,
)

RANDOM_STATE = 42
TEST_SIZE = 0.25

__all__ = [
    "DatasetValidationError",
    "FEATURE_COLUMNS",
    "LogisticRegressionExperiment",
    "OUTCOME_COLUMN",
    "OUTCOME_LABELS",
    "get_model_evaluation",
]


class LogisticRegressionExperiment:
    """Synthetic-data holdout evaluation; separate from rule-based scoring."""

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.classes: tuple[str, ...] = ()
        self.training_indices: tuple[int, ...] = ()
        self.testing_indices: tuple[int, ...] = ()
        self._test_outcomes: tuple[str, ...] = ()
        self._test_predictions: tuple[str, ...] = ()

    def train_and_evaluate(
        self,
        records: Sequence[Mapping[str, object]],
    ) -> dict:
        if not records:
            raise DatasetValidationError("The dataset contains no labeled records.")

        normalized = [
            normalize_row(record, row_number)
            for row_number, record in enumerate(records, start=2)
        ]
        features = [row[0] for row in normalized]
        outcomes = [row[1] for row in normalized]
        classes = tuple(sorted(set(outcomes)))
        class_counts = {label: outcomes.count(label) for label in classes}
        if set(classes) != set(OUTCOME_LABELS):
            raise DatasetValidationError(
                "The dataset must contain both documented synthetic outcome labels."
            )
        if len(records) < 8 or min(class_counts.values()) < 2:
            raise DatasetValidationError(
                "At least eight records and at least two records per outcome class "
                "are required for a stratified holdout split."
            )

        train_indices, test_indices = grouped_stratified_split(
            features, outcomes, random_state=RANDOM_STATE, n_splits=round(1 / TEST_SIZE)
        )
        self.training_indices = tuple(train_indices)
        self.testing_indices = tuple(test_indices)
        train_features = [features[index] for index in train_indices]
        test_features = [features[index] for index in test_indices]
        train_outcomes = [outcomes[index] for index in train_indices]
        test_outcomes = [outcomes[index] for index in test_indices]
        self.pipeline = Pipeline(
            steps=[
                ("categorical_encoder", OneHotEncoder(handle_unknown="ignore")),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        self.pipeline.fit(train_features, train_outcomes)
        self.classes = classes
        predictions = self.pipeline.predict(test_features)
        self._test_outcomes = tuple(test_outcomes)
        self._test_predictions = tuple(str(value) for value in predictions)
        class_report = classification_report(
            test_outcomes,
            predictions,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(test_outcomes, predictions, labels=classes)

        return {
            "algorithm": "Logistic Regression",
            "status": "experimental",
            "model_status": "Experimental — synthetic data",
            "dataset_source": DATASET_SOURCE,
            "dataset_type": "synthetic",
            "training_records": len(train_features),
            "testing_records": len(test_features),
            "accuracy": float(accuracy_score(test_outcomes, predictions)),
            "precision_macro": float(
                precision_score(
                    test_outcomes,
                    predictions,
                    labels=classes,
                    average="macro",
                    zero_division=0,
                )
            ),
            "recall_macro": float(
                recall_score(
                    test_outcomes,
                    predictions,
                    labels=classes,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_f1": float(
                f1_score(
                    test_outcomes,
                    predictions,
                    labels=classes,
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
                for label in classes
            ],
            "confusion_matrix": {
                "actual_classes": list(classes),
                "predicted_classes": list(classes),
                "counts": matrix.tolist(),
            },
            "evaluation_scope": (
                "Experimental holdout metrics on purpose-built synthetic records "
                "with probabilistically simulated outcomes and controlled label "
                "noise. These metrics do not measure real-world contamination "
                "detection and are not forensic validation."
            ),
        }

    def predict(self, features: Mapping[str, object]) -> dict[str, object]:
        if self.pipeline is None:
            raise RuntimeError("Train the experiment with labeled data before prediction.")
        normalized = normalize_features(features, 1)
        probabilities = self.pipeline.predict_proba([normalized])[0]
        prediction = str(self.pipeline.predict([normalized])[0])
        return {
            "experimental_prediction": prediction,
            "class_probabilities": {
                str(label): float(probability)
                for label, probability in zip(self.pipeline.classes_, probabilities)
            },
        }


def get_model_evaluation() -> dict:
    """Train and evaluate only on the bundled, reproducible synthetic dataset."""
    records = load_synthetic_records()
    return LogisticRegressionExperiment().train_and_evaluate(records)
