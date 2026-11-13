import csv
from pathlib import Path
from typing import Mapping, Sequence

from sklearn.model_selection import StratifiedGroupKFold

from app.services.synthetic_dataset import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    OUTCOME_COLUMN,
    OUTCOME_LABELS,
)

CATEGORICAL_FEATURES = {
    "evidence_type": {
        "biological",
        "trace",
        "digital",
        "firearms",
        "document",
        "other",
    },
    "collection_conditions": {"controlled", "challenging", "unknown"},
    "tool_condition": {"clean", "compromised", "unknown"},
    "packaging_condition": {"intact", "compromised", "unknown"},
    "storage_condition": {"controlled", "uncontrolled", "unknown"},
}
BOOLEAN_FEATURES = {
    "ppe_compliant",
    "separately_packaged",
    "chain_of_custody_documented",
}


class DatasetValidationError(ValueError):
    """Raised when a labeled dataset is not suitable for this experiment."""


def _normalize_boolean(value: object, row_number: int, column: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return "true"
    if normalized in {"false", "0"}:
        return "false"
    raise DatasetValidationError(
        f"Row {row_number}: {column} must be true or false."
    )


def normalize_features(
    record: Mapping[str, object], row_number: int = 1
) -> list[str]:
    values: list[str] = []
    for column in FEATURE_COLUMNS:
        value = record.get(column)
        if value is None or not str(value).strip():
            raise DatasetValidationError(
                f"Row {row_number}: missing value for {column}."
            )
        normalized = str(value).strip().lower()
        if column in BOOLEAN_FEATURES:
            normalized = _normalize_boolean(value, row_number, column)
        elif normalized not in CATEGORICAL_FEATURES[column]:
            raise DatasetValidationError(
                f"Row {row_number}: unsupported value for {column}."
            )
        values.append(normalized)
    return values


def normalize_row(
    record: Mapping[str, object], row_number: int
) -> tuple[list[str], str]:
    values = normalize_features(record, row_number)
    outcome = str(record.get(OUTCOME_COLUMN, "")).strip()
    if outcome not in OUTCOME_LABELS:
        raise DatasetValidationError(
            f"Row {row_number}: {OUTCOME_COLUMN} must be one of "
            f"{', '.join(sorted(OUTCOME_LABELS))}."
        )
    return values, outcome


def load_synthetic_records(
    dataset_path: Path = DATASET_PATH,
) -> list[dict[str, str]]:
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        if reader.fieldnames is None:
            raise DatasetValidationError("The synthetic dataset has no CSV header.")
        required_columns = {*FEATURE_COLUMNS, OUTCOME_COLUMN}
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            raise DatasetValidationError(
                f"Missing required columns: {', '.join(sorted(missing_columns))}."
            )
        records = list(reader)
    if not records:
        raise DatasetValidationError("The synthetic dataset contains no records.")
    return records


def grouped_stratified_split(
    features: Sequence[Sequence[str]],
    outcomes: Sequence[str],
    random_state: int,
    n_splits: int = 4,
) -> tuple[list[int], list[int]]:
    if len(features) != len(outcomes) or not features:
        raise DatasetValidationError(
            "Features and outcomes must contain the same non-zero number of rows."
        )
    if n_splits < 2 or n_splits > len(features):
        raise DatasetValidationError("n_splits must be between 2 and the row count.")
    if set(outcomes) != set(OUTCOME_LABELS):
        raise DatasetValidationError(
            "The dataset must contain both documented synthetic outcome labels."
        )

    profile_ids: dict[tuple[str, ...], int] = {}
    groups = [
        profile_ids.setdefault(tuple(row), len(profile_ids))
        for row in features
    ]
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_indices, test_indices = next(
        splitter.split(features, outcomes, groups=groups)
    )
    train_classes = {outcomes[index] for index in train_indices}
    test_classes = {outcomes[index] for index in test_indices}
    if train_classes != set(OUTCOME_LABELS) or test_classes != set(OUTCOME_LABELS):
        raise DatasetValidationError(
            "The grouped stratified split must contain both outcome classes "
            "in training and testing records."
        )
    if {groups[index] for index in train_indices}.intersection(
        groups[index] for index in test_indices
    ):
        raise DatasetValidationError(
            "Repeated feature profiles may not cross the train/test boundary."
        )
    return train_indices.tolist(), test_indices.tolist()
