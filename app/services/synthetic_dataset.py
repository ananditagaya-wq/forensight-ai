import csv
import math
import random
from pathlib import Path
from typing import Mapping

RECORD_COUNT = 1500
RANDOM_SEED = 20261113
LABEL_NOISE_RATE = 0.08

FEATURE_COLUMNS = (
    "evidence_type",
    "collection_conditions",
    "ppe_compliant",
    "tool_condition",
    "packaging_condition",
    "storage_condition",
    "separately_packaged",
    "chain_of_custody_documented",
)
OUTCOME_COLUMN = "synthetic_outcome"
OUTCOME_LABELS = ("synthetic_lower_risk", "synthetic_higher_risk")
DATASET_SOURCE = (
    f"Purpose-built synthetic dataset; {RECORD_COUNT} records; "
    f"generator seed {RANDOM_SEED}"
)
DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "synthetic_evidence_outcomes.csv"
)

_EVIDENCE_TYPES = ("Biological", "Trace", "Digital", "Firearms", "Document", "Other")
_EVIDENCE_TYPE_WEIGHTS = (0.22, 0.19, 0.16, 0.14, 0.13, 0.16)


def _weighted_choice(
    rng: random.Random, values: tuple[str, ...], weights: tuple[float, ...]
) -> str:
    return rng.choices(values, weights=weights, k=1)[0]


def _higher_risk_probability(record: Mapping[str, object]) -> float:
    """Map simulated handling attributes to a noisy outcome probability."""
    log_odds = -2.0
    log_odds += {
        "Biological": 0.25,
        "Trace": 0.10,
        "Digital": -0.25,
        "Firearms": 0.15,
        "Document": -0.15,
        "Other": 0.0,
    }[str(record["evidence_type"])]
    log_odds += {
        "controlled": 0.0,
        "challenging": 0.75,
        "unknown": 0.35,
    }[str(record["collection_conditions"])]
    log_odds += 0.70 if not record["ppe_compliant"] else 0.0
    log_odds += {
        "clean": 0.0,
        "compromised": 0.80,
        "unknown": 0.30,
    }[str(record["tool_condition"])]
    log_odds += {
        "intact": 0.0,
        "compromised": 0.75,
        "unknown": 0.30,
    }[str(record["packaging_condition"])]
    log_odds += {
        "controlled": 0.0,
        "uncontrolled": 0.70,
        "unknown": 0.25,
    }[str(record["storage_condition"])]
    log_odds += 0.70 if not record["separately_packaged"] else 0.0
    log_odds += 0.65 if not record["chain_of_custody_documented"] else 0.0
    log_odds += (
        0.30
        if record["collection_conditions"] == "challenging"
        and not record["ppe_compliant"]
        else 0.0
    )
    log_odds += (
        0.35
        if record["packaging_condition"] == "compromised"
        and not record["separately_packaged"]
        else 0.0
    )
    return 1.0 / (1.0 + math.exp(-log_odds))


def generate_synthetic_records(
    count: int = RECORD_COUNT, random_state: int = RANDOM_SEED
) -> list[dict[str, object]]:
    """Generate fictional feature records and probabilistic, noisy outcomes."""
    if count < 1:
        raise ValueError("count must be a positive integer.")

    rng = random.Random(random_state)
    records: list[dict[str, object]] = []
    for _ in range(count):
        record: dict[str, object] = {
            "evidence_type": _weighted_choice(
                rng, _EVIDENCE_TYPES, _EVIDENCE_TYPE_WEIGHTS
            ),
            "collection_conditions": _weighted_choice(
                rng, ("controlled", "challenging", "unknown"), (0.70, 0.22, 0.08)
            ),
            "ppe_compliant": rng.random() < 0.84,
            "tool_condition": _weighted_choice(
                rng, ("clean", "compromised", "unknown"), (0.76, 0.18, 0.06)
            ),
            "packaging_condition": _weighted_choice(
                rng, ("intact", "compromised", "unknown"), (0.78, 0.16, 0.06)
            ),
            "storage_condition": _weighted_choice(
                rng, ("controlled", "uncontrolled", "unknown"), (0.70, 0.24, 0.06)
            ),
            "separately_packaged": rng.random() < 0.78,
            "chain_of_custody_documented": rng.random() < 0.82,
        }

        higher_risk = rng.random() < _higher_risk_probability(record)
        if rng.random() < LABEL_NOISE_RATE:
            higher_risk = not higher_risk
        record[OUTCOME_COLUMN] = (
            OUTCOME_LABELS[1] if higher_risk else OUTCOME_LABELS[0]
        )
        records.append(record)
    return records


def write_synthetic_dataset(path: Path | None = None) -> Path:
    """Write the reproducible demonstration dataset as a CSV."""
    destination = path or DATASET_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = generate_synthetic_records()
    with destination.open("w", encoding="utf-8", newline="") as dataset_file:
        writer = csv.DictWriter(
            dataset_file, fieldnames=(*FEATURE_COLUMNS, OUTCOME_COLUMN)
        )
        writer.writeheader()
        writer.writerows(records)
    return destination


if __name__ == "__main__":
    print(write_synthetic_dataset())
