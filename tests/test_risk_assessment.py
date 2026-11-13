import pytest
from pydantic import ValidationError

from app.schemas import AssessmentRequest
from app.services.risk_assessment import PreliminaryRuleBasedService


@pytest.fixture
def controlled_evidence():
    return {
        "evidence_ref": "RISK-TEST-001",
        "evidence_type": "Trace",
        "collection_conditions": "controlled",
        "ppe_compliant": True,
        "tool_condition": "clean",
        "packaging_condition": "intact",
        "storage_condition": "controlled",
        "separately_packaged": True,
        "chain_of_custody_documented": True,
    }


def assess(payload):
    return PreliminaryRuleBasedService().assess(AssessmentRequest(**payload))


def test_all_controlled_conditions_have_no_elevated_indicators(controlled_evidence):
    result = assess(controlled_evidence)

    assert (result.score, result.category) == (0, "low")
    assert result.factors == [
        "No elevated handling indicators were identified in the information provided."
    ]
    assert result.recommendations == [
        "Continue routine evidence-handling, packaging, and custody procedures."
    ]


def test_all_compromised_conditions_contribute_two_points_each(controlled_evidence):
    controlled_evidence.update(
        collection_conditions="challenging",
        ppe_compliant=False,
        tool_condition="compromised",
        packaging_condition="compromised",
        storage_condition="uncontrolled",
        separately_packaged=False,
        chain_of_custody_documented=False,
    )

    result = assess(controlled_evidence)

    assert (result.score, result.category) == (14, "high")
    assert len(result.factors) == len(result.recommendations) == 7
    assert all(factor.endswith("(+2 points)") for factor in result.factors)
    assert "PPE compliance was not confirmed. (+2 points)" in result.factors


def test_mixed_conditions_sum_each_contribution_and_match_recommendations(
    controlled_evidence,
):
    controlled_evidence.update(
        collection_conditions="challenging",
        tool_condition="unknown",
        storage_condition="uncontrolled",
    )

    result = assess(controlled_evidence)

    assert (result.score, result.category) == (5, "medium")
    assert result.factors == [
        "Collection conditions were challenging. (+2 points)",
        "Collection tool condition is unknown. (+1 point)",
        "Storage conditions were reported uncontrolled. (+2 points)",
    ]
    assert result.recommendations == [
        "Document site conditions and review whether environmental controls were adequate.",
        "Verify tool preparation and cleaning records if available.",
        "Review storage logs and move evidence to an appropriate controlled environment if necessary.",
    ]


def test_unknown_conditions_receive_one_point_each(controlled_evidence):
    controlled_evidence.update(
        collection_conditions="unknown",
        tool_condition="unknown",
        packaging_condition="unknown",
        storage_condition="unknown",
    )

    result = assess(controlled_evidence)

    assert (result.score, result.category) == (4, "medium")
    assert len(result.factors) == len(result.recommendations) == 4
    assert all(factor.endswith("(+1 point)") for factor in result.factors)


@pytest.mark.parametrize(
    ("score_inputs", "expected_score", "expected_category"),
    [
        ({"collection_conditions": "challenging"}, 2, "low"),
        ({"collection_conditions": "challenging", "tool_condition": "unknown"}, 3, "medium"),
        ({"collection_conditions": "challenging", "ppe_compliant": False, "tool_condition": "unknown"}, 5, "medium"),
        ({"collection_conditions": "challenging", "ppe_compliant": False, "tool_condition": "compromised"}, 6, "high"),
    ],
)
def test_category_boundary_scores(
    controlled_evidence, score_inputs, expected_score, expected_category
):
    controlled_evidence.update(score_inputs)

    result = assess(controlled_evidence)

    assert result.category == expected_category
    assert result.score == expected_score


@pytest.mark.parametrize(
    "missing_field",
    [
        "collection_conditions",
        "ppe_compliant",
        "tool_condition",
        "packaging_condition",
        "storage_condition",
        "separately_packaged",
        "chain_of_custody_documented",
    ],
)
def test_missing_handling_conditions_are_rejected(controlled_evidence, missing_field):
    del controlled_evidence[missing_field]

    with pytest.raises(ValidationError):
        AssessmentRequest(**controlled_evidence)
