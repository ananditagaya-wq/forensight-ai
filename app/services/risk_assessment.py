from dataclasses import dataclass

from app.schemas import AssessmentRequest


@dataclass(frozen=True)
class RiskAssessment:
    category: str
    score: int
    factors: list[str]
    recommendations: list[str]
    method: str = "preliminary_rule_based"


class RiskPredictionService:
    """Interface for future validated model integrations."""

    def assess(self, evidence: AssessmentRequest) -> RiskAssessment:
        raise NotImplementedError


class PreliminaryRuleBasedService(RiskPredictionService):
    """Transparent screening rules; not a trained contamination prediction model."""

    def assess(self, evidence: AssessmentRequest) -> RiskAssessment:
        points: list[tuple[int, str, str]] = []

        if evidence.collection_conditions == "challenging":
            points.append((2, "Collection conditions were challenging.", "Document site conditions and review whether environmental controls were adequate."))
        elif evidence.collection_conditions == "unknown":
            points.append((1, "Collection conditions were not recorded.", "Clarify and document collection conditions where records are available."))

        if not evidence.ppe_compliant:
            points.append((2, "PPE compliance was not confirmed.", "Review PPE use and personnel handling records; follow applicable PPE procedures."))
        if evidence.tool_condition == "compromised":
            points.append((2, "Collection tool condition was compromised.", "Review tool cleaning or replacement records and document any possible contact transfer."))
        elif evidence.tool_condition == "unknown":
            points.append((1, "Collection tool condition is unknown.", "Verify tool preparation and cleaning records if available."))

        if evidence.packaging_condition == "compromised":
            points.append((2, "Primary packaging was reported compromised.", "Isolate the item as appropriate and document packaging condition and any repackaging."))
        elif evidence.packaging_condition == "unknown":
            points.append((1, "Primary packaging condition is unknown.", "Inspect and document packaging condition using local evidence-handling procedures."))

        if evidence.storage_condition == "uncontrolled":
            points.append((2, "Storage conditions were reported uncontrolled.", "Review storage logs and move evidence to an appropriate controlled environment if necessary."))
        elif evidence.storage_condition == "unknown":
            points.append((1, "Storage conditions are unknown.", "Check storage records and document any gaps in environmental monitoring."))

        if not evidence.separately_packaged:
            points.append((2, "Evidence was not separately packaged.", "Review whether separate packaging is required to reduce contact or cross-transfer risk."))
        if not evidence.chain_of_custody_documented:
            points.append((2, "Chain-of-custody documentation is incomplete.", "Reconcile custody records and document known transfers without backdating entries."))

        score = sum(points_item[0] for points_item in points)
        category = "low" if score <= 2 else "medium" if score <= 5 else "high"
        factors = [
            f"{description} (+{point_value} point{'s' if point_value != 1 else ''})"
            for point_value, description, _ in points
        ] or [
            "No elevated handling indicators were identified in the information provided."
        ]
        recommendations = list(dict.fromkeys(item[2] for item in points)) or [
            "Continue routine evidence-handling, packaging, and custody procedures."
        ]

        return RiskAssessment(
            category=category,
            score=score,
            factors=factors,
            recommendations=recommendations,
        )
