import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from app.models.domain import SearchCandidate
from app.models.request import IdentifyRequest

ScenarioCategory = Literal[
    "same_person_diff_fact",
    "same_fact_diff_person",
    "same_location_diff_department",
    "paraphrase_colloquial",
    "alias_abbreviation",
]


class RankingScenario(BaseModel):
    scenario_id: str = Field(min_length=1)
    category: ScenarioCategory
    query: IdentifyRequest
    expected_top1: Optional[str] = None
    expected_in_top3: List[str] = Field(default_factory=list)
    expected_in_top5: List[str] = Field(default_factory=list)
    expected_not_in_top5: List[str] = Field(default_factory=list)
    notes: str = ""


class RankedCaseResult(BaseModel):
    case_id: str
    rank: int = Field(ge=1)
    location: str
    reported_persons: List[str] = Field(default_factory=list)
    hybrid_score: float = 0.0
    rerank_score: Optional[float] = None


class ScenarioEvaluationResult(BaseModel):
    scenario_id: str
    category: ScenarioCategory
    passed: bool
    failures: List[str] = Field(default_factory=list)
    actual_top5: List[str] = Field(default_factory=list)
    ranked_cases: List[RankedCaseResult] = Field(default_factory=list)
    notes: str = ""


class CategorySummary(BaseModel):
    category: ScenarioCategory
    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0


class RankingEvaluationReport(BaseModel):
    generated_at: datetime
    manifest_path: str
    run_full_sync: bool = False
    limit: Optional[int] = None
    total_scenarios: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    category_summaries: List[CategorySummary] = Field(default_factory=list)
    scenario_results: List[ScenarioEvaluationResult] = Field(default_factory=list)


def load_manifest(path: Path) -> List[RankingScenario]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Ranking evaluation manifest must be a JSON array.")
    return [RankingScenario.model_validate(item) for item in payload]


def validate_manifest(
    scenarios: Sequence[RankingScenario],
    known_case_ids: Iterable[str],
) -> List[str]:
    errors: List[str] = []
    known_ids = set(known_case_ids)
    seen_scenario_ids = set()

    for scenario in scenarios:
        if scenario.scenario_id in seen_scenario_ids:
            errors.append("Duplicate scenario_id: {0}".format(scenario.scenario_id))
        seen_scenario_ids.add(scenario.scenario_id)

        overlaps = (
            set(scenario.expected_in_top3)
            .union(set(scenario.expected_in_top5))
            .intersection(set(scenario.expected_not_in_top5))
        )
        if overlaps:
            errors.append(
                "Scenario {0} has contradictory expectations for case_ids: {1}".format(
                    scenario.scenario_id,
                    ", ".join(sorted(overlaps)),
                )
            )

        if scenario.expected_top1 and scenario.expected_top1 in scenario.expected_not_in_top5:
            errors.append(
                "Scenario {0} sets expected_top1 inside expected_not_in_top5.".format(
                    scenario.scenario_id
                )
            )

        referenced_case_ids = set(scenario.expected_in_top3)
        referenced_case_ids.update(scenario.expected_in_top5)
        referenced_case_ids.update(scenario.expected_not_in_top5)
        if scenario.expected_top1:
            referenced_case_ids.add(scenario.expected_top1)

        unknown_case_ids = sorted(case_id for case_id in referenced_case_ids if case_id not in known_ids)
        if unknown_case_ids:
            errors.append(
                "Scenario {0} references unknown case_ids: {1}".format(
                    scenario.scenario_id,
                    ", ".join(unknown_case_ids),
                )
            )

    return errors


def evaluate_scenario(
    scenario: RankingScenario,
    ranked_candidates: Sequence[SearchCandidate],
) -> ScenarioEvaluationResult:
    ranked_cases = [
        RankedCaseResult(
            case_id=candidate.case_id,
            rank=index,
            location=candidate.location,
            reported_persons=list(candidate.reported_persons),
            hybrid_score=float(candidate.hybrid_score),
            rerank_score=candidate.rerank_score,
        )
        for index, candidate in enumerate(ranked_candidates[:5], start=1)
    ]
    actual_top5 = [item.case_id for item in ranked_cases]
    actual_top3 = actual_top5[:3]
    failures: List[str] = []

    if scenario.expected_top1:
        actual_top1 = actual_top5[0] if actual_top5 else None
        if actual_top1 != scenario.expected_top1:
            failures.append(
                "expected_top1={0}, actual_top1={1}".format(
                    scenario.expected_top1,
                    actual_top1 or "NONE",
                )
            )

    for case_id in scenario.expected_in_top3:
        if case_id not in actual_top3:
            failures.append("{0} missing from Top3".format(case_id))

    for case_id in scenario.expected_in_top5:
        if case_id not in actual_top5:
            failures.append("{0} missing from Top5".format(case_id))

    for case_id in scenario.expected_not_in_top5:
        if case_id in actual_top5:
            failures.append("{0} unexpectedly appeared in Top5".format(case_id))

    return ScenarioEvaluationResult(
        scenario_id=scenario.scenario_id,
        category=scenario.category,
        passed=not failures,
        failures=failures,
        actual_top5=actual_top5,
        ranked_cases=ranked_cases,
        notes=scenario.notes,
    )


def build_report(
    scenario_results: Sequence[ScenarioEvaluationResult],
    manifest_path: Path,
    run_full_sync: bool,
    limit: Optional[int],
) -> RankingEvaluationReport:
    category_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    total_passed = 0

    for result in scenario_results:
        category_totals[result.category]["total"] += 1
        if result.passed:
            category_totals[result.category]["passed"] += 1
            total_passed += 1

    category_summaries = []
    for category in sorted(category_totals):
        total = category_totals[category]["total"]
        passed = category_totals[category]["passed"]
        failed = total - passed
        category_summaries.append(
            CategorySummary(
                category=category,
                total=total,
                passed=passed,
                failed=failed,
                pass_rate=_pass_rate(passed, total),
            )
        )

    total_scenarios = len(scenario_results)
    return RankingEvaluationReport(
        generated_at=datetime.now(timezone.utc),
        manifest_path=str(manifest_path),
        run_full_sync=run_full_sync,
        limit=limit,
        total_scenarios=total_scenarios,
        passed=total_passed,
        failed=total_scenarios - total_passed,
        pass_rate=_pass_rate(total_passed, total_scenarios),
        category_summaries=category_summaries,
        scenario_results=list(scenario_results),
    )


def _pass_rate(passed: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(passed / total, 4)
