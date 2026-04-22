import tempfile
import unittest
from pathlib import Path

from app.evaluation.ranking_eval import build_report, evaluate_scenario, load_manifest, validate_manifest
from app.models.domain import SearchCandidate
from scripts.seed_test_cases import build_sample_cases


class RankingEvaluationTests(unittest.TestCase):
    def test_load_manifest_parses_repo_manifest(self):
        manifest_path = Path("docs/ranking_eval_manifest.json")

        scenarios = load_manifest(manifest_path)

        self.assertGreaterEqual(len(scenarios), 5)
        self.assertEqual(scenarios[0].category, "same_person_diff_fact")
        self.assertEqual(scenarios[-1].category, "alias_abbreviation")

    def test_validate_manifest_detects_conflicting_expectations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            manifest_path.write_text(
                """
                [
                  {
                    "scenario_id": "conflict-1",
                    "category": "same_person_diff_fact",
                    "query": {
                      "reported_persons": ["王建国"],
                      "reporter": "张某",
                      "location": "太原市",
                      "description": "测试描述",
                      "time_range_years": 5
                    },
                    "expected_top1": "CASE-0001",
                    "expected_in_top3": [],
                    "expected_in_top5": [],
                    "expected_not_in_top5": ["CASE-0001"],
                    "notes": ""
                  }
                ]
                """,
                encoding="utf-8",
            )

            scenarios = load_manifest(manifest_path)
            errors = validate_manifest(
                scenarios,
                known_case_ids={item["case_id"] for item in build_sample_cases()},
            )

        self.assertEqual(len(errors), 1)
        self.assertIn("expected_top1", errors[0])

    def test_evaluate_scenario_applies_topk_rules(self):
        scenario = load_manifest(Path("docs/ranking_eval_manifest.json"))[2]
        ranked_candidates = [
            SearchCandidate(
                case_id="CASE-0021",
                location="朔州市",
                reported_persons=["张明亮"],
                description_text="test-1",
                create_time="2024-01-01T00:00:00+00:00",
                hybrid_score=0.9,
                rerank_score=0.98,
            ),
            SearchCandidate(
                case_id="CASE-0020",
                location="朔州市",
                reported_persons=["张明亮"],
                description_text="test-2",
                create_time="2024-01-01T00:00:00+00:00",
                hybrid_score=0.8,
                rerank_score=0.91,
            ),
        ]

        result = evaluate_scenario(scenario, ranked_candidates)

        self.assertTrue(result.passed)
        self.assertEqual(result.actual_top5, ["CASE-0021", "CASE-0020"])

    def test_build_report_summarizes_results(self):
        scenarios = load_manifest(Path("docs/ranking_eval_manifest.json"))
        passing = evaluate_scenario(scenarios[0], [])
        passing.failures = []
        passing.passed = True
        failing = evaluate_scenario(scenarios[1], [])

        report = build_report(
            scenario_results=[passing, failing],
            manifest_path=Path("docs/ranking_eval_manifest.json"),
            run_full_sync=False,
            limit=None,
        )

        self.assertEqual(report.total_scenarios, 2)
        self.assertEqual(report.passed, 1)
        self.assertEqual(report.failed, 1)
        self.assertAlmostEqual(report.pass_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
