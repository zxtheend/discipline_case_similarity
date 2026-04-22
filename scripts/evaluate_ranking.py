import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.bootstrap import build_container, close_container
from app.evaluation.ranking_eval import build_report, evaluate_scenario, load_manifest, validate_manifest
from app.models.request import IdentifyRequest
from app.utils.logger import configure_logging
from scripts.seed_test_cases import build_sample_cases


DEFAULT_MANIFEST = PROJECT_ROOT / "docs" / "ranking_eval_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Search + rerank ranking quality.")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to ranking evaluation manifest JSON.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write the evaluation report JSON.",
    )
    parser.add_argument(
        "--run-full-sync",
        action="store_true",
        help="Run full sync before evaluation to refresh Qdrant data.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N scenarios from the manifest.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    scenarios = load_manifest(manifest_path)
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    sample_case_ids = {item["case_id"] for item in build_sample_cases()}
    manifest_errors = validate_manifest(scenarios, sample_case_ids)
    if manifest_errors:
        for error in manifest_errors:
            print("MANIFEST ERROR: {0}".format(error), file=sys.stderr)
        return 1

    container = await build_container()
    configure_logging(container.settings.log_level)
    try:
        if args.run_full_sync:
            request_id = "ranking-eval-full-sync"
            await container.sync_service.full_sync(request_id=request_id)

        scenario_results = []
        for scenario in scenarios:
            result = await _run_single_scenario(container, scenario.query, scenario)
            scenario_results.append(result)
            _print_scenario_result(result)

        report = build_report(
            scenario_results=scenario_results,
            manifest_path=manifest_path,
            run_full_sync=args.run_full_sync,
            limit=args.limit,
        )
        _print_summary(report)

        if args.report_json:
            report_path = Path(args.report_json).resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                report.model_dump_json(indent=2),
                encoding="utf-8",
            )
            print("Report written to {0}".format(report_path))
        return 0
    finally:
        await close_container(container)


async def _run_single_scenario(container, query: IdentifyRequest, scenario):
    try:
        hybrid_candidates = await container.hybrid_search_engine.search(query)
        reranked_candidates = await container.rerank_engine.rerank(
            query=query.description,
            candidates=hybrid_candidates,
        )
        return evaluate_scenario(scenario, reranked_candidates)
    except Exception as exc:
        result = evaluate_scenario(scenario, [])
        result.failures.append("execution_error: {0}".format(exc))
        result.passed = False
        return result


def _print_scenario_result(result) -> None:
    status = "PASS" if result.passed else "FAIL"
    top5 = ", ".join(result.actual_top5) if result.actual_top5 else "-"
    print("[{0}] {1} ({2})".format(status, result.scenario_id, result.category))
    print("  Top5: {0}".format(top5))
    if result.failures:
        print("  Failures: {0}".format("; ".join(result.failures)))
    if result.notes:
        print("  Notes: {0}".format(result.notes))


def _print_summary(report) -> None:
    print("")
    print("Ranking evaluation summary")
    print(
        "Overall: {0}/{1} passed, pass_rate={2:.2%}".format(
            report.passed,
            report.total_scenarios,
            report.pass_rate,
        )
    )
    for summary in report.category_summaries:
        print(
            "- {0}: {1}/{2} passed, pass_rate={3:.2%}".format(
                summary.category,
                summary.passed,
                summary.total,
                summary.pass_rate,
            )
        )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
