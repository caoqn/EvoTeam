#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks/MIXED-BENCHMARK/manifests/mixed_benchmark_official_116_plus_838_test_seed20260828.json"


def _rows(run_id: str) -> list[dict]:
    rows = []
    for path in sorted((ROOT / "runs" / run_id / "cases").glob("*/result.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("phase") == "test" and row.get("mixed_task_id"):
            row["rollout_run_id"] = run_id
            rows.append(row)
    return rows


def aggregate_records(run_ids: list[str], rows_by_run: list[list[dict]], expected_tasks: list[dict]) -> dict:
    if len(run_ids) != len(rows_by_run):
        raise ValueError("run_ids and rows_by_run must have matching lengths")
    expected = {
        (str(row["source"]["benchmark"]), str(row["mixed_task_id"]))
        for row in expected_tasks
        if row.get("phase") == "test" and row.get("enabled", True)
    }
    raw_rows: dict[str, dict[tuple[str, str], dict]] = {}
    infra_counts: dict[str, int] = defaultdict(int)
    for run_id, rows in zip(run_ids, rows_by_run):
        indexed = {}
        for row in rows:
            if row.get("phase") != "test" or not row.get("mixed_task_id"):
                continue
            key = (str(row.get("source_benchmark", "")), str(row["mixed_task_id"]))
            if key in indexed:
                raise ValueError(f"duplicate result for {key} in run {run_id}")
            indexed[key] = row
        raw_rows[run_id] = indexed

    task_scores: dict[str, list[dict]] = defaultdict(list)
    task_ids: dict[str, set[str]] = defaultdict(set)
    for benchmark, task_id in expected:
        task_ids[benchmark].add(task_id)
        valid = []
        for run_id in run_ids:
            row = raw_rows[run_id].get((benchmark, task_id))
            if row is None:
                continue
            if row.get("infrastructure_failure"):
                infra_counts[benchmark] += 1
                continue
            if row.get("official_score_eligible", True):
                valid.append(row)
        if len(valid) == len(run_ids):
            task_scores[benchmark].append({
                "avg_score": sum(float(row.get("native_score", 0.0)) for row in valid) / len(valid),
                "avg_correct": sum(bool(row.get("is_correct")) for row in valid) / len(valid),
            })

    summary = {}
    for benchmark, expected_ids in sorted(task_ids.items()):
        complete = task_scores[benchmark]
        per_run_valid = []
        for run_id in run_ids:
            per_run_valid.append(sum(
                1 for task_id in expected_ids
                if (row := raw_rows[run_id].get((benchmark, task_id))) is not None
                and not row.get("infrastructure_failure")
                and row.get("official_score_eligible", True)
            ))
        denominator = len(expected_ids) * len(run_ids)
        summary[benchmark] = {
            "expected_tasks": len(expected_ids),
            "complete_avg_at_tasks": len(complete),
            "incomplete_tasks": len(expected_ids) - len(complete),
            "infrastructure_failures": infra_counts[benchmark],
            "per_rollout_eligible_tasks": per_run_valid,
            "rollout_coverage": round(sum(per_run_valid) / denominator, 4) if denominator else 0.0,
            "avg_at_k_native_score": round(sum(row["avg_score"] for row in complete) / len(complete), 4) if complete else None,
            "avg_at_k_success_rate": round(sum(row["avg_correct"] for row in complete) / len(complete), 4) if complete else None,
        }
        summary[benchmark]["expected_task_ids"] = sorted(expected_ids)
    return {
        "avg_at": len(run_ids),
        "run_ids": run_ids,
        "complete": all(
            row["complete_avg_at_tasks"] == row["expected_tasks"]
            and all(value == row["expected_tasks"] for value in row["per_rollout_eligible_tasks"])
            for row in summary.values()
        ),
        "complete_task_policy": "include a task only when it has an eligible result in every rollout",
        "benchmark_summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+", help="Independent test run IDs, normally three *_rN runs")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.run_ids) != 3:
        parser.error("avg@3 aggregation requires exactly three run IDs")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = aggregate_records(args.run_ids, [_rows(run_id) for run_id in args.run_ids], manifest["tasks"])
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
