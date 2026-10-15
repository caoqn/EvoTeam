from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
for path in (str(THIS_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.result_manifest import ResultManifest
from core.run_manager import RunManager
from benchmarks.mixed_native.data_paths import mixed_data_root
from mixed_benchmark_manifest import MixedBenchmarkManifest, MixedTask
from task_wrapper import MixedTaskWrapper


class MixedScheduler:

    def __init__(
        self,
        manifest: MixedBenchmarkManifest,
        *,
        team_name: str | None = None,
        run_id: str | None = None,
        resume: bool = False,
        phase: str = "evolve",
    ):
        self.manifest = manifest
        self.phase = phase
        self.wrapper = MixedTaskWrapper(manifest)
        data_root = mixed_data_root(PROJECT_ROOT)
        required_paths = {"tool implementations": data_root / "tools"}
        if "gaia" in manifest.active_benchmarks:
            required_paths["GAIA data"] = data_root / "data" / "gaia"
        if "locobench" in manifest.active_benchmarks:
            required_paths["LoCoBench installation"] = data_root / "benchmarks" / "LoCoBench"
        if "locabench" in manifest.active_benchmarks:
            required_paths["LOCA-Bench installation"] = data_root / "benchmarks" / "LOCA-bench"
        missing = [name for name, path in required_paths.items() if not path.exists()]
        if "beyondswe" in manifest.active_benchmarks:
            configured = os.environ.get("BEYONDSWE_DATA_FILE", "").strip()
            candidates = [Path(configured).expanduser()] if configured else [
                data_root / "benchmarks" / "BeyondSWE" / "beyondswe.jsonl",
                data_root / "benchmarks" / "BeyondSWE" / "data" / "beyondswe.jsonl",
            ]
            if not any(path.is_file() for path in candidates):
                missing.append("BeyondSWE JSONL")
        if missing:
            raise FileNotFoundError("MIXED-BENCHMARK external assets missing: " + ", ".join(missing))
        declared_team = str(manifest.shared_evolution["team_name"])
        self.team_name = team_name or declared_team
        self.source_team_dir = PROJECT_ROOT / "agents" / self.team_name
        if not self.source_team_dir.exists():
            raise FileNotFoundError(
                f"shared MIXED-BENCHMARK team is missing: {self.source_team_dir}. "
                "Create one common Agent namespace before executing tasks."
            )
        pool_config_path = self.source_team_dir / "pool.yaml"
        pool_payload = {}
        if pool_config_path.exists():
            import yaml
            pool_payload = yaml.safe_load(pool_config_path.read_text(encoding="utf-8")) or {}
        pool_mode = (pool_payload.get("settings") or {}).get(
            "orchestration_mode", "evoteam"
        )
        if (
            self.team_name != declared_team
            and pool_mode != "baseline_shared"
            and phase != "test"
        ):
            raise ValueError(
                f"MIXED-BENCHMARK requires the manifest shared team {declared_team!r}, "
                "unless the explicitly selected pool declares "
                "settings.orchestration_mode=baseline_shared; "
                f"received {self.team_name!r} ({pool_mode!r})"
            )
        self.orchestration_mode = str(pool_mode)
        if self.orchestration_mode not in {"evoteam", "baseline_shared"}:
            raise ValueError(
                "MIXED-BENCHMARK scheduler supports only 'evoteam' and "
                f"'baseline_shared' pools, got {self.orchestration_mode!r}"
            )
        if pool_config_path.exists():
            if (pool_payload.get("settings") or {}).get("answer_protocol"):
                raise ValueError(
                    "MIXED-BENCHMARK shared pool must not define settings.answer_protocol; "
                    "each native adapter supplies its output contract per task"
                )
        self.run_id = run_id or (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + f"_{manifest.suite_id}"
        )
        self.resume = resume
        self.run_mgr = RunManager.create_run(
            run_id=self.run_id,
            source_team_dir=self.source_team_dir,
            team_name=self.team_name,
            config={
                "suite_id": manifest.suite_id,
                "benchmark": "mixed-benchmark",
                "manifest": str(manifest.suite_id),
                "ordering": dict(manifest.ordering),
                "shared_evolution": dict(manifest.shared_evolution),
                "selected_orchestration_mode": self.orchestration_mode,
            },
            resume=resume,
        )
        self.ledger = ResultManifest(
            PROJECT_ROOT / "runs" / "manifests" / f"{manifest.suite_id}.json",
            "mixed-benchmark",
            manifest.suite_id,
        )
        self._global_indices = {
            task.mixed_task_id: index
            for index, task in enumerate(
                self.manifest.tasks_for_phase("evolve", enabled_only=False)
                + self.manifest.tasks_for_phase("test", enabled_only=False)
            )
        }

    def _native_args(self, adapter: Any, task: MixedTask, evolve: bool) -> argparse.Namespace:
        args = self.wrapper._adapter_namespace(adapter, task)
        args.evolve = evolve
        args.team = self.team_name
        args.workers = 1
        args.run_id = self.run_id
        args.resume = False
        args.result_manifest = None
        args.manifest_replace_valid = False
        args.layers = None
        args.dry_run = False
        args.results_only = False
        args.rollout = 1
        args.api_circuit_file = None
        args.api_circuit_threshold = 3
        args.api_circuit_cooldown = 120.0
        args.api_case_retries = 3
        args.api_case_retry_wait = 30.0

        split = getattr(args, "split", "")
        if evolve:
            timeout = float(adapter.default_evolve_timeout)
        else:
            timeout = float(adapter.default_timeout)
        if hasattr(adapter, "PER_SPLIT_TIMEOUT"):
            timeout = float(adapter.PER_SPLIT_TIMEOUT.get(split, timeout))
        args.timeout = timeout
        if hasattr(adapter, "PER_SPLIT_MAX_COST"):
            max_cost = adapter.PER_SPLIT_MAX_COST.get(split, adapter.default_max_cost)
        else:
            max_cost = adapter.default_max_cost
        args.max_cost = float(max_cost)
        return args

    def _completed_mix_ids(self) -> set[str]:
        completed = set()
        for path in self.run_mgr.list_cases():
            result_path = path / "result.json"
            if not result_path.exists():
                continue
            try:
                row = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                row.get("mixed_task_id")
                and not row.get("infrastructure_failure")
                and row.get("official_score_eligible", True)
            ):
                completed.add(str(row["mixed_task_id"]))
        return completed

    def _runtime_task_id(self, task: MixedTask) -> str:
        global_index = self._global_indices[task.mixed_task_id]
        return f"case-{task.phase}-{global_index:03d}"

    async def _run_task(self, task: MixedTask, *, evolve: bool) -> dict[str, Any]:
        resolved = self.wrapper.resolve(task)
        adapter = resolved.adapter
        global_index = self._global_indices[task.mixed_task_id]
        item = self.wrapper.runtime_item(
            task, global_index, resolved=resolved,
        )
        args = self._native_args(adapter, task, evolve)
        runtime_task_id = self._runtime_task_id(task)
        record = await adapter.run_single_case(
            item,
            global_index,
            args,
            self.run_mgr,
            team_name=self.team_name,
            evolve=evolve,
            timeout_secs=args.timeout,
            result_manifest=None,
            run_id=self.run_id,
            task_id_override=runtime_task_id,
            native_allowed_tools=set(
                self.manifest.benchmark_bindings[task.source.benchmark][
                    "native_allowed_tools"
                ]
            ),
            allow_pool_answer_protocol_override=False,
            allow_adapter_collaboration_priors=(
                self.orchestration_mode != "evoteam"
            ),
            selection_task_builder=(
                lambda current_item, full_task: self.wrapper.selector_task(
                    task,
                    adapter,
                    current_item,
                    full_task,
                    neutral=self.orchestration_mode == "evoteam",
                )
            ),
        )
        native_record = dict(record)
        mix_record = self.wrapper.wrap_native_record(
            task, native_record, runtime_task_id=runtime_task_id,
        )
        case_dir = self.run_mgr.create_case_dir(global_index, runtime_task_id)
        self.run_mgr.save_case_result(case_dir, mix_record)
        mix_record["_result_path"] = str(case_dir / "result.json")
        self.ledger.record_attempt(
            mix_record,
            run_id=self.run_id,
            original_index=global_index,
        )
        return mix_record

    async def run_phase(
        self, phase: str, max_tasks: int | None = None,
    ) -> list[dict[str, Any]]:
        tasks = self.manifest.tasks_for_phase(phase)
        if max_tasks is not None:
            tasks = tasks[:max_tasks]
        completed = self._completed_mix_ids() if self.resume else set()
        records: list[dict[str, Any]] = []
        for task in tasks:
            if task.mixed_task_id in completed:
                print(f"[MIXED-BENCHMARK] resume: skip {task.mixed_task_id}")
                continue
            print(
                f"[MIXED-BENCHMARK] {phase} {task.order}: {task.mixed_task_id} "
                f"({task.source.benchmark}/{task.source.split})"
            )
            record = await self._run_task(task, evolve=phase == "evolve")
            records.append(record)
            if phase == "evolve" and (
                record.get("infrastructure_failure")
                or not record.get("official_score_eligible", True)
                or not record.get("evolution_eligible", True)
            ):
                raise RuntimeError(
                    "stopping sequential evolution after an ineligible result: "
                    f"{task.mixed_task_id}"
                )
        return records

    def summary(self) -> dict[str, Any]:
        rows = []
        for path in self.run_mgr.list_cases():
            result_path = path / "result.json"
            if not result_path.exists():
                continue
            try:
                row = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if row.get("mixed_task_id"):
                rows.append(row)
        expected_test_tasks: dict[str, int] = defaultdict(int)
        expected_test_task_ids: dict[str, list[str]] = defaultdict(list)
        for task in self.manifest.tasks_for_phase("test", enabled_only=False):
            if task.enabled:
                expected_test_tasks[task.source.benchmark] += 1
                expected_test_task_ids[task.source.benchmark].append(task.mixed_task_id)
        by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_benchmark[str(row["source_benchmark"])].append(row)
        benchmark_summary = {}
        for benchmark, benchmark_rows in sorted(by_benchmark.items()):
            eligible = [
                row for row in benchmark_rows
                if not row.get("infrastructure_failure")
                and row.get("official_score_eligible", True)
            ]
            benchmark_summary[benchmark] = {
                "assigned": len(benchmark_rows),
                "expected_test_tasks": expected_test_tasks.get(benchmark, 0),
                "eligible": len(eligible),
                "infrastructure_failures": sum(
                    bool(row.get("infrastructure_failure"))
                    for row in benchmark_rows
                ),
                "non_official_runs": sum(
                    not row.get("official_score_eligible", True)
                    for row in benchmark_rows
                ),
                "native_avg_score": round(
                    sum(row.get("native_score", 0.0) for row in eligible) / len(eligible), 4
                ) if eligible else None,
                "native_score_sum": round(
                    sum(row.get("native_score", 0.0) for row in eligible), 6
                ),
                "correct_count": sum(bool(row.get("is_correct")) for row in eligible),
                "success_rate": round(
                    sum(bool(row.get("is_correct")) for row in eligible) / len(eligible), 4
                ) if eligible else None,
            }
        rates = [row["success_rate"] for row in benchmark_summary.values() if row["success_rate"] is not None]
        return {
            "suite_id": self.manifest.suite_id,
            "run_id": self.run_mgr.run_id,
            "team": self.team_name,
            "records": len(rows),
            "team_versions": self.run_mgr.list_team_versions(),
            "benchmark_summary": benchmark_summary,
            "expected_test_tasks_by_benchmark": dict(expected_test_tasks),
            "expected_test_task_ids_by_benchmark": {
                benchmark: task_ids
                for benchmark, task_ids in expected_test_task_ids.items()
            },
            "test_task_results": [
                {
                    "mixed_task_id": row["mixed_task_id"],
                    "source_benchmark": row["source_benchmark"],
                    "native_score": float(row.get("native_score", 0.0) or 0.0),
                    "is_correct": bool(row.get("is_correct", False)),
                    "infrastructure_failure": bool(row.get("infrastructure_failure", False)),
                    "official_score_eligible": bool(row.get("official_score_eligible", True)),
                }
                for row in rows if row.get("phase") == "test"
            ],
            "macro_success_rate": round(sum(rates) / len(rates), 4) if rates else None,
            "result_manifest": str(self.ledger.path),
        }

    async def run(
        self, phase: str, max_tasks: int | None = None,
    ) -> dict[str, Any]:
        if phase == "all":
            await self.run_phase("evolve", max_tasks=max_tasks)
            await self.run_phase("test", max_tasks=max_tasks)
        else:
            await self.run_phase(phase, max_tasks=max_tasks)
        summary = self.summary()
        self.run_mgr.save_summary(summary)
        return summary


def aggregate_rollout_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {"rollouts": 0, "avg_at": 0, "benchmark_summary": {}}
    expected: dict[str, int] = {}
    expected_ids: dict[str, set[str]] = {}
    task_results: list[dict[tuple[str, str], dict[str, Any]]] = []
    for summary in summaries:
        for benchmark, count in (summary.get("expected_test_tasks_by_benchmark") or {}).items():
            expected[str(benchmark)] = max(expected.get(str(benchmark), 0), int(count))
        for benchmark, ids in (summary.get("expected_test_task_ids_by_benchmark") or {}).items():
            benchmark = str(benchmark)
            ids = {str(task_id) for task_id in ids}
            if benchmark in expected_ids and expected_ids[benchmark] != ids:
                raise ValueError(f"rollouts disagree on expected test task IDs for {benchmark}")
            expected_ids[benchmark] = ids
        indexed = {}
        for row in summary.get("test_task_results", []):
            key = (str(row.get("source_benchmark", "")), str(row.get("mixed_task_id", "")))
            if key in indexed:
                raise ValueError(f"duplicate test result for {key} in run {summary.get('run_id')}")
            indexed[key] = row
        task_results.append(indexed)
    for benchmark, ids in expected_ids.items():
        if benchmark in expected and expected[benchmark] != len(ids):
            raise ValueError(
                f"expected task count and task IDs disagree for {benchmark}: "
                f"{expected[benchmark]} vs {len(ids)}"
            )
        expected[benchmark] = len(ids)
    benchmark_summary: dict[str, Any] = {}
    rates: list[float] = []
    for benchmark, expected_count in sorted(expected.items()):
        candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        infra_count = 0
        valid_per_rollout = []
        per_rollout_score = []
        per_rollout_success = []
        expected_for_benchmark = expected_ids.get(benchmark)
        for indexed in task_results:
            count = 0
            rollout_scores = []
            rollout_correct = []
            candidate_ids = expected_for_benchmark
            if candidate_ids is None:
                candidate_ids = {
                    task_id for source_benchmark, task_id in indexed
                    if source_benchmark == benchmark
                }
            for task_id in candidate_ids:
                row = indexed.get((benchmark, task_id))
                if row is None:
                    continue
                if row.get("infrastructure_failure"):
                    infra_count += 1
                elif row.get("official_score_eligible", True):
                    candidates[task_id].append(row)
                    rollout_scores.append(float(row.get("native_score", 0.0)))
                    rollout_correct.append(bool(row.get("is_correct")))
                    count += 1
            valid_per_rollout.append(count)
            per_rollout_score.append(
                sum(rollout_scores) / len(rollout_scores) if rollout_scores else None
            )
            per_rollout_success.append(
                sum(rollout_correct) / len(rollout_correct) if rollout_correct else None
            )
        expected_for_benchmark = expected_ids.get(benchmark)
        task_ids = expected_for_benchmark if expected_for_benchmark is not None else set(candidates)
        complete = [
            candidates[task_id]
            for task_id in task_ids
            if len(candidates.get(task_id, [])) == len(summaries)
        ]
        task_avg_scores = [
            sum(float(row.get("native_score", 0.0)) for row in rows) / len(rows)
            for rows in complete
        ]
        task_avg_correct = [
            sum(bool(row.get("is_correct")) for row in rows) / len(rows)
            for rows in complete
        ]
        coverage_denominator = expected_count * len(summaries)
        eligible = sum(valid_per_rollout)
        benchmark_summary[benchmark] = {
            "expected_tasks_per_rollout": expected_count,
            "expected_task_ids": sorted(task_ids),
            "complete_avg_at_tasks": len(complete),
            "incomplete_tasks": max(0, expected_count - len(complete)),
            "infrastructure_failures": infra_count,
            "per_rollout_eligible_tasks": valid_per_rollout,
            "per_rollout_native_scores": per_rollout_score,
            "per_rollout_success_rates": per_rollout_success,
            "eligible": eligible,
            "rollout_coverage": round(eligible / coverage_denominator, 4) if coverage_denominator else 0.0,
            "native_avg_score": round(sum(task_avg_scores) / len(task_avg_scores), 4) if task_avg_scores else None,
            "success_rate": round(sum(task_avg_correct) / len(task_avg_correct), 4) if task_avg_correct else None,
        }
        if task_avg_correct:
            rates.append(sum(task_avg_correct) / len(task_avg_correct))
    return {
        "rollouts": len(summaries),
        "avg_at": len(summaries),
        "run_ids": [summary.get("run_id") for summary in summaries],
        "benchmark_summary": benchmark_summary,
        "macro_success_rate": round(sum(rates) / len(rates), 4) if rates else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Sequential MIXED-BENCHMARK scheduler.\n\nThe scheduler owns ordering, one shared RunManager/team lineage, and the\ncross-suite ledger. Native adapters still own environment setup, task\nconstruction, output contracts, artifact checks, and evaluation.\n')
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=["evolve", "test", "all"], default="evolve")
    parser.add_argument("--team", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-tasks", type=int, default=None,
        help="Run only the first N tasks in the selected phase.",
    )
    parser.add_argument(
        "--rollout", type=int, default=None,
        help="Independent test rollouts; defaults to 3 for --phase test and 1 otherwise.",
    )
    args = parser.parse_args()

    if args.max_tasks is not None and args.max_tasks < 1:
        parser.error("--max-tasks must be positive")
    rollout = args.rollout if args.rollout is not None else (3 if args.phase == "test" else 1)
    if rollout < 1:
        parser.error("--rollout must be positive")
    if rollout > 1 and args.phase != "test":
        parser.error("--rollout > 1 is supported only for --phase test")

    manifest = MixedBenchmarkManifest.load(args.manifest)
    if args.dry_run:
        for phase in ("evolve", "test") if args.phase == "all" else (args.phase,):
            tasks = manifest.tasks_for_phase(phase)
            if args.max_tasks is not None:
                tasks = tasks[:args.max_tasks]
            for task in tasks:
                print(f"{phase}\t{task.order}\t{task.mixed_task_id}\t{task.source.benchmark}")
        return 0
    if rollout == 1:
        scheduler = MixedScheduler(
            manifest,
            team_name=args.team,
            run_id=args.run_id,
            resume=args.resume,
            phase=args.phase,
        )
        summary = asyncio.run(scheduler.run(args.phase, max_tasks=args.max_tasks))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    base_run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_mixed_test")
    summaries = []
    for index in range(1, rollout + 1):
        run_id = f"{base_run_id}_r{index}"
        scheduler = MixedScheduler(
            manifest,
            team_name=args.team,
            run_id=run_id,
            resume=args.resume,
            phase=args.phase,
        )
        summaries.append(asyncio.run(scheduler.run(args.phase, max_tasks=args.max_tasks)))
    aggregate = aggregate_rollout_summaries(summaries)
    aggregate["suite_id"] = manifest.suite_id
    aggregate["team"] = args.team or manifest.shared_evolution["team_name"]
    aggregate["complete"] = all(
        data["complete_avg_at_tasks"] == data["expected_tasks_per_rollout"]
        and all(value == data["expected_tasks_per_rollout"] for value in data["per_rollout_eligible_tasks"])
        for data in aggregate["benchmark_summary"].values()
    )
    aggregate_path = PROJECT_ROOT / "runs" / "aggregates" / f"{base_run_id}_avg{rollout}.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    aggregate["aggregate_path"] = str(aggregate_path)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0 if aggregate["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
