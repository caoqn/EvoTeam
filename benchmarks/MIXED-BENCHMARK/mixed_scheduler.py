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
    ):
        self.manifest = manifest
        self.wrapper = MixedTaskWrapper(manifest)
        required_paths = {"tool implementations": PROJECT_ROOT / "tools"}
        if "gaia" in manifest.active_benchmarks:
            required_paths["GAIA data"] = PROJECT_ROOT / "data" / "gaia"
        if "locobench" in manifest.active_benchmarks:
            required_paths["LoCoBench installation"] = PROJECT_ROOT / "benchmarks" / "LoCoBench"
        if "locabench" in manifest.active_benchmarks:
            required_paths["LOCA-Bench installation"] = PROJECT_ROOT / "benchmarks" / "LOCA-bench"
        missing = [name for name, path in required_paths.items() if not path.exists()]
        if "beyondswe" in manifest.active_benchmarks:
            configured = os.environ.get("BEYONDSWE_DATA_FILE", "").strip()
            candidates = [Path(configured).expanduser()] if configured else [
                PROJECT_ROOT / "benchmarks" / "BeyondSWE" / "beyondswe.jsonl",
                PROJECT_ROOT / "benchmarks" / "BeyondSWE" / "data" / "beyondswe.jsonl",
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
        if self.team_name != declared_team and pool_mode != "baseline_shared":
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
    args = parser.parse_args()

    if args.max_tasks is not None and args.max_tasks < 1:
        parser.error("--max-tasks must be positive")

    manifest = MixedBenchmarkManifest.load(args.manifest)
    if args.dry_run:
        for phase in ("evolve", "test") if args.phase == "all" else (args.phase,):
            tasks = manifest.tasks_for_phase(phase)
            if args.max_tasks is not None:
                tasks = tasks[:args.max_tasks]
            for task in tasks:
                print(f"{phase}\t{task.order}\t{task.mixed_task_id}\t{task.source.benchmark}")
        return 0
    scheduler = MixedScheduler(
        manifest,
        team_name=args.team,
        run_id=args.run_id,
        resume=args.resume,
    )
    summary = asyncio.run(scheduler.run(args.phase, max_tasks=args.max_tasks))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())