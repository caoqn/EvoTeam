from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
SUPPORTED_BENCHMARKS = frozenset({
    "locobench",
    "locabench",
    "gaia",
    "beyondswe",
})
VALID_PHASES = frozenset({"evolve", "test"})


class ManifestValidationError(ValueError):
    pass


def _require_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{path} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class SourceReference:

    benchmark: str
    split: str
    task_id: str
    source_index: int
    adapter_args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], path: str) -> "SourceReference":
        if not isinstance(payload, dict):
            raise ManifestValidationError(f"{path} must be an object")
        benchmark = _require_text(payload.get("benchmark"), f"{path}.benchmark")
        if benchmark not in SUPPORTED_BENCHMARKS:
            raise ManifestValidationError(
                f"{path}.benchmark={benchmark!r} is unsupported; expected one of "
                f"{sorted(SUPPORTED_BENCHMARKS)}"
            )
        source_index = payload.get("source_index")
        if not isinstance(source_index, int) or isinstance(source_index, bool) or source_index < 0:
            raise ManifestValidationError(
                f"{path}.source_index must be a non-negative integer"
            )
        adapter_args = payload.get("adapter_args", {})
        if not isinstance(adapter_args, dict):
            raise ManifestValidationError(f"{path}.adapter_args must be an object")
        forbidden = {"team", "evolve", "run_id", "result_manifest"} & set(adapter_args)
        if forbidden:
            raise ManifestValidationError(
                f"{path}.adapter_args contains suite-owned option(s): {sorted(forbidden)}"
            )
        return cls(
            benchmark=benchmark,
            split=_require_text(payload.get("split"), f"{path}.split"),
            task_id=_require_text(payload.get("task_id"), f"{path}.task_id"),
            source_index=source_index,
            adapter_args=dict(adapter_args),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "split": self.split,
            "task_id": self.task_id,
            "source_index": self.source_index,
            "adapter_args": dict(self.adapter_args),
        }


@dataclass(frozen=True)
class HiddenAnnotations:

    collaboration_profile: tuple[str, ...] = ()
    difficulty: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: Any, path: str) -> "HiddenAnnotations":
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ManifestValidationError(f"{path} must be an object")
        labels = payload.get("collaboration_profile", [])
        if not isinstance(labels, list) or any(
            not isinstance(label, str) or not label.strip() for label in labels
        ):
            raise ManifestValidationError(
                f"{path}.collaboration_profile must be a list of non-empty strings"
            )
        if len(set(labels)) != len(labels):
            raise ManifestValidationError(
                f"{path}.collaboration_profile contains duplicate labels"
            )
        difficulty = payload.get("difficulty", "")
        notes = payload.get("notes", "")
        if not isinstance(difficulty, str) or not isinstance(notes, str):
            raise ManifestValidationError(
                f"{path}.difficulty and {path}.notes must be strings"
            )
        return cls(tuple(labels), difficulty.strip(), notes.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "collaboration_profile": list(self.collaboration_profile),
            "difficulty": self.difficulty,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class MixedTask:
    mixed_task_id: str
    phase: str
    order: int
    source: SourceReference
    hidden_annotations: HiddenAnnotations = field(default_factory=HiddenAnnotations)
    enabled: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any], path: str) -> "MixedTask":
        if not isinstance(payload, dict):
            raise ManifestValidationError(f"{path} must be an object")
        phase = _require_text(payload.get("phase"), f"{path}.phase")
        if phase not in VALID_PHASES:
            raise ManifestValidationError(
                f"{path}.phase must be one of {sorted(VALID_PHASES)}"
            )
        order = payload.get("order")
        if not isinstance(order, int) or isinstance(order, bool) or order < 0:
            raise ManifestValidationError(f"{path}.order must be a non-negative integer")
        enabled = payload.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ManifestValidationError(f"{path}.enabled must be boolean")
        return cls(
            mixed_task_id=_require_text(payload.get("mixed_task_id"), f"{path}.mixed_task_id"),
            phase=phase,
            order=order,
            source=SourceReference.from_dict(payload.get("source"), f"{path}.source"),
            hidden_annotations=HiddenAnnotations.from_dict(
                payload.get("hidden_annotations"), f"{path}.hidden_annotations"
            ),
            enabled=enabled,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mixed_task_id": self.mixed_task_id,
            "phase": self.phase,
            "order": self.order,
            "enabled": self.enabled,
            "source": self.source.to_dict(),
            "hidden_annotations": self.hidden_annotations.to_dict(),
        }


@dataclass
class MixedBenchmarkManifest:
    suite_id: str
    description: str
    active_benchmarks: frozenset[str]
    ordering: dict[str, Any]
    shared_evolution: dict[str, Any]
    isolation_policy: dict[str, Any]
    benchmark_bindings: dict[str, dict[str, Any]]
    tasks: list[MixedTask]
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def load(cls, path: str | Path) -> "MixedBenchmarkManifest":
        target = Path(path)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestValidationError(f"invalid JSON in {target}: {exc}") from exc
        manifest = cls.from_dict(payload)
        manifest.validate()
        return manifest

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MixedBenchmarkManifest":
        if not isinstance(payload, dict):
            raise ManifestValidationError("manifest root must be an object")
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ManifestValidationError(
                f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION}"
            )
        object_fields = [
            "ordering",
            "shared_evolution",
            "isolation_policy",
            "benchmark_bindings",
        ]
        for key in object_fields:
            if not isinstance(payload.get(key), dict):
                raise ManifestValidationError(f"{key} must be an object")
        rows = payload.get("tasks")
        if not isinstance(rows, list):
            raise ManifestValidationError("tasks must be a list")
        return cls(
            suite_id=_require_text(payload.get("suite_id"), "suite_id"),
            description=str(payload.get("description") or "").strip(),
            active_benchmarks=frozenset(
                payload.get("active_benchmarks")
                or payload.get("benchmark_bindings", {}).keys()
            ),
            ordering=dict(payload["ordering"]),
            shared_evolution=dict(payload["shared_evolution"]),
            isolation_policy=dict(payload["isolation_policy"]),
            benchmark_bindings={
                str(key): dict(value) if isinstance(value, dict) else value
                for key, value in payload["benchmark_bindings"].items()
            },
            tasks=[MixedTask.from_dict(row, f"tasks[{index}]") for index, row in enumerate(rows)],
        )

    def validate(self) -> None:
        strategy = self.ordering.get("strategy")
        if strategy not in {"stratified_round_robin", "global_shuffle"}:
            raise ManifestValidationError(
                "ordering.strategy must be 'stratified_round_robin' or 'global_shuffle'"
            )
        seed = self.ordering.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ManifestValidationError("ordering.seed must be an integer")
        if self.shared_evolution.get("mode") != "single_team_workspace":
            raise ManifestValidationError(
                "shared_evolution.mode must be 'single_team_workspace'"
            )
        if self.shared_evolution.get("require_common_agent_namespace") is not True:
            raise ManifestValidationError(
                "shared_evolution.require_common_agent_namespace must be true"
            )
        _require_text(self.shared_evolution.get("team_name"), "shared_evolution.team_name")
        _require_text(self.shared_evolution.get("state_root"), "shared_evolution.state_root")

        if not self.active_benchmarks or not self.active_benchmarks <= SUPPORTED_BENCHMARKS:
            raise ManifestValidationError(
                "active_benchmarks must be a non-empty subset of "
                f"{sorted(SUPPORTED_BENCHMARKS)}"
            )
        binding_keys = set(self.benchmark_bindings)
        if binding_keys != set(self.active_benchmarks):
            raise ManifestValidationError(
                "benchmark_bindings must match active_benchmarks exactly; got "
                f"bindings={sorted(binding_keys)}, active={sorted(self.active_benchmarks)}"
            )
        for benchmark, binding in self.benchmark_bindings.items():
            if not isinstance(binding, dict):
                raise ManifestValidationError(
                    f"benchmark_bindings.{benchmark} must be an object"
                )
            for key in (
                "adapter_module",
                "adapter_class",
                "task_construction",
                "native_environment",
                "execution_policy",
                "execution_policy_version",
                "output_contract",
                "submission_type",
                "native_evaluator",
            ):
                _require_text(binding.get(key), f"benchmark_bindings.{benchmark}.{key}")
            allowed_tools = binding.get("native_allowed_tools")
            if not isinstance(allowed_tools, list) or any(
                not isinstance(tool, str) or not tool.strip() for tool in allowed_tools
            ):
                raise ManifestValidationError(
                    f"benchmark_bindings.{benchmark}.native_allowed_tools must be a "
                    "list of non-empty strings"
                )

        seen_ids: set[str] = set()
        seen_sources: set[tuple[str, str, str]] = set()
        orders: dict[str, set[int]] = {phase: set() for phase in VALID_PHASES}
        for task in self.tasks:
            if task.source.benchmark not in self.active_benchmarks:
                raise ManifestValidationError(
                    f"{task.mixed_task_id} uses inactive benchmark "
                    f"{task.source.benchmark!r}"
                )
            if task.mixed_task_id in seen_ids:
                raise ManifestValidationError(f"duplicate mixed_task_id: {task.mixed_task_id}")
            seen_ids.add(task.mixed_task_id)
            source_key = (task.source.benchmark, task.source.split, task.source.task_id)
            if source_key in seen_sources:
                raise ManifestValidationError(
                    f"duplicate native task reference: {source_key}"
                )
            seen_sources.add(source_key)
            if task.order in orders[task.phase]:
                raise ManifestValidationError(
                    f"duplicate {task.phase} order: {task.order}"
                )
            orders[task.phase].add(task.order)

        if not any(orders.values()):
            raise ManifestValidationError("manifest must contain at least one task")
        for phase, phase_orders in orders.items():
            if not phase_orders:
                continue
            expected = set(range(len(phase_orders)))
            if phase_orders != expected:
                raise ManifestValidationError(
                    f"{phase} orders must be contiguous from 0; got {sorted(phase_orders)}"
                )

        if self.ordering.get("materialized_order") is True:
            declared_hash = self.ordering.get("order_hash")
            if not isinstance(declared_hash, str) or not declared_hash.strip():
                raise ManifestValidationError(
                    "ordering.order_hash is required for a materialized order"
                )
            material = "\n".join(
                f"{task.phase}\t{task.order}\t{task.mixed_task_id}"
                for task in sorted(
                    (task for task in self.tasks if task.enabled),
                    key=lambda task: (task.phase, task.order),
                )
            )
            actual_hash = hashlib.sha256(material.encode("utf-8")).hexdigest()
            if declared_hash != actual_hash:
                raise ManifestValidationError(
                    "ordering.order_hash does not match the materialized task order"
                )

        forbidden_visibility = {
            "source_benchmark",
            "benchmark",
            "adapter",
            "native_evaluator",
            "hidden_annotations",
            "collaboration_profile",
        }
        visible = set(self.isolation_policy.get("selector_visible_fields", []))
        leaked = visible & forbidden_visibility
        if leaked:
            raise ManifestValidationError(
                f"selector_visible_fields leaks routing/analysis metadata: {sorted(leaked)}"
            )

    def tasks_for_phase(self, phase: str, *, enabled_only: bool = True) -> list[MixedTask]:
        if phase not in VALID_PHASES:
            raise ValueError(f"unknown phase: {phase}")
        rows: Iterable[MixedTask] = self.tasks
        if enabled_only:
            rows = (task for task in rows if task.enabled)
        return sorted(
            (task for task in rows if task.phase == phase),
            key=lambda task: task.order,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "description": self.description,
            "active_benchmarks": sorted(self.active_benchmarks),
            "ordering": self.ordering,
            "shared_evolution": self.shared_evolution,
            "isolation_policy": self.isolation_policy,
            "benchmark_bindings": self.benchmark_bindings,
            "tasks": [task.to_dict() for task in self.tasks],
        }
