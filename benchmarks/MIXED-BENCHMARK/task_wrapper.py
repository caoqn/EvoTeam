from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.output_contract import output_contract_from_name
from mixed_benchmark_manifest import MixedBenchmarkManifest, MixedTask
from mixed_adapter import ADAPTER_REGISTRY, MixedBenchmarkAdapter


@dataclass
class ResolvedNativeTask:
    specification: MixedTask
    adapter: MixedBenchmarkAdapter
    item: dict[str, Any]


@dataclass
class PreparedMixedTask:

    resolved: ResolvedNativeTask
    env_context: Any
    execution_task: str
    selector_task: str
    output_contract: Any
    execution_policy: Any

    @property
    def mixed_task_id(self) -> str:
        return self.resolved.specification.mixed_task_id


class MixedTaskWrapper:

    def __init__(self, manifest: MixedBenchmarkManifest):
        self.manifest = manifest
        self._validate_adapter_bindings()

    @classmethod
    def from_path(cls, path: str | Path) -> "MixedTaskWrapper":
        return cls(MixedBenchmarkManifest.load(path))

    def _validate_adapter_bindings(self) -> None:
        for benchmark, binding in self.manifest.benchmark_bindings.items():
            registration = ADAPTER_REGISTRY.get(benchmark)
            if registration is None:
                raise ValueError(f"no native adapter registration for {benchmark!r}")
            if binding["adapter_module"] != registration.module:
                raise ValueError(
                    f"{benchmark} adapter_module must be {registration.module!r}"
                )
            if binding["adapter_class"] != registration.class_name:
                raise ValueError(
                    f"{benchmark} adapter_class must be {registration.class_name!r}"
                )
            adapter = registration.factory()
            execution_policy = adapter.execution_policy({})
            if binding["execution_policy"] != execution_policy.name:
                raise ValueError(
                    f"{benchmark} execution policy mismatch: manifest declares "
                    f"{binding['execution_policy']!r}, but {registration.class_name} "
                    f"provides {execution_policy.name!r}"
                )
            if str(binding["execution_policy_version"]) != execution_policy.version:
                raise ValueError(
                    f"{benchmark} execution policy version mismatch: manifest "
                    f"declares {binding['execution_policy_version']!r}, but "
                    f"{registration.class_name} provides {execution_policy.version!r}"
                )
            declared_contract = output_contract_from_name(
                str(binding["output_contract"])
            )
            native_contract = adapter.output_contract()
            if declared_contract.name != native_contract.name:
                raise ValueError(
                    f"{benchmark} output contract mismatch: manifest declares "
                    f"{binding['output_contract']!r} (canonical "
                    f"{declared_contract.name!r}), but {registration.class_name} "
                    f"provides {native_contract.name!r}"
                )
            declared_tools = set(binding["native_allowed_tools"])
            policy_tools = set(execution_policy.allowed_tools)
            if declared_tools != policy_tools:
                raise ValueError(
                    f"{benchmark} native tool mismatch: manifest declares "
                    f"{sorted(declared_tools)}, but {registration.class_name} "
                    f"execution policy provides {sorted(policy_tools)}"
                )

    @staticmethod
    def _adapter_namespace(adapter: MixedBenchmarkAdapter, task: MixedTask):
        args = adapter.build_parser().parse_args([])
        args.split = task.source.split
        for key, value in task.source.adapter_args.items():
            if not hasattr(args, key):
                raise ValueError(
                    f"unknown {task.source.benchmark} adapter argument: {key!r}"
                )
            setattr(args, key, value)
        return args

    def resolve(self, task: MixedTask) -> ResolvedNativeTask:
        registration = ADAPTER_REGISTRY[task.source.benchmark]
        adapter = registration.factory()
        args = self._adapter_namespace(adapter, task)
        items = adapter.load_dataset(args)
        index = task.source.source_index
        if index >= len(items):
            raise IndexError(
                f"{task.mixed_task_id}: source_index {index} is outside "
                f"{task.source.benchmark}/{task.source.split} ({len(items)} items)"
            )
        item = {**items[index], "_source_index": index}
        actual_id = adapter.get_item_id(item)
        if actual_id != task.source.task_id:
            raise ValueError(
                f"{task.mixed_task_id}: manifest task_id {task.source.task_id!r} "
                f"does not match adapter item {actual_id!r} at index {index}; "
                "the native dataset or filtering arguments changed"
            )
        return ResolvedNativeTask(task, adapter, item)

    def runtime_item(
        self,
        task: MixedTask,
        global_index: int,
        *,
        resolved: ResolvedNativeTask | None = None,
    ) -> dict[str, Any]:
        resolved = resolved or self.resolve(task)
        item = dict(resolved.item)
        item["_native_source_index"] = item.get("_source_index")
        item["_source_index"] = global_index
        return item

    @staticmethod
    def selector_task(
        task: MixedTask,
        adapter: MixedBenchmarkAdapter,
        item: dict[str, Any],
        full_task: str,
        *,
        neutral: bool,
    ) -> str:
        del task
        if neutral:
            return adapter.build_neutral_team_selection_task(item, full_task)
        return adapter.build_team_selection_task(item, full_task)

    def wrap_native_record(
        self,
        task: MixedTask,
        native_record: dict[str, Any],
        *,
        runtime_task_id: str,
    ) -> dict[str, Any]:
        record = dict(native_record)
        return {
            "mixed_task_id": task.mixed_task_id,
            "phase": task.phase,
            "order": task.order,
            "source_benchmark": task.source.benchmark,
            "source_split": task.source.split,
            "source_task_id": task.source.task_id,
            "native_source_index": task.source.source_index,
            "task_id": task.mixed_task_id,
            "runtime_task_id": runtime_task_id,
            "native_score": float(record.get("score", 0.0) or 0.0),
            "is_correct": bool(record.get("is_correct", False)),
            "score": float(record.get("score", 0.0) or 0.0),
            "extracted_answer": record.get("extracted_answer", ""),
            "eval_summary": record.get("eval_summary", ""),
            "run_error": record.get("run_error", record.get("error", "")),
            "infrastructure_failure": bool(record.get("infrastructure_failure", False)),
            "official_score_eligible": bool(
                record.get("official_score_eligible", True)
            ),
            "evolution_eligible": bool(record.get("evolution_eligible", True)),
            "execution_policy": dict(record.get("execution_policy") or {}),
            "native_record": record,
        }

    async def prepare(self, task: MixedTask, session: Any) -> PreparedMixedTask:
        resolved = self.resolve(task)
        env_context = resolved.adapter.setup_environment(resolved.item, session)
        if inspect.isawaitable(env_context):
            env_context = await env_context
        execution_policy = resolved.adapter.bind_execution_policy(
            resolved.adapter.execution_policy(resolved.item),
            resolved.item,
            env_context,
            session,
        )
        workspace_files = [
            path.name for path in Path(session.workspace).iterdir()
            if not path.name.startswith(".")
        ]
        execution_task = resolved.adapter.build_task(
            resolved.item, session, workspace_files,
        )
        normalized = self.selector_task(
            task, resolved.adapter, resolved.item, execution_task, neutral=False,
        )
        selector_task = normalized
        return PreparedMixedTask(
            resolved=resolved,
            env_context=env_context,
            execution_task=execution_task,
            selector_task=selector_task,
            output_contract=resolved.adapter.output_contract(),
            execution_policy=execution_policy,
        )

    @staticmethod
    def evaluate(prepared: PreparedMixedTask, predicted: str, session: Any):
        return prepared.resolved.adapter.evaluate(
            prepared.resolved.item,
            predicted,
            session,
            **prepared.env_context.data,
        )

    @staticmethod
    def build_record(
        prepared: PreparedMixedTask,
        predicted: str,
        eval_result: Any,
        session: Any,
        error: str = "",
    ) -> dict[str, Any]:
        spec = prepared.resolved.specification
        native = prepared.resolved.adapter.build_record(
            prepared.resolved.item,
            spec.source.source_index,
            predicted,
            eval_result,
            session,
            error,
        )
        return {
            "mixed_task_id": spec.mixed_task_id,
            "phase": spec.phase,
            "order": spec.order,
            "source_benchmark": spec.source.benchmark,
            "source_split": spec.source.split,
            "source_task_id": spec.source.task_id,
            "native_score": float(eval_result.score),
            "is_correct": bool(eval_result.success),
            "infrastructure_failure": bool(native.get("infrastructure_failure", False)),
            "official_score_eligible": prepared.execution_policy.official_score_eligible,
            "evolution_eligible": prepared.execution_policy.evolution_eligible,
            "execution_policy": prepared.execution_policy.to_dict(),
            "native_record": native,
        }

    @staticmethod
    def teardown(prepared: PreparedMixedTask) -> None:
        prepared.resolved.adapter.teardown_environment(prepared.env_context)