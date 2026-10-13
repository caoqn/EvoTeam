from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any


PHASES = ("evolve", "test")


def _order_hash(tasks: list[dict[str, Any]]) -> str:
    material = "\n".join(
        f"{row['phase']}\t{row['order']}\t{row['mixed_task_id']}"
        for row in tasks
        if row.get("enabled", True)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _stratified_order(
    rows: list[dict[str, Any]], *, seed: int, phase: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["source"]["benchmark"], []).append(row)
    if not groups:
        return []

    queues: dict[str, list[dict[str, Any]]] = {}
    for benchmark in sorted(groups):
        queue = [copy.deepcopy(row) for row in groups[benchmark]]
        random.Random(f"{seed}:{phase}:{benchmark}").shuffle(queue)
        queues[benchmark] = queue

    round_rng = random.Random(f"{seed}:{phase}:rounds")
    result: list[dict[str, Any]] = []
    while any(queues.values()):
        active = [benchmark for benchmark, queue in queues.items() if queue]
        round_rng.shuffle(active)
        for benchmark in active:
            result.append(queues[benchmark].pop(0))
    return result


def _global_shuffle_order(
    rows: list[dict[str, Any]], *, seed: int, phase: str,
) -> list[dict[str, Any]]:
    result = [copy.deepcopy(row) for row in rows]
    random.Random(f"{seed}:{phase}:global").shuffle(result)
    return result


def materialize_payload(
    payload: dict[str, Any], *, seed: int,
    shuffle_phases: tuple[str, ...] = ("evolve",),
    strategy: str = "stratified_round_robin",
) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    rows = result.get("tasks", [])
    if not isinstance(rows, list):
        raise ValueError("manifest.tasks must be a list")

    reordered: list[dict[str, Any]] = []
    for phase in PHASES:
        phase_rows = [row for row in rows if row.get("phase") == phase]
        if phase in shuffle_phases:
            if strategy == "global_shuffle":
                phase_rows = _global_shuffle_order(phase_rows, seed=seed, phase=phase)
            elif strategy == "stratified_round_robin":
                phase_rows = _stratified_order(phase_rows, seed=seed, phase=phase)
            else:
                raise ValueError(f"unsupported ordering strategy: {strategy}")
        else:
            phase_rows = sorted(phase_rows, key=lambda row: row["order"])
        for order, row in enumerate(phase_rows):
            row["order"] = order
            reordered.append(row)
    result["tasks"] = reordered

    ordering = dict(result.get("ordering") or {})
    ordering.update({
        "strategy": strategy,
        "seed": seed,
        "shuffle_within_round": True,
        "shuffle_scope": ",".join(shuffle_phases) if shuffle_phases else "none",
        "materialized_order": True,
    })
    result["ordering"] = ordering
    result["suite_id"] = f"{result['suite_id']}-seed{seed}"
    result["description"] = (
        f"{result.get('description', '').rstrip('.')} "
        f"Materialized {strategy} order with seed {seed}."
    ).strip()
    result["ordering"]["order_hash"] = _order_hash(reordered)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description='Materialize a reproducible stratified-random MIXED-BENCHMARK task order.\n\nTasks are shuffled within each benchmark, then interleaved in rounds. Each\nround contains at most one task from every benchmark that still has work. The\nresult is written as a normal manifest so schedulers and baselines consume the\nsame explicit sequence.\n')
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--strategy",
        choices=("stratified_round_robin", "global_shuffle"),
        default="stratified_round_robin",
    )
    parser.add_argument(
        "--shuffle-phase", action="append", choices=PHASES,
        default=None,
        help="Phase(s) to stratify-shuffle; default is evolve only.",
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = materialize_payload(
        payload, seed=args.seed,
        shuffle_phases=tuple(dict.fromkeys(args.shuffle_phase or ("evolve",))),
        strategy=args.strategy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(f"seed={args.seed} order_hash={result['ordering']['order_hash']}")
    for phase in PHASES:
        order = [
            row["source"]["benchmark"]
            for row in result["tasks"]
            if row["phase"] == phase and row.get("enabled", True)
        ]
        print(f"{phase}: " + " ".join(order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
