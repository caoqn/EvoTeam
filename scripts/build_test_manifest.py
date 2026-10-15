#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE_DIR = ROOT / "benchmarks" / "MIXED-BENCHMARK"
BASE_MANIFEST = SUITE_DIR / "manifests" / "mixed_benchmark_official_116_seed20260828.json"
LOCA_TASKS = (
    "WoocommerceNewWelcome", "WoocommerceStockAlert", "FilterLowSelling",
    "ApplyPhDEmail", "SetConfCrDdl", "CourseAssistant",
    "CanvasArrangeExam", "CanvasListTest",
)
LOCA_SPLITS = ("8k", "16k", "32k", "64k", "128k", "256k")
LOCA_SEEDS = (42, 123, 456, 789, 2024)


def _load_adapters(reference_root: Path):
    sys.path.insert(0, str(reference_root.resolve()))
    adapters = {}
    for name, module_name, class_name in (
        ("gaia", "benchmarks.adapter_gaia", "GAIAAdapter"),
        ("locobench", "benchmarks.adapter_locobench", "LoCoBenchAdapter"),
        ("locabench", "benchmarks.adapter_locabench", "LOCABenchAdapter"),
        ("beyondswe", "benchmarks.adapter_beyondswe", "BeyondSWEAdapter"),
    ):
        adapters[name] = getattr(importlib.import_module(module_name), class_name)()
    return adapters


def _items(adapter, split: str, adapter_args: dict):
    args = adapter.build_parser().parse_args([])
    args.split = split
    for key, value in adapter_args.items():
        if not hasattr(args, key):
            raise ValueError(f"{adapter.benchmark_name}: unsupported adapter argument {key!r}")
        setattr(args, key, value)
    return adapter.load_dataset(args)


def _row(template: dict, adapter, benchmark: str, split: str,
         adapter_args: dict, source_index: int, serial: int, item: dict) -> dict:
    row = copy.deepcopy(template)
    row["mixed_task_id"] = f"mixed-test-{serial:03d}"
    row["phase"] = "test"
    row["order"] = serial
    row["source"] = {
        "benchmark": benchmark,
        "split": split,
        "task_id": adapter.get_item_id(item),
        "source_index": source_index,
        "adapter_args": dict(adapter_args),
    }
    row["hidden_annotations"] = {
        "collaboration_profile": [],
        "difficulty": "",
        "notes": "Paper holdout reference materialized from native adapter split.",
    }
    return row


def _order_hash(rows: list[dict]) -> str:
    material = "\n".join(
        f"{row['phase']}\t{row['order']}\t{row['mixed_task_id']}"
        for row in rows if row.get("enabled", True)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build(reference_root: Path, seed: int) -> dict:
    payload = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    evolve = [copy.deepcopy(row) for row in payload["tasks"] if row["phase"] == "evolve"]
    templates = {}
    for row in evolve:
        templates.setdefault(row["source"]["benchmark"], row)
    adapters = _load_adapters(reference_root)
    tests: list[dict] = []

    def add_range(benchmark: str, split: str, args: dict, indices):
        adapter = adapters[benchmark]
        items = _items(adapter, split, args)
        for index in indices:
            if index >= len(items):
                raise RuntimeError(f"{benchmark}/{split}: index {index} unavailable (n={len(items)})")
            tests.append(_row(templates[benchmark], adapter, benchmark, split, args, index, len(tests), items[index]))

    add_range("gaia", "test_100", {}, range(100))
    add_range("locobench", "python", {"category": "feature_implementation"}, range(20, 100))
    add_range("locobench", "python", {"category": "cross_file_refactoring"}, range(20, 100))

    adapter = adapters["locabench"]
    for split in LOCA_SPLITS:
        for task in LOCA_TASKS:
            for seed_value in LOCA_SEEDS:
                args = {"task": task, "seed": seed_value}
                items = _items(adapter, split, args)
                if len(items) != 1:
                    raise RuntimeError(f"locabench/{split}/{task}/{seed_value}: expected 1 item, got {len(items)}")
                tests.append(_row(templates["locabench"], adapter, "locabench", split, args, 0, len(tests), items[0]))

    add_range("beyondswe", "crossrepo", {}, range(20, 200))
    add_range("beyondswe", "depmigrate", {}, range(20, 178))
    if len(tests) != 838:
        raise RuntimeError(f"built {len(tests)} test rows; expected 838")

    random.Random(f"{seed}:test:global").shuffle(tests)
    for order, row in enumerate(tests):
        row["order"] = order
    payload["suite_id"] = f"mixed-benchmark-official-116-plus-838-seed{seed}"
    payload["description"] = "Official 116-case evolution stream plus the complete 838-case paper holdout."
    payload["tasks"] = evolve + tests
    payload["ordering"] = dict(payload["ordering"])
    payload["ordering"].update({
        "seed": seed, "shuffle_scope": "test", "materialized_order": True,
        "order_hash": _order_hash(payload["tasks"]),
    })
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
        help="Path to a checkout that provides the native benchmark adapters and data.",
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, default=SUITE_DIR / "manifests" / "mixed_benchmark_official_116_plus_838_test_seed20260828.json")
    args = parser.parse_args()
    result = build(args.reference_root, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(f"evolve=116 test=838 order_hash={result['ordering']['order_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
