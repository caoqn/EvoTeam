import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUITE_DIR = ROOT / "benchmarks" / "MIXED-BENCHMARK"
sys.path.insert(0, str(SUITE_DIR))

from mixed_benchmark_manifest import MixedBenchmarkManifest
from mixed_adapter import ADAPTER_REGISTRY
from mixed_scheduler import MixedScheduler
from task_wrapper import MixedTaskWrapper


def test_source_package_boundary():
    assert {path.name for path in (ROOT / "agents").iterdir()} == {"pool_EvoTeam"}
    for excluded in ("tools", "data", "runs", "logs", "results", "sessions", "apiconfig"):
        assert not (ROOT / excluded).exists()
    assert {path.name for path in (ROOT / "benchmarks").iterdir() if path.is_dir() and path.name != "__pycache__"} == {
        "MIXED-BENCHMARK", "mixed_native"
    }


def test_fixed_manifest_and_private_backends():
    manifest = MixedBenchmarkManifest.load(
        SUITE_DIR / "manifests" / "mixed_benchmark_official_116_seed20260828.json"
    )
    assert len(manifest.tasks_for_phase("evolve")) == 116
    assert manifest.tasks_for_phase("test") == []
    assert set(ADAPTER_REGISTRY) == {
        "gaia", "locobench", "locabench", "beyondswe"
    }
    for name, binding in manifest.benchmark_bindings.items():
        assert binding["adapter_module"] == f"benchmarks.mixed_native.{name}_backend"
        assert ADAPTER_REGISTRY[name].module == binding["adapter_module"]
    MixedTaskWrapper(manifest)
    with pytest.raises(FileNotFoundError, match="external assets missing"):
        MixedScheduler(manifest)

