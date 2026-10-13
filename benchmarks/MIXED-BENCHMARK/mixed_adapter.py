from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class NativeBackendSpec:
    benchmark: str
    module: str
    class_name: str


NATIVE_BACKENDS: dict[str, NativeBackendSpec] = {
    "gaia": NativeBackendSpec(
        "gaia", "benchmarks.mixed_native.gaia_backend", "GAIAAdapter",
    ),
    "locobench": NativeBackendSpec(
        "locobench", "benchmarks.mixed_native.locobench_backend", "LoCoBenchAdapter",
    ),
    "locabench": NativeBackendSpec(
        "locabench", "benchmarks.mixed_native.locabench_backend", "LOCABenchAdapter",
    ),
    "beyondswe": NativeBackendSpec(
        "beyondswe", "benchmarks.mixed_native.beyondswe_backend", "BeyondSWEAdapter",
    ),
}


@lru_cache(maxsize=len(NATIVE_BACKENDS))
def _native_backend(benchmark: str) -> Any:
    try:
        spec = NATIVE_BACKENDS[benchmark]
    except KeyError as exc:
        raise ValueError(f"unsupported MIXED-BENCHMARK benchmark: {benchmark!r}") from exc
    module = importlib.import_module(spec.module)
    adapter_class = getattr(module, spec.class_name)
    adapter = adapter_class()
    if adapter.benchmark_name != benchmark:
        raise ValueError(
            f"native backend {spec.module}.{spec.class_name} declares "
            f"benchmark_name={adapter.benchmark_name!r}, expected {benchmark!r}"
        )
    return adapter


class MixedBenchmarkAdapter:

    def __init__(self, benchmark: str):
        if benchmark not in NATIVE_BACKENDS:
            raise ValueError(f"unsupported MIXED-BENCHMARK benchmark: {benchmark!r}")
        self.benchmark_name = benchmark

    @property
    def native_spec(self) -> NativeBackendSpec:
        return NATIVE_BACKENDS[self.benchmark_name]

    @property
    def _backend(self) -> Any:
        return _native_backend(self.benchmark_name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def __repr__(self) -> str:
        spec = self.native_spec
        return (
            f"MixedBenchmarkAdapter(benchmark={self.benchmark_name!r}, "
            f"native={spec.module}.{spec.class_name})"
        )


@lru_cache(maxsize=len(NATIVE_BACKENDS))
def get_mixed_adapter(benchmark: str) -> MixedBenchmarkAdapter:
    return MixedBenchmarkAdapter(benchmark)


@dataclass(frozen=True)
class AdapterRegistration:

    benchmark: str
    module: str
    class_name: str

    def factory(self) -> MixedBenchmarkAdapter:
        return get_mixed_adapter(self.benchmark)


ADAPTER_REGISTRY: dict[str, AdapterRegistration] = {
    benchmark: AdapterRegistration(
        benchmark=benchmark,
        module=spec.module,
        class_name=spec.class_name,
    )
    for benchmark, spec in NATIVE_BACKENDS.items()
}
