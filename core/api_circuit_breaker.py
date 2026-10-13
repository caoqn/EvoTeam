from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


class APICircuitOpenError(RuntimeError):
    pass


@dataclass(frozen=True)
class CircuitState:
    consecutive_failures: int = 0
    open_until: float = 0.0
    last_error: str = ""

    @property
    def is_open(self) -> bool:
        return self.open_until > time.time()


class SharedAPICircuitBreaker:

    def __init__(self, path: str | Path, threshold: int = 3, cooldown_seconds: float = 120.0):
        self.path = Path(path)
        self.threshold = max(1, int(threshold))
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def check(self) -> None:
        state = self._mutate(lambda current: current)
        remaining = state.open_until - time.time()
        if remaining > 0:
            raise APICircuitOpenError(
                f"API circuit open for another {remaining:.0f}s after "
                f"{state.consecutive_failures} consecutive infrastructure failures. "
                f"Last error: {state.last_error}"
            )

    def record_success(self) -> None:
        self._mutate(lambda _current: CircuitState())

    def record_infrastructure_failure(self, error: BaseException) -> CircuitState:
        message = str(error).replace("\n", " ")[:500]

        def update(current: CircuitState) -> CircuitState:
            count = current.consecutive_failures + 1
            open_until = current.open_until
            if count >= self.threshold:
                open_until = time.time() + self.cooldown_seconds
            return CircuitState(count, open_until, message)

        return self._mutate(update)

    def snapshot(self) -> CircuitState:
        return self._mutate(lambda current: current)

    def _mutate(self, mutator) -> CircuitState:
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                try:
                    raw = json.load(handle)
                except (json.JSONDecodeError, ValueError):
                    raw = {}
                current = CircuitState(
                    consecutive_failures=int(raw.get("consecutive_failures", 0)),
                    open_until=float(raw.get("open_until", 0.0)),
                    last_error=str(raw.get("last_error", "")),
                )
                if current.open_until and current.open_until <= time.time():
                    current = CircuitState()
                updated = mutator(current)
                handle.seek(0)
                handle.truncate()
                json.dump({
                    "consecutive_failures": updated.consecutive_failures,
                    "open_until": updated.open_until,
                    "last_error": updated.last_error,
                    "updated_at": time.time(),
                    "pid": os.getpid(),
                }, handle)
                handle.flush()
                os.fsync(handle.fileno())
                return updated
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def get_shared_circuit_breaker() -> SharedAPICircuitBreaker | None:
    def setting(name: str, default: str) -> str:
        return os.environ.get(name, default)

    if setting("EVOTEAM_API_CIRCUIT_ENABLED", "1").strip().lower() in {"0", "false", "no"}:
        return None
    base_dir = Path(__file__).resolve().parent.parent
    path = Path(setting(
        "EVOTEAM_API_CIRCUIT_FILE",
        str(base_dir / "runs" / ".api_circuit.json"),
    ))
    return SharedAPICircuitBreaker(
        path,
        threshold=int(setting("EVOTEAM_API_CIRCUIT_THRESHOLD", "3")),
        cooldown_seconds=float(setting("EVOTEAM_API_CIRCUIT_COOLDOWN", "120")),
    )