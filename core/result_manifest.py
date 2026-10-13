from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResultManifest:

    def __init__(self, path: str | Path, benchmark: str, split: str):
        self.path = Path(path)
        self.benchmark = benchmark
        self.split = split
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_attempt(
        self,
        record: dict[str, Any],
        *,
        run_id: str,
        original_index: int,
        force_replace_valid: bool = False,
    ) -> dict[str, Any]:
        task_id = str(record.get("task_id") or f"index:{original_index}")
        attempt = {
            "attempted_at": _now(),
            "run_id": run_id,
            "original_index": original_index,
            "task_id": task_id,
            "is_correct": bool(record.get("is_correct", False)),
            "score": float(record.get("score", 0.0) or 0.0),
            "infrastructure_failure": bool(record.get("infrastructure_failure", False)),
            "run_error": str(record.get("run_error", ""))[:1000],
            "extracted_answer": str(record.get("extracted_answer", ""))[:500],
            "eval_summary": str(record.get("eval_summary", ""))[:1000],
            "result_path": str(record.get("_result_path", "")),
        }
        answer = attempt["extracted_answer"].strip().lower()
        attempt["retry_pending"] = bool(
            attempt["infrastructure_failure"]
            or "timeout" in attempt["run_error"].lower()
            or answer.startswith("[no output")
        )

        def update(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            entries = data.setdefault("entries", {})
            key = str(original_index)
            entry = entries.setdefault("%s" % key, {
                "original_index": original_index,
                "task_id": task_id,
                "attempts": [],
                "effective": None,
            })
            entry["task_id"] = task_id
            entry["attempts"].append(attempt)
            current = entry.get("effective")
            should_replace = (
                current is None
                or (current.get("retry_pending") and not attempt["retry_pending"])
                or force_replace_valid
            )
            if should_replace:
                entry["effective"] = attempt
            return data, entry

        return self._locked_update(update)

    def effective_records(self) -> list[dict[str, Any]]:
        data = self._read()
        entries = data.get("entries", {})
        return [
            entry["effective"] | {"original_index": entry["original_index"], "task_id": entry["task_id"]}
            for _, entry in sorted(entries.items(), key=lambda pair: int(pair[0]))
            if entry.get("effective") is not None
        ]

    def summary(self) -> dict[str, Any]:
        records = self.effective_records()
        infra = [record for record in records if record.get("infrastructure_failure")]
        scored = [record for record in records if not record.get("infrastructure_failure")]
        correct = sum(bool(record.get("is_correct")) for record in scored)
        return {
            "benchmark": self.benchmark,
            "split": self.split,
            "effective_records": len(records),
            "scored_records": len(scored),
            "correct": correct,
            "accuracy": (correct / len(scored)) if scored else 0.0,
            "infrastructure_pending": len(infra),
            "infrastructure_pending_indices": [r["original_index"] for r in infra],
        }

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "benchmark": self.benchmark, "split": self.split, "entries": {}}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"schema_version": 1, "benchmark": self.benchmark, "split": self.split, "entries": {}}

    def _locked_update(self, update):
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                try:
                    data = json.load(handle)
                except (json.JSONDecodeError, ValueError):
                    data = {"schema_version": 1, "benchmark": self.benchmark, "split": self.split, "entries": {}}
                data.setdefault("schema_version", 1)
                data["benchmark"] = self.benchmark
                data["split"] = self.split
                data, result = update(data)
                data["updated_at"] = _now()
                data["updated_by_pid"] = os.getpid()
                handle.seek(0)
                handle.truncate()
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                return result
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
