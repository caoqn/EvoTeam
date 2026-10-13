import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.utils import now_iso as _now_iso

_BASE_DIR = Path(__file__).parent.parent

SESSIONS_DIR = _BASE_DIR / "sessions"

TEMPLATES_DIR = _BASE_DIR / "templates"


def _generate_session_id() -> str:
    base = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    candidate = base
    suffix = 1
    while (SESSIONS_DIR / candidate).exists():
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


class EventLog:

    def __init__(self, path: Path):
        self.path = path
        self._file = None
        self._file = open(path, "a", encoding="utf-8")
        self._count = 0

    def log(self, event_type: str, agent: str | None = None, data: dict | None = None):
        if self._file is None or self._file.closed:
            return
        entry: dict[str, Any] = {
            "ts": _now_iso(),
            "seq": self._count,
            "type": event_type,
        }
        if agent:
            entry["agent"] = agent
        if data:
            entry["data"] = data
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()
        self._count += 1

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def count(self) -> int:
        return self._count


class Session:

    def __init__(self, session_id: str, session_dir: Path, task: str,
                 team_info: dict | None = None, template: str | None = None):
        self.id = session_id
        self.dir = session_dir
        self.task = task
        self.team_info = team_info or {}
        self.template = template
        self.created_at = _now_iso()

        workspace_dir = session_dir / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        self.workspace = str(workspace_dir)

        self.event_log = EventLog(session_dir / "events.jsonl")

        self._write_meta("running")

        self.event_log.log("session.start", data={
            "task": task,
            "workspace": self.workspace,
            "template": template,
            "team": team_info,
        })

    @staticmethod
    def create(task: str, team_info: dict | None = None,
               template: str | None = None,
               session_dir: str | Path | None = None) -> "Session":
        if session_dir is not None:
            session_dir = Path(session_dir)
            session_id = session_dir.name
            session_dir.mkdir(parents=True, exist_ok=True)
        else:
            session_id = _generate_session_id()
            session_dir = SESSIONS_DIR / session_id
            session_dir.mkdir(parents=True, exist_ok=True)

        workspace_dir = session_dir / "workspace"

        if template:
            template_dir = TEMPLATES_DIR / template
            if not template_dir.exists():
                raise FileNotFoundError(
                    f"Template '{template}' not found: {template_dir}"
                )
            shutil.copytree(
                template_dir,
                workspace_dir,
                ignore=shutil.ignore_patterns("template.yaml", "__pycache__"),
            )
        else:
            workspace_dir.mkdir(parents=True, exist_ok=True)

        return Session(session_id, session_dir, task, team_info, template)

    def close(self, status: str = "completed", summary: dict | None = None):
        if getattr(self, '_closed', False):
            return
        self._closed = True
        self.event_log.log("session.end", data={
            "status": status,
            **(summary or {}),
        })
        self.event_log.close()

        self._write_meta(status, summary)

    def _write_meta(self, status: str, summary: dict | None = None):
        meta = {
            "id": self.id,
            "status": status,
            "task": self.task,
            "template": self.template,
            "workspace": self.workspace,
            "created_at": self.created_at,
            "updated_at": _now_iso(),
            "team": self.team_info,
            "summary": summary,
        }
        (self.dir / "session.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def list_sessions() -> list[dict]:
        if not SESSIONS_DIR.exists():
            return []
        result = []
        for d in sorted(SESSIONS_DIR.iterdir(), reverse=True):
            meta_path = d / "session.json"
            if meta_path.exists():
                result.append(json.loads(meta_path.read_text(encoding="utf-8")))
        return result

    @staticmethod
    def get_latest_session_id() -> str | None:
        if not SESSIONS_DIR.exists():
            return None
        dirs = sorted(
            [d for d in SESSIONS_DIR.iterdir() if d.is_dir() and (d / "session.json").exists()],
            reverse=True,
        )
        return dirs[0].name if dirs else None

    @staticmethod
    def load_events(session_id: str) -> list[dict]:
        events_path = SESSIONS_DIR / session_id / "events.jsonl"
        if not events_path.exists():
            return []
        events = []
        for line in events_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "skipping corrupt JSONL line in %s", events_path
                    )
        return events
