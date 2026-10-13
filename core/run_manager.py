import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.scope_aware_experience_persistence import persist_reflection_to_source
from core.utils import now_iso as _now_iso

_BASE_DIR = Path(__file__).parent.parent
RUNS_DIR = _BASE_DIR / "runs"


class RunManager:

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.team_dir = run_dir / "team"
        self.cases_dir = run_dir / "cases"
        self.changelog_path = run_dir / "changelog.jsonl"

    @staticmethod
    def create_run(
        run_id: str | None = None,
        source_team_dir: str | Path | None = None,
        team_name: str = "pool_EvoTeam",
        config: dict | None = None,
        resume: bool = False,
    ) -> "RunManager":
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_dir = RUNS_DIR / run_id
        if run_dir.exists():
            if resume:
                print(f"[RunManager] resuming existing run: {run_dir}")
                return RunManager(run_dir)
            raise FileExistsError(f"Run directory already exists: {run_dir}")

        run_dir.mkdir(parents=True)
        (run_dir / "cases").mkdir()
        (run_dir / "team").mkdir()

        if source_team_dir is None:
            source_team_dir = _BASE_DIR / "agents" / team_name
        source_team_dir = Path(source_team_dir)
        if not source_team_dir.exists():
            raise FileNotFoundError(f"Source team directory not found: {source_team_dir}")

        v000 = run_dir / "team" / "v000"
        shutil.copytree(
            str(source_team_dir),
            str(v000),
            ignore=shutil.ignore_patterns("__pycache__"),
        )

        config_data = {
            "run_id": run_id,
            "team_name": team_name,
            "source_team_dir": str(source_team_dir),
            "created_at": _now_iso(),
            **(config or {}),
        }
        (run_dir / "config.json").write_text(
            json.dumps(config_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        mgr = RunManager(run_dir)
        print(f"[RunManager] created run: {run_dir}")
        print(f"[RunManager] v000 initialized from: {source_team_dir}")
        return mgr

    @staticmethod
    def load_run(run_id: str) -> "RunManager":
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_dir}")
        return RunManager(run_dir)

    @staticmethod
    def load_run_from_dir(run_dir: str | Path) -> "RunManager":
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return RunManager(run_dir)


    def list_team_versions(self) -> list[str]:
        if not self.team_dir.exists():
            return []
        versions = []
        for d in sorted(self.team_dir.iterdir()):
            if d.is_dir() and d.name.startswith("v") and not d.name.endswith("._reverted"):
                versions.append(d.name)
        return versions

    def get_latest_team_version(self) -> tuple[str, Path]:
        versions = self.list_team_versions()
        if not versions:
            raise FileNotFoundError(f"No team versions found in {self.team_dir}")
        latest = versions[-1]
        return latest, self.team_dir / latest

    def revert_team_version(self, version: str) -> None:
        src = self.team_dir / version
        dst = self.team_dir / f"{version}._reverted"
        if src.exists():
            src.rename(dst)
            print(f"[RunManager] reverted team version {version} → {dst.name}")

    def get_team_version_path(self, version: str) -> Path:
        p = self.team_dir / version
        if not p.exists():
            raise FileNotFoundError(f"Team version not found: {p}")
        return p

    def persist_team_version(
        self,
        session_team_dir: str | Path,
        include_teammate_profiles: bool = False,
        teammate_profile_agent_ids: set[str] | None = None,
        *,
        include_agent_reflection: bool = True,
        include_team_reflection: bool = False,
        include_handoff_reflection: bool = False,
    ) -> tuple[str, list[str]]:
        current_version, current_path = self.get_latest_team_version()

        current_num = int(current_version[1:])
        new_version = f"v{current_num + 1:03d}"
        new_path = self.team_dir / new_version

        shutil.copytree(
            str(current_path),
            str(new_path),
            ignore=shutil.ignore_patterns("__pycache__"),
        )

        modified = persist_reflection_to_source(
            session_team_dir=session_team_dir,
            source_team_dir=new_path,
            include_teammate_profiles=include_teammate_profiles,
            teammate_profile_agent_ids=teammate_profile_agent_ids,
            include_agent_reflection=include_agent_reflection,
            include_team_reflection=include_team_reflection,
            include_handoff_reflection=include_handoff_reflection,
        )

        if modified:
            print(f"[RunManager] created team version {new_version} "
                  f"({len(modified)} files changed)")
        else:
            shutil.rmtree(str(new_path))
            print(f"[RunManager] no changes, skipped version creation")
            return current_version, []

        return new_version, modified


    def create_case_dir(self, case_index: int, task_id: str) -> Path:
        safe_id = task_id.replace("/", "_").replace("\\", "_")
        case_name = f"{case_index:03d}_{safe_id}"
        case_dir = self.cases_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir

    def list_cases(self) -> list[Path]:
        if not self.cases_dir.exists():
            return []
        return sorted(d for d in self.cases_dir.iterdir() if d.is_dir())

    def save_case_result(self, case_dir: Path, result: dict) -> None:
        (case_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def load_case_results(self) -> list[dict]:
        results = []
        for case_dir in self.list_cases():
            result_path = case_dir / "result.json"
            if result_path.exists():
                results.append(
                    json.loads(result_path.read_text(encoding="utf-8"))
                )
        return results


    def write_changelog(
        self,
        case_index: int,
        task_id: str,
        from_version: str,
        to_version: str,
        modified_files: list[str],
        evolution_decisions: dict[str, bool] | None = None,
        evolution_updates: dict[str, bool] | None = None,
    ) -> None:
        entry = {
            "ts": _now_iso(),
            "case_index": case_index,
            "task_id": task_id,
            "from_version": from_version,
            "to_version": to_version,
            "modified_files": modified_files,
        }
        if evolution_updates is not None:
            entry["evolution_updates"] = evolution_updates
        if evolution_decisions is not None:
            entry["evolution_decisions"] = evolution_decisions
        with open(self.changelog_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_changelog(self) -> list[dict]:
        if not self.changelog_path.exists():
            return []
        entries = []
        for line in self.changelog_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "skipping corrupt JSONL line in %s", self.changelog_path
                    )
        return entries


    def save_summary(self, summary: dict) -> None:
        summary["updated_at"] = _now_iso()
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def load_summary(self) -> dict | None:
        p = self.run_dir / "summary.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def load_config(self) -> dict:
        p = self.run_dir / "config.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))


    @property
    def run_id(self) -> str:
        return self.run_dir.name

    def __repr__(self) -> str:
        versions = self.list_team_versions()
        cases = self.list_cases()
        return (f"RunManager(run_id={self.run_id}, "
                f"versions={len(versions)}, cases={len(cases)})")