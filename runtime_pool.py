from __future__ import annotations

import contextlib
import io
import re
import shutil
from pathlib import Path
from typing import NamedTuple

import yaml

from core.agent import Agent
from core.tool_registry import ToolRegistry


ROOT = Path(__file__).resolve().parent


class PoolComponents(NamedTuple):
    registry: ToolRegistry
    agents: dict[str, Agent]
    chairman_name: str
    constitution: str
    settings: dict


def load_pool(pool_dir: Path) -> PoolComponents:
    registry = ToolRegistry(str(ROOT / "tools"))
    agents = {
        agent.config.name: agent
        for config_path in sorted(pool_dir.rglob("config.yaml"))
        if config_path.parent != pool_dir
        for agent in [Agent(str(config_path.parent), registry)]
    }
    config = yaml.safe_load((pool_dir / "pool.yaml").read_text(encoding="utf-8")) or {}
    chairman_name = str(config.get("chairman") or next(iter(agents), ""))
    constitution_path = pool_dir / "constitution.md"
    constitution = (
        constitution_path.read_text(encoding="utf-8").strip()
        if constitution_path.exists() else ""
    )
    return PoolComponents(
        registry,
        agents,
        chairman_name,
        constitution,
        dict(config.get("settings") or {}),
    )


def copy_team_to_session(session_dir: Path, source_team_dir: Path) -> Path:
    destination = session_dir / "team"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_team_dir, destination, ignore=shutil.ignore_patterns("__pycache__"))
    return destination


def _save_trace(session_id: str, session_dir: str) -> None:
    destination = Path(session_dir)
    from core.trace import render_overview, render_trace

    for name, renderer in (("trace.txt", render_trace), ("overview.txt", render_overview)):
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                if name == "trace.txt":
                    renderer(session_id, verbose=True, session_dir=session_dir)
                else:
                    renderer(session_id, session_dir=session_dir)
        except Exception as exc:
            print(f"[warn] failed to save {name}: {exc}")
            continue
        clean = re.sub(r"\033\[[0-9;]*m", "", buffer.getvalue())
        (destination / name).write_text(clean, encoding="utf-8")