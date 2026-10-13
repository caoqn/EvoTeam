import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_AGENT_REFLECTION_FILES = [
    "evolution/prompt_patches.md",
    "evolution/teammate_profiles.yaml",
]

_AGENT_REFLECTION_TREE_DIRS = [
    "skills",
]

_TEAM_REFLECTION_FILES = [
    "templates.json",
]

_HANDOFF_REFLECTION_FILES = [
    "handoff_rules.json",
]

def _tree_dirs_differ(session_dir: Path, source_dir: Path) -> bool:
    if not source_dir.exists():
        return True

    session_files: dict[str, str] = {}
    for f in sorted(session_dir.rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(session_dir))
            try:
                session_files[rel] = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                session_files[rel] = None

    source_files: dict[str, str] = {}
    for f in sorted(source_dir.rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(source_dir))
            try:
                source_files[rel] = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                source_files[rel] = None

    if set(session_files.keys()) != set(source_files.keys()):
        return True

    for rel, content in session_files.items():
        if content != source_files.get(rel):
            return True

    return False


def _find_all_agent_dirs(team_dir: Path) -> list[Path]:
    agents = []
    for config_path in sorted(team_dir.rglob("config.yaml")):
        agent_dir = config_path.parent
        if agent_dir == team_dir:
            continue
        agents.append(agent_dir)
    return agents


def persist_reflection_to_source(
    session_team_dir: str | Path,
    source_team_dir: str | Path,
    include_teammate_profiles: bool = False,
    teammate_profile_agent_ids: set[str] | None = None,
    *,
    include_agent_reflection: bool = True,
    include_team_reflection: bool = False,
    include_handoff_reflection: bool = False,
) -> list[str]:

    session_path = Path(session_team_dir)
    source_path = Path(source_team_dir)
    modified = []

    session_agents = _find_all_agent_dirs(session_path)
    source_agents = _find_all_agent_dirs(source_path)

    source_map = {}
    for sa in source_agents:
        rel = sa.relative_to(source_path)
        source_map[str(rel)] = sa

    if include_agent_reflection:
        for session_agent_dir in session_agents:
            rel = session_agent_dir.relative_to(session_path)
            source_agent_dir = source_map.get(str(rel))
            if not source_agent_dir:
                continue

            reflection_files = _AGENT_REFLECTION_FILES
            if not include_teammate_profiles:
                reflection_files = [
                    rel_file for rel_file in reflection_files
                    if rel_file != "evolution/teammate_profiles.yaml"
                ]
            for rel_file in reflection_files:
                if (
                    rel_file == "evolution/teammate_profiles.yaml"
                    and teammate_profile_agent_ids is not None
                    and rel.name not in teammate_profile_agent_ids
                ):
                    stale_profile = source_agent_dir / rel_file
                    if stale_profile.exists():
                        stale_profile.unlink()
                        modified.append(str(Path(rel) / rel_file) + " (cleared)")
                    continue
                session_file = session_agent_dir / rel_file
                dest = source_agent_dir / rel_file
                if session_file.exists():
                    content = session_file.read_text(encoding="utf-8")
                    if content.strip():
                        existing_content = ""
                        if dest.exists():
                            existing_content = dest.read_text(encoding="utf-8")
                        if content != existing_content:
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_text(content, encoding="utf-8")
                            modified.append(str(Path(rel) / rel_file))
                    else:
                        if dest.exists():
                            dest.unlink()
                            modified.append(str(Path(rel) / rel_file) + " (cleared)")
                else:
                    if dest.exists():
                        dest.unlink()
                        modified.append(str(Path(rel) / rel_file) + " (cleared)")

            for rel_dir in _AGENT_REFLECTION_TREE_DIRS:
                session_dir = session_agent_dir / rel_dir
                if session_dir.exists() and session_dir.is_dir():
                    has_content = any(
                        session_dir.rglob("*") if session_dir.exists() else []
                    )
                    if has_content:
                        dest = source_agent_dir / rel_dir
                        if not _tree_dirs_differ(session_dir, dest):
                            continue
                        dest_tmp = dest.with_name(dest.name + "._tmp_reflection")
                        try:
                            if dest_tmp.exists():
                                shutil.rmtree(str(dest_tmp))
                            shutil.copytree(str(session_dir), str(dest_tmp))
                            if dest.exists():
                                shutil.rmtree(str(dest))
                            dest_tmp.rename(dest)
                        except OSError as e:
                            logger.warning("failed to update %s: %s", rel_dir, e)
                            if dest_tmp.exists():
                                shutil.rmtree(str(dest_tmp), ignore_errors=True)
                            continue
                        for f in sorted(session_dir.rglob("*")):
                            if f.is_file():
                                rel_file_path = f.relative_to(session_agent_dir)
                                modified.append(str(Path(rel) / rel_file_path))

    if include_team_reflection:
        for rel_file in _TEAM_REFLECTION_FILES:
            session_file = session_path / rel_file
            if session_file.exists():
                source_file = source_path / rel_file
                session_content = session_file.read_text(encoding="utf-8")
                source_content = ""
                if source_file.exists():
                    source_content = source_file.read_text(encoding="utf-8")
                if session_content != source_content:
                    source_file.write_text(session_content, encoding="utf-8")
                    modified.append(rel_file)

    if include_handoff_reflection:
        for rel_file in _HANDOFF_REFLECTION_FILES:
            session_file = session_path / rel_file
            if session_file.exists():
                source_file = source_path / rel_file
                session_content = session_file.read_text(encoding="utf-8")
                source_content = ""
                if source_file.exists():
                    source_content = source_file.read_text(encoding="utf-8")
                if session_content != source_content:
                    source_file.write_text(session_content, encoding="utf-8")
                    modified.append(rel_file)

    session_agent_rels = {
        str(a.relative_to(session_path)) for a in session_agents
    }
    source_agent_rels = set(source_map.keys())
    if include_team_reflection:
        for session_agent_dir in session_agents:
            rel = str(session_agent_dir.relative_to(session_path))
            if rel not in source_agent_rels:
                dest = source_path / rel
                try:
                    shutil.copytree(str(session_agent_dir), str(dest))
                    modified.append(f"{rel}/ (new agent)")
                except OSError as e:
                    logger.warning("failed to copy new agent %s: %s", rel, e)

        min_required = max(1, len(source_agent_rels) // 2)
        if len(source_agent_rels) > 0 and len(session_agent_rels) >= min_required:
            for rel_str, source_agent_dir in source_map.items():
                if rel_str not in session_agent_rels:
                    try:
                        shutil.rmtree(str(source_agent_dir))
                        modified.append(f"{rel_str}/ (removed agent)")
                    except OSError as e:
                        logger.warning("failed to remove agent %s: %s", rel_str, e)
        else:
            missing = source_agent_rels - session_agent_rels
            if missing:
                logger.warning(
                    "SAFETY: skipping agent deletion — "
                    "session has %d agents vs source %d. Would have deleted: %s",
                    len(session_agent_rels), len(source_agent_rels), missing,
                )

    if modified:
        logger.info("persisted %d files to %s", len(modified), source_path)
        for f in modified:
            logger.info("  → %s", f)
    else:
        logger.info("no reflection files to persist")

    return modified


def classify_reflection_updates(modified_files: list[str]) -> dict[str, bool]:
    updates = {"agent": False, "team": False, "handoff": False}
    team_files = set(_TEAM_REFLECTION_FILES)
    handoff_files = set(_HANDOFF_REFLECTION_FILES)
    for raw_path in modified_files:
        path = raw_path.split(" (", 1)[0].rstrip("/")
        if path in handoff_files:
            updates["handoff"] = True
        elif path in team_files or "(new agent)" in raw_path or "(removed agent)" in raw_path:
            updates["team"] = True
        else:
            updates["agent"] = True
    return updates
