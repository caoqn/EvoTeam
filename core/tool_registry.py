import sys
import logging
import importlib.util
import yaml
from pathlib import Path
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


class Tool:

    def __init__(self, name: str, schema: dict, execute: Callable[..., Awaitable[str] | str]):
        self.name = name
        self.schema = schema
        self.execute = execute

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": self.schema,
        }


class Skill:

    def __init__(self, name: str, description: str, path: str, content: str = ""):
        self.name = name
        self.description = description
        self.path = path
        self.content = content


def extract_skill_metadata(filepath: str) -> Skill:
    try:
        text = Path(filepath).read_text(encoding="utf-8")
    except OSError:
        name = Path(filepath).parent.name
        return Skill(name=name, description="", path=filepath)

    lines = text.split("\n")
    in_frontmatter = False
    frontmatter_lines = []

    for line in lines:
        if line.strip() == "---":
            if in_frontmatter:
                break
            in_frontmatter = True
            continue
        if in_frontmatter:
            frontmatter_lines.append(line)

    name = ""
    description = ""
    if frontmatter_lines:
        try:
            fm = yaml.safe_load("\n".join(frontmatter_lines))
            if isinstance(fm, dict):
                name = str(fm.get("name", "")).strip()
                description = str(fm.get("description", "")).strip()
        except yaml.YAMLError:
            pass

    if not name:
        name = Path(filepath).parent.name

    return Skill(name=name, description=description, path=filepath)


def load_skill_content(filepath: str) -> str:
    try:
        text = Path(filepath).read_text(encoding="utf-8")
    except OSError:
        return f"Error: could not read skill file: {filepath}"

    lines = text.split("\n")
    in_frontmatter = False
    frontmatter_ended = False
    content_lines = []

    for line in lines:
        if line.strip() == "---":
            if in_frontmatter:
                frontmatter_ended = True
                continue
            in_frontmatter = True
            continue
        if frontmatter_ended or not in_frontmatter:
            content_lines.append(line)

    return "\n".join(content_lines).strip()


class ToolRegistry:

    def __init__(self, tools_dir: str):
        self.tools_dir = tools_dir
        self._tools: dict[str, Tool] = {}
        self._load_global_tools()

    def _load_global_tools(self):
        tools_path = Path(self.tools_dir)
        if not tools_path.exists():
            return

        for py_file in sorted(tools_path.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            name = py_file.stem
            module_name = f"tools.{name}"

            if module_name in sys.modules:
                mod = sys.modules[module_name]
            else:
                spec = importlib.util.spec_from_file_location(
                    module_name, str(py_file))
                if not spec or not spec.loader:
                    continue
                mod = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(mod)
                except Exception:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Failed to load tool module '%s' from %s", module_name, py_file,
                        exc_info=True,
                    )
                    continue
                sys.modules[module_name] = mod

            schema = getattr(mod, "SCHEMA", None)
            execute = getattr(mod, "execute", None)
            if schema and execute:
                self._tools[name] = Tool(name=name, schema=schema, execute=execute)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_tools(self, names: list[str]) -> list[Tool]:
        return [self._tools[n] for n in names if n in self._tools]

    def all_names(self) -> list[str]:
        return list(self._tools.keys())

    def to_openai_tools(self, names: list[str] | None = None) -> list[dict]:
        if names is None:
            tools = list(self._tools.values())
        else:
            tools = []
            for n in names:
                if n in self._tools:
                    tools.append(self._tools[n])
                else:
                    logger.warning("tool '%s' not found in registry, skipping", n)
        return [t.to_openai_tool() for t in tools]


def load_agent_skills(agent_dir: str) -> list[Skill]:
    skills_dir = Path(agent_dir) / "skills"
    if not skills_dir.exists():
        return []

    skills = []
    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
            skills.append(extract_skill_metadata(str(skill_md)))
    return skills
