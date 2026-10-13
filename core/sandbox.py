from pathlib import Path


class SandboxViolation(Exception):
    pass


def resolve_path(path: str, workspace: str) -> Path:
    p = Path(path)
    ws = Path(workspace).resolve()

    if not p.is_absolute():
        p = ws / p

    p = p.resolve()

    try:
        p.relative_to(ws)
    except ValueError:
        raise SandboxViolation(
            f"Access denied: '{path}' resolves to '{p}' which is outside workspace '{ws}'"
        )

    return p