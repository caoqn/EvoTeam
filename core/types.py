from dataclasses import dataclass


@dataclass
class TaskValidation:
    success: bool
    summary: str
    details: str | None = None