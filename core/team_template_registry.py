from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.evolution_models import (
    ScopeAwareTeamTemplateEvolutionDecision,
    TeamTemplate,
    TemplateFamily,
    TemplateVersionSnapshot,
    now_iso,
)


class TeamTemplateRegistry:
    SCHEMA_VERSION = 2
    GLOBAL_SERVICE_AGENT_IDS = frozenset({"answer_agent"})

    def __init__(
        self,
        families: list[TemplateFamily] | None = None,
        templates: list[TeamTemplate] | None = None,
    ) -> None:
        family_rows = list(families or [])
        template_rows = list(templates or [])
        if len({item.family_id for item in family_rows}) != len(family_rows):
            raise ValueError("duplicate template family id")
        if len({item.template_id for item in template_rows}) != len(template_rows):
            raise ValueError("duplicate template id")
        self._families = {item.family_id: item for item in family_rows}
        self._templates = {item.template_id: item for item in template_rows}
        self.validate()

    def validate(self, known_agent_ids: set[str] | None = None) -> None:
        referenced_templates: set[str] = set()
        for family_id, family in self._families.items():
            family.validate()
            template = self._templates.get(family.current_template_id)
            if template is None:
                raise ValueError(
                    f"family {family_id} references a missing current template"
                )
            if template.family_id != family_id:
                raise ValueError("current template belongs to a different family")
            if family.current_template_id in referenced_templates:
                raise ValueError("one current template cannot belong to two families")
            referenced_templates.add(family.current_template_id)
        if set(self._templates) != referenced_templates:
            raise ValueError(
                "registry may store only the single current template for each family"
            )
        for template in self._templates.values():
            template.validate(known_agent_ids)
            member_ids = set(template.allowed_agent_ids)
            member_ids.update(
                member.agent_id
                for snapshot in template.version_history
                for member in snapshot.members
            )
            if member_ids & self.GLOBAL_SERVICE_AGENT_IDS:
                raise ValueError("global service agents cannot be team-template members")

    def list_families(self) -> list[TemplateFamily]:
        return [self._families[key] for key in sorted(self._families)]

    def list_templates(self) -> list[TeamTemplate]:
        return [self._templates[key] for key in sorted(self._templates)]

    def get_family(self, family_id: str) -> TemplateFamily | None:
        return self._families.get(family_id)

    def get_current(self, family_id: str) -> TeamTemplate | None:
        family = self.get_family(family_id)
        return self._templates.get(family.current_template_id) if family else None

    def catalog(self) -> list[dict[str, Any]]:
        rows = []
        for family in self.list_families():
            template = self._templates[family.current_template_id]
            rows.append(
                {
                    "family_id": family.family_id,
                    "family_description": family.description,
                    "handoff_family_id": family.handoff_family_id,
                    "template_id": template.template_id,
                    "template_version": template.version,
                    "members": [
                        {"agent_id": member.agent_id, "role": member.role}
                        for member in template.members
                    ],
                    "evidence_count": len(template.evidence_task_ids),
                    "success_count": len(template.successful_task_ids),
                    "failure_count": len(template.failed_task_ids),
                }
            )
        return rows

    def apply_reflection(
        self,
        decision: ScopeAwareTeamTemplateEvolutionDecision,
        *,
        task_id: str,
        known_agent_ids: set[str],
        task_success: bool = True,
        actual_agent_ids: list[str] | None = None,
    ) -> TeamTemplate | None:
        decision.validate(known_agent_ids)
        current = self.get_current(decision.family_id)
        if decision.action == "no_update":
            return current
        if decision.action == "retain":
            if current is None:
                raise ValueError("retain requires an existing family template")
            if decision.target_template_id != current.template_id:
                raise ValueError("retain must target the family's current template")
            self._append_evidence(
                current,
                task_id,
                task_success=task_success,
                actual_agent_ids=actual_agent_ids,
            )
            return current
        if decision.action == "create":
            if current is not None or decision.family_id in self._families:
                raise ValueError("create cannot add a competing template family branch")
            template = TeamTemplate(
                template_id=f"tpl_{decision.family_id}_{uuid4().hex[:8]}",
                family_id=decision.family_id,
                version=1,
                members=list(decision.members),
                evidence_task_ids=[task_id] if task_id else [],
                successful_task_ids=[task_id] if task_id and task_success else [],
                failed_task_ids=[task_id] if task_id and not task_success else [],
                task_member_evidence=(
                    {task_id: list(actual_agent_ids or [])}
                    if task_id and actual_agent_ids is not None
                    else {}
                ),
            )
            template.validate(known_agent_ids)
            family = TemplateFamily(
                family_id=decision.family_id,
                description=decision.family_description.strip(),
                current_template_id=template.template_id,
                handoff_family_id=f"hf_{decision.family_id}",
            )
            family.validate()
            self._families[family.family_id] = family
            self._templates[template.template_id] = template
            self.validate(known_agent_ids)
            return template
        if current is None:
            raise ValueError("refine requires an existing family template")
        if current.template_id != decision.target_template_id:
            raise ValueError("refine must target the family's current template")
        current.version_history.append(
            TemplateVersionSnapshot(
                version=current.version,
                members=list(current.members),
                task_id=task_id,
                reason=decision.reason,
                family_description=self._families[decision.family_id].description,
            )
        )
        current.version += 1
        current.members = list(decision.members)
        current.updated_at = now_iso()
        family = self._families[decision.family_id]
        family.description = decision.family_description.strip()
        self._append_evidence(
            current,
            task_id,
            task_success=task_success,
            actual_agent_ids=actual_agent_ids,
        )
        family.updated_at = current.updated_at
        current.validate(known_agent_ids)
        return current

    @staticmethod
    def _append_evidence(
        template: TeamTemplate,
        task_id: str,
        *,
        task_success: bool,
        actual_agent_ids: list[str] | None = None,
    ) -> None:
        if task_id and task_id not in template.evidence_task_ids:
            template.evidence_task_ids.append(task_id)
        if task_id and actual_agent_ids is not None:
            template.task_member_evidence[task_id] = list(actual_agent_ids)
        if not task_id:
            return
        target = (
            template.successful_task_ids
            if task_success
            else template.failed_task_ids
        )
        other = (
            template.failed_task_ids
            if task_success
            else template.successful_task_ids
        )
        if task_id in other:
            other.remove(task_id)
        if task_id not in target:
            target.append(task_id)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.SCHEMA_VERSION,
            "families": [item.to_dict() for item in self.list_families()],
            "templates": [item.to_dict() for item in self.list_templates()],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamTemplateRegistry":
        if not isinstance(payload, dict):
            raise ValueError("template registry must be a JSON object")
        version = int(payload.get("schema_version") or 0)
        if version != cls.SCHEMA_VERSION:
            raise ValueError(f"unsupported template registry schema_version: {version}")
        family_rows = payload.get("families", [])
        template_rows = payload.get("templates", [])
        if not isinstance(family_rows, list) or any(
            not isinstance(item, dict) for item in family_rows
        ):
            raise ValueError("registry families must be a list of objects")
        if not isinstance(template_rows, list) or any(
            not isinstance(item, dict) for item in template_rows
        ):
            raise ValueError("registry templates must be a list of objects")
        families = [
            TemplateFamily.from_dict(item) for item in family_rows
        ]
        templates = [
            TeamTemplate.from_dict(item) for item in template_rows
        ]
        return cls(families=families, templates=templates)

    @classmethod
    def load(cls, path: str | Path) -> "TeamTemplateRegistry":
        registry_path = Path(path)
        if not registry_path.exists():
            return cls()
        try:
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid template registry JSON: {exc.msg}") from exc
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> None:
        registry_path = Path(path)
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = registry_path.with_name(
            f".{registry_path.name}.tmp-{uuid4().hex[:8]}"
        )
        try:
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, registry_path)
        finally:
            if temporary.exists():
                temporary.unlink()
