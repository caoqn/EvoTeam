from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


FamilyAction = Literal["reuse", "create", "no_match"]
SkillAction = Literal["create", "refine", "retain", "retire"]
HandoffAction = Literal["create", "refine", "retain", "retire", "no_update"]
TeamReflectionAction = Literal["create", "refine", "retain", "no_update"]

_FAMILY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,80}$")
_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SEMANTIC_HANDOFF_RULE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,95}$")
_FORBIDDEN_TEMPLATE_FIELDS = {
    "assignments",
    "capabilities",
    "capability",
    "dependencies",
    "handoff_rules",
    "skills",
    "slot",
    "slots",
    "workflow",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_family_id(value: str) -> str:
    family_id = str(value or "").strip()
    if not _FAMILY_ID_PATTERN.fullmatch(family_id):
        raise ValueError(
            "family_id must be 3-81 characters of lowercase snake_case"
        )
    return family_id


def validate_artifact_id(value: str, field_name: str) -> str:
    artifact_id = str(value or "").strip()
    if not _ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
        raise ValueError(f"{field_name} is missing or contains invalid characters")
    return artifact_id


def validate_semantic_handoff_rule_id(value: str) -> str:
    rule_id = str(value or "").strip()
    if not _SEMANTIC_HANDOFF_RULE_ID_PATTERN.fullmatch(rule_id):
        raise ValueError(
            "handoff_rule_id must be 3-96 lowercase letters, digits, hyphens, or underscores "
            "and start with a letter"
        )
    if re.fullmatch(r"(?:hr[_-])?[0-9a-f]{8,96}", rule_id):
        raise ValueError("handoff_rule_id must describe the rule, not be a random hex ID")
    if len(re.findall(r"[a-z]+", rule_id)) < 2:
        raise ValueError("handoff_rule_id must contain at least two semantic words")
    return rule_id


def _unique_strings(values: list[str], field_name: str) -> list[str]:
    normalized = [str(value).strip() for value in values if str(value).strip()]
    if len(normalized) != len(values) or len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique non-empty strings")
    return normalized


def _object_list(value: Any, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{field_name} must contain only objects")
    return value


def _reject_forbidden_template_fields(payload: dict[str, Any]) -> None:
    forbidden = sorted(set(payload) & _FORBIDDEN_TEMPLATE_FIELDS)
    if forbidden:
        raise ValueError(
            f"team template contains forbidden execution fields: {forbidden}"
        )


@dataclass(frozen=True)
class TemplateMember:
    agent_id: str
    role: str

    def validate(self) -> None:
        validate_artifact_id(self.agent_id, "agent_id")
        if not self.role.strip():
            raise ValueError("template member role cannot be empty")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemplateMember":
        _reject_forbidden_template_fields(payload)
        member = cls(
            agent_id=str(payload.get("agent_id") or ""),
            role=str(payload.get("role") or ""),
        )
        member.validate()
        return member


@dataclass(frozen=True)
class TemplateVersionSnapshot:
    version: int
    members: list[TemplateMember]
    task_id: str
    reason: str
    family_description: str = ""
    changed_at: str = field(default_factory=now_iso)

    def validate(self) -> None:
        if self.version < 1:
            raise ValueError("template snapshot version must be positive")
        _validate_members(self.members)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemplateVersionSnapshot":
        member_rows = _object_list(payload.get("members", []), "snapshot members")
        snapshot = cls(
            version=int(payload.get("version") or 0),
            members=[TemplateMember.from_dict(item) for item in member_rows],
            task_id=str(payload.get("task_id") or ""),
            reason=str(payload.get("reason") or ""),
            family_description=str(payload.get("family_description") or ""),
            changed_at=str(payload.get("changed_at") or now_iso()),
        )
        snapshot.validate()
        return snapshot


def _validate_members(members: list[TemplateMember]) -> None:
    if not members:
        raise ValueError("team template must contain at least one recruitable member")
    for member in members:
        member.validate()
    member_ids = [member.agent_id for member in members]
    _unique_strings(member_ids, "template member ids")


@dataclass
class TeamTemplate:
    template_id: str
    family_id: str
    version: int
    members: list[TemplateMember]
    status: str = "active"
    version_history: list[TemplateVersionSnapshot] = field(default_factory=list)
    evidence_task_ids: list[str] = field(default_factory=list)
    successful_task_ids: list[str] = field(default_factory=list)
    failed_task_ids: list[str] = field(default_factory=list)
    task_member_evidence: dict[str, list[str]] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    @property
    def allowed_agent_ids(self) -> list[str]:
        return [member.agent_id for member in self.members]

    def validate(self, known_agent_ids: set[str] | None = None) -> None:
        validate_artifact_id(self.template_id, "template_id")
        validate_family_id(self.family_id)
        if self.version < 1:
            raise ValueError("team template version must be positive")
        if self.status != "active":
            raise ValueError("the canonical registry stores only active templates")
        _validate_members(self.members)
        _unique_strings(self.evidence_task_ids, "evidence_task_ids")
        _unique_strings(self.successful_task_ids, "successful_task_ids")
        _unique_strings(self.failed_task_ids, "failed_task_ids")
        overlap = set(self.successful_task_ids) & set(self.failed_task_ids)
        if overlap:
            raise ValueError(f"template evidence has conflicting outcomes: {sorted(overlap)}")
        if not isinstance(self.task_member_evidence, dict):
            raise ValueError("task_member_evidence must be an object")
        evidence_ids = set(self.evidence_task_ids)
        for task_id, member_ids in self.task_member_evidence.items():
            if not str(task_id).strip() or task_id not in evidence_ids:
                raise ValueError(
                    "task_member_evidence keys must be non-empty evidence task ids"
                )
            if not isinstance(member_ids, list):
                raise ValueError("task_member_evidence values must be lists")
            _unique_strings(member_ids, f"task_member_evidence[{task_id}]")
        for snapshot in self.version_history:
            snapshot.validate()
        if known_agent_ids is not None:
            unknown = sorted(set(self.allowed_agent_ids) - set(known_agent_ids))
            if unknown:
                raise ValueError(f"template references unknown agents: {unknown}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamTemplate":
        _reject_forbidden_template_fields(payload)
        member_rows = _object_list(payload.get("members", []), "template members")
        history_rows = _object_list(
            payload.get("version_history", []), "template version_history"
        )
        evidence_rows = payload.get("evidence_task_ids", [])
        if not isinstance(evidence_rows, list):
            raise ValueError("evidence_task_ids must be a list")
        successful_rows = payload.get("successful_task_ids", [])
        failed_rows = payload.get("failed_task_ids", [])
        member_evidence = payload.get("task_member_evidence", {})
        if not isinstance(successful_rows, list) or not isinstance(failed_rows, list):
            raise ValueError("template outcome evidence must be lists")
        if not isinstance(member_evidence, dict):
            raise ValueError("task_member_evidence must be an object")
        template = cls(
            template_id=str(payload.get("template_id") or ""),
            family_id=str(payload.get("family_id") or ""),
            version=int(payload.get("version") or 0),
            members=[TemplateMember.from_dict(item) for item in member_rows],
            status=str(payload.get("status") or "active"),
            version_history=[
                TemplateVersionSnapshot.from_dict(item) for item in history_rows
            ],
            evidence_task_ids=[str(item) for item in evidence_rows],
            successful_task_ids=[str(item) for item in successful_rows],
            failed_task_ids=[str(item) for item in failed_rows],
            task_member_evidence={
                str(task_id): [str(agent_id) for agent_id in member_ids]
                for task_id, member_ids in member_evidence.items()
            },
            created_at=str(payload.get("created_at") or now_iso()),
            updated_at=str(payload.get("updated_at") or now_iso()),
        )
        template.validate()
        return template


@dataclass
class TemplateFamily:
    family_id: str
    description: str
    current_template_id: str
    handoff_family_id: str = ""
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def validate(self) -> None:
        validate_family_id(self.family_id)
        validate_artifact_id(self.current_template_id, "current_template_id")
        if not self.description.strip():
            raise ValueError("template family description cannot be empty")
        if not self.handoff_family_id.strip():
            raise ValueError("template family handoff_family_id cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemplateFamily":
        family = cls(
            family_id=str(payload.get("family_id") or ""),
            description=str(payload.get("description") or ""),
            current_template_id=str(payload.get("current_template_id") or ""),
            handoff_family_id=str(payload.get("handoff_family_id") or ""),
            created_at=str(payload.get("created_at") or now_iso()),
            updated_at=str(payload.get("updated_at") or now_iso()),
        )
        family.validate()
        return family


@dataclass
class TaskAdaptiveAgentTeamSelection:
    family_action: FamilyAction
    family_id: str
    chairman_id: str
    allowed_agent_ids: list[str]
    template_id: str | None = None
    template_version: int | None = None
    family_description: str = ""
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self, known_agent_ids: set[str] | None = None) -> None:
        if self.family_action not in {"reuse", "create", "no_match"}:
            raise ValueError(f"unsupported family_action: {self.family_action}")
        validate_artifact_id(self.chairman_id, "chairman_id")
        self.allowed_agent_ids = _unique_strings(
            self.allowed_agent_ids, "allowed_agent_ids"
        )
        if not self.allowed_agent_ids:
            raise ValueError("team selection requires at least one allowed agent")
        if self.chairman_id in self.allowed_agent_ids:
            raise ValueError("chairman_id must not be duplicated in allowed_agent_ids")
        if self.family_action in {"reuse", "create"}:
            validate_family_id(self.family_id)
        if self.family_action == "reuse":
            validate_artifact_id(self.template_id or "", "template_id")
            if not self.template_version or self.template_version < 1:
                raise ValueError("reuse requires a positive template_version")
        elif self.template_id is not None or self.template_version is not None:
            raise ValueError("only reuse selections may bind an existing template")
        if self.family_action == "create" and not self.family_description.strip():
            raise ValueError("cold-start selection requires family_description")
        if known_agent_ids is not None:
            unknown = sorted(
                {self.chairman_id, *self.allowed_agent_ids} - set(known_agent_ids)
            )
            if unknown:
                raise ValueError(f"selection references unknown agents: {unknown}")


@dataclass
class SkillDecision:
    action: SkillAction
    agent_id: str
    skill_id: str
    reason: str
    skill_name: str = ""
    description: str = ""
    trigger: list[str] = field(default_factory=list)
    content: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.action not in {"create", "refine", "retain", "retire"}:
            raise ValueError(f"unsupported skill action: {self.action}")
        validate_artifact_id(self.agent_id, "agent_id")
        validate_artifact_id(self.skill_id, "skill_id")
        if not self.reason.strip():
            raise ValueError("skill decision reason cannot be empty")
        if self.action == "create" and (
            not self.skill_name.strip()
            or not self.description.strip()
            or not self.trigger
            or not self.content.strip()
        ):
            raise ValueError(
                "skill create requires name, description, trigger, and content"
            )
        if self.action == "refine" and not self.content.strip():
            raise ValueError("skill refine requires replacement content")


@dataclass
class HandoffRule:
    handoff_family_id: str
    from_agent: str
    to_agent: str
    version: int
    instruction: str
    trigger_description: str = ""
    payload_schema: dict[str, Any] = field(default_factory=dict)
    required_evidence: list[str] = field(default_factory=list)
    verification: str = ""
    fallback: str = ""
    context_notes: str = ""
    evidence_task_ids: list[str] = field(default_factory=list)
    failure_evidence: list[dict[str, Any]] = field(default_factory=list)
    version_history: list[dict[str, Any]] = field(default_factory=list)
    rule_id: str = ""
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self, known_agent_ids: set[str] | None = None) -> None:
        validate_artifact_id(self.handoff_family_id, "handoff_family_id")
        validate_artifact_id(self.from_agent, "from_agent")
        validate_artifact_id(self.to_agent, "to_agent")
        if self.from_agent == self.to_agent:
            raise ValueError("handoff rule must connect two different agents")
        validate_artifact_id(self.rule_id, "handoff_rule_id")
        if self.version < 1:
            raise ValueError("handoff rule version must be positive")
        if not self.instruction.strip():
            raise ValueError("handoff instruction cannot be empty")
        if not self.trigger_description.strip():
            raise ValueError("handoff trigger_description cannot be empty")
        if self.status not in {"active", "retired"}:
            raise ValueError("handoff rule status must be active or retired")
        _unique_strings(self.evidence_task_ids, "evidence_task_ids")
        _unique_strings(self.required_evidence, "required_evidence")
        if any(not isinstance(item, dict) for item in self.failure_evidence):
            raise ValueError("handoff failure_evidence must contain objects")
        if any(not isinstance(item, dict) for item in self.version_history):
            raise ValueError("handoff version_history must contain objects")
        if known_agent_ids is not None:
            unknown = sorted(
                {self.from_agent, self.to_agent} - set(known_agent_ids)
            )
            if unknown:
                raise ValueError(f"handoff rule references unknown agents: {unknown}")


@dataclass
class ScopeAwareTeamRuleEvolutionDecision:
    action: HandoffAction
    handoff_family_id: str
    from_agent: str
    to_agent: str
    reason: str
    instruction: str = ""
    trigger_description: str = ""
    payload_schema: dict[str, Any] = field(default_factory=dict)
    required_evidence: list[str] = field(default_factory=list)
    verification: str = ""
    fallback: str = ""
    context_notes: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    rule_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.action not in {"create", "refine", "retain", "retire", "no_update"}:
            raise ValueError(f"unsupported handoff action: {self.action}")
        if not self.reason.strip():
            raise ValueError("handoff decision reason cannot be empty")
        if self.action == "no_update":
            return
        validate_artifact_id(self.handoff_family_id, "handoff_family_id")
        validate_artifact_id(self.from_agent, "from_agent")
        validate_artifact_id(self.to_agent, "to_agent")
        validate_semantic_handoff_rule_id(self.rule_id)
        if self.from_agent == self.to_agent:
            raise ValueError("handoff decision must connect two different agents")
        if self.action == "create" and not self.instruction.strip():
            raise ValueError("handoff create requires instruction")
        if self.action == "create" and not self.trigger_description.strip():
            raise ValueError("handoff create requires trigger_description")
        if self.action == "refine":
            if not self.instruction.strip():
                raise ValueError("handoff refine requires instruction")
            if not self.trigger_description.strip():
                raise ValueError("handoff refine requires trigger_description")
        if self.action in {"create", "refine"}:
            _unique_strings(self.required_evidence, "required_evidence")


@dataclass
class ScopeAwareTeamTemplateEvolutionDecision:
    action: TeamReflectionAction
    family_id: str
    reason: str
    members: list[TemplateMember] = field(default_factory=list)
    family_description: str = ""
    target_template_id: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self, known_agent_ids: set[str] | None = None) -> None:
        if self.action not in {"create", "refine", "retain", "no_update"}:
            raise ValueError(f"unsupported team reflection action: {self.action}")
        validate_family_id(self.family_id)
        if not self.reason.strip():
            raise ValueError("team reflection reason cannot be empty")
        if self.action in {"create", "refine"}:
            _validate_members(self.members)
            if not self.family_description.strip():
                raise ValueError(
                    f"template {self.action} requires family_description"
                )
        if self.action == "create":
            if self.target_template_id is not None:
                raise ValueError("create must not target an existing template")
        elif self.action == "refine":
            validate_artifact_id(
                self.target_template_id or "", "target_template_id"
            )
        elif self.action == "retain":
            validate_artifact_id(
                self.target_template_id or "", "target_template_id"
            )
        if self.action in {"retain", "no_update"}:
            if self.members or self.family_description.strip():
                raise ValueError(
                    "retain/no_update must not emit replacement team content or description"
                )
        if known_agent_ids is not None and self.action in {"create", "refine"}:
            team_ids = {member.agent_id for member in self.members}
            unknown = sorted(team_ids - set(known_agent_ids))
            if unknown:
                raise ValueError(f"team reflection references unknown agents: {unknown}")
