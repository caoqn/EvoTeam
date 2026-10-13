from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.evolution_models import ScopeAwareTeamRuleEvolutionDecision, HandoffRule, now_iso


class TeamScopedRuleGuidedCollaborationRegistry:
    SCHEMA_VERSION = 3

    def __init__(
        self,
        *,
        bindings: dict[str, str] | None = None,
        rules: list[HandoffRule] | None = None,
    ) -> None:
        self._bindings = dict(bindings or {})
        self._rules = list(rules or [])
        self.validate()

    def validate(self, known_agent_ids: set[str] | None = None) -> None:
        seen: set[tuple[str, str]] = set()
        for team_family_id, handoff_family_id in self._bindings.items():
            if not str(team_family_id).strip() or not str(handoff_family_id).strip():
                raise ValueError("handoff family bindings cannot be empty")
        for rule in self._rules:
            rule.validate(known_agent_ids)
            scope_key = (rule.handoff_family_id, rule.rule_id)
            if scope_key in seen:
                raise ValueError(
                    "duplicate handoff rule_id within handoff family: "
                    f"{rule.handoff_family_id}/{rule.rule_id}"
                )
            seen.add(scope_key)

    def bind_team_family(self, team_family_id: str, handoff_family_id: str) -> None:
        if not team_family_id.strip() or not handoff_family_id.strip():
            raise ValueError("handoff family binding requires both ids")
        existing = self._bindings.get(team_family_id)
        if existing is not None and existing != handoff_family_id:
            raise ValueError("team family is already bound to another handoff family")
        self._bindings[team_family_id] = handoff_family_id

    def handoff_family_for(self, team_family_id: str) -> str | None:
        return self._bindings.get(team_family_id)

    def rules_for_team_family(self, team_family_id: str) -> list[HandoffRule]:
        handoff_family_id = self.handoff_family_for(team_family_id)
        if not handoff_family_id:
            return []
        return [
            rule for rule in self._rules
            if rule.handoff_family_id == handoff_family_id and rule.status == "active"
        ]

    def list_rules(self) -> list[HandoffRule]:
        return sorted(
            self._rules,
            key=lambda rule: (rule.handoff_family_id, rule.rule_id),
        )

    def _find_target(self, decision: ScopeAwareTeamRuleEvolutionDecision) -> HandoffRule | None:
        if decision.rule_id:
            return next(
                (
                    rule for rule in self._rules
                    if rule.handoff_family_id == decision.handoff_family_id
                    and rule.rule_id == decision.rule_id
                ),
                None,
            )
        matching = [
            rule for rule in self._rules
            if rule.handoff_family_id == decision.handoff_family_id
            and rule.from_agent == decision.from_agent
            and rule.to_agent == decision.to_agent
            and rule.status == "active"
        ]
        if len(matching) == 1:
            return matching[0]
        return None

    @staticmethod
    def _append_failure_evidence(
        rule: HandoffRule,
        evidence: dict[str, Any] | None,
    ) -> None:
        if not evidence:
            return
        payload = dict(evidence)
        payload.setdefault("recorded_at", now_iso())
        identity = (
            str(payload.get("task_id") or ""),
            str(payload.get("message_id") or ""),
        )
        if any(
            (
                str(item.get("task_id") or ""),
                str(item.get("message_id") or ""),
            ) == identity
            for item in rule.failure_evidence
        ):
            return
        rule.failure_evidence.append(payload)

    def apply_decision(
        self,
        decision: ScopeAwareTeamRuleEvolutionDecision,
        *,
        task_id: str,
        known_agent_ids: set[str],
        team_family_id: str | None = None,
    ) -> HandoffRule | None:
        decision.validate()
        self.validate(known_agent_ids)
        if team_family_id:
            self.bind_team_family(team_family_id, decision.handoff_family_id)
        current = self._find_target(decision)
        if current is not None and (
            current.handoff_family_id != decision.handoff_family_id
            or current.from_agent != decision.from_agent
            or current.to_agent != decision.to_agent
        ):
            raise ValueError(
                "handoff_rule_id does not match the decision family or agent pair"
            )
        failure_evidence = decision.evidence.get("failure_evidence")
        if failure_evidence is not None and not isinstance(failure_evidence, dict):
            raise ValueError("handoff failure_evidence must be an object")
        if decision.action == "no_update":
            return current
        if decision.action == "retain":
            if current is None:
                raise ValueError("handoff retain requires an existing rule")
            self._append_failure_evidence(current, failure_evidence)
            if task_id and task_id not in current.evidence_task_ids:
                current.evidence_task_ids.append(task_id)
            return current
        if decision.action == "create":
            rule_id = decision.rule_id
            if any(
                rule.handoff_family_id == decision.handoff_family_id
                and rule.rule_id == rule_id
                for rule in self._rules
            ):
                raise ValueError(
                    "handoff rule_id already exists within handoff family: "
                    f"{decision.handoff_family_id}/{rule_id}"
                )
            rule = HandoffRule(
                handoff_family_id=decision.handoff_family_id,
                from_agent=decision.from_agent,
                to_agent=decision.to_agent,
                version=1,
                instruction=decision.instruction,
                trigger_description=decision.trigger_description,
                payload_schema=dict(decision.payload_schema),
                required_evidence=list(decision.required_evidence),
                verification=decision.verification,
                fallback=decision.fallback,
                context_notes=decision.context_notes,
                evidence_task_ids=[task_id] if task_id else [],
                failure_evidence=[
                    {**dict(failure_evidence), "recorded_at": now_iso()}
                ] if failure_evidence else [],
                rule_id=rule_id,
            )
            rule.validate(known_agent_ids)
            self._rules.append(rule)
            self.validate(known_agent_ids)
            return rule
        if decision.action == "retire":
            if current is None:
                raise ValueError("handoff retire requires an existing rule")
            current.version_history.append({
                "version": current.version,
                "instruction": current.instruction,
                "trigger_description": current.trigger_description,
                "payload_schema": current.payload_schema,
                "required_evidence": current.required_evidence,
                "verification": current.verification,
                "fallback": current.fallback,
                "context_notes": current.context_notes,
                "failure_evidence": list(current.failure_evidence),
                "reason": decision.reason,
                "task_id": task_id,
                "changed_at": now_iso(),
                "status": "retired",
            })
            self._append_failure_evidence(current, failure_evidence)
            current.status = "retired"
            if task_id and task_id not in current.evidence_task_ids:
                current.evidence_task_ids.append(task_id)
            current.validate(known_agent_ids)
            return current
        if current is None:
            raise ValueError("handoff refine requires an existing rule")
        current.version_history.append({
            "version": current.version,
            "instruction": current.instruction,
            "trigger_description": current.trigger_description,
            "payload_schema": current.payload_schema,
            "required_evidence": current.required_evidence,
            "verification": current.verification,
            "fallback": current.fallback,
            "context_notes": current.context_notes,
            "failure_evidence": list(current.failure_evidence),
            "reason": decision.reason,
            "task_id": task_id,
            "changed_at": now_iso(),
            "status": current.status,
        })
        self._append_failure_evidence(current, failure_evidence)
        current.version += 1
        current.instruction = decision.instruction
        current.trigger_description = decision.trigger_description
        current.payload_schema = dict(decision.payload_schema)
        current.required_evidence = list(decision.required_evidence)
        current.verification = decision.verification
        current.context_notes = decision.context_notes
        current.fallback = decision.fallback
        if task_id and task_id not in current.evidence_task_ids:
            current.evidence_task_ids.append(task_id)
        current.validate(known_agent_ids)
        return current

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.SCHEMA_VERSION,
            "bindings": dict(sorted(self._bindings.items())),
            "rules": [rule.to_dict() for rule in self.list_rules()],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamScopedRuleGuidedCollaborationRegistry":
        if not isinstance(payload, dict):
            raise ValueError("handoff registry must be an object")
        version = int(payload.get("schema_version") or 0)
        if version != cls.SCHEMA_VERSION:
            raise ValueError(f"unsupported handoff registry schema_version: {version}")
        bindings = payload.get("bindings", {})
        rows = payload.get("rules", [])
        if not isinstance(bindings, dict) or not isinstance(rows, list):
            raise ValueError("handoff registry bindings/rules have invalid types")
        return cls(
            bindings={str(k): str(v) for k, v in bindings.items()},
            rules=[HandoffRule(**row) for row in rows],
        )

    @classmethod
    def load(cls, path: str | Path) -> "TeamScopedRuleGuidedCollaborationRegistry":
        target = Path(path)
        if not target.exists():
            return cls()
        return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(target)
