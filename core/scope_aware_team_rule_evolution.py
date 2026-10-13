from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from core.evolution_models import ScopeAwareTeamRuleEvolutionDecision
from core.team_scoped_rule_guided_collaboration import TeamScopedRuleGuidedCollaborationRegistry


@dataclass
class ScopeAwareTeamRuleEvolutionContext:
    task_id: str
    team_template_id: str
    handoff_family_id: str
    task: str
    actual_agent_ids: list[str]
    current_rules: list[dict[str, Any]] = field(default_factory=list)
    execution_summary: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    handoff_trace: list[dict[str, Any]] = field(default_factory=list)
    handoff_rule_usage: list[dict[str, Any]] = field(default_factory=list)
    historical_rule_summaries: list[dict[str, Any]] = field(default_factory=list)


class ScopeAwareTeamRuleEvolutionError(ValueError):
    pass


CompletionCallable = Callable[..., Awaitable[Any]]


class LLMScopeAwareTeamRuleEvolutionReflector:
    MAX_REPAIR_ATTEMPTS = 2

    def __init__(self, *, model: str, completion: CompletionCallable | None = None,
                 api_tracker: Any | None = None) -> None:
        self.model = model
        self._completion = completion
        self._api_tracker = api_tracker

    async def reflect(self, context: ScopeAwareTeamRuleEvolutionContext) -> ScopeAwareTeamRuleEvolutionDecision:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an interaction-rule reflection assistant. Given handoff "
                    "evidence, assess and update reusable information contracts within "
                    "the current team. Call record_handoff_decision exactly once.\n\n"
                    "Scope each rule to a directed sender-recipient pair within the "
                    "current team. Preserve distinct rule_id values when the same pair "
                    "requires different contracts under different triggers; do not "
                    "overwrite a different rule merely because its endpoints match.\n\n"
                    "Ground each decision in observed handoffs and outcomes, using "
                    "evaluation feedback and relevant history. Distinguish information-transfer "
                    "problems from tool or API failures, missing capabilities, and "
                    "incorrect task assignments.\n\n"
                    "Convert task-specific details into reusable requirements for "
                    "information, evidence, provenance, uncertainty, verification, and "
                    "fallback. Express each improvement as actionable handoff guidance, "
                    "without embedding one-task filenames, identifiers, or answers in "
                    "the reusable contract.\n\n"
                    "Choose create for a useful new contract, refine for a needed "
                    "revision, or retain when an existing rule remains supported. "
                    "Choose retire only when historical evidence supports persistent "
                    "mismatch, redundancy, harm, ineffective handoffs, or replacement by "
                    "a more reliable rule. The current trace or an isolated failure alone "
                    "does not meet this historical evidence requirement. Choose no_update "
                    "when evidence is insufficient. Do not turn an isolated failure into "
                    "an unconditional prohibition.\n\n"
                    "Update information contracts only. Leave team membership and agent "
                    "behavior or skills to their respective reflection modules. Do not "
                    "replace those updates with interaction rules or turn a handoff "
                    "contract into a fixed task workflow. Preserve valid prior guidance "
                    "when refining a rule."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "task_id": context.task_id,
                    "team_template_id": context.team_template_id,
                    "handoff_family_id": context.handoff_family_id,
                    "task": context.task,
                    "actual_agent_ids": context.actual_agent_ids,
                    "current_rules": context.current_rules,
                    "execution_summary": context.execution_summary,
                    "evaluation": context.evaluation,
                    "failures": context.failures,
                    "handoff_trace": context.handoff_trace,
                    "handoff_rule_usage": context.handoff_rule_usage,
                    "historical_rule_summaries": context.historical_rule_summaries,
                    "instruction": (
                        "Return one action and its supporting handoff and outcome "
                        "evidence in reason. For every rule-specific decision (create, "
                        "refine, retain, or retire), provide rule_id, from_agent, "
                        "to_agent, trigger_description, instruction, payload_schema "
                        "for required information, required_evidence, verification, "
                        "and fallback. The contract must preserve provenance and "
                        "uncertainty requirements; context_notes may carry additional "
                        "nuances. Use the existing rule_id for refine/retain/retire; "
                        "for create, use a concise semantic ID with lowercase words "
                        "separated by '_' or '-'. For retain/retire, reproduce the "
                        "identified current contract rather than inventing replacement "
                        "content. For refine, return a complete revised rule and "
                        "explain the evidence for the revision. trigger_description "
                        "must state a reusable applicability condition, not task-specific "
                        "routing. For no_update, leave rule_id, from_agent, and to_agent "
                        "empty. A failed evaluation may support create/refine/retain "
                        "when grounded in the observed handoff, but retire still "
                        "requires the historical evidence standard."
                    ),
                }, ensure_ascii=False),
            },
        ]
        validation_errors: list[str] = []
        for attempt in range(self.MAX_REPAIR_ATTEMPTS + 1):
            response = await self._call(messages)
            calls = self._extract_calls(response)
            try:
                if len(calls) != 1:
                    raise ValueError(
                        f"expected exactly one record_handoff_decision tool call, got {len(calls)}"
                    )
                decision = ScopeAwareTeamRuleEvolutionDecision(
                    action=str(calls[0].get("action") or "").strip().lower(),
                    handoff_family_id=context.handoff_family_id,
                    rule_id=str(calls[0].get("rule_id") or "").strip(),
                    from_agent=str(calls[0].get("from_agent") or "").strip(),
                    to_agent=str(calls[0].get("to_agent") or "").strip(),
                    reason=str(calls[0].get("reason") or "").strip(),
                    instruction=str(calls[0].get("instruction") or "").strip(),
                    trigger_description=str(calls[0].get("trigger_description") or "").strip(),
                    payload_schema=calls[0].get("payload_schema") or {},
                    required_evidence=[str(x) for x in calls[0].get("required_evidence", [])],
                    verification=str(calls[0].get("verification") or "").strip(),
                    fallback=str(calls[0].get("fallback") or "").strip(),
                    context_notes=str(calls[0].get("context_notes") or "").strip(),
                    evidence={"team_template_id": context.team_template_id},
                )
                decision.validate()

                if decision.action == "no_update":
                    if decision.rule_id or decision.from_agent or decision.to_agent:
                        raise ValueError(
                            "no_update requires empty rule_id, from_agent, and to_agent"
                        )
                    return decision

                actual_agents = set(context.actual_agent_ids)
                if decision.from_agent not in actual_agents or decision.to_agent not in actual_agents:
                    raise ValueError(
                        "handoff agents must participate in this task; "
                        f"actual agents: {sorted(actual_agents)}"
                    )
                observed_pairs = {
                    (str(item.get("from_agent") or ""), str(item.get("to_agent") or ""))
                    for item in context.handoff_trace
                }
                if observed_pairs and (decision.from_agent, decision.to_agent) not in observed_pairs:
                    raise ValueError(
                        "handoff decision must target an observed interaction; "
                        f"observed pairs: {sorted(observed_pairs)}"
                    )

                rules_by_key = {
                    (
                        str(rule.get("handoff_family_id") or ""),
                        str(rule.get("handoff_rule_id") or rule.get("rule_id") or ""),
                    ): rule
                    for rule in context.current_rules
                }
                rule_key = (decision.handoff_family_id, decision.rule_id)
                if decision.action == "create" and rule_key in rules_by_key:
                    raise ValueError(
                        "create rule_id already exists within handoff family: "
                        f"{decision.handoff_family_id}/{decision.rule_id}; "
                        "use refine/retain/retire for that rule"
                    )
                if decision.action in {"refine", "retain", "retire"}:
                    current = rules_by_key.get(rule_key)
                    if current is None:
                        raise ValueError(
                            f"{decision.action} requires an existing rule_id; "
                            "known rule ids in this handoff family: "
                            f"{sorted(key[1] for key in rules_by_key if key[0] == decision.handoff_family_id)}"
                        )
                    expected_pair = (
                        str(current.get("from_agent") or ""),
                        str(current.get("to_agent") or ""),
                    )
                    if (decision.from_agent, decision.to_agent) != expected_pair:
                        raise ValueError(
                            f"rule_id {decision.rule_id} belongs to pair {expected_pair}, "
                            f"not {(decision.from_agent, decision.to_agent)}"
                        )
                return decision
            except (TypeError, ValueError) as exc:
                error = str(exc) or exc.__class__.__name__
                validation_errors.append(error)
                if attempt >= self.MAX_REPAIR_ATTEMPTS:
                    break
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous handoff decision was invalid. Repair it and return "
                        "exactly one record_handoff_decision tool call. Do not repeat the "
                        f"same error. Validation error: {error}"
                    ),
                })

        return ScopeAwareTeamRuleEvolutionDecision(
            action="no_update",
            handoff_family_id=context.handoff_family_id,
            from_agent="",
            to_agent="",
            rule_id="",
            reason="handoff reflection invalid after two semantic repair attempts",
            evidence={
                "team_template_id": context.team_template_id,
                "reflection_status": "invalid_after_repair",
                "repair_attempts": self.MAX_REPAIR_ATTEMPTS,
                "validation_errors": validation_errors,
            },
        )

    async def _call(self, messages: list[dict[str, str]]) -> Any:
        completion = self._completion
        if completion is None:
            from core.llm import complete
            completion = complete
        return await completion(
            model=self.model,
            messages=messages,
            tools=[self._tool_schema()],
            temperature=0.1,
            max_tokens=1800,
            api_tracker=self._api_tracker,
        )

    @staticmethod
    def _extract_calls(response: Any) -> list[dict[str, Any]]:
        from core.llm import extract_tool_calls
        return [
            dict(call.get("arguments") or {})
            for call in extract_tool_calls(response)
            if call.get("name") == "record_handoff_decision"
            and isinstance(call.get("arguments"), dict)
        ]

    @staticmethod
    def _tool_schema() -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": "record_handoff_decision",
            "description": "Record one team-scoped interaction-rule decision.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "refine", "retain", "retire", "no_update"]},
                    "rule_id": {
                        "type": "string",
                        "description": "Stable id of this specific handoff rule. Required for create/refine/retain/retire; empty only for no_update.",
                    },
                    "from_agent": {"type": "string"},
                    "to_agent": {"type": "string"},
                    "reason": {"type": "string"},
                    "instruction": {"type": "string"},
                    "trigger_description": {
                        "type": "string",
                        "description": "Semantic condition for selecting this rule; may include a broad domain such as history or finance.",
                    },
                    "payload_schema": {"type": "object"},
                    "required_evidence": {"type": "array", "items": {"type": "string"}},
                    "verification": {"type": "string"},
                    "fallback": {"type": "string"},
                    "context_notes": {"type": "string"},
                },
                "required": ["action", "reason", "rule_id"],
                "additionalProperties": False,
            },
        }}
