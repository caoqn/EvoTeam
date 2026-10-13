from __future__ import annotations

import json
import inspect
import hashlib
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from core.evolution_models import TaskAdaptiveAgentTeamSelection
from core.team_template_registry import TeamTemplateRegistry


@dataclass(frozen=True)
class PublicAgentProfile:
    agent_id: str
    role: str
    description: str


@dataclass
class TaskAdaptiveAgentTeamFormationContext:
    raw_task: str
    chairman_id: str
    template_catalog: list[dict[str, Any]] = field(default_factory=list)


class TaskAdaptiveAgentTeamFormationProtocol(Protocol):
    async def select(self, context: TaskAdaptiveAgentTeamFormationContext) -> TaskAdaptiveAgentTeamSelection:
        ...


class TaskAdaptiveAgentTeamFormationError(ValueError):
    pass


CompletionCallable = Callable[..., Awaitable[Any]]
AgentCatalogProvider = Callable[
    [], list[PublicAgentProfile] | Awaitable[list[PublicAgentProfile]]
]


class LLMTaskAdaptiveAgentTeamFormer:

    MAX_ATTEMPTS = 3

    def __init__(
        self,
        *,
        model: str,
        registry: TeamTemplateRegistry,
        completion: CompletionCallable | None = None,
        agent_catalog_provider: AgentCatalogProvider | None = None,
        api_tracker: Any | None = None,
    ) -> None:
        self.model = model
        self.registry = registry
        self._completion = completion
        self._agent_catalog_provider = agent_catalog_provider
        self._api_tracker = api_tracker
        self.last_diagnostics: dict[str, Any] = {"status": "not_started"}

    async def select(self, context: TaskAdaptiveAgentTeamFormationContext) -> TaskAdaptiveAgentTeamSelection:
        if not context.raw_task.strip():
            raise TaskAdaptiveAgentTeamFormationError("team selection requires a non-empty task")
        try:
            template_decision, template_selection_attempts = await self._select_team_template(context)
        except TaskAdaptiveAgentTeamFormationError as exc:
            return await self._fallback_cold_start(context, str(exc))
        action = str(template_decision.get("family_action") or "").strip().lower()
        family_id = str(template_decision.get("family_id") or "").strip()
        if action == "reuse":
            current = self.registry.get_current(family_id)
            if current is None:
                raise TaskAdaptiveAgentTeamFormationError(
                    "team-template selector returned reuse for a missing team template"
                )
            selection = TaskAdaptiveAgentTeamSelection(
                family_action="reuse",
                family_id=family_id,
                chairman_id=context.chairman_id,
                allowed_agent_ids=list(current.allowed_agent_ids),
                template_id=current.template_id,
                template_version=current.version,
                reason=str(template_decision.get("reason") or "").strip(),
                metadata={
                    "source": "task_adaptive_agent_team_formation",
                    "family_attempt_count": len(template_selection_attempts),
                    "cold_start_composer_used": False,
                },
            )
            known_agent_ids = None
        else:
            template_decision = {
                **template_decision,
                "family_id": self._provisional_team_template_id(context.raw_task),
            }
            agents = await self._load_agent_catalog()
            known_agent_ids = {
                context.chairman_id,
                *(agent.agent_id for agent in agents),
            }
            selection, composition_attempts = await self._compose_cold_start(
                context, template_decision, agents
            )
            selection.metadata.update(
                {
                    "source": "task_adaptive_agent_team_formation",
                    "family_attempt_count": len(template_selection_attempts),
                    "cold_start_composer_used": True,
                    "composition_attempt_count": len(composition_attempts),
                }
            )
        try:
            validate_task_adaptive_team_formation(
                selection,
                registry=self.registry,
                known_agent_ids=known_agent_ids,
                expected_chairman_id=context.chairman_id,
            )
        except ValueError as exc:
            self.last_diagnostics = {
                "status": "invalid_selection",
                "family_attempts": template_selection_attempts,
                "error": str(exc),
            }
            raise TaskAdaptiveAgentTeamFormationError(str(exc)) from exc
        self.last_diagnostics = {
            "status": "selected",
            "family_attempts": template_selection_attempts,
            "cold_start_composer_used": action == "cold_start",
        }
        return selection

    @staticmethod
    def _provisional_team_template_id(raw_task: str) -> str:
        digest = hashlib.sha1(raw_task.encode("utf-8")).hexdigest()[:12]
        return f"family_{digest}"

    async def _fallback_cold_start(
        self, context: TaskAdaptiveAgentTeamFormationContext, failure_reason: str,
    ) -> TaskAdaptiveAgentTeamSelection:
        agents = await self._load_agent_catalog()
        digest = hashlib.sha1(context.raw_task.encode("utf-8")).hexdigest()[:12]
        template_decision = {
            "family_id": f"fallback_{digest}",
            "family_description": "Automatic cold-start fallback after template selection failure",
            "reason": f"Template selection failed: {failure_reason}",
        }
        try:
            selection, attempts = await self._compose_cold_start(
                context, template_decision, agents,
            )
            roster_source = "llm_cold_start_composer"
        except TaskAdaptiveAgentTeamFormationError as exc:
            selection = TaskAdaptiveAgentTeamSelection(
                family_action="create",
                family_id=template_decision["family_id"],
                family_description=template_decision["family_description"],
                chairman_id=context.chairman_id,
                allowed_agent_ids=[
                    agent.agent_id for agent in agents
                    if agent.agent_id != "answer_agent"
                ],
                reason=(
                    f"Template selection and cold-start composition failed; "
                    f"using the public roster. {exc}"
                ),
            )
            attempts = []
            roster_source = "deterministic_all_public_agents"
        selection.metadata.update({
            "source": "task_adaptive_agent_team_formation_fallback",
            "fallback_reason": failure_reason,
            "cold_start_composer_used": True,
            "composition_attempt_count": len(attempts),
            "fallback_roster_source": roster_source,
        })
        validate_task_adaptive_team_formation(
            selection,
            registry=self.registry,
            known_agent_ids={context.chairman_id, *(a.agent_id for a in agents)},
            expected_chairman_id=context.chairman_id,
        )
        self.last_diagnostics = {
            "status": "fallback_cold_start",
            "fallback_reason": failure_reason,
            "composition_attempts": attempts,
        }
        return selection

    async def _select_team_template(
        self, context: TaskAdaptiveAgentTeamFormationContext
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a team-selection assistant. Given a task specification and "
                    "the team library, determine whether to reuse an existing agent team "
                    "or initiate a cold start. Call select_agent_team_template exactly once. "
                    "Match the task's collaboration pattern, work requirements, specialist "
                    "roles, and verification needs to the available teams. Choose reuse "
                    "only when an existing team supports these requirements; topic or "
                    "domain similarity alone does not justify reuse. Choose cold_start "
                    "when no team is suitable. Keep the cold-start team provisional: "
                    "persistent creation is decided by team reflection after execution. "
                    "Select a team only. Do not recruit members or assign work; the global "
                    "agent pool is not available at this stage."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": context.raw_task,
                        "template_catalog": context.template_catalog,
                        "instruction": (
                            "Return the decision (reuse or cold_start), family_id, and "
                            "selection rationale in reason. For reuse, use the exact "
                            "existing family_id. For cold_start, propose a provisional "
                            "lowercase snake_case family_id and a short family_description "
                            "of the required collaboration pattern. This does not admit "
                            "the team to the persistent library. Use the supplied task "
                            "specification and catalog as evidence; do not output agents, "
                            "assignments, workflows, skills, or interaction rules."
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            response = await self._call(messages, self._team_template_tool_schema())
            calls = self._extract_calls(response, "select_agent_team_template")
            invalid_reason = "select_agent_team_template must be called exactly once"
            if len(calls) == 1:
                invalid_reason = self._team_template_decision_error(calls[0])
            attempts.append(
                {
                    "attempt": attempt,
                    "family_call_count": len(calls),
                    "valid": not invalid_reason,
                    "invalid_reason": invalid_reason,
                }
            )
            if len(calls) == 1 and not invalid_reason:
                return calls[0], attempts
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The previous team-template decision was invalid: {invalid_reason}. "
                        "Call select_agent_team_template exactly once."
                    ),
                }
            )
        raise TaskAdaptiveAgentTeamFormationError(
            "LLM team-template selection remained invalid after bounded retries: "
            f"{attempts[-1]['invalid_reason']}"
        )

    async def _compose_cold_start(
        self,
        context: TaskAdaptiveAgentTeamFormationContext,
        template_decision: dict[str, Any],
        agents: list[PublicAgentProfile],
    ) -> tuple[TaskAdaptiveAgentTeamSelection, list[dict[str, Any]]]:
        family_id = str(template_decision.get("family_id") or "").strip()
        description = str(template_decision.get("family_description") or "").strip()
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a team-composition assistant. Given a task specification, "
                    "a provisional team description and selection rationale, a fixed "
                    "Chairman, and the public agent pool, select candidate members for "
                    "the cold-start team. Call compose_cold_start_team exactly once. "
                    "Keep the Chairman fixed and exclude it from the candidate set. "
                    "The selected members are candidates for recruitment; the Chairman "
                    "decides which candidates to activate and assigns their work."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": context.raw_task,
                        "new_family_id": family_id,
                        "family_description": description,
                        "selector_reason": str(template_decision.get("reason") or "").strip(),
                        "fixed_chairman_id": context.chairman_id,
                        "public_agent_registry": [
                            {
                                "agent_id": item.agent_id,
                                "role": item.role,
                                "description": item.description,
                            }
                            for item in agents
                            if item.agent_id not in {context.chairman_id, "answer_agent"}
                        ],
                        "instruction": (
                            "Return a non-empty provisional_agent_ids list covering the "
                            "task's required capabilities and provisional collaboration "
                            "pattern. Select only from public_agent_registry using exact "
                            "existing IDs; do not invent agents or identifiers. Exclude "
                            "fixed_chairman_id. Leave activation, work assignments, and "
                            "execution coordination to the Chairman. Do not prescribe a "
                            "workflow or create interaction rules, skills, or task assignments."
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        attempts: list[dict[str, Any]] = []
        known_agent_ids = {
            context.chairman_id,
            *(item.agent_id for item in agents),
        }
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            response = await self._call(messages, self._cold_start_tool_schema())
            calls = self._extract_calls(response, "compose_cold_start_team")
            invalid_reason = "compose_cold_start_team must be called exactly once"
            selection: TaskAdaptiveAgentTeamSelection | None = None
            if len(calls) == 1:
                provisional = calls[0].get("provisional_agent_ids", [])
                if not isinstance(provisional, list):
                    invalid_reason = "provisional_agent_ids must be a list"
                else:
                    selection = TaskAdaptiveAgentTeamSelection(
                        family_action="create",
                        family_id=family_id,
                        family_description=description,
                        chairman_id=context.chairman_id,
                        allowed_agent_ids=[str(item).strip() for item in provisional],
                        reason=str(template_decision.get("reason") or "").strip(),
                    )
                    try:
                        selection.validate(known_agent_ids)
                        invalid_reason = ""
                    except ValueError as exc:
                        invalid_reason = str(exc)
            attempts.append(
                {
                    "attempt": attempt,
                    "composition_call_count": len(calls),
                    "valid": not invalid_reason,
                    "invalid_reason": invalid_reason,
                }
            )
            if selection is not None and not invalid_reason:
                return selection, attempts
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The previous cold-start composition was invalid: "
                        f"{invalid_reason}. Call compose_cold_start_team exactly once."
                    ),
                }
            )
        raise TaskAdaptiveAgentTeamFormationError(
            "LLM cold-start composition remained invalid after bounded retries: "
            f"{attempts[-1]['invalid_reason']}"
        )

    async def _load_agent_catalog(self) -> list[PublicAgentProfile]:
        if self._agent_catalog_provider is None:
            raise TaskAdaptiveAgentTeamFormationError(
                "cold start requires an agent_catalog_provider"
            )
        result = self._agent_catalog_provider()
        if inspect.isawaitable(result):
            result = await result
        agents = list(result)
        if not agents:
            raise TaskAdaptiveAgentTeamFormationError("cold start agent catalog is empty")
        if any(not isinstance(agent, PublicAgentProfile) for agent in agents):
            raise TaskAdaptiveAgentTeamFormationError(
                "agent_catalog_provider must return PublicAgentProfile values"
            )
        agent_ids = [agent.agent_id for agent in agents]
        if any(not agent_id.strip() for agent_id in agent_ids):
            raise TaskAdaptiveAgentTeamFormationError("agent catalog contains an empty agent_id")
        if len(agent_ids) != len(set(agent_ids)):
            raise TaskAdaptiveAgentTeamFormationError("agent catalog contains duplicate agent IDs")
        return agents

    async def _call(
        self, messages: list[dict[str, str]], tool: dict[str, Any]
    ) -> Any:
        completion = self._completion
        if completion is None:
            from core.llm import complete

            completion = complete
        return await completion(
            model=self.model,
            messages=messages,
            tools=[tool],
            temperature=0.1,
            max_tokens=2048,
            api_tracker=self._api_tracker,
        )

    def _team_template_decision_error(self, payload: dict[str, Any]) -> str:
        action = str(payload.get("family_action") or "").strip().lower()
        family_id = str(payload.get("family_id") or "").strip()
        if action == "reuse":
            return "" if self.registry.get_current(family_id) else (
                "reuse must name an existing team template"
            )
        if action != "cold_start":
            return "family_action must be reuse or cold_start"
        if self.registry.get_family(family_id) is not None:
            return "cold_start requires a new team template_id"
        from core.evolution_models import validate_family_id

        try:
            validate_family_id(family_id)
        except ValueError as exc:
            return str(exc)
        if not str(payload.get("family_description") or "").strip():
            return "cold_start requires family_description"
        return ""

    @staticmethod
    def _extract_calls(response: Any, tool_name: str) -> list[dict[str, Any]]:
        from core.llm import extract_tool_calls

        return [
            dict(call.get("arguments") or {})
            for call in extract_tool_calls(response)
            if call.get("name") == tool_name
            and isinstance(call.get("arguments"), dict)
        ]

    @staticmethod
    def _team_template_tool_schema() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "select_agent_team_template",
                "description": (
                    "Choose an existing team template or request cold start."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "family_action": {
                            "type": "string",
                            "enum": ["reuse", "cold_start"],
                        },
                        "family_id": {"type": "string"},
                        "family_description": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["family_action", "family_id"],
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _cold_start_tool_schema() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "compose_cold_start_team",
                "description": "Select the provisional members for a new team template.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "provisional_agent_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["provisional_agent_ids"],
                    "additionalProperties": False,
                },
            },
        }


def validate_task_adaptive_team_formation(
    selection: TaskAdaptiveAgentTeamSelection,
    *,
    registry: TeamTemplateRegistry,
    known_agent_ids: set[str] | None,
    expected_chairman_id: str,
) -> TaskAdaptiveAgentTeamSelection:
    selection.validate(known_agent_ids)
    if selection.chairman_id != expected_chairman_id:
        raise ValueError("team selection cannot replace the configured chairman")
    current = registry.get_current(selection.family_id)
    if selection.family_action == "reuse":
        if current is None:
            raise ValueError("reuse requires an existing team template")
        if selection.template_id != current.template_id:
            raise ValueError("reuse must bind the family's current template")
        if selection.template_version != current.version:
            raise ValueError("reuse must bind the current template version")
        if selection.allowed_agent_ids != current.allowed_agent_ids:
            raise ValueError("reuse cannot change or reorder template members")
    elif selection.family_action == "create":
        if current is not None:
            raise ValueError("cold start must use a new team template_id")
    elif current is not None:
        raise ValueError("no_match cannot name an existing team template")
    return selection