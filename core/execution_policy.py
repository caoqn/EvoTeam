from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class FallbackPolicy:
    trigger: str
    mode: str
    instructions: tuple[str, ...] = ()
    official_score_eligible: bool = False
    evolution_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger,
            "mode": self.mode,
            "instructions": list(self.instructions),
            "official_score_eligible": self.official_score_eligible,
            "evolution_eligible": self.evolution_eligible,
        }


@dataclass(frozen=True)
class ExecutionPolicy:
    name: str
    version: str
    environment_mode: str
    allowed_tools: tuple[str, ...]
    artifact_requirements: tuple[str, ...] = ()
    chairman_instructions: tuple[str, ...] = ()
    special_rules: tuple[str, ...] = ()
    role_instructions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    role_instruction_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    fallback_policy: FallbackPolicy | None = None
    infrastructure_failure_conditions: tuple[str, ...] = ()
    official_score_eligible: bool = True
    evolution_eligible: bool = True
    collaboration_prior_mode: str = "native"

    @property
    def policy_id(self) -> str:
        return f"{self.name}@{self.version}"

    def contract_only(self) -> "ExecutionPolicy":
        return replace(
            self,
            chairman_instructions=(),
            role_instructions={},
            role_instruction_aliases={},
            collaboration_prior_mode="contract_only",
        )

    def bind(
        self,
        *,
        execution_mode: str | None = None,
        available_tools: tuple[str, ...] | None = None,
        workspace_paths: dict[str, str] | None = None,
        runtime_instructions: tuple[str, ...] = (),
        fallback_used: bool = False,
        fallback_paths: dict[str, list[str]] | None = None,
    ) -> "BoundExecutionPolicy":
        fallback = self.fallback_policy if fallback_used else None
        return BoundExecutionPolicy(
            specification=self,
            execution_mode=execution_mode or self.environment_mode,
            available_tools=available_tools or self.allowed_tools,
            workspace_paths=dict(workspace_paths or {}),
            runtime_instructions=runtime_instructions,
            fallback_used=fallback_used,
            fallback_paths=dict(fallback_paths or {}),
            official_score_eligible=(
                fallback.official_score_eligible
                if fallback is not None else self.official_score_eligible
            ),
            evolution_eligible=(
                fallback.evolution_eligible
                if fallback is not None else self.evolution_eligible
            ),
        )


@dataclass(frozen=True)
class BoundExecutionPolicy:
    specification: ExecutionPolicy
    execution_mode: str
    available_tools: tuple[str, ...]
    workspace_paths: dict[str, str] = field(default_factory=dict)
    runtime_instructions: tuple[str, ...] = ()
    fallback_used: bool = False
    fallback_paths: dict[str, list[str]] = field(default_factory=dict)
    official_score_eligible: bool = True
    evolution_eligible: bool = True

    def chairman_context(self) -> str:
        lines = [
            "## Current Native Execution Policy",
            "",
            f"Policy version: `{self.specification.version}`",
            f"Environment mode: `{self.execution_mode}`",
            "Available tools: " + (
                ", ".join(f"`{tool}`" for tool in self.available_tools) or "none"
            ),
        ]
        if self.workspace_paths:
            lines.extend(["", "Workspace paths:"])
            lines.extend(
                f"- {label}: `{path}`"
                for label, path in sorted(self.workspace_paths.items())
            )
        if self.specification.artifact_requirements:
            lines.extend(["", "Required artifacts/postconditions:"])
            lines.extend(f"- {rule}" for rule in self.specification.artifact_requirements)
        instructions = (
            self.specification.chairman_instructions
            + self.specification.special_rules
            + self.runtime_instructions
        )
        if instructions:
            lines.extend(["", "Execution rules:"])
            lines.extend(f"- {rule}" for rule in instructions)
        if self.fallback_used:
            lines.extend([
                "",
                "Fallback status:",
                f"- Active mode: `{self.execution_mode}`",
                "- This run is not eligible for official scoring or shared evolution.",
            ])
            for service, paths in sorted(self.fallback_paths.items()):
                lines.append(f"- {service}: {', '.join(f'`{path}`' for path in paths)}")
        elif self.specification.fallback_policy is not None and self.fallback_paths:
            lines.extend([
                "",
                "Conditional fallback:",
                f"- Trigger: `{self.specification.fallback_policy.trigger}`",
                f"- Mode: `{self.specification.fallback_policy.mode}`",
                "- Use only after the declared fatal trigger; ordinary task and parameter errors do not qualify.",
            ])
            for service, paths in sorted(self.fallback_paths.items()):
                lines.append(f"- {service}: {', '.join(f'`{path}`' for path in paths)}")
        return "\n".join(lines)

    def instructions_for(self, agent_name: str, role: str) -> tuple[str, ...]:
        rules: list[str] = []
        keys: list[str] = ["*"]
        for identity in (role, agent_name):
            keys.extend(self.specification.role_instruction_aliases.get(identity, ()))
            keys.append(identity)
        for key in dict.fromkeys(keys):
            rules.extend(self.specification.role_instructions.get(key, ()))
        return tuple(dict.fromkeys(rules))

    def agent_context(
        self,
        agent_name: str,
        role: str,
        agent_tools: tuple[str, ...] | list[str] | None = None,
    ) -> str:
        rules = self.instructions_for(agent_name, role)
        available_tools = self.available_tools
        if agent_tools is not None:
            declared = set(agent_tools)
            available_tools = tuple(
                tool for tool in self.available_tools if tool in declared
            )
        lines = [
            "## Current Native Execution Policy",
            "",
            f"Environment mode: `{self.execution_mode}`",
            "Available tools: " + (
                ", ".join(f"`{tool}`" for tool in available_tools) or "none"
            ),
        ]
        if self.workspace_paths:
            lines.extend(["", "Workspace paths:"])
            lines.extend(
                f"- {label}: `{path}`"
                for label, path in sorted(self.workspace_paths.items())
            )
        if self.specification.artifact_requirements:
            lines.extend(["", "Required artifacts/postconditions:"])
            lines.extend(
                f"- {rule}" for rule in self.specification.artifact_requirements
            )
        common_rules = self.specification.special_rules + self.runtime_instructions
        if common_rules:
            lines.extend(["", "Execution rules:"])
            lines.extend(f"- {rule}" for rule in common_rules)
        if self.fallback_used:
            lines.extend([
                "",
                "Fallback status:",
                f"- Active mode: `{self.execution_mode}`",
                "- This run is not eligible for official scoring or shared evolution.",
            ])
            for service, paths in sorted(self.fallback_paths.items()):
                lines.append(f"- {service}: {', '.join(f'`{path}`' for path in paths)}")
        elif self.specification.fallback_policy is not None and self.fallback_paths:
            lines.extend([
                "",
                "Conditional fallback:",
                f"- Trigger: `{self.specification.fallback_policy.trigger}`",
                f"- Mode: `{self.specification.fallback_policy.mode}`",
                "- Use only after the declared fatal trigger; ordinary task and parameter errors do not qualify.",
            ])
            for service, paths in sorted(self.fallback_paths.items()):
                lines.append(f"- {service}: {', '.join(f'`{path}`' for path in paths)}")
        if rules:
            lines.extend(["", "Native role rules:"])
            lines.extend(f"- {rule}" for rule in rules)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.specification.policy_id,
            "collaboration_prior_mode": (
                self.specification.collaboration_prior_mode
            ),
            "execution_mode": self.execution_mode,
            "available_tools": list(self.available_tools),
            "workspace_paths": dict(self.workspace_paths),
            "runtime_instructions": list(self.runtime_instructions),
            "role_instruction_aliases": {
                key: list(value)
                for key, value in self.specification.role_instruction_aliases.items()
            },
            "chairman_instruction_count": len(
                self.specification.chairman_instructions
            ),
            "role_instruction_keys": sorted(
                self.specification.role_instructions
            ),
            "fallback_used": self.fallback_used,
            "fallback_paths": dict(self.fallback_paths),
            "official_score_eligible": self.official_score_eligible,
            "evolution_eligible": self.evolution_eligible,
        }
