import yaml
import asyncio
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from core import llm
from core.cost_tracker import CostTrackerMixin
from core.tool_registry import ToolRegistry, load_agent_skills, load_skill_content, Skill
from core.message_store import MessageStore

logger = logging.getLogger(__name__)



_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_bootstrap_instructions() -> str:
    path = _PROMPTS_DIR / "bootstrap_skills.md"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return (
        "### How to Use Skills — MANDATORY\n\n"
        "Check skills BEFORE any response or action. "
        "Call `use_skill(skill_name=...)` when a task matches a skill."
    )


@dataclass
class AgentResult:
    output: str
    steps: int = 0
    tool_calls_total: int = 0
    cost_seconds: float = 0.0
    cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentConfig:
    name: str
    role: str = "worker"
    description: str = ""
    model: str = "gpt-5.6-luna"
    temperature: float = 1.0
    max_tokens: int = 16384
    max_steps: int = 50
    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    tool_budgets: dict = field(default_factory=dict)


def load_config(agent_dir: str) -> AgentConfig:
    import os
    config_path = Path(agent_dir) / "config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = (
        os.environ.get("EVOTEAM_MODEL")
        or raw.get("model", "gpt-5.6-luna")
    )
    return AgentConfig(
        name=raw["name"],
        role=raw.get("role", "worker"),
        description=raw.get("description", ""),
        model=model,
        temperature=raw.get("temperature", 1.0),
        max_tokens=raw.get("max_tokens", 16384),
        max_steps=raw.get("max_steps", 50),
        tools=raw.get("tools", []),
        skills=raw.get("skills", []),
        tool_budgets=raw.get("tool_budgets", {}) or {},
    )


def load_prompt(agent_dir: str) -> str:
    prompt_path = Path(agent_dir) / "prompt.md"
    if not prompt_path.exists():
        return ""
    return prompt_path.read_text(encoding="utf-8").strip()


def load_prompt_patches(agent_dir: str) -> str:
    patch_path = Path(agent_dir) / "evolution" / "prompt_patches.md"
    if not patch_path.exists():
        return ""
    return patch_path.read_text(encoding="utf-8").strip()


def load_teammate_profiles(agent_dir: str) -> str:
    profile_path = Path(agent_dir) / "evolution" / "teammate_profiles.yaml"
    if not profile_path.exists():
        return ""
    return profile_path.read_text(encoding="utf-8").strip()


class Agent(CostTrackerMixin):

    def __init__(
        self,
        agent_dir: str,
        tool_registry: ToolRegistry,
        *,
        load_teammate_profiles_for_agent: bool = True,
    ):
        self.agent_dir = agent_dir
        self.tool_registry = tool_registry
        self.config = load_config(agent_dir)
        self.prompt = load_prompt(agent_dir)
        self.prompt_patches = load_prompt_patches(agent_dir)
        self.teammate_profiles = (
            load_teammate_profiles(agent_dir)
            if load_teammate_profiles_for_agent
            else ""
        )
        self.skills: list[Skill] = load_agent_skills(agent_dir)

        self.message_store: MessageStore | None = None

        self.messages: list[dict] = []
        self._recent_reasoning_texts: list[str] = []

        self._tool_call_counts: dict[str, int] = {}

        self._state: str = "idle"
        self._state_since: float = 0.0
        self._event_log = None
        self._idle_timeout: float = 60.0
        self._phase_step_budget: int | None = None
        self._reflection_enabled: bool = False
        self._total_cost_usd: float = 0.0

    def build_system_prompt(self) -> str:
        parts = []

        if self.prompt:
            parts.append(self.prompt)

        if self.skills:
            skill_entries = []
            for skill in self.skills:
                skill_entries.append(
                    f"  <skill>\n"
                    f"    <name>{skill.name}</name>\n"
                    f"    <description>{skill.description}</description>\n"
                    f"  </skill>"
                )
            skills_section = (
                "## Skills System\n\n"
                "<available_skills>\n"
                + "\n".join(skill_entries) + "\n"
                "</available_skills>\n\n"
                + _load_bootstrap_instructions()
            )
            parts.append(skills_section)

        if self.prompt_patches:
            parts.append(f"## Behavioral Patches\n\n{self.prompt_patches}")

        if self.teammate_profiles:
            parts.append(
                "## Teammate Profiles\n\n"
                "The following profiles are based on observations from multiple "
                "actual tasks. They record observed behavior only and do not "
                "define recruitment priority. Missing information means the "
                "teammate has not yet been observed, not that the teammate is "
                "less capable.\n\n"
                f"{self.teammate_profiles}"
            )

        return "\n\n---\n\n".join(parts)

    def get_skill(self, name: str) -> "Skill | None":
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    def _build_use_skill_schema(self) -> dict | None:
        if not self.skills:
            return None
        skill_names = [s.name for s in self.skills]
        return {
            "type": "function",
            "function": {
                "name": "use_skill",
                "description": (
                    "Load a skill's full instructions to guide your work. "
                    "Call this when a task matches a skill's description."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "enum": skill_names,
                            "description": "Name of the skill to load.",
                        },
                    },
                    "required": ["skill_name"],
                },
            },
        }

    def _handle_use_skill(self, skill_name: str) -> str:
        skill = self.get_skill(skill_name)
        if not skill:
            available = ", ".join(s.name for s in self.skills)
            return f"Error: skill '{skill_name}' not found. Available: {available}"
        content = load_skill_content(skill.path)
        return f"# Skill: {skill.name}\n\n{content}"

    def bind_message_store(self, store: MessageStore) -> None:
        self.message_store = store
        store.register(self.config.name)

    def _build_messaging_schemas(self) -> list[dict]:
        if not self.message_store:
            return []

        from tools.messaging import (
            build_send_message_schema,
            build_check_agent_status_schema,
            build_wait_for_replies_schema,
            READ_MESSAGES_SCHEMA,
        )

        other_agents = [
            n for n in self.message_store.registered_agents
            if n != self.config.name
        ]
        if not other_agents:
            return []

        runner_ref = getattr(self, "_runner_ref", None)
        handoff_rules = (
            getattr(runner_ref, "team_selection_metadata", {}).get("handoff_rules", [])
            if runner_ref is not None else []
        )
        handoff_rules = [
            rule for rule in handoff_rules
            if rule.get("from_agent") == self.config.name
        ]
        schemas = [
            {"type": "function", "function": build_send_message_schema(other_agents, handoff_rules)},
            {"type": "function", "function": READ_MESSAGES_SCHEMA},
            {"type": "function", "function": build_check_agent_status_schema(other_agents)},
            {"type": "function", "function": build_wait_for_replies_schema(other_agents)},
        ]
        return schemas

    def _handle_messaging_tool(self, tool_name: str, args: dict) -> str:
        from tools.messaging import (
            execute_send_message, execute_read_messages,
        )

        if tool_name == "send_message":
            return execute_send_message(
                to=args.get("to", ""),
                content=args.get("content", ""),
                _agent_name=self.config.name,
                _message_store=self.message_store,
                _runner=getattr(self, "_runner_ref", None),
                handoff_rule_id=args.get("handoff_rule_id"),
            )
        elif tool_name == "read_messages":
            return execute_read_messages(
                sender=args.get("sender"),
                unread_only=args.get("unread_only", True),
                limit=args.get("limit", 20),
                _agent_name=self.config.name,
                _message_store=self.message_store,
            )
        elif tool_name == "check_agent_status":
            try:
                runner_ref = getattr(self, '_runner_ref', None)
                if runner_ref:
                    from tools.messaging import execute_check_agent_status
                    return execute_check_agent_status(
                        agent_name=args.get("agent_name"),
                        _agent_name=self.config.name,
                        _message_store=self.message_store,
                        _agent_states=runner_ref.get_agent_states(),
                    )
            except Exception as e:
                return f"Error checking agent status: {e}"
            return "Agent status not available in this context."
        elif tool_name == "wait_for_replies":
            return "Error: wait_for_replies must be handled asynchronously."
        return f"Error: unknown messaging tool '{tool_name}'"

    _MESSAGING_TOOL_NAMES = frozenset({
        "send_message", "read_messages",
        "check_agent_status", "wait_for_replies",
    })

    async def _handle_wait_for_replies(self, args: dict) -> str:
        from tools.messaging import execute_wait_for_replies

        name = self.config.name
        reason = args.get("reason", "")
        from_agents = args.get("from_agents")
        event_log = getattr(self, "_event_log", None)
        idle_timeout = getattr(self, "_idle_timeout", 120.0)

        if self._state == "working":
            self._state = "idle"
            self._state_since = time.time()
            if event_log:
                event_log.log("agent.state", agent=name, data={
                    "from": "working", "to": "idle",
                    "trigger": "wait_for_replies",
                    "reason": reason,
                })

        if event_log:
            event_log.log("agent.loop.idle", agent=name, data={
                "step": -1,
                "reason": reason,
            })

        result = await execute_wait_for_replies(
            reason=reason,
            from_agents=from_agents,
            _agent_name=name,
            _message_store=self.message_store,
            _timeout=idle_timeout,
        )

        if from_agents and "No messages received" in result:
            runner_ref = getattr(self, '_runner_ref', None)
            if runner_ref:
                states = runner_ref.get_agent_states()
                still_working = [
                    a for a in from_agents
                    if states.get(a, {}).get("state") == "working"
                ]
                if still_working:
                    names = ", ".join(still_working)
                    result += (
                        f"\n\n⚠️ {names} is still WORKING. "
                        f"They are making progress — use wait_for_replies() again to keep waiting."
                    )

        if self._state == "idle" and not self.message_store.is_terminated():
            self._state = "working"
            self._state_since = time.time()
            if event_log:
                event_log.log("agent.state", agent=name, data={
                    "from": "idle", "to": "working",
                    "trigger": "wait_for_replies_returned",
                })

        return result

    def _build_all_tools(self, extra_tools: list[dict] | None = None) -> list[dict] | None:
        tools = self.tool_registry.to_openai_tools(self.config.tools)
        use_skill_schema = self._build_use_skill_schema()
        if use_skill_schema:
            tools.append(use_skill_schema)
        tools.extend(self._build_messaging_schemas())
        refresher = getattr(self, '_primitive_tools_refresher', None)
        if refresher:
            tools.extend(refresher())
        elif extra_tools:
            tools.extend(extra_tools)
        return tools or None

    async def _execute_tool_call(
        self, tc_name: str, tc_args: dict, cwd: str | None = None,
    ) -> str:
        if tc_name == "use_skill":
            return self._handle_use_skill(tc_args.get("skill_name", ""))

        if tc_name == "wait_for_replies":
            return await self._handle_wait_for_replies(tc_args)

        if tc_name in self._MESSAGING_TOOL_NAMES:
            return self._handle_messaging_tool(tc_name, tc_args)

        from tools.primitives import ALL_TOOL_NAMES, get_handler
        if tc_name in ALL_TOOL_NAMES:
            if "_parse_error" in tc_args:
                return f"Error: invalid arguments for '{tc_name}': {tc_args['_parse_error']}"
            runner_ref = getattr(self, '_runner_ref', None)
            handler = get_handler(runner_ref, tc_name) if runner_ref else None
            if handler:
                result = await handler(**tc_args, _caller_name=self.config.name)
                return str(result)
            return f"Error: primitive tool '{tc_name}' has no handler (runner not bound)"

        if not self.config.tools or tc_name not in self.config.tools:
            allowed = self.config.tools or []
            return (
                f"Error: tool '{tc_name}' is not in your allowed tools list. "
                f"Allowed: {allowed}"
            )

        budget = (self.config.tool_budgets or {}).get(tc_name, 0)
        if budget and budget > 0:
            used = self._tool_call_counts.get(tc_name, 0)
            if used >= budget:
                if hasattr(self, "_event_log") and self._event_log:
                    self._event_log.log("agent.tool_budget_exceeded",
                        agent=self.config.name, data={
                            "tool": tc_name,
                            "used": used,
                            "budget": budget,
                        })
                return (
                    f"[quota exceeded: {tc_name} already used {used} times "
                    f"(budget={budget}). Do NOT call {tc_name} again. "
                    f"Based on all the information you have already gathered, "
                    f"provide your final answer NOW. If you are the Chairman, "
                    f"follow the Runtime Submission Protocol in your context with "
                    f"the complete final answer as the `output` argument.]"
                )

        tool = self.tool_registry.get(tc_name)
        if not tool:
            return f"Error: tool '{tc_name}' not found"
        self._tool_call_counts[tc_name] = self._tool_call_counts.get(tc_name, 0) + 1
        try:
            args = dict(tc_args)
            if "_parse_error" in args:
                return f"Error: invalid arguments for '{tc_name}': {args['_parse_error']}"
            if cwd and "cwd" not in args:
                args["cwd"] = cwd
            result = tool.execute(**args)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    async def run(self, task: str, event_log=None, cwd: str | None = None) -> AgentResult:
        name = self.config.name
        start = time.time()

        logger.info(
            "[%s] run() started: model=%s, max_steps=%d",
            name, self.config.model, self.config.max_steps,
        )

        if event_log:
            event_log.log("agent.start", agent=name, data={"task": task[:500]})

        system = self.build_system_prompt()
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        openai_tools = self._build_all_tools()

        steps = 0
        tool_calls_total = 0

        while steps < self.config.max_steps:
            steps += 1

            response = await llm.complete(
                model=self.config.model,
                messages=messages,
                tools=openai_tools,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_tracker=getattr(getattr(self, "_runner_ref", None), "api_tracker", None),
            )

            self._track_cost_usd(response)

            if event_log:
                usage = getattr(response, "usage", None)
                event_log.log("llm.call", agent=name, data={
                    "model": self.config.model,
                    "step": steps,
                    "has_tool_calls": llm.has_tool_calls(response),
                    "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "cost_usd": round(self._total_cost_usd, 4),
                })

            if not llm.has_tool_calls(response):
                output = llm.extract_text(response)
                if event_log:
                    event_log.log("agent.think", agent=name, data={
                        "content": output,
                        "step": steps,
                    })
                    event_log.log("agent.finish", agent=name, data={
                        "output": output[:1000],
                        "steps": steps,
                        "tool_calls_total": tool_calls_total,
                        "cost_seconds": round(time.time() - start, 2),
                        "cost_usd": round(self._total_cost_usd, 4),
                    })
                elapsed = time.time() - start
                logger.info(
                    "[%s] run() finished: %.1fs elapsed, %d steps, "
                    "%d tool_calls, $%.4f cost",
                    name, elapsed, steps, tool_calls_total,
                    self._total_cost_usd,
                )
                return AgentResult(
                    output=output,
                    steps=steps,
                    tool_calls_total=tool_calls_total,
                    cost_seconds=elapsed,
                    cost_usd=self._total_cost_usd,
                )

            assistant_msg = llm.response_to_message(response)
            messages.append(assistant_msg)

            reasoning_text = llm.extract_text(response)
            if reasoning_text:
                self._recent_reasoning_texts.append(reasoning_text)
                if len(self._recent_reasoning_texts) > 8:
                    self._recent_reasoning_texts.pop(0)
            if event_log and reasoning_text:
                event_log.log("agent.think", agent=name, data={
                    "content": reasoning_text,
                    "step": steps,
                })

            tool_calls = llm.extract_tool_calls(response)
            tool_calls_total += len(tool_calls)

            for tc in tool_calls:
                tc_name = tc["name"]
                tc_args = tc["arguments"]

                if event_log:
                    event_log.log("tool.call", agent=name, data={
                        "tool": tc_name, "args": tc_args,
                    })

                result_str = await self._execute_tool_call(tc_name, tc_args, cwd)

                if event_log:
                    event_log.log("tool.result", agent=name, data={
                        "tool": tc_name, "output": result_str[:10000],
                    })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

        elapsed = time.time() - start
        logger.info(
            "[%s] run() finished: %.1fs elapsed, %d steps, "
            "%d tool_calls, $%.4f cost, stopped=max_steps",
            name, elapsed, steps, tool_calls_total,
            self._total_cost_usd,
        )
        if event_log:
            event_log.log("agent.finish", agent=name, data={
                "output": "[max steps reached]",
                "steps": steps,
                "tool_calls_total": tool_calls_total,
                "cost_seconds": round(elapsed, 2),
                "cost_usd": round(self._total_cost_usd, 4),
                "stopped": "max_steps",
            })
        return AgentResult(
            output="[max steps reached]",
            steps=steps,
            tool_calls_total=tool_calls_total,
            cost_seconds=elapsed,
            cost_usd=self._total_cost_usd,
            metadata={"stopped": "max_steps"},
        )

    async def run_loop(
        self,
        event_log=None,
        cwd: str | None = None,
        initial_task: str | None = None,
        system_context: str | None = None,
        extra_tools: list[dict] | None = None,
        idle_timeout: float = 120.0,
        max_idle_rounds: int = 3,
        max_steps: int | None = None,
    ) -> AgentResult:
        name = self.config.name
        effective_max_steps = max_steps if max_steps is not None else self.config.max_steps
        start = time.time()

        logger.info(
            "[%s] run_loop() started: model=%s, max_steps=%d, "
            "has_initial_task=%s, idle_timeout=%.0fs",
            name, self.config.model, effective_max_steps,
            bool(initial_task), idle_timeout,
        )

        self._state = "idle"
        self._state_since = time.time()
        self._event_log = event_log
        self._idle_timeout = idle_timeout

        if event_log:
            event_log.log("agent.loop.start", agent=name, data={
                "initial_task": initial_task[:500] if initial_task else "",
                "system_context": system_context[:500] if system_context else "",
                "has_system_context": bool(system_context),
                "initial_state": "working" if initial_task else "idle",
            })

        system = self.build_system_prompt()
        if system_context:
            system += "\n\n---\n\n" + system_context
        messages: list[dict] = [
            {"role": "system", "content": system},
        ]
        self.messages = messages

        if initial_task:
            self._append_conversation_message(
                messages, {"role": "user", "content": initial_task}
            )
            self._state = "working"
            self._state_since = time.time()
            if event_log:
                event_log.log("agent.state", agent=name, data={
                    "from": "idle", "to": "working", "trigger": "initial_task",
                })

        openai_tools = self._build_all_tools(extra_tools)
        self._extra_tools = extra_tools

        steps = 0
        tool_calls_total = 0
        consecutive_idle = 0
        max_steps_resume_used = False
        last_output = ""

        try:
            while True:
                if self._phase_step_budget is not None:
                    budget = self._phase_step_budget
                    self._phase_step_budget = None
                    steps = 0
                    effective_max_steps = budget
                    consecutive_idle = 0
                    if event_log:
                        event_log.log("agent.loop.budget_reset", agent=name, data={
                            "new_max_steps": budget,
                        })

                if self.message_store and self.message_store.is_terminated():
                    if event_log:
                        event_log.log("agent.loop.terminated", agent=name, data={
                            "reason": self.message_store.terminate_reason,
                            "steps": steps,
                        })
                    break

                if steps >= effective_max_steps:
                    if self._reflection_enabled:
                        if self._state != "idle":
                            self._state = "idle"
                            self._state_since = time.time()
                            if event_log:
                                event_log.log("agent.state", agent=name, data={
                                    "from": "working", "to": "idle",
                                    "trigger": "max_steps_reflection_wait",
                                })
                        if self.message_store:
                            has_msg = await self.message_store.wait_for_message(
                                name, timeout=idle_timeout
                            )
                            if self.message_store.is_terminated():
                                break
                            if has_msg:
                                self._inject_new_messages(messages, event_log)
                                consecutive_idle = 0
                                self._state = "working"
                                self._state_since = time.time()
                                runner_ref = getattr(self, "_runner_ref", None)
                                is_chairman = (
                                    runner_ref is not None
                                    and name == getattr(runner_ref, "chairman_name", None)
                                )
                                is_task_execution = (
                                    runner_ref is None
                                    or getattr(runner_ref, "_phase", "task_execution")
                                    == "task_execution"
                                )
                                if (
                                    is_chairman
                                    and is_task_execution
                                    and not max_steps_resume_used
                                ):
                                    steps = max(0, steps - 5)
                                    max_steps_resume_used = True
                                if event_log:
                                    event_log.log("agent.state", agent=name, data={
                                        "from": "idle", "to": "working",
                                        "trigger": "message_received",
                                    })
                            continue
                        else:
                            break
                    else:
                        break

                if not messages[-1:] or messages[-1]["role"] == "system":
                    if self.message_store:
                        has_msg = await self.message_store.wait_for_message(name, timeout=idle_timeout)
                        if self.message_store.is_terminated():
                            break
                        if not has_msg:
                            consecutive_idle += 1
                            if consecutive_idle >= max_idle_rounds and not self._reflection_enabled:
                                break
                            continue
                        self._inject_new_messages(messages, event_log)
                        consecutive_idle = 0
                        if self._state == "idle":
                            self._state = "working"
                            self._state_since = time.time()
                            if event_log:
                                event_log.log("agent.state", agent=name, data={
                                    "from": "idle", "to": "working",
                                    "trigger": "message_received",
                                })
                    else:
                        break

                steps += 1

                _MAX_CONTEXT_MESSAGES = 150
                if len(messages) > _MAX_CONTEXT_MESSAGES + 1:
                    system_msg = messages[0]
                    initial_task_msg = None
                    if len(messages) > 1 and messages[1].get("role") == "user":
                        initial_task_msg = messages[1]
                    cut_start = len(messages) - _MAX_CONTEXT_MESSAGES
                    if initial_task_msg and cut_start <= 1:
                        cut_start = 2
                    min_cut = 2 if initial_task_msg else 1
                    while cut_start > min_cut:
                        msg = messages[cut_start]
                        role = msg.get("role", "")
                        if role == "tool":
                            cut_start -= 1
                        elif role == "assistant" and msg.get("tool_calls"):
                            break
                        else:
                            break
                    while cut_start < len(messages) and messages[cut_start].get("role") == "tool":
                        cut_start += 1
                    recent = messages[cut_start:]
                    truncated_count = cut_start - (2 if initial_task_msg else 1)
                    preserved = [system_msg]
                    if initial_task_msg:
                        preserved.append(initial_task_msg)
                    preserved.append({
                        "role": "user",
                        "content": (
                            f"[Context trimmed: {truncated_count} earlier messages removed "
                            f"to fit context window. Recent conversation follows.]"
                        ),
                    })
                    messages = preserved + recent
                    self.messages = messages
                    if event_log:
                        event_log.log("agent.loop.context_trimmed", agent=name, data={
                            "truncated_count": truncated_count,
                            "remaining": len(messages),
                            "initial_task_preserved": initial_task_msg is not None,
                        })

                openai_tools = self._build_all_tools(self._extra_tools)

                response = await llm.complete(
                    model=self.config.model,
                    messages=messages,
                    tools=openai_tools,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    api_tracker=getattr(getattr(self, "_runner_ref", None), "api_tracker", None),
                )

                self._track_cost_usd(response)

                if event_log:
                    usage = getattr(response, "usage", None)
                    event_log.log("llm.call", agent=name, data={
                        "model": self.config.model,
                        "step": steps,
                        "mode": "loop",
                        "has_tool_calls": llm.has_tool_calls(response),
                        "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                        "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                        "cost_usd": round(self._total_cost_usd, 4),
                    })

                if not llm.has_tool_calls(response):
                    text = llm.extract_text(response)
                    self._append_conversation_message(
                        messages, {"role": "assistant", "content": text},
                        execution_message=llm.response_to_message(response),
                    )
                    last_output = text

                    runner_ref = getattr(self, "_runner_ref", None)
                    contract = getattr(runner_ref, "output_contract", None)
                    answer_prefix = getattr(contract, "answer_agent_prefix", "")
                    is_answer_service = (
                        runner_ref is not None
                        and self.config.name == getattr(
                            contract, "answer_agent_name", "answer_agent",
                        )
                        and bool(answer_prefix)
                        and text.lstrip().lower().startswith(answer_prefix.lower())
                    )
                    if is_answer_service and self.message_store:
                        chairman_name = runner_ref.chairman_name
                        self.message_store.send(
                            sender=self.config.name,
                            receiver=chairman_name,
                            content=text.strip(),
                        )
                        if event_log:
                            event_log.log("agent.answer_auto_delivered", agent=name, data={
                                "receiver": chairman_name,
                                "prefix": answer_prefix,
                            })

                    if self._state == "working":
                        self._state = "idle"
                        self._state_since = time.time()
                        if event_log:
                            event_log.log("agent.state", agent=name, data={
                                "from": "working", "to": "idle",
                                "trigger": "no_tool_calls",
                            })

                    if event_log:
                        event_log.log("agent.think", agent=name, data={
                            "content": text,
                            "step": steps,
                        })
                        event_log.log("agent.loop.idle", agent=name, data={
                            "step": steps,
                        })

                    if self.message_store:
                        has_msg = await self.message_store.wait_for_message(
                            name, timeout=idle_timeout
                        )
                        if self.message_store.is_terminated():
                            break
                        if has_msg:
                            self._inject_new_messages(messages, event_log)
                            consecutive_idle = 0
                            self._state = "working"
                            self._state_since = time.time()
                            if event_log:
                                event_log.log("agent.state", agent=name, data={
                                    "from": "idle", "to": "working",
                                    "trigger": "message_received",
                                })
                        else:
                            consecutive_idle += 1
                            if consecutive_idle >= max_idle_rounds and not self._reflection_enabled:
                                break
                            self._append_conversation_message(messages, {
                                "role": "user",
                                "content": (
                                    f"[idle timeout — no new messages after {int(idle_timeout)}s. "
                                    f"Idle count: {consecutive_idle}/{max_idle_rounds}. "
                                    f"Use wait_for_replies() to continue waiting, "
                                    f"check_agent_status() to see if agents are still working, "
                                    f"or proceed with available information.]"
                                ),
                            })
                    else:
                        break
                    continue

                consecutive_idle = 0
                assistant_msg = llm.response_to_message(response)
                self._append_conversation_message(messages, assistant_msg)

                reasoning_text = llm.extract_text(response)
                if reasoning_text:
                    self._recent_reasoning_texts.append(reasoning_text)
                    if len(self._recent_reasoning_texts) > 8:
                        self._recent_reasoning_texts.pop(0)
                if event_log and reasoning_text:
                    event_log.log("agent.think", agent=name, data={
                        "content": reasoning_text,
                        "step": steps,
                    })

                tool_calls = llm.extract_tool_calls(response)
                tool_calls_total += len(tool_calls)

                for tc in tool_calls:
                    tc_name = tc["name"]
                    tc_args = tc["arguments"]

                    if event_log:
                        event_log.log("tool.call", agent=name, data={
                            "tool": tc_name, "args": tc_args,
                        })

                    result_str = await self._execute_tool_call(tc_name, tc_args, cwd)

                    if event_log:
                        event_log.log("tool.result", agent=name, data={
                            "tool": tc_name, "output": result_str[:10000],
                        })

                    self._append_conversation_message(messages, {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_str,
                    })
        except asyncio.CancelledError:
            if event_log:
                event_log.log("agent.loop.cancelled", agent=name, data={
                    "steps": steps,
                    "cost_seconds": round(time.time() - start, 2),
                })
            raise
        except Exception as exc:
            if event_log:
                event_log.log("agent.loop.error", agent=name, data={
                    "error": str(exc)[:500],
                    "error_type": type(exc).__name__,
                    "steps": steps,
                    "cost_seconds": round(time.time() - start, 2),
                })
            last_output = f"[agent error: {type(exc).__name__}: {str(exc)[:200]}]"

        stopped_reason = None
        if steps >= effective_max_steps:
            stopped_reason = "max_steps"

        elapsed = time.time() - start
        logger.info(
            "[%s] run_loop() finished: %.1fs elapsed, %d steps, "
            "%d tool_calls, $%.4f cost%s",
            name, elapsed, steps, tool_calls_total,
            self._total_cost_usd,
            f", stopped={stopped_reason}" if stopped_reason else "",
        )

        if event_log:
            log_data = {
                "steps": steps,
                "tool_calls_total": tool_calls_total,
                "cost_seconds": round(time.time() - start, 2),
                "cost_usd": round(self._total_cost_usd, 4),
            }
            if stopped_reason:
                log_data["stopped"] = stopped_reason
            event_log.log("agent.loop.end", agent=name, data=log_data)

        metadata = {"stopped": stopped_reason} if stopped_reason else {}
        return AgentResult(
            output=last_output,
            steps=steps,
            tool_calls_total=tool_calls_total,
            cost_seconds=time.time() - start,
            cost_usd=self._total_cost_usd,
            metadata=metadata,
        )

    def _append_conversation_message(
        self, messages: list[dict], message: dict,
        *, execution_message: dict | None = None,
    ) -> None:
        messages.append(message)
        runner = getattr(self, "_runner_ref", None)
        if runner is not None:
            runner._record_execution_message(
                self.config.name,
                message if execution_message is None else execution_message,
            )

    def _inject_new_messages(self, messages: list[dict], event_log=None) -> None:
        if not self.message_store:
            return

        unread = self.message_store.read(reader=self.config.name, unread_only=True, limit=50)
        if not unread:
            return

        parts = []
        senders = []
        for msg in unread:
            parts.append(
                f"**[{msg.sender}]**: {msg.content}"
            )
            if msg.sender not in senders:
                senders.append(msg.sender)
        content = "## New Messages\n\n" + "\n\n---\n\n".join(parts)
        remaining = self.message_store.count_unread(self.config.name)
        if remaining > 0:
            content += f"\n\n[{remaining} more unread message(s) — call read_messages() to see them]"
        self._append_conversation_message(
            messages, {"role": "user", "content": content}
        )

        if event_log:
            event_log.log("agent.wake", agent=self.config.name, data={
                "senders": senders,
                "count": len(unread),
                "content": content,
            })
