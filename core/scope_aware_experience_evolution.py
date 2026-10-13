import asyncio
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from core.utils import load_prompt_template as _load_prompt_template

logger = logging.getLogger(__name__)


class ScopeAwareExperienceEvolutionMixin:



    def _record_execution_message(
        self, agent_id: str, message: dict[str, Any]
    ) -> None:
        if (
            self._phase != "task_execution"
            or self._composite_execution_trace is not None
            or agent_id in self._global_service_agent_ids
            or message.get("role") == "system"
        ):
            return
        self._execution_trace_sequence += 1
        self._execution_agent_traces.setdefault(agent_id, []).append({
            "sequence": self._execution_trace_sequence,
            "timestamp": time.time(),
            **deepcopy(message),
        })

    def _freeze_execution_trace(self, event_log=None) -> None:
        if self._composite_execution_trace is not None:
            return
        self._freeze_handoff_trace(event_log)
        participant_ids = [self.chairman_name, *self._recruited_agent_ids]
        handoffs = [
            {
                "message_id": message.id,
                "from_agent": message.sender,
                "to_agent": message.receiver,
                "content": message.content,
                "handoff_rule_id": message.handoff_rule_id,
                "timestamp": message.timestamp,
            }
            for message in self.message_store.iter_all_messages()
            if message.sender != "[SYSTEM]"
            and message.receiver != "[SYSTEM]"
            and message.sender not in self._global_service_agent_ids
            and message.receiver not in self._global_service_agent_ids
        ]
        self._composite_execution_trace = {
            "phase": "task_execution",
            "agent_traces": {
                agent_id: deepcopy(self._execution_agent_traces.get(agent_id, []))
                for agent_id in dict.fromkeys(participant_ids)
                if agent_id not in self._global_service_agent_ids
            },
            "handoff_trace": handoffs,
            "handoff_rule_usage": [
                {
                    key: item[key]
                    for key in ("message_id", "handoff_rule_id", "from_agent", "to_agent")
                }
                for item in handoffs
                if item["handoff_rule_id"]
            ],
        }
        if event_log:
            event_log.log("runner.composite_execution_trace_frozen", data={
                "agent_message_counts": {
                    agent_id: len(trace)
                    for agent_id, trace in self._composite_execution_trace[
                        "agent_traces"
                    ].items()
                },
                "handoff_count": len(handoffs),
            })

    def _freeze_handoff_trace(self, event_log=None) -> None:
        self._handoff_trace = [
            {
                "message_id": message.id,
                "from_agent": message.sender,
                "to_agent": message.receiver,
                "content": message.content[:4000],
                "handoff_rule_id": getattr(message, "handoff_rule_id", None),
                "timestamp": message.timestamp,
            }
            for message in self.message_store.iter_all_messages()
            if message.sender != "[SYSTEM]"
            and message.receiver != "[SYSTEM]"
        ]
        self._handoff_rule_usage = [
            {
                "message_id": item["message_id"],
                "handoff_rule_id": item["handoff_rule_id"],
                "from_agent": item["from_agent"],
                "to_agent": item["to_agent"],
            }
            for item in self._handoff_trace
            if item.get("handoff_rule_id")
        ]
        if event_log:
            event_log.log("runner.handoff_trace_frozen", data={
                "message_count": len(self._handoff_trace),
                "rule_usage_count": len(self._handoff_rule_usage),
            })

    async def set_final_output(self, output: str) -> str:
        if self._final_output_locked:
            return "Error: final task output is locked and cannot be overwritten during reflection."
        if self._phase != "task_execution":
            return "Error: final task output can only be set during task execution."
        self._result = output
        self._final_output_locked = True
        if self._finalization_mode == "none":
            self._finalization_mode = "chairman"
        event_log = self._event_log

        if self._task_validator:
            try:
                self._task_validation = await self._task_validator(output)
                if event_log:
                    event_log.log("runner.task_validation", data={
                        "success": self._task_validation.success,
                        "summary": self._task_validation.summary,
                    })
            except Exception as e:
                if event_log:
                    event_log.log("runner.task_validation_error", data={
                        "error": str(e),
                    })
                self._task_validation = None

        return "Output recorded successfully."

    async def finalize_task(self, output: str) -> str:
        if self._phase != "task_execution":
            return (
                "Error: finalize_task has already been called. "
                f"Current phase: {self._phase}. "
                "Continue with the current reflection phase tools."
            )

        self._result = output
        self._final_output_locked = True
        if self._finalization_mode == "none":
            self._finalization_mode = "chairman"
        event_log = self._event_log

        if self._task_validator:
            try:
                self._task_validation = await self._task_validator(output)
                if event_log:
                    event_log.log("runner.task_validation", data={
                        "success": self._task_validation.success,
                        "summary": self._task_validation.summary,
                    })
            except Exception as e:
                if event_log:
                    event_log.log("runner.task_validation_error", data={
                        "error": str(e),
                    })
                self._task_validation = None

        hook = getattr(self, '_pre_reflection_hook', None)
        if hook:
            try:
                hook()
            except Exception as e:
                if event_log:
                    event_log.log("runner.pre_reflection_hook_error", data={"error": str(e)})

        self._freeze_execution_trace(event_log)

        self._phase = "l1_reflection"
        self._all_reflection_agents = list(self.agents.keys())

        for name, agent in self.agents.items():
            if name not in self._agent_dirs and hasattr(agent, 'agent_dir'):
                self._agent_dirs[name] = str(agent.agent_dir)

        if event_log:
            event_log.log("runner.phase_change", data={
                "from": "task_execution",
                "to": "l1_reflection",
                "task_output": output[:200],
                "reflection_agents": self._all_reflection_agents,
            })

        l1_prompt = _load_prompt_template("reflection/l1_agent.md")
        if not l1_prompt:
            l1_prompt = (
                "Reflect on your own performance. Use update_prompt_patch / "
                "update_skill / skip_l1_reflection."
            )
        self._broadcast_l1_start(l1_prompt)

        self._restart_all_agents_for_phase("l1_reflection")

        self._start_reflection_phase_guard("l1_reflection")

        overview = _load_prompt_template("reflection/overview.md") or ""
        reflection_ctx = self._build_reflection_context()
        return (
            "Task finalized. Output recorded.\n\n"
            + (overview + "\n\n" if overview else "")
            + reflection_ctx + "\n\n"
            "---\n\n"
            "**ENTERING L1 REFLECTION PHASE**\n\n"
            + l1_prompt + "\n\n"
            "Complete your L1 reflection using the tools above, then wait. "
            "The system will automatically advance to L2 once all agents finish L1."
        )

    def terminate(self, reason: str = "chairman_terminate") -> None:
        if self._done.is_set():
            return
        self.message_store.terminate(reason=reason)
        self._done.set()


    def _check_and_advance_phase(self, completed_level: str, agent_name: str) -> str:
        self._auto_complete_exited_agents(completed_level)

        _completed_map = {"L1": self._l1_completed, "L2": self._l2_completed}
        completed = _completed_map.get(completed_level, set())
        reflection_agents = self._agents_for_reflection_level(completed_level)
        total = len(reflection_agents)
        remaining = [n for n in reflection_agents if n not in completed]

        if remaining:
            return (
                f"\n\n{completed_level} progress: {len(completed)}/{total} agents completed. "
                f"Remaining: {', '.join(remaining)}"
            )

        advance_result = self._auto_advance_phase(completed_level)
        return f"\n\nAll agents completed {completed_level}. {advance_result}"

    def _agents_for_reflection_level(self, completed_level: str) -> list[str]:
        if completed_level == "L2":
            return list(getattr(self, "_l2_reflection_agents", []))
        return list(self._all_reflection_agents)

    def _auto_advance_phase(self, completed_level: str) -> str:
        event_log = self._event_log

        if completed_level == "L1" and self._phase == "l1_reflection":
            self._l2_reflection_agents = self.l2_reflection_agents()
            if not self._l2_reflection_agents:
                self._phase = "terminated"
                if event_log:
                    event_log.log("runner.l1_auto_terminate", data={
                        "l1_completed": list(self._l1_completed),
                        "disabled_phase": "l2_teammate_profiles",
                    })
                self.message_store.terminate(reason="l1_all_completed_l2_disabled")
                self._done.set()
                return (
                    "All agents completed L1 reflection. L2 teammate profiles "
                    "are disabled; session ending."
                )
            self._phase = "l2_reflection"
            if event_log:
                event_log.log("runner.phase_change", data={
                    "from": "l1_reflection",
                    "to": "l2_reflection",
                    "l1_completed": list(self._l1_completed),
                    "reflection_agents": self._l2_reflection_agents,
                })
            self._restart_all_agents_for_phase("l2_reflection")
            self._broadcast_l2_start()
            self._start_reflection_phase_guard("l2_reflection")
            return "Auto-advancing to L2 reflection phase."

        elif completed_level == "L2" and self._phase == "l2_reflection":
            self._phase = "terminated"
            if event_log:
                event_log.log("runner.l2_auto_terminate", data={
                    "l2_completed": list(self._l2_completed),
                    "next_stage": "adapter_level_reflectors",
                })
            self.message_store.terminate(reason="l2_all_completed")
            self._done.set()
            return "All agents completed L2 reflection; session ending."

        return ""

    def _auto_complete_exited_agents(self, level: str) -> None:
        _level_map = {"L1": self._l1_completed, "L2": self._l2_completed}
        completed = _level_map.get(level)
        if completed is None:
            return
        event_log = self._event_log

        for name, task in self._agent_tasks.items():
            if not task.done():
                continue
            if name in completed or name not in self._agents_for_reflection_level(level):
                continue
            completed.add(name)
            if event_log:
                event_log.log(f"reflection.{level.lower()}.auto_complete", agent=name, data={
                    "reason": "agent run_loop exited",
                })

            if level == "L2":
                for other_name in self._agents_for_reflection_level("L2"):
                    if other_name == name or other_name in completed:
                        continue
                    partners = self.message_store.get_communication_partners(other_name)
                    if name in partners:
                        self.message_store.send(
                            sender="[SYSTEM]",
                            receiver=other_name,
                            content=(
                                f"[L2 UPDATE] {name} has exited and will not participate "
                                f"in L2 discussion. Proceed with your own L2 reflection."
                            ),
                        )


    async def _enter_forced_reflection(self, event_log=None) -> None:
        termination_reason = self._forced_termination or "unknown"

        hook = getattr(self, '_pre_reflection_hook', None)
        if hook:
            try:
                hook()
            except Exception as e:
                if event_log:
                    event_log.log("runner.pre_reflection_hook_error", data={"error": str(e)})

        self._freeze_execution_trace(event_log)

        if event_log:
            event_log.log("runner.forced_reflection_start", data={
                "termination_reason": termination_reason,
                "task_output": (self._result or "[no output]")[:200],
                "agents": list(self.agents.keys()),
                "total_cost_usd": round(self._get_total_cost_usd(), 4),
                "in_place": True,
            })

        self._done.clear()

        if self.message_store.is_terminated():
            self.message_store.reset_termination()

        self._phase = "l1_reflection"
        self._all_reflection_agents = list(self.agents.keys())

        for name, agent in self.agents.items():
            if name not in self._agent_dirs and hasattr(agent, 'agent_dir'):
                self._agent_dirs[name] = str(agent.agent_dir)

        if event_log:
            event_log.log("runner.phase_change", data={
                "from": "task_execution",
                "to": "l1_reflection",
                "task_output": (self._result or "[no output — terminated before completion]")[:200],
                "reflection_agents": self._all_reflection_agents,
                "forced": True,
                "termination_reason": termination_reason,
            })

        forced_context = self._build_forced_reflection_context(termination_reason)

        l1_prompt = _load_prompt_template("reflection/l1_forced.md")
        if not l1_prompt:
            l1_prompt = _load_prompt_template("reflection/l1_agent.md")
        if not l1_prompt:
            l1_prompt = (
                "Reflect on your own performance. Use update_prompt_patch / "
                "update_skill / skip_l1_reflection."
            )

        self._broadcast_forced_l1_start(forced_context, l1_prompt)

        self._restart_all_agents_for_phase("l1_reflection")

        self._start_reflection_phase_guard("l1_reflection")

    def _build_forced_reflection_context(self, termination_reason: str) -> str:
        parts: list[str] = []

        parts.append("## ⚠ TASK TERMINATED — FORCED REFLECTION")
        parts.append(
            f"**Termination reason**: {termination_reason}\n\n"
            "The task was forcibly terminated before normal completion. "
            "Reflect on the available execution evidence. Resource exhaustion alone "
            "does not establish a reusable agent-behavior error."
        )

        task_text = getattr(self, '_task', '') or ''
        if task_text:
            task_preview = task_text[:1000]
            if len(task_text) > 1000:
                task_preview += "\n... (truncated)"
            parts.append(f"## Original Task\n```\n{task_preview}\n```")

        output = self._result or "[no output — the task was terminated before any result was produced]"
        if len(output) > 2000:
            output = output[:2000] + "\n... (truncated)"
        parts.append(f"## Partial Task Output\n```\n{output}\n```")

        parts.append(self._build_resource_summary())

        parts.append(self._build_communication_summary())

        parts.append(
            "## Reflection Focus\n"
            "Ground reflection in **your own observed execution behavior**:\n"
            "1. Identify repeatable behavioral errors or successful behavior supported by the available trace and feedback.\n"
            "2. Propose efficiency improvements only when the trace supports a causal link between your repeatable behavior and wasted resources.\n"
            "3. Leave recruitment, task assignments, team membership, and handoff contracts to team and rule reflection.\n"
            "4. Do not turn model, API, network, tool, Docker, or MCP failures into agent updates.\n"
            "5. Do not weaken correctness or verification, or impose arbitrary universal quotas because resources were exhausted.\n\n"
            "Choose one action and return one structured decision under the L1 instructions. "
            "Choose no_update when evidence is insufficient. "
            "Do not continue debugging or writing code. Use at most 1-2 read-only "
            "commands if needed to check evidence, then complete L1 using the provided tools."
        )

        if self._task_validation is not None:
            v = self._task_validation
            status = "SUCCESS ✓" if v.success else "FAILED ✗"
            parts.append(f"## Task Validation: {status}")
            parts.append(v.summary)
        else:
            parts.append(
                "## Task Validation: NOT PERFORMED\n"
                "The task was terminated before validation could run."
            )

        return "\n\n".join(parts)

    def _broadcast_forced_l1_start(self, forced_context: str, l1_prompt: str) -> None:
        overview = _load_prompt_template("reflection/overview.md") or ""
        for name in self._all_reflection_agents:
            parts = []
            if overview:
                parts.append(overview)
            parts.append(forced_context)
            parts.append("---\n\n**[PHASE: L1 REFLECTION — FORCED]**\n\n" + l1_prompt)
            if name == self.chairman_name:
                parts.append(
                    "### Chairman-Specific Reflection\n"
                    "Apply the same Agent Reflection standard to your own observed execution behavior.\n"
                    "Do not encode team membership, task assignments, or handoff contracts "
                    "in your prompt patches or skills; team and rule reflection own those decisions.\n"
                    "Choose no_update if the available evidence does not support a reusable agent-level improvement."
                )
            self.message_store.send(
                sender="[SYSTEM]",
                receiver=name,
                content="\n\n".join(parts),
            )


    def _broadcast_l1_start(self, l1_prompt: str) -> None:
        overview = _load_prompt_template("reflection/overview.md") or ""
        context = self._build_reflection_context()
        for name in self._all_reflection_agents:
            parts = []
            if overview:
                parts.append(overview)
            parts.append(context)
            phase_header = "---\n\n**[PHASE: L1 REFLECTION]**\n\n" + l1_prompt
            if name == self.chairman_name:
                phase_header += (
                    "\n\n(This is a backup notification. You may have already received "
                    "L1 instructions via finalize_task. Proceed with your L1 reflection.)"
                )
            parts.append(phase_header)
            self.message_store.send(
                sender="[SYSTEM]",
                receiver=name,
                content="\n\n".join(parts),
            )

    def _broadcast_l2_start(self) -> None:
        l2_prompt = _load_prompt_template("reflection/l2_communication.md")
        if not l2_prompt:
            l2_prompt = (
                "Reflect on teammate interactions. Use "
                "update_teammate_profile / skip_l2_reflection."
            )

        l2_agents = self._agents_for_reflection_level("L2")
        chairman_only = (
            l2_agents == [self.chairman_name]
            and not getattr(self, "member_teammate_profiles_enabled", False)
        )
        initiators = self.message_store.get_pairwise_initiators(l2_agents)

        for name in l2_agents:
            if chairman_only:
                role_guidance = (
                    "### Chairman-Only Profile Reflection\n\n"
                    "Review the actual recruited members' task-execution messages and "
                    "outcomes. Update only stable, task-relevant capability observations "
                    "for members you actually started. Do not start a new L2 discussion, "
                    "and do not encode handoff procedures in these profiles. After recording "
                    "any useful observations, call skip_l2_reflection."
                )
                self.message_store.send(
                    sender="[SYSTEM]",
                    receiver=name,
                    content=(
                        "[PHASE: L2 REFLECTION] Chairman-only teammate-profile reflection.\n\n"
                        + role_guidance
                    ),
                )
                continue
            partners = self.message_store.get_communication_partners(name)
            partners = [p for p in partners if p in l2_agents]

            initiate_to: list[str] = []
            wait_from: list[str] = []
            for p in partners:
                pair = (min(name, p), max(name, p))
                initiator = initiators.get(pair)
                if initiator == name:
                    initiate_to.append(p)
                else:
                    wait_from.append(p)

            role_guidance = self._build_l2_role_guidance(name, initiate_to, wait_from)

            self.message_store.send(
                sender="[SYSTEM]",
                receiver=name,
                content=(
                    "[PHASE: L2 REFLECTION] Moving to L2 pairwise communication reflection.\n\n"
                    + l2_prompt + "\n\n"
                    + role_guidance
                ),
            )

    @staticmethod
    def _build_l2_role_guidance(
        agent_name: str,
        initiate_to: list[str],
        wait_from: list[str],
    ) -> str:
        lines: list[str] = []
        lines.append("### Your L2 Discussion Assignments\n")

        if initiate_to:
            names = ", ".join(initiate_to)
            lines.append(
                f"**IF you need to discuss a problem**, you are the initiator for: {names}\n"
                f"Send them a message about the specific issue, "
                f"wait for their reply, then update their teammate profile.\n"
                f"**If collaboration was smooth, do NOT send them a message.** "
                f"Just record your observations and call `skip_l2_reflection`.\n"
            )
        if wait_from:
            names = ", ".join(wait_from)
            lines.append(
                f"**Wait for** possible discussion from: {names}\n"
                f"They may or may not reach out. If they do, reply to their questions, "
                f"then update their teammate profile.\n"
                f"If they don't reach out, just record observations and `skip_l2_reflection`.\n"
            )
        if not initiate_to and not wait_from:
            lines.append(
                "You had no direct communication partners during this task. "
                "Call `skip_l2_reflection` immediately.\n"
            )
        return "\n".join(lines)

    def _restart_all_agents_for_phase(self, phase: str) -> None:
        event_log = self._event_log
        cwd = self._cwd
        max_steps = self.REFLECTION_MAX_STEPS
        restarted: list[str] = []
        refreshed: list[str] = []
        chairman_extra_tools = None

        phase_agents = (
            self._agents_for_reflection_level("L2")
            if phase == "l2_reflection"
            else list(self._all_reflection_agents)
        )
        for name in phase_agents:
            task = self._agent_tasks.get(name)
            if task is None:
                continue
            agent = self.agents.get(name)
            if agent is None:
                continue

            if task.done():
                if event_log:
                    event_log.log("runner.agent_unexpected_exit", agent=name, data={
                        "phase": phase,
                        "note": "fallback restart — context will be lost",
                    })

                system_context = (
                    self._build_chairman_context()
                    if name == self.chairman_name
                    else self._build_agent_context(agent)
                )

                if name == self.chairman_name:
                    from tools.primitives import build_primitive_tools, rebuild_primitive_schemas
                    agent._primitive_tools_refresher = lambda: rebuild_primitive_schemas(
                        self,
                        include_chairman_tools=True,
                        agent_name=self.chairman_name,
                    )
                    chairman_extra_tools = build_primitive_tools(
                        self, agent_name=self.chairman_name,
                    )

                agent._reflection_enabled = self.enable_reflection

                restart_extra_tools = chairman_extra_tools if name == self.chairman_name else None

                new_task = asyncio.create_task(
                    agent.run_loop(
                        event_log=event_log,
                        cwd=cwd,
                        initial_task=None,
                        system_context=system_context,
                        extra_tools=restart_extra_tools,
                        idle_timeout=self.idle_timeout,
                        max_idle_rounds=10,
                        max_steps=max_steps,
                    ),
                    name=f"agent-{name}",
                )
                self._agent_tasks[name] = new_task
                restarted.append(name)
            else:
                agent._phase_step_budget = max_steps
                refreshed.append(name)

        if (restarted or refreshed) and event_log:
            event_log.log("runner.restart_agents_for_phase", data={
                "phase": phase,
                "restarted": restarted,
                "refreshed": refreshed,
                "max_steps": max_steps,
            })


    def _build_reflection_context(self) -> str:
        parts: list[str] = []

        desc_key = getattr(self, '_evolution_description_key', '')
        if desc_key:
            desc_dir = Path(__file__).parent.parent / "agents" / "evolving_description"
            desc_path = desc_dir / f"{desc_key}.md"
            if desc_path.exists():
                try:
                    desc_content = desc_path.read_text(encoding="utf-8").strip()
                    if desc_content:
                        parts.append(desc_content)
                except Exception as e:
                    logger.debug("Failed to read evolution description %s: %s", desc_path, e)

        parts.append("## Final Output Submitted by Chairman")
        output_preview = self._result or ""
        if len(output_preview) > 3000:
            output_preview = output_preview[:3000] + "\n... (truncated)"
        parts.append(f"```\n{output_preview}\n```")

        if self._task_validation is not None:
            v = self._task_validation
            status = "SUCCESS ✓" if v.success else "FAILED ✗"
            parts.append(f"## Task Validation: {status}")
            parts.append(v.summary)
            if v.details:
                parts.append(f"### Validation Details\n```\n{v.details[:2000]}\n```")
        else:
            parts.append(
                "## Task Validation: NOT AVAILABLE\n"
                "No external validation was performed. "
                "Reflect based on your own judgment of output quality."
            )

        parts.append(self._build_resource_summary())

        parts.append(self._build_communication_summary())
        
        execution_trace = self._build_execution_trace()
        if execution_trace:
            parts.append(execution_trace)

        return "\n\n".join(parts)

    def _build_execution_trace(self) -> str:
        lines = ["## What Your Team Did (Execution Trace)"]
        
        messages_by_agent: dict[str, list] = {}
        
        for msg in self.message_store.iter_all_messages():
            sender = msg.sender
            if sender == "[SYSTEM]":
                continue
            
            if sender not in messages_by_agent:
                messages_by_agent[sender] = []
            
            msg_content = msg.content
            if "calling" in msg_content.lower() or "tool" in msg_content.lower():
                messages_by_agent[sender].append({
                    'content': msg_content[:200],
                    'time_offset': msg.timestamp if hasattr(msg, 'timestamp') else None
                })
        
        if not messages_by_agent:
            lines.append("No execution trace available.")
            return "\n".join(lines)
        
        for agent_name in sorted(messages_by_agent.keys()):
            if agent_name == self.chairman_name:
                lines.append(f"\n**{agent_name}** (Team Lead):")
            else:
                lines.append(f"\n**{agent_name}**:")
            
            msgs = messages_by_agent[agent_name][:5]
            for i, msg_info in enumerate(msgs, 1):
                preview = msg_info['content'].replace('\n', ' ')[:100]
                lines.append(f"  {i}. {preview}")
        
        lines.append("")
        return "\n".join(lines)

    def _build_resource_summary(self) -> str:
        total_cost = self._get_total_cost_usd()
        elapsed = time.time() - self._run_start_time

        agent_costs = {}
        for name, agent in self.agents.items():
            agent_costs[name] = round(getattr(agent, "_total_cost_usd", 0.0), 2)

        lines = [
            "## Resource Usage",
            f"- **Total cost**: ${total_cost:.2f} / ${self.max_cost_usd:.2f} limit",
            f"- **Task time**: {elapsed:.0f}s ({elapsed/60:.1f}min) / {self.max_seconds:.0f}s budget",
            f"- **API recovery time**: {self.api_tracker.recovery_seconds():.0f}s across {len(self.api_tracker.incidents)} failed request(s)",
            f"- **Messages exchanged**: {self.message_store.total_count}",
            f"- **Cost by agent**: {agent_costs}",
            f"- **Team size**: {len(self.agents)} agents active",
        ]
        return "\n".join(lines)

    def _build_communication_summary(self) -> str:
        lines = ["## Communication Summary"]

        pair_stats: dict[tuple[str, str], list[int]] = {}
        for msg in self.message_store.iter_all_messages():
            sender = msg.sender
            receiver = msg.receiver
            if sender == "[SYSTEM]" or receiver == "[SYSTEM]":
                continue
            pair = (sender, receiver)
            if pair not in pair_stats:
                pair_stats[pair] = []
            pair_stats[pair].append(len(msg.content))

        if not pair_stats:
            lines.append("No inter-agent communication during task execution.")
            return "\n".join(lines)

        for (sender, receiver), lengths in sorted(pair_stats.items()):
            count = len(lengths)
            avg_len = sum(lengths) // count if count else 0
            lines.append(f"- {sender} → {receiver}: {count} messages (avg {avg_len} chars)")

        return "\n".join(lines)

    def _build_team_timeline(self) -> str:
        start = self._run_start_time
        if not start:
            return ""

        lines = ["## Team Timeline"]

        agent_first_msg: dict[str, float] = {}
        agent_last_msg: dict[str, float] = {}
        for msg in self.message_store.iter_all_messages():
            if msg.sender == "[SYSTEM]":
                continue
            name = msg.sender
            ts = msg.timestamp
            if name not in agent_first_msg or ts < agent_first_msg[name]:
                agent_first_msg[name] = ts
            if name not in agent_last_msg or ts > agent_last_msg[name]:
                agent_last_msg[name] = ts

        events: list[tuple[float, str]] = []
        for name, ts in agent_first_msg.items():
            rel = ts - start
            cost = round(getattr(self.agents.get(name), "_total_cost_usd", 0.0), 2)
            events.append((rel, f"{name} first active (cost so far: ${cost})"))
        for name, ts in agent_last_msg.items():
            rel = ts - start
            events.append((rel, f"{name} last message"))

        events.sort(key=lambda x: x[0])

        for rel_time, desc in events:
            mins, secs = divmod(max(0, int(rel_time)), 60)
            lines.append(f"- {mins}:{secs:02d}  {desc}")

        elapsed = time.time() - start
        mins, secs = divmod(int(elapsed), 60)
        lines.append(f"- {mins}:{secs:02d}  → entered reflection")

        lines.append("\n**Final cost by agent:**")
        for name, agent in self.agents.items():
            cost = round(getattr(agent, "_total_cost_usd", 0.0), 2)
            lines.append(f"- {name}: ${cost}")

        return "\n".join(lines)


    def _start_reflection_phase_guard(self, phase: str) -> None:
        if self._reflection_phase_guard and not self._reflection_phase_guard.done():
            self._reflection_phase_guard.cancel()
        self._reflection_phase_start = time.time()
        self._reflection_phase_guard = asyncio.create_task(
            self._guard_reflection_phase(phase, self._event_log)
        )

    async def _guard_reflection_phase(self, phase: str, event_log=None) -> None:
        try:
            await asyncio.sleep(self.reflection_phase_timeout)
            if event_log:
                event_log.log("runner.reflection_phase_timeout", data={
                    "phase": phase,
                    "timeout": self.reflection_phase_timeout,
                })
            level_map = {
                "l1_reflection": ("L1", self._l1_completed),
                "l2_reflection": ("L2", self._l2_completed),
            }
            level, completed = level_map.get(phase, (None, None))
            if level and completed is not None:
                for name in self._agents_for_reflection_level(level):
                    if name not in completed:
                        completed.add(name)
                        if event_log:
                            event_log.log(
                                f"reflection.{level.lower()}.auto_complete",
                                agent=name,
                                data={"reason": "phase_timeout"},
                            )
                if level in ("L1", "L2"):
                    self._auto_advance_phase(level)
        except asyncio.CancelledError:
            pass