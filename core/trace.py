import json
from core.session import SESSIONS_DIR
from core.utils import (parse_ts, fmt_elapsed, truncate, load_session_data,
                         load_session_data_from_dir, collect_agent_stats, calc_cost)



_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}

_AGENT_COLORS = ["blue", "green", "yellow", "magenta", "cyan"]


def _c(color: str, text: str) -> str:
    return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"


def _bold(text: str) -> str:
    return f"{_COLORS['bold']}{text}{_COLORS['reset']}"


def _dim(text: str) -> str:
    return f"{_COLORS['dim']}{text}{_COLORS['reset']}"


_truncate = truncate
_parse_ts = parse_ts
_fmt_elapsed = fmt_elapsed



def _load_session(session_id: str, session_dir=None):
    if session_dir:
        from pathlib import Path
        events, meta = load_session_data_from_dir(Path(session_dir))
    else:
        events, meta = load_session_data(session_id, SESSIONS_DIR)
    if not events:
        print(f"No events found for session {session_id}")
        return None, None
    return events, meta


def _collect_agents_and_stats(events: list) -> tuple[list[str], dict, dict]:
    agents_seen, stats = collect_agent_stats(events)

    agent_color = {}
    for i, a in enumerate(agents_seen):
        agent_color[a] = _AGENT_COLORS[i % len(_AGENT_COLORS)]

    return agents_seen, stats, agent_color



def render_overview(session_id: str, session_dir=None) -> None:
    events, meta = _load_session(session_id, session_dir=session_dir)
    if events is None:
        return

    task = meta.get("task", "")
    t0 = _parse_ts(events[0].get("ts", ""))
    agents_seen, stats, agent_color = _collect_agents_and_stats(events)

    run_end = {}
    for e in events:
        if e.get("type") in ("committee.end", "runner.end"):
            run_end = e.get("data", {})

    w = 72
    print()
    print(_bold("╔" + "═" * w + "╗"))
    print(_bold(f"║  OVERVIEW — {session_id}".ljust(w + 2) + "║"))
    print(_bold("╠" + "═" * w + "╣"))
    ow = w - 4
    task_lines = task.split("\n")
    ow_lines: list[str] = []
    for tl in task_lines:
        while len(tl) > ow:
            ow_lines.append(tl[:ow])
            tl = tl[ow:]
        ow_lines.append(tl)
    if ow_lines:
        print(f"║  Task: {ow_lines[0]}".ljust(w + 2) + "║")
        for line in ow_lines[1:]:
            print(f"║    {line}".ljust(w + 2) + "║")
    else:
        print(f"║  Task: (empty)".ljust(w + 2) + "║")

    t_last = _parse_ts(events[-1].get("ts", ""))
    duration = t_last - t0 if t0 else 0
    total_tokens_in = sum(s["tokens_in"] for s in stats.values())
    total_tokens_out = sum(s["tokens_out"] for s in stats.values())
    print(f"║  Duration: {_fmt_elapsed(duration)}  |  "
          f"Events: {len(events)}  |  "
          f"Agents: {len(agents_seen)}  |  "
          f"Tokens: {total_tokens_in}in/{total_tokens_out}out".ljust(w + 1) + "║")

    pricing = (meta.get("summary") or {}).get("pricing")
    if pricing:
        ci = calc_cost(total_tokens_in, total_tokens_out,
                       pricing.get("input_price", 0), pricing.get("output_price", 0))
        print(f"║  Cost: ${ci['total_cost']:.6f}  "
              f"(in: ${ci['cost_in']:.6f}  out: ${ci['cost_out']:.6f})".ljust(w + 2) + "║")

    print(_bold("╚" + "═" * w + "╝"))

    print()
    print(_bold("  AGENTS"))
    print(f"  {'Name':<14} {'Steps':>5} {'Tools':>5} {'MsgOut':>6} {'MsgIn':>5} "
          f"{'TokIn':>7} {'TokOut':>7}")
    print(f"  {'─' * 14} {'─' * 5} {'─' * 5} {'─' * 6} {'─' * 5} {'─' * 7} {'─' * 7}")
    for a in agents_seen:
        s = stats[a]
        color = agent_color[a]
        print(f"  {_c(color, a):<24} {s['steps']:>5} {s['tool_calls']:>5} "
              f"{s['msgs_sent']:>6} {s['msgs_recv']:>5} "
              f"{s['tokens_in']:>7} {s['tokens_out']:>7}")

    print()
    print(_bold("  TIMELINE"))
    _render_timeline(events, t0, agents_seen, agent_color, stats)

    print()
    print(_bold("  MESSAGE FLOW"))
    _render_message_flow(events, agents_seen, agent_color)

    _render_governance_summary(events, t0, agent_color)

    _render_final_output(events, t0)

    print()
    print(_dim(f"  Full trace: python main.py --trace {session_id}"))
    print(_dim(f"  Agent view: python main.py --trace {session_id} --agent <name>"))
    print(_dim(f"  Verbose:    python main.py --trace {session_id} --verbose"))
    print()


def _render_timeline(events, t0, agents_seen, agent_color, stats):
    for e in events:
        ts = _parse_ts(e.get("ts", ""))
        elapsed = ts - t0 if t0 else 0
        etype = e.get("type", "")
        agent = e.get("agent", "")
        data = e.get("data", {})
        t_str = _fmt_elapsed(elapsed)
        color = agent_color.get(agent, "white")

        if etype == "committee.start":
            members = data.get("members", [])
            print(f"  {_dim(t_str)}  {_bold('▶ START')} "
                  f"members=[{', '.join(members)}]")

        elif etype == "runner.start":
            chairman = data.get("chairman", "")
            pool_dir = data.get("pool_dir", "")
            print(f"  {_dim(t_str)}  {_bold('▶ START')} "
                  f"chairman={chairman} pool={pool_dir.split('/')[-1] if pool_dir else ''}")

        elif etype == "agent.loop.start":
            initial_state = data.get("initial_state", "")
            icon = "🟢" if initial_state == "working" else "⏸️"
            print(f"  {_dim(t_str)}  {_c(color, agent):>22} {icon} {initial_state}")

        elif etype == "agent.think":
            content = data.get("content", "")
            if content:
                summary = content.replace("\n", " ")[:80]
                print(f"  {_dim(t_str)}  {_c(color, agent):>22} 🧠 {_dim(summary)}")

        elif etype == "tool.call":
            tool = data.get("tool", "")
            if tool in ("send_message",):
                args = data.get("args", {})
                to = args.get("to", "?")
                if isinstance(to, list):
                    to_str = ", ".join(to)
                else:
                    to_str = str(to)
                content = str(args.get("content", ""))[:60].replace("\n", " ")
                print(f"  {_dim(t_str)}  {_c(color, agent):>22} 📨→{to_str} {_dim(content)}")
            elif tool == "wait_for_replies":
                args = data.get("args", {})
                reason = args.get("reason", "")[:60]
                print(f"  {_dim(t_str)}  {_c(color, agent):>22} ⏳ {_dim(f'waiting: {reason}')}")
            elif tool in ("review_plan", "raise_concern", "override_concern",
                          "dispatch_to_team", "finalize_task"):
                args = data.get("args", {})
                brief = _brief_tool_args(tool, args)
                print(f"  {_dim(t_str)}  {_c(color, agent):>22} 🔧 {_bold(tool)} {_dim(brief)}")
            elif tool in ("list_pool", "start_agent", "stop_agent",
                          "set_final_output", "finalize_task", "terminate"):
                args = data.get("args", {})
                brief = _brief_tool_args(tool, args)
                print(f"  {_dim(t_str)}  {_c(color, agent):>22} 🔧 {_bold(tool)} {_dim(brief)}")

        elif etype == "gate.check":
            gate = data.get("gate", "")
            passed = data.get("passed", False)
            icon = _c("green", "✅") if passed else _c("red", "❌")
            print(f"  {_dim(t_str)}  {'':>22} 🔒 GATE[{gate}] {icon}")

        elif etype == "review.change":
            action = data.get("action", "")
            if action == "review_plan":
                verdict = data.get("verdict", "")
                v_icon = {"approve": "👍", "concern": "⚠️", "reject": "❌"}.get(verdict, "📋")
                print(f"  {_dim(t_str)}  {_c(color, agent):>22} {v_icon} review: {_bold(verdict)}")
            elif action == "raise_concern":
                sev = data.get("severity", "")
                s_icon = {"critical": "🚨", "warning": "⚠️"}.get(sev, "ℹ️")
                desc = data.get("description", "")[:60]
                print(f"  {_dim(t_str)}  {_c(color, agent):>22} {s_icon} concern[{sev}]: {_dim(desc)}")
            elif action == "override_concern":
                cid = data.get("concern_id", "")
                print(f"  {_dim(t_str)}  {_c(color, agent):>22} ⚡ override {cid}")

        elif etype == "agent.wake":
            senders = data.get("senders", [])
            count = data.get("count", 0)
            print(f"  {_dim(t_str)}  {_c(color, agent):>22} ◀ woken ({count} msgs from {', '.join(senders)})")

        elif etype == "agent.loop.terminated":
            reason = data.get("reason", "")
            print(f"  {_dim(t_str)}  {_c(color, agent):>22} 🛑 {reason}")

        elif etype == "agent.state":
            from_s = data.get("from", "")
            to_s = data.get("to", "")
            trigger = data.get("trigger", "")
            icon = "🟢" if to_s == "working" else "⏸️"
            print(f"  {_dim(t_str)}  {_c(color, agent):>22} {icon} {from_s}→{to_s} ({trigger})")

        elif etype == "committee.end":
            cost = data.get("cost_seconds", 0)
            print(f"  {_dim(t_str)}  {_bold('■ END')} {cost:.1f}s")

        elif etype == "runner.end":
            cost = data.get("cost_seconds", 0)
            cost_usd = data.get("cost_usd", 0)
            agents_used = data.get("agents_used", [])
            usd_str = f" ${cost_usd:.4f}" if cost_usd else ""
            print(f"  {_dim(t_str)}  {_bold('■ END')} {cost:.1f}s{usd_str} "
                  f"agents=[{', '.join(agents_used)}]")

        elif etype == "runner.timeout":
            print(f"  {_dim(t_str)}  {_c('red', _bold('⏱ TIMEOUT'))} "
                  f"max_seconds={data.get('max_seconds')}")

        elif etype == "runner.guard":
            reason = data.get("reason", "")
            print(f"  {_dim(t_str)}  {_c('yellow', _bold('⚠ GUARD'))}: {reason}")

        elif etype in ("committee.phase_change", "runner.phase_change"):
            to_p = data.get("to", "")
            phase_icons = {
                "l1_reflection": "🔵 Agent Self-Reflection",
                "l2_reflection": "🟢 Teammate Profile Reflection",
                "terminated": "⬛ Terminated",
            }
            label = phase_icons.get(to_p, to_p)
            print(f"  {_dim(t_str)}  {_c('magenta', _bold(f'━━ {label} ━━'))}")

        elif etype == "committee.restart_agents":
            restarted = data.get("restarted", [])
            print(f"  {_dim(t_str)}  {_c('magenta', '🔄 restart')}: {', '.join(restarted)}")

        elif etype == "runner.restart_agents_for_phase":
            phase = data.get("phase", "")
            restarted = data.get("restarted", [])
            refreshed = data.get("refreshed", [])
            phase_label = {
                "l1_reflection": "Agent Self-Reflection",
                "l2_reflection": "Teammate Profile Reflection",
            }.get(phase, phase)
            parts = []
            if restarted:
                parts.append(f"restarted=[{', '.join(restarted)}]")
            if refreshed:
                parts.append(f"refreshed=[{', '.join(refreshed)}]")
            print(f"  {_dim(t_str)}  {_c('magenta', '🔄 restart for')} {phase_label}: {' '.join(parts)}")

        elif etype == "reflection.l1.patch":
            print(f"  {_dim(t_str)}  {_c(color, agent):>22} 📝 Self-reflection patch")

        elif etype == "reflection.l1.skill":
            skill = data.get("skill_name", "")
            print(f"  {_dim(t_str)}  {_c(color, agent):>22} 🔧 Self-reflection skill: {skill}")

        elif etype == "reflection.l1.skip":
            print(f"  {_dim(t_str)}  {_c(color, agent):>22} ⏭ Self-reflection complete")

        elif etype == "reflection.l2.profile":
            teammate = data.get("teammate", "")
            print(f"  {_dim(t_str)}  {_c(color, agent):>22} 👤 Teammate profile → {teammate}")

        elif etype == "reflection.l2.correlation":
            partner = data.get("partner", "")
            print(f"  {_dim(t_str)}  {_c(color, agent):>22} 🤝 Profile correlation ↔ {partner}")

        elif etype == "reflection.l2.skip":
            print(f"  {_dim(t_str)}  {_c(color, agent):>22} ⏭ Profile reflection complete")

        elif etype == "committee.reflection_abort":
            reason = data.get("reason", "")
            print(f"  {_dim(t_str)}  {_c('red', _bold('⚠ REFLECTION ABORT'))} {reason}")

        elif etype == "runner.reflection_skipped_time_budget":
            remaining = data.get("remaining", 0)
            print(f"  {_dim(t_str)}  {_c('yellow', '⏭ REFLECTION SKIPPED')} "
                  f"only {remaining:.0f}s remaining")

        elif etype.startswith("reflection.") and etype.endswith(".auto_complete"):
            print(f"  {_dim(t_str)}  {_c(color, agent):>22} ⏩ auto-complete")


def _render_message_flow(events, agents_seen, agent_color):
    flow: dict[tuple[str, str], int] = {}

    for i, e in enumerate(events):
        if e.get("type") != "tool.call":
            continue
        data = e.get("data", {})
        tool = data.get("tool", "")
        sender = e.get("agent", "")
        if tool == "send_message":
            to = data.get("args", {}).get("to", "")
            if isinstance(to, list):
                recipients = to
            elif isinstance(to, str) and to:
                recipients = [to]
            else:
                recipients = []
            for r in recipients:
                if sender and r:
                    flow[(sender, r)] = flow.get((sender, r), 0) + 1

    if not flow:
        print(f"  {_dim('(no messages)')}")
        return

    for (s, t), count in sorted(flow.items(), key=lambda x: -x[1]):
        sc = agent_color.get(s, "white")
        tc = agent_color.get(t, "white")
        bar = "█" * min(count, 20)
        print(f"  {_c(sc, s):>22} → {_c(tc, t):<14} {bar} {count}")


def _render_governance_summary(events, t0, agent_color):
    reviews = []
    concerns = []
    gates = []
    for e in events:
        etype = e.get("type", "")
        data = e.get("data", {})
        agent = e.get("agent", "")
        if etype == "review.change":
            action = data.get("action", "")
            if action == "review_plan":
                reviews.append((agent, data.get("verdict", "")))
            elif action == "raise_concern":
                concerns.append((agent, data.get("severity", ""), data.get("description", "")[:60]))
            elif action == "override_concern":
                concerns.append((agent, "override", data.get("concern_id", "")))
        elif etype == "gate.check":
            gates.append((data.get("gate", ""), data.get("passed", False)))

    if not reviews and not concerns and not gates:
        return

    print()
    print(_bold("  GOVERNANCE"))
    if reviews:
        print(f"  Reviews: ", end="")
        parts = []
        for agent, verdict in reviews:
            color = agent_color.get(agent, "white")
            v_icon = {"approve": "👍", "concern": "⚠️", "reject": "❌"}.get(verdict, "?")
            parts.append(f"{_c(color, agent)}:{v_icon}{verdict}")
        print("  ".join(parts))
    if concerns:
        for agent, sev, desc in concerns:
            color = agent_color.get(agent, "white")
            if sev == "override":
                print(f"  ⚡ {_c(color, agent)} override → {desc}")
            else:
                s_icon = {"critical": "🚨", "warning": "⚠️"}.get(sev, "ℹ️")
                print(f"  {s_icon} {_c(color, agent)} [{sev}] {_dim(desc)}")
    if gates:
        for gate, passed in gates:
            icon = _c("green", "✅ PASS") if passed else _c("red", "❌ BLOCK")
            print(f"  🔒 Gate[{gate}]: {icon}")


def _render_final_output(events, t0):
    for e in reversed(events):
        if e.get("type") == "tool.call":
            data = e.get("data", {})
            if data.get("tool") in ("finalize_task", "set_final_output"):
                output = data.get("args", {}).get("output", "")
                if output:
                    print()
                    print(_bold("  FINAL OUTPUT"))
                    for line in output[:500].split("\n"):
                        print(f"  {line}")
                    if len(output) > 500:
                        print(f"  {_dim(f'[...{len(output) - 500} more chars]')}")
                return


def _brief_tool_args(tool: str, args: dict) -> str:
    if tool == "review_plan":
        return f"verdict={args.get('verdict', '')}"
    if tool == "raise_concern":
        return f"[{args.get('severity', '')}] {str(args.get('description', ''))[:50]}"
    if tool == "override_concern":
        return f"id={args.get('concern_id', '')}"
    if tool == "dispatch_to_team":
        return f"→ {args.get('team_name', '')}:{str(args.get('task_description', ''))[:40]}"
    if tool == "finalize_task":
        return f"output={str(args.get('output', ''))[:40]}..."
    if tool == "list_pool":
        return ""
    if tool == "start_agent":
        return f"agent={args.get('agent_name', '')}"
    if tool == "stop_agent":
        return f"agent={args.get('agent_name', '')}"
    if tool == "set_final_output":
        return f"output={str(args.get('output', ''))[:40]}..."
    if tool == "terminate":
        return f"reason={args.get('reason', '')}"
    return ""



def render_trace(
    session_id: str,
    verbose: bool = False,
    agent_filter: str | None = None,
    session_dir=None,
) -> None:
    events, meta = _load_session(session_id, session_dir=session_dir)
    if events is None:
        return

    task = meta.get("task", "")
    text_limit = 0 if verbose else 500

    t0 = _parse_ts(events[0].get("ts", ""))

    agents_seen, stats, agent_color = _collect_agents_and_stats(events)

    if agent_filter:
        if agent_filter not in agents_seen:
            print(f"Agent '{agent_filter}' not found. Available: {', '.join(agents_seen)}")
            return
        print(f"\n{_dim(f'  Filtering: only showing events for {agent_filter}')}")

    run_end_data = {}

    _print_header(session_id, task, events, agents_seen)

    printed_steps: set[tuple[str, int]] = set()
    agent_step: dict[str, int] = {}
    last_printed_agent = ""
    pending_call_ts: dict[str, float] = {}
    current_phase = "task_execution"

    _PHASE_LABELS = {
        "l1_reflection": " [Self-Reflection]",
        "l2_reflection": " [Profiles]",
    }

    def _phase_label() -> str:
        return _PHASE_LABELS.get(current_phase, "")

    def _step_display(agent: str, step: int) -> str:
        agent_stats = stats.get(agent, {})
        phase_steps = agent_stats.get("phase_steps", {})
        phase_map = phase_steps.get(current_phase, {})
        if phase_map:
            local_step = phase_map.get(step, step)
            max_local = len(phase_map)
            return f"step {local_step}/{max_local}{_phase_label()}"
        max_step = agent_stats.get("steps", "?")
        return f"step {step}/{max_step}{_phase_label()}"

    def _ensure_step_header(agent: str, step: int, elapsed_str: str) -> None:
        nonlocal last_printed_agent
        key = (agent, step)
        if key in printed_steps:
            last_printed_agent = agent
            return
        printed_steps.add(key)
        last_printed_agent = agent
        color = agent_color.get(agent, "white")
        display = _step_display(agent, step)
        print(f"\n{_dim(f'[{elapsed_str}]')} "
              f"{_c(color, f'{agent}')} {display}")

    def _ensure_agent_context(agent: str, elapsed_str: str) -> None:
        nonlocal last_printed_agent
        if agent and agent != last_printed_agent:
            step = agent_step.get(agent, 0)
            key = (agent, step)
            if key not in printed_steps:
                _ensure_step_header(agent, step, elapsed_str)
            else:
                last_printed_agent = agent
                color = agent_color.get(agent, "white")
                display = _step_display(agent, step)
                print(f"\n{_dim(f'[{elapsed_str}]')} "
                      f"{_c(color, f'{agent}')} {display}")

    for e in events:
        ts = _parse_ts(e.get("ts", ""))
        elapsed = ts - t0 if t0 else 0
        etype = e.get("type", "")
        agent = e.get("agent", "")
        data = e.get("data", {})
        elapsed_str = _fmt_elapsed(elapsed)

        if agent_filter and agent and agent != agent_filter:
            if etype == "tool.call" and data.get("tool") == "send_message":
                to = data.get("args", {}).get("to", "")
                if isinstance(to, list):
                    if agent_filter in to:
                        pass
                    else:
                        continue
                elif to == agent_filter:
                    pass
                else:
                    continue
            else:
                continue

        if etype == "committee.start":
            members = data.get("members", [])
            chair = data.get("chair", "")
            print(f"\n{_dim(f'[{elapsed_str}]')} {_bold('▶ COMMITTEE START')}")
            print(f"   members: {', '.join(members)}  chair: {_bold(chair)}")
            gate = data.get("gate_config", {})
            if gate:
                print(f"   gates: dispatch_review={'on' if gate.get('dispatch_requires_review') else 'off'}"
                      f"  min_reviews={gate.get('min_reviews_for_dispatch')}"
                      f"  finalize_block_critical={'on' if gate.get('finalize_blocks_on_critical') else 'off'}")
            continue

        if etype == "runner.start":
            chairman = data.get("chairman", "")
            pool_dir = data.get("pool_dir", "")
            pool_name = pool_dir.split("/")[-1] if pool_dir else ""
            reflection = data.get("enable_reflection", False)
            print(f"\n{_dim(f'[{elapsed_str}]')} {_bold('▶ RUNNER START')}")
            print(f"   chairman: {_bold(chairman)}  pool: {pool_name}"
                  f"  max_seconds: {data.get('max_seconds', '?')}"
                  f"  reflection: {'on' if reflection else 'off'}")
            continue

        if etype == "committee.end":
            run_end_data = data
            continue

        if etype == "runner.end":
            run_end_data = data
            continue

        if etype == "committee.timeout":
            print(f"\n{_dim(f'[{elapsed_str}]')} {_c('red', '⏱ COMMITTEE TIMEOUT')}"
                  f" max_seconds={data.get('max_seconds')}")
            continue

        if etype == "runner.timeout":
            print(f"\n{_dim(f'[{elapsed_str}]')} {_c('red', '⏱ RUNNER TIMEOUT')}"
                  f" max_seconds={data.get('max_seconds')}"
                  f" elapsed={data.get('elapsed', '?')}s")
            continue

        if etype == "runner.guard":
            reason = data.get("reason", "")
            print(f"\n{_dim(f'[{elapsed_str}]')} {_c('yellow', '⚠ GUARD TRIGGERED')}"
                  f" reason={reason}")
            continue

        if etype in ("committee.phase_change", "runner.phase_change"):
            from_p = data.get("from", "")
            to_p = data.get("to", "")
            current_phase = to_p
            phase_icons = {
                "l1_reflection": "🔵 Agent Self-Reflection",
                "l2_reflection": "🟢 Teammate Profile Reflection",
                "terminated": "⬛ Terminated",
            }
            label = phase_icons.get(to_p, to_p)
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{_c('magenta', _bold(f'━━━ PHASE: {label} ━━━'))}")
            for key, label in (
                ("l1_completed", "self-reflection completed"),
                ("l2_completed", "profile reflection completed"),
            ):
                completed = data.get(key, [])
                if completed:
                    print(f"      {label}: {', '.join(completed)}")
            continue

        if etype in ("session.start", "session.end"):
            continue

        if etype in ("committee.restart_agents", "runner.restart_agents_for_phase"):
            restarted = data.get("restarted", [])
            refreshed = data.get("refreshed", [])
            reason = data.get("reason", "")
            phase = data.get("phase", "")
            label = phase or reason or ""
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{_c('magenta', f'🔄 RESTART AGENTS')}: {', '.join(restarted)}")
            if refreshed:
                print(f"      refreshed: {', '.join(refreshed)}")
            if label:
                print(f"      {_dim(label)}")
            continue

        if etype == "committee.max_messages":
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{_c('yellow', '⚠ MAX MESSAGES')} "
                  f"total={data.get('total_count', '?')}")
            continue

        if etype == "committee.reflection_abort":
            reason = data.get("reason", "")
            phase = data.get("phase", "")
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{_c('red', _bold(f'⚠ REFLECTION ABORT'))}"
                  f" phase={phase}: {reason}")
            continue

        if etype == "runner.reflection_skipped_time_budget":
            remaining = data.get("remaining", 0)
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{_c('yellow', _bold('⏭ REFLECTION SKIPPED'))}"
                  f" only {remaining:.0f}s remaining (need {data.get('min_required', '?')}s)")
            continue

        if etype == "runner.task_validation":
            success = data.get("success", False)
            summary = data.get("summary", "")
            icon = _c("green", "✅") if success else _c("red", "❌")
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{icon} {_bold('TASK VALIDATION')}: {summary[:200]}")
            continue

        if etype.startswith("reflection.") and etype.endswith(".auto_complete"):
            reason = data.get("reason", "")
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{_c('magenta', f'⏩ AUTO-COMPLETE')} {_bold(agent)}: {_dim(reason)}")
            continue

        color = agent_color.get(agent, "white")

        if etype == "agent.loop.start":
            initial = data.get("initial_task", "")
            initial_state = data.get("initial_state", "idle")
            ctx = data.get("system_context", "")
            last_printed_agent = agent
            print(f"\n{_dim(f'[{elapsed_str}]')} {_c(color, _bold(f'━━━ {agent.upper()} START ━━━'))}")
            if initial:
                task_display = ""
                if ctx:
                    import re as _re
                    m = _re.search(r"## Current Task\s*\n\n(.*?)(?=\n\n## |\Z)", ctx, _re.DOTALL)
                    if m:
                        task_display = m.group(1).strip()
                if not task_display:
                    task_display = initial
                display = _truncate(task_display, text_limit) if text_limit else task_display
                print(f"   {_dim('📥 INITIAL TASK:')}")
                _print_indented(display, indent=6)
            elif initial_state == "idle":
                print(f"   {_dim('⏸️ IDLE — waiting for messages')}")
            continue

        if etype == "llm.call":
            step = data.get("step", 0)
            agent_step[agent] = step
            continue

        if etype == "agent.think":
            content = data.get("content", "")
            step = data.get("step", agent_step.get(agent, 0))
            if not content:
                continue
            _ensure_step_header(agent, step, elapsed_str)
            display = _truncate(content, text_limit) if text_limit else content
            print(f"   {_dim('🧠 THINK:')}")
            _print_indented(display, indent=6)
            continue

        if etype == "tool.call":
            tool = data.get("tool", "")
            args = data.get("args", {})
            step = agent_step.get(agent, 0)
            _ensure_step_header(agent, step, elapsed_str)
            args_str = _format_args(tool, args, text_limit)
            print(f"   {_dim('🔧 ACTION:')} {_bold(tool)}({args_str})")
            if tool == "wait_for_replies":
                pending_call_ts[agent] = ts
            continue

        if etype == "tool.result":
            tool = data.get("tool", "")
            output = data.get("output", "")
            _ensure_agent_context(agent, elapsed_str)
            last_printed_agent = agent
            display = _truncate(output, text_limit) if text_limit else output
            wait_suffix = ""
            if tool == "wait_for_replies" and agent in pending_call_ts:
                wait_secs = ts - pending_call_ts.pop(agent)
                if wait_secs >= 1.0:
                    wait_suffix = f" {_dim(f'(waited {wait_secs:.1f}s)')}"
            suffix = _message_flow_suffix(tool, data)
            print(f"   {_dim('📤 RESULT:')} {display}{wait_suffix}")
            if suffix:
                print(f"   {suffix}")
            continue

        if etype == "agent.wake":
            senders = data.get("senders", [])
            count = data.get("count", 0)
            content = data.get("content", "")
            senders_str = ", ".join(senders)
            last_printed_agent = agent
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{_c(color, agent)} "
                  f"{_dim(f'◀ woken by: {senders_str} ({count} msgs)')}")
            if content:
                display = _truncate(content, text_limit) if text_limit else content
                print(f"   {_dim('📥 MESSAGES:')}")
                _print_indented(display, indent=6)
            continue

        if etype == "agent.loop.idle":
            _ensure_agent_context(agent, elapsed_str)
            last_printed_agent = agent
            print(f"   {_dim('⏸ IDLE — waiting for messages...')}")
            continue

        if etype == "agent.loop.terminated":
            _ensure_agent_context(agent, elapsed_str)
            last_printed_agent = agent
            reason = data.get("reason", "")
            print(f"   {_dim(f'🛑 TERMINATED — {reason}')}")
            continue

        if etype == "agent.loop.cancelled":
            _ensure_agent_context(agent, elapsed_str)
            last_printed_agent = agent
            steps = data.get("steps", 0)
            cost = data.get("cost_seconds", 0)
            print(f"   {_dim(f'⚡ CANCELLED — {steps} steps, {cost:.1f}s')}")
            continue

        if etype == "agent.state":
            from_s = data.get("from", "")
            to_s = data.get("to", "")
            trigger = data.get("trigger", "")
            icon = "🟢" if to_s == "working" else "⏸️"
            _ensure_agent_context(agent, elapsed_str)
            last_printed_agent = agent
            print(f"   {icon} {_dim(f'STATE: {from_s} → {to_s}')} ({trigger})")
            continue

        if etype == "agent.loop.end":
            _ensure_agent_context(agent, elapsed_str)
            last_printed_agent = agent
            steps = data.get("steps", 0)
            tc = data.get("tool_calls_total", 0)
            cost = data.get("cost_seconds", 0)
            print(f"   {_dim(f'■ END — {steps} steps, {tc} tool calls, {cost:.1f}s')}")
            continue

        if etype == "dispatch.timeout":
            team_name = data.get("team_name", "")
            leader = data.get("leader_name", "")
            timeout = data.get("timeout", 0)
            el = data.get("elapsed", 0)
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{_c('red', '⏱ DISPATCH TIMEOUT')} "
                  f"team={_bold(team_name)} leader={leader} "
                  f"({el}s / {timeout}s limit)")
            continue

        if etype == "reflection.l1.patch":
            patch = data.get("patch", "")
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{_c('magenta', f'📝 SELF-REFLECTION PATCH')} by {_bold(agent)}")
            if patch:
                _print_indented(patch, indent=6)
            continue

        if etype == "reflection.l1.skill":
            skill_name = data.get("skill_name", "")
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{_c('magenta', f'🔧 SELF-REFLECTION SKILL')} by {_bold(agent)}: {skill_name}")
            continue

        if etype == "reflection.l1.skip":
            reason = data.get("reason", "")
            completed = data.get("completed", [])
            total = data.get("total", 0)
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{_c('magenta', f'⏭ SELF-REFLECTION COMPLETE')} by {_bold(agent)}: {reason}")
            print(f"      progress: {len(completed)}/{total} completed: {', '.join(completed)}")
            continue

        if etype == "reflection.l2.profile":
            teammate = data.get("teammate", "")
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{_c('cyan', f'👤 TEAMMATE PROFILE')} by {_bold(agent)} → {teammate}")
            continue

        if etype == "reflection.l2.correlation":
            partner = data.get("partner", "")
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{_c('cyan', f'🤝 PROFILE CORRELATION')} by {_bold(agent)} ↔ {partner}")
            continue

        if etype == "reflection.l2.skip":
            reason = data.get("reason", "")
            completed = data.get("completed", [])
            total = data.get("total", 0)
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{_c('cyan', f'⏭ PROFILE REFLECTION COMPLETE')} by {_bold(agent)}: {reason}")
            print(f"      progress: {len(completed)}/{total} completed: {', '.join(completed)}")
            continue

        if etype == "reflection.skipped":
            reason = data.get("reason", "")
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{_c('yellow', f'⏭ REFLECTION SKIPPED')}: {reason}")
            continue

        if etype == "reflection.validation_failed":
            fname = data.get("file", "")
            error = data.get("error", "")
            print(f"\n{_dim(f'[{elapsed_str}]')} "
                  f"{_c('red', f'❌ REFLECTION VALIDATION FAILED')}: {fname}")
            if error:
                _print_indented(error, indent=6)
            continue

        if etype == "gate.check":
            _ensure_agent_context(agent, elapsed_str)
            last_printed_agent = agent
            gate = data.get("gate", "")
            passed = data.get("passed", False)
            detail = data.get("detail", "")
            icon = _c("green", "✅ PASSED") if passed else _c("red", "❌ BLOCKED")
            print(f"   {_dim('🔒 GATE')} [{gate}] {icon}")
            if not passed and detail:
                display = _truncate(detail, text_limit) if text_limit else detail
                _print_indented(display, indent=6)
            continue

        if etype == "review.change":
            _ensure_agent_context(agent, elapsed_str)
            last_printed_agent = agent
            action = data.get("action", "")
            if action == "review_plan":
                verdict = data.get("verdict", "")
                comment = data.get("comment", "")
                icon = {"approve": "👍", "concern": "⚠️", "reject": "❌"}.get(verdict, "📋")
                print(f"   {icon} {_dim('REVIEW:')} {_bold(verdict)}")
                if comment:
                    display = _truncate(comment, text_limit) if text_limit else comment
                    _print_indented(display, indent=6)
            elif action == "raise_concern":
                severity = data.get("severity", "")
                description = data.get("description", "")
                icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(severity, "📋")
                print(f"   {icon} {_dim(f'CONCERN [{severity}]:')} {description[:200]}")
            elif action == "override_concern":
                cid = data.get("concern_id", "")
                reason = data.get("reason", "")
                print(f"   {_dim('⚡ OVERRIDE')} {cid}: {reason[:200]}")
            continue

        if etype == "agent.start":
            last_printed_agent = agent
            print(f"\n{_dim(f'[{elapsed_str}]')} {_c(color, f'{agent}')} {_dim('agent.start')}")
            continue

        if etype == "agent.finish":
            last_printed_agent = agent
            output = data.get("output", "")
            steps = data.get("steps", 0)
            print(f"   {_dim(f'■ FINISH — {steps} steps')}")
            if output:
                display = _truncate(output, text_limit) if text_limit else output
                _print_indented(display, indent=6)
            continue

    print()
    _print_footer(
        session_id, events, agents_seen, stats, agent_color,
        run_end_data, t0, meta,
    )



def _print_header(session_id: str, task: str, events: list, agents: list[str]) -> None:
    w = 68
    print()
    print(_bold("╔" + "═" * w + "╗"))
    print(_bold(f"║  Agent Team Trace — {session_id}".ljust(w + 2) + "║"))

    task_content_width = w - 4
    task_lines = task.split("\n")
    wrapped: list[str] = []
    for tl in task_lines:
        while len(tl) > task_content_width:
            wrapped.append(tl[:task_content_width])
            tl = tl[task_content_width:]
        wrapped.append(tl)
    if wrapped:
        first = f"║  Task: {wrapped[0]}"
        print(_bold(first.ljust(w + 2) + "║"))
        for line in wrapped[1:]:
            cont = f"║    {line}"
            print(_bold(cont.ljust(w + 2) + "║"))
    else:
        print(_bold(f"║  Task: (empty)".ljust(w + 2) + "║"))

    info = f"║  Events: {len(events)} | Agents: {len(agents)}"
    print(_bold(info.ljust(w + 2) + "║"))
    print(_bold("╚" + "═" * w + "╝"))


def _print_footer(
    session_id: str,
    events: list,
    agents: list[str],
    stats: dict,
    agent_color: dict,
    run_end: dict,
    t0: float,
    meta: dict | None = None,
) -> None:
    w = 68
    print(_bold("╔" + "═" * w + "╗"))
    print(_bold("║  Summary".ljust(w + 2) + "║"))
    print(_bold("╠" + "═" * w + "╣"))

    for a in agents:
        s = stats.get(a, {})
        color = agent_color.get(a, "white")
        line = (f"║  {_c(color, a)}: "
                f"{s.get('steps', 0)} steps | "
                f"{s.get('tool_calls', 0)} tool calls | "
                f"{s.get('msgs_sent', 0)} msgs sent | "
                f"tokens: {s.get('tokens_in', 0)}in/{s.get('tokens_out', 0)}out")
        print(line)

    reviews_from_end = run_end.get("reviews", 0)
    reviews_from_events = sum(
        1 for e in events
        if e.get("type") == "review.change"
        and e.get("data", {}).get("action") == "review_plan"
    )
    reviews = max(reviews_from_end, reviews_from_events)
    concerns = run_end.get("active_concerns", 0)
    dispatches = run_end.get("dispatches", 0)
    cost = run_end.get("cost_seconds", 0)
    cost_usd = run_end.get("cost_usd", 0)
    print(_bold("║" + "─" * w + "║"))
    summary_line = f"║  Reviews: {reviews} | Active concerns: {concerns} | "
    if dispatches:
        summary_line += f"Dispatches: {dispatches} | "
    summary_line += f"Duration: {cost:.1f}s"
    if cost_usd:
        summary_line += f" | Cost: ${cost_usd:.4f}"
    print(summary_line)

    pricing = ((meta or {}).get("summary") or {}).get("pricing")
    if pricing:
        total_in = sum(s.get("tokens_in", 0) for s in stats.values())
        total_out = sum(s.get("tokens_out", 0) for s in stats.values())
        ci = calc_cost(total_in, total_out,
                       pricing.get("input_price", 0), pricing.get("output_price", 0))
        print(f"║  Cost: ${ci['total_cost']:.6f}  "
              f"({total_in} in × ${pricing['input_price']}/1M + "
              f"{total_out} out × ${pricing['output_price']}/1M)")

    print(_bold("╚" + "═" * w + "╝"))


def _print_indented(text: str, indent: int = 6) -> None:
    prefix = " " * indent
    for line in text.split("\n"):
        print(f"{prefix}{line}")


def _format_args(tool: str, args: dict, text_limit: int) -> str:
    if not args:
        return ""
    parts = []
    effective_limit = text_limit if text_limit else 0
    for k, v in args.items():
        v_str = str(v)
        if effective_limit and len(v_str) > effective_limit:
            v_str = v_str[:effective_limit] + "..."
        elif not effective_limit:
            pass
        elif len(v_str) > 80:
            v_str = v_str[:80] + "..."
        parts.append(f"{k}={json.dumps(v_str, ensure_ascii=False)}")
    return ", ".join(parts)


def _message_flow_suffix(tool: str, data: dict) -> str:
    output = data.get("output", "")
    if tool == "send_message" and "Message sent" in output:
        return f"   {_dim('──── 📨 message')}"
    return ""
