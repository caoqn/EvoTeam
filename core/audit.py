import json
from collections import defaultdict
from core.session import SESSIONS_DIR
from core.utils import (parse_ts, fmt_elapsed, truncate, load_session_data,
                         load_session_data_from_dir, collect_agent_stats, calc_cost)



def _load(session_id: str, session_dir=None) -> tuple[list[dict], dict]:
    if session_dir:
        from pathlib import Path
        return load_session_data_from_dir(Path(session_dir))
    return load_session_data(session_id, SESSIONS_DIR)


_parse_ts = parse_ts
_fmt_elapsed = fmt_elapsed
_truncate = truncate


def _collect_stats(events: list) -> tuple[list[str], dict]:
    return collect_agent_stats(events)


def _group_by_agent(events: list) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        a = e.get("agent", "")
        if a:
            groups[a].append(e)
    return dict(groups)


def _build_steps(agent_events: list[dict]) -> list[dict]:
    steps: list[dict] = []
    current_step = 0
    current_events: list[dict] = []

    for e in agent_events:
        t = e.get("type", "")
        d = e.get("data", {})

        if t == "llm.call":
            step_num = d.get("step", 0)
            if step_num != current_step and current_events:
                steps.append({"step": current_step, "events": current_events})
                current_events = []
            current_step = step_num

        current_events.append(e)

    if current_events:
        steps.append({"step": current_step, "events": current_events})

    return steps


def _build_message_threads(events: list) -> list[dict]:
    t0 = _parse_ts(events[0].get("ts", "")) if events else 0
    messages = []
    for e in events:
        if e.get("type") != "tool.call":
            continue
        d = e.get("data", {})
        tool = d.get("tool", "")
        args = d.get("args", {})
        sender = e.get("agent", "")

        if tool == "send_message" and sender:
            elapsed = _parse_ts(e.get("ts", "")) - t0 if t0 else 0
            to = args.get("to", "")
            if isinstance(to, list):
                to_str = ", ".join(to)
            else:
                to_str = str(to)
            messages.append({
                "from": sender,
                "to": to_str,
                "content": args.get("content", ""),
                "ts": e.get("ts", ""),
                "elapsed": _fmt_elapsed(elapsed),
            })
    return messages


def _build_governance_trail(events: list) -> dict:
    t0 = _parse_ts(events[0].get("ts", "")) if events else 0
    reviews = []
    concerns = []
    gates = []

    for e in events:
        elapsed = _parse_ts(e.get("ts", "")) - t0 if t0 else 0
        t_str = _fmt_elapsed(elapsed)
        etype = e.get("type", "")
        agent = e.get("agent", "")
        d = e.get("data", {})

        if etype == "review.change":
            action = d.get("action", "")
            if action == "review_plan":
                reviews.append({
                    "time": t_str, "agent": agent,
                    "verdict": d.get("verdict", ""),
                    "comment": d.get("comment", ""),
                })
            elif action == "raise_concern":
                concerns.append({
                    "time": t_str, "agent": agent, "type": "raise",
                    "severity": d.get("severity", ""),
                    "description": d.get("description", ""),
                })
            elif action == "override_concern":
                concerns.append({
                    "time": t_str, "agent": agent, "type": "override",
                    "concern_id": d.get("concern_id", ""),
                    "reason": d.get("reason", ""),
                })
        elif etype == "gate.check":
            gates.append({
                "time": t_str, "gate": d.get("gate", ""),
                "passed": d.get("passed", False),
                "detail": d.get("detail", ""),
            })

    return {"reviews": reviews, "concerns": concerns, "gates": gates}



def _render_summary_txt(session_id: str, events: list, meta: dict) -> str:
    lines: list[str] = []
    w = lines.append

    task = meta.get("task", "")
    status = meta.get("status", "unknown")
    agents, stats = _collect_stats(events)
    t0 = _parse_ts(events[0].get("ts", "")) if events else 0
    t_last = _parse_ts(events[-1].get("ts", "")) if events else 0
    duration = t_last - t0 if t0 else 0
    total_tok_in = sum(s["tokens_in"] for s in stats.values())
    total_tok_out = sum(s["tokens_out"] for s in stats.values())

    w("=" * 72)
    w(f"AUDIT SUMMARY — {session_id}")
    w("=" * 72)
    w(f"Task:     {task}")
    w(f"Status:   {status}")
    w(f"Duration: {_fmt_elapsed(duration)}")
    w(f"Events:   {len(events)}")
    w(f"Agents:   {len(agents)}")
    w(f"Tokens:   {total_tok_in} in / {total_tok_out} out")

    pricing = (meta.get("summary") or {}).get("pricing")
    if pricing:
        ci = calc_cost(total_tok_in, total_tok_out,
                       pricing.get("input_price", 0), pricing.get("output_price", 0))
        w(f"Cost:     ${ci['total_cost']:.6f}  "
          f"(in: ${ci['cost_in']:.6f}  out: ${ci['cost_out']:.6f})")

    w("")

    w("AGENTS")
    w(f"  {'Name':<16} {'Steps':>5} {'Tools':>5} {'MsgOut':>6} {'MsgIn':>5} {'TokIn':>7} {'TokOut':>7}")
    w(f"  {'-'*16} {'-'*5} {'-'*5} {'-'*6} {'-'*5} {'-'*7} {'-'*7}")
    for a in agents:
        s = stats[a]
        w(f"  {a:<16} {s['steps']:>5} {s['tool_calls']:>5} "
          f"{s['msgs_sent']:>6} {s['msgs_recv']:>5} "
          f"{s['tokens_in']:>7} {s['tokens_out']:>7}")
    w("")

    gov = _build_governance_trail(events)
    if gov["reviews"] or gov["concerns"] or gov["gates"]:
        w("GOVERNANCE")
        for r in gov["reviews"]:
            w(f"  [{r['time']}] {r['agent']} review: {r['verdict']}")
            if r["comment"]:
                w(f"           {_truncate(r['comment'], 200)}")
        for c in gov["concerns"]:
            if c["type"] == "raise":
                w(f"  [{c['time']}] {c['agent']} concern [{c['severity']}]: "
                  f"{_truncate(c['description'], 200)}")
            else:
                w(f"  [{c['time']}] {c['agent']} override: {c['concern_id']} "
                  f"reason={_truncate(c.get('reason', ''), 100)}")
        for g in gov["gates"]:
            status_str = "PASS" if g["passed"] else "BLOCK"
            w(f"  [{g['time']}] gate[{g['gate']}]: {status_str}")
            if not g["passed"] and g["detail"]:
                w(f"           {_truncate(g['detail'], 200)}")
        w("")

    w("PER-AGENT STEPS")
    grouped = _group_by_agent(events)
    for a in agents:
        agent_evts = grouped.get(a, [])
        step_groups = _build_steps(agent_evts)
        w(f"  [{a}] ({len(step_groups)} step groups)")
        for sg in step_groups:
            step_num = sg["step"]
            evts = sg["events"]
            tools_used = []
            think_summary = ""
            for ev in evts:
                et = ev.get("type", "")
                ed = ev.get("data", {})
                if et == "tool.call":
                    tool_name = ed.get("tool", "")
                    if tool_name == "send_message":
                        to = ed.get("args", {}).get("to", "?")
                        if isinstance(to, list):
                            to_str = ",".join(to)
                        else:
                            to_str = str(to)
                        tools_used.append(f"send_message→{to_str}")
                    elif tool_name == "wait_for_replies":
                        tools_used.append("wait_for_replies")
                    elif tool_name in ("dispatch_to_team", "finalize_task",
                                       "review_plan", "raise_concern",
                                       "override_concern", "update_plan",
                                       "list_pool", "start_agent", "stop_agent",
                                       "set_final_output", "terminate"):
                        brief = _brief_tool(tool_name, ed.get("args", {}))
                        tools_used.append(brief)
                    else:
                        tools_used.append(tool_name)
                elif et == "agent.think" and not think_summary:
                    content = ed.get("content", "")
                    think_summary = _truncate(content.replace("\n", " "), 120)

            tools_str = ", ".join(tools_used) if tools_used else "(no tools)"
            w(f"    step {step_num}: {tools_str}")
            if think_summary:
                w(f"      think: {think_summary}")
        w("")

    messages = _build_message_threads(events)
    if messages:
        w("MESSAGE LOG")
        for msg in messages:
            content_preview = _truncate(msg["content"].replace("\n", " "), 150)
            w(f"  [{msg['elapsed']}] {msg['from']}→{msg['to']}: {content_preview}")
        w("")

    for e in reversed(events):
        if e.get("type") == "tool.call":
            d = e.get("data", {})
            if d.get("tool") in ("finalize_task", "set_final_output"):
                output = d.get("args", {}).get("output", "")
                if output:
                    w("FINAL OUTPUT")
                    w(_truncate(output, 1000))
                    w("")
                break

    anomalies = _detect_anomalies(events)
    if anomalies:
        w("ANOMALIES")
        for anomaly in anomalies:
            w(f"  ⚠ {anomaly}")
        w("")

    w("=" * 72)
    return "\n".join(lines)


def _brief_tool(tool: str, args: dict) -> str:
    if tool == "dispatch_to_team":
        team = args.get("team_name", "?")
        desc = _truncate(str(args.get("task_description", "")), 40)
        return f"dispatch→{team}({desc})"
    if tool == "finalize_task":
        return f"finalize({_truncate(str(args.get('output', '')), 30)})"
    if tool == "review_plan":
        return f"review({args.get('verdict', '')})"
    if tool == "raise_concern":
        return f"concern[{args.get('severity', '')}]"
    if tool == "override_concern":
        return f"override({args.get('concern_id', '')})"
    if tool == "update_plan":
        return "update_plan"
    if tool == "list_pool":
        return "list_pool"
    if tool == "start_agent":
        return f"start({args.get('agent_name', '?')})"
    if tool == "stop_agent":
        return f"stop({args.get('agent_name', '?')})"
    if tool == "set_final_output":
        return f"set_output({_truncate(str(args.get('output', '')), 30)})"
    if tool == "terminate":
        return f"terminate({args.get('reason', '')})"
    return tool


def _detect_anomalies(events: list) -> list[str]:
    anomalies = []

    for e in events:
        etype = e.get("type", "")
        if etype == "committee.timeout":
            anomalies.append(
                f"Committee timeout at max_seconds={e.get('data', {}).get('max_seconds')}")
        elif etype == "runner.timeout":
            anomalies.append(
                f"Runner timeout at max_seconds={e.get('data', {}).get('max_seconds')}"
                f" (elapsed={e.get('data', {}).get('elapsed', '?')}s)")
        elif etype == "runner.guard":
            reason = e.get("data", {}).get("reason", "")
            anomalies.append(f"Runner guard triggered: {reason}")

    for e in events:
        if e.get("type") == "gate.check" and not e.get("data", {}).get("passed"):
            gate = e.get("data", {}).get("gate", "")
            anomalies.append(f"Gate [{gate}] blocked")

    raised = {}
    overridden = set()
    for e in events:
        if e.get("type") == "review.change":
            d = e.get("data", {})
            if d.get("action") == "raise_concern" and d.get("severity") == "critical":
                cid = d.get("concern_id", "")
                if cid:
                    raised[cid] = e.get("agent", "")
            elif d.get("action") == "override_concern":
                overridden.add(d.get("concern_id", ""))
    for cid, agent in raised.items():
        if cid not in overridden:
            anomalies.append(f"Unresolved critical concern '{cid}' by {agent}")

    for e in events:
        if e.get("type") == "agent.loop.terminated":
            reason = e.get("data", {}).get("reason", "")
            if reason != "finalized":
                anomalies.append(
                    f"{e.get('agent', '')} terminated: {reason}")

    return anomalies



def _render_report_html(session_id: str, events: list, meta: dict) -> str:
    from html import escape as esc

    task = meta.get("task", "")
    status = meta.get("status", "unknown")
    agents, stats = _collect_stats(events)
    t0 = _parse_ts(events[0].get("ts", "")) if events else 0
    t_last = _parse_ts(events[-1].get("ts", "")) if events else 0
    duration = t_last - t0 if t0 else 0
    total_tok_in = sum(s["tokens_in"] for s in stats.values())
    total_tok_out = sum(s["tokens_out"] for s in stats.values())

    grouped = _group_by_agent(events)
    gov = _build_governance_trail(events)
    messages = _build_message_threads(events)
    anomalies = _detect_anomalies(events)

    palette = ["#61afef", "#98c379", "#e5c07b", "#c678dd", "#56b6c2",
               "#e06c75", "#d19a66", "#a9b1d6", "#7aa2f7", "#9ece6a",
               "#e0af68", "#bb9af7"]
    agent_color = {a: palette[i % len(palette)] for i, a in enumerate(agents)}

    parts: list[str] = []
    p = parts.append

    p("<!DOCTYPE html>")
    p("<html lang='en'><head><meta charset='utf-8'>")
    p(f"<title>Audit — {esc(session_id)}</title>")
    p("<style>")
    p("""
body { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
       background: #1e1e2e; color: #cdd6f4; margin: 0; padding: 20px;
       font-size: 13px; line-height: 1.6; }
h1 { color: #89b4fa; font-size: 18px; margin-bottom: 4px; }
h2 { color: #a6e3a1; font-size: 15px; margin-top: 24px; border-bottom: 1px solid #45475a;
     padding-bottom: 4px; }
h3 { color: #f9e2af; font-size: 13px; margin: 12px 0 4px 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; }
th, td { text-align: left; padding: 4px 10px; border-bottom: 1px solid #313244; }
th { color: #89b4fa; font-weight: normal; }
.stat { color: #bac2de; }
.anomaly { color: #f38ba8; font-weight: bold; }
.pass { color: #a6e3a1; }
.fail { color: #f38ba8; }
.dim { opacity: 0.6; }
.msg-from { font-weight: bold; }
.msg-content { white-space: pre-wrap; background: #181825; padding: 6px 10px;
               border-radius: 4px; margin: 2px 0 8px 20px; max-height: 300px;
               overflow-y: auto; border-left: 3px solid #45475a; }
details { margin: 2px 0; }
details > summary { cursor: pointer; padding: 4px 0; }
details > summary:hover { color: #89b4fa; }
details[open] > summary { color: #a6e3a1; }
.step-box { margin-left: 16px; padding: 4px 0 4px 12px;
            border-left: 2px solid #313244; margin-bottom: 4px; }
.tool-call { color: #cba6f7; }
.think { color: #94e2d5; font-style: italic; }
.result { color: #a6adc8; }
.badge { display: inline-block; padding: 1px 6px; border-radius: 3px;
         font-size: 11px; margin-left: 4px; }
.badge-approve { background: #2d4a2d; color: #a6e3a1; }
.badge-concern { background: #4a3d1d; color: #f9e2af; }
.badge-reject  { background: #4a1d1d; color: #f38ba8; }
.badge-critical { background: #4a1d1d; color: #f38ba8; }
.badge-warning  { background: #4a3d1d; color: #f9e2af; }
.toc { position: sticky; top: 0; background: #1e1e2e; padding: 8px 0;
       border-bottom: 1px solid #45475a; z-index: 10; }
.toc a { color: #89b4fa; text-decoration: none; margin-right: 16px; }
.toc a:hover { text-decoration: underline; }
pre { white-space: pre-wrap; word-break: break-all; }
""")
    p("</style></head><body>")
    p(f"<h1>Audit: {esc(session_id)}</h1>")
    p(f"<p>Status: {esc(status)} | Duration: {duration:.1f}s | "
      f"Tokens: {total_tok_in}in/{total_tok_out}out</p>")

    p("<h2>Agent Stats</h2><table><tr><th>Agent</th><th>Steps</th>"
      "<th>Tool Calls</th><th>Tokens In</th><th>Tokens Out</th></tr>")
    for a in agents:
        s = stats[a]
        color = agent_color.get(a, "#cdd6f4")
        p(f"<tr><td style='color:{color}'>{esc(a)}</td>"
          f"<td>{s['steps']}</td><td>{s['tool_calls']}</td>"
          f"<td>{s['tokens_in']}</td><td>{s['tokens_out']}</td></tr>")
    p("</table>")

    if anomalies:
        p("<h2>Anomalies</h2><ul>")
        for a in anomalies:
            p(f"<li class='anomaly'>{esc(a)}</li>")
        p("</ul>")

    if messages:
        p("<h2>Messages</h2>")
        for msg in messages[:50]:
            sender = msg.get("sender", "?")
            receiver = msg.get("receiver", "?")
            content = msg.get("content", "")[:500]
            p(f"<div><span class='msg-from'>{esc(sender)} &rarr; {esc(receiver)}</span>"
              f"<div class='msg-content'><pre>{esc(content)}</pre></div></div>")

    p("<h2>Agent Execution Details</h2>")
    for agent_name in agents:
        color = agent_color.get(agent_name, "#cdd6f4")
        agent_events = grouped.get(agent_name, [])
        steps = _build_steps(agent_events)
        p(f"<h3 style='color:{color}'>{esc(agent_name)} ({len(steps)} steps)</h3>")
        for step in steps[:30]:
            p("<details><summary>")
            if step.get("tool"):
                p(f"<span class='tool-call'>{esc(step['tool'])}</span>")
            if step.get("think"):
                think_preview = step["think"][:100]
                p(f" <span class='think'>{esc(think_preview)}</span>")
            p("</summary>")
            if step.get("result"):
                result_preview = step["result"][:500]
                p(f"<div class='step-box'><pre class='result'>{esc(result_preview)}</pre></div>")
            p("</details>")

    if gov:
        p("<h2>Governance Trail</h2>")
        for verdict_type, verdicts in gov.items():
            p(f"<h3>{esc(verdict_type)}</h3><ul>")
            for v in verdicts[:20]:
                badge_class = f"badge-{v.get('level', 'approve')}"
                p(f"<li><span class='badge {badge_class}'>{esc(v.get('level', ''))}</span> "
                  f"{esc(v.get('summary', '')[:200])}</li>")
            p("</ul>")

    p("</body></html>")
    return "\n".join(parts)


def audit(session_id: str, session_dir=None, quiet: bool = False):
    events, meta = _load(session_id, session_dir=session_dir)
    if not events:
        if not quiet:
            print(f"No events found for session {session_id}")
        return None

    if session_dir:
        from pathlib import Path
        out_dir = Path(session_dir)
    else:
        out_dir = SESSIONS_DIR / session_id

    summary_txt = _render_summary_txt(session_id, events, meta)
    summary_path = out_dir / "audit-summary.txt"
    summary_path.write_text(summary_txt, encoding="utf-8")

    report_html = _render_report_html(session_id, events, meta)
    report_path = out_dir / "audit-report.html"
    report_path.write_text(report_html, encoding="utf-8")

    if not quiet:
        print(f"audit-summary.txt: {summary_path} ({len(summary_txt)} chars)")
        print(f"audit-report.html: {report_path} ({len(report_html)} chars)")

    return str(summary_path), str(report_path)