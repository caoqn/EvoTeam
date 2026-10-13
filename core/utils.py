import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)



def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")



_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt_template(relative_path: str) -> str:
    path = _PROMPTS_DIR / relative_path
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def parse_ts(ts_str: str) -> float:
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def fmt_elapsed(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[+{len(text) - limit}]"


def load_session_data(session_id: str, sessions_dir: Path) -> tuple[list[dict], dict]:
    return load_session_data_from_dir(sessions_dir / session_id)


def load_session_data_from_dir(session_dir: Path) -> tuple[list[dict], dict]:
    session_dir = Path(session_dir)
    events_path = session_dir / "events.jsonl"
    if not events_path.exists():
        return [], {}
    events = []
    for line in events_path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping corrupt JSONL line in %s", events_path)
    meta_path = session_dir / "session.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return events, meta


def collect_agent_stats(events: list) -> tuple[list[str], dict]:
    agents_seen: list[str] = []
    for e in events:
        a = e.get("agent", "")
        if a and a not in agents_seen:
            agents_seen.append(a)

    stats: dict[str, dict] = {
        a: {"steps": 0, "tool_calls": 0, "msgs_sent": 0,
            "msgs_recv": 0, "tokens_in": 0, "tokens_out": 0,
            "phase_steps": {}}
        for a in agents_seen
    }

    current_phase = "task_execution"
    agent_phase_steps: dict[str, dict[str, list[int]]] = {
        a: {} for a in agents_seen
    }

    for e in events:
        a = e.get("agent", "")
        d = e.get("data", {})
        t = e.get("type", "")

        if t in ("committee.phase_change", "runner.phase_change"):
            current_phase = d.get("to", current_phase)
            continue

        if a and a in stats:
            if t == "llm.call":
                step = d.get("step", 0)
                stats[a]["steps"] = max(stats[a]["steps"], step)
                stats[a]["tokens_in"] += d.get("input_tokens", 0)
                stats[a]["tokens_out"] += d.get("output_tokens", 0)
                phase_list = agent_phase_steps[a].setdefault(current_phase, [])
                if step not in phase_list:
                    phase_list.append(step)
            elif t == "tool.call":
                stats[a]["tool_calls"] += 1
                if d.get("tool") == "send_message":
                    stats[a]["msgs_sent"] += 1
            elif t == "agent.wake":
                stats[a]["msgs_recv"] += d.get("count", 0)

    for a in agents_seen:
        phase_steps: dict[str, dict[int, int]] = {}
        for phase, steps in agent_phase_steps[a].items():
            sorted_steps = sorted(steps)
            mapping = {}
            for i, gs in enumerate(sorted_steps, 1):
                mapping[gs] = i
            phase_steps[phase] = mapping
        stats[a]["phase_steps"] = phase_steps

    return agents_seen, stats


def calc_cost(
    tokens_in: int,
    tokens_out: int,
    input_price: float,
    output_price: float,
) -> dict:
    cost_in = tokens_in * input_price / 1_000_000
    cost_out = tokens_out * output_price / 1_000_000
    return {
        "total_in": tokens_in,
        "total_out": tokens_out,
        "cost_in": cost_in,
        "cost_out": cost_out,
        "total_cost": cost_in + cost_out,
    }



_MODEL_PRICING: list[tuple[str, float, float]] = [
    ("gpt-5.6-luna", 0.20, 1.20),
    ("opus-4",        5.0,  25.0),
    ("sonnet-4",      3.0,  15.0),
    ("haiku-4",       1.0,   5.0),
    ("haiku-3",       0.25,  1.25),
    ("sonnet-3",      3.0,  15.0),
    ("opus-3",       15.0,  75.0),
    ("gpt-5-nano",   0.05,  0.40),
    ("gpt-5.1-codex", 2.50, 10.0),
    ("gpt-5",        1.25, 10.0),
    ("gpt-4o-mini",   0.15,  0.60),
    ("gpt-4o",        2.50, 10.0),
    ("gpt-4-turbo",  10.0,  30.0),
    ("o4-mini",       1.10,  4.40),
    ("o3-mini",       1.10,  4.40),
    ("deepseek",      0.27,  1.10),
]

_DEFAULT_PRICING = (3.0, 15.0)


def estimate_llm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    model_lower = model.lower()
    input_price, output_price = _DEFAULT_PRICING

    for substr, ip, op in _MODEL_PRICING:
        if substr in model_lower:
            input_price, output_price = ip, op
            break

    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
