import asyncio
import sys
import os
import json
import logging
import argparse
import threading
from pathlib import Path
from collections import defaultdict
from typing import Any

_BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BASE_DIR))

LOCABENCH_DIR = _BASE_DIR / "benchmarks" / "LOCA-bench"
sys.path.insert(0, str(LOCABENCH_DIR))
sys.path.insert(0, str(LOCABENCH_DIR / "mcp_convert"))

from benchmarks.adapter import BenchmarkAdapter, EvalResult, EnvContext
from core.execution_policy import (
    BoundExecutionPolicy,
    ExecutionPolicy,
    FallbackPolicy,
)

os.environ["LOCA_QUIET"] = "1"
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)

_eval_lock = threading.Lock()

LOCA_NATIVE_CONTRACT_PATH = "agent_workspace/native_contract.json"


def _native_contract_skeleton(
    *,
    task_name: str,
    context_level: str,
    execution_mode: str,
    tool_catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_id": "loca_native_task_contract_v1",
        "contract_path": LOCA_NATIVE_CONTRACT_PATH,
        "status": "draft",
        "task": {
            "task_name": task_name,
            "context_level": context_level,
            "execution_mode": execution_mode,
        },
        "available_service_capabilities": [
            {
                "name": str(tool.get("name") or ""),
                "description": str(tool.get("description") or "")[:300],
            }
            for tool in tool_catalog
            if isinstance(tool, dict) and tool.get("name")
        ],
        "authoritative_sources": [],
        "selection": {
            "population_definition": "",
            "inclusion_rules": [],
            "exclusion_rules": [],
            "deduplication_key": "",
            "ordering_rules": [],
            "source_count": None,
            "target_count": None,
            "coverage_status": "unresolved",
            "unresolved_items": [],
        },
        "transformations": [],
        "artifacts": [],
        "mutations": [],
        "payload_contracts": [],
        "protected_state": [],
        "verification": {
            "requirements": [],
            "authoritative_readbacks": [],
            "expected": None,
            "completed": 0,
            "verified": 0,
            "failed": 0,
            "remaining": None,
            "missing": None,
            "duplicate": None,
            "unintended_extra": None,
            "unverified": None,
            "status": "pending",
        },
        "completion_gate": {
            "contract_resolved": False,
            "source_coverage_complete": False,
            "selection_reconciled": False,
            "artifacts_verified": False,
            "mutations_read_back": False,
            "payloads_verified": False,
            "protected_state_verified": False,
            "no_pending_or_unverified_items": False,
            "ready": False,
            "blocking_reasons": [],
        },
        "evidence": {
            "manifest_path": "agent_workspace/task_manifest.json",
            "parameters_path": "agent_workspace/task_params.json",
            "verification_report_path": "agent_workspace/verification_report.json",
        },
    }


def _ensure_native_contract_skeleton(
    task_dir: str | Path,
    *,
    task_name: str,
    context_level: str,
    execution_mode: str,
    tool_catalog: list[dict[str, Any]],
) -> Path:
    contract_path = Path(task_dir) / LOCA_NATIVE_CONTRACT_PATH
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    if not contract_path.exists():
        contract_path.write_text(
            json.dumps(
                _native_contract_skeleton(
                    task_name=task_name,
                    context_level=context_level,
                    execution_mode=execution_mode,
                    tool_catalog=tool_catalog,
                ),
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    return contract_path


def _validate_native_contract(contract_path: str | Path) -> dict[str, Any]:
    path = Path(contract_path)
    result: dict[str, Any] = {
        "path": str(path),
        "present": path.exists(),
        "valid": False,
        "status": None,
        "issues": [],
    }
    if not path.exists():
        result["issues"].append("native contract file is missing")
        result["issue_count"] = 1
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["issues"].append(f"native contract is not valid JSON: {exc}")
        result["issue_count"] = 1
        return result
    if not isinstance(payload, dict):
        result["issues"].append("native contract root must be an object")
        result["issue_count"] = 1
        return result

    status = str(payload.get("status") or "").strip().lower()
    result["status"] = status or None
    if status in {"", "draft", "failed", "completed_with_failures"}:
        result["issues"].append(
            f"contract status must be a completed state, got {status or 'missing'!r}"
        )

    gate = payload.get("completion_gate")
    if not isinstance(gate, dict):
        result["issues"].append("completion_gate is missing or not an object")
        gate = {}
    for key in (
        "contract_resolved",
        "source_coverage_complete",
        "selection_reconciled",
        "no_pending_or_unverified_items",
        "ready",
    ):
        if gate.get(key) is not True:
            result["issues"].append(f"completion_gate.{key} is not true")
    if gate.get("blocking_reasons"):
        result["issues"].append("completion_gate has blocking_reasons")

    selection = payload.get("selection")
    if not isinstance(selection, dict):
        result["issues"].append("selection is missing or not an object")
    elif str(selection.get("coverage_status") or "").lower() != "complete":
        result["issues"].append("selection.coverage_status is not complete")

    verification = payload.get("verification")
    if not isinstance(verification, dict):
        result["issues"].append("verification is missing or not an object")
        verification = {}
    for key in ("failed", "missing", "duplicate", "unintended_extra", "unverified"):
        value = verification.get(key)
        if isinstance(value, (int, float)) and value > 0:
            result["issues"].append(f"verification.{key}={value}")
    expected = verification.get("expected")
    verified = verification.get("verified")
    if isinstance(expected, (int, float)) and isinstance(verified, (int, float)):
        if expected > 0 and verified < expected:
            result["issues"].append(
                f"verification.verified={verified} is below expected={expected}"
            )

    for field, gate_key in {
        "artifacts": "artifacts_verified",
        "mutations": "mutations_read_back",
        "payload_contracts": "payloads_verified",
        "protected_state": "protected_state_verified",
    }.items():
        value = payload.get(field)
        if isinstance(value, list) and value and gate.get(gate_key) is not True:
            result["issues"].append(
                f"{field} contains entries but completion_gate.{gate_key} is not true"
            )

    result["valid"] = not result["issues"]
    result["ready"] = gate.get("ready") is True
    result["issue_count"] = len(result["issues"])
    return result


def _fallback_contract(task_dir: str | Path, servers: list[str]) -> dict[str, dict[str, Any]]:
    root = Path(task_dir) / "local_db"
    contracts: dict[str, dict[str, Any]] = {}
    for server in servers:
        key = str(server).lower()
        if key == "woocommerce":
            paths = [root / "woocommerce" / "products.json", root / "woocommerce" / "orders.json"]
        elif key == "email":
            users_data = root / "emails" / "users_data"
            paths = (
                list(users_data.glob("*/emails.json"))
                + list(users_data.glob("*/folders.json"))
                if users_data.exists() else []
            )
        elif key == "google_cloud":
            paths = [root / "google_cloud" / "bigquery_data.db"]
        elif key == "google_sheet":
            paths = [root / "google_sheet" / "cells.json", root / "google_sheet" / "rows.json"]
        else:
            paths = []
        contracts[key] = {
            "mode": "local_db_available" if any(path.exists() for path in paths) else "unavailable",
            "paths": [str(path.relative_to(Path(task_dir))) for path in paths if path.exists()],
        }
    return contracts


def _validate_mcp_catalog(config: dict, catalog: list[dict], task_dir: str | Path) -> dict[str, dict[str, Any]]:
    servers = [str(name) for name in config.get("mcp_servers", {})]
    tool_names = [str(tool.get("name", "")).lower() for tool in catalog if isinstance(tool, dict)]
    contracts = _fallback_contract(task_dir, servers)
    if not tool_names:
        unavailable = [
            server for server in servers
            if contracts.get(server, {}).get("mode") == "unavailable"
        ]
        if unavailable:
            raise ConnectionError(
                "LOCA MCP initialization exposed no tools and no local_db "
                f"fallback for enabled servers: {unavailable}"
            )
    return contracts


def _ensure_email_accounts(config: dict, task_dir: str | Path) -> list[dict[str, str]]:
    email_server = config.get("mcp_servers", {}).get("email", {})
    if not email_server or not email_server.get("enabled", True):
        return []

    task_root = Path(task_dir)
    params = email_server.get("params", {})
    data_dir_value = str(params.get("data_dir") or "{task_workspace}/local_db/emails")
    data_dir = Path(
        data_dir_value
        .replace("{task_workspace}", str(task_root))
        .replace("{agent_workspace}", str(task_root / "agent_workspace"))
    )

    identities: dict[str, dict[str, str]] = {}

    def add_identity(raw: Any, source: str) -> None:
        if not isinstance(raw, dict):
            return
        email = str(raw.get("email") or "").strip()
        password = str(raw.get("password") or "").strip()
        if not email or "@" not in email or not password:
            return
        identities[email] = {
            "email": email,
            "password": password,
            "name": str(raw.get("name") or email.split("@", 1)[0]).strip(),
            "source": source,
        }

    add_identity(params, "mcp_config")
    config_candidates = [
        task_root / "emails_config.json",
        task_root / "initial_workspace" / "emails_config.json",
        task_root / "agent_workspace" / "emails_config.json",
    ]
    for path in config_candidates:
        if not path.exists():
            continue
        try:
            add_identity(json.loads(path.read_text(encoding="utf-8")), str(path.name))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConnectionError(f"invalid LOCA email account config {path}: {exc}") from exc

    if not identities:
        return []

    _ensure_pythonpath()
    from mcps.email.database_utils import EmailDatabase
    db = EmailDatabase(data_dir=str(data_dir))
    repaired: list[dict[str, str]] = []
    for email, identity in identities.items():
        changed = False
        if email not in db.users:
            db.create_user(email, identity["name"], identity["password"])
            changed = True
        else:
            user = db.users[email]
            if user.get("password") != identity["password"]:
                user["password"] = identity["password"]
                changed = True
            if identity["name"] and user.get("name") != identity["name"]:
                user["name"] = identity["name"]
                changed = True
            if changed:
                db._save_json_file("users.json", db.users)

        user_dir = data_dir / "users_data" / email
        required = ("emails.json", "folders.json", "drafts.json")
        if not all((user_dir / filename).exists() for filename in required):
            defaults = {
                "emails.json": {},
                "folders.json": {
                    "INBOX": {"name": "INBOX", "total": 0, "unread": 0},
                    "Sent": {"name": "Sent", "total": 0, "unread": 0},
                    "Drafts": {"name": "Drafts", "total": 0, "unread": 0},
                    "Trash": {"name": "Trash", "total": 0, "unread": 0},
                },
                "drafts.json": {},
            }
            user_dir.mkdir(parents=True, exist_ok=True)
            for filename, default in defaults.items():
                target = user_dir / filename
                if not target.exists():
                    db._save_json_file(str(target), default)
                    changed = True

        db.login(email, identity["password"])
        db.logout()
        repaired.append({
            "email": email,
            "source": identity["source"],
            "status": "repaired" if changed else "ready",
        })

    return repaired


async def _run_with_progress_guard(awaitable, run_dir: Path, case_index: int, timeout: float):
    task = asyncio.create_task(awaitable)
    started = asyncio.get_running_loop().time()
    last_progress = started
    event_path: Path | None = None
    read_offset = 0
    stall_limit = float(os.environ.get("LOCA_PROGRESS_STALL_TIMEOUT", "600"))
    idle_event_types = {"llm.call", "agent.think", "agent.loop.idle"}
    try:
        while not task.done():
            remaining = timeout - (asyncio.get_running_loop().time() - started)
            if remaining <= 0:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise asyncio.TimeoutError("LOCA case task timeout exhausted")
            await asyncio.sleep(min(30.0, remaining))
            if event_path is None:
                matches = sorted((run_dir / "cases").glob(f"{case_index:03d}_*/events.jsonl"))
                if matches:
                    event_path = matches[-1]
            try:
                if event_path is not None:
                    with event_path.open("r", encoding="utf-8", errors="replace") as stream:
                        stream.seek(read_offset)
                        new_lines = stream.readlines()
                        read_offset = stream.tell()
                else:
                    new_lines = []
            except OSError:
                new_lines = []

            meaningful = False
            for line in new_lines:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(event.get("type") or "") not in idle_event_types:
                    meaningful = True
                    break
            if meaningful:
                last_progress = asyncio.get_running_loop().time()
            if asyncio.get_running_loop().time() - last_progress >= stall_limit:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise ConnectionError(
                    "LOCA coordination stall: no meaningful event progress for "
                    f"{int(stall_limit)}s; preserving the previous evolution version"
                )
        return task.result()
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        raise


def _completion_protocol_status(event_text: str) -> dict[str, Any]:
    set_final_output = False
    finalize_task = False
    terminate = False
    runner_end = False
    for line in (event_text or "").splitlines():
        try:
            event = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        event_type = str(event.get("type") or "").lower()
        if event_type in {"runner.end", "runner_end"}:
            runner_end = True
        if event_type != "tool.call":
            continue
        data = event.get("data") or {}
        tool_name = str(data.get("tool") or data.get("name") or "").lower()
        set_final_output |= tool_name in {"set_final_output", "set-final-output"}
        finalize_task |= tool_name in {"finalize_task", "finalize-task"}
        terminate |= tool_name == "terminate"
    final_output_submitted = set_final_output or finalize_task
    if finalize_task:
        finalization_tool = "finalize_task"
    elif set_final_output:
        finalization_tool = "set_final_output"
    else:
        finalization_tool = None
    return {
        "set_final_output": set_final_output,
        "finalize_task": finalize_task,
        "finalization_tool": finalization_tool,
        "final_output_submitted": final_output_submitted,
        "terminate": terminate,
        "runner_end": runner_end,
        "complete_handshake": bool(
            final_output_submitted and (terminate or runner_end)
        ),
    }


TASK_CONFIGS_DIR = LOCABENCH_DIR / "task-configs"

CONTEXT_LEVELS = [
    "8k", "16k", "32k", "64k", "96k", "128k", "256k",
    "evolve_96k",
]

STD_EVAL_SEEDS = [42, 123, 456, 789, 2024]
EVOLVE_SEEDS = [101, 102]

S2L_TASK_NAMES = [
    "ABTestingS2LEnv",
    "AcademicWarningS2LEnv",
    "ApplyPhDEmailS2LEnv",
    "CanvasArrangeExamS2LEnv",
    "CanvasListTestS2LEnv",
    "CourseAssistantS2LEnv",
    "ExcelMarketResearchS2LEnv",
    "FilterLowSellingProductsS2LEnv",
    "MachineOperatingS2LEnv",
    "NhlB2bAnalysisS2LEnv",
    "PayableInvoiceCheckerS2LEnv",
    "SetConfCrDdlS2LEnv",
    "UpdateMaterialInventoryS2LEnv",
    "WoocommerceNewWelcomeS2LEnv",
    "WoocommerceStockAlertS2LEnv",
]

TASK_SHORT_NAMES = {
    "ABTesting": "ABTestingS2LEnv",
    "AcademicWarning": "AcademicWarningS2LEnv",
    "ApplyPhDEmail": "ApplyPhDEmailS2LEnv",
    "CanvasArrangeExam": "CanvasArrangeExamS2LEnv",
    "CanvasListTest": "CanvasListTestS2LEnv",
    "CourseAssistant": "CourseAssistantS2LEnv",
    "ExcelMarketResearch": "ExcelMarketResearchS2LEnv",
    "FilterLowSelling": "FilterLowSellingProductsS2LEnv",
    "MachineOperating": "MachineOperatingS2LEnv",
    "NhlB2bAnalysis": "NhlB2bAnalysisS2LEnv",
    "PayableInvoiceChecker": "PayableInvoiceCheckerS2LEnv",
    "SetConfCrDdl": "SetConfCrDdlS2LEnv",
    "UpdateMaterialInventory": "UpdateMaterialInventoryS2LEnv",
    "WoocommerceNewWelcome": "WoocommerceNewWelcomeS2LEnv",
    "WoocommerceStockAlert": "WoocommerceStockAlertS2LEnv",
}



def load_locabench_configs(
    context_level: str = "128k",
    task_filter: str | None = None,
    seed_filter: int | None = None,
) -> list[dict]:
    if context_level == "evolve_96k":
        config_file = TASK_CONFIGS_DIR / "evolve_96k_set_config.json"
        display_level = "96k"
    else:
        config_file = TASK_CONFIGS_DIR / f"final_{context_level}_set_config.json"
        display_level = context_level

    if not config_file.exists():
        raise FileNotFoundError(
            f"LOCA-bench config not found: {config_file}\n"
            f"Available: {[f.name for f in TASK_CONFIGS_DIR.glob('*.json')]}"
        )

    with open(config_file, encoding="utf-8") as f:
        data = json.load(f)

    configs = data.get("configurations", [])

    for i, c in enumerate(configs):
        c["_index"] = i
        c["_context_level"] = display_level
        c["_split"] = context_level
        c["_seed"] = c.get("env_params", {}).get("seed", 0)
        c["_task_name"] = c.get("name", "")

    if task_filter:
        full_name = TASK_SHORT_NAMES.get(task_filter, task_filter)
        configs = [c for c in configs if c["_task_name"] == full_name]

    if seed_filter is not None:
        configs = [c for c in configs if c["_seed"] == seed_filter]

    return configs



def _create_loca_env(config: dict, task_dir: str):
    from inference.run_react import dynamic_import_class

    _ensure_pythonpath()

    EnvClass = dynamic_import_class(config["env_class"])
    env_params = dict(config["env_params"])
    env_params["task_dir"] = task_dir

    with _eval_lock:
        return EnvClass(**env_params)


def _ensure_pythonpath():
    import os
    loca_root = str(LOCABENCH_DIR.resolve())
    mcp_root = str((LOCABENCH_DIR / "mcp_convert").resolve())

    interpreter_dir = str(Path(sys.executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if interpreter_dir not in path_entries:
        os.environ["PATH"] = os.pathsep.join([interpreter_dir, *path_entries])

    current = os.environ.get("PYTHONPATH", "")
    paths = current.split(os.pathsep) if current else []

    changed = False
    if loca_root not in paths:
        paths.insert(0, loca_root)
        changed = True
    if mcp_root not in paths:
        paths.insert(1, mcp_root)
        changed = True

    if changed:
        os.environ["PYTHONPATH"] = os.pathsep.join(paths)


def _setup_mcp_tool(config: dict, task_dir: str):
    import threading
    from inference.run_react import setup_mcp_servers
    from gem.tools.mcp_tool import MCPTool

    _ensure_pythonpath()
    task_ws = Path(task_dir)
    agent_ws = task_ws / "agent_workspace"
    agent_ws.mkdir(parents=True, exist_ok=True)

    result_holder = {}

    def _init_in_thread():
        try:
            mcp_config = setup_mcp_servers(config["mcp_servers"], task_ws, agent_ws)
            tool = MCPTool(mcp_config, validate_on_init=False, execution_timeout=120.0)
            _register_mcp_tool(tool)
            available = tool.get_available_tools()
            result_holder["tool"] = tool
            result_holder["catalog"] = available
        except Exception as exc:
            result_holder["error"] = exc

    thread = threading.Thread(target=_init_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=120)
    if "error" in result_holder:
        raise result_holder["error"]
    if "tool" not in result_holder:
        raise TimeoutError("MCP tool initialization timed out (120s)")

    tool = result_holder["tool"]
    try:
        available = result_holder["catalog"]
        catalog = [
            {"name": t["name"], "description": t.get("description", "")}
            for t in available
        ]
        return tool, catalog
    except Exception:
        try:
            tool.close()
        finally:
            _unregister_mcp_tool(tool)
        raise


def _check_preprocess_integrity(task_dir: str, task_name: str, strict: bool = False) -> None:
    td = Path(task_dir)
    local_db = td / "local_db"
    agent_ws = td / "agent_workspace"

    local_db_files = sum(1 for _ in local_db.rglob("*") if _.is_file()) if local_db.exists() else 0
    agent_ws_files = sum(1 for _ in agent_ws.rglob("*") if _.is_file()) if agent_ws.exists() else 0

    if local_db_files == 0 and agent_ws_files == 0:
        msg = (
            f"  [LOCAbench] ⚠️  WARNING: preprocess for {task_name} produced no data!\n"
            f"  [LOCAbench]   local_db/ has {local_db_files} files, agent_workspace/ has {agent_ws_files} files.\n"
            f"  [LOCAbench]   This usually means the preprocess subprocess failed silently.\n"
            f"  [LOCAbench]   Check PYTHONPATH or run: cd benchmarks/LOCA-bench && python gem/envs/*/preprocess/main.py --help\n"
            f"  [LOCAbench]   The task will run but evaluation will return reward=0.0."
        )
        print(msg)
        if strict:
            raise RuntimeError(
                f"preprocess for {task_name} produced no data in {task_dir} "
                f"(local_db={local_db_files}, agent_workspace={agent_ws_files}); "
                f"refusing to proceed under LOCA_STRICT_PREPROCESS=1"
            )
    else:
        print(f"  [LOCAbench] Preprocess OK: local_db={local_db_files} files, agent_workspace={agent_ws_files} files")



def _extract_eval_log(task_dir: str, task_name: str) -> str:
    if not task_dir:
        return ""

    candidates = [
        Path(task_dir) / "logs" / "env.log",
        Path(task_dir) / "workspace" / "logs" / "env.log",
    ]
    content = ""
    for log_path in candidates:
        if log_path.exists():
            try:
                content = log_path.read_text(encoding="utf-8", errors="replace")
                break
            except Exception:
                continue
    if not content:
        return ""

    try:
        marker = "Starting task evaluation"
        idx = content.rfind(marker)
        if idx < 0:
            for alt in ["Evaluation result", "reward", "PASS", "FAIL",
                         "expected", "actual", "mismatch", "incorrect"]:
                idx = content.rfind(alt)
                if idx >= 0:
                    idx = content.rfind("\n", 0, idx)
                    if idx < 0:
                        idx = 0
                    break

        if idx < 0:
            return content[-2000:].strip() if len(content) > 2000 else content.strip()

        eval_section = content[idx:]
        if len(eval_section) > 3000:
            eval_section = eval_section[:3000] + "\n... (truncated)"
        return eval_section.strip()
    except Exception:
        return ""



import atexit
import threading

_owned_mcp_tools: list = []
_mcp_lock = threading.Lock()


def _register_mcp_tool(tool) -> None:
    with _mcp_lock:
        _owned_mcp_tools.append(tool)


def _unregister_mcp_tool(tool) -> None:
    with _mcp_lock:
        try:
            _owned_mcp_tools.remove(tool)
        except ValueError:
            pass


def _cleanup_mcp_tools() -> None:
    with _mcp_lock:
        to_clean = list(_owned_mcp_tools)
    if not to_clean:
        return
    for tool in to_clean:
        try:
            tool.close()
        except Exception:
            pass


atexit.register(_cleanup_mcp_tools)



class LOCABenchAdapter(BenchmarkAdapter):
    benchmark_name = "locabench"
    default_team = "pool_EvoTeam"
    preserve_timeout_handoff_trace = False
    aggregate_handoff_reflection_by_pair = False
    default_timeout = 7200.0
    default_evolve_timeout = 7200.0
    default_max_cost = 12.0
    default_split = "128k"
    split_choices = CONTEXT_LEVELS

    def execution_policy(self, item: dict) -> ExecutionPolicy:
        return ExecutionPolicy(
            name="stateful_service_operations",
            version="2",
            environment_mode="native_mcp",
            allowed_tools=("read_file", "write_file", "bash", "loca_mcp"),
            artifact_requirements=(
                "Resolve authoritative sources, selection rules, transformations, exact artifacts, external mutations, payloads, protected state, and verification requirements before claiming completion.",
                "For multi-item work, maintain agent_workspace/task_manifest.json with stable source IDs, non-overlapping ownership, and pending/completed/failed/unverified status.",
                "Record stable identities, destinations, schemas, templates, timezones, and other task parameters in agent_workspace/task_params.json.",
                "Verify evaluator-visible artifact paths and authoritative external state; a corrected local file is not proof that an earlier submission or mutation was corrected.",
                "Report exact expected, completed, verified, failed, remaining, missing, duplicate, unintended-extra, and unverified counts where applicable.",
            ),
            chairman_instructions=(
                "Use an execution-first workflow: once a record's task-required target, payload, and preconditions are verified, execute it without waiting for the entire contract or all reports to be complete. Assign one writer per target; readers and verifiers remain read-only unless explicitly reassigned ownership.",
                "Recruit by required work product: source coverage, analytical selection, cross-system mapping, local artifact creation, controlled mutation, and independent verification.",
                "Every artifact or mutation handoff must identify its source evidence, exact target/destination, schema or payload, preconditions, postconditions, idempotency or duplicate policy, and required read-back.",
                "Do not finalize from aggregate counts or teammate narrative alone; require independent evidence for every applicable completion-gate clause.",
                "If a worker stalls, inspect its checkpoint and authoritative state, then reassign ownership or take over only unresolved items. Pending work, many tool calls, incomplete paperwork, or a repairable schema error are not terminal blockers. Continue while budget remains and safe task actions are possible; do not request a final blocked report merely because a batch is incomplete.",
            ),
            special_rules=(
                "Use one authoritative writer per mutable resource unless disjoint target ownership is explicitly recorded; never repeat a successful mutation because a message is delayed.",
                "Executing all requested external mutations (sending required notification/welcome emails, updating spreadsheets, inserting/updating database records) is MANDATORY. Missing non-essential fields (such as supplier contacts or optional links) must NEVER block execution — use empty strings/defaults and complete all required writes and emails.",
                "When sending batch emails or mutating multiple records (e.g. 5+ items), complete the entire sequence across all targets in a continuous loop or via a batch execution script. Do not stop midway after the first item.",
                "Email credentials must be read from admin_credentials.txt in the workspace (default email: admin@woocommerce.local, password: admin).",
                "When syncing records across systems (e.g. syncing customers to BigQuery and sending welcome emails), ensure all associated status fields (such as welcome_email_sent=1 and welcome_email_date) are explicitly updated and marked in the destination database.",
            ),
            role_instructions={
                "context_analyst": (
                    "Acquire complete source evidence, pagination coverage, identities, and source counts; update authoritative_sources and selection evidence without mutating external state.",
                ),
                "evidence_reader": (
                    "Acquire complete source evidence, pagination coverage, identities, and source counts; update authoritative_sources and selection evidence without mutating external state.",
                ),
                "analyst": (
                    "Make selection and transformation rules reproducible: record inclusions, exclusions, joins, deduplication, ordering, boundary conditions, and unresolved rows before execution.",
                ),
                "artifact_handler": (
                    "Create task-required local deliverables from verified source rows and destination schemas. Keep bookkeeping lightweight and update it alongside execution; an unfinished contract document does not block verified rows. Do not invent extra deliverables or acceptance conditions.",
                ),
                "implementer": (
                    "Execute task-authorized rows whose identity, required payload, and preconditions are verified; no additional approval is needed per row. Checkpoint result IDs and continue from a successful canary through the remaining set. Correct tool arguments after validation errors; isolate genuinely unresolved rows and continue independent rows. Never repeat confirmed sends or invent missing required values.",
                ),
                "verifier": (
                    "Remain read-only; reconcile every applicable contract clause, exact artifact or payload, manifest row, authoritative postcondition, protected state, and missing/duplicate/extra/unverified item.",
                ),
                "integrator": (
                    "Resolve cross-system identity mappings and dependencies per batch. Report actionable discrepancies and a continuation plan; isolate unmatched rows without blocking unrelated verified rows. Distinguish different query filters from genuinely conflicting source state.",
                ),
                "controlled_operator": (
                    "Consume an approved mutation plan without redefining selection. Confirm current identity and pre-state, execute idempotently in bounded batches, record result IDs, and obtain authoritative post-state evidence.",
                ),
                "researcher": (
                    "Analyze task-visible and local source evidence without external web research or external-state mutation; send any MCP acquisition need to a tool-capable context analyst.",
                ),
            },
            role_instruction_aliases={
                "context_analyst": ("evidence_reader", "analyst"),
                "researcher": ("analyst",),
                "evidence_researcher": ("researcher", "analyst"),
                "implementer": ("artifact_handler", "controlled_operator"),
                "integrator": ("analyst",),
                "verifier": ("evidence_reader",),
            },
            fallback_policy=FallbackPolicy(
                trigger="fatal_non_recoverable_mcp_failure",
                mode="local_db_fallback",
                instructions=(
                    "Use local_db only after fatal non-recoverable MCP initialization or tool failure.",
                    "Do not switch for parameter validation, business-logic, or ordinary task errors.",
                    "Stop calling the failed MCP path and independently verify all fallback writes.",
                ),
                official_score_eligible=False,
                evolution_eligible=False,
            ),
            infrastructure_failure_conditions=(
                "preprocess cannot create an authoritative task state",
                "MCP is unavailable and complete task-specific local_db fallback paths are unavailable",
                "native claim_done evaluator cannot execute",
            ),
        )

    def bind_execution_policy(
        self, policy: ExecutionPolicy, item: dict, env_ctx: EnvContext, session,
    ) -> BoundExecutionPolicy:
        mode = str(env_ctx.data.get("execution_mode") or "native_mcp")
        fallback_contracts = env_ctx.data.get("fallback_contracts") or {}
        fallback_paths = {
            str(service): list(contract.get("paths") or [])
            for service, contract in fallback_contracts.items()
            if contract.get("mode") == "local_db_available"
        }
        fallback_used = mode == "local_db_fallback"
        runtime_rules = (
            f"Per-case native contract: {LOCA_NATIVE_CONTRACT_PATH}.",
            "Update the contract alongside execution; it is a progress aid, not a prerequisite to executing verified task-authorized rows or evaluator ground truth.",
            "Complete all requested mutations (emails, spreadsheets, database sync) without blocking on missing optional attributes.",
        )
        tools = policy.allowed_tools
        if fallback_used:
            tools = tuple(tool for tool in tools if tool != "loca_mcp")
            runtime_rules += (
                policy.fallback_policy.instructions
                if policy.fallback_policy else ()
            )
        return policy.bind(
            execution_mode=mode,
            available_tools=tools,
            workspace_paths={
                "workspace": str(session.workspace),
                "agent_workspace": str(Path(session.workspace) / "agent_workspace"),
                "local_state": str(Path(session.workspace) / "local_db"),
            },
            runtime_instructions=runtime_rules,
            fallback_used=fallback_used,
            fallback_paths=fallback_paths,
        )

    PER_SPLIT_MAX_COST = {split: 12.0 for split in CONTEXT_LEVELS}

    PER_SPLIT_TIMEOUT = {
        "8k": 1800.0,
        "16k": 1800.0,
        "32k": 3600.0,
        "64k": 3600.0,
        "96k": 5400.0,
        "128k": 7200.0,
        "256k": 7200.0,
        "evolve_96k": 7200.0,
    }

    def cli(self) -> None:
        parser = self.build_parser()
        args = parser.parse_args()

        split = getattr(args, "split", "") or self.default_split

        if args.timeout is None:
            if args.evolve:
                args.timeout = float(self.default_evolve_timeout)
            else:
                args.timeout = float(self.PER_SPLIT_TIMEOUT.get(
                    split, self.default_timeout,
                ))

        if args.max_cost is None:
            args.max_cost = float(self.PER_SPLIT_MAX_COST.get(
                split, self.default_max_cost,
            ))

        if args.evolve and args.workers > 1:
            print(
                "[error] --evolve and --workers > 1 are mutually exclusive. "
                "Evolution mode requires sequential execution (--workers 1)."
            )
            sys.exit(1)

        if args.api_case_retries < 0 or args.api_case_retry_wait < 0:
            parser.error("--api-case-retries and --api-case-retry-wait must be non-negative")

        print(
            f"[LOCABench] split={split} timeout={args.timeout:.0f}s "
            f"max_cost=${args.max_cost:.1f}"
        )

        import asyncio
        exit_code = asyncio.run(self.run(args))
        if exit_code:
            sys.exit(exit_code)

    def add_extra_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--task", type=str, default=None,
            help="Run specific task (short or full name)")
        parser.add_argument(
            "--seed", type=int, default=None,
            help="Run specific seed (42, 123, 456, 789, 2024, or 101/102 for evolve_96k)")

    def load_dataset(self, args: argparse.Namespace) -> list[dict]:
        context_level = getattr(args, "split", "128k") or "128k"
        task_filter = getattr(args, "task", None)
        seed_filter = getattr(args, "seed", None)

        items = load_locabench_configs(
            context_level=context_level,
            task_filter=task_filter,
            seed_filter=seed_filter,
        )

        if not items:
            print(f"[LOCABench] No configurations found for split={context_level}")
            if task_filter:
                print(f"  task={task_filter}")
                print(f"  Available tasks: {list(TASK_SHORT_NAMES.keys())}")
            if seed_filter:
                print(f"  seed={seed_filter}")

        return items

    def get_item_id(self, item: dict) -> str:
        task_name = item.get("_task_name", "unknown")
        seed = item.get("_seed", 0)
        return f"{task_name}_seed{seed}"

    def build_task(self, item: dict, session: Any, workspace_files: list[str]) -> str:
        task_instruction = item.get("_task_instruction", "")
        task_name = item.get("_task_name", "")
        context_level = item.get("_context_level", "")
        tool_catalog = item.get("_tool_catalog", [])
        execution_mode = item.get("_execution_mode", "native_mcp")

        if execution_mode == "local_db_fallback":
            fallback_contracts = item.get("_fallback_contracts", {})
            path_lines = []
            for service, contract in sorted(fallback_contracts.items()):
                paths = contract.get("paths") or []
                if paths:
                    path_lines.append(
                        f"  - {service}: " + ", ".join(f"`{path}`" for path in paths)
                    )
            return f"""## Task: {task_name} ({context_level})

{task_instruction}

The primary service-tool environment failed with a fatal initialization error.
Use only the task-specific local state paths listed below. Paths are relative
to your workspace `{session.workspace}`:

{chr(10).join(path_lines)}

Rules for this fallback execution:
1. Do not call the unavailable service tool.
2. Preserve the existing local data format and make idempotent changes.
3. Read and populate `{LOCA_NATIVE_CONTRACT_PATH}` from task-visible evidence before writing.
4. For bulk work, record exact pending/completed/failed/unverified rows in `agent_workspace/task_manifest.json`.
5. Independently read back the final local state, update the verification and completion-gate sections, and report exact incomplete or uncertain scope.
6. Finish using the authoritative Runtime Submission Protocol supplied by the runner.
"""

        tool_list = "\n".join(
            f"  - `{t['name']}`: {t['description'][:100]}"
            for t in tool_catalog
        )

        stock_selection_guidance = ""
        if task_name == "WoocommerceStockAlertS2LEnv":
            stock_selection_guidance = """
## WooCommerce stock-selection rule

For stock-alert tasks, if the task specifies a per-product `stock_threshold`,
that predicate is authoritative: enumerate the complete product population
(using pagination where needed) and select products with
`stock_quantity < stock_threshold` (strictly less than). Do not use
`woo_reports_low_stock` or any fixed `<10` low-stock report as the target
population; such a report is only supplementary evidence and may use a
different filter. Reconcile the complete selected ID set before writing the
sheet or sending notifications.
"""

        task = f"""## Task: {task_name} ({context_level})


{task_instruction}
{stock_selection_guidance}


You can interact with the environment using the `loca_mcp` tool. The following MCP tools are available:

{tool_list}


**List all tools:**
```
loca_mcp(action="list_tools")
```

**Call a specific tool:**
```
loca_mcp(action="call", tool_name="<tool_name>", arguments='{{...}}')
```


## Native task contract

`{LOCA_NATIVE_CONTRACT_PATH}` is the per-case contract skeleton. It contains no
hidden answers. Update it from task-visible evidence alongside execution, not as
an all-or-nothing approval gate. Verify each action's required target, payload,
and preconditions before that action; do not wait for unrelated rows or complete
paperwork. The task instruction defines the work, not the bookkeeping skeleton:

1. Identify authoritative sources, current identity/session, complete source
   coverage, inclusion/exclusion rules, joins, deduplication, ordering, and
   boundary conditions.
2. Specify every required local artifact at its evaluator-visible path with
   exact schema, names, ordering, encoding, format, and source mapping.
3. Specify every external mutation with target, precondition, exact payload,
   idempotency or duplicate policy, protected state, and authoritative
   postcondition. For cross-system work, record identity mappings and operation
   dependency order.
4. For multi-item work, keep a lightweight `agent_workspace/task_manifest.json`.
   Validate a canary when needed, then continue through ALL remaining eligible
   records in bounded batches, checkpointing result IDs. A canary or checkpoint
   is not a stopping condition. Retry repairable parameter errors with corrected
   arguments; isolate unresolved records without halting independent work.
5. Independently read back the actual evaluator-visible artifact or external
   state. Do not treat a local rebuild, tool success response, nonzero file, or
   aggregate count as proof that the submitted state is correct.
6. Set `completion_gate.ready` only when all applicable requirements are
   directly verified and pending, failed, missing, duplicate, unintended-extra,
   and unverified scope is zero. While work remains, treat ready=false as a
   progress state: continue execution and repair, not final submission. If a
   worker stalls, return its checkpoint to the chairman for reassignment or
   takeover of unresolved items. Do not request repetitive blocked reports.
   Only end incomplete when budget is exhausted or concrete errors prevent all
   remaining safe actions after feasible recovery. Report such failures honestly;
   never fabricate completion, recipients, payloads, or successful sends.

Use `loca_mcp(action="list_tools")` to inspect the available native operations,
execute the task step by step, and finish only through the authoritative Runtime
Submission Protocol supplied by the runner. The native evaluator runs after
submission, so complete the environment work and verification first.


Your workspace is: `{session.workspace}`
"""
        return task

    def build_team_selection_task(self, item: dict, full_task: str) -> str:
        tools = item.get("_tool_catalog", [])
        return json.dumps({
            "work_request": item.get("_task_instruction", ""),
            "context_level": item.get("_context_level", ""),
            "native_workflow_requirements": [
                "authoritative source coverage and reproducible target selection",
                "exact evaluator-visible local artifacts when requested",
                "controlled idempotent external mutations when requested",
                "cross-system identity and dependency reconciliation when requested",
                "independent artifact and authoritative-state read-back",
            ],
            "available_tool_capabilities": [
                str(tool.get("description") or "") for tool in tools
                if isinstance(tool, dict)
            ],
        }, ensure_ascii=False)

    def build_neutral_team_selection_task(self, item: dict, full_task: str) -> str:
        return json.dumps({
            "work_request": item.get("_task_instruction", ""),
        }, ensure_ascii=False)

    def setup_environment(self, item: dict, session: Any) -> EnvContext:
        from tools.loca_mcp import set_mcp_tool

        task_dir = session.workspace
        config = item

        item.pop("_cached_eval", None)

        print(f"  [LOCAbench] Creating env: {config['_task_name']}...")
        env = _create_loca_env(config, task_dir)
        obs = env._get_instructions() if hasattr(env, '_get_instructions') else ""
        print(f"  [LOCAbench] Env created. Instruction: {len(obs)} chars")

        _strict = os.environ.get("LOCA_STRICT_PREPROCESS", "").strip() in ("1", "true", "True")
        _check_preprocess_integrity(task_dir, config["_task_name"], strict=_strict)

        item["_task_instruction"] = obs

        print(f"  [LOCAbench] Starting MCP servers: {list(config['mcp_servers'].keys())}...")
        enabled_servers = [
            str(name) for name, server in config.get("mcp_servers", {}).items()
            if not isinstance(server, dict) or server.get("enabled", True)
        ]
        fallback_contracts = _fallback_contract(task_dir, enabled_servers)
        mcp_tool = None
        catalog = []
        execution_mode = "native_mcp"
        try:
            mcp_tool, catalog = _setup_mcp_tool(config, task_dir)
            _validate_mcp_catalog(config, catalog, task_dir)
            if not catalog and enabled_servers and all(
                fallback_contracts.get(server, {}).get("mode") == "local_db_available"
                for server in enabled_servers
            ):
                execution_mode = "local_db_fallback"
        except Exception as exc:
            complete_fallback = bool(enabled_servers) and all(
                fallback_contracts.get(server, {}).get("mode") == "local_db_available"
                for server in enabled_servers
            )
            if not complete_fallback:
                raise ConnectionError(
                    "LOCA MCP initialization failed and complete local_db fallback "
                    f"is unavailable: {exc}"
                ) from exc
            execution_mode = "local_db_fallback"
            print(f"  [LOCAbench] MCP unavailable; using local_db fallback: {exc}")

        if execution_mode == "local_db_fallback" and mcp_tool is not None:
            try:
                mcp_tool.close()
            except Exception:
                pass
            _unregister_mcp_tool(mcp_tool)
            mcp_tool = None
            catalog = []

        print(
            f"  [LOCAbench] Environment mode: {execution_mode}; "
            f"{len(catalog)} MCP tools discovered"
        )

        item["_tool_catalog"] = catalog
        item["_execution_mode"] = execution_mode
        item["_fallback_contracts"] = fallback_contracts
        contract_path = _ensure_native_contract_skeleton(
            task_dir,
            task_name=str(config.get("_task_name") or ""),
            context_level=str(config.get("_context_level") or ""),
            execution_mode=execution_mode,
            tool_catalog=catalog,
        )
        item["_native_contract_path"] = str(
            contract_path.relative_to(Path(task_dir))
        )

        if mcp_tool is not None:
            set_mcp_tool(mcp_tool, catalog)
        else:
            set_mcp_tool(None, [])

        return EnvContext(data={
            "env": env,
            "mcp_tool": mcp_tool,
            "catalog": catalog,
            "execution_mode": execution_mode,
            "fallback_contracts": fallback_contracts,
            "native_contract_path": str(contract_path),
            "task_dir": task_dir,
            "task_name": config["_task_name"],
            "seed": config["_seed"],
            "context_level": config["_context_level"],
        })

    def post_process(self, item: dict, session: Any, env_ctx: EnvContext) -> None:
        mcp_tool = env_ctx.data.get("mcp_tool")
        if mcp_tool is not None and (
            getattr(mcp_tool, "_evoteam_fallback_triggered", False)
        ):
            env_ctx.data["execution_mode"] = "local_db_fallback"
            item["_execution_mode"] = "local_db_fallback"
        event_path = Path(session.dir) / "events.jsonl"
        try:
            event_text = event_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            item["_completion_protocol"] = {}
        else:
            item["_completion_protocol"] = _completion_protocol_status(event_text)


    def teardown_environment(self, env_ctx: EnvContext) -> None:
        from tools.loca_mcp import clear_mcp_tool

        clear_mcp_tool()

        mcp_tool = env_ctx.data.get("mcp_tool")
        if mcp_tool:
            try:
                mcp_tool.close()
            except Exception as e:
                print(f"  [LOCAbench] Warning: MCP cleanup error: {e}")
            finally:
                _unregister_mcp_tool(mcp_tool)

    def evaluate(self, item: dict, predicted: str, session: Any, **ctx) -> EvalResult:
        cached = item.get("_cached_eval")
        if cached is not None:
            return cached

        env = ctx.get("env")
        if env is None:
            return EvalResult(
                success=False, score=0.0,
                summary="No LOCA-bench environment available for evaluation",
            )

        try:
            with _eval_lock:
                obs, reward, terminated, truncated, info = env.step("claim_done")

            score = float(reward)
            success = score >= 1.0

            eval_details = info.get("evaluation", "")
            if isinstance(eval_details, dict):
                eval_details = json.dumps(eval_details)[:500]
            elif not isinstance(eval_details, str):
                eval_details = str(eval_details)[:500]

            return EvalResult(
                success=success,
                score=score,
                summary=f"reward={score:.1f} ({'PASS' if success else 'FAIL'})",
                details=eval_details if eval_details else None,
                extra={
                    "reward": score,
                    "task_name": item.get("_task_name", ""),
                    "seed": item.get("_seed", 0),
                    "context_level": item.get("_context_level", ""),
                },
            )
        except Exception as e:
            return EvalResult(
                success=False, score=0.0,
                summary=f"Evaluation error: {e}",
                details=str(e)[:500],
            )

    def build_task_validator(self, item: dict, session: Any, **ctx):
        from core.types import TaskValidation

        env = ctx.get("env")
        if env is None:
            return None

        async def _validator(output: str) -> TaskValidation:
            try:
                with _eval_lock:
                    obs, reward, terminated, truncated, info = env.step("claim_done")
                score = float(reward)
                success = score >= 1.0

                eval_details = info.get("error", "") or info.get("evaluation", "")
                if isinstance(eval_details, dict):
                    eval_details = json.dumps(eval_details)[:500]
                item["_cached_eval"] = EvalResult(
                    success=success,
                    score=score,
                    summary=f"reward={score:.1f} ({'PASS' if success else 'FAIL'})",
                    details=str(eval_details)[:500] if eval_details else None,
                    extra={
                        "reward": score,
                        "task_name": item.get("_task_name", ""),
                        "seed": item.get("_seed", 0),
                        "context_level": item.get("_context_level", ""),
                    },
                )

                task_name = item.get("_task_name", "")
                summary = f"LOCA-bench {task_name}: reward={score:.1f} ({'PASS' if success else 'FAIL'})"

                details_parts = [summary]
                if not success:
                    error_info = info.get("error", "")
                    if error_info:
                        details_parts.append(f"Failure reason: {str(error_info)[:800]}")

                    eval_log = _extract_eval_log(ctx.get("task_dir", ""), task_name)
                    if eval_log:
                        details_parts.append(f"Evaluation log excerpt:\n{eval_log}")

                    if obs and isinstance(obs, str) and len(obs.strip()) > 0:
                        details_parts.append(f"Environment observation: {obs[:800]}")

                    if output and len(output.strip()) > 0:
                        details_parts.append(f"Agent claimed output:\n{output[:1000]}")

                    details_parts.append(
                        "\nKey areas to improve:\n"
                        "  - Ensure ALL items are processed — count them before terminating\n"
                        "  - Verify output format matches exactly (directory names, column headers, email subjects)\n"
                        "  - Check pagination completeness (iterate ALL pages until empty)\n"
                        "  - Confirm no items were skipped or duplicated\n"
                        "  - For customer/order filtering: check the COMPLETE dataset criteria, not just surface-level counts\n"
                        "  - Do NOT call claim_done_claim_done — it doesn't exist. Just call set_final_output + terminate"
                    )

                return TaskValidation(
                    success=success,
                    summary=summary,
                    details="\n".join(details_parts),
                )
            except Exception as e:
                return TaskValidation(
                    success=False,
                    summary=f"Evaluation error: {e}",
                    details=str(e)[:500],
                )

        return _validator

    def build_record(self, item, idx, predicted, eval_result, session, error):
        extra = eval_result.extra or {}

        err_str = error or ""
        forced_termination = None
        if err_str:
            low = err_str.lower()
            if "timeout" in low:
                forced_termination = "timeout"
            elif "cost_limit" in low or "cost limit" in low:
                forced_termination = "cost_limit"
            elif "max_messages" in low:
                forced_termination = "max_messages"

        return {
            "idx": idx,
            "task_name": extra.get("task_name", item.get("_task_name", "")),
            "seed": extra.get("seed", item.get("_seed", 0)),
            "context_level": extra.get("context_level", item.get("_context_level", "")),
            "split": item.get("_split", item.get("_context_level", "")),
            "reward": extra.get("reward", 0.0),
            "score": eval_result.score,
            "success": eval_result.success,
            "eval_summary": eval_result.summary[:200] if eval_result.summary else "",
            "eval_details": eval_result.details[:500] if eval_result.details else "",
            "model_output": predicted[:2000] if predicted else "",
            "run_error": error,
            "forced_termination": forced_termination,
            "completion_protocol": item.get("_completion_protocol", {}),
            "session_id": session.id,
        }

    def _make_error_record(self, idx: int, item: dict, error: Exception) -> dict:
        err_str = str(error)
        low = err_str.lower()
        forced_termination = None
        if "timeout" in low:
            forced_termination = "timeout"
        elif "cost_limit" in low or "cost limit" in low:
            forced_termination = "cost_limit"
        elif "max_messages" in low:
            forced_termination = "max_messages"
        return {
            "idx": idx,
            "index": idx,
            "task_name": item.get("_task_name", ""),
            "seed": item.get("_seed", 0),
            "context_level": item.get("_context_level", ""),
            "split": item.get("_split", item.get("_context_level", "")),
            "status": "error",
            "error": err_str,
            "run_error": err_str,
            "forced_termination": forced_termination,
            "completion_protocol": item.get("_completion_protocol", {}),
            "score": 0.0,
            "success": False,
            "reward": 0.0,
            "eval_summary": f"Error: {err_str[:150]}",
            "eval_details": "",
            "model_output": "",
            "session_id": "",
        }

    def compute_summary(self, records, args):
        total = len(records)
        if total == 0:
            return {"total": 0, "avg_score": 0, "pass_rate": 0}

        scores = [r.get("score", 0.0) for r in records]
        passed = sum(1 for s in scores if s >= 1.0)
        avg = sum(scores) / total

        by_task = defaultdict(list)
        for r in records:
            by_task[r.get("task_name", "?")].append(r.get("score", 0.0))

        task_summary = {}
        for task, task_scores in sorted(by_task.items()):
            task_summary[task] = {
                "count": len(task_scores),
                "pass": sum(1 for s in task_scores if s >= 1.0),
                "avg": round(sum(task_scores) / len(task_scores), 3),
            }

        return {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total else 0,
            "avg_score": round(avg, 4),
            "by_task": task_summary,
        }

    def print_summary(self, records: list[dict]) -> None:
        total = len(records)
        if total == 0:
            print("No results.")
            return

        scores = [r.get("score", 0.0) for r in records]
        passed = sum(1 for s in scores if s >= 1.0)
        avg = sum(scores) / total

        print(f"\n{'=' * 60}")
        print(f"LOCA-bench Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
        print(f"Average score: {avg:.3f}")
        print(f"{'=' * 60}")

        by_task = defaultdict(list)
        for r in records:
            by_task[r.get("task_name", "?")].append(r.get("score", 0.0))

        print("\nBy Task:")
        for task in sorted(by_task):
            vals = by_task[task]
            p = sum(1 for s in vals if s >= 1.0)
            print(f"  {task:45s}: {p}/{len(vals)} passed, avg={sum(vals)/len(vals):.3f}")

        by_seed = defaultdict(list)
        for r in records:
            by_seed[r.get("seed", 0)].append(r.get("score", 0.0))

        if len(by_seed) > 1:
            print("\nBy Seed:")
            for seed in sorted(by_seed):
                vals = by_seed[seed]
                p = sum(1 for s in vals if s >= 1.0)
                print(f"  seed={seed}: {p}/{len(vals)} passed, avg={sum(vals)/len(vals):.3f}")

        errors = [r for r in records if r.get("run_error")]
        if errors:
            print(f"\nErrors: {len(errors)}")
            for r in errors[:5]:
                print(f"  [{r.get('idx')}] {r.get('task_name')}: {r.get('run_error', '')[:100]}")

    def dry_run_print(self, items, args):
        context_level = getattr(args, "split", "128k")
        print(f"LOCA-bench ({context_level}): {len(items)} configurations\n")

        by_task = defaultdict(list)
        for item in items:
            by_task[item["_task_name"]].append(item["_seed"])

        print("By Task:")
        for task in sorted(by_task):
            seeds = sorted(by_task[task])
            print(f"  {task:45s}: seeds={seeds}")

        print(f"\nTotal: {len(by_task)} task types × {len(items)//len(by_task) if by_task else 0} seeds = {len(items)} configurations")

        print("\nFirst 10 items:")
        for i, item in enumerate(items[:10]):
            name = item["_task_name"]
            seed = item["_seed"]
            servers = list(item.get("mcp_servers", {}).keys())
            print(f"  [{i}] {name} seed={seed}")
            print(f"      MCP servers: {servers}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")



if __name__ == "__main__":
    LOCABenchAdapter().cli()
