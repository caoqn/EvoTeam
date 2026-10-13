import sys
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

_BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BASE_DIR))

from benchmarks.adapter import BenchmarkAdapter, EvalResult, EnvContext
from core.execution_policy import BoundExecutionPolicy, ExecutionPolicy


LOCOBENCH_DIR = _BASE_DIR / "benchmarks" / "LoCoBench"
DATA_DIR = LOCOBENCH_DIR / "data"
SCENARIOS_DIR = DATA_DIR / "output" / "scenarios"
GENERATED_DIR = DATA_DIR / "generated"

TASK_CATEGORIES = [
    "architectural_understanding", "cross_file_refactoring",
    "feature_implementation", "bug_investigation",
    "multi_session_development", "code_comprehension",
    "integration_testing", "security_analysis",
]

LANGUAGES = [
    "python", "cpp", "java", "c", "csharp",
    "javascript", "typescript", "go", "rust", "php",
]

SPLIT_MAP = {
    "all": {},
    "python": {"language": "python"},
    "java": {"language": "java"},
    "go": {"language": "go"},
    "cpp": {"language": "cpp"},
    "javascript": {"language": "javascript"},
    "typescript": {"language": "typescript"},
    "rust": {"language": "rust"},
    "c": {"language": "c"},
    "csharp": {"language": "csharp"},
    "php": {"language": "php"},
    "python_easy": {"language": "python", "difficulty": "easy"},
    "python_medium": {"language": "python", "difficulty": "medium"},
    "python_hard": {"language": "python", "difficulty": "hard"},
    "python_easy_medium": {"language": "python", "difficulty_in": ["easy", "medium"]},
    "python_expert": {"language": "python", "difficulty": "expert"},
}



def _extract_project_id(scenario_id: str) -> str:
    _CATEGORY_PATTERNS = [
        "_architectural_understanding_",
        "_cross_file_refactoring_",
        "_feature_implementation_",
        "_bug_investigation_",
        "_multi_session_development_",
        "_code_comprehension_",
        "_integration_testing_",
        "_security_analysis_",
    ]
    for pat in _CATEGORY_PATTERNS:
        idx = scenario_id.find(pat)
        if idx >= 0:
            return scenario_id[:idx]
    parts = scenario_id.rsplit("_", 4)
    return parts[0] if len(parts) > 4 else scenario_id


def _extract_language(scenario_id: str) -> str:
    return scenario_id.split("_")[0]



def load_locobench_data(
    split: str = "all",
    category: str | None = None,
    max_items: int | None = None,
    index_start: int | None = None,
    index_end: int | None = None,
) -> list[dict]:
    if not SCENARIOS_DIR.exists():
        raise FileNotFoundError(
            f"LoCoBench scenarios dir not found: {SCENARIOS_DIR}\n"
            f"Download data.zip and extract to benchmarks/LoCoBench/data/"
        )

    filters = SPLIT_MAP.get(split, {})
    lang_filter = filters.get("language")
    diff_filter = filters.get("difficulty")
    diff_in_filter = filters.get("difficulty_in")

    items = []
    for f in sorted(SCENARIOS_DIR.glob("*.json")):
        try:
            scenario = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            continue

        sid = scenario.get("id", "")
        lang = _extract_language(sid)
        diff = scenario.get("difficulty", "")
        cat = scenario.get("task_category", "")

        if lang_filter and lang != lang_filter:
            continue
        if diff_filter and diff != diff_filter:
            continue
        if diff_in_filter and diff not in diff_in_filter:
            continue
        if category and cat != category:
            continue

        if index_start is not None or index_end is not None:
            import re
            match = re.search(r'_(\d{3})_', sid)
            if match:
                idx = int(match.group(1))
                if index_start is not None and idx < index_start:
                    continue
                if index_end is not None and idx > index_end:
                    continue
            else:
                continue

        scenario["_language"] = lang
        scenario["_project_id"] = _extract_project_id(sid)
        items.append(scenario)

    if max_items:
        items = items[:max_items]

    return items



def _collect_solution_files(workspace: str) -> dict[str, str]:
    solution_dir = Path(workspace) / "solution"
    return _collect_solution_files_from_dir(solution_dir)


def _collect_solution_files_from_dir(solution_dir: Path) -> dict[str, str]:
    if not solution_dir.exists():
        return {}

    files = {}
    for f in sorted(solution_dir.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            rel = str(f.relative_to(solution_dir))
            try:
                files[rel] = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
    return files


def _evaluate_with_metrics(scenario: dict, solution_code: dict[str, str]) -> dict:
    global _cached_metrics_module, _cached_validator_module
    if "_cached_metrics_module" not in globals():
        _cached_metrics_module = None
        _cached_validator_module = None

    import importlib.util

    if _cached_metrics_module is None:
        _metrics_path = LOCOBENCH_DIR / "locobench" / "generation" / "metric_algorithms.py"
        try:
            spec = importlib.util.spec_from_file_location(
                "locobench_metrics", str(_metrics_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _cached_metrics_module = mod
        except Exception as e:
            return {"score": 0.0, "error": f"LoCoBench metrics not importable: {e}"}

    try:
        LoCoBenchMetricsCalculator = _cached_metrics_module.LoCoBenchMetricsCalculator
    except Exception as e:
        return {"score": 0.0, "error": f"LoCoBench metrics not importable: {e}"}

    calc = LoCoBenchMetricsCalculator()
    solution_code = calc._sanitize_solution_code(solution_code)

    if not solution_code:
        return {"score": 0.0, "details": "empty solution"}

    try:
        acs = calc.calculate_architectural_coherence_score(scenario, solution_code)
    except Exception:
        acs = 0.0
    try:
        dta = calc.calculate_dependency_traversal_accuracy(scenario, solution_code)
    except Exception:
        dta = 0.0
    try:
        rs = calc.calculate_robustness_score(scenario, solution_code)
    except Exception:
        rs = 0.0
    try:
        cs = calc.calculate_comprehensiveness_score(scenario, solution_code)
    except Exception:
        cs = 0.0
    try:
        ins = calc.calculate_innovation_score(scenario, solution_code)
    except Exception:
        ins = 0.0
    try:
        sts = calc.calculate_system_thinking_score(scenario, solution_code)
    except Exception:
        sts = 0.0
    try:
        ses = calc.calculate_solution_elegance_score(scenario, solution_code)
    except Exception:
        ses = 0.0
    try:
        cfrd = calc.calculate_cross_file_reasoning_depth(scenario, solution_code)
    except Exception:
        cfrd = 0.0

    se_score = (acs + dta + cfrd + sts + rs + cs + ins + ses) / 8.0

    try:
        icu = calc.calculate_information_coverage_utilization(scenario, solution_code)
    except Exception:
        icu = 0.0
    try:
        mmr = calc.calculate_multi_session_memory_retention(scenario, solution_code)
    except Exception:
        mmr = 0.0

    lcu_score = (icu + mmr) / 2.0

    try:
        idc = calc.calculate_incremental_development_capability(scenario, solution_code)
    except Exception:
        idc = 0.0

    _validator_path = LOCOBENCH_DIR / "locobench" / "validation" / "code_validator.py"
    compilation_score = 0.0
    security_score = 0.5
    quality_score = 0.5
    language = _extract_language(scenario.get("id", "python"))

    if _cached_validator_module is None:
        try:
            spec2 = importlib.util.spec_from_file_location(
                "locobench_validator", str(_validator_path))
            vmod = importlib.util.module_from_spec(spec2)
            spec2.loader.exec_module(vmod)
            _cached_validator_module = vmod
        except Exception:
            _cached_validator_module = False

    try:
        if _cached_validator_module and _cached_validator_module is not False:
            validator = _cached_validator_module.CodeValidator()

            if language == "python":
                import py_compile, tempfile, os
                all_ok = True
                for fname, code in solution_code.items():
                    if not fname.endswith(".py"):
                        continue
                    try:
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                                          delete=False) as tmp:
                            tmp.write(code)
                            tmp_path = tmp.name
                        py_compile.compile(tmp_path, doraise=True)
                        os.unlink(tmp_path)
                    except py_compile.PyCompileError:
                        all_ok = False
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
                    except Exception:
                        pass
                compilation_score = 1.0 if all_ok else 0.0
            else:
                compilation_score = 0.5

            try:
                sec_patterns = validator._get_security_patterns(language)
                all_vulns = []
                for fname, code in solution_code.items():
                    vulns = validator._scan_security_patterns(code, sec_patterns, fname)
                    all_vulns.extend(vulns)
                if all_vulns:
                    severity_weights = {"high": 0.3, "medium": 0.2, "low": 0.1}
                    deduction = sum(severity_weights.get(v.get("severity", "low"), 0.1)
                                    for v in all_vulns)
                    security_score = max(0.0, 1.0 - deduction)
                else:
                    security_score = 1.0
            except Exception:
                security_score = 0.5

            try:
                complexity_scores = []
                maintain_scores = []
                for fname, code in solution_code.items():
                    complexity_scores.append(validator._calculate_complexity(code, language))
                    maintain_scores.append(validator._calculate_maintainability(code, language))
                if complexity_scores:
                    quality_score = (
                        sum(complexity_scores) / len(complexity_scores) * 0.5 +
                        sum(maintain_scores) / len(maintain_scores) * 0.5
                    )
                else:
                    quality_score = 0.5
            except Exception:
                quality_score = 0.5

    except Exception:
        pass

    fc_score = compilation_score * 0.5 + idc * 0.5

    cq_score = security_score * 0.5 + quality_score * 0.5

    lcbs = 5.0 * (0.4 * se_score + 0.3 * fc_score + 0.2 * cq_score + 0.1 * lcu_score)

    return {
        "lcbs": round(lcbs, 4),
        "se_score": round(se_score, 4),
        "fc_score": round(fc_score, 4),
        "cq_score": round(cq_score, 4),
        "lcu_score": round(lcu_score, 4),
        "details": {
            "ACS": round(acs, 3), "DTA": round(dta, 3), "CFRD": round(cfrd, 3),
            "STS": round(sts, 3), "RS": round(rs, 3), "CS": round(cs, 3),
            "IS": round(ins, 3), "SES": round(ses, 3),
            "ICU": round(icu, 3), "MMR": round(mmr, 3), "IDC": round(idc, 3),
            "Compilation": round(compilation_score, 3),
            "Security": round(security_score, 3),
            "Quality": round(quality_score, 3),
        },
        "solution_files": len(solution_code),
        "solution_lines": sum(len(c.split("\n")) for c in solution_code.values()),
    }



class LoCoBenchAdapter(BenchmarkAdapter):
    benchmark_name = "locobench"
    default_team = "pool_EvoTeam"
    preserve_timeout_handoff_trace = False
    aggregate_handoff_reflection_by_pair = False
    default_timeout = 1800.0
    default_evolve_timeout = 1800.0
    default_max_cost = 2.40
    default_split = "python_easy_medium"
    split_choices = list(SPLIT_MAP.keys())

    def execution_policy(self, item: dict) -> ExecutionPolicy:
        return ExecutionPolicy(
            name="long_context_solution_workspace",
            version="1",
            environment_mode="isolated_context_solution_workspace",
            allowed_tools=("read_file", "write_file", "bash"),
            artifact_requirements=(
                "Write at least one complete file under solution/; an empty solution scores zero.",
                "Match the requested language, interfaces, and requirements from the supplied context.",
                "For analysis tasks, use source files with analysis in comments rather than Markdown.",
            ),
            chairman_instructions=(
                "Partition relevant context files into non-overlapping evidence assignments.",
                "Have evidence agents report concrete code evidence directly to the implementer.",
                "Require independent verification of solution paths, interfaces, and requirement coverage before finalization.",
            ),
            special_rules=(
                "Be selective when reading context and do not attempt to execute the supplied context project.",
                "The completion summary is not a substitute for files under solution/.",
            ),
            role_instructions={
                "researcher": (
                    "Treat the supplied context as the only evidence corpus; identify requirement-relevant files and return source-grounded findings without external research.",
                ),
                "context_analyst": (
                    "Read only assigned context files and report concrete snippets, interfaces, dependencies, and paths to the implementer.",
                ),
                "implementer": (
                    "Write complete, syntactically valid files under solution/ and match existing project patterns.",
                ),
                "verifier": (
                    "Check solution/ placement, language/API contracts, cross-file consistency, and every stated requirement without creating an alternative solution.",
                ),
                "integrator": (
                    "Reconcile cross-file interfaces and confirm that the final artifact is internally consistent.",
                ),
            },
            role_instruction_aliases={
                "reader-1": ("context_analyst",),
                "reader-2": ("context_analyst",),
                "reader-3": ("context_analyst",),
                "reader-4": ("context_analyst",),
                "developer": ("implementer",),
                "reviewer": ("verifier",),
                "solution_verifier": ("verifier",),
                "feature_integrator": ("integrator",),
                "dependency_architect": ("context_analyst", "integrator"),
                "evidence_researcher": ("researcher",),
            },
            infrastructure_failure_conditions=(
                "native scenario data is missing",
                "required context source files cannot be provisioned",
                "native metrics implementation cannot be loaded",
            ),
        )

    def bind_execution_policy(
        self, policy: ExecutionPolicy, item: dict, env_ctx: EnvContext, session,
    ) -> BoundExecutionPolicy:
        return policy.bind(workspace_paths={
            "workspace": str(session.workspace),
            "context": str(Path(session.workspace) / "context"),
            "solution": str(Path(session.workspace) / "solution"),
        })

    def add_extra_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--max-items", type=int, default=None,
            help="Max instances to load")
        parser.add_argument(
            "--category", type=str, default=None,
            choices=TASK_CATEGORIES,
            help="Run specific task category only")
        parser.add_argument(
            "--language", type=str, default=None,
            choices=LANGUAGES,
            help="Filter by language (overrides --split)")
        parser.add_argument(
            "--index-start", type=int, default=None,
            help="Start index for project filtering (e.g., 20 for test set)")
        parser.add_argument(
            "--index-end", type=int, default=None,
            help="End index for project filtering (e.g., 99 for test set)")

    def load_dataset(self, args: argparse.Namespace) -> list[dict]:
        max_items = getattr(args, "max_items", None)
        category = getattr(args, "category", None)
        index_start = getattr(args, "index_start", None)
        index_end = getattr(args, "index_end", None)

        lang = getattr(args, "language", None)
        if lang:
            split = lang
        else:
            split = args.split

        return load_locobench_data(split, category=category, max_items=max_items,
                                   index_start=index_start, index_end=index_end)

    def get_item_id(self, item: dict) -> str:
        return item.get("id", "unknown")

    def build_task(self, item: dict, session, workspace_files: list[str]) -> str:
        title = item.get("title", "")
        description = item.get("description", "")
        task_category = item.get("task_category", "")
        difficulty = item.get("difficulty", "")
        language = item.get("_language", "")
        context_files = item.get("context_files", [])
        workspace = session.workspace

        task_prompt = item.get("task_prompt", "")
        if isinstance(task_prompt, dict):
            parts = []
            for key in sorted(task_prompt.keys()):
                parts.append(f"**{key.upper()}**: {task_prompt[key]}")
            task_prompt = "\n\n".join(parts)

        evaluation_criteria = item.get("evaluation_criteria", [])
        criteria_text = "\n".join(f"  - {c}" for c in evaluation_criteria)

        normalized_files = [f.replace("//", "/") for f in context_files]
        file_list = "\n".join(f"  - `context/{f}`" for f in normalized_files[:50])
        if len(normalized_files) > 50:
            file_list += f"\n  - ... and {len(normalized_files) - 50} more files"

        task = f"""## Task: {title}

**Category**: {task_category.replace('_', ' ').title()}
**Difficulty**: {difficulty}
**Language**: {language}


{description}


{task_prompt}


The following source code files are available in your workspace under `context/`:

{file_list}

Use `read_file` to examine these files. **You do NOT need to read all files** — analyze which files are relevant to the task and read only those.

**Workspace location**: `{workspace}`
- Context files: `{workspace}/context/...`
- Write solutions to: `{workspace}/solution/...`


{criteria_text}


Write your solution code files into the `solution/` directory using `write_file`. For example:
- `write_file(path="solution/main.py", content="...")`
- `write_file(path="solution/utils/helper.py", content="...")`

Create complete, working {language} code files that address all requirements. The solution should be well-structured and follow {language} best practices.

**IMPORTANT**: You MUST write at least one file to `solution/` before finishing. An empty solution scores 0.
"""
        return task

    def build_team_selection_task(self, item: dict, full_task: str) -> str:
        task_prompt = item.get("task_prompt", "")
        return json.dumps({
            "work_type_hint": item.get("task_category", ""),
            "work_request": task_prompt or item.get("description", ""),
            "implementation_language": item.get("_language", ""),
            "context_file_count": len(item.get("context_files", [])),
            "verification_strategy": item.get("evaluation_criteria", []),
        }, ensure_ascii=False)

    def build_neutral_team_selection_task(self, item: dict, full_task: str) -> str:
        task_prompt = item.get("task_prompt", "")
        return json.dumps({
            "work_request": task_prompt or item.get("description", ""),
            "available_resources": {
                "implementation_language": item.get("_language", ""),
                "context_file_count": len(item.get("context_files", [])),
            },
        }, ensure_ascii=False)

    def setup_environment(self, item: dict, session) -> EnvContext:
        project_id = item.get("_project_id", "")
        context_file_names = item.get("context_files", [])
        workspace = Path(session.workspace)

        context_dir = workspace / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        (workspace / "solution").mkdir(parents=True, exist_ok=True)

        copied = 0
        total_chars = 0
        for rel_path in context_file_names:
            rel_path_normalized = rel_path.replace("//", "/")
            src = GENERATED_DIR / project_id / rel_path_normalized
            if not src.exists():
                src = GENERATED_DIR / project_id / rel_path
            if not src.exists():
                continue
            dst = context_dir / rel_path_normalized
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(str(src), str(dst))
                total_chars += src.stat().st_size
                copied += 1
            except Exception:
                pass

        print(f"  Context: {copied}/{len(context_file_names)} files "
              f"({total_chars / 1024:.0f} KB) → workspace/context/")

        return EnvContext(data={
            "project_id": project_id,
            "workspace": str(workspace),
            "context_files_count": copied,
            "context_total_chars": total_chars,
        })

    def teardown_environment(self, env_ctx: EnvContext) -> None:
        pass

    def get_patch_for_snapshot(self, env_ctx) -> str:
        workspace = env_ctx.data.get("workspace", "")
        if not workspace:
            return ""
        import shutil
        solution_dir = Path(workspace) / "solution"
        backup_dir = Path(workspace) / "solution_pre_reflection"
        if solution_dir.exists() and not backup_dir.exists():
            try:
                shutil.copytree(str(solution_dir), str(backup_dir))
            except Exception:
                pass
        return ""

    def build_task_validator(self, item: dict, session, **ctx):
        from core.types import TaskValidation

        async def _validator(output: str) -> TaskValidation:
            solution_code = _collect_solution_files(session.workspace)
            if not solution_code:
                return TaskValidation(
                    success=False,
                    summary="No solution files found in workspace/solution/.",
                    details="Agent did not write any files to the solution/ directory.",
                )
            metrics = _evaluate_with_metrics(item, solution_code)
            lcbs = metrics.get("lcbs", 0.0)
            details_dict = metrics.get("details", {})

            lines = [f"LCBS = {lcbs:.2f} / 5.0"]
            lines.append(f"SE={metrics.get('se_score',0):.2f} (40%) | "
                         f"FC={metrics.get('fc_score',0):.2f} (30%) | "
                         f"CQ={metrics.get('cq_score',0):.2f} (20%) | "
                         f"LCU={metrics.get('lcu_score',0):.2f} (10%)")
            lines.append("")

            _METRIC_ADVICE = {
                "ACS": ("Architectural Coherence", "Match existing project structure and patterns more closely"),
                "DTA": ("Dependency Traversal", "Ensure imports and cross-file references are correct"),
                "CFRD": ("Cross-File Reasoning", "Reference and integrate logic from multiple context files"),
                "STS": ("System Thinking", "Consider the system holistically, not just local changes"),
                "RS": ("Robustness", "Add error handling, input validation, and edge case checks"),
                "CS": ("Comprehensiveness", "Write MORE files covering all aspects of the requirement"),
                "IS": ("Innovation", "Go beyond minimal implementation — add useful features"),
                "SES": ("Solution Elegance", "Improve code clarity, naming, and structure"),
                "ICU": ("Context Utilization", "Read and reference MORE context files in your solution"),
                "MMR": ("Multi-Session Memory", "Maintain consistency across solution files"),
                "IDC": ("Incremental Development", "Build on existing patterns rather than starting from scratch"),
                "Compilation": ("Compilation", "Ensure all code is syntactically correct"),
                "Security": ("Security", "Avoid hardcoded secrets, SQL injection, unsafe patterns"),
                "Quality": ("Code Quality", "Reduce complexity, improve maintainability"),
            }

            low_metrics = []
            for key, score in details_dict.items():
                if isinstance(score, (int, float)) and score < 0.5:
                    name, advice = _METRIC_ADVICE.get(key, (key, "Improve this dimension"))
                    low_metrics.append((key, score, name, advice))

            if low_metrics:
                low_metrics.sort(key=lambda x: x[1])
                lines.append("⚠ LOW-SCORING DIMENSIONS (improve these in future tasks):")
                for key, score, name, advice in low_metrics[:5]:
                    lines.append(f"  {key}={score:.2f} ({name}): {advice}")
            else:
                lines.append("All dimensions above 0.5 — good baseline performance.")

            lines.append("")
            lines.append("Full scores: " + json.dumps(details_dict))

            details_str = "\n".join(lines)[:1900]
            is_good = lcbs >= 3.0

            return TaskValidation(
                success=is_good,
                summary=(f"LCBS={lcbs:.2f}/5.0 ({'Good+' if is_good else 'needs improvement'}). "
                         f"{metrics.get('solution_files',0)} files. "
                         f"{len(low_metrics)} low dimensions."),
                details=details_str,
            )

        return _validator

    def evaluate(self, item: dict, predicted: str, session, **ctx) -> EvalResult:
        backup_dir = Path(session.workspace) / "solution_pre_reflection"
        if backup_dir.exists():
            solution_code = _collect_solution_files_from_dir(backup_dir)
        else:
            solution_code = _collect_solution_files(session.workspace)

        if not solution_code:
            return EvalResult(
                success=False, score=0.0,
                summary="No solution files in workspace/solution/",
            )

        metrics = _evaluate_with_metrics(item, solution_code)
        lcbs = metrics.get("lcbs", 0.0)
        score_01 = min(lcbs / 5.0, 1.0)

        return EvalResult(
            success=lcbs >= 3.0,
            score=round(score_01, 4),
            summary=f"LCBS={lcbs:.2f}/5.0 | SE={metrics.get('se_score',0):.2f} "
                    f"FC={metrics.get('fc_score',0):.2f} CQ={metrics.get('cq_score',0):.2f} "
                    f"LCU={metrics.get('lcu_score',0):.2f} | "
                    f"{metrics.get('solution_files',0)} files, "
                    f"{metrics.get('solution_lines',0)} lines",
            details=json.dumps(metrics.get("details", {}))[:500],
            extra={
                "lcbs": lcbs,
                "se_score": metrics.get("se_score", 0),
                "fc_score": metrics.get("fc_score", 0),
                "cq_score": metrics.get("cq_score", 0),
                "lcu_score": metrics.get("lcu_score", 0),
                "details": metrics.get("details", {}),
                "solution_files": metrics.get("solution_files", 0),
                "solution_lines": metrics.get("solution_lines", 0),
                "language": item.get("_language", ""),
                "task_category": item.get("task_category", ""),
                "difficulty": item.get("difficulty", ""),
            },
        )

    def build_record(self, item, idx, predicted, eval_result, session, error):
        extra = eval_result.extra or {}
        return {
            "idx": idx,
            "scenario_id": item.get("id", ""),
            "language": item.get("_language", ""),
            "task_category": item.get("task_category", ""),
            "difficulty": item.get("difficulty", ""),
            "title": item.get("title", "")[:100],
            "lcbs": extra.get("lcbs", 0),
            "score": eval_result.score,
            "se_score": extra.get("se_score", 0),
            "fc_score": extra.get("fc_score", 0),
            "cq_score": extra.get("cq_score", 0),
            "lcu_score": extra.get("lcu_score", 0),
            "details": extra.get("details", {}),
            "solution_files": extra.get("solution_files", 0),
            "solution_lines": extra.get("solution_lines", 0),
            "eval_summary": eval_result.summary[:200] if eval_result.summary else "",
            "model_output": predicted[:2000],
            "run_error": error,
            "session_id": session.id,
        }

    def compute_summary(self, records, args):
        total = len(records)
        if total == 0:
            return {"total": 0, "avg_score": 0, "avg_lcbs": 0}

        lcbs_scores = [r.get("lcbs", 0) for r in records]
        avg_lcbs = sum(lcbs_scores) / total

        excellent = sum(1 for s in lcbs_scores if s >= 4.0)
        good = sum(1 for s in lcbs_scores if 3.0 <= s < 4.0)
        fair = sum(1 for s in lcbs_scores if 2.0 <= s < 3.0)
        poor = sum(1 for s in lcbs_scores if s < 2.0)

        return {
            "total": total,
            "avg_score": round(avg_lcbs / 5.0, 4),
            "avg_lcbs": round(avg_lcbs, 3),
            "grade_distribution": {
                "excellent": excellent, "good": good,
                "fair": fair, "poor": poor,
            },
        }

    def print_summary(self, records: list[dict]) -> None:
        total = len(records)
        if total == 0:
            print("No results.")
            return

        lcbs_scores = [r.get("lcbs", 0) for r in records]
        avg = sum(lcbs_scores) / total

        print(f"\n{'=' * 60}")
        print(f"LoCoBench Results: avg LCBS={avg:.2f}/5.0 ({total} scenarios)")
        print(f"{'=' * 60}")

        by_lang = defaultdict(list)
        for r in records:
            by_lang[r.get("language", "?")].append(r.get("lcbs", 0))
        if len(by_lang) > 1:
            print("\nBy Language:")
            for lang in sorted(by_lang):
                vals = by_lang[lang]
                print(f"  {lang:12s}: avg={sum(vals)/len(vals):.2f} ({len(vals)} cases)")

        by_cat = defaultdict(list)
        for r in records:
            by_cat[r.get("task_category", "?")].append(r.get("lcbs", 0))
        print("\nBy Task Category:")
        for cat in sorted(by_cat):
            vals = by_cat[cat]
            print(f"  {cat:35s}: avg={sum(vals)/len(vals):.2f} ({len(vals)} cases)")

        by_diff = defaultdict(list)
        for r in records:
            by_diff[r.get("difficulty", "?")].append(r.get("lcbs", 0))
        print("\nBy Difficulty:")
        for diff in ["easy", "medium", "hard", "expert"]:
            if diff in by_diff:
                vals = by_diff[diff]
                print(f"  {diff:8s}: avg={sum(vals)/len(vals):.2f} ({len(vals)} cases)")

        excellent = sum(1 for s in lcbs_scores if s >= 4.0)
        good = sum(1 for s in lcbs_scores if 3.0 <= s < 4.0)
        fair = sum(1 for s in lcbs_scores if 2.0 <= s < 3.0)
        poor = sum(1 for s in lcbs_scores if s < 2.0)
        print(f"\nGrade Distribution:")
        print(f"  Excellent (4-5): {excellent} | Good (3-4): {good} | "
              f"Fair (2-3): {fair} | Poor (0-2): {poor}")

    def dry_run_print(self, items, args):
        print(f"LoCoBench ({args.split}): {len(items)} scenarios\n")

        by_lang = defaultdict(int)
        by_cat = defaultdict(int)
        by_diff = defaultdict(int)
        for item in items:
            by_lang[item.get("_language", "?")] += 1
            by_cat[item.get("task_category", "?")] += 1
            by_diff[item.get("difficulty", "?")] += 1

        for i, item in enumerate(items[:10]):
            sid = item.get("id", "?")
            title = item.get("title", "?")[:60]
            diff = item.get("difficulty", "?")
            cat = item.get("task_category", "?")
            n_files = len(item.get("context_files", []))
            ctx_len = item.get("context_length", 0)
            print(f"  [{i}] {sid[:60]}")
            print(f"      {cat} | {diff} | {n_files} files | {ctx_len:,} chars")
            print(f"      {title}")
            print()

        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more\n")

        print("By Language:")
        for lang in sorted(by_lang):
            print(f"  {lang}: {by_lang[lang]}")
        print("\nBy Category:")
        for cat in sorted(by_cat):
            print(f"  {cat}: {by_cat[cat]}")
        print("\nBy Difficulty:")
        for diff in ["easy", "medium", "hard", "expert"]:
            if diff in by_diff:
                print(f"  {diff}: {by_diff[diff]}")


if __name__ == "__main__":
    LoCoBenchAdapter().cli()
