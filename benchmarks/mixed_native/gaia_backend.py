import sys
import json
import re
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

_BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BASE_DIR))

from benchmarks.mixed_native.data_paths import mixed_data_root

from benchmarks.adapter import BenchmarkAdapter, EvalResult, EnvContext
from core.execution_policy import BoundExecutionPolicy, ExecutionPolicy
from core.output_contract import GAIA_OUTPUT_CONTRACT



DATA_DIR = mixed_data_root(_BASE_DIR) / "data" / "gaia"
VAL_FILES_DIR = DATA_DIR / "val_files"

SPLIT_MAP = {
    "all": ("val_new", None),
    "level1": ("val_new", {1}),
    "level2": ("val_new", {2}),
    "level3": ("val_new", {3}),
    "train_20": ("train_20", None),
    "test_100": ("test_100", None),
}

_TEXT_PROCESSABLE_EXTS = {
    "txt", "py", "json", "jsonld", "csv", "md",
}
_BINARY_PROCESSABLE_EXTS = {
    "xlsx", "xls", "pdf", "docx", "pptx", "zip", "pdb",
}



def load_gaia_data(
    split: str = "all",
    max_items: int | None = None,
) -> list[dict]:
    if split not in SPLIT_MAP:
        raise ValueError(f"Unknown split: {split!r}. Valid: {list(SPLIT_MAP)}")

    file_kind, level_filter = SPLIT_MAP[split]
    file_suffix = {
        "val_new": "val_new",
        "train_20": "train_20",
        "test_100": "test_100",
    }[file_kind]

    items = []
    for level in [1, 2, 3]:
        if level_filter is not None and level not in level_filter:
            continue

        data_file = DATA_DIR / f"level_{level}_{file_suffix}.json"
        if not data_file.exists():
            raise FileNotFoundError(
                f"GAIA data file not found: {data_file}\n"
                f"Ensure data/gaia/ directory is complete.\n"
                f"Split files (train_20/test_100) should be pre-included in the repository."
            )

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            item["_level"] = level
            items.append(item)

    if max_items:
        items = items[:max_items]

    return items



def normalize_answer(s: str) -> str:
    s = s.lower()
    return re.sub(r'[^a-z0-9]', '', s)


def extract_final_answer(model_output: str) -> str:
    parts = re.split(r'(?i)final\s*answer\s*:', model_output, maxsplit=1)
    if len(parts) > 1:
        return parts[1].strip()
    return model_output.strip()


def check_answer(model_output: str, standard_answer: str) -> bool:
    extracted = extract_final_answer(model_output)
    return normalize_answer(extracted) == normalize_answer(standard_answer)



class GAIAAdapter(BenchmarkAdapter):
    def output_contract(self):
        return GAIA_OUTPUT_CONTRACT

    def execution_policy(self, item: dict) -> ExecutionPolicy:
        return ExecutionPolicy(
            name="knowledge_question_with_optional_attachment",
            version="1",
            environment_mode="attachment_workspace",
            allowed_tools=(
                "read_file", "write_file", "bash", "web_search", "web_fetch",
            ),
            artifact_requirements=(
                "Produce one evidence-supported concise answer in the requested form.",
                "Inspect the provided attachment when the question depends on it.",
            ),
            chairman_instructions=(
                "Recruit the smallest useful evidence team based on the question and attachment type.",
                "Request independent verification for calculations, conflicting evidence, or strict formatting.",
                "Send the original question, supported findings, candidate answer, and format constraints to the AnswerAgent.",
            ),
            special_rules=(
                "Compute rather than estimate and preserve the most specific wording supported by evidence.",
                "Do not add units, articles, explanations, or formatting unless requested.",
            ),
            role_instructions={
                "researcher": (
                    "Report source-backed facts and URLs; distinguish primary evidence from inference.",
                ),
                "context_analyst": (
                    "Inspect attachments with the appropriate parser and cite sheet, page, row, slide, or section locations.",
                ),
                "implementer": (
                    "Use reproducible code or calculations for structured transformations, and report exact inputs, operations, and outputs without inventing unsupported facts.",
                ),
                "verifier": (
                    "Check calculations, source support, attachment locations, and strict answer-format compliance.",
                ),
                "integrator": (
                    "Reconcile web, attachment, and calculation evidence; preserve disagreements and hand the Chairman one supported candidate answer with its format constraints.",
                ),
            },
            role_instruction_aliases={
                "web_agent": ("researcher",),
                "reading_agent": ("researcher",),
                "file_agent": ("context_analyst",),
                "data_agent": ("context_analyst", "implementer"),
                "coding_agent": ("implementer",),
                "science_agent": ("implementer", "verifier"),
                "verification_agent": ("verifier",),
                "coordinator_agent": ("integrator",),
                "writing_agent": ("integrator",),
                "evidence_researcher": ("researcher",),
            },
            infrastructure_failure_conditions=(
                "required dataset file is missing",
                "required attachment is missing or unreadable",
                "native evaluator has no standard answer",
            ),
        )

    def bind_execution_policy(
        self, policy: ExecutionPolicy, item: dict, env_ctx: EnvContext, session,
    ) -> BoundExecutionPolicy:
        paths = {"workspace": str(session.workspace)}
        file_name = str(env_ctx.data.get("file_name") or "")
        if file_name:
            paths["attachment"] = str(Path(session.workspace) / file_name)
        return policy.bind(workspace_paths=paths)

    benchmark_name = "gaia"
    default_team = "pool_EvoTeam"
    preserve_timeout_handoff_trace = False
    aggregate_handoff_reflection_by_pair = False
    default_timeout = 900.0
    default_evolve_timeout = 900.0
    default_max_messages = 300
    default_max_cost = 2.40
    default_split = "all"
    split_choices = list(SPLIT_MAP.keys())


    def add_extra_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--max-items", type=int, default=None,
            help="Max number of instances to load")
        parser.add_argument(
            "--level", type=int, default=None, choices=[1, 2, 3],
            help="Run only specified level (overrides --split)")


    def load_dataset(self, args: argparse.Namespace) -> list[dict]:
        max_items = getattr(args, "max_items", None)

        level = getattr(args, "level", None)
        if level:
            split = f"level{level}"
        else:
            split = args.split

        return load_gaia_data(split, max_items=max_items)

    def get_item_id(self, item: dict) -> str:
        return item.get("task_id", "unknown")


    def build_task(self, item: dict, session, workspace_files: list[str]) -> str:
        question = item.get("Question", "")
        file_name = item.get("file_name", "")
        level = item.get("_level", item.get("Level", 0))
        workspace = session.workspace

        task_parts = []

        task_parts.append(f"## Question (Level {level})\n\n{question}")

        if file_name:
            ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
            file_path = f"{workspace}/{file_name}"
            task_parts.append(
                f"## Attached File\n\n"
                f"A file is provided at: `{file_path}`\n"
                f"File type: {ext}\n\n"
                f"Use appropriate tools to read and analyze this file.\n"
                f"**IMPORTANT**: When using `bash`, always use the full path `{file_path}`. "
                f"When using `read_file`, use `read_file(path=\"{file_name}\")`."
            )

            if ext in ("xlsx", "xls"):
                task_parts.append(
                    f"**Tip**: Use `bash(command=\"python3 -c \\\"import pandas as pd; "
                    f"df = pd.read_excel('{file_path}'); print(df.to_string())\\\"\")`"
                )
            elif ext == "csv":
                task_parts.append(
                    f"**Tip**: Use `bash(command=\"python3 -c \\\"import pandas as pd; "
                    f"df = pd.read_csv('{file_path}'); print(df.to_string())\\\"\")`"
                )
            elif ext == "pdf":
                task_parts.append(
                    f"**Tip**: Use `bash(command=\"python3 -c \\\"import pdfplumber; "
                    f"pdf = pdfplumber.open('{file_path}'); "
                    "[print(p.extract_text()) for p in pdf.pages]\\\"\")`"
                )
            elif ext == "docx":
                task_parts.append(
                    f"**Tip**: Use `read_file(path=\"{file_name}\")` or "
                    f"`bash(command=\"python3 -c \\\"import docx; doc = docx.Document('{file_path}'); "
                    "print('\\\\n'.join([p.text for p in doc.paragraphs]))\\\"\")`"
                )
            elif ext == "pptx":
                task_parts.append(
                    f"**Tip**: Use `bash(command=\"python3 -c \\\"from pptx import Presentation; "
                    f"prs = Presentation('{file_path}'); "
                    "[print(s.shapes.title.text if s.shapes.title else '') for s in prs.slides]\\\"\")`"
                )
            elif ext == "zip":
                task_parts.append(
                    f"**Tip**: Use `bash(command=\"cd {workspace} && unzip {file_name} -d extracted && ls extracted\")` "
                    "to extract and examine contents."
                )
            elif ext == "pdb":
                task_parts.append(
                    f"**Tip**: Use `bash(command=\"python3 -c \\\"f = open('{file_path}'); print(f.read()[:5000])\\\"\")`"
                )
            elif ext in _TEXT_PROCESSABLE_EXTS:
                task_parts.append(
                    f"**Tip**: Use `read_file(path=\"{file_name}\")` to read the file."
                )

        task_parts.append(
            "## Answer Format\n\n"
            "YOUR FINAL ANSWER should be a number OR as few words as possible OR "
            "a comma separated list of numbers and/or strings.\n"
            "- If asked for a number: don't use comma to write your number, "
            "don't use units such as $ or percent sign unless specified otherwise.\n"
            "- If asked for a string: don't use articles, don't use abbreviations "
            "(e.g. for cities), write digits in plain text unless specified otherwise.\n"
            "- Example: respond with 'Saint Louis' instead of 'St. Louis'; "
            "respond with '17' instead of '17m'.\n\n"
            "Submit your answer using: `set_final_output(output=\"FINAL ANSWER: <your answer>\")`"
        )

        return "\n\n".join(task_parts)

    def build_team_selection_task(self, item: dict, full_task: str) -> str:
        file_name = str(item.get("file_name") or "")
        attachment_type = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "none"
        return json.dumps({
            "work_request": item.get("Question", ""),
            "attachment_type": attachment_type,
            "complexity_hint": item.get("_level", item.get("Level", 0)),
        }, ensure_ascii=False)

    def build_neutral_team_selection_task(self, item: dict, full_task: str) -> str:
        file_name = str(item.get("file_name") or "")
        attachment_type = (
            file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "none"
        )
        return json.dumps({
            "work_request": item.get("Question", ""),
            "available_resource": {"attachment_type": attachment_type},
        }, ensure_ascii=False)


    def setup_environment(self, item: dict, session) -> EnvContext:
        file_name = item.get("file_name", "")

        if file_name:
            src = VAL_FILES_DIR / file_name
            if not src.exists():
                task_id = item.get("task_id", "")
                src = VAL_FILES_DIR / task_id / file_name
            if not src.exists():
                task_id = item.get("task_id", "")
                for candidate in VAL_FILES_DIR.iterdir():
                    if candidate.name == task_id and candidate.is_dir():
                        src = candidate / file_name
                        if src.exists():
                            break

            if src.exists():
                dst = Path(session.workspace) / file_name
                if src.is_dir():
                    shutil.copytree(str(src), str(dst))
                else:
                    shutil.copy2(str(src), str(dst))
                print(f"  Attached file: {file_name} → {dst}")
            else:
                print(f"  [warn] Attached file not found: {file_name}")
                print(f"         Searched in: {VAL_FILES_DIR}")

        return EnvContext(data={
            "file_name": file_name,
            "task_id": item.get("task_id", ""),
        })

    def teardown_environment(self, env_ctx: EnvContext) -> None:
        pass


    def build_task_validator(self, item: dict, session, **ctx):
        from core.types import TaskValidation

        standard_answer = item.get("Final answer", "")
        if not standard_answer:
            return None

        async def _validator(output: str) -> TaskValidation:
            extracted = extract_final_answer(output)
            is_correct = check_answer(output, standard_answer)

            if is_correct:
                return TaskValidation(
                    success=True,
                    summary="Answer matches the expected result after normalization.",
                    details=(
                        f"Your submitted answer: {extracted!r}\n"
                        f"Expected answer: {standard_answer!r}\n"
                        f"After normalization: PASS.\n\n"
                        "Reflect on the reasoning and workflow that led to this success: "
                        "what patterns worked well and could be reused on similar tasks?"
                    ),
                )

            norm_model = normalize_answer(extracted)
            norm_standard = normalize_answer(standard_answer)

            if not norm_model:
                err_class = "empty_answer"
                err_hint = (
                    "Your submitted answer is empty after normalization. "
                    "You may have submitted pure reasoning text without a concrete answer, "
                    "or the FINAL ANSWER line was missing."
                )
            elif norm_standard in norm_model and norm_model != norm_standard:
                err_class = "over_specified"
                err_hint = (
                    "Your answer CONTAINS the expected answer but with extra tokens. "
                    "Typical over-specification patterns: units ('17 hours' vs '17'), "
                    "articles ('the United States' vs 'United States'), "
                    "descriptive qualifiers, or explanatory words. "
                    "The GAIA grader strips non-alphanumerics and requires EXACT match. "
                    "Reflect on whether answer_agent enforced the minimal-form rule strictly."
                )
            elif norm_model in norm_standard and norm_model != norm_standard:
                err_class = "under_specified"
                err_hint = (
                    "Your answer is a substring of the expected answer — "
                    "you stopped short of the complete answer. "
                    "Did you miss part of a multi-part answer (list, compound entity)?"
                )
            elif len(norm_model) > 3 * max(1, len(norm_standard)):
                err_class = "verbose"
                err_hint = (
                    "Your answer is much longer than expected — "
                    "you likely included reasoning/explanation in FINAL ANSWER "
                    "instead of a single concise token."
                )
            else:
                err_class = "different"
                err_hint = (
                    "Your answer is substantively different from what was expected "
                    "(not just a format mismatch). Reflect on the reasoning: "
                    "did you misread the question, use the wrong source, "
                    "or misinterpret what was being asked?"
                )

            return TaskValidation(
                success=False,
                summary=(
                    f"INCORRECT: {err_class}. "
                    f"Expected {standard_answer!r}, got {extracted!r} "
                    f"(normalized: {norm_standard!r} vs {norm_model!r})."
                ),
                details=(
                    f"## Error diagnosis\n"
                    f"- Your submitted answer: {extracted!r}\n"
                    f"- Your answer (normalized): {norm_model!r}\n"
                    f"- Expected answer: {standard_answer!r}\n"
                    f"- Expected (normalized): {norm_standard!r}\n"
                    f"- Error category: **{err_class}**\n\n"
                    f"## What the error means\n"
                    f"{err_hint}\n\n"
                    f"## ⚠️ REFLECTION OUTPUT CONSTRAINT\n"
                    f"The expected answer above is provided ONLY so you can accurately\n"
                    f"diagnose WHERE your reasoning went wrong. It is NOT a fact to\n"
                    f"memorize. Your reflection products (prompt_patches, skills,\n"
                    f"constitution) will be applied to FUTURE UNSEEN tasks.\n"
                    f"Therefore:\n"
                    f"  ❌ Do NOT copy the expected answer string into your patches.\n"
                    f"  ❌ Do NOT use the current task's specific entities as examples.\n"
                    f"  ✅ DO extract the generalizable pattern using abstract placeholders\n"
                    f"     (`<compound>`, `<X>`, `<Y>`) and record WORKFLOW improvements."
                ),
            )

        return _validator


    def evaluate(self, item: dict, predicted: str, session, **ctx) -> EvalResult:
        standard_answer = item.get("Final answer", "")

        if not standard_answer:
            return EvalResult(
                success=False, score=0.0,
                summary="No standard answer available for this item.",
            )

        extracted = extract_final_answer(predicted)
        is_correct = check_answer(predicted, standard_answer)

        return EvalResult(
            success=is_correct,
            score=1.0 if is_correct else 0.0,
            summary=(
                f"{'CORRECT' if is_correct else 'INCORRECT'}: "
                f"extracted='{extracted[:100]}' "
                f"expected='{standard_answer[:100]}'"
            ),
            details=json.dumps({
                "extracted_answer": extracted[:200],
                "standard_answer": standard_answer[:200],
                "normalized_model": normalize_answer(extracted),
                "normalized_standard": normalize_answer(standard_answer),
            })[:500],
            extra={
                "is_correct": is_correct,
                "extracted_answer": extracted[:200],
                "standard_answer": standard_answer[:200],
                "level": item.get("_level", item.get("Level", 0)),
            },
        )


    def build_record(self, item, idx, predicted, eval_result, session, error):
        extra = eval_result.extra or {}
        return {
            "idx": idx,
            "task_id": item.get("task_id", ""),
            "level": item.get("_level", item.get("Level", 0)),
            "question": item.get("Question", "")[:200],
            "standard_answer": item.get("Final answer", ""),
            "extracted_answer": extra.get("extracted_answer", ""),
            "is_correct": extra.get("is_correct", False),
            "score": eval_result.score,
            "eval_summary": eval_result.summary[:200] if eval_result.summary else "",
            "model_output": predicted[:2000],
            "run_error": error,
            "session_id": session.id,
            "file_name": item.get("file_name", ""),
        }


    def compute_summary(self, records, args):
        infrastructure_failures = [
            record for record in records if record.get("infrastructure_failure")
        ]
        scored_records = [
            record for record in records if not record.get("infrastructure_failure")
        ]
        total = len(scored_records)
        correct = sum(1 for r in scored_records if r.get("is_correct"))

        by_level = defaultdict(lambda: {"total": 0, "correct": 0})
        for r in scored_records:
            level = r.get("level", 0)
            by_level[level]["total"] += 1
            if r.get("is_correct"):
                by_level[level]["correct"] += 1

        level_stats = {}
        for level, stats in sorted(by_level.items()):
            t, c = stats["total"], stats["correct"]
            level_stats[f"level_{level}"] = {
                "total": t,
                "correct": c,
                "accuracy": round(c / t, 4) if t else 0.0,
            }

        return {
            "submitted_cases": len(records),
            "total": total,
            "correct": correct,
            "avg_score": round(correct / total, 4) if total else 0.0,
            "infrastructure_failure_count": len(infrastructure_failures),
            "infrastructure_failure_task_ids": [
                record.get("task_id", "") for record in infrastructure_failures
            ],
            "by_level": level_stats,
        }

    def print_summary(self, records: list[dict]) -> None:
        infrastructure_failures = [
            record for record in records if record.get("infrastructure_failure")
        ]
        scored_records = [
            record for record in records if not record.get("infrastructure_failure")
        ]
        total = len(scored_records)
        if total == 0:
            print("No scored results (all completed cases had infrastructure failures).")
            return
        correct = sum(1 for r in scored_records if r.get("is_correct"))
        accuracy = correct / total

        print(f"\n{'=' * 60}")
        print(f"GAIA Results: {correct}/{total} correct "
              f"({100 * accuracy:.1f}%)")
        if infrastructure_failures:
            print(
                f"Excluded infrastructure failures: {len(infrastructure_failures)} "
                "(retry these exact task IDs; do not score them as wrong answers)"
            )
        print(f"{'=' * 60}")

        by_level = defaultdict(list)
        for r in scored_records:
            by_level[r.get("level", 0)].append(r.get("is_correct", False))

        print("\nBy Level:")
        for level in sorted(by_level):
            vals = by_level[level]
            c = sum(vals)
            t = len(vals)
            print(f"  Level {level}: {c}/{t} ({100 * c / t:.1f}%)")

        with_file = [r for r in scored_records if r.get("file_name")]
        without_file = [r for r in scored_records if not r.get("file_name")]
        if with_file:
            wf_correct = sum(1 for r in with_file if r.get("is_correct"))
            print(f"\n  With file:    {wf_correct}/{len(with_file)} "
                  f"({100 * wf_correct / len(with_file):.1f}%)")
        if without_file:
            nf_correct = sum(1 for r in without_file if r.get("is_correct"))
            print(f"  Without file: {nf_correct}/{len(without_file)} "
                  f"({100 * nf_correct / len(without_file):.1f}%)")

        print("\nDetails:")
        for r in records:
            icon = (
                "🌐" if r.get("infrastructure_failure")
                else ("✅" if r.get("is_correct") else "❌")
            )
            err = (f" [ERR: {r['run_error'][:30]}]"
                   if r.get("run_error") else "")
            answer = r.get("extracted_answer", "")[:30]
            expected = r.get("standard_answer", "")[:30]
            file_tag = f" 📎" if r.get("file_name") else ""
            print(f"  {icon} [{r.get('idx', '?'):>3}] L{r.get('level', '?')} "
                  f"ans='{answer}' exp='{expected}'{file_tag}{err}")

    def dry_run_print(self, items, args):
        print(f"GAIA ({args.split}): {len(items)} items\n")

        by_level = defaultdict(int)
        with_file_count = 0
        for i, item in enumerate(items):
            level = item.get("_level", item.get("Level", 0))
            by_level[level] += 1
            if item.get("file_name"):
                with_file_count += 1

            if i < 10 or args.cases:
                task_id = item.get("task_id", "?")[:20]
                question = item.get("Question", "")[:80]
                answer = item.get("Final answer", "")[:30]
                file_name = item.get("file_name", "")
                file_tag = f" | file: {file_name}" if file_name else ""

                print(f"  [{i}] L{level} {task_id}...")
                print(f"      Q: {question}...")
                print(f"      A: {answer}{file_tag}")
                print()

        if len(items) > 10 and not args.cases:
            print(f"  ... and {len(items) - 10} more\n")

        print("By level:")
        for level in sorted(by_level):
            print(f"  Level {level}: {by_level[level]}")
        print(f"\nWith attached file: {with_file_count}/{len(items)}")


if __name__ == "__main__":
    GAIAAdapter().cli()
