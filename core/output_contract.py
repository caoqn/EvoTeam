from dataclasses import dataclass


@dataclass(frozen=True)
class OutputContract:

    name: str = "generic"
    answer_agent_enabled: bool = False
    answer_agent_name: str = "answer_agent"
    enforce_answer_agent: bool = False
    answer_agent_prefix: str = "FINAL OUTPUT:"
    answer_agent_instruction: str = (
        "Return the complete final output to the Chairman using the required "
        "prefix exactly."
    )
    chairman_instruction: str = (
        "Send the original task, concise verified evidence, candidate output, "
        "and required format to the AnswerAgent before submitting."
    )
    submission_type: str = "text"
    chairman_submission_hint: str = (
        "Submit the complete final output required by the current task and "
        "benchmark evaluator."
    )
    emergency_mode: str = "disabled"
    emergency_instruction: str = (
        "Provide the best complete final output now, based only on information "
        "already gathered. Do not make tool calls or describe next steps."
    )
    emergency_prefix: str = ""

    @property
    def has_emergency_finalize(self) -> bool:
        return self.emergency_mode != "disabled"


GAIA_OUTPUT_CONTRACT = OutputContract(
    name="gaia",
    answer_agent_enabled=True,
    enforce_answer_agent=True,
    answer_agent_prefix="FINAL ANSWER:",
    answer_agent_instruction=(
        "Return only the concise answer after `FINAL ANSWER:`. Do not include "
        "reasoning, a report, units, or extra explanation."
    ),
    chairman_instruction=(
        "Send the original question, concise verified evidence, candidate short "
        "answer, and strict answer format to the AnswerAgent."
    ),
    submission_type="strict_short_answer",
    chairman_submission_hint=(
        "Submit only the concise GAIA answer required by the question; do not "
        "include reasoning, a report, units, or extra explanation."
    ),
    emergency_mode="strict_tagged_answer",
    emergency_instruction=(
        "Output the best concise GAIA answer now. Do not make tool calls, "
        "explain reasoning, or say what you would do next."
    ),
    emergency_prefix="FINAL ANSWER:",
)


def output_contract_from_name(name: str | None) -> OutputContract:
    if not name or name == "generic":
        return OutputContract()
    aliases = {
        "gaia_short_answer": "gaia",
        "researchrubrics": "deep_research_report",
        "deepresearch": "deep_research_report",
        "swebench_pro": "swe_patch_summary",
        "beyondswe": "beyondswe_patch_summary",
        "locobench": "locobench_solution_summary",
        "locabench": "locabench_completion_summary",
    }
    name = aliases.get(name, name)
    if name == "gaia":
        return GAIA_OUTPUT_CONTRACT
    profiles = {
        "deep_research_report": OutputContract(
            name="deep_research_report",
            answer_agent_enabled=True,
            enforce_answer_agent=True,
            answer_agent_instruction=(
                "Return the complete evidence-backed Markdown report after "
                "`FINAL OUTPUT:` and preserve citations and the confidence line."
            ),
            chairman_instruction=(
                "Send the research question, evidence packet, citations, draft "
                "report, and rubric requirements to the AnswerAgent."
            ),
            submission_type="markdown_research_report",
            chairman_submission_hint=(
                "Submit the complete evidence-backed Markdown research report "
                "required by the rubric, including citations and its confidence line."
            ),
            emergency_mode="tagged_output",
            emergency_instruction=(
                "Return the best complete Markdown research report supported by "
                "the evidence; preserve citations and the Confidence line."
            ),
            emergency_prefix="FINAL OUTPUT:",
        ),
        "swe_patch_summary": OutputContract(
            name="swe_patch_summary",
            answer_agent_enabled=True,
            enforce_answer_agent=True,
            answer_agent_instruction=(
                "Return a concise 1-3 sentence implementation and test summary "
                "after `FINAL OUTPUT:` without inventing evidence."
            ),
            chairman_instruction=(
                "Send the task, patch evidence, changed files, and verified test "
                "results to the AnswerAgent."
            ),
            submission_type="repository_patch_summary",
            chairman_submission_hint=(
                "Submit a concise 1-3 sentence summary of the implemented and "
                "tested change; the repository patch is the real benchmark artifact."
            ),
            emergency_mode="tagged_output",
            emergency_instruction=(
                "Return a concise 1-3 sentence summary of the implemented and "
                "tested change. Do not invent patch or test evidence."
            ),
            emergency_prefix="FINAL OUTPUT:",
        ),
        "beyondswe_patch_summary": OutputContract(
            name="beyondswe_patch_summary",
            answer_agent_enabled=True,
            enforce_answer_agent=True,
            answer_agent_instruction=(
                "Return a concise 1-3 sentence fix and test summary after "
                "`FINAL OUTPUT:` without inventing evidence."
            ),
            chairman_instruction=(
                "Send the task, repository patch evidence, and verified test "
                "results to the AnswerAgent."
            ),
            submission_type="repository_patch_summary",
            chairman_submission_hint=(
                "Submit a concise 1-3 sentence summary of the implemented and "
                "tested fix; the repository patch is the real benchmark artifact."
            ),
            emergency_mode="tagged_output",
            emergency_instruction=(
                "Return a concise 1-3 sentence summary of the implemented and "
                "tested fix. Do not invent patch or test evidence."
            ),
            emergency_prefix="FINAL OUTPUT:",
        ),
        "locobench_solution_summary": OutputContract(
            name="locobench_solution_summary",
            answer_agent_enabled=True,
            enforce_answer_agent=True,
            answer_agent_instruction=(
                "Confirm from the evidence packet that the required complete files "
                "exist under `solution/`, then return a concise completion summary "
                "after `FINAL OUTPUT:`."
            ),
            chairman_instruction=(
                "Send the task, solution file paths, implementation evidence, and "
                "verification results to the AnswerAgent."
            ),
            submission_type="solution_files_plus_completion_summary",
            chairman_submission_hint=(
                "Submit a concise completion summary after confirming that the "
                "required files exist under solution/; those files are the artifact."
            ),
            emergency_mode="tagged_output",
            emergency_instruction=(
                "Return a concise completion summary only after checking the "
                "solution artifact evidence in the packet."
            ),
            emergency_prefix="FINAL OUTPUT:",
        ),
        "locabench_completion_summary": OutputContract(
            name="locabench_completion_summary",
            answer_agent_enabled=True,
            enforce_answer_agent=True,
            answer_agent_instruction=(
                "Confirm the required environment actions and output format from "
                "the evidence packet, then return a concise completion summary "
                "after `FINAL OUTPUT:`."
            ),
            chairman_instruction=(
                "Send the task, completed environment actions, postconditions, and "
                "verification evidence to the AnswerAgent."
            ),
            submission_type="environment_actions_plus_completion_summary",
            chairman_submission_hint=(
                "Submit a concise completion summary after confirming the required "
                "actions and exact output format were completed in the environment."
            ),
            emergency_mode="tagged_output",
            emergency_instruction=(
                "Return a concise completion summary supported by the verified "
                "environment evidence in the packet."
            ),
            emergency_prefix="FINAL OUTPUT:",
        ),
    }
    try:
        return profiles[name]
    except KeyError as exc:
        raise ValueError(f"Unknown output contract: {name!r}") from exc


def output_contract_from_settings(
    settings: dict | None,
    benchmark_name: str | None = None,
    *,
    native_contract: OutputContract | None = None,
    allow_pool_override: bool = True,
) -> OutputContract:
    settings = settings or {}
    pool_protocol = settings.get("answer_protocol")
    if pool_protocol and not allow_pool_override:
        raise ValueError(
            "shared heterogeneous pools must not define answer_protocol; "
            "the native adapter supplies the contract per task"
        )
    if native_contract is not None and not pool_protocol:
        contract = native_contract
    else:
        protocol = pool_protocol or benchmark_name
        contract = output_contract_from_name(protocol)
    if settings.get("answer_agent_required", contract.answer_agent_enabled):
        return contract
    return OutputContract(
        name=contract.name,
        answer_agent_instruction=contract.answer_agent_instruction,
        chairman_instruction=contract.chairman_instruction,
        submission_type=contract.submission_type,
        chairman_submission_hint=contract.chairman_submission_hint,
        emergency_mode=contract.emergency_mode,
        emergency_instruction=contract.emergency_instruction,
        emergency_prefix=contract.emergency_prefix,
    )
