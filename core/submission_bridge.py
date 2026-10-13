from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BridgeResult:

    used: bool
    answer: str = ""
    mode: str = ""


class SubmissionBridge:

    def __init__(self, runner: Any):
        self.runner = runner

    @property
    def contract(self) -> Any:
        return self.runner.output_contract

    def extract_answer(self) -> str:
        if not self.contract.answer_agent_enabled:
            return ""
        prefix = re.escape(self.contract.answer_agent_prefix)
        for message in reversed(self.runner.message_store.peek(
            self.runner.chairman_name,
            sender=self.contract.answer_agent_name,
        )):
            match = re.search(
                rf"(?is)^\s*{prefix}\s*(.*?)\s*$", message.content,
            )
            if match and match.group(1).strip():
                return match.group(1).strip()
        return ""

    def validate(self, output: str) -> str | None:
        if not self.contract.enforce_answer_agent:
            return None
        expected = self.extract_answer()
        prefix = self.contract.answer_agent_prefix
        if not expected:
            return (
                f"AnswerAgent has not returned a `{prefix}` line. Send it the "
                "original task, concise evidence packet, candidate answer, and "
                "required format; wait for its reply before submitting."
            )
        if output.strip() != expected:
            return (
                "Submission must equal AnswerAgent's exact answer after "
                f"`{prefix}`. Submit output={expected!r} without a label or "
                "explanation."
            )
        return None

    async def finalize_available_answer(
        self,
        *,
        reason: str,
        event_log: Any | None = None,
    ) -> BridgeResult:
        runner = self.runner
        if runner._phase != "task_execution" or runner._result:
            return BridgeResult(False)
        answer = self.extract_answer()
        if not answer:
            return BridgeResult(False)

        mode = "finalize_task" if runner.enable_reflection else "set_final_output"
        if event_log:
            event_log.log("runner.submission_bridge_finalize", data={
                "reason": reason,
                "mode": mode,
                "answer_len": len(answer),
                "contract": self.contract.name,
            })

        if runner.enable_reflection:
            await runner.finalize_task(answer)
        else:
            await runner.set_final_output(answer)
            runner.terminate(reason=f"submission_bridge:{reason}")

        runner._submission_bridge_used = True
        runner._finalization_mode = "answer_agent_bridge"
        return BridgeResult(True, answer=answer, mode=mode)