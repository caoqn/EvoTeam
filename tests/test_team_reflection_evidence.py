import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from litellm import ModelResponse

from benchmarks.adapter import EvalResult, _evolve_team_template
from core import llm
from core.agent import Agent
from core.evolution_models import ScopeAwareTeamTemplateEvolutionDecision, TemplateMember
from core.output_contract import OutputContract
from core.runner import Runner, RunnerResult
from core.team_template_registry import TeamTemplateRegistry
from core.tool_registry import ToolRegistry


def _agent(tmp_path, name, registry):
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    (directory / "config.yaml").write_text(
        f"name: {name}\nmodel: test-model\nmax_steps: 100\n", encoding="utf-8"
    )
    return Agent(str(directory), registry)


def _response(content, tool=None, arguments=None, call_id="call_1", reasoning=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool:
        message["tool_calls"] = [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool,
                "arguments": json.dumps(arguments or {}),
            },
        }]
    return ModelResponse(choices=[{"message": message}])


@pytest.mark.parametrize("existing_template", [False, True])
def test_local_trace_survives_trimming_and_restart_and_reaches_reflector(
    tmp_path, monkeypatch, existing_template
):
    async def scenario():
        tools = ToolRegistry(str(tmp_path / "tools"))
        runner = Runner(
            pool_dir=tmp_path, chairman_name="chairman", tool_registry=tools,
            output_contract=OutputContract(answer_agent_enabled=True),
        )
        worker = _agent(tmp_path, "worker", tools)
        worker._runner_ref = runner
        runner.agents["worker"] = worker
        runner._record_agent_participation("worker", recruited=True)
        monkeypatch.setattr(worker, "_build_all_tools", lambda extra: [])
        monkeypatch.setattr(worker, "_track_cost_usd", lambda response: None)
        original_task = "Implement the requested behavior and verify the artifact."
        artifact = "artifact body\n" * 1000 + "END_OF_ARTIFACT"
        call_count = 0

        async def complete(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 90:
                return _response(
                    f"Inspect and verify step {call_count}", "inspect_artifact",
                    {"path": f"artifact_{call_count}.txt"}, f"call_{call_count}",
                )
            return _response("Verified final result", reasoning="All artifacts checked.")

        async def execute(name, args, cwd):
            return artifact if args["path"] == "artifact_1.txt" else "verified"

        monkeypatch.setattr(llm, "complete", complete)
        monkeypatch.setattr(worker, "_execute_tool_call", execute)
        await worker.run_loop(initial_task=original_task)
        assert all(message.get("tool_call_id") != "call_1" for message in worker.messages)

        waiting_task = asyncio.create_task(asyncio.Event().wait())
        runner._agent_tasks["worker"] = waiting_task
        assert runner.stop_agent("worker")
        await asyncio.gather(waiting_task, return_exceptions=True)
        replacement = _agent(tmp_path, "worker", tools)
        replacement._runner_ref = runner
        runner.agents["worker"] = replacement
        runner._record_agent_participation("worker", recruited=True)
        monkeypatch.setattr(replacement, "_build_all_tools", lambda extra: [])
        monkeypatch.setattr(replacement, "_track_cost_usd", lambda response: None)
        await replacement.run_loop(initial_task="Recheck the earlier artifact")

        runner.message_store.register("chairman")
        runner.message_store.register("worker")
        runner.message_store.register("answer_agent")
        handoff = runner.message_store.send(
            "worker", "chairman", artifact, handoff_rule_id="rule_evidence"
        )
        runner.message_store.send(
            "answer_agent", "chairman", "global service output",
            handoff_rule_id="service_rule",
        )
        runner._record_execution_message(
            "answer_agent", {"role": "assistant", "content": "service-only trace"}
        )
        runner._freeze_execution_trace()
        trace = runner._composite_execution_trace
        local_trace = trace["agent_traces"]["worker"]
        first_call = next(item for item in local_trace if item.get("tool_calls"))
        first_result = next(item for item in local_trace if item.get("tool_call_id") == "call_1")
        assert first_call["tool_calls"][0]["function"]["name"] == "inspect_artifact"
        assert first_result["content"] == artifact
        assert local_trace[-1]["content"] == "Verified final result"
        assert local_trace[-1]["reasoning_content"] == "All artifacts checked."
        assert any(item["content"] == "Recheck the earlier artifact" for item in local_trace)
        assert [item["sequence"] for item in local_trace] == sorted(item["sequence"] for item in local_trace)
        assert "answer_agent" not in trace["agent_traces"]
        assert [item["message_id"] for item in trace["handoff_trace"]] == [handoff.id]
        assert trace["handoff_trace"][0]["content"] == artifact
        assert trace["handoff_rule_usage"][0]["handoff_rule_id"] == "rule_evidence"

        template_registry = TeamTemplateRegistry()
        selection = SimpleNamespace(
            family_id="evidence_verification", allowed_agent_ids=["worker", "candidate"]
        )
        if existing_template:
            template_registry.apply_reflection(
                ScopeAwareTeamTemplateEvolutionDecision(
                    action="create", family_id=selection.family_id,
                    reason="Historical complementary contributions.",
                    members=[TemplateMember("worker", "worker"), TemplateMember("candidate", "reviewer")],
                    family_description="Evidence synthesis with independent verification.",
                ),
                task_id="prior_task", known_agent_ids={"worker", "candidate"},
                actual_agent_ids=["worker"],
            )
        pool = SimpleNamespace(
            chairman_name="chairman", settings={},
            agents={
                name: SimpleNamespace(config=SimpleNamespace(
                    name=name, role="worker", description="Verification", model="test-model"
                ))
                for name in ("chairman", "worker", "candidate", "answer_agent")
            },
        )
        captured = []

        async def reflect(**kwargs):
            captured.append(json.loads(kwargs["messages"][1]["content"]))
            return _response("", "record_team_template_decision", {
                "action": "no_update", "reason": "Keep evidence for later assessment.",
                "structural_diagnosis": {
                    "is_team_structure_failure": False,
                    "failure_type": "execution", "basis": "The tools returned usable evidence.",
                },
            })

        monkeypatch.setattr(llm, "complete", reflect)
        output = "supported result\n" * 400 + "END_OF_OUTPUT"
        await _evolve_team_template(
            runtime_team_dir=tmp_path, pool=pool, registry=template_registry,
            selection=selection, task_id="current_task", task=original_task,
            result=RunnerResult(output=output, metadata={
                "recruited_agent_ids": ["worker", "answer_agent"],
                "global_service_agent_ids": ["answer_agent"],
                "composite_execution_trace": trace,
            }),
            eval_result=EvalResult(
                success=not existing_template, score=0.0 if existing_template else 1.0,
                summary="Evaluated", details="Native evaluator feedback",
            ),
            error_msg="",
        )
        payload = captured[0]
        assert payload["task"] == original_task
        assert payload["candidate_agent_ids"] == ["worker", "candidate"]
        assert payload["actual_agent_ids"] == ["worker"]
        assert payload["execution_summary"]["output"] == output
        assert payload["composite_execution_trace"] == trace
        assert payload["evaluation"]["details"] == "Native evaluator feedback"
        assert bool(payload["current_template"]) is existing_template
        assert bool(payload["historical_task_summaries"]) is existing_template
        assert "answer_agent" not in {item["agent_id"] for item in payload["agents"]}

    asyncio.run(scenario())


@pytest.mark.parametrize("ending", ["normal", "forced", "reflection_disabled"])
def test_runner_freezes_execution_before_reflection_and_returns_evidence(
    tmp_path, monkeypatch, ending
):
    async def scenario():
        tools = ToolRegistry(str(tmp_path / "tools"))
        runner = Runner(
            pool_dir=tmp_path, chairman_name="chairman", tool_registry=tools,
            enable_reflection=ending != "reflection_disabled",
            max_seconds=0.01 if ending == "forced" else 1,
        )
        chairman = _agent(tmp_path, "chairman", tools)
        monkeypatch.setattr(runner, "load_agent_from_pool", lambda name: chairman)
        monkeypatch.setattr(runner, "_build_chairman_context", lambda: "")
        monkeypatch.setattr(runner, "_build_reflection_context", lambda: "context")
        monkeypatch.setattr(runner, "_build_forced_reflection_context", lambda reason: "context")
        monkeypatch.setattr(runner, "_start_reflection_phase_guard", lambda phase: None)
        runner.message_store.register("worker")
        runner._record_agent_participation("worker", recruited=True)

        def reflection_chatter(*args):
            runner.message_store.send("worker", "chairman", "REFLECTION_ONLY")
            chairman._append_conversation_message(chairman.messages, {
                "role": "assistant", "content": "REFLECTION_ONLY"
            })

        monkeypatch.setattr(runner, "_broadcast_l1_start", reflection_chatter)
        monkeypatch.setattr(runner, "_broadcast_forced_l1_start", reflection_chatter)
        monkeypatch.setattr(
            runner, "_restart_all_agents_for_phase",
            lambda phase: runner.terminate() if ending == "forced" else None,
        )

        async def run_loop(**kwargs):
            chairman._append_conversation_message(chairman.messages, {
                "role": "assistant", "content": "TASK_EXECUTION_ONLY"
            })
            runner.message_store.send("worker", "chairman", "TASK_HANDOFF_ONLY")
            if ending == "forced":
                await asyncio.Event().wait()
            elif ending == "normal":
                await runner.finalize_task("final result")
                runner.terminate()
            else:
                await runner.set_final_output("final result")
                runner.terminate()

        monkeypatch.setattr(chairman, "run_loop", run_loop)
        primitives = ModuleType("tools.primitives")
        primitives.build_primitive_tools = lambda *args, **kwargs: []
        primitives.rebuild_primitive_schemas = lambda *args, **kwargs: []
        primitives.cleanup_handlers = lambda *args: None
        monkeypatch.setitem(sys.modules, "tools.primitives", primitives)
        result = await runner.run("original task")
        trace = result.metadata["composite_execution_trace"]
        assert trace["agent_traces"]["chairman"][0]["content"] == "TASK_EXECUTION_ONLY"
        assert trace["handoff_trace"][0]["content"] == "TASK_HANDOFF_ONLY"
        assert "REFLECTION_ONLY" not in json.dumps(trace)
        assert "REFLECTION_ONLY" not in json.dumps(result.metadata["handoff_trace"])
        runner._freeze_execution_trace()
        assert runner._composite_execution_trace == trace
        if ending == "forced":
            assert result.metadata["forced_termination"]

    asyncio.run(scenario())