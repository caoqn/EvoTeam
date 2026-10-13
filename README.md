# EvoTeam: Multi-Agent Collaboration via Evolving Teams

This repository implements EvoTeam and its MIXED-BENCHMARK evaluation protocol.
It includes one shared agent
pool, a mixed adapter facade, four native backends, and the evolution manifest.
Dataset payloads, tool implementations, credentials, and run outputs are not
included.

## EvoTeam method modules

The implementation follows the three modules described in the paper:

- **Task-Adaptive Agent Team Formation** is implemented in
  `core/task_adaptive_agent_team_formation.py`; the reusable team-template
  catalog and persistence live in `core/team_template_registry.py`.
- **Team-Scoped Rule-Guided Collaboration** is implemented in
  `core/team_scoped_rule_guided_collaboration.py`, which loads and applies
  interaction rules associated with the selected agent team.
- **Scope-Aware Experience Evolution** uses
  `core/scope_aware_experience_evolution.py` for runtime reflection and
  agent-skill evolution. The team-template and interaction-rule reflectors
  are in `core/scope_aware_team_template_evolution.py` and
  `core/scope_aware_team_rule_evolution.py`; shared updates are persisted by
  `core/scope_aware_experience_persistence.py`.

Persisted team state uses `family_id` and `handoff_family_id` to connect team
templates with their scoped handoff rules.

Team reflection receives the original execution task, the selected template
or cold-start candidates, the actual recruited members, the final output,
evaluation feedback, and relevant history. Its `composite_execution_trace`
contains each recruited member's local execution messages plus the Chairman's
coordination trace, and the full handoff payloads with interaction-rule IDs.
Local traces retain emitted reasoning, tool calls, tool results, and intermediate
artifacts exposed in those messages. They are recorded independently of bounded
LLM contexts, survive stopping/restarting members, and are frozen before agent
reflection starts. Static system prompts and post-task reflection messages are
excluded. Team-reflection execution evidence is not sliced or truncated; long executions
therefore require a reflection model with sufficient context capacity.

The entry point is `benchmarks/MIXED-BENCHMARK/mixed_scheduler.py`. The standard
manifest is `benchmarks/MIXED-BENCHMARK/manifests/mixed_benchmark_official_116_seed20260828.json`
(116 evolution cases and no test-phase rows); held-out cases are supplied separately.

## External assets required to run

- Provide GAIA split files and attachments at `data/gaia/`.
- Install LoCoBench at `benchmarks/LoCoBench/`, including its scenario data and
  native validation code.
- Install LOCA-Bench at `benchmarks/LOCA-bench/`, including `task-configs/`,
  `gem/`, `mcp_convert/`, and the required MCP services.
- Provide BeyondSWE data at `benchmarks/BeyondSWE/beyondswe.jsonl` or set
  `BEYONDSWE_DATA_FILE`; its repository tasks require Docker.
- Provide a `tools/` package with the tool names declared by the shared pool
  and manifest. The LOCA tool must expose `set_mcp_tool`, `clear_mcp_tool`,
  and the `_evoteam_fallback_triggered` flag on a crashed MCP session.
- Configure the model endpoint using environment variables or a private env
  file selected by `EVOTEAM_ENV_FILE`. Do not place credentials in this copy.

These components are deliberately omitted. Structural validation works
without them; native resolution and execution do not.

## Commands

```bash
python benchmarks/MIXED-BENCHMARK/validate_manifest.py \
  benchmarks/MIXED-BENCHMARK/manifests/mixed_benchmark_official_116_seed20260828.json
python benchmarks/MIXED-BENCHMARK/mixed_scheduler.py \
  --manifest benchmarks/MIXED-BENCHMARK/manifests/mixed_benchmark_official_116_seed20260828.json \
  --phase all --dry-run
```

After installing the external assets, validate native references with
`--resolve-native`. To execute the evolution stream, supply `--run-id` and run
`--phase evolve`. For holdout evaluation, prepare a manifest with the same
116 evolution rows plus the held-out cases as `test` rows, then run
`--phase test --resume` with the same run ID. The scheduler writes output
under `runs/`.

The shared agent pool is `agents/pool_EvoTeam`, and its orchestration mode is
`evoteam`. The model override is `EVOTEAM_MODEL`; a private environment file
can be selected with `EVOTEAM_ENV_FILE`. Circuit-breaker settings use the
`EVOTEAM_API_CIRCUIT_` prefix, and `EVOTEAM_RESULT_MANIFEST` can override the
native result-manifest path.
