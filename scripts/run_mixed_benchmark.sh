#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON_BIN="${META_TEAM_PYTHON:-python3.11}"
MANIFEST="${MIXED_MANIFEST:-benchmarks/MIXED-BENCHMARK/manifests/mixed_benchmark_official_116_plus_838_test_seed20260828.json}"
ACTION="${1:?Usage: $0 <build-manifest|validate|evolve|test|aggregate> ...}"
shift

case "${ACTION}" in
  build-manifest) exec "${PYTHON_BIN}" scripts/build_test_manifest.py "$@" ;;
  validate) exec "${PYTHON_BIN}" benchmarks/MIXED-BENCHMARK/validate_manifest.py "${MANIFEST}" ;;
  evolve)
    RUN_ID="${1:?Usage: $0 evolve <run_id> [max_tasks]}"
    MAX_TASKS="${2:-}"
    extra=()
    [[ -n "${MAX_TASKS}" ]] && extra+=(--max-tasks "${MAX_TASKS}")
    exec "${PYTHON_BIN}" benchmarks/MIXED-BENCHMARK/mixed_scheduler.py --manifest "${MANIFEST}" --phase evolve --run-id "${RUN_ID}" "${extra[@]}"
    ;;
  test)
    TEAM="${1:?Usage: $0 test <frozen_team> <base_run_id> [rollouts]}"
    BASE_RUN_ID="${2:?Usage: $0 test <frozen_team> <base_run_id> [rollouts]}"
    ROLLOUTS="${3:-3}"
    exec "${PYTHON_BIN}" benchmarks/MIXED-BENCHMARK/mixed_scheduler.py --manifest "${MANIFEST}" --phase test --team "${TEAM}" --run-id "${BASE_RUN_ID}" --rollout "${ROLLOUTS}"
    ;;
  aggregate) exec "${PYTHON_BIN}" scripts/aggregate_avg3.py "$@" ;;
  *) echo "Unknown action: ${ACTION}" >&2; exit 2 ;;
esac
