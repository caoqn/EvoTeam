#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:?Usage: $0 <run_id> [version] [team_name]}"
VERSION="${2:-latest}"
TEAM_NAME="${3:-pool_EvoTeam_frozen}"

if [[ "${VERSION}" == "latest" ]]; then
  VERSION="$(find "${ROOT}/runs/${RUN_ID}/team" -mindepth 1 -maxdepth 1 -type d -name 'v[0-9]*' -print | sort | tail -n 1 | xargs -n1 basename)"
fi
SOURCE="${ROOT}/runs/${RUN_ID}/team/${VERSION}"
DEST="${ROOT}/agents/${TEAM_NAME}"
[[ -d "${SOURCE}" ]] || { echo "missing team version: ${SOURCE}" >&2; exit 1; }
[[ ! -e "${DEST}" ]] || { echo "destination exists: ${DEST}; remove it explicitly before replacing" >&2; exit 1; }
cp -R "${SOURCE}" "${DEST}"
printf 'Frozen %s/%s -> %s\n' "${RUN_ID}" "${VERSION}" "${DEST}"
