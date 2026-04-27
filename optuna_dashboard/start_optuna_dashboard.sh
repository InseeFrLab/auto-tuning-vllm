#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DB_PATH="${1:-${SCRIPT_DIR}/study.db}"
HOST="${OPTUNA_DASHBOARD_HOST:-127.0.0.1}"
PORT="${OPTUNA_DASHBOARD_PORT:-8080}"

show_help() {
    cat <<EOF
Start Optuna Dashboard with the example SQLite database.

Usage:
  ./optuna_dashboard/start_optuna_dashboard.sh [path/to/study.db]

Environment variables:
  OPTUNA_DASHBOARD_HOST   Host to bind (default: 127.0.0.1)
  OPTUNA_DASHBOARD_PORT   Port to bind (default: 8080)

Examples:
  ./optuna_dashboard/start_optuna_dashboard.sh
  OPTUNA_DASHBOARD_PORT=8081 ./optuna_dashboard/start_optuna_dashboard.sh
  ./optuna_dashboard/start_optuna_dashboard.sh ./optuna_dashboard/study.db
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

if [[ ! -f "${DB_PATH}" ]]; then
    echo "Database file not found: ${DB_PATH}" >&2
    exit 1
fi

if [[ -x "${REPO_ROOT}/venv/bin/optuna-dashboard" ]]; then
    DASHBOARD_CMD=("${REPO_ROOT}/venv/bin/optuna-dashboard")
elif [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
    DASHBOARD_CMD=("${REPO_ROOT}/venv/bin/python" -m optuna_dashboard)
elif command -v optuna-dashboard >/dev/null 2>&1; then
    DASHBOARD_CMD=("$(command -v optuna-dashboard)")
else
    cat <<EOF >&2
optuna-dashboard is not installed.

Recommended installation from the repository root:
  python3 -m venv venv
  ./venv/bin/pip install optuna-dashboard

Then run:
  ./optuna_dashboard/start_optuna_dashboard.sh
EOF
    exit 1
fi

ABS_DB_PATH="$(cd "$(dirname "${DB_PATH}")" && pwd)/$(basename "${DB_PATH}")"
STORAGE_URL="sqlite:///${ABS_DB_PATH}"

cat <<EOF
Starting Optuna Dashboard
  Database: ${ABS_DB_PATH}
  Storage : ${STORAGE_URL}
  URL     : http://${HOST}:${PORT}

Useful views:
  - History for trial evolution
  - Parallel Coordinate for parameter interactions
  - Importance for parameter ranking
  - Table for detailed trial inspection
EOF

exec "${DASHBOARD_CMD[@]}" --host "${HOST}" --port "${PORT}" "${STORAGE_URL}"
