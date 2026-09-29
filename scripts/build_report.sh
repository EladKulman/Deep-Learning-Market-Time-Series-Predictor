#!/usr/bin/env bash
# Run from any directory. REPORT_PYTHON can select an explicitly configured Python.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${REPORT_PYTHON:-}" ]]; then
    report_python="$REPORT_PYTHON"
elif [[ -x "$repo_dir/.venv/bin/python" ]]; then
    report_python="$repo_dir/.venv/bin/python"
else
    report_python="python3"
fi
exec "$report_python" "$repo_dir/scripts/build_report.py" "$@"
