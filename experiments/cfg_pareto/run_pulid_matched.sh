#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec .venv/bin/python -u experiments/cfg_pareto/run_pulid_matched.py "$@"
