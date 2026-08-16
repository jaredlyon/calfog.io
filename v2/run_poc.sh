#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec /home/pa/work/venv/bin/python v2/run_poc.py
