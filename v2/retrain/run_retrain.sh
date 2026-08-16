#!/usr/bin/env bash
set -euo pipefail
cd /home/pa/work/calfog.io
exec /home/pa/work/venv/bin/python v2/retrain/run_retrain.py 2>&1 | tee v2/retrain/run_retrain.log
