#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
exec "${PYTHON:-python}" ops/retrain/run_retrain.py 2>&1 | tee ops/retrain/run_retrain.log
