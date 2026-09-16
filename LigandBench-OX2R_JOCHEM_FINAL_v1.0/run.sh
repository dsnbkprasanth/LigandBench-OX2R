#!/usr/bin/env bash
set -euo pipefail

# One-command reproducibility workflow for the frozen OX2R dataset.
python scripts/reproduce_main.py "$@"
