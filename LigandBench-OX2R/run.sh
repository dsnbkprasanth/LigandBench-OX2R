#!/usr/bin/env bash
# Convenience launcher for the LigandBench GUI.
set -e
cd "$(dirname "$0")"

if ! python -c "import streamlit" 2>/dev/null; then
  echo "Installing dependencies (first run) ..."
  pip install -r requirements.txt
fi

echo "Launching LigandBench — open the Local URL it prints below."
streamlit run app.py
