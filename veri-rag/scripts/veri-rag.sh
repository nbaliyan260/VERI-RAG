#!/usr/bin/env bash
# Run veri-rag CLI with the project .venv (works when only conda base is active).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "No .venv found. Create it first:" >&2
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[llm]'" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
exec veri-rag "$@"
