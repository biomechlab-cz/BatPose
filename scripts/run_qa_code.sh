#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WT="$ROOT/../wt-qa-code"

COMMON="$ROOT/agent_prompts/00_common.md"
ROLE="$ROOT/agent_prompts/30_qa_code.md"

cd "$WT"

echo "== Launching Claude in: $WT =="
echo
echo "----- COPY INTO CLAUDE (COMMON + ROLE) -----"
echo
cat "$COMMON"
echo
echo "-----"
echo
cat "$ROLE"
echo
echo "----- END PROMPT -----"
echo
echo "Tip: paste the text above as the first message in Claude."
echo

exec claude
