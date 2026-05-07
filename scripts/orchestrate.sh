#!/usr/bin/env bash
# =============================================================================
# orchestrate.sh — Automated multi-agent development loop
#
# Flow per cycle:
#   1.  QA-Code   → finds code bugs
#   2.  App       → fixes the bugs        ↑ repeats until clean or max iters
#   3.  QA-UI     → black-box GUI testing
#   4.  USER      → manual testing pause
#   5.  Marketing → (only when QA-UI + user found no bugs)
#   6.  Restart from step 1
#
# Agents talk via git branches in the shared worktree repo.
# Each agent writes  reports/<agent>_status.json  so we can parse "bugs_found".
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"          # e.g. wt-qa-code
MAIN_WT="$(cd "$REPO_ROOT/../BatPose" && pwd)"     # the main worktree (main branch)
WT_APP="$(cd "$REPO_ROOT/../wt-app" && pwd)"
WT_QA_CODE="$REPO_ROOT"                            # we're already here
WT_QA_UI="$(cd "$REPO_ROOT/../wt-qa-ui" && pwd)"
PROMPTS="$REPO_ROOT/agent_prompts"

# ── Tuning ────────────────────────────────────────────────────────────────────
MAX_QA_ITER=5           # max QA-Code ↔ App inner loops before giving up
CLAUDE_BUDGET="2.00"    # max USD per agent run (guards against runaway cost)
CLAUDE_MODEL="sonnet"   # agent model alias

# ── Colour logging ────────────────────────────────────────────────────────────
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
RED='\033[0;31m'; BOLD='\033[1m'; RESET='\033[0m'

log()      { printf "${CYAN}[%s]${RESET} %s\n"         "$(date '+%H:%M:%S')" "$*" >&2; }
log_ok()   { printf "${GREEN}[%s] ✓ %s${RESET}\n"      "$(date '+%H:%M:%S')" "$*" >&2; }
log_warn() { printf "${YELLOW}[%s] ⚠  %s${RESET}\n"    "$(date '+%H:%M:%S')" "$*" >&2; }
log_err()  { printf "${RED}[%s] ✗ %s${RESET}\n"        "$(date '+%H:%M:%S')" "$*" >&2; }
banner()   {
    echo "" >&2
    printf "${BOLD}╔══════════════════════════════════════════════════════╗${RESET}\n" >&2
    printf "${BOLD}║  %-52s║${RESET}\n" "$*" >&2
    printf "${BOLD}╚══════════════════════════════════════════════════════╝${RESET}\n" >&2
    echo "" >&2
}

# ── Prerequisite check ────────────────────────────────────────────────────────
check_prereqs() {
    command -v claude  >/dev/null || { log_err "claude CLI not found"; exit 1; }
    command -v git     >/dev/null || { log_err "git not found"; exit 1; }
    command -v python3 >/dev/null || { log_err "python3 not found"; exit 1; }
    for d in "$MAIN_WT" "$WT_APP" "$WT_QA_CODE" "$WT_QA_UI" "$PROMPTS"; do
        [[ -d "$d" ]] || { log_err "Directory not found: $d"; exit 1; }
    done
    for f in 00_common.md 20_app_architecture.md 30_qa_code.md \
             40_qa_ui_biomech.md 50_marketing.md; do
        [[ -f "$PROMPTS/$f" ]] || { log_err "Prompt file missing: $PROMPTS/$f"; exit 1; }
    done
    log_ok "Prereqs OK"
}

# ── Git helpers ───────────────────────────────────────────────────────────────

# Merge main into the agent's worktree (so it sees all recent fixes)
sync_from_main() {
    local wt="$1" branch
    branch="$(git -C "$wt" branch --show-current)"
    log "Syncing $branch ← main …"
    if ! git -C "$wt" merge main --no-edit -m "Merge main into $branch [auto]" 2>&1; then
        log_err "Merge conflict while syncing $branch. Please resolve then press Enter."
        read -r _
    fi
}

# Merge the agent's branch into main (from the main worktree)
merge_to_main() {
    local branch="$1"
    log "Merging $branch → main …"
    if ! git -C "$MAIN_WT" merge "$branch" --no-edit \
         -m "Merge $branch into main [auto]" 2>&1; then
        log_warn "Merge of $branch into main had conflicts — attempting to skip empty merge."
    fi
}

# ── Status-file helpers ───────────────────────────────────────────────────────

# Delete stale status file before each agent run
clear_status() {
    local status_file="$1"
    rm -f "$status_file"
}

# Read bugs_found from the agent's status JSON.
# Returns 0 (success/true) when bugs_found > 0.
has_bugs() {
    local status_file="$1" fallback_pattern="$2" fallback_file="$3"
    if [[ -f "$status_file" ]]; then
        python3 - "$status_file" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    sys.exit(0 if int(d.get("bugs_found", 0)) > 0 else 1)
except Exception:
    sys.exit(1)
PY
        return $?
    fi
    # Fallback: pattern search in the report file
    grep -q "$fallback_pattern" "$fallback_file" 2>/dev/null
}

# ── Agent runner ──────────────────────────────────────────────────────────────

# Run one Claude Code agent non-interactively.
# Args: display_name  worktree_dir  prompt_string
run_agent() {
    local name="$1" wt="$2" prompt="$3"
    local log_file="/tmp/orchestrate_${name}_$(date '+%Y%m%d_%H%M%S').log"
    log "▶ Starting agent: $name  (log → $log_file)"
    (
        cd "$wt"
        claude --print "$prompt" \
            --dangerously-skip-permissions \
            --model "$CLAUDE_MODEL" \
            --max-budget-usd "$CLAUDE_BUDGET" \
            2>&1 | tee "$log_file"
    ) || log_warn "Agent $name exited non-zero (check $log_file)"
    log_ok "Agent finished: $name"
}

# ── Prompt builders ───────────────────────────────────────────────────────────

prompt_qa_code() {
    local cycle="$1"
    cat "$PROMPTS/00_common.md"
    echo ""
    cat "$PROMPTS/30_qa_code.md"
    cat <<AUTOEOF

---
## AUTOMATED RUN — Cycle ${cycle}

You are running non-interactively as part of a CI loop.

**Your full task:**
1. Read the latest code, run \`make lint && make test && make e2e\`, and analyse all source
   files for bugs (logic errors, missing validation, edge cases, API contract violations).
2. Add new tests for any gaps you find. Fix any failures in lint/test/e2e.
3. Commit ALL new/modified files: \`git add -A && git commit -m "qa-code: cycle ${cycle}"\`
4. Write \`reports/qa_code_report.md\` (update it — keep the same format as before, add a
   "## Cycle ${cycle}" section with new findings).
5. Write \`reports/qa_code_status.json\` with exactly this structure and commit it:
   \`{"agent": "qa-code", "cycle": ${cycle}, "bugs_found": N}\`
   Set \`bugs_found\` to the count of NEW source-code bugs discovered this cycle
   (0 = code is fully clean, no action needed from the App agent).

Do not stop early. Complete all five steps before exiting.
AUTOEOF
}

prompt_app() {
    local cycle="$1"
    cat "$PROMPTS/00_common.md"
    echo ""
    cat "$PROMPTS/20_app_architecture.md"
    cat <<AUTOEOF

---
## AUTOMATED RUN — Cycle ${cycle}

You are running non-interactively as part of a CI loop.

**Your full task:**
1. Read \`reports/qa_code_report.md\` — focus on the "## Cycle ${cycle}" section (or the
   most recent "## Bugs Noted" section) to find what needs to be fixed.
2. Fix every listed bug. Keep fixes minimal and correct.
3. Run \`make lint && make test && make e2e\` — ALL must be green before you finish.
4. Commit all changes: \`git add -A && git commit -m "fix: bugs from qa-code cycle ${cycle}"\`

Do NOT write reports. Do NOT add features beyond what is needed to fix listed bugs.
AUTOEOF
}

prompt_qa_ui() {
    local cycle="$1"
    cat "$PROMPTS/00_common.md"
    echo ""
    cat "$PROMPTS/40_qa_ui_biomech.md"
    cat <<AUTOEOF

---
## AUTOMATED RUN — Cycle ${cycle}

You are running non-interactively as part of a CI loop.

**Your full task:**
1. Launch the app: \`make run-gui &\` then \`sleep 5\` to let it start.
   Interact via the GUI test driver if available: \`python reports/gui_test_driver.py\`
   (use xvfb-run if no display: \`xvfb-run -a make run-gui &\`)
2. Execute ALL test scenarios from your role prompt. Be thorough.
3. Write (overwrite) these three reports:
   - \`reports/ui_test_report.md\`
   - \`reports/bugs.md\`
   - \`reports/ux_recommendations.md\`
4. Write \`reports/qa_ui_status.json\` with exactly:
   \`{"agent": "qa-ui", "cycle": ${cycle}, "bugs_found": N}\`
   Set \`bugs_found\` to the count of bugs in \`reports/bugs.md\` (0 = zero bugs).
5. Commit all reports: \`git add reports/ && git commit -m "qa-ui: cycle ${cycle} reports"\`
AUTOEOF
}

prompt_marketing() {
    local cycle="$1"
    cat "$PROMPTS/00_common.md"
    echo ""
    cat "$PROMPTS/50_marketing.md"
    cat <<AUTOEOF

---
## AUTOMATED RUN — Cycle ${cycle}

You are running non-interactively.

Complete your full marketing analysis and write \`reports/marketing_report.md\`.
Then commit: \`git add reports/marketing_report.md && git commit -m "docs: marketing report cycle ${cycle}"\`
AUTOEOF
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

check_prereqs
cycle=0

while true; do
    cycle=$((cycle + 1))
    banner "CYCLE ${cycle} — START"

    # ── Status file paths (worktree-local) ────────────────────────────────────
    QA_CODE_STATUS="$WT_QA_CODE/reports/qa_code_status.json"
    QA_UI_STATUS="$WT_QA_UI/reports/qa_ui_status.json"

    # ── Phase 1: QA-Code ↔ App inner loop ────────────────────────────────────
    qa_iter=0
    while true; do
        qa_iter=$((qa_iter + 1))
        banner "Cycle ${cycle} · QA-Code pass #${qa_iter}"

        sync_from_main "$WT_QA_CODE"
        clear_status "$QA_CODE_STATUS"

        run_agent "qa-code-c${cycle}-i${qa_iter}" \
                  "$WT_QA_CODE" \
                  "$(prompt_qa_code "$cycle")"

        merge_to_main "agent/qa-code"

        if has_bugs "$QA_CODE_STATUS" \
                    "^### CODE-BUG-" \
                    "$WT_QA_CODE/reports/qa_code_report.md"; then

            log "QA-Code found bugs in cycle ${cycle} pass ${qa_iter}"

            if [[ $qa_iter -ge $MAX_QA_ITER ]]; then
                log_warn "Reached max inner iterations ($MAX_QA_ITER). Skipping to QA-UI."
                break
            fi

            banner "Cycle ${cycle} · App pass #${qa_iter}"
            sync_from_main "$WT_APP"

            run_agent "app-c${cycle}-i${qa_iter}" \
                      "$WT_APP" \
                      "$(prompt_app "$cycle")"

            merge_to_main "agent/app"
        else
            log_ok "QA-Code: no bugs found — code is clean this cycle!"
            break
        fi
    done

    # ── Phase 2: QA-UI ────────────────────────────────────────────────────────
    banner "Cycle ${cycle} · QA-UI"
    sync_from_main "$WT_QA_UI"
    clear_status "$QA_UI_STATUS"

    run_agent "qa-ui-c${cycle}" \
              "$WT_QA_UI" \
              "$(prompt_qa_ui "$cycle")"

    merge_to_main "agent/qa-ui"

    # Show QA-UI summary
    echo "" >&2
    if has_bugs "$QA_UI_STATUS" \
                "^## BUG-" \
                "$WT_QA_UI/reports/bugs.md"; then
        log_warn "QA-UI found bugs this cycle — see $WT_QA_UI/reports/bugs.md"
    else
        log_ok "QA-UI found no bugs this cycle"
    fi

    # ── Phase 3: USER TESTING ─────────────────────────────────────────────────
    banner "Cycle ${cycle} — YOUR TURN TO TEST"
    cat >&2 <<INFO

  The app has been updated. To launch it:
    cd ${REPO_ROOT} && make run-gui

  Review the QA-UI report first if you like:
    ${WT_QA_UI}/reports/ui_test_report.md
    ${WT_QA_UI}/reports/bugs.md

INFO
    printf "  Did YOU find any bugs? [y = yes  /  n = no  /  q = quit]: " >&2
    read -r user_answer

    case "${user_answer,,}" in
        q|quit)
            log "Stopped by user after cycle ${cycle}."
            exit 0
            ;;
        y|yes)
            log "User found bugs → restarting full cycle"
            continue
            ;;
    esac

    # If QA-UI also found bugs, loop back even if user says clean
    if has_bugs "$QA_UI_STATUS" \
                "^## BUG-" \
                "$WT_QA_UI/reports/bugs.md"; then
        log_warn "QA-UI report still lists bugs → restarting cycle"
        continue
    fi

    # ── Phase 4: MARKETING EXPERT ─────────────────────────────────────────────
    banner "Cycle ${cycle} · Marketing Expert"

    run_agent "marketing-c${cycle}" \
              "$MAIN_WT" \
              "$(prompt_marketing "$cycle")"

    # Commit marketing report into main if agent didn't already
    (
        cd "$MAIN_WT"
        git add reports/marketing_report.md 2>/dev/null || true
        if ! git diff --cached --quiet 2>/dev/null; then
            git commit -m "docs: add marketing report (cycle ${cycle}) [auto]"
        fi
    ) || true

    banner "Cycle ${cycle} COMPLETE ✓"
    cat >&2 <<DONE

  Marketing report: ${MAIN_WT}/reports/marketing_report.md

  Press Enter to start cycle $((cycle + 1)), or type 'q' to stop.
DONE
    read -r cmd
    [[ "${cmd,,}" == "q" || "${cmd,,}" == "quit" ]] && break

done

log "Orchestration ended. Completed ${cycle} cycle(s)."
