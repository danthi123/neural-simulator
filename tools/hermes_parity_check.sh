#!/usr/bin/env bash
# hermes_parity_check.sh — doctor for "can Hermes develop this repo with the same workflow
# discipline Claude Code uses here?" See docs/HERMES_WORKFLOW_PARITY.md for the full mapping this
# script verifies against.
#
# Two tiers of check:
#   BLOCKING — (1) the git-level pre-commit gate: fires on any `git commit` regardless of which
#              agent runs it, so its absence is a real regression, not just an unapplied config
#              snippet; (2) HERMES.md + the durable-state re-anchor rule: with only ~100K context
#              (vs Claude Code's ~1M) Hermes's reliability has to come from ON-DISK STATE it
#              RE-READS EVERY TURN, never from context surviving a session boundary -- so a missing
#              HERMES.md, a missing live_state.md re-read mandate, or a missing pre_llm_call bridge
#              script are treated as regressions too, not just unapplied config. Failing any of
#              these exits 1.
#   INFO     — everything that additionally depends on the owner applying hermes-parity/*.yaml|sh
#              to their global ~/.hermes/ (this script never touches that itself: the hooks
#              actually being REGISTERED, project-skill trust, the heartbeat cron job). Reported as
#              LIVE / PENDING / (skipped: hermes not on PATH); never affects the exit code, because
#              "not yet applied" is expected right after this lands, not a bug in this checker.
#
# Runs with no GPU and no heavy imports (bash + grep + a stdlib-only `python3 -m py_compile`), so
# it is safe to run from a Monitor / cron tick / CI without a VRAM or RSS budget concern.
set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO_ROOT" ]; then
    echo "⛔ not inside a git checkout — cannot verify the pre-commit gate"
    exit 1
fi
cd "$REPO_ROOT" || exit 1

COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
CANONICAL="$(dirname "$COMMON_DIR")"
FAIL=0
pass()  { printf '  \xe2\x9c\x93 %s\n' "$1"; }              # ✓
fail()  { printf '  \xe2\x9b\x94 %s\n' "$1"; FAIL=1; }       # ⛔
info()  { printf '  \xe2\x84\xb9 %s\n' "$1"; }               # ℹ
live()  { printf '  \xe2\x9c\x93 LIVE    %s\n' "$1"; }
pend()  { printf '  \xe2\x97\x8b PENDING %s\n' "$1"; }
skip()  { printf '  - skipped: %s\n' "$1"; }

echo "════ 1. GIT PRE-COMMIT GATE (BLOCKING — fires on any \`git commit\`, any agent) ════"

HOOKS_PATH="$(git config --get core.hooksPath 2>/dev/null || true)"
if [ -z "$HOOKS_PATH" ]; then
    fail "core.hooksPath is not set — the repo's gates (tools/gates/*.py) will NOT run on commit."
    info "fix: git config core.hooksPath tools/githooks"
else
    # core.hooksPath may be relative (tools/githooks) or absolute; resolve against CANONICAL.
    case "$HOOKS_PATH" in
        /*) RESOLVED="$HOOKS_PATH" ;;
        *)  RESOLVED="$CANONICAL/$HOOKS_PATH" ;;
    esac
    if [ "$(readlink -f "$RESOLVED" 2>/dev/null)" = "$(readlink -f "$CANONICAL/tools/githooks" 2>/dev/null)" ]; then
        pass "core.hooksPath -> $HOOKS_PATH (resolves to tools/githooks)"
    else
        fail "core.hooksPath is set to '$HOOKS_PATH', which does not resolve to tools/githooks"
    fi
fi

PRECOMMIT="$CANONICAL/tools/githooks/pre-commit"
if [ ! -f "$PRECOMMIT" ]; then
    fail "missing: $PRECOMMIT"
elif [ ! -x "$PRECOMMIT" ]; then
    fail "not executable: $PRECOMMIT (chmod +x)"
else
    pass "tools/githooks/pre-commit exists and is executable"
fi

if bash -n "$PRECOMMIT" 2>/tmp/hermes_parity_syntax.$$; then
    pass "tools/githooks/pre-commit: syntax OK (bash -n)"
else
    fail "tools/githooks/pre-commit: syntax error — $(cat /tmp/hermes_parity_syntax.$$)"
fi
rm -f /tmp/hermes_parity_syntax.$$

PY="$CANONICAL/.venv/bin/python"
if [ -x "$PY" ]; then
    pass "canonical engine interpreter present: $PY"
else
    fail "missing canonical engine interpreter: $PY (the hook exits 1 without it, for ANY agent)"
fi

echo
echo "════ 2. LEAN CONTEXT FILE + DURABLE-STATE RE-ANCHOR (the #1 reliability concern: Hermes's ════"
echo "════    ~100K context vs Claude's ~1M means reliability must come from ON-DISK STATE re-read  ════"
echo "════    every turn, never from context, and a session must never start from scratch)          ════"

# HERMES.md wins Hermes's context-file priority over CLAUDE.md (.hermes.md/HERMES.md > AGENTS.md >
# CLAUDE.md > .cursorrules, first found in the cwd-to-git-root walk wins, only ONE loads) --
# verified directly against agent/prompt_builder.py's build_context_files_prompt with the real
# installed Hermes: with HERMES.md present, the composed prompt contains "## HERMES.md" and no
# "## CLAUDE.md" section at all. So HERMES.md is the intended small file Hermes loads every
# session; CLAUDE.md (25x larger) stays available for on-demand deep reads, never bulk-loaded.
if [ -f "$REPO_ROOT/HERMES.md" ]; then
    pass "HERMES.md present at repo root (supersedes CLAUDE.md in Hermes's context-file priority)"
    HSIZE=$(wc -c < "$REPO_ROOT/HERMES.md" 2>/dev/null || echo 0)
    if [ "$HSIZE" -gt 15000 ]; then
        info "HERMES.md is ${HSIZE}B -- getting large for a 100K-context agent; keep it a lean pointer, not a copy of CLAUDE.md"
    else
        pass "HERMES.md is ${HSIZE}B (lean)"
    fi
    if grep -q "research/coordination/live_state.md" "$REPO_ROOT/HERMES.md" 2>/dev/null; then
        pass "HERMES.md mandates re-reading research/coordination/live_state.md every turn"
    else
        fail "HERMES.md does not mention research/coordination/live_state.md -- the durable re-anchor rule is missing"
    fi
else
    fail "HERMES.md missing at repo root -- Hermes will fall back to loading the full CLAUDE.md (or nothing) every session, with no compact per-turn re-anchor rule"
fi

# A stray .hermes.md (the dotfile variant) is checked BEFORE HERMES.md in the same directory
# (agent/prompt_builder.py::_HERMES_MD_NAMES = (".hermes.md", "HERMES.md")) -- if one exists with
# different/stale content it silently shadows HERMES.md.
if [ -f "$REPO_ROOT/.hermes.md" ]; then
    fail ".hermes.md also exists at repo root -- it is checked BEFORE HERMES.md and will shadow it if the two differ"
else
    pass "no .hermes.md shadowing HERMES.md"
fi

if [ -f "$REPO_ROOT/CLAUDE.md" ]; then
    pass "CLAUDE.md present (available for on-demand deep reads; not auto-loaded while HERMES.md exists)"
else
    info "CLAUDE.md missing -- HERMES.md's pointers to it for depth would 404"
fi
info "Hermes only discovers HERMES.md/CLAUDE.md when its cwd resolves into this repo -- point terminal.cwd / --in / cron --workdir at $REPO_ROOT"

if [ -f "$REPO_ROOT/research/coordination/live_state.md" ]; then
    pass "research/coordination/live_state.md present (the durable frontier/next-actions file HERMES.md points at and the pre_llm_call hook re-injects)"
else
    info "research/coordination/live_state.md absent -- will be generated on first run of tools/live_state.py"
fi

# The per-turn re-injection hook itself (see section 3 for the LIVE/PENDING check against the
# owner's actual ~/.hermes/config.yaml) -- here we just confirm the bridge script this depends on
# exists, since HERMES.md's rule is a fallback instruction for when the hook is NOT configured.
if [ -f "$REPO_ROOT/tools/hermes/hook_live_state_context.py" ]; then
    pass "tools/hermes/hook_live_state_context.py present (the pre_llm_call re-injection hook -- Hermes's verified UserPromptSubmit equivalent, see hooks doc worked example 3)"
else
    fail "tools/hermes/hook_live_state_context.py missing -- no automatic per-turn re-injection; HERMES.md's manual-read rule is the only fallback"
fi

echo
echo "════ 3. HERMES-SIDE PIECES (each needs hermes-parity/*.{yaml,sh} applied to ~/.hermes/) ════"

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
HERMES_BIN="$(command -v hermes 2>/dev/null || true)"

# --- bridge scripts in-repo (these ARE live the moment this branch merges; no owner step) ---
for f in tools/hermes/hook_live_state_context.py tools/hermes/hook_post_edit.py tools/hermes/hook_block_dangerous_kill.py; do
    if [ -f "$REPO_ROOT/$f" ]; then
        if python3 -m py_compile "$REPO_ROOT/$f" 2>/tmp/hermes_parity_pyc.$$; then
            live "$f (compiles cleanly)"
        else
            fail "$f fails to compile: $(cat /tmp/hermes_parity_pyc.$$)"
        fi
        rm -f /tmp/hermes_parity_pyc.$$
    else
        pend "$f (expected in-repo bridge script, not found)"
    fi
done

# --- .hermes/skills/ copies in-repo (deliberately COPIES, not symlinks -- a symlink pointing
# outside the skill's own directory trips a CRITICAL "traversal" finding in Hermes's project-skill
# scanner and gets silently quarantined; verified directly, see tools/hermes/sync_skills.sh) ---
SKILL_COUNT=0
DRIFTED=""
for name in neural-simulator sync-documentation evolve-skills verify-go vikunja cost-routing; do
    dst="$REPO_ROOT/.hermes/skills/$name/SKILL.md"
    src="$REPO_ROOT/.claude/skills/$name/SKILL.md"
    if [ -f "$dst" ]; then
        SKILL_COUNT=$((SKILL_COUNT + 1))
        if [ -f "$src" ] && ! cmp -s "$src" "$dst"; then
            DRIFTED="$DRIFTED $name"
        fi
    fi
done
if [ "$SKILL_COUNT" -eq 6 ]; then
    live ".hermes/skills/ (6/6 repo-local skills present, synced copies of .claude/skills/)"
else
    pend ".hermes/skills/ ($SKILL_COUNT/6 present)"
fi
if [ -n "$DRIFTED" ]; then
    fail ".hermes/skills/ out of sync with .claude/skills/ for:$DRIFTED -- run tools/hermes/sync_skills.sh"
fi

# --- global ~/.hermes/config.yaml hooks block ---
if [ -f "$HERMES_HOME/config.yaml" ]; then
    if grep -q "hook_live_state_context.py" "$HERMES_HOME/config.yaml" 2>/dev/null; then
        live "~/.hermes/config.yaml hooks: block (hermes-parity/config.hooks.snippet.yaml applied)"
    else
        pend "~/.hermes/config.yaml hooks: block (apply hermes-parity/config.hooks.snippet.yaml)"
    fi
else
    pend "~/.hermes/config.yaml not found at $HERMES_HOME/config.yaml"
fi

# --- global skills trust ---
if [ -f "$HERMES_HOME/config.yaml" ]; then
    if grep -q "$REPO_ROOT" "$HERMES_HOME/config.yaml" 2>/dev/null; then
        live "project skill trust (repo path found in ~/.hermes/config.yaml)"
    else
        pend "project skill trust (run hermes-parity/skills_trust.sh)"
    fi
fi

# --- heartbeat cron job ---
if [ -x "$HERMES_HOME/scripts/sim_heartbeat.sh" ]; then
    live "~/.hermes/scripts/sim_heartbeat.sh installed"
else
    pend "~/.hermes/scripts/sim_heartbeat.sh (run hermes-parity/apply_cron.sh)"
fi
if [ -n "$HERMES_BIN" ]; then
    if timeout 15 hermes cron list 2>/dev/null | grep -qi "sim-heartbeat"; then
        live "hermes cron job 'sim-heartbeat' registered"
    else
        pend "hermes cron job 'sim-heartbeat' (run hermes-parity/apply_cron.sh)"
    fi
else
    skip "hermes cron list (hermes not found on PATH)"
fi

echo
echo "════ 4. AUTONOMOUS-MODE PIECES (BLOCKING — these are in-repo files, same rationale as ════"
echo "════    section 2: their absence is a regression, not just an unapplied ~/.hermes step)   ════"

for f in tools/hermes_say.sh tools/hermes_autonomous.sh tools/hermes_health_check.sh tools/hermes_desktop_control.sh; do
    if [ ! -f "$REPO_ROOT/$f" ]; then
        fail "missing: $f"
    elif ! bash -n "$REPO_ROOT/$f" 2>/tmp/hermes_parity_bashn.$$; then
        fail "$f: syntax error — $(cat /tmp/hermes_parity_bashn.$$)"
    else
        pass "$f present, bash -n clean"
    fi
    rm -f /tmp/hermes_parity_bashn.$$
done

if [ -f "$REPO_ROOT/tools/hermes/hook_live_state_context.py" ] && grep -q "_drain_feedback_queue\|hermes_feedback_queue" "$REPO_ROOT/tools/hermes/hook_live_state_context.py" 2>/dev/null; then
    pass "pre_llm_call hook drains the hermes_say.sh feedback queue (owner can queue feedback without interrupting)"
else
    fail "tools/hermes/hook_live_state_context.py does not drain the feedback queue — hermes_say.sh's messages would never surface"
fi

if [ -f "$REPO_ROOT/hermes-parity/scripts/sim_heartbeat.sh" ] && grep -q "HERMES_ACTIVE" "$REPO_ROOT/hermes-parity/scripts/sim_heartbeat.sh" 2>/dev/null && grep -q "GAME_MODE" "$REPO_ROOT/hermes-parity/scripts/sim_heartbeat.sh" 2>/dev/null; then
    pass "hermes-parity/scripts/sim_heartbeat.sh gates on HERMES_ACTIVE/GAME_MODE (a stray cron tick is a safe no-op)"
else
    fail "hermes-parity/scripts/sim_heartbeat.sh missing the HERMES_ACTIVE/GAME_MODE gate — a stray tick while Claude drives or the owner is paused would still run the full heartbeat body"
fi

if [ -f "$REPO_ROOT/tools/hermes_takeover.sh" ] && grep -q "hermes_autonomous.sh" "$REPO_ROOT/tools/hermes_takeover.sh" 2>/dev/null; then
    pass "tools/hermes_takeover.sh wires autonomous mode on/off into the driver handoff"
else
    fail "tools/hermes_takeover.sh does not call hermes_autonomous.sh — autonomous mode would not turn on/off with the driver switch"
fi

if [ -f "$REPO_ROOT/tools/game.sh" ] && grep -q "hermes_autonomous.sh" "$REPO_ROOT/tools/game.sh" 2>/dev/null; then
    pass "tools/game.sh pauses/resumes autonomous mode around a gaming break"
else
    fail "tools/game.sh does not call hermes_autonomous.sh — a gaming pause would not pause the autonomous heartbeat cron"
fi

# --- autonomous-mode ~/.hermes pieces (owner-applied; INFO, not BLOCKING) ---
if [ -n "$HERMES_BIN" ]; then
    if timeout 15 hermes gateway status 2>/dev/null | grep -qi "gateway service is running"; then
        live "hermes gateway running (the cron ticker can fire)"
    else
        pend "hermes gateway (run: bash tools/hermes_autonomous.sh on)"
    fi
else
    skip "hermes gateway status (hermes not found on PATH)"
fi

echo
if [ "$FAIL" -eq 0 ]; then
    echo "RESULT: OK — the git-gate + durable-state-reanchor + autonomous-mode parity checks are"
    echo "        intact. See PENDING lines above for owner steps still needed on this machine's"
    echo "        ~/.hermes/ (hermes-parity/README.md has the order)."
else
    printf 'RESULT: FAIL \xe2\x80\x94 one or more BLOCKING checks failed (see \xe2\x9b\x94 lines above).\n'
fi
exit $FAIL
