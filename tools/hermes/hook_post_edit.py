#!/usr/bin/env python3
"""Hermes `post_tool_call` shell hook (matcher: write_file|patch) — merges what Claude Code splits
across two PostToolUse scripts: .claude/hooks/live_state_nudge.py (regenerate the durable LIVE-STATE
file on a landing) and .claude/hooks/check_doc_drift.py (run the W1/W2 doc-structure checker + nudge
sync-documentation / keep-webapp-current after an edit that historically causes drift).

WHY ONE SCRIPT, AND WHY IT WRITES A FILE INSTEAD OF PRINTING A MESSAGE. Hermes's post_tool_call is
OBSERVER-ONLY: model_tools.py's `_emit_post_tool_call_hook()` calls `invoke_hook("post_tool_call",
...)` and discards the return value (no `systemMessage` channel exists for this event the way Claude
Code's PostToolUse has). So this hook cannot show the agent anything directly -- it can only cause a
SIDE EFFECT. It regenerates research/coordination/live_state.md immediately (same as Claude Code),
and for anything a human/agent should actually be told (a docs/WRITING.md violation, a
sync-documentation nudge) it appends to research/coordination/.hermes_pending_advisory, a small
queue that tools/hermes/hook_live_state_context.py (the pre_llm_call sibling) drains and injects as
context on the NEXT LLM call. One call behind Claude Code's immediate systemMessage, not zero.

Registration (~/.hermes/config.yaml, see hermes-parity/config.hooks.snippet.yaml):

    hooks:
      post_tool_call:
        - matcher: "write_file|patch"
          command: "python3 /home/dant123/Projects/sim/tools/hermes/hook_post_edit.py"
          timeout: 30

Fail-open: any error is swallowed. Never blocks (post_tool_call cannot block in Hermes anyway).
"""
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PENDING_ADVISORY = os.path.join(REPO, "research", "coordination", ".hermes_pending_advisory")

# Same patterns as .claude/hooks/live_state_nudge.py -- editing one of these changes the
# frontier / next-actions / last-decision that LIVE-STATE carries.
DURABLE_SOURCE = re.compile(
    r"(?:^|/)GAP_CLOSURE_MISSION\.md$"
    r"|(?:^|/)docs/PRODUCTION_INTEGRATION_LEDGER\.yaml$"
    r"|(?:^|/)research/coordination/backlog\.json$"
    r"|(?:^|/)research/findings/[^/]+\.md$"
)

# Same set as .claude/hooks/check_doc_drift.py.
GOVERNED_DOCS = {
    "CLAUDE.md", "GAP_CLOSURE_MISSION.md", "ROADMAP.md", "README.md", "docs/TERMS.md",
    "docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md",
}
SIGNIFICANT_PATTERN = re.compile(
    r"(?:^|/)(?:sim|experiment)/[^/]+\.py$"
    r"|(?:^|/)research/runners/[^/]+\.py$"
    r"|(?:^|/)research/findings/[^/]+\.md$"
)
WEBAPP_RELEVANT_PATTERN = re.compile(
    r"(?:^|/)research/runners/g11_bg_runner\.py$"
    r"|(?:^|/)sim/(bridge|regions|neuromodulators)\.py$"
)


def _append_advisory(text: str) -> None:
    try:
        os.makedirs(os.path.dirname(PENDING_ADVISORY), exist_ok=True)
        with open(PENDING_ADVISORY, "a", encoding="utf-8") as fh:
            fh.write(text.strip() + "\n\n")
    except Exception:
        pass


def _regen_live_state_fast() -> None:
    try:
        env = dict(os.environ)
        env["LIVE_STATE_FAST"] = "1"
        subprocess.run(
            [sys.executable, os.path.join(REPO, "tools", "live_state.py")],
            cwd=REPO, capture_output=True, text=True, timeout=20, env=env,
        )
    except Exception:
        pass


def _check_docs_violation(rel: str) -> str:
    checker = os.path.join(REPO, "tools", "check_docs.py")
    if not os.path.exists(checker):
        return ""
    py = os.path.join(REPO, ".venv", "bin", "python")
    if not os.path.exists(py):
        py = sys.executable
    try:
        r = subprocess.run([py, checker], cwd=REPO, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            out = "\n".join(
                ln for ln in r.stdout.splitlines() if ln.strip() and not ln.startswith("OK")
            )[:1200]
            return (
                "docs/WRITING.md VIOLATION after editing " + rel + ":\n" + out +
                "\nFix now (tools/split_long_doc_lines.py --apply for W2; add a retraction marker "
                "or a docs/RETRACTED.md row for W1)."
            )
    except Exception:
        pass
    return ""


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    tool_input = payload.get("tool_input") or {}
    path = tool_input.get("path") or tool_input.get("file_path") or ""
    if not path:
        return
    normalized = str(path).replace("\\", "/")

    if DURABLE_SOURCE.search(normalized):
        _regen_live_state_fast()

    # rel-ize against the repo root the same way check_doc_drift.py does.
    rel = normalized.split(os.path.basename(REPO) + "/")[-1] if REPO else normalized
    if rel in GOVERNED_DOCS:
        violation = _check_docs_violation(rel)
        if violation:
            _append_advisory(violation)
            return  # matches check_doc_drift.py: a structure violation short-circuits the nudge

    significant = bool(SIGNIFICANT_PATTERN.search(normalized))
    webapp_relevant = bool(WEBAPP_RELEVANT_PATTERN.search(normalized))
    if significant or webapp_relevant:
        skills = []
        if significant:
            skills.append("sync-documentation")
        if webapp_relevant:
            skills.append("keep-webapp-current")
        _append_advisory(
            f"Code change in {path} may have made the {' and '.join(skills)} "
            f"contract(s) stale. Run the relevant skill(s) to verify field names, flag lists, "
            f"recipe headlines, and webapp launcher presets."
        )


if __name__ == "__main__":
    main()
