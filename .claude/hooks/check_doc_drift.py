"""PostToolUse hook: nudge to run sync-documentation and/or
keep-webapp-current skills after edits that historically cause drift.

Reads the tool-call JSON from stdin and emits a hookSpecificOutput JSON to
stdout iff the changed file matches one of two patterns:

1. SIGNIFICANT_PATTERN: drifts the prose docs (line numbers, recipe
   headlines, flag lists). Nudges sync-documentation.
2. WEBAPP_RELEVANT_PATTERN: drifts the dashboard's contract with the
   simulator (field names, print formats, flag names). Nudges
   keep-webapp-current.

A single change can trigger both.

Stays silent on tests, UI internals, etc.

Wired up in .claude/settings.json under hooks.PostToolUse with matcher
"Write|Edit".
"""
import json
import os
import re
import subprocess
import sys

# Files governed by docs/WRITING.md's two structure rules. Editing one runs the checker
# IMMEDIATELY, so a violation is reported while the edit is fresh rather than at some later
# audit. Earned 2026-07-28: three stale citations sat in these files (one on the MASTER
# ROADMAP, presenting a retracted attribution as current) until a checker was finally run.
GOVERNED_DOCS = {
    "CLAUDE.md", "GAP_CLOSURE_MISSION.md", "ROADMAP.md", "README.md", "docs/TERMS.md",
    "docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md",
}

# Drifts written documentation (CLAUDE.md line numbers, INDEX entries, etc.)
SIGNIFICANT_PATTERN = re.compile(
    r"(?:^|/)(?:sim|experiment)/[^/]+\.py$"
    r"|(?:^|/)research/runners/[^/]+\.py$"
    r"|(?:^|/)research/findings/[^/]+\.md$"
)

# Drifts the webapp's contract with the simulator (subset of significant — the
# webapp doesn't care about findings text, but does care about runner flags
# and recorded JSON field names).
WEBAPP_RELEVANT_PATTERN = re.compile(
    r"(?:^|/)research/runners/g11_bg_runner\.py$"
    r"|(?:^|/)sim/(bridge|regions|neuromodulators)\.py$"
)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return

    tool_input = data.get("tool_input") or {}
    tool_response = data.get("tool_response") or {}
    path = tool_input.get("file_path") or tool_response.get("filePath") or ""
    if not path:
        return

    normalized = path.replace("\\", "/")

    # --- docs/WRITING.md structure check: run it, do not merely nudge ---
    rel = normalized.split("Projects/sim/")[-1]
    if rel in GOVERNED_DOCS:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        checker = os.path.join(root, "tools", "check_docs.py")
        if os.path.exists(checker):
            try:
                r = subprocess.run([sys.executable, checker], capture_output=True, text=True,
                                   cwd=root, timeout=30)
                if r.returncode != 0:
                    out = "\n".join(l for l in r.stdout.splitlines()
                                    if l.strip() and not l.startswith("OK"))[:1200]
                    print(json.dumps({"systemMessage":
                        "docs/WRITING.md VIOLATION after editing " + rel + ":\n" + out +
                        "\nFix now (tools/split_long_doc_lines.py --apply for W2; add a ⛔ marker or a "
                        "docs/RETRACTED.md row for W1). These are structure rules only — they do not "
                        "check whether the claim is TRUE."}))
                    return
            except Exception:
                pass

    significant = bool(SIGNIFICANT_PATTERN.search(normalized))
    webapp_relevant = bool(WEBAPP_RELEVANT_PATTERN.search(normalized))
    if not (significant or webapp_relevant):
        return

    skills = []
    if significant:
        skills.append("sync-documentation")
    if webapp_relevant:
        skills.append("keep-webapp-current")
    skills_str = " and ".join(skills)

    msg = (
        f"Code change in {path} may have made the {skills_str} "
        f"contract(s) stale. Run the relevant skill(s) to verify field "
        f"names, flag lists, recipe headlines, and webapp launcher presets."
    )
    print(json.dumps({"systemMessage": msg}))


if __name__ == "__main__":
    main()
