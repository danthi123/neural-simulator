#!/usr/bin/env bash
# Re-sync .hermes/skills/<name>/SKILL.md from .claude/skills/<name>/SKILL.md.
#
# WHY A COPY, NOT A SYMLINK (found empirically, 2026-08-30). Hermes's project-skill loader runs a
# content scanner (tools/skills_guard.py) over every trusted project skill before loading it, and
# a "dangerous" verdict silently excludes it from `_find_all_skills()` -- from skills_list,
# skill_view, AND slash-command dispatch (agent/skill_utils.py::is_quarantined_project_skill,
# "fail-closed: a scanner crash or missing scanner quarantines the skill"). A symlink whose target
# resolves OUTSIDE the skill's own directory trips a CRITICAL "traversal" finding
# ("symlink points outside the skill directory") on its own, independent of the linked content --
# verified directly against the scanner: all 6 of this repo's skills scored "dangerous" as
# symlinks into ../../../.claude/skills/, and "safe"/"caution" (both of which DO load; only
# "dangerous" is excluded) as plain file copies of the identical bytes. So the two directories
# cannot be symlinked together; this script is the sync step that keeps them from drifting apart.
#
# Run it whenever a .claude/skills/<name>/SKILL.md changes (or add it to a pre-commit / CI step --
# not wired automatically here, see docs/HERMES_WORKFLOW_PARITY.md).
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

for name in neural-simulator sync-documentation evolve-skills verify-go vikunja cost-routing; do
    src=".claude/skills/$name/SKILL.md"
    dst=".hermes/skills/$name/SKILL.md"
    if [ ! -f "$src" ]; then
        echo "  skip: $src does not exist"
        continue
    fi
    mkdir -p "$(dirname "$dst")"
    if [ -f "$dst" ] && cmp -s "$src" "$dst"; then
        echo "  up to date: $dst"
    else
        cp "$src" "$dst"
        echo "  synced: $src -> $dst"
    fi
done

echo
echo "Verify with (from a machine that has hermes installed, after 'hermes skills trust $ROOT'):"
echo "  hermes skills list --source local"
