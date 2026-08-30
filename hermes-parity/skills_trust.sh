#!/usr/bin/env bash
# Trust the neural-simulator repo so Hermes loads its repo-local skills.
#
# NOT RUN AUTOMATICALLY -- review, then run by hand. `hermes skills trust` writes to
# ~/.hermes/config.yaml (skills.trusted_project_dirs), which the task that produced this file was
# told not to touch itself.
#
# BACKGROUND: Hermes's repo-local skill discovery looks for ./.hermes/skills/<name>/SKILL.md or
# ./.agents/skills/<name>/SKILL.md at the project root (agent/skill_utils.py; confirmed against
# tests/agent/test_project_skills.py) -- NOT .claude/skills/, which is Claude-Code-specific. The
# SKILL.md FORMAT is identical (YAML frontmatter `name:`/`description:` + a markdown body), so no
# content conversion was needed. This worktree adds .hermes/skills/<name>/SKILL.md as a PLAIN COPY
# of the existing .claude/skills/<name>/SKILL.md for each of the 6 skills:
#   .hermes/skills/neural-simulator/SKILL.md    (copy of .claude/skills/neural-simulator/SKILL.md)
#   .hermes/skills/sync-documentation/SKILL.md  (copy of .claude/skills/sync-documentation/SKILL.md)
#   .hermes/skills/evolve-skills/SKILL.md       (copy of .claude/skills/evolve-skills/SKILL.md)
#   .hermes/skills/verify-go/SKILL.md           (copy of .claude/skills/verify-go/SKILL.md)
#   .hermes/skills/vikunja/SKILL.md             (copy of .claude/skills/vikunja/SKILL.md)
#   .hermes/skills/cost-routing/SKILL.md        (copy of .claude/skills/cost-routing/SKILL.md)
#
# A COPY, DELIBERATELY NOT A SYMLINK (found empirically, not assumed). Hermes scans every trusted
# project skill with tools/skills_guard.py before loading it; a "dangerous" verdict silently
# excludes it (fail-closed). A symlink whose target resolves outside the skill's own directory
# trips a CRITICAL "traversal" finding ("symlink points outside the skill directory") BY ITSELF --
# verified directly against the scanner, all 6 skills scored "dangerous" as symlinks into
# ../../../.claude/skills/, and "safe" or "caution" (both LOAD; only "dangerous" is excluded) as
# plain copies of the byte-identical content. Two of the six (neural-simulator, verify-go) score
# "caution" on real content -- a `os.environ` mention inside a past-bug narrative reads as a
# potential env-dump to the scanner -- but caution still loads (matches the hub's own behavior for
# prose-level keyword hits; only "dangerous" quarantines).
#
# Because it is a copy, IT WILL DRIFT if .claude/skills/<name>/SKILL.md is edited later. Re-run
# tools/hermes/sync_skills.sh after any such edit (not wired into a hook here -- see
# docs/HERMES_WORKFLOW_PARITY.md's "known gaps" section).
#
# Until trusted, Hermes only prints an untrusted-skills-root NOTICE
# (get_untrusted_project_skills_root()) and loads nothing from .hermes/skills/ -- this is the
# one-time consent step.
set -euo pipefail
hermes skills trust /home/dant123/Projects/sim
echo "done. Verify with:  hermes skills list --source local   (the 6 sim skills should appear)"
