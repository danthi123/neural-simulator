# Startup-context condense: summary for the owner (2026-09-25)

Branch `research/condense-startup-context`, not merged. Nothing in your memory folder was changed.

## Sizes, before and after

| What loads at the start of a session | Before | After |
|---|---|---|
| `CLAUDE.md` | 25,670 bytes (≈6.4K tokens) | 11,126 bytes (≈2.8K tokens) |
| Memory index `MEMORY.md` (99 lines) | 19,425 chars (≈4.9K tokens) | 5,932 chars, 65 lines (≈1.5K tokens), once you apply it |
| **Whole first request of `llm claude`**, measured (after = new CLAUDE.md + proposed index + `llm.sh` trims) | 111,696 chars (≈27.9K tokens, ~43% of the 64K window) | 81,081 chars (≈20.3K tokens, ~31%) |

The last row is a real measurement, not an estimate: I pointed Claude Code (in `-p` mode) at a small local recorder
instead of the model and captured what it actually sends. Tokens are characters ÷ 4 (approximate). What is
left after the change is mostly Claude Code itself: the tool definitions (~45,700 chars), its system prompt and
environment block (~15,000 chars), and the LIVE-STATE note (~1,900 chars, which currently arrives twice on the first
turn: once from the session-start hook and once from the per-message hook).

## What moved where

- **`CLAUDE.md`** now holds the mission, the pointers (master roadmap, `GAP_CLOSURE_MISSION.md`, `ROADMAP.md`), 12
  non-negotiables, the workflow habits that no gate enforces, the compaction instruction (same keep-list), the
  `cfg.seed` trap in two lines, units and commands. **Every rule is still there**; the stories, incident history and
  your full directive wording moved **verbatim** into `docs/CLAUDE_RATIONALE_ARCHIVE.md`, which holds the entire old
  file plus a table of where each old section's rule now lives.
- **Memory:** `.claude/memory_condense/PROPOSED_MEMORY.md` is the new index (grouped: your directives, project
  state, operations, working habits). `ARCHIVE_LIST.md` lists the 35 notes to archive, each with its reason, plus the
  exact commands to apply it. None is a directive still in force.
- **Local model only:** `tools/local_llm/llm.sh claude` now (1) tells Claude Code the model's real window (65,536
  tokens). It did not know it for a model named "local", so it would have compacted too late. (2) It drops two tools
  that cannot work or are never used locally: web search and the code-review reporter. MCP servers and the
  bio-research plugin are also forced off, but a terminal session loads neither today, so those two flags are only
  a guard. `LLM_CLAUDE_FULL=1 llm claude` skips the tool/plugin flags. Tested with the real Claude Code binary against
  the recorder; your running model was not touched.
- Small pointer fixes so nothing depends on removed text: `HERMES.md`, a comment in `tools/lane_dispatch.sh`.
  `CLAUDE.md` keeps the phrases other files quote ("ACTIVE MISSION", "ONE spiking substrate", "recent-output", the
  ROADMAP "skim" link, "no watchdog/daemon"). The doc checks, `tests/test_doc_rules.py` and all 45 gate selftests
  pass.

## What you should decide

1. **Merge the branch?** Two lines in `CLAUDE.md` are new: they note your newer rulings where the old text had gone
   stale. One is the 09-19 permanent Qwen mouth, the other the 09-18 per-mechanism realism-for-speed trade. Remove
   them if you prefer the old wording.
2. **Apply the memory condense** (commands in `ARCHIVE_LIST.md`). Four archives are marked *owner: confirm*: they are
   older directives whose content newer notes now carry.
3. **Conflicts that need your ruling** (all 20 are listed in `ARCHIVE_LIST.md`):
   - Is learning-over-time now in scope before scaffold retirement is finished? The 09-04 order says wait; the 09-18,
     09-24 and 09-25 rulings point at learning now.
   - Is the extra-3090 hardware plan on hold after the 09-18 "don't buy hardware yet" finding?
   - The April note "don't merge to main without approval" conflicts with today's practice.
   - The affect-marker note still says "never flip without sign-off"; you have since retired the marker word, so the
     note itself should be updated.
4. **Optional, for the local model:** stop the LIVE-STATE note being injected twice on the first turn. It also costs
   about 480 tokens on every message you type. This needs a small hook change, outside this task's scope.
5. **Found, not fixed:** `.claude/style.md` is not in git, so it has no backup. It is also never auto-loaded. The
   `sync-documentation` skill still checks `CLAUDE.md` for line counts and recipes that moved out on 2026-07-31.
