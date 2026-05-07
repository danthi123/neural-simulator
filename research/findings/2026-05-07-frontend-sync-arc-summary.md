# 2026-05-07 — Frontend-sync arc + Tier 2.1 chat demo

**Triggered by user prompt:** "have you been keeping the frontend
updated as well to allow access to new features and capabilities and
such?" — followed by "update the autonomous-runs skill to include
frontend updates as part of its flow."

Honest answer: NO. Over ~16 hours of autonomous work I'd shipped 8+
new runners (Phase 1.3, 1.4, 1.5, Tier 2.3, chat demos, Phase 2.1/2.2)
but the webapp launcher only had the original 4 presets. The
dashboard — the user's primary observation surface — was lying.

## What this arc fixed

### 1. autonomous-runs skill hardened (principle #10)

`C:\Users\dant123\.claude\skills\autonomous-runs/SKILL.md` now has a
new principle: **Frontend stays in sync with backend capabilities.**
The decision rule is "backend + frontend = single unit of work, not
'implement now, expose later'." References the existing
`keep-webapp-current` and `sync-documentation` skills so the
autonomous skill orchestrates them rather than reinventing.

Also added 2 new entries to the Anti-patterns table and updated
Step 2 of the Process to require a frontend-sync sweep after each
new user-facing capability.

### 2. Webapp drift caught + fixed (2 bugs)

Running `keep-webapp-current` flagged two real bugs:

- **Bug 1 (commit f19ae0e):** `phase_2_1_abc` and `phase_2_2_shakespeare`
  presets pointed to `cortex_pretraining.py`, which lives only on the
  `path-f-hybrid` branch. Clicking them from main would fail with
  "No module named research.runners.cortex_pretraining". Removed both
  presets, left a comment pointing users to `git checkout path-f-hybrid`.

- **Bug 2 (commit 0341bb9):** Launcher was injecting
  `--interactive-control-file` and `--progress-print-interval` for ALL
  non-text-io presets, but 6 new runners (chat_demo, chat_continual_demo,
  continual_forgetting_eval, consolidation_trainer, continual_eval_suite,
  phrase_trainer) reject these flags as "unrecognized arguments". Fix:
  `supports_live_mode = (preset not in PRESET_RUNNERS) and not is_text_io`.
  Only g11_bg_runner presets get live-mode flags.

Both fixed via the standard plan→implement→test loop. Each bug got a
regression test (commit ac740f6) that's now part of `pytest
tests/test_webapp_server.py` (33 tests, all passing).

### 3. Doc-sync drift fixed (commit 547af55)

Running `sync-documentation` caught CLAUDE.md drift accumulated over
~2 weeks of sim/ growth:
- sim/: 13 → 15 modules, 11.8K → 12.3K lines (+ bioparameter.py,
  progress.py)
- bridge.py: 6037 → 6093 lines
- VisualizationConfig: line 292 → 357
- RuntimeState: line 312 → 377
- GPUConfig: line 327 → 392
- _run_one_simulation_step: line 4210 → 4236
- _initialize_simulation_data: line 823 → 831
- runners: 26 → 57
- findings: 93+ → 177+
- tests: 40 → 57

### 4. New capabilities shipped

- **`chat_synonym_demo.py`** (commit 92b133f): Tier 2.1 8-word synonym
  chat demo. User types "north" OR "up" → motor_N activates. Built on
  validated Tier 2.1 v4 scale-up arch (n_lang=4096, n_motor=1000,
  n_motor_fs=120). ~10 min single seed. Wired into webapp PRESETS
  same-commit.

- **`chat_demo_aggregate.py`** (commit d48bd48): Multi-seed aggregator
  that handles all 3 chat demo types (Tier 1, synonym, continual).
  Reports mean/std/range accuracy, per-action breakdown, and
  primary-vs-synonym split for synonym demos.

- **`scripts/multiseed_chat_demo.sh`** (commit 520eb04): Bash helper
  that sequentially launches N seeds via the webapp API and aggregates
  results. Default 6 seeds (Phase 1.4 protocol). Works for any chat
  demo preset.

### 5. Dashboard now exposes everything

- 26 presets total (was 4 when arc started)
- Webapp running with `--reload` so future changes auto-pick-up
- All presets verified to launch with clean cmd via regression tests
- `/api/info` confirms: chat_demo, chat_continual_demo,
  chat_synonym_demo, phase_1_3_consolidation, phase_1_4_forgetting,
  phase_1_5_unified, tier_2_3_phrases all dispatchable

### 6. Documentation

- README "Try it in 60 seconds" now includes chat demo entry points
  (commit 350dad2)
- CHAT-DEMO-GUIDE.md section 1b for Tier 2.1 synonym demo
  (commit 5754932)
- Phase 1.3 + Tier 2.1 combined design plan
  (commit c192dec) — next-phase research direction

## In-flight at session end

- **chat_synonym_demo seed 42** via dashboard launch API (run_id
  `55bfaa8600b0`, started ~15:56). Validates the new runner
  end-to-end. ETA ~13-15 min total (Tier 2.1 v4 scale-up is slower
  than initially estimated 10 min).

- **chat_continual_demo seed 43** finished earlier in the session:
  primary 38% / retention 60% — MODERATE single-seed result, not
  comparable to Phase 1.4 BRANCH A's 6-seed mean 103% retention
  (single-seed variance is expected; 1/6 seeds in Phase 1.4 also
  failed at 22% retention).

## Commit history

```
520eb04 feat: scripts/multiseed_chat_demo.sh -- N-seed launcher
2d813e8 chore: remove stale .pid files from prior session
c192dec plan: Phase 1.3 + Tier 2.1 combined consolidation design
350dad2 docs: README -- add chat demo entry points
d48bd48 feat: chat_demo_aggregate -- multi-seed aggregator
5754932 docs: CHAT-DEMO-GUIDE -- add Tier 2.1 synonym demo entry
92b133f feat: chat_synonym_demo runner + webapp preset
ac740f6 test: regression tests for live-mode flag gating
0341bb9 fix: webapp -- gate live-mode flags by runner type
547af55 docs: CLAUDE.md -- sync drift
f19ae0e fix: webapp -- remove phase_2_* presets
7a18d42 feat: webapp -- 8 new presets (earlier in arc)
```

## What the user can verify

1. **Dashboard launchable demos:** open http://127.0.0.1:8765, click
   the launcher tab. Should see 26 presets including 3 chat demos.
   Click any → cmd line is clean (no flag rejection).

2. **Skill update:** `cat
   ~/.claude/skills/autonomous-runs/SKILL.md` should show the new
   "### 10. Frontend stays in sync with backend capabilities" section.

3. **Tests:** `pytest tests/test_webapp_server.py -v` — 33 tests pass,
   including the 2 new regression tests
   (`test_launch_skips_live_mode_flags_for_overridden_runners`,
   `test_launch_keeps_live_mode_flags_for_g11_runners`).

4. **CLAUDE.md drift gone:** numbers in the project structure block
   match actual code state.

## Next directions

1. **Multi-seed chat_demo run** (~36 min): run
   `bash scripts/multiseed_chat_demo.sh chat_demo` to validate the
   ~33-45% Tier 1 baseline at 6 seeds.

2. **Multi-seed chat_synonym_demo run** (~60 min): same but with
   Tier 2.1. Should match the validated 6-seed BREAKTHROUGH numbers
   (W→A 5/6, A→W 6/6 aligned, A→W mean 63.7%).

3. **Phase 1.3 + Tier 2.1 combined** (~30 min/seed × 3 = 90 min):
   implement per `docs/plans/2026-05-07-Phase1.3-Tier2.1-combined-design.md`,
   smoke test, then 3-seed validation. Tests CLS theory at
   synonym scale.

4. **Tier 2.1 12-word vocab chat demo:** Tier 2.1 also validated
   12-word; could ship a variant that demonstrates the larger vocab.

## Lessons learned

- **Frontend drift is invisible.** The webapp's `/api/info` endpoint
  cheerfully reported its old preset list while I was happily shipping
  new runners. Without the user's prompt, this would have continued.
  Principle #10 in autonomous-runs is the systemic fix.

- **Project skills > one-off audits.** Both `keep-webapp-current` and
  `sync-documentation` existed before this arc but I wasn't routinely
  invoking them. Adding them to autonomous-runs as periodic-sweep
  recommendations should make this drift visible early.

- **Smoke-test through the actual user path.** I added 8 presets via
  the PRESETS dict and assumed they'd work. Smoke-testing through
  `/api/runs/launch` immediately exposed the live-mode flag bug. Lesson:
  always exercise the integration path, not just the unit tests.

- **`uvicorn --reload` on Windows is unreliable.** WatchFiles often
  doesn't pick up edits; manual kill + restart is more reliable. Worth
  a project memo somewhere.
