# Test-triage WORKLOG (research/draft-test-triage)

## 2026-10-01 08:22 (local model, Hermes) — section 5 idle: suite in flight

- **Why this task:** no section 4 task can move (T1 STOP #2; T2 pool-hold; rest pool-gated / done /
  need-ruling; T13 cupy rc pending on a busy GPU). Section 5 "Idle (only when no task can move)" = test triage.
- **Run:** `SIM_NO_PROVENANCE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider
  --continue-on-collection-errors --ignore=tests/test_game_pause_resume.py --junitxml=_scratch/suite.xml -rfE tests`
  from `.claude/worktrees/draft-test-triage` (branch `research/draft-test-triage` from main; `.venv` symlinked to
  main's — no editable install, cwd-based imports). GPU hidden (CPU-only).
- **memcap deviation (honest):** `tools/memcap.sh 12` REFUSED in this container — no `systemd --user`
  (no systemctl, no `/run/user/99`) and cgroup v2 is read-only. Ran **uncapped** per the script's own
  "run the job yourself only if you accept the OOM risk" clause. Risk assessed low: no live training on S,
  23 GB free of 61 GB, and the OOM target would be the test process (largest), not the small session.
- **State at handoff:** suite running in background (`proc_dd364382776e`, pytest pid 22903), ~4% at handoff.
  Long run (~1.5–2h). Log: `/tmp/triage_run.log`. On completion: read `_scratch/suite.xml`, group failures by
  first error line (GPU-hidden = ENVIRONMENT), write the triage finding in `docs/drafts/test-triage/`, commit
  through the gates. No fixes (per section 5).
## 2026-10-01 12:45 (local model, Hermes) — suite complete, triage written

- **Suite finished 12:07:** `175 failed, 7258 passed, 497 skipped, 3 xfailed, 4 errors in 13587.50s (3:46:27)`,
  rc=1 (expected for a triage run). `_scratch/suite.xml` (5.6 MB) written. Started 08:20 without `nice`
  (the `server-load` ruling landed after start; isolation re-runs used `nice -n 10`).
- **Grouping (section 5):** 179 failed/errored tests → **76 distinct first-error lines** via
  `_scratch` script parsing suite.xml (`E `-prefixed first line, message-attr fallback; 0 ungrouped).
  Full list: `docs/drafts/test-triage/grouped-first-error-lines.txt`.
- **Classification** (evidence-backed, counts sum to 179): **ENVIRONMENT 66** (no cupy/torch/dearpygui/rsync/
  systemctl in container; gitignored `data/` absent from worktree — verified `git check-ignore`; hardcoded
  `/home/dant123/Projects/sim` POOL_ROOT absent on S); **POLLUTION 33** (vocabulary KeyErrors — verified
  representative tests PASS in isolation, import-time `cpd_v1.*_VOCAB` mutation by other runners);
  **REAL 28** (composer-D mismatch, emerge61 broadcast, PROTECTED divergence — verified FAIL in isolation
  AND on the main checkout, `%o` format bug, provenance sidecar); **UNVERIFIED 52** (grouped only, no
  isolation run, no fixes per section 5).
- **Document:** `docs/drafts/test-triage/TRIAGE.md` (372 lines, max line 304 ≤ W2's 800). No fixes.
- **Deviations:** memcap refused in container (no systemd --user) → uncapped per its documented clause;
  run predated the `server-load` nice ruling.