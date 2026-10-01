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
