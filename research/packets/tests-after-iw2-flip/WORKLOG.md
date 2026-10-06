# tests-after-iw2-flip — §4 triage WORKLOG

## 2026-10-06 04:30-04:45 — triage (ruling tests-after-iw2-flip, runbook line 921)

- Command (local): `.venv/bin/python -m pytest tests/test_open_ended_generation_fluent.py -q` -> **2 failed,
  3 passed**: test_render_hypothesis_fluent_flagged_guess_stub (:69), test_render_hypothesis_template_fallback_
  without_mouth (:100). Full tb: /opt/data/cache/scratch/fluent_fail.log.
- Symptom 1: stub surface = "perhaps the dog likes the fish  [a guess ...]" but :69 expects startswith("maybe ")
  and :72 asserts NOT "perhaps". Symptom 2: with renderer=None, fluent=True (expected False) at :100.
- Root (verified in code): ONE cause = the SPIKING BROCA mouth, default ON. `render_hypothesis_verified`
  (research/runners/brain_chat_tui.py:1810) runs the spiking mouth BEFORE the `raw_mode or renderer is None`
  check (:1816), so (a) the fluent surface carries the spiking Broca's own hedge "perhaps the <S> <V-3sg> the <O>"
  (docstrings :17, :1879, :1895) instead of the pre-spiking "_frame_guess" 'Maybe ...' lead (:1837), and
  (b) renderer=None no longer reaches the raw flagged template. Escape is only BRAIN_SPIKING_MOUTH=0 (:1846-1850).
  Extra stale spot: `render_hypothesis` docstring :1787 still promises the 'Maybe <fluent>' frame.
- Chronology (git): test file e35f6740e = 2026-08-12; spiking route 55be6d7ab = 2026-08-12 (same day) — the tests
  encode the pre-spiking contract and have failed on main since the spiking default landed (matches the ruling:
  "two failures on main predate this branch").
- Command (S): `ssh workstation 'bash -lc "... pytest tests/test_rank2_integrated_loop_thread.py::
  test_webapp_integrated_loop_env_default_off ..."'` -> **1 passed in 1.00s** (workstation @ e80251f6d): the
  cloud failure was the missing-fastapi artifact the ruling suspected; real state = green on S. No further action.
- Fix = JUDGMENT (not the driver's): (a) update the two tests + the :1787 docstring to pin the spiking contract,
  or (b) make renderer=None/raw mode also gate the spiking mouth (the tests' original --no-renderer semantics).
  Filed as NEEDS [tests-after-iw2-flip] ASK. (Guess, marked: which side is intended cannot be read from the record
  alone — the spiking mouth's GO is EMERGE-59/61, 6-seed, while the tests were smoke-verified with
  --no-verify: lane-starvation.)
