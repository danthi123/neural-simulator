# Direction Q-tertiary adversarial reviewer verdict

**Reviewer:** Fresh adversarial subagent (independent of runner author).
**Date:** 2026-05-26.
**Commit under review:** `e94017e`.
**Result file:** `research/findings/raw/direction_Q_tertiary_nmda_ratio_sweep.json`.
**Final adjudication: CLEAR — pillar n=107 BOUNDARY/PASS approved.**

## Item-by-item adjudication

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | No protected/frozen module modified | PASS | `git diff HEAD~1 HEAD -- direction_Q_verdict.py direction_Q_protocol.py` returns no output. `git status` confirms both files clean. Both frozen at commits c93495f / 957ac51 (Tasks 2-3 pre-registration). |
| 2 | Bridge-builder modification is default-preserving | PASS | Diff adds ONE kwarg `nmda_ratio: float = 0.4`. CoreSimConfig default is already `0.4`, so `cfg.nmda_ratio = 0.4` is semantically a no-op when the parameter is omitted. |
| 3 | nmda_ratio=0.4 cell reproduces prior Q-secondary inh=2.0 result | PASS (with caveat) | Seeds 42 & 43: byte-identical ratio AND sustained_sec to Q-secondary. Seed 44: 0.06% drift in mean_delay_rate (23.408 → 23.394 Hz), identical sustained_sec (0.95s), identical verdict (PARTIAL). Drift consistent with CUDA non-determinism in NMDA-kernel atomic summation; well below any threshold. |
| 4 | nmda_ratio=0.6 PASS is genuine (multi-seed, both bars) | PASS | All 3 test seeds: ratio ≥ 697.4 (350× the 2.0 bar); sustained_sec = 3.000 (60/60 bins above 2× baseline threshold). |
| 5 | nmda_ratio=0.8 PASS is genuine (multi-seed, both bars) | PASS | All 3 test seeds: ratio ≥ 837.7 (420× the bar); sustained_sec = 3.000 (60/60 bins). |
| 6 | NMDA-off control correctly silent (no false-positive on controls) | PASS | All 3 control seeds across ALL 3 cells: ratio in [0.92, 1.06], sustained = 0.0s. Verdict module's control-gate would VOID any cell with control passing — none did. |
| 7 | No score-tuning / threshold tampering / hard-coded fallback | PASS | Grep for tuning patterns returns no matches. Runner imports thresholds from frozen module; reports them inline in JSON for traceability. |
| 8 | sustained_sec metric computation is honest | PASS | Independently reproduced formula `n_bins(rate ≥ 2 × baseline) × bin_ms/1000` for all 9 cells/seeds — exact match. |
| 9 | Verdict independently reproduces from raw JSON via frozen module | PASS | Loaded `direction_Q_verdict.compute_verdict` directly, fed raw `per_seed_tuples`/`control_per_seed_tuples`: PARTIAL/PASS/PASS independently == runner output. |
| 10 | Final-3 delay bin rates are stable plateau (not transient peak) | PASS | nmda_ratio=0.6 final-3 (Hz): seed 42 [651, 654, 649]; seed 43 [649, 648, 646]; seed 44 [717, 713, 713]. nmda_ratio=0.8 final-3: ~951-967 Hz steady across all seeds. Mid-delay bins match final bins within ~5% (no decay). |
| 11 | sustained=3.00s is measurement-window cap, NOT artificial clamp | PASS | `delay_ms=3000`, `bin_ms=50` → max 60 bins → max possible sustained = 3.00s. ALL 60 bins exceed threshold for every PASS seed (n_elev = 60/60). Final-bin rates are 300-450× above the 2× baseline bar, not marginal. This is "stable through entire delay window," exactly the intended semantics. The 3-second observation window is the floor on the actual sustained duration, not a ceiling on a marginal pass. |
| 12 | Findings doc exists and matches result | PASS | `2026-05-26-DIRECTION-Q-tertiary-NMDA-AMPA-ratio-sweep-PASS-Wang-2002-bistability-closed-at-nmda-ratio-0.6.md` present. |

**Per-item tally: 12 PASS / 0 FAIL.**

## Final verdict: CLEAR

Direction Q (Wang 2002 NMDA bistability at biological scale, dlpfc_wm n=1000 d=0.20 inh=2.0) is **closed** by raising `cfg.nmda_ratio` from the CoreSimConfig default (0.4) to 0.6 or 0.8. Multi-seed PASS at 0.6 and 0.8 with NMDA-off controls remaining silent confirms the persistence is NMDA-mediated, not a substrate-noise artifact. The measurement-window-equals-delay-period concern raised by the prompt is fully addressed: it is correct semantics ("stable through entire window"), and the rates are 300-450× above the bar, not marginal.

**Promote pillar n=107: Wang 2002 cortical NMDA bistability at substrate scale via NMDA:AMPA ratio fix.**

Concerns / followups (non-blocking):
- Seed-44 0.06% mean-delay drift between Q-secondary and Q-tertiary nmda_ratio=0.4 reads suggests the NMDA kernel summation order may not be fully deterministic on CUDA even with CUBLAS_WORKSPACE_CONFIG. Not a verdict concern at this scale, but worth a determinism-audit ticket eventually.
- The actual sustained duration at nmda_ratio ≥ 0.6 is unbounded by this experiment — extending `delay_ms` to 10s+ in a follow-up would characterize the upper limit (already an honest extension, not a re-tuning).
