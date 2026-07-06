# objrel spike-count-tie recovery — REAL (objrel-slot0 recovers on all 10, load-bearing, anti-cheats hold) BUT the mechanism is a MINORITY-THEME-on-tie PRIOR, not a genuine per-pool calibration (adversarial-verify caught the overclaim). The genuine answer-independent fix (per-pool GAIN normalization / graded-drive tie-break) is the next lever.

**Date:** 2026-07-06
**Method:** adversarial-verify Workflow `wgxmgy82f` (3 skeptics + synthesizer, all running their own controls) — 3/3 refute the mechanism-framing half.
**Verdict:** GO on RECOVERY, RETRACT the "calibration" mechanism framing (the Nth self-caught overclaim this session; the discipline working).
**Builds on:** `2026-07-06-objrel-CORRECTION-no-reservoir-problem-residual-is-ALL-readout.md` (the ridge reads objrel 1.00 on all 10; the spiking read fails on 103/104 at a slot0 count-tie).

## What is REAL (confirmed by the adversarial-verify)
`recovery_is_genuine=True, anticheats_hold=True`: the 2-stage tie-gated spiking read recovers objrel-slot0 to ≥0.90 on ALL 10 seeds (103: 0.00→1.00, 104: 0.17→1.00), and every anti-cheat holds — RAW `_score_spiking` still causally FAILS 103/104 (load-bearing on exactly those 2 seeds); TASK-BLIND (the reference is the class-balanced mean of TRAIN features + train labels only, bias0 byte-identical across test draws → no test-label peek); CANONICAL not regressed (1.00 raw AND calibrated, canon has 0 slot0 ties); Dale-legal (a per-pool scalar subtraction on the output-LIF spike-COUNT vector, weights + signs untouched); held-out (train rng ≠ test rng). **objrel-slot0 IS recoverable on all 10 seeds.**

## What was OVERCLAIMED and is now RETRACTED (mechanism = `equivalent_to_minority_prior`)
The builder framed the tie-break as a "genuine per-pool-GAIN calibration (measured, answer-independent, SEED-CONSISTENT offset), NOT a minority prior." The adversarial-verify refuted that on 4 independent controls:
1. **A hard "pick THEME on any slot0 AGENT-vs-THEME tie" (ZERO measured content) reproduces the calibrated result BYTE-IDENTICALLY on every real tie — and is STRICTLY BETTER** (it also recovers seed 45, which the measured bias leaves at 0.92 because there b_AGENT==b_THEME so the tie survives). The measured magnitudes do NO work the hard prior doesn't.
2. **Any random per-pool vector whose only property is "THEME component smallest" recovers 1.00/1.00 on every seed × 10 draws** — only the ORDERING (THEME favored on tie), not the measured gain, matters.
3. **The measured bias is NOT seed-consistent:** b_AGENT−b_THEME ∈ {−1,0,+1,+2} across seeds (on seed 42 AGENT UNDER-responds). Falsifies "seed-consistent per-pool gain."
4. **Applying the bias UNCONDITIONALLY (not tie-gated) DESTROYS clean seeds** (42/46/102 → 0.00) → the measured magnitudes carry no useful away-from-tie information; only the tie-gate + THEME-favoring ordering is load-bearing.
Plus: the distinction is **structurally UNTESTABLE on this corpus** — canonical (true-AGENT) slot0 produces 0 ties; every failing objrel tie is true-THEME. So "measured calibration" and "THEME-on-tie prior" make identical predictions here.

⇒ The demonstrated mechanism is a **minority-THEME-on-tie prior** (biologically defensible as a novelty-salience / rare-reading Bayesian bias — Schultz novelty-DA — but NOT a per-pool gain calibration), and on THIS corpus it is task-specific (it encodes "objrel-slot0 = THEME"). It is NOT a genuine, answer-independent close.

## The ROOT of the tie (the real mechanism to fix) + the genuine next lever
The 103/104 failure is a **SATURATION tie**: at the graded op-point both the AGENT and THEME output pools are driven above threshold and fire at the SAME (max) rate → count `[4,0,4]` → argmax defaults to AGENT — even though the graded output DRIVE genuinely favors THEME (ridge membrane [0.25 AGENT, 0.75 THEME]; the reservoir encodes it, ridge reads it). More spike-count resolution (READ_T 4×) does NOT help (both saturate); a uniform gain drop breaks canonical. The GENUINE, answer-independent fixes (untested — the next lever):
1. **Per-pool GAIN normalization (the RIGHT Turrigiano homeostatic mechanism):** normalize each output pool's GAIN so it operates in its LINEAR (non-saturated) range → the spike RATE tracks the graded drive → THEME's higher drive fires MORE → the tie breaks toward THEME on objrel AND toward AGENT on canonical (answer-independent). Distinct from the refuted bias-SUBTRACTION.
2. **Graded-drive / sub-threshold-membrane tie-break:** on a spike-count tie, break it by the graded output drive (a real, more-biological neural quantity the pure count quantizes away) — answer-independent (gives whichever role the drive favors).
These are genuinely testable (they must give AGENT on a constructed canonical slot0 tie, THEME on objrel) — unlike the minority prior. Test them + adversarially verify (does a hard THEME-on-tie prior STILL match? if the gain-norm gives AGENT on a canonical tie where the prior gives THEME, it is genuinely distinguishable).

## Honest scope
objrel-slot0 is recoverable on all 10 seeds (the info is present; the ridge reads it), and a novelty-salience minority-on-tie prior demonstrably recovers it — but that prior is task-specific on this corpus and is NOT a genuine answer-independent read-out mechanism. The objrel spiking read is NOT yet genuinely closed; the genuine close needs the per-pool gain normalization / graded-drive tie-break (answer-independent), which is the next lever. NO overclaim.

## Files
- `research/runners/_rungB1c_objrel_reservoir_robustness_sweep_derisk.py` (`--read calibrated`/`spiking`/`ridge`), `research/findings/raw/_cal10_s*.json` + `_raw10_s*.json` (10-seed calibrated + raw regression), adversarial-verify `wgxmgy82f`.
