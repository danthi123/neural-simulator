# objrel spiking read — GENUINELY closed on all 10 seeds by a graded-drive tie-break (`gradedtie`), answer-independent (controller-verified inline; subagent adversarial-verify rate-limited)

**Date:** 2026-07-06
**Runner:** `research/runners/_rungB1c_objrel_reservoir_robustness_sweep_derisk.py` (`--read gradedtie` / `spiking` / `--distinguish`)
**Verdict:** GENUINE answer-independent close of the analytic spiking read on all 10 seeds — the first genuine (non-prior) close in the objrel read-out arc.
**Verification:** CONTROLLER-INLINE (the subagent adversarial-verify Workflow `wwkcuo594` hit the weekly rate limit; the load-bearing checks were run by the controller from the fan-out data + the distinguishing anti-cheat). A full independent subagent re-run is deferred to when the limit resets.
**Supersedes the prior attempts:** `calibrated` (bias-subtraction) + `gainnorm` were both REFUTED as minority-THEME-on-tie priors (`2026-07-06-objrel-tie-break-recovery-real-but-mechanism-is-a-MINORITY-PRIOR-not-calibration.md`).

## The mechanism
The objrel-slot0 failure on 103/104 is a SATURATION TIE: both the AGENT + THEME output pools fire at MAX rate → spike count `[4,0,4]` → argmax defaults AGENT, though the graded output DRIVE genuinely favors THEME (the ridge discriminant, which reads objrel 1.00 on all 10). `gradedtie`: on an exact slot0 count-tie, break by `argmax` of the read-out's OWN graded analog drive `_graded_output_drive = f·IN_SCALE @ W_e + (f·IN_SCALE @ W_fi) @ W_io` — the sub-threshold membrane the spike COUNT quantizes away. No labels, no "THEME" — it follows the actual drive.

## Controller-inline verification (all anti-cheats hold)
1. **RECOVERY:** `gradedtie` objrel-slot0 ≥ 0.90 on ALL 10 seeds (103: 0.00→1.00, 104: 0.17→1.00); canonical ≥ 0.90 (1.00) on all 10; clean seeds unperturbed (tie-break fires only on exact ties).
2. **RAW CAUSAL CONTROL:** `--read spiking` (count-only) still FAILS 103/104 → the graded-drive tie-break is load-bearing.
3. **ANSWER-INDEPENDENCE (the decisive control, `--distinguish`):** synthesize AGENT-favoring slot0 ties (real data has none — canonical slot0 never ties, objrel ties always favor THEME) and check what each mechanism gives. On **seed 103 — the only seed with BOTH synthetic AGENT-favoring (24) AND THEME-favoring (12) ties — `gradedtie` is `answer_independent=True`: AGENT-fav→AGENT 1.0, THEME-fav→THEME 1.0.** RAW (always AGENT), `gainnorm` (always THEME), `calibrated` (always THEME) are all `answer_independent=False`. gradedtie is the ONLY mechanism that follows the actual drive in BOTH directions. (On seeds 44/45/101 the blend sweep produced only AGENT-favoring ties → their `answer_independent` field reads False as a TEST-COVERAGE artifact, NOT a gradedtie failure — gradedtie gives AGENT correctly on all their AGENT-favoring ties.)
4. **CONSTRUCT VALIDITY:** example genuine tie (seed 103): `counts=[4,0,4]` (exact tie), `graded_drive=[0.1262, 0, 0.1240]` (AGENT>THEME), `true_role=AGENT` → gradedtie reads the drive → AGENT (correct); a THEME-prior gives THEME (wrong).
5. **Dale-legal** (the decision consults the analog drive but does not flip weight signs: W_e≥0, W_fi≥0, W_io≤0 untouched); **held-out** (test rng ≠ train rng); **canon not regressed**.

## The honest DIRECTIVE framing + the remaining pieces
- The graded output drive IS a real neural quantity (the output neuron's sub-threshold membrane input); consulting it on a saturation-tie is a legitimate GRADED neural read, distinct from the refuted minority prior (proven by the answer-independence test). The **brain-based-PUREST realization is FIRST-SPIKE LATENCY** (the higher-drive neuron fires EARLIER — a pure spike-TIMING quantity reading the same graded info): confirming latency reproduces gradedtie's answer-independence would make the close fully spike-native rather than a graded-membrane read. **[DEFERRED — subagent rate limit; the confirming refinement.]**
- This closes the ANALYTIC (ideal, ridge-weight) spiking read on all 10. The EMERGENT LEARNED read-out (delta-rule plasticity) has its own separate 45/101 learning fragility; threading `gradedtie` into the learned read-out for the END-TO-END emergent close is the remaining step. **[DEFERRED — subagent rate limit.]**

## Net objrel arc status (this session)
Reservoir encodes objrel on all 10 (ridge 1.00) · read-out plasticity genuinely learns it emergently on clean seeds (BPTT 0/6) · the 2/10 spike-count-tie failure is now GENUINELY closed by an answer-independent graded-drive tie-break (`gradedtie`) · 5+ overclaims caught + corrected by the adversarial-verify discipline along the way. Remaining: latency-purity confirmation + threading into the learned read-out for the end-to-end emergent close.

## Files
- `research/runners/_rungB1c_objrel_reservoir_robustness_sweep_derisk.py` (`--read gradedtie`); `research/findings/raw/_gt10_s*.json` (10-seed recovery), `_dist_s*.json` (distinguishing test), `_raw10_s*.json` (RAW causal control).
