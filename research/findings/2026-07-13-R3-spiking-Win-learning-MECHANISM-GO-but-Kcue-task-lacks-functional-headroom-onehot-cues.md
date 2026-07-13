# R3 spiking-W_in-learning: the on-bridge BDSP MECHANISM is validated GO, but the K-cue distal-decode DE-RISK TASK does NOT test the functional "learn W_in beats fixed W_in" claim — a fixed random W_in already solves it (one-hot cues are maximally separable; the RATE reference confirms learn≈fixed). The functional test needs STRUCTURED input (the language regime), not arbitrary orthogonal symbols

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_onbridge_learn_win_derisk.py` (committed 69b56c09; the R3 spiking realization — learn W_in on a FROZEN spiking Izhikevich reservoir via the committed `enable_bdsp` rule on a plastic input→reservoir pathway; NO `sim/` edit). Dist sweep `raw/_r3_distsweep.log`.
**Status:** MECHANISM GO + a METHODOLOGY NEGATIVE on the task — the cheap K-cue de-risk cannot evaluate the R3 functional claim; the diagnosis names the valid task.

## The mechanism is validated (all controls GO, every operating point)
On the K-cue distal-decode (`[CUE_k] filler×d [QUERY]`, K=12, n_pool=200), the committed BDSP rule learns the input→reservoir W_in correctly:
- **W_in moves, W_rec frozen:** `dw_win` 0.014→0.135 (grows with dist/eligibility) while `dw_rec = 0.000000` at every dist (the frozen recurrence is untouched — verified by the COO pathway masks against `sim/bridge.py:7246-7273`).
- **Directed credit:** `learn_win` moves W_in ~2–5× the `apical_lesion` (apical=0) arm; `B_rises` True (apical raises the measured burst rate → directed credit); `wrong_sign` mirrors `learn_win`'s dw (anti-symmetric credit).
- **Anti-cheats collapse:** `input_lesion` → 0.083 (chance); `scramble` → 0.000; `no_weight_transport` True (Y own RandomState, never modified).
⇒ the R3 spiking realization — "the committed dendritic burst rule learns the reservoir's INPUT projection on spikes, no `sim/` edit, no weight transport" — WORKS as a mechanism.

## The functional claim FAILS on this task — and the RATE reference proves it is the TASK, not spiking
Dist sweep (seed 42, K=12, n_pool=200), decode acc (chance 0.083):
| dist | fixed_win (spiking) | learn_win (spiking) | RATE-ref fixed | RATE-ref learn |
|---|---|---|---|---|
| 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 1.000 | 1.000 | — | — |
| 16 | 1.000 | 1.000 | — | — |
| 24 | 1.000 | 1.000 | 0.917 | 0.917 |
| 32 | 1.000 | 1.000 | 0.833 | **0.667** |
- **A fixed random W_in decodes all 12 cues perfectly to dist=32** — the reservoir HOLDS the cue that far AND a random projection separates it. There is NO collision → NO headroom for learning W_in.
- **The RATE reference (the R3 mechanism at its full-gradient best) shows learn ≈ fixed** (0.917=0.917 at dist=24) and learn WORSE than fixed at dist=32 (0.667<0.833) — i.e. **the task has no "learn W_in beats fixed W_in" property AT ALL**, independent of the spiking substrate.

## Root cause (systematic-debugging): one-hot cues have NOTHING to learn
The cues are arbitrary ORTHOGONAL one-hot symbols → they are already maximally separable, so a LEARNED input projection cannot beat a RANDOM one (both separate orthogonal inputs equally; learning only adds instability, hence learn<fixed at dist=32). The R3 reframe's "learn W_in is worth ~3× learning W_rec / beats full BPTT" was measured on the **LANGUAGE next-token task** (TinyStories V=2000), where the input tokens have **distributional/semantic STRUCTURE** that a random projection scrambles but a learned embedding exploits. The K-cue task strips exactly that structure out. ⇒ the scoping doc's assumption ("K large enough that a fixed-random W_in COLLIDES the cues") does not hold with orthogonal one-hot cues + an over-provisioned reservoir; a random projection separates ≤~n_pool orthogonal cues regardless of distance.

## ⇒ The redirect (the valid functional test)
Testing the R3 W_in-learning claim on spikes needs a task with **input-representation headroom** — where a fixed random W_in genuinely underperforms a learned one:
1. **STRUCTURED cue codes** (the cheap faithful fix): cues = OVERLAPPING/correlated codes (shared input features), so a random W_in scrambles the exploitable structure but a learned W_in maps them to a separable subspace. Validate the regime with the RATE reference FIRST (does rate-learn > rate-fixed?), then run the spiking arm only where the regime is proven valid.
2. **The LANGUAGE task on spikes** (the real R3 regime): learn W_in on the spiking reservoir, by-depth next-token CE, vs fixed W_in — expensive (the fork base's on-bridge W_rec LM run boundaried at 2/6), but the faithful regime.
The RATE reference is the load-bearing GATE for task validity: only run the expensive spiking arm at an operating point where the rate reference SHOWS the R3 property (rate-learn > rate-fixed). NO `sim/` edit.

## UPDATE — the cheap RATE PROBE confirms the diagnosis + measures the (modest) structured headroom
`research/runners/_r3_structured_input_rate_probe.py` (numpy, self-contained): a FIXED ESN reservoir, distal-decode, cue codes ORTHOGONAL (one-hot) vs STRUCTURED (overlapping — each cue = a sum of a small shared random-atom pool), fixed W_in (read-out only) vs learn W_in (BPTT-frozen-W_rec = the R3 ceiling). Seed 42:
| regime | fixed_win | learn_win | margin |
|---|---|---|---|
| K=16, d=6 (easy) — orthogonal | 1.000 | 1.000 | +0.000 |
| K=16, d=6 (easy) — structured | 1.000 | 1.000 | +0.000 |
| **K=48, d=12, n_pool=64 (collision) — orthogonal** | **1.000** | **1.000** | **+0.000** |
| **K=48, d=12, n_pool=64 (collision) — structured** (8 atoms) | 0.688 | 0.750 | **+0.062** |
| structured, 6 atoms (heavier overlap) | 0.333 | 0.396 | +0.062 |
| structured, 5 atoms | 0.208 | 0.250 | +0.042 |
| structured, 4 atoms (near-degenerate) | 0.042 | 0.083 | +0.042 |
- **ORTHOGONAL codes: learn NEVER beats fixed (margin 0.000 at every difficulty)** — a random W_in separates orthogonal cues to the reservoir's capacity, so there is nothing to learn (the confirmed root cause).
- **STRUCTURED (overlapping) codes: fixed COLLIDES (0.69→0.04 as overlap grows) and learn RECOVERS a REAL but MODEST +0.04–0.06** — structure-specific, so input structure IS the R3 headroom, but the magnitude on distal-decode is small (a bounded code-de-mixing gain), NOT the language task's +4 nats. The margin plateaus ~+0.06 across overlap levels — a genuine ceiling of this task class, not under-training.
- ⇒ **the distal-decode task is a WEAK R3 regime**: learning W_in helps only modestly (de-mix overlapping codes), because the task's learnable input structure is thin. A +0.06 RATE ceiling means the noisier SPIKING version would very likely NOT clear a robust +0.10 gate → a predictable BOUNDARY on the cheap proxy.

## ⇒ Verdict + the faithful next test
The R3 spiking-W_in-learning **mechanism is validated GO**; the **cheap distal-decode proxy cannot demonstrate a strong functional win** (orthogonal = no headroom; structured = only +0.06, below a robust spiking gate). The **faithful strong test is the LANGUAGE task on spikes** — learn W_in on the fixed spiking reservoir on real text, by-depth next-token CE, vs fixed W_in (the regime where the R3 rate result is +4 nats). It is expensive (the fork base's on-bridge W_rec LM run boundaried at 2/6; the W_in version is the R3-STABLE lever, better-odds but untested on spikes) — the next major de-risk, pivoting `_emerge_reservoir_lm_onbridge_bdsp_derisk.py` to a plastic input→reservoir pathway (frozen recurrence) on the by-depth LM metric.

## ⚠️ CORROBORATION + THE ARC'S ACTUAL LATEST STATE (a-1 caught this AFTER the re-derivation — the lesson: dig to the LATEST arc finding, not just the scoping doc)
This finding (2026-07-13) INDEPENDENTLY RE-DERIVED the early conclusions our own record ALREADY reached on **2026-07-12** — a belated a-1 (`--corpus finding`, query "cue task right instrument") surfaced them. The 2026-07-12 arc is thoroughly mapped + CONVERGED, well beyond this re-derivation:
- **`2026-07-12-cue-task-is-wrong-instrument-*`:** the K-cue task is the wrong instrument (Johnson-Lindenstrauss — a random projection preserves distinct-input distinctness, so cue CLASSIFICATION never needs a learned W_in). The R3 benefit is next-token PREDICTION + distributional GENERALIZATION. The **correct instrument** (class-structured next-token task: shared class dims + class-irrelevant identity confound + held-out synonyms) shows a **STRONG rate headroom: learn 0.900 vs fixed 0.322 (+0.578), 6-seed.** (My structured-code probe found the SAME direction — orthogonal 0.0, structured +0.06 — but weaker, because "which cue" is still classification, not held-out generalization; the 2026-07-12 instrument adds the held-out-synonym generalization the +0.578 needs.)
- **`2026-07-12-spiking-learn-Win-...BDSP-coarseness-boundary`:** the SPIKING version of the correct instrument was built → the generalization benefit does NOT transfer (the spiking BDSP credit is too coarse to learn the confound-suppressing input representation).
- **`2026-07-12-dendritic-per-compartment-gain-SURPASSES-*`:** a DEVELOPMENTAL dendritic per-compartment divisive gain (frequency-adapted, PPMI-family) SURPASSES the point-neuron confound boundary — MODEST + operating-point-sensitive + a binary-code artifact (6/6 direction, ~0.35 magnitude); compose-with-learn-W_in REFUTED. **⇒ THE CONVERGENCE: on the spiking substrate, confound-suppression for generalization is a DEVELOPMENTAL local-normalization capability (dendritic gain, already PPMI-delivered on real count-corpus), NOT a credit-learned one (BDSP too coarse) NOR cross-neuron decorrelation (red herring).**
⇒ the R3 spiking-W_in arc is CONVERGED; the only genuinely-open rung is D2 (the fully-spiking two-compartment online-adapted dendritic gain, `enable_two_compartment_dap`) — a purity/completeness build (the capability is PPMI-delivered), not a capability unlock. This 2026-07-13 finding stands as an independent corroboration of the instrument diagnosis, NOT a new frontier.

## Files
`_reslm_onbridge_learn_win_derisk.py`; `_r3_structured_input_rate_probe.py`; `raw/_r3_distsweep.log`. Corroborates `2026-07-12-cue-task-is-wrong-instrument-*` + the dendritic-surpass convergence. Ties to `2026-07-11-R3-REFRAME-*` (the language regime where W_in-learning DOES win, +4 nats).
