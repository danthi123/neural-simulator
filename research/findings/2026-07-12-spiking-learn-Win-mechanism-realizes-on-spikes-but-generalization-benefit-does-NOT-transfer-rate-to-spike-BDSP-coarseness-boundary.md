# The R3 spiking learn-W_in realizes the MECHANISM cleanly on spikes, but does NOT transfer the rate generalization benefit — a like-for-like-verified rate→spike BDSP credit-coarseness boundary on confound-suppression

**Date:** 2026-07-12
**Status:** 🟧 HONEST BOUNDARY (like-for-like verified) — the committed on-bridge BDSP rule LEARNS the input projection W_in on spikes cleanly (mechanism confirmed) but does NOT reproduce the rate version's held-out generalization benefit. First-class negative: it maps exactly where the substrate can/can't do the R3 mechanism. Reuse-by-import; NO `sim/` edit.
**Frontier:** the R3-reframe long-range/generalization path — *fixed spiking reservoir + e-prop/BDSP-learned INPUT projection W_in + local read-out*. Prior: `2026-07-11-R3-REFRAME-...md` (rate, validated ~78% of ceiling), `2026-07-12-cue-task-is-wrong-instrument-...md` (the correct instrument = prediction+generalization, strong rate headroom).

## What runs cleanly on spikes (the mechanism)

`research/runners/_reslm_onbridge_generalize_derisk.py` (reuse `WinLearnReservoir` + `_run_arm` + the committed `enable_bdsp`; `GenReservoir` drives a multi-hot code): on ONE `SimulationBridge`, the committed BDSP rule (apical = k·(Y@δ), fixed-random Y) MOVES the plastic input→reservoir W_in — `dw_win` ~1.0–1.4, `dw_rec ≡ 0` (recurrence frozen), `no_weight_transport True`, B rises under apical drive (the D1 coupling). The anti-cheats hold (input-lesion → chance; label-scramble → chance). **The learning mechanism realizes on the spiking substrate** — this half is a GO.

**Operating point (read the substrate).** The EMERGE-82 reservoir is E/I-balanced-sparse by design (`_INH_W 8.0 > _EXC_W 6.0`, "keeps the pool from saturating"). A broad multi-hot code recruits the reservoir's inhibitory neurons, so it needs STRONG focused drive to spike during the clean read; at `in_hi≈3000, res_bias≈120` it reads richly (0.17–0.21 spikes/step) and the confound-FREE task reads well (fixed 0.8; a minimal 6-code task 0.833). Weak drive (`in_hi=700`) leaves the clean read silent → chance; scale (n=120→500) and E/I overrides did not substitute for drive.

## The boundary (the functional generalization benefit does NOT transfer)

At configs where the RATE version shows a strong learn-vs-fixed generalization benefit, the SPIKING `learn_win` == `fixed_win` (margin 0), across every knob swept:

| config (sf=3, id_pool=60) | RATE learn / fixed / margin | SPIKING learn / fixed / margin |
|---|---|---|
| n=60, idn=30 | 0.867 / 0.389 / **+0.478** | 0.30 / 0.30 / **+0.00** |
| n=100, idn=30 | 0.756 / 0.533 / **+0.222** | 0.133 / 0.133 / **+0.00** |
| n=60, idn=20 | 0.767 / 0.433 / +0.333 | (learn==fixed) |

And across confound strength (idn 3→30: spiking learn==fixed at every step — idn=3 both 0.8, idn=8 both 0.367, idn=30 both 0.30), training budget (epochs 3→20, bdsp_lr 0.02→0.05, dw_win 1.0→1.4), and scale (n 60→500). **The learned W_in moves substantially but does NOT suppress the identity confound**, so it does not recover the held-out generalization the rate version achieves.

## Methodology catch (the like-for-like discipline earned its keep)

The first spiking tests were at sf=3/idn=8/n=120 — where I nearly wrote "the spiking BDSP fails to transfer the rate benefit." The load-bearing check: **at that exact config the RATE version ALSO has no headroom** (learn 0.867 == fixed 0.867, margin +0.000) — the confound is too weak / the reservoir too big for it to bite. There was nothing to transfer. Re-scanning for configs where the rate benefit genuinely exists (small n + strong confound) and re-testing spiking THERE gives the valid boundary above. Without the like-for-like check the boundary claim would have been a confound-config artifact. (The skill's COMPARE-LIKE-FOR-LIKE step, applied to itself.)

## Root cause + why it's a boundary, not a wall

The rate confound-suppression is a fine, per-synapse statistical-averaging effect: the class dims' e-prop updates accumulate (consistent sign vs the class error) while the identity dims' average out (random vs class) → learned W_in up-weights class, suppresses identity. The committed spiking credit — burst-multiplexed feedback-alignment via a BINARY burst detector (apical = k·(Y@δ)) — is too COARSE to realize that fine per-dim discrimination: it moves W_in in bulk but cannot selectively down-weight the confound dims. This is the same rate→spike coarseness the prior on-bridge BDSP arc documented (AUTONOMOUS_STATE: "burst-multiplexed FA credit is COARSER than the rate RFLO; the rate e-prop's within-reach recovery doesn't robustly transfer to spikes at this scale").

**3-seed confirmation** (n=60, idn=30, id_pool=60, in_hi=3000, res_bias=120, epochs=10; rate benefit at this config = +0.478): learn **0.222** vs fixed **0.200**, margin **+0.022** (noise-level) — per-seed 0.3/0.3, 0.167/0.1, 0.2/0.2; `dw_win` 1.28–1.34, `dw_rec ≡ 0` all seeds, `apical_lesion == learn` and `wrong_sign == learn` (the apical credit isn't load-bearing for a benefit that isn't there), input-lesion → chance, no weight transport. Anti-cheats clean; the benefit simply does not transfer.

## The credit-vehicle lever (M2.6 graded clean-error) — BUILT + TESTED → RULED OUT (the boundary is deeper)

The a-1 RAG step named the graded clean-error credit as the most-likely fix; I built it and tested it. **It does NOT close the boundary.** Added an additive/default-off/byte-identical `sim/` flag `enable_bdsp_graded_credit` (`sim/config.py` + the 11-line guarded `sim/bridge.py` kernel edit): the committed FF rule credits with the noisy MEASURED burst `B − P̄·E`; the flag swaps it for the graded expectation `E·(P − P̄)` (the kernel's own identity `B − P̄·E == E·(P − P̄)`; P is the smooth burst probability, B its stochastic sample) — the on-bridge M2.6 low-variance credit. **3-seed at the boundary config (rate benefit +0.478): graded learn 0.222 vs fixed 0.211 (margin +0.011) — statistically identical to the measured-B result (learn 0.222 vs fixed 0.200, +0.022).** So the confound-suppression failure is **NOT credit-vehicle-noise-bound** — a clean ruling-out of the leading hypothesis. (40 determinism/kernel tests pass; the default path is byte-identical by construction — `_B_post` IS `cp_bdsp_B` when off.)

**⇒ The boundary is DEEPER: read-out/delta-consistency on the sparse spiking substrate.** The rate confound-suppression needs a CONSISTENT per-class read-out error `δ` over many examples (so class dims' W_in updates accumulate while identity dims' average out). The credit `k·(Y@δ)` is only as clean as `δ`, and `δ` comes from the read-out on the sparse spike-count reservoir read — which is noisy enough that the per-class error isn't consistent across examples, so the statistical W_in organization never accumulates. Making the CREDIT graded (this test) doesn't help because the noise enters at the READ-OUT δ, upstream of the credit vehicle. Thoroughly characterized: the boundary survives scale (n 60→500), drive/operating-point, read-window (t_step), E/I excitability, AND credit-vehicle (graded M2.6) levers.

## Deeper next lever (a fresh research-gated arc — do NOT force a positive)
The binding limit is the sparse spiking read-out δ consistency, not the credit rule. Candidate directions (research-gate before building): (1) a richer/population read-out that yields a consistent per-class δ (population coding — the documented rate-code-wall lift); (2) a graded/membrane read of the reservoir (vs the sparse spike count) for δ; (3) the dendritic input-representation frontier (the owner's standing priority) where the input representation is learned with a substrate that can carry a clean continuous error. The **rate mechanism stands validated** (`2026-07-11-R3-REFRAME`, ~78%); the spiking residual is now precisely located at the read-out δ, and the M2.6 credit-vehicle hypothesis is ruled out.

## (Appendix) the a-1-surfaced candidate that this rung tested — for the trail

The skill's new (a-1) "check our own findings first" step surfaced the exact lever — no external gate needed: **the M2.6 CLEAN-ERROR credit** (`2026-07-07-D1-microcircuit-noise-robust-deep-credit-clears-bar-on-spikes.md`). That finding shows the committed BDSP's burst-fraction credit `b = B − P̄·E` is batch/noise-FRAGILE (held-out 0.788 at batch 128, collapsing to 0.51) while a **low-variance clean-error feedback-alignment credit** `e = φ′(E)·(Yᵀ@e_upper)` (Urbanczik-Senn M2.6 somatic-rate rule, no weight transport) is batch-ROBUST and clears **0.964**. That is *exactly my boundary's failure mode*: my runner's credit vehicle is the noisy `(B − P̄·E)` burst read, and confound-suppression needs precisely the low-variance per-synapse signal the clean-error credit provides. Next rung:
1. **Swap the credit vehicle to the clean-error channel** on the input→reservoir W_in update — read the apical credit as the graded clean error (`enable_bdsp_microcircuit`'s interneuron-cancellation returns the apical to the clean residual) instead of the binary-ish burst-fraction detector.
2. **HONEST CAVEAT (from that finding, read in depth):** the M2.6 clean-error result is a NUMPY REFERENCE (depth-2 MLP); on-bridge the interneuron cancellation is validated for the burst READOUT, but the M2.6 FF *weight update* lives runner-side and its fully-on-bridge realization is undemonstrated. Realizing the clean-error credit for the on-bridge W_in update likely needs a small additive `sim/` change to the BDSP credit read (graded vs binary) — a research-gated `sim/` rung (additive/default-off/guarded), legitimate because a faithful low-variance dendritic credit IS the mechanism, not a cheat.
This is the genuine **rate-RFLO-vs-spiking-BDSP credit-precision gap**; the rate mechanism is validated (`2026-07-11-R3-REFRAME`), and our own record already names the credit rule that closes it. Do NOT force a positive; the clean-error swap is the next single-variable de-risk.

## Verdict
The R3 learn-W_in **mechanism** realizes on spikes (moves W_in cleanly, no weight transport, dw_rec≡0); the **functional generalization benefit** does not transfer via the committed BDSP rule at this scale — a precisely-located, like-for-like-verified rate→spike credit-coarseness boundary. The rate path (`2026-07-11-R3-REFRAME`) stands validated; the spiking realization's honest residual is the credit precision.

## Files
- `research/runners/_reslm_onbridge_generalize_derisk.py` (spiking runner; `--t-step`/`--exc-w`/`--inh-w` operating-point levers).
- `raw/_reslm_gen_boundary_n60_3seed.json` (the 3-seed confirmation), `raw/_genR_n{60,100}.json` (the like-for-like configs).
