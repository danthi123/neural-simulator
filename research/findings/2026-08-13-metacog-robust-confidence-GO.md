---
type: finding
status: live
date: 2026-08-13
mechanism: metacog-confidence-read
integration_faculty: onebrain-merge-organs-pool2
---

# Metacog robust confidence (GO): a DIVISIVE-NORMALIZED NMDA-CONDUCTANCE balance read makes the confident/uncertain decision INVARIANT to the pool-#2 per-region init re-draw — metacog answer-preservation 1/6 → 6/6, the pool-#2 production-default flip is UNBLOCKED (conditional on adopting the read)

**Date:** 2026-08-13 · **Build:** `research/runners/_metacog_robust_confidence_derisk.py`
(`RobustMetacogProductionOrgan`, a `MetacogProductionOrgan` subclass that only replaces the confidence MARGIN) ·
**Re-runs:** the pool-#2 answer-preservation A/B (`_onebrain_production_flip2_verify`) with the metacog read
swappable · **Artifact:** `research/findings/raw/_metacog_robust_confidence_6seed.json` (6 seeds 42/43/44/100/101/102,
`SIM_BACKEND=numpy`) · **Unblocks:** `2026-08-13-onebrain-second-pool-SCOPED.md` (the withheld flip).

## The blocker

Pool #2 (`onebrain_merge_production2.py`) merges the metacog + pragmatic production organs onto ONE shared spiking
bridge BYTE-IDENTICALLY (merge GO 6/6; `per_region_wiring_seed` exercised). Pragmatic is answer-preserving (6/6).
Metacog is NOT (1/6): the shared pool REQUIRES the three region-scoped seams (`per_region_parameter_heterogeneity` /
`per_region_threshold_heterogeneity` / `per_region_wiring_seed`), which re-draw the workspace's per-neuron Izhikevich
params + firing thresholds name-keyed instead of from the global-RNG order, and metacog's balance-of-evidence
confidence — the ABSOLUTE spike-rate margin `|rate(asm1)-rate(asm0)|` off `cp_firing_states` — shifts enough under that
re-draw to flip the `confident`/`uncertain` decision at mid-range evidence. So the production-default flip was withheld,
naming as its fix "a robust / self-calibrating confidence read invariant to the re-draw."

## The real root cause (measured, deeper than "narrow dynamic range")

The FIRST lever — divisive-normalizing the SPIKE-rate margin by the summed co-active rate `|r1-r0|/(r1+r0)` — FAILED
(still 0/1; see `_metacog_robust_confidence_derisk --read balance` control). The flattening is not a pure gain change a
ratio could cancel: the spike-count margin sits at the NOISE FLOOR. The workspace assemblies fire at ~0.1% at this
operating point, so `|rate(asm1)-rate(asm0)|` is a difference of ~0.1 spikes — even the STANDALONE build's
margin(evidence) is only ~0.5 monotone (near-random). The mid-range `confident` call is reading noise, and the
per-region re-draw reshuffles that noise → the flip. There is no signal in the spike counts to normalize. (This matches
the prior `2026-08-02-laneC-metacog-margin-comparator-PARTIAL…` finding, which already attributed its stress failures to
"relay spike-count / settling noise".)

## The fix — divisive normalization of the NMDA conductance

Read the balance off the assemblies' slow-NMDA recurrent conductance (`cp_conductance_g_nmda`) — the GRADED synaptic
accumulator the metacog faculty was explicitly designed around ("slow NMDA lets meta INTEGRATE the settled
balance-of-evidence"; Wang persistent NMDA) — instead of the coarse spike count:

    conf_norm(evidence) = |g_nmda(asm1) − g_nmda(asm0)| / (g_nmda(asm1) + g_nmda(asm0) + eps)

Both terms come from the SAME substrate: the NMDA conductance is driven by presynaptic spikes through NMDA synapses (a
genuine spiking-substrate state, NOT the injected current — the lesion that removes the evidence differential collapses
it), and slow-NMDA integrates the sparse spikes into a SMOOTH graded signal with real SNR. This is a Carandini & Heeger
DIVISIVE NORMALIZATION off conductances — exactly the anti-cheat's sanctioned form ("divisive-norm off
`cp_firing_states`/CONDUCTANCES") — NOT a host rescale of the answer: numerator = the two competing accumulators'
balance, denominator = their summed co-active NMDA drive (the normalization pool). The confident/uncertain threshold
self-calibrates on this margin exactly as before (the same synthetic hi/lo-evidence battery in the gap).

Measured: the NMDA margin tracks evidence monotonically in BOTH the standalone and the merged build, with the today and
merged margin curves nearly overlapping across the sweep, where the spike-rate margin is near-random (monotonicity
~0.5). The self-calibrated threshold therefore lands at the SAME evidence boundary (mid-range) in both builds, and
the decision is invariant to the re-draw.

## Result — GO 6/6 (up from 1/6)

Every read is through the real organ APIs: metacog `judge(evidence)` over the pool-#2 sweep MC_EVID×8, pragmatic
`interpret(utterance)` over {none, some, all}. TODAY = each organ on its own bridge (== flag-off production); MERGED =
both organs on ONE shared bridge; CORESIDENT = each organ on its own bridge with the three merge seams ON.

| axis (6 seeds) | absolute spike read (baseline) | NMDA-conductance divisive-norm (fix) |
|---|---|---|
| A. ONE shared pool (metacog.bridge IS pragmatic.bridge IS substrate, N=450) | 6/6 | 6/6 |
| B. MERGED == CORESIDENT byte-identical (Δ==0.0) | 6/6 | 6/6 |
| **MERGE-GO (A + B)** | **6/6** | **6/6** |
| C. answer preserved MERGED-vs-TODAY — PRAGMATIC | 6/6 | 6/6 |
| C. answer preserved MERGED-vs-TODAY — **METACOG** (`confident` bool) | **1/6** | **6/6** |
| MONO — confident monotone-nondecreasing in evidence (today / merged) | — | 6/6 / 6/6 |
| DGN — non-degenerate / tracks-evidence (low→uncertain, high→confident) | — | 6/6 / 6/6 |
| **FULL FLIP-GO (A + B + C both organs + MONO + DGN)** | **1/6** | **6/6** |

The NMDA read yields the IDENTICAL clean monotone pattern `[F,F,F,F,T,T,T,T]` (confident/uncertain boundary at evidence
~0.5) on ALL 6 seeds, both TODAY and MERGED — not merely answer-preserving but a consistent, seed-robust confidence
code. The absolute baseline reproduces the pre-existing 1/6 blocker EXACTLY (its own standalone pattern is non-monotone
noise: seed 43 boundary drifts to ~0.82; seed 102 calls ZERO-evidence "confident" and evid 0.75 "uncertain").

## Anti-cheats / why the GO holds

- **Load-bearing (lever):** the `--read balance` control (absolute spike margin) is 1/6 in the SAME runner — the
  conductance divisive-norm is the cause of the invariance, not the harness. `lever()` confirms the read values move
  substantially between the absolute and NMDA reads (the runner's printed LEVER line). Lesion (remove the evidence
  differential) collapses the margin (inherited).
- **Not trivial invariance:** merged==today is not "read made constant" — B (merged==coresident, Δ==0.0) shows
  co-residence adds no footprint, and MONO + DGN + tracks-evidence confirm the read genuinely discriminates
  (`[F,F,F,F,T,T,T,T]`, boundary in the mid-range, monotone in both builds, 6/6).
- **Brain-based:** the read is a divisive normalization of two `cp_conductance_g_nmda` accumulator reads (a synaptic,
  spike-driven substrate state), NOT the injected current and NOT a host rescale of the answer.
- **Winning interpretation:** under the MERGE (merged vs today, both robust) the metacog decision is IDENTICAL 6/6, and
  pragmatic's implicature/enriched interpretation is unchanged 6/6.

## Honest residual (declared, not deferred)

The fix is a NEW confidence read, not the currently-shipped absolute-spike read (`metacog_production_organ`,
`confidence_read="balance"`, default-ON since 2026-08-12). Adopting it changes some STANDALONE confidence calls vs the
old read (full-sweep agreement 4/6; clear-extreme agreement 5/6) — but every change is a DE-NOISING correction: the old
read is at the noise floor (non-monotone; on seed 102 it calls zero-evidence "confident"), and the NMDA read is monotone
and consistent across seeds. So the pool-#2 flip is UNBLOCKED **conditional on the metacog organ adopting the
NMDA-conductance divisive-normalized read** — a read UPGRADE that also stabilizes the standalone organ. This de-risk
proves the mechanism; it does NOT itself swap the shipped organ's read nor flip `_MERGE2_DEFAULT_ON`.

## Named next rung (production wiring)

Swap the metacog organ's confidence read to the divisive-normalized NMDA-conductance margin (behind an additive,
default-preserving flag, then flip the default), re-run `_onebrain_production_flip2_verify` end-to-end, and flip
`onebrain_merge_production2._MERGE2_DEFAULT_ON=True`. Affect remains a separate rung (its own pool / the recall-composer
bridge — the structural name-collision + global-OU exclusion from `2026-08-13-onebrain-second-pool-SCOPED.md`).

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._metacog_robust_confidence_derisk \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_metacog_robust_confidence_6seed.json
```

## Scope / non-claims

- `wired: NO (the robust divisive-normalized NMDA-conductance confidence read is de-risked in
  research/runners/_metacog_robust_confidence_derisk.py::RobustMetacogProductionOrgan; it is NOT yet swapped into the
  shipped metacog_production_organ, and the pool-#2 default is NOT flipped) / on_by_default: NO / scaffold_retired: none
  (de-risk only — the production wiring is the named next rung).` Functional read-out only; no phenomenal claim.
- NO `sim/` edit; reuse-by-import; process backend (numpy in this de-risk → bit-exact; cupy in production — the read is
  deterministic given the substrate, exactly as the existing organ).
- The GO is on the pool-#2 answer-preservation A/B (the flip blocker). It does not re-assert the E1 type-2 SDT / meta-d'
  faculty gate, which is a separate metric on a separate runner.
