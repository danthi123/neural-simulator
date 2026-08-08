---
type: finding
status: contributing
date: 2026-08-08
mechanism: pragmatic-success-two-input-coincidence-detector
lane: D-pragmatics
runner: research/runners/_pragmatic_success_coincidence_derisk.py
biology: research/biology/dendritic-plateau-coincidence-burst.md
artifacts:
  - research/findings/raw/_pragmatic_success/summary_6seed.json
  - research/findings/raw/_pragmatic_success/smoke.json
---

# Communicative success is a two-input NEURAL coincidence detector (belief × intent), 6/6-seed GO — Leg 1

**The claim, in one line.** "Did the listener come to believe what I intended" = the OVERLAP of the listener's
belief and the speaker's intent = neural `<belief, intent>`, and that overlap is computed ON THE SPIKING
SUBSTRATE as a MULTIPLICATIVE COINCIDENCE — with a LINEAR summator at matched total input rate provably UNABLE
to compute it. This is Leg 1 of workstream D (pragmatics): the communicative-success signal that Leg 2 will read
back to train speaking. 6/6 seeds GO on the decisive core; NO `sim/` edit.

## What was built

A `success[k]` detector per communicable state `k` receives TWO spiking afferents: `belief[k]` — the listener's
inferred posterior over states, sourced from the RSA speaker-listener bridge (`build_rsa_bridge`, the social
environment, reused by import) — and `intent[k]` — the one-hot communicative goal. `success[k]` is a genuine
AND: it fires only when `belief[k]` AND `intent[k]` are co-active at the SAME `k`. The scalar communicative
success is `Σ_k rate(success[k])` = the neural inner product by coincidence.

The AND is the **engine-native dendritic-coincidence plateau** (`enable_coincidence_detection`, per-pathway
`coincidence_detector=True`) — the Poirazi-Brannon-Mel 2003 two-layer subunit / Larkum distal+proximal
conjunction → plateau → burst (biology binding: `research/biology/dendritic-plateau-coincidence-burst.md`). Each
afferent alone delivers a per-step coincident COUNT below `coincidence_k_threshold` (sub-plateau); only the two
together clear it and fire the detector. This is the additive/default-off path — **no `sim/` edit**, reuse by
import of the RSA bridge + the GNW wash-out snapshot/restore machinery. Substrate seeded via `cfg.seed`
(verified byte-identical: rebuild at seed 42 hashes identically; seed 43 differs).

**Why coincidence and not summation.** The belief drive is normalized to a FIXED TOTAL and intent is always
one-hot, so the TOTAL afferent input is matched across aligned and misaligned trials. A linear read gives
`f(bel) + f(int)` — the same whether the mass overlaps at one `k` or is split across two. Only a supralinear
same-`k` conjunction distinguishes them. This is exactly the linear-summation wall documented in
`2026-06-09-coincidence-substrate-upgrade-design.md`, here turned into the discriminating mechanism.

## Result (6 seeds: 42 43 44 100 101 102, numpy-CPU)

<!--derived-->

| metric | mean | gate | verdict |
|---|---|---|---|
| coincidence AUC (aligned success > misaligned) | 0.969 | ≥ 0.85 | PASS (per-seed min 0.931) |
| LINEAR-SHAM AUC (matched total input) | 0.256 | ≤ 0.62 | PASS — cannot separate |
| SHUFFLED-K AUC (permuted belief→success topography) | 0.300 | ≤ 0.65 | PASS — collapses |
| REAL lesion AUC (silence the success column) | 0.500 | ≤ 0.65 | PASS — collapses to chance |
| MATCHED-SHAM lesion AUC (silence equal-size unrelated pool) | 0.969 | ≥ 0.80 | PASS — preserved |
| speaker read-back top-1 (REPORTED, not gated) | 0.944 | — | 5/6 seeds = 1.0; seed 100 = 0.667 |

**Verdict: GO, 6/6 seeds, moat_intact.** The decisive teeth all behave in their failing direction:

- **Linear-sham is the load-bearing control.** The SAME neurons/wiring with the plateau OFF (plain E_TO_E
  summation) at matched total input gives AUC 0.256 — it does not merely fail to separate, it ANTI-separates
  (misaligned > aligned) because point-neuron AMPA summation saturates, so `f(bel+int) < f(bel)+f(int)`. The
  separation is 152% attributable to the nonlinearity (the control moves OPPOSITE the treatment). This proves the
  separation is the coincidence, not the neurons or the drive.
- **Real vs matched-sham lesion.** Silencing the success detector pool (hyperpolarizing clamp) drops AUC to
  chance (0.500) and aligned success to 0.000; the SAME clamp on an equal-size unrelated `decoy` pool leaves AUC
  at 0.969. The flip is specific to the coincidence column, not to "any lesion." (Not tautological: the sham is
  the identical operation on the same neuron count and does NOT flip the metric.)
- **Shuffled-k** (a derangement of belief→success so no belief group meets its own intent group) collapses AUC to
  0.300 at matched total input — the same-`k` topography is load-bearing.

## Honest scope (carried into the code)

`Σ_k rate(success[k])` is a population-rate READ-OUT; the MULTIPLY (belief × intent) is done by the plateau
kernel, not the host — there is **no host index-multiply and no host argmax** in the success computation (the two
Wave-1 shortcuts that produced false passes). Belief is a legitimate social input (the RSA listener posterior),
as W4/W5 treat the truth lexicon and the situation→valence appraisal as input. This is a FUNCTIONAL
communicative-success correlate — dissociable, collapses under linear-sham / shuffled-k / real-lesion, survives
the matched sham-lesion — NOT a claim of understanding another mind.

**Speaker read-back is REPORTED, not gated.** For each intent, does success rank the RSA-aligned utterance top?
5/6 seeds = 1.000; seed 100 = 0.667 <!--derived--> (a heterogeneity-adverse seed where two RSA posteriors are
near-degenerate for one intent; rounded from the per-seed value in the cited summary). This previews Leg 2's
read-back-to-speaking, but the actual NEURAL speaker CHOICE (a WTA over a
LEARNED intent→utterance assembly, DA-gated three-factor) is Leg 2 — so it characterizes the teaching signal
rather than gating the decisive Leg-1 result. It is not lifted from a negative arm; it is a supporting metric
reported with its dip visible.

**No engram in Leg 1.** The coincidence column is a FIXED structural mechanism at a fixed operating point; nothing
learns (plasticity/reward/OU all off). The anti-cheats are therefore structural (linear-sham / shuffled-k /
lesion), not an untrained-engram arm — the untrained-engram teeth belongs to Leg 2, where the assembly is learned.

## Calibration

<!--derived-->

Operating point from a sweep (2026-08-08): `ITEM=80` neurons/assembly, `coincidence_k_threshold=44` (0.55·ITEM),
`coincidence_gain=4`, synaptic weight 2.0 (fast-AMPA kept sub-threshold; the plateau does the AND), belief/intent
drive 2500 pA. `ITEM=40` left seed 100 marginal (AUC 0.829, from a superseded ITEM=40 calibration run, not the
shipped config); widening the single-vs-double coincident-count gap (80 neurons) made all six seeds robust (min
AUC 0.931, derived over the six per-seed AUCs in the cited summary).

## Next (Leg 2, the real risk)

Wire the coincidence rate → group-scoped DA and train the intent→utterance assembly with three-factor plasticity
so the speaker CHOICE becomes a WTA over a LEARNED assembly (not a host argmax over an imported RSA table). The
coincidence-contingent reward is better-posed than the 2026-08-03 vocal-credit v1 NO-GO (a naive DA→three-factor
loop over-reinforced the early-active utterance): a mismatch → no coincidence → no DA. Declared fallback if Leg 2
fails the yoked/convergence teeth: a spiking value-critic baseline (still no `sim/` edit).

## Reproduce

```bash
# 6-seed verdict (numpy-CPU):
SIM_BACKEND=numpy python -u -m research.runners._pragmatic_success_coincidence_derisk \
    --seeds 42 43 44 100 101 102 --json research/findings/raw/_pragmatic_success/summary_6seed.json
# single-seed smoke:
SIM_BACKEND=numpy python -u -m research.runners._pragmatic_success_coincidence_derisk --smoke --seed 42 \
    --json research/findings/raw/_pragmatic_success/smoke.json
```
