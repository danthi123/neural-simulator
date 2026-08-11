---
type: finding
status: contributing
date: 2026-08-11
mechanism: pragmatic-graded-implicature-belief-source
lane: D-pragmatics
seeds: [42, 43, 44, 100, 101, 102]
instrument: A/B (onehot [leg2_v2 baseline] vs graded [W4 depth-2 RSA posterior]) on the reward-misspec finding's pragmatic-alignment metric — succ_opt==aligned (STEP 1 deterministic ceiling) + learned-aligned (STEP 2 v3 state-value learner, PLAIN reward, fix+yoked). Belief calibration = L1 distance to the analytic Frank-Goodman RSA posterior. Moat = the normalization-lesion of the graded implicature.
artifacts:
  - research/findings/raw/_pragmatic_success/graded_belief_source_6seed.json
  - research/findings/raw/_pragmatic_success/graded_belief_step1_6seed.json
---

# W4 graded-implicature RSA wired as the speaking-pipeline belief source: the INTEGRATION is real (belief now carries the "some→not all" content, 12× better calibrated, moat intact) but it does NOT move the argmax pragmatic-alignment metric — the residual is the DETECTOR base-rate artifact + argmax-insensitivity, exactly the reward-misspec finding's own re-diagnosis, now confirmed by direct A/B

Wires in the graded belief the 2026-08-10 reward-misspec finding named as the INTEGRATION gap
(`2026-08-10-reward-misspec-distinctiveness-PARTIAL-rediagnosed-as-detector-artifact-not-RSA-belief.md`): "the belief
gap is really an INTEGRATION gap (the substrate's depth-2 implicature is a 6/6 GO, just not wired into this
pipeline)". Depth-2 scalar implicature is a standing 6/6 GO (W4,
`2026-08-01-W4-recursive-theory-of-mind-2nd-order-false-belief-plus-depth2-scalar-implicature-6seed-GO.md`). This
connects the two GO pieces (graded implicature + the learn-to-speak state-value critic) and MEASURES the metric.

## Read the substrate first: the one-hot was the neural L1 collapsing, not a missing implicature

<!--derived-->

The leg2_v2 `_belief_sources` already reads the neural depth-2 L1 via `_rsa_recursion(..., settle_ms=25)` — but at the
W4-calibrated operating point (strong FS divisive normalization) the FINAL L1 `_compete` hard-suppresses the losing
state to EXACTLY 0: `L1("some")=[none 0, SBNA 0.0372, all 0.0]` → normalized to the one-hot `[0,1,0]`; `L1("all")`
dies to all-zeros → literal fallback `[0,0,1]`. So the effective belief is the IDENTITY matrix (reproduced: the
committed `pragmatic_distinctiveness_step1_6seed.json` records `belief_u_t == I` on all 6 seeds). That IS the
"winner-take-all one-hot" the finding named — and it is a READOUT collapse, not a missing capability.

The FAITHFUL graded posterior. In RSA the depth-2 listener posterior is `L1(s|u) ∝ prior(s)·S1(u|s)` (uniform prior).
The substrate's S1 rates (the depth-2 RSA SPEAKER distribution — itself a W4 GO component, neural rates from the FS
divisive normalization) are GRADED: `S1("some")=[0, 0.0439, 0.0161]` → normalized over states = `[0, 0.731, 0.269]`,
matching the analytic Frank-Goodman `L1("some")=[0, 0.75, 0.25]`. So the graded belief reads
`L1(s|u)=normalize_states(S1_neural[u,:])` — the TRUE graded posterior, read one competition-step before the operating
point's final hard-WTA `_compete` zeroes the loser. It carries the real "some→not all" content (SBNA ~0.73 preferred,
`all` still ~0.27-possible) instead of the false one-hot claim that `all` is impossible after "some". Same host
normalization the existing pipeline already applies to its neural L1 rates (`v/v.sum()`), so faithfulness parity holds.

## Result — 6 seeds: the graded belief is faithfully wired, but the argmax alignment metric does not move

<!--derived-->

| read-out (6 seeds) | onehot (leg2_v2 baseline) | graded (W4 RSA) | verdict |
|---|---|---|---|
| **STEP 1** succ_opt==aligned | **8/18** | **7/18** | not moved (−1) |
| **STEP 2** learned-aligned (PLAIN reward) | **0.444** | **0.389** | not moved (−0.056) |
| belief calibration: L1(some)→analytic RSA (lower=better) | 0.500 | **0.041** | 12× better |
| graded implicature margin (SBNA−all) | — | **+0.506** | present |
| — under normalization-lesion (moat) | — | **+0.006** | collapses (98.9% attributable) |

The onehot arm reproduces the reward-misspec finding's numbers EXACTLY — succ_opt==aligned **8/18** and STEP-2
learned-aligned **0.444** (the finding's ~0.44 PLAIN cap) — a byte-identical belief source (`onehot` ==
`_belief_sources`), anchoring the A/B. The graded arm does NOT improve either: succ_opt 7/18 (unchanged on 5/6 seeds;
seed 102 slips 1→0), learned-aligned 0.389 (equal on 4/6 seeds, worse on seed 100 where training noise flips
0.667→0.333, so −0.056 in aggregate). Contingency held IN AGGREGATE for both belief arms (mean fix weight-separation
≫ yoked: onehot 1.17 vs −0.07, graded 1.74 vs 0.20) — the finding's own honest bound is that contingency rests on
this separation, not the brittle per-seed binary flag (which counted 4/6 onehot, 3/6 graded).

## Why it does not move the metric (two structural reasons, both already implied by the finding)

<!--derived-->

1. **The metric is argmax; the graded refinement is sub-argmax.** `aligned[t]=argmax_u belief[u][t]` is the IDENTITY
   for BOTH beliefs — the graded belief only adds 0.27 mass on `all` under "some", which never changes an argmax. So
   the target the metric scores against is byte-identical; only belief MAGNITUDES change, and succ_opt/learned-aligned
   are argmax reads that are structurally INSENSITIVE to magnitude refinement.
2. **The succ_opt gap is the DETECTOR, not the belief.** The reward-misspec finding already re-diagnosed the
   `succ_opt != aligned` gap as a substrate coincidence-DETECTOR artifact (per-utterance base rate + per-(t,u)
   margin-SNR heterogeneity that corrupts the diagonal). Swapping the belief drive does not touch that detector
   artifact. This A/B is the direct confirmation: a faithfully graded belief, fed through the SAME detector, lands the
   same 7-8/18. **The wall lives in the detector, as the finding said — not the listener's belief.**

## What IS banked (the integration is real), and what it is worth

<!--derived-->

The integration itself is real and banked: the speaking pipeline can now source a GRADED, calibrated listener belief
that carries the depth-2 scalar implicature (12× closer to the analytic RSA posterior), collapsing to flat under the
normalization-lesion — so the graded content is the substrate's FS divisive normalization (the W4 mechanism), not a
host-injected table. This removes the FALSE one-hot claim ("`all` is impossible after `some`") from the pipeline's
world-model, which is the representationally-correct pragmatic calibration the north-star asks for — even though the
argmax alignment metric cannot reward it. The two GO pieces are now connected; `--belief onehot` keeps the exact
leg2_v2 baseline (additive, default-off).

## The residual + the named next mechanisms (per THE LAW: a wall on a METHOD, not the capability)

<!--derived-->

- **For the metric (the detector-SNR wall):** the standing 2026-07-08 **dendritic dAP READOUT** GO (`2026-07-08-riii-onsubstrate-dendritic-dAP-completion-SURPASS-6seed.md`) — a READOUT nonlinearity, explicitly NOT the tested-NEGATIVE dendritic/BDSP deep-CREDIT-assignment rule (`2026-05-17-dendritic-credit-assignment-NEGATIVE.md`, `2026-07-22-gap4-real-issue-NOT-dendrites-and-timing-FIRST-CLASS-deep-research.md`) — applied to the detector pool (held-out completion 0.571 vs point-neuron 0.007, 6-seed).
  Its regenerative plateau supplies the selectivity-decoupled-from-magnitude the point-soma coincidence detector cannot
  hold on the corrupted diagonal (the margin-SNR / point-soma wall the reward-misspec finding redirected to) — the
  indicated build for succ_opt.
- **For rewarding graded calibration:** the argmax metric cannot see the refinement, so a **magnitude-sensitive
  pragmatic reward** (informativeness = the listener's graded posterior mass on the true intent, `belief[u][t]`, read
  through the neural coincidence rate rather than an argmax) is the reward-side change that would let the calibrated
  belief pay off. A fully-neural soft-competition L1 read (in place of the host `normalize(S1)`) is the belief-side
  upgrade. Neither is a re-sweep of the current method.

## Honest scope

<!--derived-->

A FUNCTIONAL pragmatics correlate: a listener-belief source carrying the depth-2 scalar implicature (graded,
collapsing under the normalization-lesion), wired as the speaker's environment. NOT a claim of phenomenal access to
another mind; self-report would be a functional read-out. The graded posterior's final normalization is a host op on
neural rates — the same footing as the existing pipeline's L1 normalization; a fully-neural soft-competition read is
the stated upgrade. Plasticity off in STEP 1 (fixed operating point, as in the W4/leg2 GOs); STEP 2 learns only via
the reward-modulated three-factor rule with a host-EMA state-value baseline (per v3). numpy-CPU on real spiking
Izhikevich bridges; additive NEW runner (reuse-by-import of W4 + leg2_v2 + the v3 learner); NO `sim/` edit;
`--belief onehot` reproduces the leg2_v2 baseline byte-for-byte.

Artifacts: `research/findings/raw/_pragmatic_success/graded_belief_source_6seed.json` (both steps),
`research/findings/raw/_pragmatic_success/graded_belief_step1_6seed.json` (STEP-1 ceiling). Reproducer:
`research/runners/_pragmatic_graded_belief_source_derisk.py` (`--step both --seeds 42 43 44 100 101 102 --n-train 360`).
SIM_BACKEND=numpy.
