---
type: finding
status: contributing
date: 2026-08-10
mechanism: pragmatic-success-signal
lane: D-pragmatics
seeds: [42, 43, 44, 100, 101, 102]
instrument: succ_opt-vs-aligned agreement PLAIN vs PRAGMATIC (STEP 1 deterministic ceiling) + learned-aligned PLAIN vs PRAG with the fix/yoked contingency decomposition (STEP 2). The distinctiveness term's effect is attributed by the plain-vs-prag agreement lift and the fix≫yoked separation.
---

# The learn-to-speak reward-misspecification is a PARTIAL win with a decisive RE-DIAGNOSIS: it is a substrate coincidence-DETECTOR artifact (per-pool base-rate + per-(t,u) heterogeneity), NOT an RSA/Gricean belief gap — a distinctiveness term helps 2/6 seeds; the base-rate half is closeable but the per-cell residual is the recurring margin-SNR wall (dendritic surpass), and the "belief gap" is really an INTEGRATION gap (the substrate's depth-2 implicature is a 6/6 GO, just not wired into this pipeline)

Follows the learn-to-speak learning-wall fix (`2026-08-10-learn-to-speak-...-the-learning-wall-fixed.md`), which named
this reward misspecification (success-optimal == belief-aligned only 8/18) as the distinct next frontier.

## The re-diagnosis (the load-bearing result)

<!--derived-->

The hypothesized fix was an RSA/Gricean DISTINCTIVENESS term (the reward rewards the utterance that DISTINGUISHES the
intent). Reading the code + a deterministic table diagnostic (`_pragmatic_success_distinctiveness_derisk.py`): **the
"RSA belief gap" framing is a MISCHARACTERIZATION.** `s(t,u)` reads a neural coincidence pool driven by the listener
belief `belief[u]` — and for ALL 6 seeds the artifact's `belief_u_t` matrix is exactly the **IDENTITY**
`[[1,0,0],[0,1,0],[0,0,1]]` (verified from `pragmatic_distinctiveness_step1_6seed.json`). The pipeline's belief source
(`_belief_sources` in `_pragmatic_success_readback_leg2_v2_derisk.py`, `_rsa_recursion(..., settle_ms=25)`) collapses
the depth-2 L1 to WINNER-TAKE-ALL, so `aligned[u]=u` is trivial identity naming with NO graded implicature content for
the reward to reflect. The entire `succ_opt != aligned` gap is therefore a **substrate coincidence-DETECTOR artifact**:
a per-utterance BASE RATE (some utterance columns are intrinsically hot for every intent) + per-(t,u) heterogeneity +
belief-independent baseline firing. ⇒ **the wall lives in the DETECTOR, not the listener's belief.**

**IMPORTANT (RAG-checked — the substrate CAN reason pragmatically; this is an INTEGRATION gap, not a substrate limit).**
Depth-2 scalar implicature is already a **6/6 GO on the spiking substrate**
(`2026-08-01-W4-recursive-theory-of-mind-2nd-order-false-belief-plus-depth2-scalar-implicature-6seed-GO.md`): the
neural depth-2 RSA turns L0("some")=[SBNA .5, all .5] into a GRADED L1 that prefers SBNA (a real "some→not all"
implicature, ~+0.033 margin, lesion/permute collapse it). This pragmatic-SPEAKING pipeline simply does NOT source its
belief from that graded RSA — it uses the leg2_v2 winner-take-all build. So the upstream frontier is **wiring the W4
graded-implicature RSA into the speaking pipeline's belief source** (connecting two existing GO pieces — graded
implicature + the learn-to-speak state-value critic), NOT an open "can the substrate produce graded posteriors"
question.

## The PARTIAL win (the distinctiveness term is mechanistically correct, and helps)

<!--derived-->

`s_prag(t,u) = S[t,u] − mean_{t'≠t} S[t',u]` (subtractive contrast over intents; a divisive variant equivalent)
removes the shared BASE-RATE component:
- STEP 1 (deterministic ceiling, 6 seeds): succ_opt==aligned **8/18 (PLAIN) → 11/18 (PRAG)** — fixes seeds 42/100/101
  fully; 0/6 seeds degenerate (non-degeneracy control passes).
- STEP 2 (plug into the v3 spiking state-value learner, `_pragmatic_success_distinctiveness_learn_derisk.py`, n_train=360):
  learned-aligned **0.444 (PLAIN, reproduces the ~0.44 cap) → 0.556 (PRAG, +0.11)**, driven by seeds 42/100 climbing
  0.667→1.0. Contingency PRESERVED (fix target-weight-separation 0.8-2.5 ≫ yoked ≤0.42; and the mean-centered
  pragmatic reward has a SMALLER yoked leak than plain — a clean-contingency side benefit).

## Why it does NOT close it + the real next levers

<!--derived-->

3/6 seeds are unmoved because their residual is IDIOSYNCRATIC per-(t,u) detector heterogeneity that corrupts the
DIAGONAL itself (e.g. seed 43: `S[2][2]=0.033` is the LOWEST cell in the table) — no marginal reward-shape contrast
(subtractive or divisive) can touch a per-cell defect.

**The record already characterizes "detector homeostasis" — it is a base-rate fix, NOT a closer (RAG-checked).**
The distinctiveness contrast `S[t,u] − mean_{t'≠t}S[t',u]` IS per-utterance-column baseline removal done host-side,
and its DIVISIVE variant scored identically (both 11/18). The nav cascade already ran the SUBSTRATE form of exactly
this — per-pool baseline equalization / Carandini-Heeger divisive normalization (`input_divisive_norm`,
`sim/bridge.py:6076-6080`, the point-neuron-proven in-sim primitive): in `2026-06-20-cascade-north-bias-FIX.md`
**FIX 2** cut the all-saturate tie fraction 0.329→0.181 (removed the base-rate component, as intended) but did **NOT
further improve the score** — "the residual is now a **margin-SNR** issue". So the base-rate half is closeable (the
host column-subtraction already does it, 11/18), but the residual is margin-SNR, not a base-rate the record can strip
further. **And a PER-NEURON rate homeostat is specifically NOT the lever for this DETECTOR**, because the pragmatic
detector pools are SEPARABLE/disjoint: the same-day WTA-premise reframe
(`2026-08-10-neural-WTA-separable-assemblies-weight-controllable-homeostat-premise-REFRAMED.md`; and the ⚠️ banner on
`2026-08-10-neural-WTA-afferent-winner-common-mode-removal-research-gate.md`) established that disjoint assemblies have
NO per-assembly common-mode for a per-neuron threshold homeostat to strip — the homeostat design is reserved for the
CO-RESIDENT/dendritic case only. (A per-neuron threshold homeostat was separately NEGATIVE at a suprathreshold
operating point, `2026-08-10-value-critic-homeostat-NEGATIVE-*`.) ⇒ the residual is a genuine margin-SNR wall, not a
homeostat away.

**The real residual is the RECURRING margin-SNR / point-soma wall, and it has a partial biological surpass.** The same
"selectivity decoupled from magnitude on a point neuron" wall shows up in (i) nav selection (above), (ii) the
value-critic readout SNR (`2026-08-10-value-critic-...`), and (iii) CA3 completion (the trilemma,
`2026-08-10-ca3-point-neuron-attractor-completion-trilemma-NEGATIVE-*`). For (iii) the STANDING surpass is the
two-compartment **dendritic dAP READOUT** (`2026-07-08-riii-onsubstrate-dendritic-dAP-completion-SURPASS-6seed.md`:
held-out completion 0.571 vs LINEAR point-neuron 0.007, 6-seed, 4 controls) — its regenerative plateau supplies
selectivity decoupled from magnitude, exactly the per-cell horn the point-soma detector cannot hold. **Honest bound on
that surpass:** it is a READOUT on a HAND-INSTALLED attractor; the LEARNED + seed-robust bistable version is NOT closed
(the 2026-07-18 "learned CLOSED" claims were retracted as self-sustaining / plasticity+noise confounds; sparse
recurrence gives only partial specificity; the current open leads are assembly-selective inhibition + a somatic
slow-NMDA reverberatory attractor — `2026-07-18-gap5-bistable-completion-mechanism-research-gate.md`). ⇒ **the
indicated build is the dendritic-plateau nonlinearity on the detector pool (a hand-installed-readout probe first, per
the standing 2026-07-08 result), not another reward-shape or a point-neuron homeostat.**

**And an UPSTREAM INTEGRATION gap (arguably the more important frontier):** this pipeline's belief is one-hot identity
because it sources from the leg2_v2 winner-take-all RSA, NOT because the substrate can't reason — depth-2 scalar
implicature is a standing 6/6 GO (W4, above). Wiring the graded-implicature RSA into the speaking pipeline's belief
source (so `aligned` carries real "some→not all" content) is the indicated integration — it connects two existing GO
pieces rather than opening a new substrate question.

## Honest bounds

<!--derived-->

The distinctiveness contrast is a HOST readout over neural coincidence rates (same footing as the existing scalar
success readout); a fully-neural lateral-inhibition pool is the upgrade if pursued. STEP 1 is a small K=3 DETERMINISTIC
coincidence-table ceiling (exact); STEP 2 is the REAL spiking bridge (880-neuron Izhikevich, the v3 state-value
learner, n_train=360/seed) — its learned WTA readout adds tie-resolution noise (seed 101 plain hit 1.0 by a near-tie),
so the headline is the aligned-rate lift + the contingency, not a single seed's flag. Contingency rests on fix≫yoked
separation (robust), not the brittle binary pass flag. Both runs SIM_BACKEND=numpy; step2 reproduced locally
(PLAIN 0.4445 → PRAG 0.5555, byte-matching the pooled run).

Artifacts: `research/findings/raw/_pragmatic_success/pragmatic_distinctiveness_step1_6seed.json` (the ceiling),
`research/findings/raw/_pragmatic_success/pragmatic_distinctiveness_step2_6seed.json` (the learner). Reproducers:
`research/runners/_pragmatic_success_distinctiveness_derisk.py`, `_pragmatic_success_distinctiveness_learn_derisk.py`.
NO `sim/` edit. SIM_BACKEND=numpy.
