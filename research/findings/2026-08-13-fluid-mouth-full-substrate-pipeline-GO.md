---
type: finding
status: qualified
claim_check: synthesis
verdict: >
  GO (6-seed, 6/6). The ENTIRE mouth next-word path is now on the spiking substrate END-TO-END: the WKV recurrent
  leaky STATE (held as [WK]'s slow recurrent-NMDA conductance cp_conductance_g_nmda_recurrent) feeds [CE]'s two chained
  substrate signed-graded reads — the output projection Wo_sp@state then the read-out head_w@h with head_b as a tonic
  bias-input population — producing the next-word winner off cp_conductance_g_e/g_i, with 0 host matmul on the state or
  the margin and 0 host RNG draws on the read path. Over real TinyStories context the full substrate state->logits
  pipeline reproduces the FULL HOST MOUTH's next-word decision at recov_argmax 0.9137 mean / 0.8887 min (6/6), top-1
  argmax_agree 0.694, deep-context (>=8 tokens) argmax_agree 0.629. The composition is essentially MULTIPLICATIVE, not a
  collapse: within the SAME run and eval set, the host-state->substrate-read arm (== the [CE] read chain) recovers
  0.9482 and the substrate-state->host-read arm (== the [WK] state) recovers 0.9620, and 0.9482 x 0.9620 = 0.912 ~
  the observed 0.9137 — so moving the STATE onto the substrate on top of the substrate reads costs only ~0.035 recov.
  All brain-based anti-cheats collapse 6/6: lesion the state input (zero-input) 0.694->0.125, reset the NMDA
  conductance every token (memoryless) ->0.257, lesion the projection read stage (zero-state) ->0.031, lesion the
  read-out read stage (zero-feature) ->0.031, scramble ->0.000 (chance 0.001); the inhibitory read-out shadow stays
  load-bearing (argmax_agree > positive-only on all 6). Both stages reuse their arcs' OWN fixed seed-42 calibrations
  (no new tuning): [WK]'s per-channel affine scale/off, [CE]'s proj_out_scale=0.30 + bias_scale=0.14. STILL host: the
  input projection Wv (BPTT weights; the LEARNING rule is the 2026-08-12 diagonal e-prop GO), the r_h gate, the LN, the
  trained decay/Wo_sp/head weights, and the three fixed unit-scalars. A named residual (the whole state->logits chain)
  moved onto neurons/synapses/graded conductances end-to-end; a real step, honestly bounded. NOT "fully spiking" / NOT
  production-wired. Runner-only, default-off, NO sim/ edit.
lane: gap#1 / A1 (brain-native open-prose mouth — the full state->logits path on the substrate)
date: 2026-08-13
mechanism: >
  CHAIN two separately-validated GO substrates (wiring, NO new mechanism). STAGE 0 (the STATE, [WK]'s
  GradedRecurrentState): per token rate=[relu(+v),relu(-v)] (host input projection Wv, DECLARED residual) drives a
  carrier population over a long window (t_step=40) onto F=2D subthreshold channels via nmda_slow; the channel's slow
  recurrent-NMDA conductance is a dual-exp leaky integral of relu(v) = the WKV state ap/an, read LINEARLY and mapped by
  a fixed per-channel affine cst = scale*g + off onto the host-state scale. STAGE 1+2 (the READS, [CE]'s
  ComposedEndToEndRead composed_biaspop): cst = [ap_sub, an_sub] is fed in place of the host state; the substrate
  output projection reconstructs Wo_sp@cst off cp_conductance_g_e/g_i (gated by the host r_h, scaled by proj_out_scale),
  the dual-nonneg feature drives the read-out bridge, Wp/Wn (head_w Dale-split) wire onto V word-pools, and a tonic
  bias-input population (bias_e/bias_i, weights ~ head_b) injects the base-rate prior as a synaptic conductance; winner
  = argmax over the net-current margin df_e*g_e + df_i*g_i. The reference is the full host mouth
  ro.logits(ap_host,an_host,tid). Three arms per position isolate WHERE composition costs: A fullsub (substrate state ->
  substrate reads), B hoststate_subread (host state -> substrate reads == [CE]), C substate_hostread (substrate state ->
  host read == [WK]). cfg.seed-controlled substrate; runner-only, default-off, NO sim/ edit.
artifacts:
  - research/runners/_wkv_mouth_full_substrate_pipeline_derisk.py
  - research/findings/raw/_wkv_full_substrate_pipeline_6seed.json
  - research/findings/raw/_wkv_full_substrate_pipeline_smoke.json
---

# gap#1 / A1 — the FULL mouth state->logits path on the substrate END-TO-END (GO, 6-seed 6/6)

## What was separately validated, and what this composes

The mouth's per-token next-word chain, with the substrate boundary as of 2026-08-13:

    (1) v      = Wv @ LN(emb[tid])                     # input projection    (host, BPTT weights — DECLARED residual)
    (2) ap,an  = decay*ap+relu(v), decay*an+relu(-v)   # WKV leaky STATE     <<< [WK] substrate slow-NMDA conductance
    (3) r_h    = sigmoid(Wr @ LN(emb[tid]))            # receptance gate     (host — DECLARED residual)
    (4) h      = r_h * (Wo_sp @ [ap,an])               # OUTPUT PROJECTION   <<< [CE] substrate graded read
    (5) logits = head_w @ h + head_b                   # read-out            <<< [CE] substrate graded read + bias pop

Two pieces were each a GO IN ISOLATION, but each took the other's input host-side:

- **[WK]** `2026-08-13-fluid-mouth-wkv-state-graded-conductance-integrator-GO` — step (2) held on the real Izhikevich
  bridge as the slow recurrent-NMDA conductance `cp_conductance_g_nmda_recurrent` (a clean dual-exp leaky integral of a
  graded carrier drive), state_corr 0.793 6/6, reproducing the host next-word decision 0.797 THROUGH A HOST READ. Its
  named next rung #2 was verbatim this composition ("feed the substrate state into the graded Wo_sp@state into the
  graded head_w@h").
- **[CE]** `2026-08-13-fluid-mouth-endtoend-substrate-read-GO` (composed_biaspop arm) — steps (4)+(5) as ONE substrate
  signed-graded pipeline (every matmul a `cp_conductance_g_e/g_i` read) + head_b as a tonic bias-input population,
  recov_argmax 0.9495 6/6 — but it took the STATE HOST-SIDE (its residual #4: "the WKV recurrent STATE ... is host").

This lane feeds [WK]'s SUBSTRATE state (calibrated `cst = scale*g_nmda_recurrent + off`, on the host-state scale) into
[CE]'s substrate read chain in place of the host `[ap,an]`. So the ENTIRE next-word path — state integration ->
projection -> read-out -> logits — is substrate end-to-end. Both calibrations are the arcs' OWN fixed seed-42 values;
no new tuning was introduced by this lane.

## RESULT — 6-seed (42/43/44/100/101/102; V=1000; D=128; t_step=40; carrier_pop=24; read_window=150; n_eval=200; GPU cupy)

<!--derived: research/findings/raw/_wkv_full_substrate_pipeline_6seed.json-->

| seed | A fullsub recov | A argmax_agree (>pos) | A deep_agree (n) | B host-state (==CE) | C host-read (==WK) | zero-in | mless | zero-state | zero-feat | scr | GO |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42  | 0.9443 | 0.770 (0.600) | 0.867 (15) | 0.9647 | 0.9765 | 0.150 | 0.292 | 0.058 | 0.058 | 0.000 | ✓ |
| 43  | 0.9074 | 0.705 (0.575) | 0.571 (7)  | 0.9381 | 0.9555 | 0.117 | 0.242 | 0.033 | 0.033 | 0.000 | ✓ |
| 44  | 0.9292 | 0.705 (0.570) | 0.818 (11) | 0.9599 | 0.9603 | 0.100 | 0.225 | 0.025 | 0.025 | 0.000 | ✓ |
| 100 | 0.8887 | 0.645 (0.575) | 0.450 (20) | 0.9535 | 0.9698 | 0.100 | 0.258 | 0.017 | 0.017 | 0.000 | ✓ |
| 101 | 0.9040 | 0.675 (0.550) | 0.600 (10) | 0.9461 | 0.9567 | 0.167 | 0.283 | 0.033 | 0.033 | 0.000 | ✓ |
| 102 | 0.9084 | 0.665 (0.550) | 0.467 (15) | 0.9270 | 0.9531 | 0.117 | 0.242 | 0.017 | 0.017 | 0.000 | ✓ |
| **mean** | **0.9137** (min 0.8887) | **0.694** (0.570) | **0.629** | **0.9482** | **0.9620** | **0.125** | **0.257** | **0.031** | **0.031** | **0.000** | **6/6** |

**Headline.** The full substrate state->logits pipeline reproduces the full host mouth's next-word decision at
**recov_argmax 0.9137** (6/6, min 0.8887) and **top-1 argmax_agree 0.694** (deep-context 0.629), NOT collapsed. It sits
a MEASURED ~0.035 recov below the host-state substrate read (arm B, 0.9482) — the honest cost of also moving the STATE
onto the substrate on top of the substrate reads.

## The composition is MULTIPLICATIVE, not a collapse (the decomposition)

Three arms share the SAME eval positions, so the deltas isolate WHERE composition costs. Arm B (host state ->
substrate reads) is the [CE] read chain; arm C (substrate state -> host read) is the [WK] state. The full pipeline
(arm A) recovers **0.9137**, and **recov(B) x recov(C) = 0.9482 x 0.9620 = 0.912 ~ 0.9137**. The two stages' errors
compose almost exactly as independent factors — the substrate state and the substrate reads each shave a few percent
of the peaked, near-tied next-word mass, and chaining them does not amplify into a collapse. This is the composed
analog of [WK]'s 0.797 argmax (state through a host read) and [CE]'s 0.769 argmax (host state through the substrate
reads): the full-substrate top-1 agreement 0.694 is close to their product-of-agreements and well above every
anti-cheat floor.

## Anti-cheats (all 6 seeds; each MUST collapse — brain-based, negatives load-bearing)

- **Lesion the STATE (zero-input, cache-immune):** driving the WKV carriers with a ZERO rate (the state decays to ~0)
  drops the full-pipeline argmax_agree 0.694 -> **0.125** — the SUBSTRATE state drives the whole downstream chain.
- **Memoryless (recurrence load-bearing):** resetting `cp_conductance_g_nmda_recurrent` every token drops it -> **0.257**
  — the substrate is genuinely INTEGRATING across tokens, not re-reading a per-token transient.
- **Lesion the PROJECTION read stage (zero-state at the projection input):** -> **0.031** (chance 0.001).
- **Lesion the READ-OUT read stage (zero-feature at the read-out input):** -> **0.031**.
- **Scramble (post-hoc pool->word relabel):** -> **0.000** on every seed.
- **Signed shadow load-bearing:** argmax_agree > positive-only on all 6 (e.g. 0.770 vs 0.600) — the inhibitory `Wn`
  read-out weights carry signal, not decoration.
- **Provenance:** state read off `cp_conductance_g_nmda_recurrent`; winner off `cp_conductance_g_e/g_i`; head_b via a
  spiking tonic-bias synapse; `host_rng_draws_on_read_path == 0`, pools spike ~1.85/read, bias-pop ~0.85/read, on all 6.
  0 host matmul on the state or the margin.

## External grounding (composed from the two parent findings' resolving anchors)

The mechanism inherits the biology of its two stages, each with a source anchor that resolves in the parent finding:
the slow recurrent-NMDA integrator state (Wang XJ 1999, J Neurosci 19(21):9587 — persistent-activity integrator state
requires recurrent excitation dominated by a SLOW NMDA component); the faithful LINEAR graded-conductance read
(Holt & Koch 1997 — the subthreshold membrane current at rest is a linear combination of synaptic conductances); and
head_b as a per-pool tonic baseline / starting-point offset (Mulder 2012, J Neurosci 32(7):2335 — prior probability
biases choice via the accumulation starting point). This lane adds no new biology; it composes the three.

## Honest residuals — what is substrate, what is host

1. **ON the substrate now (end-to-end):** the WKV recurrent leaky STATE integration (slow recurrent-NMDA conductance,
   persisting + decaying across tokens), the per-token input DELIVERY (graded carrier drive), the STATE READ (linear
   graded conductance), the OUTPUT PROJECTION (signed graded read off g_e/g_i), the READ-OUT (signed graded margin),
   and head_b (tonic bias-input population). Every matmul between the state and the winner is a conductance read; the
   winner is an argmax over the substrate net-current margin. 0 host matmul on the state/margin, 0 host RNG draws.
2. **A measured ~0.035 recov composition penalty, not a collapse.** The full pipeline (0.9137) trails the host-state
   substrate read (0.9482) because the substrate state reconstructs ap/an imperfectly (state_corr 0.79 in [WK]) and
   that error is amplified at near-ties; it composes multiplicatively with the read chain, holding recov ~0.91.
3. **deep-NLL gap is positive (0.2-1.75 nats).** The read is an argmax DECISION, not a calibrated distribution; the
   mass it lands on the true target is below the host softmax's. recov_argmax (mass on the substrate winner / mass on
   the host winner) is the clean fidelity metric; the deep-NLL gap is reported as characterization, not a parity claim.
4. **STILL host / BPTT:** the input projection Wv (BPTT weights; the transport-free diagonal e-prop LEARNING rule is the
   separate 2026-08-12 GO), the r_h receptance gate, the LN inside the state, the trained decay/Wo_sp/head_w/head_b
   VALUES, and three fixed unit-scalars (scale/off, proj_out_scale, bias_scale). rate v=Wv@LN(emb) is host-computed.
5. **NOT "fully spiking" / NOT production-wired.** A named residual (the whole state->logits chain) moved onto the
   substrate end-to-end at recov 0.91 — a real step on the deepest gap, honestly bounded. Functional read-outs only;
   no claim of phenomenal experience.

## Named next rungs (in tractability order)

1. **Shunt the r_h gate onto the substrate** (the projection finding's own named residual): make r_h a divisive
   (shunting-inhibition) conductance on the projection pools rather than a host elementwise multiply — removes one of
   the two remaining host operations between sensation and the winner.
2. **Substrate the input projection v = Wv@LN(emb)** as its own graded read (the projection-GO instrument applied to Wv)
   with the 2026-08-12 diagonal e-prop rule training Wv — closes the last matmul, leaving only LN + the embedding.
3. **Lift the state-delivery dead-zone** ([WK]'s residual #2: a matched-pair common-mode-subtracting carrier so small
   relu(v) fires linearly) — raises state_corr toward the 0.90 clean-input core, lifting the composition penalty.

## Files
- Runner: `research/runners/_wkv_mouth_full_substrate_pipeline_derisk.py`
- Raw: `research/findings/raw/_wkv_full_substrate_pipeline_6seed.json` (+ `_smoke.json`, + `.prov.json` sidecars)
- Builds on: `2026-08-13-fluid-mouth-wkv-state-graded-conductance-integrator-GO.md` (the substrate STATE this feeds in),
  `2026-08-13-fluid-mouth-endtoend-substrate-read-GO.md` (the substrate READ chain this feeds), and their upstream
  projection/read-out/parity-close GOs.
