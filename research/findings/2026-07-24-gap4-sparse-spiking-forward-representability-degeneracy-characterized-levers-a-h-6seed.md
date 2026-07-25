# gap#4: the on-bridge spiking blocker is a SPARSE-SPIKING FORWARD-REPRESENTABILITY degeneracy — precisely characterized (levers a–h, 6-seed, held-out) and reconciled with the 2026-07-22 credit-signal finding; NOT dendritic, NOT credit-at-sparse, NOT a pooling fix (2026-07-24)

## What this adds (and does NOT overturn)
The banked gap#4 on-bridge NO-GO (`2026-07-24-gap4-...-tonic-pinned-frozen-representation-root-cause.md`) attributed the
freeze to φ'-vanishing credit + a tonic-pinned representation. Taking that surpass further (levers a–h), the root cause
is **deeper and upstream of credit**: at the sparse spiking operating point the FORWARD hidden representation does not
preserve the input's **generalizable** class structure — so no credit rule has separable features to shape. This is
**consistent with, and a precise characterization of**, the 2026-07-22 deep-research finding's named *"rate-code /
point-neuron degeneracy … the SAME algo hits 97% on numpy graded signals, degenerate on sparse spikes"*
(`2026-07-22-gap4-real-issue-NOT-dendrites-...`). It does **NOT** overturn that finding's primary recommendation (test
the RANK-1 learned interneuron microcircuit / RANK-2 PAL credit fixes on informative-credit signals), and it explicitly
does **NOT** support a dendritic escape (2026-07-22: dendrite topology is faithful; nonlinear subunits are RANK-5).

## The decisive metric: HELD-OUT decodability (6 seeds, semantic-inheritance task, k=5, n_heldout=27)
Train a probe on the 96 train items' representation, evaluate on 27 HELD-OUT inheritance items. INPUT is the ceiling
(the generalizable structure that exists); a hidden readout "preserves" structure iff it generalizes near INPUT.

| representation | train lin | **ho lin** | train mlp | **ho mlp** |
|---|---|---|---|---|
| INPUT | 0.521 | 0.284 | 1.000 | **0.988** |
| H2 event **pooled** (32 logical) | 0.339 | 0.333 | 0.356 | **0.340** |
| H2 event **UNPOOLED** (256 raw) | 0.781 | 0.284 | 1.000 | **0.247** |
| H2 **graded-soma-V** pooled | 0.399 | 0.352 | 0.477 | **0.290** |

**Reading:** INPUT is nonlinearly separable and GENERALIZES (ho mlp 0.988) — the task needs a nonlinear hidden transform
(ho lin only 0.284). Every hidden readout of the sparse-spiking forward **fails to generalize** (ho ≤ 0.34 vs 0.99).
The unpooled raw code's train mlp 1.000 is **high-dim OVERFITTING** (256 features, 96 samples) — it collapses to
ho 0.247 ≈ chance. ⇒ the sparse-spiking forward genuinely does not carry generalizable class structure to the hidden
layer; it is not hiding in the raw neurons.

## The lever ladder (each a distinct candidate fix, each ruled out) — cheapest-first, all reuse-by-import, NO `sim/` edit
- **(a) tonic-vs-input rebalance** (make the hidden input-driven, not tonic-pinned): input_cv rose 0.27→0.52 across
  configs but the hidden linear-probe stayed frozen (+0.000); the densest config went silent. ⇒ tonic-pinning is not
  the binding constraint.
- **(b) DECOLLE local-readout credit** (each hidden layer a direct local target — bypasses φ'-vanishing entirely):
  probe frozen (+0.000), all 3 arms (correct / shuffled / output-only) IDENTICAL. **Verify (2-seed):** hidden FF weights
  move **30–65%** entry-wise, forward representation moves **40%**, but pooled-Fisher class-separability ≈ 0 and DROPS.
  ⇒ credit reaches the weights and moves the representation, but never toward class-discriminative features → **credit
  efficacy is NOT the bottleneck** (refutes the reservoir-artifact: weights genuinely move).
- **(c) graded-soma-V readout** (read the clean analog potential, not the quantized event rate): pooled graded-V (ho
  0.29) > pooled event (ho 0.34 train / 0.29 ho) but still ≪ INPUT — a partial recovery, not an escape; training does
  not improve it.
- **(d) denser operating point** (raise drive): E stayed pinned ~0.04 at H1/H2 across all drive configs (drive cannot
  densify the BDSP event rate) → the "denser spiking" branch is not reachable via drive.
- **(g) population size** (wider hidden × higher pool_k, 608→16,768 neurons): pooled event mlp stays 0.354
  (majority-class degenerate) at EVERY size; graded-V reaches only 0.60 (train) at the largest. ⇒ more population does
  not rescue the pooled code.
- **(h) unpooled vs pooled**: the raw code overfits (above) — not a pooling artifact.

## Verdict (per THE LAW — method(s) banked, capability OPEN, next action named)
- **The sparse-spiking forward-representability degeneracy is a CHARACTERIZED boundary of the point-neuron rate code**
  (same family as the documented Mikulasch-Priesemann whitening/decorrelation point-neuron limit): the input's
  generalizable structure (ho 0.99) is not preserved through the sparse-spiking hidden layers (ho ≤ 0.34), and it is not
  recovered by credit (a,b), readout (c), drive (d), population (g), or unpooling (h). An honest, precisely-bounded
  negative — the scientific deliverable that maps what the substrate can/can't do.
- **NOT a wall / NOT a defer.** Two forward threads remain, both named and neither dendritic:
  1. **Credit-capability thread (2026-07-22 RANK-1/2, the recommended next BUILD):** the deep-credit rule works on
     GRADED/informative-credit signals (97%); the **learned interneuron self-predicting microcircuit** (Sacramento 2018,
     `enable_bdsp_microcircuit` stub exists but its cancellation is runner-supplied, not learned in-engine) fixes
     accuracy AND the no-confab moat at once. **Test it where the forward IS representable (graded), then port.**
  2. **Spiking-forward-representability thread (the spiking-realization blocker THIS finding isolates):** a richer /
     more-reliable spiking forward CODE (spike-timing / phase coding rather than a sparse rate; or integration/
     reliability) so the forward preserves generalizable structure. This is the point-neuron rate-code limit — the
     harder, spiking-native thread; explicitly NOT more mean-pooling (fails) and NOT more population (fails).
- **NEXT ACTION:** build + test the RANK-1 learned interneuron microcircuit on a graded/informative-credit signal (the
  concrete untested build the 2026-07-22 research already ranked #1), keeping the spiking-forward-representability thread
  as the parallel spiking-realization arc.

## Process note — two verify-go saves this turn (both premature conclusions of MINE, caught by discipline)
1. **RAG-check before concluding** caught me about to re-derive "the escape is dendritic subunits" — which the
   owner-pushed 2026-07-22 4-agent finding had explicitly rejected 2 days earlier (drift #12, the #1 re-derivation
   cause). 
2. **The held-out instrument lens** caught that the unpooled train-separability (mlp 1.000) was high-dim overfitting
   (ho 0.247) before I banked "the raw code is separable, pooling is the artifact." A train-only probe in 256-dim was
   nearly unfalsifiable; held-out settled it.

## Provenance
Diagnostics (6-seed): `scratchpad/gap4_representability_verify.py`, `lever_{a..h}*.py` (+ logs). Reconciled with
`2026-07-22-gap4-real-issue-NOT-dendrites-and-timing-FIRST-CLASS-deep-research.md`. NO `sim/` edit anywhere in the arc.
The load-bearing representability probe is promoted to `research/runners/_gap4_representability_probe.py`.
