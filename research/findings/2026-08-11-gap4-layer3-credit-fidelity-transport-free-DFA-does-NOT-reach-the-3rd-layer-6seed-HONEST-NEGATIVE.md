---
type: finding
status: contributing
date: 2026-08-11
mechanism: deep-credit-on-spikes — LAYER-3 CREDIT FIDELITY (does transport-free DFA e-prop error reach the 3rd hidden layer on a genuinely depth-3 target)
lane: gap#4 / deep-credit
verdict: where TESTABLE (the BP depth-3 ceiling holds, e.g. seed 42), transport-free DFA e-prop does NOT reach the 3rd hidden layer (clean single-seed HONEST NEGATIVE); but the tent^3/width-8 ceiling is SEED-FRAGILE so the 6-seed AGGREGATE is UNDEFINED — the instrument needs a more robustly-fittable depth-3-engaging target
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_gap4_layer3_credit_fidelity_6seed.json
  - research/findings/raw/_gap4_layer3_credit_fidelity_smoke42.json
runner: research/runners/_gap4_layer3_credit_fidelity_derisk.py
instrument: the ACHIEVABLE gap#4 test named by the crux-reframe (`2026-08-11-gap4-depth3-obligatory-task-is-provably-impossible...`). Train a 3-hidden-layer DendriticMLP([n_in,8,8,8,1]) to FIT a depth-3-composed target (tent^3 regression, MSE) and ask whether transport-free DFA e-prop drives the fit toward the backprop-depth-3 oracle. Arms: BP-depth-3 (ceiling), BP-depth-2 (depth-separation), DFA (transport-free), permuted-target, B=0-lesion. SIM_BACKEND=numpy.
---

# gap#4 LAYER-3 CREDIT FIDELITY — where testable, transport-free DFA e-prop does NOT reach the 3rd hidden layer; but the tent^3/width-8 ceiling is seed-fragile (6-seed UNDEFINED)

The crux reframe (`2026-08-11-gap4-depth3-obligatory-task-is-provably-impossible-reframe-to-layer3-credit-fidelity.md`)
established that gap#4 deep-credit must be tested as LAYER-3 CREDIT FIDELITY: does transport-free error reach the 3rd
hidden layer, on a target that provably FITS only with layer 3. This de-risk builds that test. Two honest results.

## Result 1 — where the ceiling HOLDS (seed 42): a clean HONEST NEGATIVE (`research/findings/raw/_gap4_layer3_credit_fidelity_smoke42.json`)

<!--derived-->
On seed 42 the BP depth-3 oracle robustly fits (loss 0.00018 ≪ target var), while BP depth-2 does NOT (0.0419) — so the
target is genuinely depth-3-ENGAGING there. On that valid instrument, **transport-free DFA e-prop does NOT fit: loss
0.0395, sitting at the SAME mean-predictor / BP-depth-2 floor as its null controls** (permuted-target and B=0-lesion also
~0.042). DFA is indistinguishable from "no learning"; it closes ~0% of the BP2→BP3 fit gap. **NO-GO (honest negative):
without weight transport, error does NOT reach the 3rd hidden layer** — the literal gap#4 answer where it is testable.

## Result 2 — the instrument caveat: the 6-seed AGGREGATE is UNDEFINED, honestly (`_gap4_layer3_credit_fidelity_6seed.json`)

<!--derived-->
Across seeds 42/43/44/100/101/102 the BP depth-3 ceiling is SEED-FRAGILE: mean bp3 loss 0.0241 (vs 0.00018 on seed 42) —
backprop-depth-3 fits the (highly oscillatory) tent^3 target at width 8 on only ~1/6 seeds. The runner's
`backprop_oracle_ceiling_exists` precondition therefore FAILS on the aggregate, and it correctly returns **UNDEFINED, NOT
a negative** — you cannot conclude "DFA can't reach layer 3" on seeds where the ORACLE itself can't fit. (This is the
verdict-preconditions discipline working: a failed precondition yields UNDEFINED, never a fabricated negative.)

## The instrument correction banked here (why the ALIGNMENT metric was invalid, and the FIT is the signal)

<!--derived-->
The build initially gated on the layer-3 (a3, output-adjacent) DFA-vs-BP cosine ALIGNMENT — invalid, because a3 is the
output-adjacent hidden layer, so its alignment is **output-layer feedback-alignment (W₃↔B₃): TARGET-INDEPENDENT** (the
permuted-target run also shows a3 alignment ~1.0). The verdict was corrected to gate on the **FIT** (does the arm learn
the target below the BP-depth-2 / mean-predictor floor), with a3 alignment demoted to REPORTED and the deepest-from-output
a1 alignment reported (unreliable at the stall). "The instrument is part of the emulation": a mechanism you cannot
measure correctly you tune in the wrong direction.

## Scope / honesty + the two named next steps (per THE LAW — the capability stays OPEN)

<!--derived-->
NO-EXTERNAL-NEEDED: the seed-42 negative reproduces the KNOWN failure of Direct Feedback Alignment on DEEP layers
(fixed-random feedback aligns the OUTPUT layer but does not propagate to reach early hidden layers). The surpasses are
standard + already in the gap#4 ledger, so no new external read is required.

- **Science next-mechanism (the capability):** the feedback must be LEARNED to align with the forward pathway so error
  reaches deep layers — **weight-mirror / Kolen-Pollack (PAL-KP) learned feedback**, or the **φ′-vanishing fix**
  (per-layer gain / activation). DFA (fixed feedback) is measured insufficient for depth-3 reach on the FIT, where testable.
- **Instrument next-step (to make the 6-seed decisive):** a more robustly-fittable depth-3-ENGAGING target so the BP
  depth-3 ceiling holds on every seed while BP depth-2 still underfits — e.g. a slightly wider net with a target tuned to
  stay depth-2-hard, or the quasigroup-depth-3 (train-fit) alternative, or per-seed ceiling-gated scoring (score only the
  seeds where the oracle fits). Then the seed-42 negative becomes a clean 6-seed negative.
- **Data-only, NO `sim/` edit.** Provenance: the build agent stalled before committing; the coordinator RECOVERED the
  uncommitted runner + smoke, FIXED the verdict scoping (a3-alignment gate → FIT gate, per the agent's own in-transcript
  diagnosis), and ran the 6-seed — which honestly surfaced the ceiling-fragility the single-seed smoke had hidden.
