# Gap #5 re-opening: the CA3 recurrents are NOT "functionally silent" — a direct g_e probe shows they transmit and scale with weight; the 2026-07-08 "transmission bug" was a weak-drive/weight-not-applied artifact. The real question is attractor STRENGTH, not a sim/ transmission bug.

**2026-07-17.** Owner chose gap #5 (CA3 completion / imaginative replay) as the next gap ("whichever closes quicker").
a-1 said the root cause was a MECHANICAL bug — `2026-07-08-riii-CORRECTION-ca3-recurrents-functionally-silent-not-point-neuron-limit.md`
claimed the ca3→ca3 recurrent synapses deliver ~0.2 mV (~1000× too weak) and are WEIGHT-INVARIANT (24× weight →
byte-identical Vm), i.e. a transmission/scaling bug in `sim/`. Per the silent-failure discipline (a refutation needs
the instrument verified as much as a confirmation), I read the substrate + instrumented the delivery directly BEFORE
theorizing a sim/ fix. **The claim does not survive.**

## What the code actually is (read before theorizing)
- The ca3→ca3 pathway (`text_minimal_isolation.py:1126`) is a plain-AMPA `RegionPathway` (NO `exc_receptor` →
  default AMPA; only a *plasticity* gate `ca3_swr_burst`, which freezes weight UPDATES, not current). So the
  nmda_slow AMPA-suppression path (`bridge.py:6086`) does NOT apply, and the plasticity gate does NOT block current.
- The spike matvec (`bridge.py:6200-6216`) is `effective_connections_matrix.T @ prev_fired` × `propagation_strength`
  (default **0.05**), orientation correct ([pre,post] → `.T` delivers to post). `propagation_strength` is small but
  **weight-proportional** (weight 120 → 6.0/spike, weight 5 → 0.25/spike) — so weight-invariance would require the
  recurrent to be *excluded* from the matvec, not merely weak.

## The direct g_e/Vm probe (the decisive instrument)
Build the CA3 bridge, OVERRIDE the ca3→ca3 weights in `cp_connections.data` directly (isolating current-delivery
from training), freeze plasticity, drive 24 presynaptic CA3 hard (~125 spikes), release, step once, measure the
**target** neurons' g_e + Vm change:

| set weight | n ca3→ca3 syn | driver spikes | target g_e (before→after) | g_e max Δ | Vm max Δ |
|---|---|---|---|---|---|
| 5 | 4262 | 125 | 0.167 → 0.152 | 0.026 | **1.43 mV** |
| 120 | 4262 | 132 | 9.73 → 10.86 | 4.59 | **3.66 mV** |

**The recurrents clearly transmit and scale with weight** (weight 120 delivers g_e ~10 and 3.66 mV, ~2.6× the
weight-5 depolarization). This directly **refutes** "silent / ~1000× too weak / weight-invariant."

## Reconciliation — why the 2026-07-08 finding read "silent + weight-invariant"
Two candidate artifacts, both plausible, both away from a sim/ bug:
1. **Weak drive.** The finding's direct test drove only **8 presynaptics / 18 spikes** (vs my 24 / ~125). Near the
   floor, both weight 5 and 120 give small Vm changes that look "invariant." (A first probe of MINE reproduced
   "weight-invariant = 1.50" — but that was because `_build(train=False)` HARDCODES weight 1.5 regardless of the
   `ca3w` arg (`_riii_..._derisk.py:29`); the finding's `train=True` path DID apply 4.997 vs 119.928, but its
   *direct-transmission* sub-test may have hit the same or a weak-drive regime.)
2. **The current IS weight-proportional** — so a real "silent" reading can only come from too-few presynaptic spikes,
   not from a scaling bug in delivery.

## What this means for gap #5 (honest, not overclaimed)
- Gap #5 completion is **NOT blocked by a sim/ transmission bug** — there is nothing to "fix" in the recurrent
  current delivery; it works and scales with weight. The old "the next step is a sim/-internals fix to recurrent
  current delivery" is **retired** as a mis-diagnosis.
- The REAL question is **attractor STRENGTH**: weight 120 gives only ~3.66 mV of recurrent depolarization from a
  strong drive — not enough to fire non-cue ensemble members from rest, so a *partial cue* won't complete unless the
  attractor is stronger (higher recurrent weight / density / genuine recurrent LTP during encoding). That is a
  tractable knob-sweep, not a substrate transmission wall.
- **In flight (verification + working-regime search):** a weight×density sweep of the finding's own held-out
  completion diagnostic (`_riii_ca3_completion_specificity_derisk.py`, weight 120/300/600 × density 0.30/0.50/0.60)
  to find where trained held-out completion clears the no-train control (recurrence-gain > 0.15, trained > 0.30 =
  its GO gate). If a stronger attractor GOes → gap #5 completion closes via attractor-strength, no sim/ edit. If even
  weight-600/density-0.60 stays at chance → the point-neuron attractor genuinely IS too weak in this regime and the
  dendritic-plateau completion mechanism (already 6-seed GO as a read-out) is the method — but on a HONEST basis, not
  a mis-diagnosed transmission bug.

⇒ per THE LAW, the "silent recurrents transmission bug" was a **disguised boundary / mis-diagnosis**; questioning it
with a direct instrument re-opened gap #5 toward a tractable attractor-strength path. Verification sweep pending.
