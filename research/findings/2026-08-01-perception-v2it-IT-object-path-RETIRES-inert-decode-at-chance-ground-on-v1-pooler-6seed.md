---
type: finding
status: contributing
date: 2026-08-01
mechanism: invariance-from-temporal-continuity
artifacts:
  - research/findings/raw/lanes/perception_v2it_aggregate.json
---

# Perception (lane D): the V1→V2→IT object path RETIRES 6/6 — the trained IT is INERT (held-position decode ≈ chance), grounding should use the V1→pooler codon

**One-line verdict:** the "validate-or-retire" de-risk for the ventral V1→V2→IT object-recognition path resolves
to **RETIRE on all 6 seeds**. The trained IT layer does NOT achieve position-invariant, category-discriminative
object codes: held-out-position decode is **0.245 ≈ chance 0.25** (min 0.094), it does not beat the retinotopic
V1-complex (0.260) or a *frozen* IT (0.25), and `it_fires_all_seeds` is False — the IT layer is effectively
inert. Per the gate, that retires the trained-IT object path; downstream grounding should standardize on the
V1→pooler codon, not an IT invariant code that isn't there. An honest boundary, mapped, run concurrently with the
other lanes.

Artifact: `research/findings/raw/lanes/perception_v2it_aggregate.json` (backend numpy/CPU).

## Result — 6 seeds {42,43,44,100,101,102}, chance 0.25

| read-out (mean held-out-position decode) | value | note |
|---|---|---|
| trained IT | 0.245 | ≈ chance (min 0.094) — **inert** |
| V1-complex (retinotopic) | 0.260 | IT does not beat it |
| frozen IT (untrained) | 0.250 | training adds nothing |
| IT scramble control | 0.302 | no learned structure to collapse |

`overall_verdict = RETIRE` (6/6). `it_rsa_pixels_mean = 0.0` (no pixel-RSA structure). The trained IT neither
fires reliably nor carries position-invariant category information beyond the retinotopic input.

## What this means (and the honest next)

This does NOT retire *perception* — it retires **this path to invariance** (a V1→V2→IT feedforward object code
trained by the current rule at this operating point). The invariance-from-temporal-continuity mechanism
(`2026-07-02-emerge50-trace-rule-GO`) is the validated route to invariant codes; the v2it IT path here is a
separate, weaker attempt and it does not hold. Grounding/readout should attach to the **V1→pooler codon** (which
does carry usable structure), and if an IT-level invariant code is wanted, it needs the trace-rule / temporal-
continuity mechanism driving it, not this feedforward STDP path. Named next: wire grounding to the V1→pooler
codon; revisit an IT invariant code only via the trace rule. A mapped boundary with the next mechanism named — no
capability abandoned.
