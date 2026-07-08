# Deep-credit real-task, part 7 (role-filler binding) + the FEEDFORWARD ARC COMPLETE — single role-filler binding is NOT depth-required (depth-1: one hidden layer forms the role×filler conjunction directly); the XOR-over-pool POSITIVE CONTROL validates the harness (depth-required, microcircuit learns it 0.994). ⇒ the feedforward deep-credit depth-benefit is validated but NARROW; the genuine depth-required LANGUAGE capability is MULTI-HOP / RECURSIVE / SEQUENTIAL composition = the recurrent frontier (the deep lever's lead-orientation target).

**Date:** 2026-07-08
**Runner:** `research/runners/_rolefiller_binding_deep_credit_derisk.py` (reuse-by-import of the part-2 harness + `--positive-control`; NO `sim/` edit). Self-correcting Stage-0 gate.
**Verdict:** honest boundary (single-bind not depth-required) + a harness-validating positive control + the feedforward-arc-complete convergence.

## Stage-0 — single role-filler binding is NOT depth-required
R roles × F fillers (arbitrary codes); target = the VALUE CLASS of the (role,filler) CONJUNCTION (per-property xor/and/mux gate); held-out NOVEL combos = systematic recombination. 1-seed (14×14, bind=xor): linear 0.42 / 1-layer 0.42 / 2-layer 0.20 / deep-best 0.28, depth-gap **−0.14**; linear-probe 0.18 (< chance 0.36 = genuinely nonlinear, not a concat/score shortcut). DEPTH-SEPARATING **False**, robust across bind ∈ {xor,and,mux}, key_bits ∈ {2,3}. **The nonlinearity is a SINGLE-hidden-layer nonlinearity** — one wide hidden layer reads both disjoint code blocks jointly (universal approximation) and generalizes the bind to novel combos; depth doesn't help (overfits the arbitrary id codes). Stage-1 correctly SKIPPED.

## The POSITIVE CONTROL — the harness CAN detect depth (so the 4 negatives are trustworthy)
`--positive-control` (XOR-over-pool): DEPTH-SEPARATING **True** — deep-best **0.993** vs 1-layer **0.499**, depth-gap **+0.494**, lin-probe 0.503. Stage-1 ran clean: **microcircuit LEARNS it 0.994** vs floor 0.495; wrong-sign alignment FAILS (deep 0.01, Trap-B defeat); permuted→chance; lesion collapses; no-weight-transport True; per-layer alignment [0.20, 0.83, 1.00]. ⇒ the whole harness (depth gate + deep-credit arms + anti-cheats) works; the negatives are real boundaries, not harness failures.

## THE FEEDFORWARD DEEP-CREDIT REAL-TASK ARC — COMPLETE (the comprehensive map)
| task | depth-required? | why |
|---|---|---|
| CIFAR (perception) | NO (wrong instrument) | image depth is CONVOLUTIONAL, not FC |
| raw-PPMI (category) | NO | word embeddings make categories LINEARLY decodable |
| transitive inference (order) | NO | any MONOTONE SCALAR SCORE is transitive |
| single role-filler binding | NO | one hidden layer forms the role×filler conjunction (depth-1) |
| **XOR-over-pool (part-2 + pos-control)** | **YES — rule learns it (0.69/5-6; pos-control 0.994)** | **nonlinear conjunction over a POOLED representation (2-layer: pool→nonlinear-gate)** |

⇒ **supervised deep-credit's FEEDFORWARD depth-benefit is real but NARROW**: it needs a nonlinear conjunction over a POOLED/composed representation (2 hidden layers), which the "natural" perceptual/semantic/ordinal/single-bind tasks do NOT require (they have convolutional/linear/scalar/depth-1 shortcuts). **The genuine depth-required LANGUAGE capability is MULTI-HOP / RECURSIVE / SEQUENTIAL composition** — bind→re-bind, transitive chains that can't shortcut to a scalar score, nested structure — which requires holding + re-composing intermediate bindings over TIME. That is the RECURRENT frontier (the deep lever's lead orientation: "a simulated recurrent sequence/language cortex"), and the acknowledged unsolved-field-wide problem (deep×recurrent×spiking; the D3 target).

## What this establishes for the deep lever (the honest, comprehensive landing)
The feedforward deep-credit mechanism is validated (ports to spikes; learns the nonlinear-conjunctive-over-pool composition; the microcircuit clean; all anti-cheats). The real-task arc has COMPREHENSIVELY mapped where its depth-benefit lives (nonlinear conjunction over a pooled rep) and where it does not (the shortcut-able natural + single-hop tasks). **⇒ the genuine next frontier — the depth-required language capability — is the RECURRENT/multi-hop composition (D3), which the project already substantially explores via the EMERGE HTM sequence cortex (unsupervised, EMERGE-9/10/14) + the fronto-striatal reservoir comprehension (EMERGE-78..85).** The feedforward exploration converges on the recurrent target the deep lever was always building toward. NEXT: the recurrent multi-hop-composition de-risk — does a recurrent credit path (BPTT-SNN / e-prop / the reservoir) learn a MULTI-HOP bind→re-bind that the feedforward net provably cannot (the XOR-over-pool control is the feedforward ceiling)?

## Files
`research/runners/_rolefiller_binding_deep_credit_derisk.py`; `research/findings/raw/_rolefiller_binding_seed42_smoke.json` (boundary) + `_rolefiller_binding_poscontrol_seed42_smoke.json` (control). Arc: `2026-07-07/08-deep-credit-real-task-*.md`.
