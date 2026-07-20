# gap#4 RUNG 7b — PRE-REGISTRATION: weight-dependent BTSP on a VALID instrument (filed BEFORE the run)

**Filed 2026-07-20 before any rung-7b result exists.** Seeds **1600-1605**, never used.

## What changed since rung 7, and why this is a repair not a re-tune

Rung 7 was **invalid**: a single global `btsp_w_max = 300` (chosen for the layer-2 pathway at weight ~150) inflated
the layer-1 pathway (natural weight ~0.6) by **92x**, destroying field formation before the rule could be assessed.
Two pathways, natural scales differing **250x**, one bound.

The repair is a **per-synapse `cp_btsp_wmax`**, exactly mirroring the per-synapse `cp_btsp_theta` this codebase
already carries for the identical reason (commit `a5a5e341`: *"one global theta genuinely cannot serve both"*,
measured ratio 27.4x). `None` => the cfg scalar => byte-identical, asserted end-to-end.

**No rule parameter changed.** `alpha_pot = 0.24`, `alpha_dep = 0.09` remain Milstein's published values;
`k_pot = k_dep = 0.02` remains unchanged, so the fixed point is still set purely by the published sigmoid ratio.
The only change is that each pathway is now bounded on its own scale.

## PF-6 — the fix verified BEFORE pre-registering

| | rung 7 (one global bound) | rung 7b (per-pathway) |
|---|---|---|
| distinct bounds present | 300 only | **[5.0, 300.0]** |
| layer-1 weight inflation | **92x** (0.6002 -> 55.2791) | **2.0x** (0.6002 -> 1.2232) |

Layer-1 weights now stay on their own scale.

## PRE-REGISTERED PREDICTIONS (unchanged from rung 7)

0. **P0 — stage 1 forms:** `map_ok = 1` on >= 5/6. *(This is what rung 7 failed, for the configuration reason above.)*
1. **P1 — adjacent contrast (THE GOAL):** >= **1.60x**, on >= 5/6.
2. **P2 — far contrast retained:** >= 2.0x, on >= 5/6.
3. **P3 — rule is load-bearing:** `k_dep = 0` control reproduces ~1.213x / 2.609x, 6/6.
4. **P4 — no floor pinning:** < 5% of synapses at `w_min`, 6/6.

**FALSIFIED if P1 fails.** With the instrument now valid, a P1 failure means weight-dependent bidirectional
plasticity — the mechanism biology actually uses, whose fixed point is confirmed on deployed traces (PF-5) and
whose thresholds are verified against 8.5M deployed samples — **does not deliver neighbour-contrast in this task.**
That would be a real negative about the task, not another mis-implementation.

## Cap — now genuinely binding

**One run.** The instrument is repaired, the parameters are published, and there is no remaining free knob I could
honestly adjust. If P1 fails I do **not** touch `k`, the thresholds, or the bounds. The next question becomes
whether the task's evenly-spaced field layout — which the literature says has **no empirical basis** (real spacing
is Poisson with a modal gap of zero) — is itself generating the deficit. That runner option is already implemented
(`--poisson-cells`) and would be the successor experiment.
