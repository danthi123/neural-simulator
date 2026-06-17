# Fixed-role + learned-filler bundling — capacity localization = RE-OPEN the on-bridge binding build

**Date:** 2026-06-17 (CYCLE 146/149 — the localization that closed the deferred-build loop)
**Status:** **RE_OPEN_BUILD.** The fixed-role + learned-filler bundling lifts monotonically to **full parity**
with the fixed-algebra ceiling as the bind-space dimension grows — the 0.603 the dendritic build was deferred on
(at D_h=64) is a **capacity artifact, not a ceiling.** 6 seeds, numpy/CPU, reuse-by-import, NO `sim/` edit.
**Runner:** `research/runners/_phaseB_frlf_capacity_sweep.py`
**Raw:** `research/findings/raw/_phaseB_frlf_capacity_sweep.json`

## Why this sweep

The 6-seed A/B (`2026-06-17-fixed-role-learned-filler-bundling-derisk.md`) resolved GO on the science (a fixed
self-inverse role + LEARNED filler codes recovers bundled multi-attribute facts where a learned *linear* inverse
cannot), but the learned-filler version landed at **0.603** — well below the fully-fixed FHRR algebra's **0.993**.
The build call was DEFER, with the pre-registered BOUNDARY clause: *"the LEARNED fillers cost accuracy vs the
fully-fixed algebra; localize (more capacity / a multiplicative cleanup) before committing the build."* This sweep
is that localization: hold everything fixed and vary only the bind-space dimension D_h ∈ {64, 128, 256},
measuring the FR+LF bundled held-out recall against the matched ceiling at each D_h (the exact same eval path,
zero A/B drift).

## Result — 6 seeds (42, 43, 44, 100, 101, 102)

| bind-space D_h | FR+LF bundled held-out | ceiling | gap to ceiling | single held-out | near-parity |
|---|---|---|---|---|---|
| 64 (the A/B point) | 0.603 | 0.993 | +0.390 | 0.806 | 0/6 |
| 128 | 0.832 | 1.000 | +0.168 | 0.889 | 3/6 |
| **256** | **0.988** | 1.000 | **+0.012** | **1.000** | **6/6** |

D_h=256 per-seed: [1.00, 1.00, 1.00, 1.00, 0.93, 1.00]. chance 0.062.

## Reading it

- **Monotonic lift to full parity.** The gap to the fixed-algebra ceiling collapses +0.390 → +0.168 → +0.012 as
  D_h doubles, and every seed reaches near-parity at D_h=256 (6/6). The learned-filler bundling is **not** capped
  below the fixed algebra — it simply needed enough bind-space to hold the 3-way superposition of learned codes.
- **The deferred build RE-OPENS as justified.** The reason for the DEFER (a real-but-modest lever ~0.39 below the
  ceiling) is removed: at the working dimension the learned binder matches the fixed algebra (0.988 vs 1.000)
  while generalizing systematically (single held-out 1.000). The learned representations (codes + read-out) carry
  the bundling, which is exactly the idealization-removing piece.
- **Single-binding generalization tracks too** (0.806 → 1.000), so this is not memorization at higher capacity —
  the held-out (role, filler) combinations recover as well as the trained ones.

## Consequence (what this licensed)

This RE-OPEN, together with the on-bridge Step-1 result (`2026-06-17-onbridge-learned-filler-binding-step1-GO.md`,
the bundling survives real LIF spiking at 0.969 = 98% of numpy) and Step-2
(`_phaseB_onbridge_learned_composer_derisk`, the learned spiking binder does who/what Q&A + the no-confab moat),
makes the owner-greenlit on-bridge binding build a *wiring* task at the parity dimension rather than a
mechanism-invention task. Honest scope: parity is reached by capacity (D_h=256), and the on-substrate realization
uses the existing ON/OFF population substrate (the fixed ±1 role makes binding a linear channel-swap; the
population rate-code carries the superposition), so the `fused_coincidence_plateau` dendritic-multiplication
primitive is not required for this binding.

## Reproduce
```bash
SIM_BACKEND=numpy python -u -m research.runners._phaseB_frlf_capacity_sweep \
    --seeds 42,43,44,100,101,102 --dims 64,128,256
```
