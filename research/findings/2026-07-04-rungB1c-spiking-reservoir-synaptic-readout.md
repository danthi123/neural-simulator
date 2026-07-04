# RUNG B-1c — spiking reservoir co-resident (c1 **GO**); the full synaptic read-out (c2) hits a sub-1% margin-resolution **BOUNDARY**

**Date:** 2026-07-04
**Runner:** `research/runners/_rungB1c_spiking_reservoir_synaptic_readout_derisk.py` (`--mode c1|c2`)
**Test:** `tests/test_rungB1c_spiking_reservoir_synaptic_readout.py`
**Raw:** `research/findings/raw/_rungB1c_c1_3seed.json` (GO 3/3), `_rungB1c_c2_3seed.json` (PARTIAL 1/3), `_rungB1c_c2_seed42.json` (GO)

## Why (the LAST two host shortcuts in role selection)

RUNG B-1b removed the host `argmax`, but two host computations remained: the reservoir feature `f` (a host RATE reservoir)
and the read-out `f @ Ws[k]` (a host matmul). RUNG B-1c removes them: the reservoir becomes SPIKING and co-resident on the
one bridge (c1), and `Ws` becomes real reservoir→ensemble SYNAPSES (c2), so the whole comprehend→select→bind turn runs on
ONE `UnifiedBrainBridge` with nothing load-bearing host-computed.

## c1 — the spiking reservoir is co-resident on the bridge — **GO** (3/3)

A recurrent Izhikevich liquid-state machine (300 neurons, the EMERGE-82 statistics; fixed-random recurrence + `W_in`
input, 20% inhibitory subset via `cp_traits=1`) is allocated as a slice on the `UnifiedBrainBridge` (the additive
`reservoir_n` param — default 0, verified byte-identical: B-1/B-1b fast tests still 8/8) and wired runner-side
(`set_pathway_weights(add_missing=True)`). It replaces the host RATE reservoir; the read-out is still host `f@Ws` → the
B-1b WTA. **3/3 GO** (seeds 42/43/44): route 12/12 each, all nine B-1b anti-cheats hold on the co-resident spiking
substrate. ⇒ the host RATE reservoir is removed; comprehension is now a spiking LSM on the one bridge.

## c2 — the full synaptic read-out (`Ws'` synapses, NO host `f@Ws`) — **BOUNDARY** (GO 1/3)

`Ws_shifted = Ws − Ws.min()` (Dale-legal, purely excitatory) is wired as reservoir→ensemble synapses (per content slot),
replacing the host `f@Ws` drive: the WTA ensembles are driven SYNAPTICALLY by the reservoir's firing. On **seed 42 this
GOes** — the whole turn runs synaptically on one bridge (route 10/12 ≥0.8n, synaptic-readout-lesion collapses 0<10,
route/res-lesion collapse, Ws-scramble collapses, source-check clean). But **seeds 43/44 fail** — an honest, precisely
located boundary.

**The boundary (a real finding):** no single `Ws_shifted` scale gives BOTH route == host-dict recall (12/12) AND a
load-bearing reservoir-lesion, robustly across seeds. Two coupled causes:
1. **Sub-1% margin resolution.** After the Dale shift the winner beats the runner-up by only ~0.3–1.4% of total drive; the
   spiking read-out resolves this only with enough ensemble size + integration to average the Izhikevich/OU noise. This
   integration used the B-1b **P=20 / T=12 / replay-3** regime — the exact regime the B-1c CRUX de-risk found INVERTS the
   top-2 (the crux needed **P=80 / T=30** for 6/6). So the boundary is very likely UNDER-RESOLUTION, not a wall.
2. **The per-role bias intercept prior.** The ridge `Ws` has a per-role bias row that encodes each slot's role PRIOR;
   implemented as a lesion-immune per-ensemble tonic, it carries the canonical AGENT/PREDICATE slots even when the
   reservoir is lesioned → the reservoir is genuinely load-bearing only for the patient slot on some draws (seed 42 yes,
   43 no). On seed 44 the feature + margin degrade until the synaptic route recovers 0/12 (host-dict itself only 8/12).

## Honest findings (self-caught, no faking)

- **Hebbian + OU must be toggled OFF during the reservoir read.** The unified bridge runs global Hebbian ON + OU noise; a
  fixed-random LSM must not learn (with Hebbian ON the recurrence drifts, feature discrimination 1.000 → 0.14). Toggling
  both off for the self-contained read window (mirroring `elaborate`'s dlPFC OU toggle) restores 1.000 — legitimate.
- **The `Ws` bias row is PER-ROLE, not a role-independent constant** (a correction to the crux, which only tested slot-0):
  dropping it breaks the argmax on the AGENT/PREDICATE slots; carrying `Ws_shifted[bias, r]` as a per-role tonic fixes it.
- **In c2 the WTA mutual-inhibition is no longer load-bearing** — the selection genuinely moved from inhibition-competition
  to the synaptic read-out (so the B-1b WTA-lesion anti-cheat is superseded by the syn-readout-lesion, which DOES collapse).

## The surpass (boundary = undiscovered mechanism — IN FLIGHT)

The boundary is precisely located, so it is a lead: apply the crux's validated resolution (**P=80 role ensembles + longer
integration**, not B-1b's P=20/T=12) to the co-located read-out, and make the reservoir load-bearing for ALL slots (the
per-role bias intercept must not be a lesion-immune tonic — route it so a reservoir-lesion degrades every slot, or use a
non-canonical test set where position cannot substitute for the reservoir's structural read). Then the sub-1% margin is
resolved robustly and the whole turn closes out on one substrate.

## Files
- `research/runners/_rungB1c_spiking_reservoir_synaptic_readout_derisk.py` — c1/c2 modes.
- `tests/test_rungB1c_spiking_reservoir_synaptic_readout.py` — 5 fast + c1-GO/c2-seed42 slow gates.
- `research/runners/unified_brain_bridge.py` — the additive `reservoir_n` param (default-off, byte-identical).
