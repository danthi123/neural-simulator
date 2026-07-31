---
type: finding
status: contributing
date: 2026-06-05
mechanism: fhrr
---

# FHRR phasor pivot — de-risk (owner-chosen Option A) — 2026-06-05

Owner chose **Option A: FHRR phasor pivot, de-risk first** for the opponency SNR wall
(`2026-06-05-B-opponency-rate-coded-SNR-wall-CONFIRMED.md`). Two de-risk layers run before committing to the rework.
**Layer 1 (representation): GO.** **Layer 2 (on-the-bridge realization): the scope is much larger than a reuse —
it requires a PROTECTED `sim/` model extension (the bridge has no native FHRR substrate).** Surfaced to owner.

## De-risk 1 — does the FHRR representation clear the opponency wall + unbind at parity? GO
`research/runners/spiking_phasor_fhrr.py` self-test (the validated spiking-phasor reference, Orchard phase-sum /
phase-subtraction / phase-midpoint over a phase-timing code), the project's compositional task, frozen bar 0.80:

| load | compositional accuracy | groundable sim min | ungroundable sim max | abstention separates |
|---|---|---|---|---|
| 2 | **1.0000** | 0.597 | 0.114 | ✅ |
| 3 | **1.0000** | 0.454 | 0.114 | ✅ |
| 5 | **1.0000** | 0.303 | 0.112 | ✅ |

**Perfect accuracy at every load, clean abstention separation.** Where the ±1 ON/OFF scheme hit a hard opponency
wall at load 3–4, FHRR clears it — because in the phasor algebra there is NO common mode and NO small signed
difference (every component is unit-magnitude; the info is in PHASE). The groundable similarity DEGRADES gracefully
with load (0.60→0.45→0.30 — the SNR ≈ 2N/M dimension dial the research flagged) but stays cleanly separable. The
representation is the right one; the research's Option-A claim is validated. (This confirms the numpy reference; it
is NOT yet on the core sim — see de-risk 2.)

## De-risk 2 — can FHRR be realized ON the SimulationBridge (the owner's "no bolted-on numpy" mandate)? Needs a model extension
The production composer's bind/unbind runs ON the bridge (genuine coincidence neurons, RATE-coded). The two FHRR
references (`spiking_phasor_fhrr.py`, `resonate_fire_fhrr.py`) are explicitly NUMPY-REFERENCE, not on the bridge. To
satisfy the mandate, FHRR's phase arithmetic must run on the bridge. **The bridge lacks every natural FHRR
substrate:**

| FHRR needs | bridge has | verdict |
|---|---|---|
| per-synapse **delays** (phase-sum via delay lines) | `max_synaptic_delay_ms`/`max_delay_steps` config fields BUT **no delay buffer/queue in the step** — synapses are instantaneous (`cp_conductance_g_e += increase` same-step). The config is **vestigial**. | ✗ not implemented |
| **resonate-and-fire** complex-state neuron (Frady-Sommer phasor; the `resonate_fire_fhrr.py` model Z=V+iU) | Izhikevich / HH / AdEx — all **real-valued** spiking; no complex-state / zero-crossing-phase model | ✗ absent |
| **complex-valued synapses** (bind = complex multiply) | real-valued conductances `g_e, g_i` | ✗ absent |
| **timing/phase** code | the composer bind is **RATE**-coded (firing rate over a 150-step window) | ✗ representational shift |

The one near-fit — HH inferior-olive neurons DO have subthreshold oscillations (`g_CaT`/`g_NaP`) — is not designed
for clean phase arithmetic and is 10–20× slower (dt=0.05ms). **Conclusion: realizing FHRR on the bridge requires
EXTENDING the bridge's core model** — adding per-synapse delay support (a delay buffer in the conductance path) OR a
resonate-and-fire neuron model OR complex synapses. **Any of these is a PROTECTED `sim/` edit** — the first protected
change of the entire consolidation arc (all prior work was reuse-by-import, zero `sim/` edits), plus a rate→timing
recoding of the composer and a full re-validation of the capability matrix. This is the **biggest architectural arc**
since the conversational pipeline began — months-scale.

## The decision (surfaced to owner — material scope change)
The de-risk did its job: the representation is GO, but the on-bridge realization is far heavier than "reuse native
primitives." The honest trade-off, now visible:
- **Both DEEP shortcuts (cleanup, storage) are already CLEARED.** The composer's nonlinear core is fully spiking.
- The opponency is the **LAST linear-glue op** and is **already DISCLOSED** (the audit boundary, n=111). The
  superposition half is genuine in-network (cos 0.97); only the opponency resists rate-coded spikes.
- On-bridge FHRR would clear it (literal full clear) AND add the F=3 two-attribute resonator — but at the cost of a
  **months-scale protected bridge-model extension** (delays or resonate-and-fire) + full re-validation, when the
  higher-value spine item (the fully-grounded run) awaits.
- Honest recommendation: the FHRR representation is proven, and per-synapse delays / resonate-and-fire are both
  biologically real and generally useful bridge capabilities (not just for FHRR) — so the extension has standalone
  merit IF the owner wants to fund it. But purely to clear ONE already-disclosed linear-glue op, **Option D (accept
  the disclosed boundary, pivot to the grounded run)** is the higher cost/benefit unless the owner specifically
  values the literal full clear + the F=3 resonator. Owner's call.

## Artifacts
- `research/runners/spiking_phasor_fhrr.py` self-test (de-risk 1, GO) → `research/findings/raw/spiking_phasor_fhrr_selftest.json`
- Bridge-capability probe (de-risk 2): no delay buffer in `sim/bridge.py` step; neuron models in `sim/enums.py`
  (Izhikevich/HH/AdEx, no resonate-and-fire); composer bind rate-coded (`core_sim_composition.py`).
- NO `sim/` edits in the de-risk. CuPy/numpy as appropriate (de-risk 1 is the numpy phasor reference).
