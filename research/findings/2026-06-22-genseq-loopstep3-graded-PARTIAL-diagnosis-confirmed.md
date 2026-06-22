# Loop-step 3 de-risk #2 (graded readout) — Phase A CONFIRMS the diagnosis: the rate-saturation was a READOUT artifact; graded un-saturates + recovers per-layer fidelity (cumulative needs population coding) (2026-06-22)

**Scope:** the #1 resolution of the loop-step-3 rate-saturation NEGATIVE — read the GRADED membrane (`a_cont`)
instead of the saturated spike rate. `research/runners/_genseq_loopstep3_graded_derisk.py`, GPU. **NO `sim/` edit.**
Phase A (graded + per-block gain calibration + scale sweep) was DECISIVE; Phase B (population coding) OOM'd at
n_per=8 (~83K neurons / ~800M synapses on the 24 GB card) and stalled — re-run at a feasible scale is the next step.
On `main`.

## Phase A result — the diagnosis is CONFIRMED, with a partial lift
| readout | per-block Spearman | cumulative | `a_cont` saturated? |
|---|---|---|---|
| de-risk #1 (SPIKE-RATE) | [0.321, −0.019, 0.009] | 0.009 | (spike-rate pinned at 0.5) |
| **de-risk #2 (GRADED, best scale=20)** | **[0.865, 0.596, 0.327]** | **0.327** | **False — NO saturation** |

(scale sweep: 20→cum 0.327, 40→0.027, 80→−0.196; best at 20.)

⇒ The rate-saturation wall was indeed a **READOUT artifact**. Reading the graded membrane `a_cont`:
- **un-saturates** — `a_cont_sat=False` at every scale → the `[VERIFY]` open question is answered: the graded path
  does NOT re-saturate under a dense transformer layer's full fan-in;
- **recovers layer-0 from 0.32 → 0.865** (near step-0's 0.92 — the spike-rate readout had been destroying even the
  first layer);
- **carries real signal through the dense hidden layer** (layer-1 0.596 vs the spike-rate readout's −0.019 chance).

## The remaining gap (per-layer error accumulation, NOT saturation)
Cumulative fidelity degrades across the 3 layers (0.865 → 0.596 → 0.327) — below the GO bar (≥0.8). This is per-layer
error accumulation, not saturation. The documented fix is **POPULATION coding** (the project's prior rate-code-wall
lift: single-neuron 47% → n_per=8 100% → n_per=32 108% of host, `2026-06-15-…GO.md`). Phase B escalated to it but
OOM'd at n_per=8 — population coding multiplies the bridge size (n_per² synapse blow-up at full fan-in).

## Verdict — PARTIAL (diagnosis confirmed; graded works; cumulative needs pop-coding at a feasible scale)
The rate-saturation wall IS a readout artifact, surpassable with the in-bridge graded path (NO `sim/` edit, the
single most important finding). The multi-layer cumulative (0.327) needs the documented population-coding lift,
which must be re-run at an OOM-safe scale.

## Next
Re-run de-risk #2 Phase B (population coding) at a FEASIBLE scale — n_per=4 (not 8), a NARROWER hidden width, or
fewer layers, to avoid the n_per² synapse blow-up — and measure whether pop-coding lifts the cumulative to ≥0.8.
If yes, the multi-layer consolidation is surpassed (graded + pop-coding, no `sim/` edit) → resume the loop-step-3
ladder (attention #2 → full forward #3) → C2. The spiking-convert + P2-knowledge GO stand regardless.
