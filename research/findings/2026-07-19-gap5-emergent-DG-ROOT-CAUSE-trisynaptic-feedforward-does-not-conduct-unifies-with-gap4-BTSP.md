# gap#5 (ii) emergent-DG — ROOT CAUSE (read-your-substrate): the trisynaptic FEEDFORWARD chain does NOT conduct a volley; the "amplification" framing chased the wrong problem; the real fix UNIFIES with gap#4 one-shot BTSP

**2026-07-19.** Instrumented the emergent-DG runner's 0-firing directly (per the read-your-substrate discipline —
`feedback_read_own_substrate_before_theorizing`) instead of sweeping amplification operating points. The instrumentation
localized the blocker precisely, and it is NOT amplification.

## What I measured (seed 42, `_gap5_emergent_dg_selection_derisk` default build, GPU)
Traced firing at every stage of the trisynaptic loop `language_input → ec → dg → ca3` under a driven input, and read the
CA3 conductances:

1. **Every feedforward hop fails to conduct** (drive each stage directly, count cells that fire at all over 50 steps):
   - `language_input` fires 38/384 (the driven pattern) — ✓
   - drive EC directly → EC fires 30/200 but **DG = 0/300** (`ec→dg` perforant w=6 does not propagate)
   - drive DG directly → DG fires 45/300 but **CA3 = 0/400** (`dg→ca3` mossy w=8 does not propagate)
   - drive `language_input` (the real path) → EC = 0 (the FIRST hop `lang→ec` w=4 already dies: EC g_e peaks ~0.8
     while EC g_i ~1.1, so EC v crawls −60→−58 mV and never nears its −42 mV threshold).
2. **The mossy→CA3 conductance is ~10–30× too weak.** CA3 g_e scales LINEARLY with mossy weight and is INVARIANT to
   feedback inhibition (so inhibition is NOT the cause): g_e peak 0.17 (w=8) → 0.53 (w=25) → 1.06 (w=50) → 2.13 (w=100)
   → 4.25 (w=200) → 7.63 (w=400); across ca3_fb_inhib ∈ {2, 5, 20} the g_e is byte-identical. CA3 needs g_e ~5+ to
   cross its 22 mV rest→threshold gap.
3. **Strong mossy fires cells but NO assembly latches.** As mossy weight rises, "ever-fires" grows 1→17→114→178/400
   (reaching the finding's "~43% distributed code" at w=400 = a DENSE TRANSIENT activation) — but the SUSTAINED assembly
   (cells firing ≥30% of the settled window) is **0 at every weight**. The bistability keystone (`plateau_self_regen=0.15,
   apical_kir_g=3.0`) IS enabled in amplify mode, yet nothing latches.

## Why the loop was never exposed as broken
Every prior CA3 result (SWR completion, coincidence completion, the whole completion arc) drove **CA3 directly** (a
partial of a stored ensemble → the trained recurrent completes it). The trisynaptic FEEDFORWARD weights + the sparsifying
DG feedforward inhibition (dg_ffi) were tuned for **sparsity**, never for **volley propagation**, so the feedforward
break stayed invisible until this instrumentation drove the loop end-to-end.

## The precise reframe (two problems, not one; amplification was neither)
- **Problem A — feedforward conduction:** the mossy detonator (biologically the strongest synapse in the brain) is here
  ~10–30× too weak. Fixable by strong feedforward weights (w~100+ fires a set), but then it is DENSE (needs the E%-max
  `ca3_ff_inhib` to sparsify — the layer-3 build I already committed).
- **Problem B — no self-sustaining assembly (the deeper one):** with plasticity OFF (the read pass sets
  `enable_hebbian_learning=False`), NO recurrent attractor can form, so a driven set fires TRANSIENTLY and decays. The
  bistable latch does not engage because `plateau_self_regen`/`apical_kir` are triggered by the APICAL/recurrent
  compartment, and a NOVEL assembly with random recurrent (ca3w=4, density=0.05) has no coincident recurrent drive to
  trigger the plateau — a chicken-and-egg (latch needs recurrent coincidence needs an assembly needs the latch).

## ⇒ the mechanism (per THE LAW — the boundary names the next build), and it UNIFIES gap#5 with gap#4
The emergent-DG needs an assembly to **form AND self-sustain in one shot from a novel input**. The biological mechanism
for exactly this is **one-shot BTSP** (Bittner-Magee behavioral-timescale plasticity — how hippocampal place fields form
in a single traversal): a plateau-gated one-shot potentiation of the RECURRENT among the co-active set instantly builds
the attractor, so the set self-sustains after the mossy volley is removed. This is the SAME mechanism as gap#4 (the
plateau-gated BTSP keystone). ⇒ the emergent-DG (gap#5 ii) and the dendritic-credit keystone (gap#4) are ONE problem:
**one-shot plateau-gated recurrent potentiation.** The full emergent-DG recipe is now precisely grounded:
1. strong feedforward mossy (fire a sparse set from the DG volley) — Problem A,
2. E%-max feedforward inhibition (`ca3_ff_inhib`, already built) to hold it sparse (~5%),
3. one-shot BTSP (plasticity ON) to potentiate the recurrent among the co-active set → a self-sustaining attractor —
   Problem B = the gap#4 mechanism.

## Status
- **Honest reframe recorded.** The 3-layer "amplification" framing (2026-07-18) was chasing the wrong variable — the
  input never reached CA3. This instrumentation (≈20 min) found in the substrate what the operating-point sweeps missed,
  exactly the read-your-substrate lesson.
- **NEXT build (GPU, unifies with gap#4):** drive a sparse CA3 set via strong mossy WITH plasticity ON → one-shot BTSP
  the recurrent among the co-active set → remove drive → test self-sustain (Jaccard of the re-presented assembly) +
  separation across inputs + the mossy-lesion/permute anti-cheats. This is the gap#4 BTSP mechanism applied to the CA3
  recurrent — the two gap items close together.
- Diagnostics: `scratchpad/emergdg_{diag,chain,ec,stagedrive,mossy}.py` (read-your-substrate instrumentation). NO sim/ edit.
