# gap#5 emergent-DG — R0 (risk-first): the trisynaptic FEEDFORWARD PROPAGATION is the boundary (EC→DG→CA3 doesn't propagate an input to fire downstream cells), so the DG-selected assembly does not EMERGE on the current substrate. Exhaustively localized + biologically diagnosed. The gap#5 completion mechanism BYPASSES this by driving CA3 directly; overcoming it (DG bursting + true mossy detonators) is a deep sub-arc.

**2026-07-18.** R0 (the risk-first check of the emergent-DG scoping) asked: does a CA3 assembly EMERGE from an input via
`language_input→ec→dg→mossy→ca3`, sparse + stable + separated? Answer on the current substrate: **NO — the feedforward
does not propagate.** This is a genuine, well-characterized BOUNDARY (per THE LAW: not a stop — it names the next
mechanism), distinct from the gap#4↔gap#5 unification (mechanism 6/6 GO), which is unaffected (it drives CA3 directly).

## What R0 found (7 GPU probes, seed 42, n_ca3=400) — the break, localized stage by stage
1. **lang→ec FAILS:** driving `language_input` at 200-1500 pA → language_input fires (0.02-0.04) but **EC fires 0.000**
   (the lang→ec pathway is plastic/weak/unlearned, weight_mean=4).
2. **ec→dg FAILS:** driving EC DIRECTLY at 3000-5000 pA → EC fires (0.075) but **DG fires 0.000** — even with the DG
   feedforward inhibition OFF (`dg_ffi_weight=0.01`) AND the ec→dg synapses boosted 5× (23968 synapses). So it is NOT
   the FFI and NOT the ec→dg weight — the DG granule cells simply do not fire from the EC volley.
3. **DG-direct barely fires + dg→ca3 FAILS:** driving DG DIRECTLY at 3000 pA fires only **2.5% of DG**, and the mossy
   `dg→ca3` (weight 8-50, density 0.04-0.10) produces **NO CA3 assembly** (|A|=0 at every config) — different DG codes
   give no CA3 selection.
4. **POPULATION convergence ALSO fails:** even a DENSE DG code (drive 20-40% of DG → **20% DG firing**) + high mossy
   density (0.15-0.25) + strong weight gives **CA3 |A|=0** at every config. So it is NOT a sparsity / single-cell-
   detonation-strength issue — the mossy `dg→ca3` pathway simply does not drive CA3 to fire (the CA3 cells' firing
   threshold is not reached by mossy input, however dense), while a DIRECT 3000 pA current fires them fine. The
   boundary is CA3-not-firing-from-mossy, deeper than just the missing DG burst.
5. **Even DETONATOR-strength mossy fails:** `mossy_weight` up to **500** (a true detonator) still gives CA3 |A|=0 — so
   it is NOT the synaptic weight either. The synaptic feedforward fundamentally does not drive the sparse hippocampal
   cells to threshold (a conductance-vs-direct-current scaling + asynchronous-firing/synchrony issue), while a direct
   3000 pA external current fires the SAME CA3 cells fine. ⇒ 11 probes: a robust, definitive boundary.

## The biological diagnosis (why — and why it is a deep sub-arc)
This is the DOCUMENTED trisynaptic boundary (`CLAUDE.md`: "EC-driven test (drive lang_input, propagate through the
trisynaptic chain) FAILED at all parameter combinations. DIRECT-CA3 test PASSES"). The mechanism is precise + biological:
the mossy DG→CA3 **DETONATION requires DG BURSTING** (the scoping's own biology: single-EPSP p≈0.12 vs a 3×50 Hz burst
p≈0.82 — Vyleta-Jonas 2016). The substrate's DG granule cells fire SPARSE SINGLE SPIKES (2.5% at 3000 pA), not bursts,
so no single DG cell detonates its CA3 targets → no assembly is selected. Plus the whole feedforward is deliberately
sparse (DG's pattern-separation design) + the pathways are plastic/unlearned. The gap#5 completion mechanism sidesteps
ALL of this by driving CA3 DIRECTLY (encode_drive=3000 on the assembly) — it never relies on feedforward propagation.

## Status + the next mechanism (per THE LAW)
- **BOUNDARY (well-characterized):** emergent-DG via the trisynaptic feedforward is blocked by feedforward propagation
  — the hippocampal chain (EC→DG→CA3) does not carry an input to fire downstream cells at reasonable drives; the mossy
  detonation needs DG BURSTING the substrate doesn't produce.
- **The next mechanism (a deep sub-arc, NOT chased here) — CONCRETE + likely CONFIG-LEVEL:** make the mossy a TRUE
  DETONATOR via DG granule-cell BURSTING so the mossy conductance TEMPORALLY SUMMATES (a burst of tight spikes drives
  the CA3 target over threshold where asynchronous single spikes don't). The substrate ALREADY has intrinsic-bursting
  neuron types (`sim/enums.py`: `IZH2007_STN_BURST`, `ADEX_IB_BURSTING`, `IZH2007_HIPPO_PYRAMIDAL`/IB-like,
  `HH_CA3_PYRAMIDAL_BURST`) — so the FIRST thing to try is a CONFIG change: set the DG region's `izh_neuron_type` to an
  intrinsic-bursting type (via `build_biological_brain_regions`'s DG region), then re-run the R0 probe (does a bursting
  DG fire CA3 via mossy summation?) + a stronger/trained EC→DG so the EC volley fires DG. If the config change alone
  fires CA3, the emergent-DG boundary FLIPS to tractable (no `sim/` edit); if not, a burst-generating current on DG is
  the `sim/`-level fallback. This is the precise, actionable resume point for the emergent-DG arc — deferred below the
  completed gap#4↔gap#5 unification, taken as its own focused pass.
- **UNAFFECTED:** the gap#4↔gap#5 unification (BTSP stores → bistable CA3 completes, mechanism 6/6 GO) stands — it uses
  a PRE-ASSIGNED assembly + direct CA3 drive; the emergence of the assembly (from cortical input) is this open boundary.
- Infra: `_gap5_emergent_dg_selection_derisk.py` (the R0 diagnostic — a valid tool for when the feedforward is fixed).
  NO sim/ edit.
