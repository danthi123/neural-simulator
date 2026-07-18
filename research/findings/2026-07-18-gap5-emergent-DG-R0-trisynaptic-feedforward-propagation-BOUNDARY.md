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

## REFRAME (the richer picture from the membrane + synchrony probes — the actionable path)
Two later probes change the picture from "hard boundary" to "a tractable INTEGRATED select-and-store":
- **The mossy DOES reach CA3:** at `mossy_weight`=500 some CA3 cells cross threshold (v_max −31.7 > the −40 threshold),
  though v_mean stays −65. So the conductance arrives; the problem is that ASYNCHRONOUS sparse DG firing gives each CA3
  cell only transient, non-coincident input.
- **Synchronizing the DG volley makes CA3 INPUT-SPECIFIC:** a GAMMA-pulsed DG drive (2-3 on / 2-4 off) raises the
  CA3-rate-vector separation `sep_cos` from 0.00 → **0.53** (distinct inputs → distinct CA3 responses) — the mossy IS
  selecting an input-specific CA3 seed. It still does not SUSTAIN firing (|A|=0 at ≥0.15) because the CA3 recurrent is
  weak here (ca3w=1.5, no attractor amplification).
- ⇒ **the emergent-DG is an INTEGRATED select-and-store, not select-THEN-store:** the synchronized mossy SEEDS
  input-specific CA3 co-activity; the assembly SUSTAINS only once the CA3 recurrent exists — which is exactly what BTSP
  BUILDS. So the tractable path is to run the mossy-seeding + the BTSP store TOGETHER (on the first synchronized
  presentation the mossy seed co-fires CA3 → BTSP stores it → the recurrent grows → the assembly emerges + completes),
  at the completion-scale config (n_ca3=2000, the gap#5 recall machinery). R0's strict "sustained |A_m| before any
  storing" metric was too strict for the SEED. **Resume: build the integrated select-and-store** — a synchronized DG
  volley seeds CA3 co-activity, `encode_btsp` stores it (the assembly = the mossy-seeded co-firing cells, read live),
  the bistable CA3 completes; anti-cheats = input-driven (permute input → different assembly), pattern-separation
  (sep_cos < 0.4 across inputs), + the completion nocue/perm/no-encode. This reuses everything already built.

## Status + the next mechanism (per THE LAW)
- **BOUNDARY (well-characterized):** emergent-DG via the trisynaptic feedforward is blocked by feedforward propagation
  — the hippocampal chain (EC→DG→CA3) does not carry an input to fire downstream cells at reasonable drives; the mossy
  detonation needs DG BURSTING the substrate doesn't produce.
- **The next mechanism (a deep sub-arc, NOT chased here) — CORRECTED by reading the substrate:** my first hypothesis
  ("set DG to a bursting neuron type") is WRONG — the DG region ALREADY uses `IZH2007_HIPPO_PYRAMIDAL` (an IB-like
  bursting type; `text_minimal_isolation.py:698`). So the boundary is NOT a missing DG-bursting type. The two REAL
  residuals, from the probes: (i) DG fires very SPARSELY (only 2.5% even at 3000 pA direct — its threshold + the
  `dg_pv_basket` FFI keep it near-silent), so few DG cells are available to detonate; and (ii) even a DENSE DG code +
  detonator-strength mossy (weight 500) does NOT fire CA3, while a direct 3000 pA current fires the SAME CA3 cells —
  i.e. the mossy synaptic CONDUCTANCE doesn't reach CA3 threshold (a conductance-magnitude / driving-force / synchrony
  issue, distinct from the external-current path). ⇒ the actionable resume point is a DEEPER investigation: (a) why the
  mossy conductance (weight×(E−V)) is so much weaker than an equivalent external current at firing CA3 — measure the
  actual mossy PSC vs the 3000 pA current, check the reversal potential / conductance scaling / the per-step decay vs
  DG-firing synchrony; (b) whether DG can be made to fire densely + synchronously (a gamma-paced DG volley) so the
  mossy summates. This is a `sim/`-level or deep-config hippocampal-feedforward-excitability build, deferred below the
  completed gap#4↔gap#5 unification, taken as its own focused pass. (Lesson: read the region's actual neuron type
  before proposing a neuron-type fix — the substrate already had the bursting type.)
- **UNAFFECTED:** the gap#4↔gap#5 unification (BTSP stores → bistable CA3 completes, mechanism 6/6 GO) stands — it uses
  a PRE-ASSIGNED assembly + direct CA3 drive; the emergence of the assembly (from cortical input) is this open boundary.
- Infra: `_gap5_emergent_dg_selection_derisk.py` (the R0 diagnostic — a valid tool for when the feedforward is fixed).
  NO sim/ edit.
