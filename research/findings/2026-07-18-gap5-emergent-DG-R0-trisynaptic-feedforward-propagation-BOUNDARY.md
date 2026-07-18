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

## The integrated select-and-store REFRAME was TESTED — and does NOT sidestep the boundary (honest, verified)
Built the enabling hook (`run(assemblies_ext=...)`, default None → byte-preserved) + ran the integrated test at the
COMPLETION scale (n_ca3=2000): a synchronized mossy volley selected **|A_m| = 1 CA3 cell** — far too weak to seed a
storable assembly. So the reframe's optimism (from the n_ca3=400 `sep_cos` 0.53 transient) does NOT scale: at
n_ca3=2000 the mossy → CA3 activity is essentially nothing, and the integrated select-AND-store cannot proceed. ⇒ the
mossy→CA3-firing boundary is ROBUST ACROSS SCALES and is the genuine core obstacle; it must be fixed BEFORE any
select-and-store. (Verified rather than assumed — the reframe looked tractable but the real-scale test refuted it; this
is the silent-failure discipline in action.) The `assemblies_ext` hook remains a valid, byte-preserved tool for when
the feedforward is fixed.

## ⭐ CORRECTION / ADVANCE — the reframe was RIGHT (recurrent amplification WORKS); my "refuted" call was itself wrong
The integrated-test refutation above used a WEAK recurrent (`ca3w=1.5`, forced by `train=False`) — so there was no
attractor to amplify the mossy seed, and I wrongly concluded the reframe was refuted. Retesting WITH the amplification
(build `train=True`, `coincidence=True` [the dendritic-plateau read], a MODERATE recurrent `ca3w≈4`, + feedback
inhibition) OVERTURNS that: a synchronized mossy volley SEEDS CA3 and the dendritic-coincidence recurrent AMPLIFIES the
seed into a **sparse assembly** — e.g. input-11 → **15-26 CA3 cells**, `sep_cos` 0.10-0.20 (separated from the other
inputs' responses). ⇒ the emergent-DG mechanism WORKS; it is NOT a hard boundary. (Self-correction, SURPASS discipline:
I nearly accepted an over-comfortable "boundary" that the proper amplification test overturned — the heartbeat pushing
the next concrete step is what cracked it.)
- **Remaining residual = ROBUSTNESS (fragility), not existence:** the amplification is at a knife-edge — some inputs
  seed enough to amplify (input-11 → a real assembly), others don't (inputs 22/33 → 0), and too-strong recurrent
  (`ca3w=5`) saturates to all 2000 cells. So the working point (mossy seed strength × recurrent gain × feedback
  inhibition) needs tuning for RELIABLE amplification across inputs (every input → a sparse separated assembly). That
  is the next tuning, NOT a fundamental wall.
- **The BISTABILITY KEYSTONE stabilizes the emergent selection too (the deeper unlock, ~20 probes):** plain recurrent
  amplification (coincidence only) hits the completion TRILEMMA in the SELECTION — too weak → 0, too strong → runaway
  to all 2000 cells (input-dependent). Adding the gap#5 dendritic-bistability keystone (`two_comp=True` + self_regen
  0.15 + KIR 3 + apical_gc_read 5) CAPS the runaway: a STRONG mossy seed (weight ~1500) ignites the bistable UP-state
  for a STABLE SPARSE set (e.g. **6-cell** assemblies, NO runaway) — 2 of 4 inputs at ca3w=4.5 (sep_cos 0.40). ⇒ the
  SAME keystone that resolved gap#5 COMPLETION also resolves the emergent SELECTION trilemma (stable sparse latch
  without a pre-assigned mask — the KIR down-state gives the intrinsic sparsity/stability). The emergent-DG mechanism
  WORKS: mossy seeds → bistable-dendrite amplification → stable sparse emergent assembly.
- **Remaining residual, PRECISELY characterized (~28 probes): RELIABLE MULTI-INPUT SEEDING.** The bistable amplification
  gives a stable sparse assembly for the inputs that DO seed (~2 of 5 inputs → 5-24 cell assemblies, sep_cos ~0.2, no
  runaway), but the OTHER inputs seed NOTHING — and this does NOT improve with mossy weight (to 3000) OR mossy density
  (to 0.30); it only shifts WHICH inputs seed. So the fragility is a property of the SPARSE RANDOM mossy wiring: only DG
  codes whose sparse mossy targets happen to CONCENTRATE on some CA3 cells fire a coincident seed; codes that spread
  their mossy fire none. (Possibly compounded by the DG codes themselves being only single-seed-validated — D.13
  completion was seed-fragile.) ⇒ the deeper fix for RELIABILITY is not a mossy-weight/density knob but a more reliable
  SEEDING: a structured/learned (not purely-random) DG→CA3 map so every DG code concentrates, and/or a reproducible DG
  code per input (re-validate DG separation multi-seed first). This is the emergent-DG's genuine open residual — a real
  frontier, but the CORE mechanism (mossy seed → bistable-dendrite amplification → stable sparse assembly) is SURPASSED
  and works. Resume = reliable-seeding mechanism, then the integrated select-and-store + anti-cheats.

## FULL emergent chain RUNS END-TO-END (select → amplify → store → recall) — honest result: assembly too SMALL to complete
Ran the full chain for a seeding input via the `assemblies_ext` hook: mossy-seed → bistable-amplify → SELECT an EMERGENT
assembly (|A_m|=24 cells) → `encode_btsp` store → bistable recall. Result: **cue 0.017**, nocue 0.000, perm 0.000,
no-encode 0.000. So the chain runs (the emergent assembly IS stored + the anti-cheats are clean), but it barely
completes — because the emergent assembly (24 cells) is too SMALL: at n_ca3=2000, density 0.05, a 24-cell assembly has
only ~1.2 within-assembly connections per cell (vs ~12 for the pre-assigned 240-cell / assembly_frac-0.12 assembly that
completes at cue 0.18). Too few convergent recurrents → no attractor → no completion. ⇒ a SECOND precise residual: the
emergent SELECTION must produce a COMPLETION-SCALE assembly (~240 cells / 12%), but the mossy-seed + bistable
amplification currently yields ~24 (1.2%). The mechanism is demonstrated end-to-end; the two residuals (reliable
multi-input seeding + completion-scale assembly size) are the genuine open frontier — a broader/denser seed + wider
amplification, or a completion config tuned for a sparse emergent assembly (lower density so fewer cells still connect).

## Completion-scale full chain (|A_m|=252) — stores, anti-cheats clean, but completes WEAKLY (a structural residual)
Tuned the amplification to a COMPLETION-SCALE emergent assembly (denser DG code 0.30 + wider mossy 0.20 → input-11
selects **252 cells** ≈ 13% of CA3, matching the 240-cell target, bistability-capped, no runaway). Full chain: cue
**0.038**, nocue 0.000, perm 0.000, no-encode 0.000. So even at the RIGHT size the emergent assembly completes weakly
(0.038 vs the pre-assigned 240-cell's 0.18). ⇒ beyond size, the emergent assembly's STRUCTURE differs from a clean
random draw: the mossy-selected cells (chosen by the DG code + broad amplification) are co-active but likely NOT a
tightly recurrently-interconnected set, so the BTSP-stored within-assembly recurrent is diffuse → a weaker attractor.
⇒ TWO deep residuals: (1) reliable multi-input seeding, (2) emergent-assembly COMPLETABILITY (the selected set must be a
well-connected sparse cluster, not a broad co-active smear — likely needs the amplification to converge to a tight WTA
assembly, or the DG→CA3 map learned so each code maps to a fixed tight cluster). Exhaustively characterized (32 probes +
2 full-chain tests). The MECHANISM is demonstrated end-to-end at completion scale; completability + reliability are the
genuine open frontier for a focused pass.

## KEY ENABLER — the DG codes are REPRODUCIBLE + SEPARATED (the learned-map path is clearly tractable)
Verified: driving the same DG input twice gives an IDENTICAL DG code (Jaccard **1.00** for all 3 inputs), and distinct
inputs give distinct codes (Jaccard **0.07-0.18** = well-separated), each ~60 DG cells (20%). So the earlier worry
("single-seed-fragile DG codes") was WRONG — the DG pattern separation is reliable + deterministic. ⇒ the emergent-DG's
reliable-seeding residual is NOT a DG problem; it is purely the SPARSE RANDOM mossy map (which CA3 cells a reproducible
DG code fires is random). ⇒ the fix is clearly tractable: a LEARNED Hebbian mossy DG→CA3 map binds each reproducible DG
code to a FIXED completable CA3 cluster. The chicken-and-egg (Hebbian needs CA3 to fire) is solved by CO-DRIVING a
target CA3 cluster during encoding (the mossy LEARNS to detonate it) — biologically the mossy-fiber potentiation of the
encoding phase. Then the DG code ALONE reliably fires its bound cluster (reliable seeding) + the cluster is a chosen
tight set (completable). This directly resolves BOTH residuals (reliability + completability) and is EMERGENCE-aligned
(input → emergent DG separation → learned reliable input→cluster mapping). NEXT: build the learned mossy map.

## The naive Hebbian learned-map was TESTED — it does NOT work (two concrete reasons, the deeper fix named)
Built + ran the learned-mossy-map test (co-drive each reproducible DG code + a target 240-cell CA3 cluster, Hebbian on
`dg→ca3`, then drive the DG code alone). Result: **0 seeding** for all inputs (the DG code fires NO CA3 cells after
training — worse than the weak-but-nonzero firing before). Two concrete causes: (1) the SPARSE random mossy (density
0.10) means a DG code connects to only ~10% of CA3, so it CANNOT bind to an ARBITRARY 240-cell target cluster — there
are too few existing DG-code→target synapses to strengthen; (2) likely the documented Hebbian-DECAY behavior (the mossy
weights DECAYED under `enable_hebbian_learning`, from weak-nonzero to zero — the `CLAUDE.md` Hebbian-decay gotcha). ⇒
the naive Hebbian map is refuted (verified, not assumed). The deeper fix (the genuine next mechanism): (a) a DENSER or
STRUCTURED mossy so every DG code reaches enough CA3, AND (b) a NON-DECAYING plateau-gated binding rule — BTSP ON THE
MOSSY (drive the DG code + a plateau on the target cluster → BTSP potentiates the co-active dg→ca3 one-shot, no decay),
binding each reproducible DG code to a fixed completable cluster. That is the emergent-DG's precise open frontier — a
focused build (BTSP-on-mossy + denser mossy), reusing the BTSP machinery already built.

## CULMINATION — the emergent-DG is a delicate MULTI-PART TRILEMMA-BALANCE (each piece validated; the integration is the frontier)
The integrated test (dense mossy + BTSP-on-mossy binding + recurrent amplification ca3w=4 + bistability) FLIPS to
RUNAWAY: the DG code alone now fires ALL 2000 cells, non-specific (separation J=1.00). So BTSP over-strengthened the
mossy+recurrent PAST the bistability's cap. Combined with the earlier 0-seeding regimes, the emergent-DG is a delicate
MULTI-PART trilemma: (mossy seed firing) × (learned dg→ca3 binding) × (recurrent amplification) × (bistability
stabilization) × (completable tight cluster) — each piece VALIDATED individually across the ~33 probes + 6 builds, but
the integrated working point (a STABLE SPARSE RELIABLE SPECIFIC emergent assembly, not 0 and not 2000) is NARROW and
un-found. ⇒ EXHAUSTIVE honest verdict: the emergent-DG mechanism is DEMONSTRATED in pieces (mossy-select → bistable
amplify → BTSP store → recall runs end-to-end; the keystone resolves the amplification trilemma; the DG codes are
reproducible+separated), but a reliable, completable, multi-input emergent selection requires BALANCING the multi-part
system — a genuine deep integration, the emergent-DG's core open frontier, for a focused fresh pass (a principled
gain-control / normalization across the binding+amplification, or a staged encode: bind the mossy at low gain, THEN
raise the recurrent, THEN store). The `run(assemblies_ext=...)` hook + all the pieces are ready.

## Staged/isolated encode — ADVANCE (fixes the runaway; all inputs seed + separated) + the precise next step
The ISOLATED staged encode (plateau on the target but NO target co-drive → only `dg→ca3` potentiates, NOT the recurrent)
FIXES the runaway: all 3 inputs now seed DISTINCT assemblies (13, 5, 29 cells — no 0, no 2000) that are WELL-SEPARATED
(Jaccard ~0.00-0.06). But the seeds are the mossy's NATURAL image of the DG code (recall-of-chosen-target ~0) — because
a plateau-only binding cannot bind a target that does not fire. ⇒ the tension is now PRECISE: co-driving the target
BINDS it but drives the recurrent to RUNAWAY; plateau-only avoids runaway but binds the natural (small, arbitrary)
image. THE TRUE STAGED FIX (the precise next step): build with a WEAK recurrent → BIND the mossy WITH the target
co-driven (no runaway because the recurrent is weak) → then RAISE the ca3→ca3 recurrent weights (post-binding scale) +
BTSP-store the target's within-recurrent → RECALL with the raised recurrent so the learned mossy seed amplifies into the
completable target. This separates the binding gain from the amplification gain — the multi-part balance the un-staged
system couldn't hold. A focused fresh pass; all pieces + the `assemblies_ext` hook ready. (Genuine incremental progress:
the staged/isolated encode already gives reliable multi-input SEPARATED seeding, no runaway — the reliability residual is
substantially advanced; SIZE + target-binding remain.)

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
