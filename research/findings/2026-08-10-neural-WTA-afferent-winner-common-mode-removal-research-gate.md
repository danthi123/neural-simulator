---
type: research-gate
status: active
date: 2026-08-10
mechanism: cortical-afferent-winner-selection
lane: EPISODIC / WTA-readout
---

# Research gate — the neural WTA that selects on AFFERENT drive, not intrinsic strength: a region-scoped firing-rate HOMEOSTAT (the common-mode remover) + E%-max/gamma de-latch (the vehicle). Buildable-now, runner-side, NO sim edit. DR-gated design, verified anchors.

> ⚠️ **PREMISE REFRAMED 2026-08-10 (same day) — see `2026-08-10-neural-WTA-separable-assemblies-weight-controllable-homeostat-premise-REFRAMED.md`.** The cheapest decisive test (the pragmatics afferent-swap probe below) showed the SEPARABLE-assembly WTA is ALREADY weight-controllable at learnable magnitudes (committed v2 oracle probe = 1.0/6 at W=8); the "0.167 latch negative" was a strong-inhibition + high-tonic OPERATING-POINT artifact, and disjoint assemblies have NO per-assembly common-mode for a homeostat to strip. **⇒ Do NOT build the homeostat against the separable case (pragmatics/episodic/cortex-wta) — the fix there is weaker lateral inhibition + a learnable afferent weight.** This gate's homeostat design remains valid ONLY as the far-future CO-RESIDENT/dendritic direction (a different substrate). The refuted-lever analysis + the co-resident→dendritic scoping below still stand.

**Verdict: DESIGN READY / buildable_now = YES.** This is the surpass for the cross-arc-characterized neural-WTA
wall. Produced by a 4-angle deep-research workflow (local corpus + external literature); the two "obvious" levers
are REFUTED with the reason, so they are not re-proposed; the load-bearing file anchors are VERIFIED in code this
session. The honest risk is an operating-point axis, not the build.

## THE DEFECT (do not re-derive — characterized across four arcs)

A shared-inhibition winner-take-all latches on the assembly with the largest **INTRINSIC** strength (per-neuron
threshold heterogeneity + a shared-core/mixed-episode boost + the shared-FS latch: the first assembly to ignite
suppresses the rest), NOT the assembly the **cue-specific AFFERENT** signal points at. So a neural WTA readout of a
heteroassociative recall or a plastic-afferent speaker/source decision is NEGATIVE — "a forced source afferent does
NOT move the winner." Findings: `2026-08-08-pragmatics-readback-leg2-WTA-speaker-NEGATIVE-value-critic-fallback.md`,
`2026-08-07-source-monitor-joint-scale-NO-GO-...`, `2026-04-26-cortex-wta.md`,
`2026-08-08-episodic-CA3-completion-CLOSED-...-WTA-still-negative.md`.

## REFUTED levers (the DR round killed these — do NOT re-propose)

<!--derived-->

- **Pooled Carandini-Heeger divisive normalization on the soma — REFUTED-HERE (mathematical, not contingent).** A
  pooled divisor is ONE common scalar; division by a common scalar is RANK-PRESERVING (Carandini-Heeger 2012, Nat
  Rev Neurosci 13:51). It cancels only a GLOBAL common-mode (contrast/query-scale), whereas the defect's common-mode
  is PER-ASSEMBLY, living inside each competitor's own numerator (`D_i = core_i + unique_i`) — so a common divisor
  leaves its rank untouched and normalization's own WTA property SHARPENS the intrinsic-strong winner. Already run as
  the objrel RANK-2 build → BOUNDARY (`2026-07-05-objrel-rank2-divisive-norm-BOUNDARY.md`). The soma primitive
  (`enable_input_divisive_norm`) is a known-failing config, not a fresh lever.
- **Recall-time subtractive feedforward inhibition (a rank-1 inhibitory pool) — REFUTED-HERE ×3** (objrel BOUNDARY,
  phaseB WALL, source-monitor NO-GO). A point-neuron rank-1 pool delivers only SCALAR-UNIFORM subtraction
  (Holt-Koch 1997: shunting is subtractive on rate; Mikulasch-Priesemann: a rank-1 pool cannot do per-dimension
  subtraction) → it removes the equal-across-competitors component but never the per-assembly heterogeneity that IS
  the defect, and a fixed subtraction induces an anti-correlated see-saw. (Its ENCODING-time variant survives as a
  later lever — see cheap-first sequencing.)

**Why this refines the composer-DC-offset analogy (the insight that seeded this gate):** the composer's DC offset
was a GLOBAL constant C (one label-free subtraction removed it, `1f448d26`); the WTA's intrinsic-strength bias is
PER-ASSEMBLY (a different constant inside each competitor), which is strictly harder — a global divide/subtract
cannot touch it. The correct remover must act PER-NEURON, not pooled.

## THE MECHANISM TO BUILD (composition; attribution split is the whole point)

<!--derived-->

| Arm | Removes which common-mode component | Grounding |
|---|---|---|
| **Region-scoped per-neuron firing-rate HOMEOSTAT (Diehl-Cook adaptive threshold), run over an encoding/settling exposure phase on the competing readout assemblies** | the **per-assembly magnitude bias** — a rate homeostat RAISES the threshold of always-firing shared-core cells (shrinks their fact-independent common drive) and LOWERS thresholds of rarely-firing cue-specific cells (grows their differential drive): decorrelation-by-intrinsic-plasticity, PER-NEURON (what a pooled divide cannot do). | Diehl-Cook 2015; the repo's own `2026-07-18-gap5-emergent-DG-gain-balance-research-gate.md` ("a homeostatic per-neuron adaptive threshold… equalizes excitability so no cell wins structurally") |
| **E%-max + gamma de-latch (de Almeida-Idiart-Lisman 2009)** | the **temporal first-igniter latch** — discharging feedback inhibition every ~25 ms re-runs the contest on CURRENT drive; rank-preserving under an equal shift, so it tolerates the benign equal residual. | de Almeida-Idiart-Lisman 2009 (PMID 19515917); the built `ca3_ff_basket` E%-max block |

Dropped: pooled divisive-norm (refuted), recall-time subtractive FFI (refuted), SFA/STD as the SELECTOR (wrong
polarity — silences the weaker-sustaining rival; wrong timescale ~100-300ms vs the ~10-25ms latch; NEGATIVE alone
`2026-07-23-gap5…intrinsic-fatigue-alone-NEGATIVE`) — SFA kept only as an optional de-latch assist.

## BUILDABLE-NOW (runner-side, additive/default-off, ZERO sim edit) — anchors VERIFIED this session

<!--derived-->

- **Homeostat (remover):** `BrainRegion.enable_homeostasis=True` region-scoped on the readout assemblies only
  (per-region mask `sim/bridge.py:525-533`; consumed `bridge.py:1519`; update rule reads `activity_ema`/`rate_error`
  → `cp_neuron_firing_thresholds`). **CRITICAL — the operating point IS the mechanism:** the GLOBAL defaults are
  deliberately SLOW (`sim/config.py:686-687`: `homeostasis_threshold_adapt_rate=0.0005` ≈0.5 mV/s;
  `homeostasis_ema_alpha=0.0002` ≈5 s EMA), so on a brief post-cue probe the default homeostat NEVER engages as an
  equalizer (the "companion process proxied with a too-slow constant" trap, CLAUDE.md). The build MUST (a) region-
  scope it, (b) RAISE the region's adapt-rate, (c) run an encoding/SETTLING exposure phase so the rate-error
  accumulates before readout.
- **E%-max/gamma (vehicle):** reuse the built ff-basket `research/runners/_riii_ca3_coincidence_completion_derisk.py:80-102`
  (`ca3_ff_basket`, `exc_fraction=0.0`, additive/`default-None` ⇒ byte-identical when off); pace its fast GABA-A
  ~40 Hz to decay/renew each cycle. If STP assists the de-latch, carve the STORED-cue synapses out
  (`RegionPathway.stp_disabled` → `cp_stp_disabled_mask`) so depression hits the latch, not the store.
- **Lesion to the current negative:** region-homeostasis off + `ff_inhib=None` ⇒ byte-identical to the present
  latched WTA.

## THE DECISIVE ANTI-CHEAT (afferent-swap under fixed intrinsic strength) + attribution lesions

<!--derived-->

- **Primary (positive):** hold every assembly's intrinsic strength FIXED (same `cfg.seed` heterogeneity + same
  shared-core membership) and SWAP which assembly carries the cue-specific afferent advantage. Remover ON → the
  winner MUST follow the swap on ≥5/6 blind seeds (42/43/44/100/101/102). Concrete first test: the pragmatics oracle
  probe (intent[t]→utter[t]=30, all other blocks=1, tonic 0) must move from the latched **0.1667 (below 1/3 chance)**
  to ~1.0.
- **No-remover control (MUST FAIL):** remover OFF (or left at the default-slow global rate) → the winner must NOT
  flip; it stays on the intrinsic-strong assembly (the current negative). A flip with the remover off = a wiring
  artifact → reject.
- **Attribution lesions (prevent the E%-max overclaim):** homeostat OFF / gamma ON → the STRUCTURAL winner returns
  (proves the homeostat is the remover, not E%-max); homeostat ON / gamma-reset OFF (static inhibition) → isolates
  the de-latch; ALL off → byte-identical.
- **Guards (objrel-earned) + quantitative verdict:** SCRAMBLE-label must collapse the read (else a heterogeneity/
  position artifact); no-afferent → chance; permuted afferent → follows the permutation; the easy/canonical case
  must NOT regress (the see-saw tell); and SWEEP the afferent advantage DOWN from 30× toward learnable magnitudes,
  report the CROSSOVER drive that flips the winner vs the intrinsic gap — small crossover = real common-mode
  reduction; needs-huge-afferent = cosmetic.

## SCOPE + cheap-first sequencing

<!--derived-->

- **Scope = SEPARABLE-assembly WTA only** (pragmatics-speaker, episodic-CA3 = pending task #7, cortex-wta). Does NOT
  target the CO-RESIDENT source-monitor joint-scale case (unique summed into ONE soma rate on the same cells as the
  core) — no somatic operator can separate two signals in one scalar; that is already mapped to the DENDRITIC
  substrate (`enable_dendritic_divisive_gain`, `sim/config.py:543`) and MUST NOT be re-run here.
- **Cheap-first:** run the runner-side homeostat + E%-max FIRST (no sim edit). If a PER-ASSEMBLY residual survives
  (one competitor structurally contains more core), add the ENCODING-time synaptic lever — inhibition-gated
  Hebbian/BTSP eligibility / heterosynaptic-LTD so shared cells cannot potentiate equally to every source (Cecchini
  et al. 2026; the phaseB C1b-ii recommendation) — a guarded default-off `sim/` edit — ONLY then.

## First build (next focused pass)

New runner (build): `research/runners/_wta_afferent_winner_homeostat_derisk.py` (or extend the pragmatics WTA-speaker
runner) — the pragmatics oracle afferent-swap probe is the cheapest decisive test (small, CPU-runnable, no GPU);
region-scoped raised-adapt-rate homeostat + settling phase + E%-max basket; the full afferent-swap / no-remover /
attribution-lesion / scramble panel; 6 seeds via `cfg.seed`; build-twice threshold-hash determinism first. Serves
task #7 (episodic neural WTA) once the pragmatics probe confirms the remover. Corpus check logged this session.

NO-EXTERNAL-NEEDED beyond this round: Diehl-Cook 2015 + de Almeida-Idiart-Lisman 2009 are the recorded grounding;
the surpass is a runner-side composition on existing, verified substrate.
