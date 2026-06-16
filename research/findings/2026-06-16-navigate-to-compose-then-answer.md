# Navigate-to-COMPOSE-then-answer — the agent COMPOSES what it perceived, on ONE brain (step-3 behavioral build)

**Date:** 2026-06-16
**Runner:** `research/runners/navigate_to_compose_then_answer.py`
**Raw:** `research/findings/raw/navigate_to_compose_then_answer_smoke_seed42.json` (single-seed) + `..._6seed.json` (6-seed)
**Design:** `research/findings/2026-06-16-step3-integration-build-scoping.md` (the controller-verified scoping; §2 build, §4 anti-cheats, §6 task list)
**Predecessor de-risks (all GO, all committed):** `2026-06-16-step3-live-cortex-grounded-compose-cheap-first.md`
(cheap-first + scaled 6-seed + production-composer 6-seed @ D=2048 + correlation boundary + the merged-bridge
co-resident cheap-first).
**Verdict:** **GO** — owner-greenlit GPU build, executed subagent-driven, controller trust-but-verified.

---

## What this is — RECALL → COMPOSE, on one brain

The `navigate_to_see_then_answer.py` milestone (the (B) perception→memory task) had the agent navigate, perceive
objects live, engram-tag them, and afterward **RECALL** what it saw ("I saw the apple"). Its honest boundary: it
**could not algebraically bind a perceived object into a NEW fact** — the engram tag is an opaque ensemble pointer,
and the navigation perception is a RATE code while the composer consumes PHASOR codes (the rate-vs-phasor wall).

This build dissolves that wall behaviorally. The agent NAVIGATES the merged nav+conversation bridge, PERCEIVES +
**GROUNDS** each encountered object IN-EPISODE (a fixed complex projection maps the live `cortex_it` spiking
firing-rate into a composer phasor code: `composer.concepts[o] = angle(M @ live_cortex_it_rate)`), then **COMPOSES**
a novel perceived-object fact on the co-resident resonate-and-fire (`rf`) slice and answers a who/what query about
it — while abstaining on unstored queries (the no-confab moat). It upgrades **RECALL → COMPOSE** on ONE
`SimulationBridge` with the navigation cascade, the conversational parser, the dlPFC planner, the `rf` composer, and
now the perception region all co-resident.

## The one brain (one bridge, one step loop)

- **Body:** the basal-ganglia action cascade on the merged bridge selects each move (the `motor_X` / `sel_X`
  disinhibition winner — a neural decision, not a coordinate argmax). The merged cascade selects moves out of the
  box via `motor_X`; the optional `enable_spiking_wta_readout` pass-through adds the higher-SNR `sel_X` selection
  `navigate_to_see` was validated with (a selection-quality upgrade, not a requirement).
- **Perception:** the bare `cortex_it` region (256 neurons, `co_resident_perception=True`) — the percept is
  rendered into it on arrival (the legitimate sensory render), and its live spiking rate is read OFF the merged
  bridge with OU on (the real episode-noise condition the cheap-first retired).
- **The grounded-code map M:** the de-risked fixed complex projection (rate → phasor), reused verbatim.
- **Compose:** the production fixed FHRR (Fourier Holographic Reduced Representation) bind/bundle/unbind/cleanup on
  the co-resident `rf` slice (`MergedRFComposer`), unchanged.

## Results

**Single-seed GPU smoke (seed 42, `SIM_BACKEND=cupy`, RTX 3090):** GO.

| metric | result |
|--------|--------|
| objects grounded IN-EPISODE | 3 (apple, cat, dog — agent navigated the corridor, reached waypoints) [≥2 required] |
| held-out compose clean | 1.000 (≫ memorization floor 0.500; chance 0.250) |
| no-confab moat (unstored → None) | abstain 1/1 |
| positive recall (stored fact retrieves) | 1/1 |
| LESION (grounding severed) | compose collapses 1.000 → 0.167 (≈ chance) |
| ISO-perception (no body → no encounter) | 0 grounded → no compose |
| T0 byte-identity (cortex_it after rf) | True |

**6-seed validation (seeds 42/43/44/100/101/102, GPU): GO on all 6.**

| seed | grounded in-episode | held-out compose | mem-floor | LESION | moat | pos | ISO | byte-id |
|------|---------------------|------------------|-----------|--------|------|-----|-----|---------|
| 42   | 3 | 1.000 | 0.500 | 0.167 | 1/1 | 1 | 0 | True |
| 43   | 3 | 1.000 | 0.500 | 0.167 | 1/1 | 1 | 0 | True |
| 44   | 3 | 1.000 | 0.333 | 0.167 | 1/1 | 1 | 0 | True |
| 100  | 3 | 1.000 | 0.500 | 0.167 | 1/1 | 1 | 0 | True |
| 101  | 3 | 1.000 | 0.500 | 0.000 | 1/1 | 1 | 0 | True |
| 102  | 3 | 1.000 | 0.333 | 0.000 | 1/1 | 1 | 0 | True |
| **mean** | **3** | **1.000** | **0.444** | **0.111** | **6/6** | **6/6** | **0** | **True** |

Every seed: grounds the 3 objects on its (seed-randomized) route in-episode, composes held-out perceived-object
facts at 1.000 ≫ the memorization floor (mean 0.444, all ≫ chance 0.250), the no-confab moat abstains (6/6) while a
stored fact still retrieves (6/6), **lesioning the grounding collapses the compose** (to 0.167 or 0.000 — far below
the un-lesioned 1.000), no body → 0 grounded, and the byte-identity holds. Raw:
`navigate_to_compose_then_answer_6seed.json`. (Honest note: the fixed corridor route passes 3 of the 4 placed
objects, so the agent grounds 3/4 — a body-trajectory scaffold, not a perception limit; ≥2 grounded is the gate, met
on every seed. The 4-object vocabulary makes the absolute compose a ceiling, as in the de-risks — the discriminating
signals are held-out ≫ floor + the lesion collapse + the moat, all of which hold 6/6.)

## Anti-cheats (all required; all pass)

1. **LESION the grounded-code map (the navsee cut-after-encode form).** The held-out composites are composed under
   the GROUNDED codebook (so each bound filler is the percept-derived code). After the grounded objects'
   `composer.concepts[o]` are restored to random codes (the live-percept grounding severed), re-cleaning up the SAME
   stored composites against the now-lesioned codebook no longer recovers the perceived object → compose collapses
   to ≈ chance. (A naive grounded→random lesion BEFORE compose would NOT collapse it — the FHRR algebra is
   code-agnostic; the cut-after-encode form is the correct control, and proves the recovery rides the live-percept
   grounding, not a structural bias.)
2. **HELD-OUT novel fact (compose ≠ recall).** A (perceived-object, role) combination NEVER composed in any setup
   step unbinds correctly ≫ chance AND ≫ a memorization-floor recall baseline — the capability RECALL lacks (the
   (B) milestone could only recall stored ensembles). This is THE control that separates COMPOSE from RECALL.
3. **ISOLATED-perception (the task needs the perceiving BODY).** With no navigating body to encounter objects,
   nothing is perceived/grounded → no compose. Only the coupled brain (navigating + perceiving) closes the loop.
4. **PROVENANCE + the no-confab MOAT.** Structural: `composer.concepts[o]` IS the live-rate-derived grounded code
   (`== grounded_phases(rate, M)`), and the bind ran on the merged bridge (`cp_rf_w_re is not None` after a store —
   not a silent standalone fallback). The moat: every unstored query returns `None`; a stored fact still retrieves.
   The moat was never weakened.

## BRAIN-BASED-ONLY accounting

Host code is legitimate ONLY for the **environment** (the grid, the agent position, rendering an object's identity
into `cortex_it` on arrival — the sensory render) and the **body** (stepping the agent per the neural cascade
winner). Everything cognitive is the brain's: the move SELECTION is the cascade disinhibition winner; the percept is
a LIVE spiking rate read (`cp_firing_states`, the substrate's own response); the grounded code is that rate through
the fixed projection (the shared-grounded-code map); the COMPOSE is the validated fixed FHRR primitive on the `rf`
slice; the abstention is the composer's. **No `sim/` edit** — reuse-by-import; the two builder kwargs
(`co_resident_perception`, `enable_spiking_wta_readout`) are additive default-False on the RUNNER builder, and the
co-resident composer's sliced `rf_kick` is the already-landed, default-off-byte-identical STEP-2b/5b edit the build
merely uses (regression: `test_merged_rf_composer_coresident` 5/5 + `test_nav_conv_step2b_coresident` 7/7).

## HONEST SCOPE

- This composes perceived **flat-distinct OBJECT** facts via shared grounded codes — the perceived object becomes a
  filler the FHRR algebra can bind into a novel fact. It is the genuine "the agent composes what it sees."
- It uses the **FIXED composer algebra** (the production FHRR bind on the `rf` slice), NOT a learned cortical bind
  (the 2026-06-16 capability map settled that multi-attribute BUNDLING is not learnable from-scratch on point
  neurons; the fixed self-inverse primitive is load-bearing + biology-grounded, not a shortcut).
- It is **NOT** generalization across SIMILAR concepts (treating "dog"/"cat" alike because their codes are similar)
  — that is the separate dendritic / PPMI frontier (the correlation boundary map showed the algebra *tolerates*
  correlation but tolerance ≠ generalization). A deliberate, deferred owner call.
- The grounding is for OBJECTS (the perceived fillers); abstract relata (verbs) use the composer's existing concept
  codes (no sensory grounding for them — the composer's own honest limit).
