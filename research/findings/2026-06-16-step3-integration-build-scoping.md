# Step-3 INTEGRATION BUILD scoping — wiring the grounded-code map onto the merged nav+conv bridge so the agent COMPOSES a perceived-object fact in-episode (2026-06-16)

**Status:** READ-ONLY implementation-scoping design. No `sim/` code, no GPU, no experiment run. The single
deliverable is this doc. Every load-bearing claim is cited to file + line. This SCOPES the owner-gated payoff
build identified by the just-completed de-risk arc (`2026-06-16-step3-live-cortex-grounded-compose-cheap-first.md`
"What remains (owner-gated) #1"); it does not build it.

**Author role:** read-only computational-neuroscience scoping subagent.

**Predecessor docs (read first; this extends them, does not repeat them):**
- `research/findings/2026-06-16-step3-compose-perceived-content-scoping.md` — the deep-research scoping that ranked
  Option (a) "shared grounded codes" as the recommended path and specified the cheap-first probe. THAT probe has
  since been RUN and is GO (below). This doc is the *next* step: the integration build it gated.
- `research/findings/2026-06-16-step3-live-cortex-grounded-compose-cheap-first.md` — the de-risk results being
  extended (cheap-first GO 4-obj; scaled GO to 32-obj 6-seed; production-composer drop-in GO D=2048 6-seed,
  moat intact; correlated-percept boundary map).

---

## 0. Terms (defined once — no undefined acronyms)

- **bridge** — one `sim.bridge.SimulationBridge`: a network of simulated spiking neurons stepped by one
  `_run_one_simulation_step` loop. "The brain."
- **merged bridge** — the single `SimulationBridge` built by
  `research/runners/nav_conv_merged_bridge.py:build_merged_nav_conv_bridge` (line 199) holding the navigation
  basal-ganglia cascade + the conversational parser + the dlPFC dialogue planner, with an OPTIONAL co-resident
  resonate-and-fire composer slice. This is the integration target.
- **rate code** — information in a pool's firing-rate magnitude over a window. The navigation perception
  (`cortex_it`, the ventral object-identity ensembles) is a rate code on Izhikevich neurons.
- **phasor code** — information in the PHASE of a unit-magnitude complex value, in `[0,1)^D`. The conversational
  composer's concept codes. Realized on `RESONATE_AND_FIRE` (RF) neurons + complex synapses.
- **FHRR (Fourier Holographic Reduced Representation)** — the production vector-symbolic-algebra (VSA) scheme:
  bind = complex product of phasors, unbind = multiply by the conjugate, bundle = complex sum, cleanup = max
  phase-cosine over the codebook. Realized in `research/runners/rf_phasor_composer.py`.
- **role / filler / bind / unbind / bundle** — a *role* is a slot (agent/action/patient/…); a *filler* is a
  concept. **bind**=(role,filler)→composite; **bundle**=sum bound pairs into a fact; **unbind**=recover a filler.
- **grounded code** — a concept code that is a deterministic function of an object's perceptual features (vs a
  free random code). The composer takes them via `RFPhasorComposer(grounded_codes={word: phases[D]})`
  (`rf_phasor_composer.py:63,86–89`).
- **grounded-code map / projection M** — the fixed (or learned) transform from a live `cortex_it` rate vector to a
  composer phasor code. The de-risk's `_projection` + `grounded_phases`
  (`_step3_grounded_codes_production_composer_derisk.py:51,76`).
- **engram tag** (Tonegawa, catalog D.14) — the set of neurons that fired in a window
  (`start_engram_recording`→`commit_engram_tag`); `stimulate_tag` re-drives that ensemble (causal recall).
- **the rate-vs-phasor wall** — the load-bearing obstacle: the perceived object is a *rate* ensemble; the composer
  consumes *phasor* codes. The grounded-code map dissolves it by making the percept a phasor.
- **MergedRFComposer** — the `RFPhasorComposer` subclass (`nav_conv_merged_bridge.py:517`) whose RF bind/unbind
  ops run on a SLICE (`rf` region) of the merged bridge instead of on its own per-op bridges (STEP 2b co-residence,
  owner-approved, complete).

---

## 1. DIAGNOSIS — what exists, and the exact gap

### 1.1 The three completed pieces this build composes

**(i) The (B) navigate-to-see-then-answer RECALL milestone — one merged brain, GO 3/3.**
`research/runners/navigate_to_see_then_answer.py` builds ONE bridge (`build_navsee_bridge`, line 149) holding the
navigation cascade (the BODY) + `cortex_it` (PERCEPTION) + `language_output` (the RECALL channel) + a DENSE plastic
`cortex_it→language_output` route trained by Hebbian co-firing. In a LIVE episode the agent navigates (the BG
cascade selects each move, `_cascade_select_move` line 407), perceives objects rendered into `cortex_it` on arrival
(`_perceive_and_tag` line 475), tags each from the live perception window
(`bridge.commit_engram_tag(..., region_filter=["cortex_it"])`, line 515), and AFTER the episode RECALLS them by
`stimulate_tag` → `language_output` read-through (`recall_what_seen` line 593). Its honest scope line (lines 85–88):
**this is RECALL ("I saw the apple"), NOT composition over perceived content — "you cannot yet algebraically bind a
perceived object into a novel role-filler fact … the rate-vs-phasor wall."**

**(ii) The masked-RF co-resident composer on the merged bridge — STEP 2b, COMPLETE.**
`build_merged_nav_conv_bridge(..., co_resident_rf=True)` (line 199) reserves a contiguous `rf` region (`7*rf_D`
neurons, no out-edges into navigation, lines 258–267), and `MergedNavConvAgent(co_resident_composer=True)`
(line 585) drives the composer's RF ops on that slice via the owner-approved sliced `rf_kick`
(`MergedRFComposer._resonate`, line 548). Acceptance gates GREEN: CPU bit-exactness vs standalone +
byte-isolation (`tests/test_merged_rf_composer_coresident.py`, 5/5), the full conversational matrix + no-confab
moat co-resident at D=128 (`tests/test_nav_conv_step2b_coresident.py`, 7/7 GPU). So a fact-composing FHRR
composer ALREADY runs on the merged bridge's own neurons, sharing the one step loop.

**(iii) The grounded-code map — de-risked, drops into the PRODUCTION composer, GO 6-seed.**
`_step3_grounded_codes_production_composer_derisk.py` reads a live `cortex_it` spiking firing-rate vector
(`read_cortex_it_rate`, line 56), projects it through a FIXED complex projection to composer phases
(`grounded_phases`, line 76), and feeds the result as `RFPhasorComposer(grounded_codes={obj: phases})` (line 113).
Result (`-cheap-first.md` §"PRODUCTION-COMPOSER drop-in"): at D=2048 production tier, 6 seeds, the grounded
composer stores a 3-way Subject-Verb-Object fact over PERCEIVED objects, recalls patient + agent (6/6 every seed),
and ABSTAINS (returns `None`) on every unstored query (no-confab moat intact, 3/3 every seed), behavior identical
to the random-code baseline (parity True every seed). The correlated-percept boundary map showed the compose
algebra tolerates percept/code correlation up to code-similarity ≈0.98.

### 1.2 The exact gap (recall → compose), stated mechanically

The (B) milestone and the grounded-code de-risk are **two halves that have never been joined on one bridge:**

| | (B) navigate-to-see (`navigate_to_see_then_answer.py`) | grounded-code de-risk (`_step3_grounded_codes_production_composer_derisk.py`) |
|---|---|---|
| `cortex_it` perception live in an episode? | **YES** (rendered on arrival, tagged live) | NO (a static `build_probe_bridge` percept read, off-episode) |
| co-resident composer on the bridge? | NO (no `rf` region; recall is engram-only) | NO (the production `RFPhasorComposer` on its own bridges) |
| perceived object an algebra operand? | **NO** (engram tag = opaque ensemble pointer) | **YES** (grounded phasor code) — but on a *separate* probe bridge |
| can it COMPOSE a perceived object into a novel fact? | **NO** (recall only) | YES, but the percept source is `build_probe_bridge`, not the *live merged-bridge episode* |

The gap is precisely: **the perceived object's live `cortex_it` rate code never becomes a grounded concept code IN
the co-resident composer's codebook DURING a merged-bridge episode.** (B) has the live perception but no composer;
the de-risk has the grounded-code→composer path but not on the merged bridge in-episode. The build joins them.

### 1.3 The central architectural fact the build must resolve

**`build_merged_nav_conv_bridge` does NOT contain `cortex_it` by default.** It calls
`build_bg_brain_regions(n_cortex=n_cortex)` (line 246) with `enable_visual_cortex` defaulting to False; the
`cortex_it` region is built inside `build_bg_brain_regions` **only when `enable_visual_cortex=True`**
(`g11_bg_runner.py:2428,2473–2481` — `cortex_it` is appended inside the `if enable_visual_cortex:` block). So the
merged bridge's regions are: the BG cascade (`cortex_{N,E,S,W}`→str→gpi→thal→motor/sel) + parser
(`parse_conj`, `parse_role`) + dlPFC (`cortex_ctx`, `dlpfc_wm`) + optional `rf` — **no perception region.**

The perception region `cortex_it` + the `cortex_it→language_output` trained read-out live in
`navigate_to_see_then_answer.py:build_navsee_bridge` (lines 173–257), NOT in the merged builder. **The build's core
structural task is to bring `cortex_it` (and the grounded-code transcoder) onto the merged bridge alongside the
co-resident composer** — i.e. merge the navsee perception regions into the merged builder. Both builders already use
the identical brain-region-framework idiom (`inject_explicit_wiring` over `region_manager.build_wiring_plan`,
`nav_conv_merged_bridge.py:335–341`, `navigate_to_see_then_answer.py:251–257`), so this is a region-list union, not
new machinery.

---

## 2. THE BUILD — concrete wiring (reuse-by-import; flag every glue point)

### 2.1 The data flow (in-episode)

```
 [environment renders object o]            (legitimate sensory render — the body/world; the only host write
        │                                   into the percept code, exactly as (B) renders into cortex_it)
        ▼
  cortex_it band_o  ── live spiking rate read over a window ──►  rate_vec (N_CORTEX_IT,)   [the grounded RATE code]
        │                                                              │
        │  (B) path: engram tag (recall, unchanged, kept)              │  grounded-code map M (fixed complex projection)
        ▼                                                              ▼
  stimulate_tag → language_output  (RECALL: "I saw o")        phases_o = angle(M @ rate_vec) ∈ [0,1)^D   [PHASOR code]
                                                                       │
                                                                       ▼
                              composer.concepts[o] = phases_o   (inject into the co-resident MergedRFComposer codebook)
                                                                       │
                                                                       ▼
                composer.store(agent=o, action=v, patient=p)  ──►  bind+bundle on the merged `rf` slice  [COMPOSE]
                composer.query_patient(o, v) / query_agent(v, p)  ──►  unbind+cleanup → the PERCEIVED object
```

The bind/bundle/unbind/cleanup are the existing fixed FHRR primitive on the co-resident `rf` slice (STEP 2b,
unchanged). The ONLY new in-episode write into a concept code is `composer.concepts[o] = M @ (live cortex_it rate)`
— a grounded percept, never a host-set label. The engram-tag RECALL path is RETAINED unchanged (the build ADDS the
compose capability; it does not remove recall).

### 2.2 Reuse points (file:line) and the minimal new glue

**A new runner `research/runners/navigate_to_compose_then_answer.py`** (the compositional successor to
`navigate_to_see_then_answer.py`) is the home for the glue. Everything else is reuse-by-import:

1. **Merged bridge + co-resident composer (the substrate).**
   `nav_conv_merged_bridge.build_merged_nav_conv_bridge(seed, vocab, co_resident_rf=True, rf_D=...)` (line 199) +
   `MergedNavConvAgent(seed, vocab, co_resident_composer=True)` (line 585). The agent already exposes
   `hear`/`what_does`/`who_does`/`store`-equivalents and `self.composer` is a `MergedRFComposer`.
   **GLUE (new):** the merged builder must be extended to ALSO include the perception region + read-out (next item).
   This is a small additive change to `build_merged_nav_conv_bridge` (a new `co_resident_perception=True` kwarg,
   default False = STEP-2b byte-preserved), not a new file's worth of logic.

2. **The perception region + the (B) read-out (bring `cortex_it` onto the merged bridge).**
   The `cortex_it` + `language_output` `BrainRegion`s and the dense plastic `cortex_it→language_output` route +
   its Hebbian co-firing trainer + freeze are ALL in `navigate_to_see_then_answer.build_navsee_bridge`
   (lines 173–309) and `_train_route` (line 318). The constants
   (`N_CORTEX_IT`, `N_LANG_OUTPUT`, `IT_TO_LANG_GATE`, `PERCEPT_SPARSITY`, `PERCEPT_DRIVE_PA`, `ENGRAM_TOP_K`,
   `_object_band_indices`, `_render_percept`, `_recall_lang_output_pattern`) are imported from
   `funcint_perception_to_memory_probe` (already done in `navigate_to_see_then_answer.py:112–119`).
   **GLUE (new):** append `it_region` + `lang_out_region` to the merged builder's `union_regions`, and add the
   `cortex_it→language_output` dense population to the combined injection (exactly as `build_navsee_bridge` does
   it at lines 234–257). This is the same `inject_explicit_wiring` union the merged builder already performs for
   `dlpfc_loop` (`nav_conv_merged_bridge.py:335–341`); the read-out population is one more entry in `union_plan`.
   **Ordering caution (load-bearing):** append the perception regions in a position that does NOT shift the
   navigation/parser/dlPFC/rf index bases the existing STEP-2a/2b gates depend on. The `rf` region is appended
   LAST precisely to preserve those bases (`nav_conv_merged_bridge.py:262–268`); the perception regions should go
   between the dlPFC and rf regions (or after rf with rf's base recomputed), and the build must re-assert the
   STEP-2b byte-identity of the nav/parser slices (Task-1 anti-cheat).

3. **The grounded-code map M (rate → phasor) — the de-risk's exact construction.**
   `_step3_grounded_codes_production_composer_derisk._projection(D, n_in, seed)` (line 51) and
   `grounded_phases(rate_vec, proj)` (line 76) ARE the map. `read_cortex_it_rate(bridge, it_indices, obj_idx)`
   (line 56) is the live rate read. **GLUE (new):** call these against the MERGED bridge's `cortex_it` slice
   (`it_indices = rm.indices("cortex_it")`) instead of the probe bridge's, during the episode. The map is a fixed
   numpy projection (host arithmetic on the substrate's own live rate read — see §5 BRAIN-BASED accounting).

4. **Inject the grounded code into the co-resident composer's codebook.**
   `MergedRFComposer` inherits `RFPhasorComposer.concepts` (a `{word: phases[D]}` dict, `rf_phasor_composer.py:80`)
   and the `grounded_codes` override hook (lines 86–89). **GLUE (new):** on arrival at object `o`, after reading the
   live rate and mapping to `phases_o`, set `agent.composer.concepts[o] = phases_o`. (Equivalently, rebuild the
   composer once with `grounded_codes=` after a perceive-all pass — but the in-episode `concepts[o] = …` write is
   the more faithful "the agent grounds the code as it sees the object" form and is what proves COMPOSE-in-episode.)
   This is a dict assignment; no `sim/` touch, no composer change.

5. **Compose + query (the payoff).** `agent.hear("o v p")` (line 633) / `agent.composer.store(o, v, p)` /
   `agent.what_does(o, v)` (line 644) / `agent.who_does(v, p)` (line 648) — UNCHANGED. They now operate on a fact
   whose agent/patient codes are the live percept-derived grounded codes, composed on the `rf` slice.

6. **The episode scaffold (navigate, arrive, perceive).** `navigate_to_see_then_answer.run_episode` (line 534),
   `_cascade_select_move` (line 407), `default_object_layout` (line 522), `_steer_toward` (line 463) — reuse the
   navigation episode loop. **GLUE (new):** replace the `_perceive_and_tag`-only arrival handler with one that ALSO
   does the grounded-code read+inject (item 3+4). The tag write is RETAINED (recall stays available alongside
   compose) — so on arrival the agent BOTH tags (for recall) AND grounds the concept code (for compose).

**Net new code:** one runner (`navigate_to_compose_then_answer.py`) + one additive kwarg on
`build_merged_nav_conv_bridge` (`co_resident_perception=True`, default False) that unions in the `cortex_it`/
`language_output` regions + the read-out population. No new mechanism — every piece (perception region, read-out,
grounded-code map, co-resident composer, episode loop, store/query) already exists and is validated.

### 2.3 Does ANY `sim/` protected-module edit get touched? — NO.

**No `sim/` edit is required.** Every reuse point is a public API or a reuse-by-import:
- The perception region + read-out are framework `BrainRegion`/population injected via the public
  `inject_explicit_wiring` (`nav_conv_merged_bridge.py:341` already calls it; `navigate_to_see_then_answer.py:257`
  already does the identical read-out injection).
- The co-resident composer (`MergedRFComposer`, STEP 2b) is already landed and used the owner-approved sliced
  `rf_kick` (`neuron_mask=`) edit, which is ALREADY in `sim/` (the 5b protected edit, complete and default-off
  byte-identical). The build USES it; it adds nothing to `sim/`.
- The grounded-code map is host numpy on a live rate read (`bridge.cp_firing_states` is a public read).
- `composer.concepts[o] = phases_o` is a numpy dict assignment.

The owner's "byte-level diff review of any sim/ edit" requirement is therefore **not triggered** — reuse-by-import
only. (If, as a follow-on, the grounded-code map were made LEARNED via the on-bridge `cortex_it→language_output`
co-firing read-out extended to project onto the concept code — the predecessor scoping's Option (a2) — that ALSO
needs no `sim/` edit, since it reuses the existing dense-route + Hebbian-co-firing machinery, `_train_route`
`navigate_to_see_then_answer.py:318`. The fixed map (a1) is the recommended first build; a2 is the brain-based
follow-on.)

---

## 3. CHEAP-FIRST DE-RISK — the smallest CPU/numpy smoke before any GPU build

**The single load-bearing question the smoke must answer:** when the perceived object's grounded code is sourced
from the LIVE `cortex_it` rate ensemble OF THE MERGED BRIDGE (not a standalone `build_probe_bridge`, and with the
nav cascade + parser + dlPFC + `rf` all co-resident), does the perceived object still COMPOSE into a NOVEL fact and
unbind back to the correct percept — i.e. does the grounded-code map survive co-residence on the full merged
substrate?

The grounded-code→production-composer drop-in is ALREADY GO (D=2048, 6-seed) — but on a SEPARATE
`build_probe_bridge`, not the merged bridge, and at production D. The cheap smoke isolates the ONE new variable:
**reading the grounded rate from the merged bridge's `cortex_it` slice (co-resident with everything else) and
composing on the merged `rf` slice.** Everything else is already de-risked.

> **Probe `research/runners/_step3_merged_grounded_compose_smoke.py` (numpy, CPU):**
> 1. Build the merged bridge WITH the new perception regions:
>    `build_merged_nav_conv_bridge(seed, vocab, co_resident_rf=True, co_resident_perception=True, rf_D=small)`
>    at a TINY D (e.g. D=8–16, matching `test_merged_rf_composer_coresident.py`'s D=8) and a tiny vocab
>    (the 4 `OBJECT_WORDS` apple/river/dog/cat + a couple of verbs), on `SIM_BACKEND=numpy`.
> 2. For each object: read its live `cortex_it` rate on the MERGED bridge (`read_cortex_it_rate` against
>    `rm.indices("cortex_it")`), map → phases via the fixed `_projection`/`grounded_phases`, set
>    `agent.composer.concepts[o] = phases_o`.
> 3. Split the (agent, patient) ordered distinct-object pairs into MEMORIZED vs HELD-OUT (the
>    `_step3_live_cortex_grounded_compose_probe.py` anti-cheat, lines 120–123). For each HELD-OUT fact:
>    `store`/`query_patient`/`query_agent` via the co-resident composer; check the unbind recovers the perceived
>    object. Score a memorization-floor recall baseline on the SAME held-out facts (lines 128–140).
> 4. Assert the no-confab moat: `query_patient` of an unstored (agent, action) returns `None`.

**The exact GATE (mirrors the cheap-first probe's GO gate,
`_step3_live_cortex_grounded_compose_probe.py:188–190`, on the MERGED substrate):**
- **GO** if, on merged-bridge `cortex_it`-rate-derived grounded codes composed on the merged `rf` slice:
  **held-out clean compose ≥ 0.90** (unbind agent+patient → correct perceived object) AND **held-out compose ≥
  the memorization floor + 0.30** (compose generalizes to never-composed pairings; a recall baseline does not) AND
  **the no-confab moat abstains** on every unstored query (returns `None`). 3 seeds (42/43/44) for the smoke.
- **NO-GO / honest negative** if held-out compose collapses toward chance (1/N) or to the memorization floor
  (→ the merged-substrate `cortex_it` rate read is noisier than the standalone probe's — localize to a population-
  code lift or more read steps, the documented fix), or the moat breaches (HARD STOP — never weaken the moat).

**Why this is the right cheapest move:** it is < 30 min CPU/numpy, ZERO `sim/` edit, and isolates the single
unproven variable (the live read off the merged bridge's `cortex_it` co-resident with the whole stack) — the
grounded-code→composer path itself and the co-resident composer are each ALREADY GO independently. A GO here gates
the GPU build at production D; a NO-GO localizes the limit without spending GPU.

---

## 4. ANTI-CHEAT CONTROLS (all required)

Each defeats a specific way to fake "the agent composes what it perceived." All four are required; the standing
constraints below are binding.

1. **LESION the grounded-code map (compose must collapse).** Zero / sever the grounded codes (set the perceived
   objects' `composer.concepts[o]` back to their random codes, OR zero the projection M so `phases_o` carries no
   percept) → composing the perceived object must collapse to chance: the unbind no longer recovers the perceived
   object, because the fact's filler is no longer the percept-derived code. This proves the COMPOSE rides the
   live-percept grounding, not a fixed structural bias or a leak. (Mirrors the (B) lesion,
   `navigate_to_see_then_answer.py:_lesion_route` line 615, and the predecessor scoping's anti-cheat 1.)

2. **HELD-OUT novel fact (compose ≠ recall — generalize above the memorization floor).** A (perceived-object, role)
   combination NEVER composed in any setup step must unbind correctly ≫ chance AND ≫ a memorization-floor recall
   baseline. This is the capability recall LACKS (recall reactivates a stored ensemble; compose generalizes to new
   pairings). Reuse the leakage-free MEMORIZED/HELD-OUT split + the `_mem_recall` lookup-table floor
   (`_step3_live_cortex_grounded_compose_probe.py:118–140`): a recall-only system scores at the floor on held-out;
   a composing system beats it by ≥ 0.30. **This is THE control that separates COMPOSE from RECALL** — it is the
   whole point of the build (the (B) milestone could already recall; only this proves composition).

3. **PROVENANCE (the grounded code is the live perception, not a host copy; the only recall-time write is
   legitimate).** Assert structurally that the filler code in any composed fact is the percept-derived code read
   from the merged bridge's `cortex_it` firing (`bridge.cp_firing_states[it_indices]`), never a host-set concept
   vector. The ONLY writes into a perceived object's code are (i) the environment's sensory render into `cortex_it`
   (the legitimate world render, exactly as (B)) and (ii) `composer.concepts[o] = M @ (that live rate)`. No host
   code copies a labeled "apple" phasor in. (Mirrors `navigate_to_see_then_answer.py:provenance_check` line 626 +
   the predecessor scoping's anti-cheat 1.) Additionally assert the STEP-2b co-residence anti-cheat holds: the
   compose actually ran on the merged `rf` slice (`agent.composer._merged is agent._merged_bridge`,
   `cp_rf_w_re is not None` after a store — `test_nav_conv_step2b_coresident.py:43–47`), not a silent standalone
   fallback.

4. **The no-confab MOAT must stay intact (abstain on unstored queries).** Every unstored (agent, action) /
   (action, patient) query must return `None`/`"unknown"` (the abstention the production composer already gives,
   `rf_phasor_composer.py:316,331,340,364`). The grounded-code drop-in de-risk already showed the moat survives
   grounding (3/3 abstain every seed, D=2048); the build must re-confirm it ON the merged bridge. **NEVER weaken
   the moat to make a recall/compose number look better** — a moat breach is a HARD STOP
   (`_step3_grounded_codes_production_composer_derisk.py:159–161`).

**Standing constraints (binding on every gate):**
- **GPU/CuPy for decisive runs; numpy only for the tiny smoke.** The cheap-first §3 smoke is numpy/CPU; the
  behavioral gate + the 6-seed validation run on `SIM_BACKEND=cupy` (the merged bridge + RF complex matvec at
  production D run on CuPy — `test_nav_conv_step2b_coresident.py:23–26`).
- **6-seed before claiming generalization.** The behavioral gate runs ≥ 6 seeds (42/43/44/100/101/102) before any
  "the agent composes what it sees" claim — matching the de-risk's own 6-seed bar for the production drop-in.
- **Never weaken the frozen bars or the moat.** The parser/dlPFC/nav frozen-weight isolation (the 5a/STEP-2a
  byte-identity) and the no-confab moat are load-bearing; the build must preserve them, not relax them.

---

## 5. HONEST SCOPE LINE

**What this build IS:** the agent NAVIGATES the merged nav+conv bridge, PERCEIVES an object live (rendered into
`cortex_it`), grounds that live spiking rate into a composer concept code via a fixed projection, and COMPOSES the
perceived object into a NOVEL role-filler fact on the co-resident `rf` slice — then answers a who/what query about
that fact and abstains on unstored ones. It upgrades the (B) milestone from RECALL ("I saw the apple") to COMPOSE
("the apple is red" / "dog chase <perceived cat>"), realized on ONE bridge with the perception + the composer +
the navigation + the parser + the dlPFC all co-resident. It is a consolidation of EXISTING, separately-validated
capabilities (live perception + grounded-code map + co-resident FHRR composer + episode loop) onto one substrate
in-episode — the genuine "the agent composes what it sees."

**What this build is NOT:**
- It is **NOT** the dendritic / PPMI generalization-across-similar-concepts frontier (CLAUDE.md step-3 fork,
  Option c). It composes FLAT-DISTINCT object facts via shared grounded codes; it does NOT transfer knowledge from
  "dog" to "cat" because their codes are similar. The correlated-percept boundary map (`-cheap-first.md` §"CORRELATED")
  explicitly notes "the algebra tolerates correlation" ≠ "correlation provides generalization" — only the former is
  in scope; the latter remains the deferred dendritic-substrate arc (a protected `NeuronModel` edit, months-scale,
  a deliberate owner call).
- It uses the **FIXED composer algebra** (the production FHRR bind/bundle/unbind/cleanup on the `rf` slice), NOT a
  learned cortical bind. The 2026-06-16 capability map settled that multi-attribute BUNDLING is not learnable
  from-scratch on point neurons; the fixed self-inverse primitive (or a dendrite) is load-bearing and
  biology-grounded, not a shortcut. The grounded code is learned-from-perception in the (a2) follow-on, but the
  BIND stays the fixed primitive.
- The grounding is for OBJECTS only (the perceived fillers). ABSTRACT relata ("red", "on", "near") use the
  composer's existing concept codes — there is no sensory grounding for them (the composer's own honest limit,
  `rf_phasor_composer.py:83–85`).
- The recommended first build uses the FIXED projection (a1) for internal-consistency compose (perceive→ground→
  store→query all use the same percept-derived code). Making the agent answer with the WORD "apple" (the percept
  landing on the NAMED concept) needs the LEARNED map (a2) — the brain-based follow-on, same co-firing mechanism,
  still no `sim/` edit.

**BRAIN-BASED accounting:** the grounded code is a LIVE spiking rate read (`cp_firing_states` over the read window)
on the merged bridge — the substrate's own response to the rendered percept, not a host stand-in. The
bind/unbind/bundle is the validated fixed FHRR primitive on the co-resident `rf` slice. The grounded-code map M is
the shared-grounded-code projection (host arithmetic on the substrate's own live rate — the predecessor scoping's
recommended Option a mechanism, §2). Host code is legitimately the episode scaffold (rendering the percept = the
environment's sensory render; the navigation trajectory + body stepping = the body) + the scoring. The compose +
the abstention are the brain's (the composer's) job, on its neurons.

---

## 6. RANKED BUILD TASK LIST (cheap-first → behavioral gate → 6-seed)

Ordered; each task names its gate. No task starts before the prior task's gate is GREEN.

| # | Task | Gate |
|---|------|------|
| **T0** | **Merged-builder perception union (the one structural change).** Add `co_resident_perception=True` (default False) to `build_merged_nav_conv_bridge`: union in `cortex_it` + `language_output` `BrainRegion`s (between dlPFC and `rf`) + the dense `cortex_it→language_output` read-out population into the combined injection, reusing the navsee constants + injection idiom (`navigate_to_see_then_answer.py:173–257`). | A construction smoke (CPU): the merged bridge now has `cortex_it`, `language_output`, AND `parse_conj`/`dlpfc_wm`/`rf`/`cortex_N`; the read-out gate registers; the nav/parser/rf index bases are UNCHANGED vs `co_resident_perception=False` (STEP-2b byte-identity preserved). |
| **T1** | **Cheap-first merged-grounded-compose smoke** (`_step3_merged_grounded_compose_smoke.py`, §3): live `cortex_it` rate off the MERGED bridge → fixed map → `composer.concepts[o]` → store/query on the merged `rf` slice; held-out vs memorization-floor + moat. CPU/numpy, tiny D + 4 objects, 3 seeds. | **GO** = held-out clean compose ≥ 0.90 AND held-out ≥ mem-floor + 0.30 AND moat abstains (all 3 seeds). NO-GO localizes (population-code lift / more read steps) WITHOUT spending GPU. |
| **T2** | **Anti-cheat battery on the smoke** (§4): lesion the grounded map (compose collapses), provenance assert (filler = live `cortex_it` read, co-residence holds), held-out vs floor (already in T1), moat (already in T1). CPU. | Lesion drops held-out compose to ≤ chance; provenance assertions pass (tag/code ⊆ percept, `composer._merged is merged_bridge`, `cp_rf_w_re is not None` post-store); moat 100% abstain. |
| **T3** | **The behavioral gate runner** (`navigate_to_compose_then_answer.py`, the compositional successor to `navigate_to_see_then_answer.py`): a LIVE merged-bridge episode — navigate, perceive+ground each encountered object in-episode, COMPOSE a novel fact involving a perceived object, then answer a who/what query about it + abstain on unstored. Single-seed GPU smoke first. | A single-seed GPU smoke: the agent navigates (cascade moves), grounds ≥ 2 perceived objects in-episode, composes a held-out fact, `what_does`/`who_does` recover the perceived object, and `what_does(unstored)` is `None`. The ISOLATED controls (no perception → nothing to ground → no compose; map-lesion → compose collapses) hold. |
| **T4** | **6-seed validation** of the behavioral gate (seeds 42/43/44/100/101/102) at production D (=128 or 2048 per the merged-bridge tier), `SIM_BACKEND=cupy`. | **GO** = held-out perceived-object compose recovers ≥ the de-risk's bar (clean ≥ 0.90, ≫ mem-floor) on all 6 seeds, moat intact (abstain on every unstored query) on all 6 seeds, lesion collapses on all 6 — i.e. the agent reliably COMPOSES what it perceived, in-episode, on one bridge. |
| **T5** *(follow-on, owner-gated)* | **Learned grounding map (a2):** extend the on-bridge `cortex_it→language_output` co-firing read-out (`_train_route`, `navigate_to_see_then_answer.py:318`) to project the percept onto the composer's NAMED concept code (so the agent answers with the WORD), still reuse-by-import, no `sim/` edit. | The learned map lands the perceived object on its NAMED concept code at compose grade (held-out compose names the right word ≫ chance), 6-seed — the brain-based percept→concept alignment. |

---

## SUMMARY (executive)

**The build in 2 sentences.** Bring the navigation perception region `cortex_it` onto the already-merged
nav+conv+`rf` bridge (one additive `co_resident_perception` kwarg on `build_merged_nav_conv_bridge`, reusing the
navsee perception regions + read-out verbatim), and in a live episode read each perceived object's `cortex_it`
spiking rate, map it through the de-risked fixed complex projection into a phasor, write it into the co-resident
`MergedRFComposer`'s codebook (`composer.concepts[o] = phases_o`), and `store`/`query` a novel role-filler fact —
upgrading the (B) navigate-to-see milestone from RECALL ("I saw the apple") to COMPOSE ("dog chase <perceived
cat>") on one brain. Every piece (live perception, the grounded-code map, the co-resident FHRR composer, the
episode loop, store/query, the no-confab moat) already exists and is separately validated; the build joins them
in-episode.

**The single biggest implementation risk.** The live `cortex_it` rate read OFF THE MERGED BRIDGE — co-resident
with the navigation cascade, parser, dlPFC, and `rf` all firing under OU + tonic drive — being noisier than the
standalone `build_probe_bridge` read the de-risk used (which composed at D=2048 6-seed in isolation). If the
co-resident rate read degrades the grounded code below the compose bar, the documented fix is the population-code
lift / more read steps; the cheap-first smoke (T1) isolates exactly this before any GPU spend. The secondary risk
is region-ordering: the perception regions must union in WITHOUT shifting the nav/parser/dlPFC/rf index bases the
STEP-2a/2b byte-identity gates depend on (the build re-asserts that byte-identity as T0's gate).

**The cheap-first gate.** `_step3_merged_grounded_compose_smoke.py`, CPU/numpy, tiny D + 4 objects, 3 seeds:
**GO** iff held-out clean compose ≥ 0.90 AND held-out compose ≥ memorization-floor + 0.30 (compose generalizes
where recall does not) AND the no-confab moat abstains on every unstored query — sourcing the grounded code from
the merged bridge's own `cortex_it` slice and composing on its own `rf` slice. < 30 min, zero `sim/` edit.

**Is any `sim/` edit required? NO.** The build is reuse-by-import only: the perception region + read-out are
public-API framework injection (`inject_explicit_wiring`), the grounded-code map is host numpy on a public live
rate read, the codebook write is a numpy dict assignment, and the co-resident composer (`MergedRFComposer` + the
sliced `rf_kick`) is an ALREADY-LANDED, default-off-byte-identical STEP-2b/5b edit the build merely USES. The
owner's byte-level sim/-diff review is therefore not triggered. (The follow-on learned-map (a2) also needs no
`sim/` edit.)
