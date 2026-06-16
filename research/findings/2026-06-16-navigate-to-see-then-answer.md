# Navigate-to-see-then-answer — the (B) PERCEPTION→MEMORY behavioral task on ONE brain: **GO (6/6 seeds)**

> **6-seed confirmation (2026-06-16, controller):** extended to seeds 42/43/44/100/101/102 (`navigate_to_see_then_answer_6seed.json`) — COUPLED recall **3/3 every seed**, LESION **0/3 every seed**, ISOLATED-NAV/ISOLATED-PERCEPTION no recall, SCRAMBLE specificity tracks the layout, provenance clean — all 6 seeds. The original 3-seed result was also controller-reproduced independently (== the build subagent's numbers). The milestone meets the 6-seed standing-rigor rule.

**Date:** 2026-06-16
**Runner:** `research/runners/navigate_to_see_then_answer.py`
**Raw:** `research/findings/raw/navigate_to_see_then_answer.json`
**Design:** `docs/plans/2026-06-10-functional-integration-one-brain-design.md` §2 (the follow-on task), §3.3 (the
(B) engram mechanism), §5 (the anti-cheats), §7 step 4.
**Verdict:** **GO** — seeds 42/43/44, GPU (`SIM_BACKEND=cupy`). Exit 0.

---

## What this is

The (B) counterpart of the (A) language→action milestone (`spoken_instruction_nav.py`). (A) had the conversational
channel drive the body; **(B) has the body's PERCEPTION (navigation-side) write into memory, which the
conversational channel later recalls** — the deeper cross-region "one brain" interaction, demonstrated
**behaviorally within a live navigation episode** (the basal-ganglia action cascade firing, the agent moving, OU
noise running) rather than a static probe.

**The task ("navigate-to-see-then-answer"):** the agent NAVIGATES a gridworld corridor (the BG cascade
`cortex_{N,E,S,W}→str_D1→gpi→thal→sel/motor` selects each move; the body steps). Objects sit at grid cells. As the
agent ARRIVES at an object's cell, the environment renders that object's identity into the perception region
`cortex_it` (a legitimate sensory render) WHILE a live engram recording accumulates, and the perceived ensemble is
committed as the tag `seen_<obj>` — captured DURING the episode. AFTER the episode, queried "what did you see?",
each tag is stimulated (`stimulate_tag`) and the `language_output` reactivation is read through a TRAINED
`cortex_it→language_output` route → the recalled words. Recall accuracy on the objects the agent ACTUALLY
encountered on its path is scored vs chance (1/4).

This **moves the two GO (B) static de-risks** (`funcint_perception_to_memory_probe.py` clean read-out, and
`funcint_perception_to_memory_trained_probe.py` TRAINED noisy `cortex_it→language_output` read-out) **into a live
navigation episode** — the perceive→tag→recall loop now runs with the cascade live and the body moving. The recall
+ training + lesion + provenance logic is reused VERBATIM from those de-risks (`encode_percept_engram` logic,
`_recall_lang_output_pattern`, `_recall_metrics`, `provenance_check`, the trained-route `train_readout` logic).

---

## The one brain (one `SimulationBridge`, one step loop)

- **Body:** the BG action cascade from `g11_bg_runner.build_bg_brain_regions(n_cortex=100,
  enable_spiking_wta_readout=True)` — `cortex_{N,E,S,W}` → striatum (D1/D2) → GPe/GPi → thalamus →
  `sel_{N,E,S,W}` (the spiking winner-take-all selection layer). The committed move is the disinhibited spiking
  winner (the `sel_X` pool with the most spikes), **not a coordinate argmax**. Validated direction-selective: a
  biased `cortex_d` releases its own `sel_d` (N→N, E→E, S→S, W→W in the cascade probe).
- **Perception:** `cortex_it` (256 neurons) — the ventral "what"-stream object-identity ensembles; the engram
  source.
- **Recall channel:** `language_output` (256 neurons) + the DENSE plastic `cortex_it→language_output` route
  (~42 k synapses), whose per-object selectivity is GROWN by Hebbian co-firing (the Pulmüller / b3 / concept-pool
  embodied co-firing recipe), NOT hand-wired. Trained on-diagonal ≫ off-diagonal (≈8.1/0.67) — a lossy, learned
  map, so recall correctness is a genuine signal above chance, not ~1.0.

---

## Results (3 seeds, GPU)

| Seed | COUPLED recall (encountered) | LESION recall | ISO-NAV / ISO-PERC tags | SCRAMBLE specificity | route on/off |
|------|------------------------------|---------------|--------------------------|----------------------|--------------|
| 42   | **3/3** (apple, cat, dog)    | 0/3           | 0 / 0                    | ✓ (→ river,dog,cat)  | 8.16 / 0.67  |
| 43   | **3/3** (river, dog, cat)    | 0/3           | 0 / 0                    | ✓ (→ apple,cat,dog)  | 8.45 / 0.68  |
| 44   | **3/3** (river, cat, apple)  | 0/3           | 0 / 0                    | ✓ (→ dog,apple,cat)  | 7.98 / 0.61  |

Chance recall (top-1 of 4 by margin) = 0.25. **COUPLED = 1.00 on every seed; LESION = 0.00 on every seed.** The
object layout is seed-randomized, so the encountered set differs per seed; each seed recalls exactly the 3 objects
on its path and **abstains** on the unencountered 4th (the `None` top-1: river@42, apple@43, dog@44 are never
perceived → no tag → not recalled — the no-confab behavior, it does not hallucinate the unseen object).

---

## Anti-cheats (design §5 — all required, all pass on all seeds)

1. **LESION (primary):** the COUPLED episode runs identically (the agent perceives + tags the 3 objects), then the
   trained `cortex_it→language_output` route is cut (42 025 synapses zeroed). Stimulating the (intact) engram tags
   no longer reaches `language_output` → **recall collapses 3/3 → 0/3 on every seed.** ⇒ the recall RIDES the
   synaptic route, not a leak or a Python copy. (`_lesion_route`.)
2. **ISOLATED controls (the task needs BOTH brains):**
   - **ISOLATED-NAV** — the agent navigates the SAME route but nothing is rendered/tagged (`perceive=False`) → **0
     tags, 0 recall** on every seed. ⇒ the body alone produces no recall.
   - **ISOLATED-PERCEPTION** — perception with NO body (the cascade is omitted, `with_body=False`) → the agent
     never traverses the grid, never arrives at an object cell → **0 tags, 0 recall** on every seed. ⇒ perception
     without the navigating body to ENCOUNTER objects produces no recall. Only the COUPLED brain closes the loop.
3. **PROVENANCE:** every committed tag is a SUBSET of `cortex_it` (asserted structurally — the tag IS the perceived
   ensemble); the only recall-time write is `stimulate_tag` (drives the tagged perceived ensemble, NOT a copied
   percept vector); `language_output` is never driven at recall. The only perception-side write is the orthogonal
   object code into `cortex_it` during the live encounter (the sensory render). No parser/percept-derived value is
   written into a non-perception drive at recall. (`provenance_check` — clean on all seeds.)
4. **Object-scramble / specificity:** permuting WHICH objects are present (reversing the layout) changes the
   encountered set (e.g. seed 42: {apple,cat,dog} → {river,dog,cat}) and the recall TRACKS the new set (3/3, no
   spurious tags) on every seed. ⇒ the recall reflects the objects ACTUALLY encountered on THIS path, not a fixed
   structural bias.
5. **Multi-seed:** 3 (42/43/44), all GO.

---

## BRAIN-BASED-ONLY accounting

Host code is legitimate ONLY for (1) the **environment** — the grid, the agent position, the object placement, and
RENDERING an object's identity into `cortex_it` when the agent arrives (a sensory render, the perception-side
analogue of (A) rendering the command word into `language_input`; the Gabor/retina front-end is separately
validated, so a direct object-identity render to `cortex_it` is in-scope per design §3.3), plus the cascade tonic
pacemakers + the cortex steer-bias toward the next route waypoint (body/environment scaffolding — the agent's
trajectory + intrinsic/brainstem drive); and (2) the **body** — stepping the agent per the spiking `sel`/`motor`
winner. Everything cognitive is neural: the move SELECTION is the cascade disinhibition winner, the memory write is
the engram (the neurons that fired ARE the memory), and the recall is NEURAL REACTIVATION through the trained
synapses — **not a Python lookup.** The route's per-object selectivity is LEARNED (co-firing), not hand-wired.
**No `sim/` edit** (the cascade, the engram API, the Hebbian co-firing, and the trained route are all public APIs /
reuse-by-import).

---

## One implementation note worth recording (the merged-bridge clip tension)

The standalone (B) trained probe used `hebbian_max_weight = stdp_w_max = 25` (so the trained `cortex_it→language_output`
route stays SELECTIVE — a low ceiling keeps the absolute on/off-diagonal spread small). On the MERGED bridge that
clip is a foot-gun: the bridge applies the global Hebbian/STDP weight CLIPS **ungated** (CLAUDE.md "5a
plasticity-isolation … the two global weight CLIPS are UNGATED"), so a low ceiling clips the nav cascade's strong
`cortex_X→str_D1_X` corticostriatal synapses (`weight_mean ≈ 125`) down to 25 → `str_D1` goes silent → the whole
disinhibition cascade dies (diagnosed precisely: `cortex_E` fires 525 spikes but `str_D1_E` fires 0; raising the
clip to 400 revived it but then the route lost selectivity, on/off ≈ 386/191). The clean resolution (in
`_train_route`): the resting config keeps the LOW route ceiling, and the route-training pass **snapshots the
non-route (cascade + internal) synapse weights before the pass and restores them after** — the cascade is a fixed
structural circuit in this episode, so the low-clip route-training pass cannot crush it, while the route alone keeps
its trained values. Result: route on/off ≈ 8/0.67 (selective) AND the cascade selects all 4 cardinals correctly.

---

## HONEST SCOPE

- This is a **RECALL** interaction — "I saw the apple" → later recall "apple", driven by the live perception during
  a real navigation episode. It is **NOT composition over perceived content** (you cannot yet algebraically bind a
  perceived object into a novel role-filler fact). The compositional version genuinely requires shared grounded
  codes / the learned-cortex step-3 (the rate-vs-phasor cross-code wall, design §6); the engram-tag mechanism
  SIDESTEPS that wall but only as recall — that boundary is the deliberate scope line, not a defect.
- The navigation here is a faithful **moving body** (the BG cascade selecting each move neurally, OU on, the agent
  stepping), but it is NOT the flagship goal-optimal nav benchmark — the body's job in this task is to ENCOUNTER
  objects along a route so the perception write happens live. The agent's trajectory toward each waypoint is an
  environment scaffold (the "where to go next" drive that in the full nav stack comes from place cells / superior
  colliculus); the move it commits is the spiking winner, read not asserted.
- Validated at 4 candidate objects (chance 0.25), corridor layout, 3 seeds. Recall is exact (1.00) because the
  trained labeled-line read-out, though lossy/learned, is clean enough at this object count; a larger object
  vocabulary would lower the per-object recall toward the trained-route's signal-to-noise floor (the
  `funcint_perception_to_memory_trained_probe` characterizes that floor).

---

## Functions to trust-but-verify

- `build_navsee_bridge` — builds the ONE bridge (nav cascade + `cortex_it` + `language_output` + the dense plastic
  route) and trains+freezes the route.
- `_train_route` — the route Hebbian co-firing trainer + the non-route snapshot/restore (the clip-tension fix).
- `_cascade_select_move` — ONE neural nav decision (the BG disinhibition winner; the body's move).
- `_perceive_and_tag` — the LIVE perception write during the episode (render object → `cortex_it`, engram-record
  over the arrival window with the cascade live, `commit_engram_tag(region_filter=["cortex_it"])`).
- `recall_what_seen` — the after-episode recall by neural reactivation (`stimulate_tag` → read `language_output`).
- `_lesion_route` — the primary anti-cheat (zero the trained route synapses).
- `provenance_check` — the structural audit (every tag ⊆ `cortex_it`; recall's only write is `stimulate_tag`).
- `run_episode` / `verdict_from` — the episode loop (the body traversing the route, encountering + tagging objects)
  and the GO/PARTIAL/NEGATIVE gate.
