# N2 + N7 characterization verdicts (both DEFENSIBLE) + the honest full navigation-cheat ledger

**Date:** 2026-06-10
**Task:** the explicitly-named next step after the N9 fully-spiking reward loop — characterize the two remaining "perception" navigation cheats (N2 goal-rendering, N7 V1 receptive-field pre-initialization) as *defensible legitimate perception* or *host cheat*, and write the verdict into a "fully biologized navigation" finding.
**Standard applied:** the owner's BRAIN-BASED-ONLY bar — anything done by a host (non-neural) computation *between sensation and action* is a shortcut, *even if the host calculation is biologically correct*. Host code is legitimate **only** for (1) the environment (the world's state + **rendering the agent's sensory input**) and (2) the body (acting on the motor output). Everything cognitive (perception, salience, reward, value, dopamine, action selection) must be neurons/synapses.

This finding does two things. First, it closes the assigned N2/N7 characterization with grounded **DEFENSIBLE** verdicts. Second — and this is the load-bearing honesty step — it writes the **complete** navigation-cheat ledger as the code actually stands, which shows that navigation is **not yet** "fully biologized" by the strict bar: N2/N7/N6/N9 are closed, but two host computations (the superior-colliculus orienting reflex and the reward *value*) remain between sensation and action. Declaring nav "fully biologized" at this point would overclaim.

---

## Part 1 — N2: the goal rendered into the agent's retina. **Verdict: DEFENSIBLE (legitimate environment-rendered sensory input; beacon/landmark visual navigation).**

**What the code does** (`sim/visual_cortex.py:render_gridworld_to_image`, lines 155–207): given the agent's grid position and the goal's grid position, it paints a two-channel (ON / OFF) 32×32 image — the agent as a bright ON-channel block (intensity 1.0 with a 0.7 halo), the goal as a *dimmer* ON-channel block (intensity 0.5) at the goal's pixel location, and the grid lines as a faint OFF-channel edge signal. That image is then fed into the neural retina (`image_to_retina_drive` → `retina` region → V1 → V2 → IT → dorsal/motor planning) whenever `--enable-visual-cortex` is set.

**Why it is defensible, not a cheat.** The decisive question under the BRAIN-BASED-ONLY bar is: *does the goal's coordinate get fed into cognition as a number, or does it get rendered as a visible cue (light) that the neural retina then perceives?* Here the goal coordinate is used **only to decide where in the image to draw the light** — which is exactly what a physical environment does: a beacon at world-position (gx, gy) emits light that lands on the retina at the corresponding retinal position. The brain never receives (gx, gy) as a number; it receives a pattern of photoreceptor activations and must *perceive* the bright spot's retinal location and convert it to an action through the neural visual hierarchy. This is **beacon / landmark navigation** — a textbook, biologically real strategy (animals move toward visual cues directly associated with a goal). It falls squarely inside the bar's explicit legitimate-environment clause: "rendering the agent's sensory input (the retinal image the neural retina then receives)."

**Catalog grounding.** Cluster E ("Sensory perception & cortical encoding") catalogues *beacon/landmark sensors* as the project's established perception model; the catalog's own "where"-stream entry (E, line ~1493) draws exactly the distinction this verdict turns on — "navigation uses direct (gx, gy) **cheats** OR **beacon proxy**." N2 is the beacon proxy made literally visual (the cue rendered into the retina, not a coordinate handed to cognition).

**Honest simplifications (documented; environment/sensory-model choices, not cognitive cheats):**
1. **Single, trivially-distinct blob** — the goal is a clean bright spot with no clutter, distractors, occlusion, or object-recognition difficulty. This reduces perceptual *difficulty*, not the *kind* of computation. Enrichment: render the goal as a textured multi-pixel object among distractors so the plastic V2/IT must *recognize-then-localize* it (the recognition machinery exists downstream of V1). ~1–2 weeks; a capability addition, not a cheat removal.
2. **Top-down allocentric render** — the "retina" sees a god's-eye map (the agent sees its own avatar and the goal from above), not an egocentric first-person view. The retinotopic re-centering (agent-relative offset) is a legitimate parietal operation; an egocentric render with looming/size depth cues is an enrichment, not a correction.
3. **Goal always visible** — no line-of-sight, gaze, or distance falloff on visibility (the falloff is in the reward, not the render). An idealized always-foveated beacon.

**Bottom line (N2):** legitimate environment-rendered sensory input. The coordinate places the light; the brain perceives it. Not a cheat. The simplifications are defensible reductions / named enrichments.

---

## Part 2 — N7: V1 receptive fields pre-initialized as fixed Gabor filters. **Verdict: DEFENSIBLE (a faithful model of innate / early V1 orientation tuning; neural synaptic computation, not a host computation).**

**What the code does** (`sim/visual_cortex.py:build_v1_simple_weights` lines 76–152; `apply_v1_gabor_weights` lines 223–310): it constructs fixed oriented receptive fields (8 orientations × 4 spatial frequencies × 16×16 positions) as Gabor filters — a Gaussian envelope times a cosine carrier — and installs them as the **synaptic weights** on the `retina → cortex_v1_simple` pathway, replacing the random initialization. V1 simple-cell weights are then *fixed*; the downstream `cortex_v2` and `cortex_it` regions remain plastic (learn via spike-timing-dependent plasticity).

**Why it is defensible, not a host cheat — two independent reasons.**

1. **The computation is neural, not host.** Unlike the bar's listed cheats ("a prediction error computed by a Python formula," "a reflex that reads pixels and returns a cardinal in code," "an argmax over spike counts"), the Gabor here is *not* a function evaluated in host code between sensation and action. It is a set of **fixed afferent synaptic weights**. Retina spikes are weighted by those synapses and drive V1 simple cells to fire — a synaptic matrix-vector product performed by the substrate. The only thing a formula sets is *where the weights start*; the computation (retina → V1 spiking) is done by neurons. This is "initializing a synaptic weight matrix," not "computing a function off-substrate."

2. **The initialization models real innate biology.** V1 orientation tuning is **largely innate and present at eye-opening, before visual experience** — established in visually-inexperienced kittens (Hubel & Wiesel 1963), in ferret V1 before eye-opening (Chapman, Stryker & Bonhoeffer 1996), and scaffolded by **retinal waves** (spontaneous correlated retinal activity before the eyes open, whose ON/OFF structure can drive orientation-selective wiring; Meister et al. 1991; Gjorgjieva & Eglen). Pre-installing Gabor receptive fields stands in for the *converged endpoint* of this genetic + spontaneous-activity development. A V1 that is "born tuned" is what the biology says.

**Catalog grounding.** Catalog **E.08** ("V1 simple cells — oriented bar detectors," Kandel 6e Ch 22): "Orientation-tuned, position-specific receptive fields built from aligned LGN center-surround inputs (Hubel-Wiesel). Linear filter + threshold approximation — **Gabor-like RFs**." The visual_cortex module *is* the implementation of E.08 (the catalog's stale "missing" status predates the Cluster K v1 module). Catalog **L.05** ("Spontaneous-activity-driven refinement — retinal waves," Kandel 6e Ch 49 pp 1218–1222): the developmental route in which V1 receptive fields *emerge* from patterned spontaneous activity before experience — explicitly the deeper, fully-developmental version, and explicitly flagged in the catalog as an enrichment.

**Honest residual / named enrichment (NOT a cheat, but the deeper neural target by the project's own standard):** the strictest reading of "anything that CAN be spiking is made spiking" says the V1 receptive fields should *develop* from retinal-wave-driven plasticity on a *plastic* `retina → V1` pathway, rather than be installed by a formula at their converged endpoint. That is catalog L.05, and it is a real, fundable enrichment (~1–2 weeks: a stimulus-blind wave generator + a plastic V1 pathway + a developmental pretraining/freeze phase + a check that the *learned* receptive fields are orientation-tuned, ~30° half-width-at-half-height, with no nav regression). The critical anti-cheat there: the wave generator must be **stimulus-blind** (structured noise, *not* the gridworld image) — leaking the task image into "spontaneous" activity would be the cheat. Until that enrichment is built, the fixed Gabor is a **legitimate innate prior** (the converged endpoint of L.05), consistent with the project's endorsed "innate scaffold" pattern (the same status as the innate superior-colliculus reflex).

**Bottom line (N7):** a faithful innate-V1 model realized as fixed synaptic weights (neural computation). Not a host cheat. The activity-dependent developmental version (retinal-wave growth of the receptive fields) is a named enrichment, not a correction.

---

## Part 3 — the honest full navigation-cheat ledger (what is actually closed vs still host)

Characterizing N2/N7 is necessary but **not sufficient** to declare navigation "fully biologized." The strict BRAIN-BASED-ONLY bar must be applied to *every* computation between sensation and action. As the code stands:

| Axis / cheat | Mechanism in code | Strict-bar status |
|---|---|---|
| **Action selection (N6)** | Deployed nav uses the spiking commit-burst / spiking winner-take-all selection layer (`enable_commit_burst=True`, `readout_source="spiking_wta"`); the host argmax (`g11_bg_runner.py:2764`) is in the **pretraining** loop only, and even there reads *which* spiking pool fired (= the body reading the motor output). | **Spiking / closed.** |
| **Dopamine RPE (N9)** | Spiking SNc dopamine cell computing δ = r − V; `reward_us` (PPN-like) delivers the reward burst; the striosome critic subtracts V via GABA_B/GIRK. Completed this session. | **Spiking / closed.** |
| **Goal cue (N2)** | Goal rendered as a visible light-blob into the retina; coordinate places the light, never enters cognition; neural visual hierarchy perceives it. | **Defensible / closed** (this finding). |
| **V1 receptive fields (N7)** | Fixed Gabor afferent weights on retina→V1 (neural synaptic computation); innate-V1 prior; V2/IT plastic. | **Defensible / closed** (this finding). |
| **SC orienting reflex (N1)** | Host `sc_orienting_cardinal_from_image` reads the rendered image (pixels) and returns a cardinal direction that drives the matching cortex pool. It is a **weanable teaching scaffold** (`sc_reflex_wean_start`, ramps the reflex to zero as the learned circuit matures; default −1 = never wean). | **Host scaffold — NOT closed by the strict bar.** Defensible *as a weaning scaffold* (the bar explicitly endorses "innate reflex teaches a learned circuit"); the strict-bar target is a **spiking superior-colliculus** retinotopic map that computes the orienting signal in neurons. Wherever the reflex is left un-weaned, it is a live host driver. |
| **Reward value (N5)** | Host `sc_salience_offset_from_image` reads the rendered image at the old and new positions, computes the goal's retinal eccentricity each time, and sets `reward = sign(Δ eccentricity)` (`g11_bg_runner.py:6620–6635`). **Coordinate-free** (no gx,gy enter the reward), but a **host distance-formula**. The spiking `reward_us` population only *delivers* this scalar as a burst. | **Reward delivery spiking; reward COMPUTATION host — NOT closed by the strict bar.** This is precisely the bar's named cheat "a reward computed by a distance formula" / "a reflex that reads pixels." The strict-bar target is a **neural approach/looming detector** — a population whose firing *is* "the goal got closer," computed from the dorsal "where" signal across time, rather than a numpy eccentricity difference. |
| **Learned perception drive** | Host `sc_salience_offset_from_image` reads the goal's retinal offset and uses it to drive a *learned* `sensory → cortex_X` population (which then learns the where→action map). | Borderline: the offset is a retinotopic sensory feature (the kind the retina/SC compute), but it is extracted in host numpy. Same root as N1/N5 — the strict-bar version reads the offset from the neural SC/dorsal map. |

### What this means

- **Closed (strict bar):** N6 (spiking selection), N9 (spiking dopamine RPE).
- **Closed (defensible legitimate perception):** N2 (beacon cue), N7 (innate V1).
- **Open (strict bar) — the remaining host computations between sensation and action:** **N1** (the superior-colliculus orienting reflex — a host pixels→cardinal reader, currently a weaning scaffold) and **N5's reward *value*** (a host sign-of-eccentricity distance-formula; only its delivery is spiking). The shared root of both, plus the learned-perception offset drive, is the same: a host function (`sc_orienting_cardinal_from_image` / `sc_salience_offset_from_image`) reads the rendered image and returns a cognitive quantity (an orienting direction; an approach reward). The strict-bar replacement is a **spiking superior colliculus** that computes the goal's retinal position and its motion neurally, feeding (a) the orienting drive and (b) the approach signal that gates `reward_us`.

### Honest verdict on "fully biologized navigation"

Navigation is **biology-faithful in dopamine (N9 spiking RPE), action selection (N6 spiking commit), and the perceptual *representation* (N2 beacon cue + N7 innate V1)** — four axes genuinely closed. It is **not yet fully biologized by the strict "anything that CAN be spiking is made spiking" bar**, because the **superior-colliculus orienting reflex (N1)** and the **reward *value* (N5)** are still host pixel-formulas. A prior survey (`2026-06-08-remaining-nav-cheats-full-biologization-research.md`) judged N1/N5 "defensible" because they are *coordinate-free* and *biologically shaped* — a principled bar — but the owner's later, authoritative re-classification is explicit that *coordinate-free host computation is still a shortcut*, and that a spiking superior colliculus + a neural reward system are "the real target."

**Therefore the "fully biologized → single-instance unification" gate is honestly held, not passed.** Per the owner's "continue biologization autonomously" + "everything feasible spiking," the next biologization targets are pre-registered:
1. **N5 reward-value → a neural approach/looming detector** (cheaper-first: reuses `reward_us`, the existing retinal salience map, and the dorsal "where" signal; converts the *value* computation, not just the delivery).
2. **N1 SC reflex → a spiking superior-colliculus retinotopic map** (a 2D map + winner-take-all → orienting drive; also feeds the N5 approach signal — one spiking SC can close both).

These are genuine host→spiking conversions (feasible, so in-scope under the strict bar), to be scoped by the standing deep-research-first practice before building.

---

## Files / lines verified (trust-but-verify)

- `sim/visual_cortex.py` — N2 `render_gridworld_to_image` (155–207); N7 `gabor_kernel` (39–73), `build_v1_simple_weights` (76–152), `apply_v1_gabor_weights` (223–310).
- `research/runners/g11_bg_runner.py` — N1 reflex call + wean (6342–6365); N5 reward value (6620–6635); learned-perception offset drive (6373–6383); N6 spiking selection (`enable_commit_burst` default True, 355/3501; deployed vs pretraining-argmax 2764); `spiking_reward_us` delivery (1015–1022, 5336–5343, 6814 region).
- Catalog `sim-catalog/references/feature-catalog.md` — Cluster E (beacon/landmark perception); E.08 (V1 simple cells, Gabor RFs, Hubel-Wiesel); L.05 (retinal-wave developmental refinement).
- Prior survey: `research/findings/2026-06-08-remaining-nav-cheats-full-biologization-research.md` (its N2/N7 "defensible / CHARACTERIZE-only" verdicts are confirmed; its "after N5+N9, fully biologized" framing is *refined here* — N5's reward *value* and N1's reflex remain host under the strict bar).

## Status

N2 and N7 are characterized **DEFENSIBLE** (legitimate perception, grounded in catalog E.08 / L.05 and Hubel-Wiesel / retinal-wave biology) — the assigned task is complete. The honest full ledger shows navigation is **not yet fully biologized** by the strict bar: N1 (SC orienting reflex) + N5 (reward value) are the remaining host→spiking conversions, with a spiking superior colliculus as the shared target. The unification stays parked. Continuing autonomously to scope the cheapest-first spiking conversion (N5 neural approach detector) per the deep-research-first standing practice.
