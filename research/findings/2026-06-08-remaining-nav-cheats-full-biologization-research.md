# Remaining navigation cheats — full-biologization research (N5, N9, N2, N7)

**Date:** 2026-06-08
**Type:** Deep-research / findings (READ-ONLY; no code changed, no GPU run)
**Scope:** The four residual cheats in the gridworld navigation agent
(`research/runners/g11_bg_runner.py`) after N8 (thalamic disinhibition),
N6 (spiking accumulate-then-commit decision), and N1 (SC orienting reflex +
learned dorsal/PPC vision circuit) were biologized.
**Owner mandate (MEMORY.md):** "artificial life with a proper brain analogue,
biology-translatable insights; honest negatives under strict biology ARE the
scientific deliverable." This document is written to that bar — it labels two
of the four items as *defensible modeling choices, not cheats*, with the
developmental-neuroscience evidence, rather than inventing busy-work.

---

## EXECUTIVE SUMMARY (read this first)

| # | Cheat | One-line verdict | Cheapest grounded mechanism / why defensible |
|---|---|---|---|
| **N5** | Reward is coordinate-based (`reward=+1 if Manhattan dist ↓`, computed from raw `(gx,gy,x,y)`; even `--sensed-reward` recomputes the same coordinate distance) | **BIOLOGIZE (cheap, ~1 day)** | Replace the coordinate-distance reward with a **perceived-approach reward**: `reward = sign(Δ|offset|)` where `offset = sc_salience_offset_from_image(img)` — the *retinal eccentricity* of the goal blob the reflex already reads from pixels. This is appetitive/incentive-salience "getting-closer" reward (Berridge; Schultz "reward function 1 = positive reinforcer", catalog O.23 / C.27). The machinery already exists (`g11_bg_runner.py:106`) and is anti-cheat-clean (coordinates never enter). |
| **N9** | Dopamine is a scalar (`current_reward_signal` → broadcast DA; no spiking SNc computing a reward-prediction error) | **BIOLOGIZE (medium, ~3–5 days)** | Add a **learned scalar value/critic** `V` (an EMA or a small striosome-value readout) and feed the SNc pool the **RPE δ = r − V** (Rescorla-Wagner/TD; catalog C.22/C.28/C.30; Schultz 1998; Frémaux-Sprekeler-Gerstner 2013 spiking actor-critic). Reuses the **already-existing `snc` spiking region** (`g11_bg_runner.py:851`, IZH2007_DOPAMINE) + the `dopamine` neuromodulator's `from_reward` rule which **already subtracts a `reward_baseline`** (`sim/neuromodulators.py:962`). Cheapest grounded version = make that baseline a *learned* `V(s)` instead of a constant. |
| **N2** | Goal is rendered into the image at its coordinates; agent localizes it from pixels | **CHARACTERIZE (defensible perception, not a cheat)** | This is **beacon navigation** — a textbook navigation strategy where a salient visible goal is perceived and localized (Chan et al. 2012; SC retinotopic goal map, catalog A.07/H.25). The goal *is* a visible object; perceiving its location IS the task. The only honest simplification is that the blob is *trivially* visually distinct (no object-recognition difficulty), and apparent-size doesn't encode distance (eccentricity does). A non-cheat richer version = textured/cluttered scene + learned object recognition. |
| **N7** | V1 Gabor receptive fields are pre-set at init, not learned | **CHARACTERIZE (faithful innate/early V1, not a cheat)** | V1 orientation tuning is **present at eye-opening**, established by **spontaneous retinal waves before visual experience** (ferret P14; Hubel-Wiesel; catalog L.05/L.04/L.19; Crair-Gillespie-Stryker 1998; Smith-Häusser/Ko 2013). Pre-setting Gabors is a defensible model of *innate* V1. An activity-dependent version (retinal-wave Hebbian developmental pretraining) is an available *enrichment*, not a correction — and the catalog (L.05) already flags it as a future build. |

**Recommended order:**
1. **N5 first** (cheapest, highest-leverage, removes the single largest remaining coordinate leak from the *learning* loop, and it directly de-risks N9 because the perceived reward becomes the `r` the RPE consumes).
2. **N9 second** (builds on N5's perceived `r`; promotes the existing scalar DA to a spiking-SNc RPE — the headline biology win; composes with `--enable-tonic-da`).
3. **N2 and N7 are CHARACTERIZE-only** — write the honest verdict into the "fully biologized" definition (§"Fully biologized navigation" below). No build required unless the owner wants the *enrichment* versions (textured scene for N2; retinal-wave pretraining for N7), which are separate, larger efforts.

**Net:** after N5 + N9, the navigation loop is biology-faithful in all four canonical axes — **action** (N6 spiking decision), **reward** (N5 perceived approach reward), **dopamine** (N9 spiking-SNc RPE), and **perception** (N1 SC reflex + learned vision; N7 innate V1; N2 beacon localization). The residual simplifications are then all *defensible modeling choices*, not cheats. See the explicit enumeration in the final section.

---

## Diagnosis grounding (what the code actually does today)

All four claims were verified against the current source, not assumed:

- **N5 — reward is coordinate-based.** `g11_bg_runner.py:4710–4734`: the reward is
  `+1 if dist_after < dist_before else −1 else 0`, where `dist = manhattan(x,y) = |x−gx|+|y−gy|`
  (`g11_bg_runner.py:3562`, `:4010`, `:4704`) — raw coordinates. The `--sensed-reward`
  "beacon gradient" branch (`:4714–4727`) **also** recomputes `d_before`/`d_after` from
  `(gx,gy,x,y)` and only applies a `peak/(1+falloff·d)` cosmetic transform before taking
  the same sign — it is **not** coordinate-free. (The project's own comment at `:4710`
  calls the default "cheat: uses raw (gx,gy)"; the beacon comment at `:4711–4713` claims it
  "operates on the perceptual signal" but the code shows it does not — it operates on the
  coordinate distance.) This reward gates the reward-modulated cortico-striatal STDP via
  `bridge.core_config.current_reward_signal` (`:4836`).
- **N9 — dopamine is a scalar.** The reward feeds `current_reward_signal` (`:4836`,
  `:4870`), a single signed float that the bridge multiplies into the eligibility trace ×
  STDP path. `--enable-tonic-da` (`:2631`, `:3185`) promotes DA to a `dopamine`
  neuromodulator (`sim/neuromodulators.py:_default_dopamine_config`, `:914`), but its
  `from_reward` production rule still consumes `current_reward_signal` directly — there is
  **no population computing a reward *prediction error***. Critically, a **spiking SNc
  region already exists** (`g11_bg_runner.py:851–860`, `name="snc"`,
  `IZH2007_DOPAMINE`, `n_dopamine` neurons) but with `internal_density=0` and **no afferent
  projections** — it is a structural placeholder that does not compute anything; the teaching
  signal bypasses it entirely.
- **N2 — goal rendered at its coordinates.** `sim/visual_cortex.render_gridworld_to_image`
  (`:155–207`) paints the goal as a fixed dimmer ON blob (`0.5`) at
  `gx·pixels_per_cell, gy·pixels_per_cell` (`:190–199`). The agent reads it back from pixels
  via `sc_orienting_cardinal_from_image` (`:72`) / `sc_salience_offset_from_image` (`:106`),
  which take only the image array.
- **N7 — V1 Gabors pre-set.** `sim/visual_cortex.build_v1_simple_weights` (`:76–152`)
  constructs fixed Gabor RFs (8 orientations × 4 freqs × 16×16 positions) at init;
  `apply_v1_gabor_weights` (`:223`) overwrites the random `retina→cortex_v1_simple` weights
  with them. The module docstring (`:18–19`) states the design: "V1 simple weights are fixed
  Gabor at init; V2/IT learn via STDP." V2/IT downstream **are** plastic.

---

## N5 — coordinate-based reward → perceived-approach (phototaxis-like) reward

### Diagnosis
The reflex is innate, so navigation doesn't strictly *need* reward to perform — but the
reward is still in the loop gating BG plasticity (the learned dorsal/PPC circuit from N1,
and any future learned policy, depend on it). As long as that reward is computed from raw
`(gx,gy,x,y)`, the *learning* signal leaks coordinates even though the *action* signal no
longer does. This is the last coordinate leak in the learning loop.

### Biology-grounded mechanism: appetitive approach / incentive-salience reward
The biologically correct coordinate-free signal is a **perceived approach reward** — the
agent senses the goal getting closer *in its sensory field* and that approach is itself the
appetitive/consummatory teaching signal.

- **Catalog O.23 (three reward functions, Schultz 2016 NRN):** reward function **1 =
  positive reinforcer** (induces learning via DA × eligibility-trace plasticity) and
  function **2 = goal object for approach** are exactly the two functions the project should
  implement; approach-to-a-perceived-goal is the canonical operationalization. The project
  "cleanly implements function 1 … partially implements function 2" — a perceived-approach
  reward strengthens function 2.
- **Catalog C.27 (wanting vs liking, Berridge):** DA mediates "**wanting**" = incentive
  motivation = **approach behavior** toward a perceived goal (not hedonic "liking"). DA is
  "the incentive-salience / wanting signal." A reward that fires when the perceived goal
  grows nearer is precisely an incentive-salience/approach signal.
- **Literature:** approach/incentive-salience is "a motivational magnet quality of
  rewarding stimuli that makes them desirable goals … rendering approach behavior more
  likely" (Berridge & Robinson; motivational-salience review). In **beacon / sign-tracking**
  paradigms animals "move towards visual cues directly associated with the goal" — the visual
  proximity of the goal cue *is* the reinforcer (Chan et al. 2012). This is the navigational
  homologue of **phototaxis** (moving up a sensed intensity gradient), which is the simplest,
  most ancient form of an approach reward: the reinforcement is the *change in a perceived
  scalar* (here, goal proximity in the visual field), not a coordinate.
- **It is a legitimate biologization for the same reason the N1 reflex is:** the reflex is
  defensible because it reads the goal's *retinal position* from pixels (not coordinates);
  by the identical logic, reading the goal's *retinal proximity* (eccentricity magnitude)
  from pixels and rewarding its decrease is the matched reward-side biologization. The agent
  is allowed to perceive the goal (N2 is defensible perception, below); deriving an approach
  reward from that same percept is consistent.

### Concrete cheap mechanism (reuses existing machinery)
`sc_salience_offset_from_image(image)` (`g11_bg_runner.py:106–128`) already returns the
continuous goal offset `(dx, dy)` in grid-cell units, **read from the rendered image alone**
(6/6 unit tests pin that coordinates never enter — see
`research/findings/2026-06-08-Rank2-learned-vision-circuit-and-teacher-correction.md`). The
perceived offset *magnitude* `|offset| = hypot(dx, dy)` is the goal's retinal eccentricity.
The reward becomes:

```
off_before = sc_salience_offset_from_image(img_before)   # pixels only
off_after  = sc_salience_offset_from_image(img_after)
ecc_before = hypot(*off_before);  ecc_after = hypot(*off_after)
reward = +1 if ecc_after < ecc_before - eps else (-1 if ecc_after > ecc_before + eps else 0)
```

This is a near-drop-in for the existing sign-only reward at `:4729–4734`, just sourcing the
distance from pixels instead of `(gx,gy)`. (If `sc_salience_offset_from_image` returns `None`
because the agent is on the goal — both blobs merged — emit `+1` for arrival, matching the
co-located case.)

### Pitfall the owner flagged — apparent SIZE vs ECCENTRICITY
The render paints a **fixed-size** goal blob (`render_gridworld_to_image:190–199` always
spreads the goal over a 3×3 region at amplitude `0.5`, independent of distance). So a
"looming/size-based" proximity cue is **NOT** available — apparent size does not encode
distance in this world. The eccentricity (agent-blob centroid → goal-blob centroid distance
on the retina) **does** encode proximity, and that is what `sc_salience_offset_from_image`
computes. So the mechanism must use **centroid eccentricity, not blob size/intensity**. This
is exactly what the offset helper already returns; no looming model is needed or warranted.
(A second, subtler pitfall: on a discrete grid both the coordinate Manhattan distance and the
perceived Euclidean eccentricity move in lockstep for cardinal steps, so the *labels* the two
rewards emit are nearly identical — which is the point: the perceived reward is behaviorally
equivalent but coordinate-free, so it should not regress the nav score. The win is provenance,
not a different reward.)

### Cheap-first de-risk (falsifiable smoke, gated on the real nav score)
- **Smoke:** add `--perceived-approach-reward` (sources `r` from
  `sc_salience_offset_from_image`), run the flagship multi-goal-deterministic config (the one
  from CLAUDE.md "🎯 LATEST BREAKTHROUGH": `--moving-goal --goal-schedule multi
  --deterministic` + the cluster flags + `--enable-visual-cortex` + `--sc-orienting-reflex`)
  at **6 seeds** (project rule: 6-seed validation required). **Acceptance:** summed-reward
  metric within noise of the coordinate-reward baseline (no regression) — the perceived
  reward should track the coordinate reward because both encode goal approach. **Falsifier:**
  if the perceived reward systematically diverges (e.g. blob-centroid quantization on the
  32-px render makes `ecc` non-monotone for some diagonal steps → wrong-signed reward → score
  craters), the helper needs sub-pixel centroiding or a small hysteresis `eps`.
- **Anti-cheat:** assert the reward branch references **only** `img`/the offset helper, never
  `gx,gy,x,y` (grep the new branch; the helper's signature structurally excludes coordinates,
  same guarantee as the N1 reflex). No coordinate may enter the reward computation.
- **Multi-seed:** 6 seeds (42–44, 100–102 per the documented set) on grid-8 and grid-32.

### Cost estimate
**~1 day.** One new CLI flag + one reward branch in the runner (additive, default-off, **no
`sim/` edit**), reusing `sc_salience_offset_from_image`. A 6-seed × 2-grid smoke (~minutes
each on GPU, but **not run here** per constraints) plus the anti-cheat grep. This is the
highest leverage-to-cost item.

---

## N9 — scalar dopamine → spiking SNc computing a reward-prediction error

### Diagnosis
The reward feeds `current_reward_signal` and is broadcast as DA gain; even with
`--enable-tonic-da` the DA concentration tracks **raw reward**, not a **prediction error**.
There is no learned value/prediction, so the system is an **actor-only** architecture
(catalog C.30 / O.20 say exactly this) — it does policy *improvement* via DA-gated STDP but
has no policy *evaluation* (no critic), so it can settle into local optima a critic would have
escaped, and it cannot reproduce the canonical Schultz signatures (cue-shift, omission dip).

### Biology-grounded mechanism: spiking-SNc RPE (Schultz 1998; actor-critic)
- **Catalog C.22 (Schultz RPE), C.28 (TD error = δ), C.30 (actor-critic):** phasic VTA/SNc
  dopamine encodes the **reward-prediction error** δ. The minimal Rescorla-Wagner/TD form
  (Schultz 1998 Eq. 6a; catalog C.22 supplemental) is **δ = r(t) − P(t−1)** where `P` is a
  learned prediction; the full bootstrap form (catalog C.28) is **δ = r + γV(s′) − V(s)**.
  C.30 gives the anatomical mapping: **SNc/VTA DA = critic δ output; striosome-patch =
  critic state-value V(s); striatal matrix = actor preferences H(s,a); corticostriatal
  matrix synapses = actor weights modified by δ** (Houk-Adams-Barto 1995; Schultz 1998 Fig.
  9C). The project is the "actor-only, critic-missing" special case (C.30 sim-status).
- **The project ALREADY has the SNc region** (`g11_bg_runner.py:851`, `name="snc"`,
  `IZH2007_DOPAMINE`). Today it is silent (no afferents, `internal_density=0`). The
  biologization is to (a) make a small **value/prediction `V`** and (b) drive the SNc pool's
  firing (or the `dopamine` neuromodulator concentration) with **δ = r − V** rather than `r`.
- **Spiking realization exists in the literature** — this is a solved problem, not novel
  research:
  - **Frémaux, Sprekeler & Gerstner 2013** (*PLoS Comput Biol* 9(4):e1003024),
    "Reinforcement Learning Using a Continuous Time Actor-Critic Framework with Spiking
    Neurons" — the canonical **continuous-time spiking actor-critic**: a spiking critic
    population estimates `V`, its TD error modulates reward-modulated STDP. It specifically
    solves the "RL is discrete but spikes are continuous-time" mismatch that the project's
    per-step reward injection papers over. This is the closest published blueprint.
  - **Potjans, Diesmann & Morrison 2011** (*Front. Comput. Neurosci.*; "An imperfect
    dopaminergic error signal can drive temporal-difference learning", PMC3093351) — shows a
    **biologically realistic, non-ideal spiking SNc RPE still drives TD learning**. Directly
    de-risks the worry that a spiking (noisy, rate-coded) δ won't be clean enough.
  - **Joel, Niv & Ruppin 2002** (*Neural Networks* 15:535) — the standard review of
    **actor-critic models of the basal ganglia**, the anatomical mapping C.30 cites.
- **It composes with what's there.** `--enable-tonic-da` already moves DA into the
  neuromodulator subsystem as a concentration; its `from_reward` rule **already subtracts a
  `reward_baseline`** (`sim/neuromodulators.py:962–965`, with `reward = current_reward_signal
  − reward_baseline` at `:634–635`). **The cheapest grounded RPE = replace that constant
  `reward_baseline` with a learned `V`.** A single scalar EMA of reward (`R̄`, catalog O.21
  average-reward R-learning) is the minimal critic and is *already implicitly present* in
  `--adaptive-da`'s reward-EMA — catalog O.21 explicitly says declaring an `R̄`
  neuromodulator with a slow `from_reward` EMA and subtracting it "would unify the EMA-gating
  with the average-reward RL formalism." So the de-risked first step is an algorithmic δ; the
  *spiking* SNc is the second step.

### Cheapest grounded version (staged, each step falsifiable)
1. **Algorithmic δ (½–1 day).** Add `δ = r − V`, where `V` is a slow reward EMA `R̄`
   (R-learning; O.21). Feed `δ` (not `r`) into `current_reward_signal`. This is a
   Rescorla-Wagner critic — the smallest thing that is a genuine prediction error.
   **Acceptance:** flagship 6-seed summed reward ≥ the raw-reward baseline (a correct critic
   should match or beat actor-only — C.30/O.20 predict the evaluator escapes local optima).
2. **State-dependent value `V(s)` (1–2 days).** Make `V` a readout of a small **striosome /
   value population** (C.30 maps the critic's state-value to striosome-patch) driven by the
   perceived state (the N1 sensory→cortex code), trained by `δ` itself (bootstrapping). This
   is the actor-critic proper.
3. **Drive the spiking `snc` pool with δ (1–2 days).** Project the value/error onto the
   existing `snc` IZH2007_DOPAMINE region (give it afferents) so the **DA broadcast is a
   spiking SNc burst/dip** whose rate encodes δ — the literal biology (Frémaux 2013; Potjans
   2011). The neuromodulator concentration is then produced from SNc firing rather than the
   scalar.

### De-risk / anti-cheat
- **Canonical-signature smoke (cheap, biology-faithful, gated on a real metric):** the
  catalog gives the acceptance test for free (C.22/C.28 behavioral-validation): a 2-cue
  Pavlovian schedule — with a working critic, the DA signal must **shift from US to CS** over
  trials (cue-shift) and show an **omission dip** on reward omission. This is a stronger,
  more diagnostic falsifier than the nav score for whether the RPE is *real* vs cosmetic.
  Run it as a tiny instrumentation harness in addition to the nav-score regression.
- **Nav-score gate:** flagship multi-goal-deterministic 6-seed; δ-reward must not regress
  summed reward vs the raw-reward baseline.
- **Anti-cheat:** `V` must be learned from the reward/state, never seeded with `(gx,gy)` or
  true distance; assert no coordinate enters the critic. (If N5 is done first, `r` is already
  perceived/coordinate-free, so the whole RPE loop is coordinate-free.)
- **Compose check:** verify it stacks with `--enable-tonic-da` (don't double-count — if the
  neuromodulator path consumes δ, the per-synapse scalar path must not also apply δ).

### Cost estimate
**~3–5 days** for the full staged build (step 1 alone is ~½ day and already a genuine
prediction error). Step 1 needs no `sim/` edit (runner-side δ). Steps 2–3 likely touch the
neuromodulator/region wiring (`from_region_firing` already exists for a learned-value readout
— `sim/neuromodulators.py:902`), but additively/opt-in. **N9 is the headline biology win**:
it converts the project from "actor-only / scalar DA" to "actor-critic / spiking-SNc RPE,"
closing catalog C.22/C.28/C.30/O.20 in one arc.

---

## N2 — goal rendered into the image at its coordinates (CHEAT vs DEFENSIBLE?)

### Honest verdict: **DEFENSIBLE PERCEPTION, not a cheat — but characterize the simplification.**

**Why it is NOT a cheat.** The agent does not receive coordinates; it receives an **image**
in which the goal is a **visible object**, and it must **perceive and localize** that object
from pixels (which it does — `sc_orienting_cardinal_from_image`, `sc_salience_offset_from_image`,
the learned dorsal/PPC circuit). This is exactly how real animals navigate to a **visible
goal/landmark**:

- **Beacon navigation** is a textbook, biologically real strategy: animals "move towards
  visual cues directly associated with the goal"; "objects can serve as beacons, which
  require only recognition for effective goal localization" (Chan, Baumann, Bellgrove & Mattingley
  2012, *Front. Psychol.* 3:304, "From Objects to Landmarks"). Routes with salient landmark
  objects are learned better than routes without (ibid.).
- **The superior colliculus** maintains a **retinotopic map of behaviorally relevant goal
  locations** and "identifies salient points in the environment and coordinates orienting …
  toward such locations" (catalog A.07/H.25; Krauzlis et al.; "Goal Representations Dominate
  Superior Colliculus Activity", Boehnke & Munoz / White et al.). Perceiving a salient visible
  goal and orienting to it is the SC's actual job, not a shortcut.

So: **a salient visible goal that the agent localizes from pixels is the navigation TASK, not
a cheat.** Feeding the agent the goal's `(gx,gy)` *coordinates directly* would be the cheat
(and that is the N1/N5 cheat, being removed). Painting the goal as a perceptible object and
making the agent find it is legitimate.

**What the genuine simplifications are (characterize these honestly):**
1. **The goal is *trivially* visually distinct** — a single dim ON blob at a fixed amplitude
   (`0.5`), separable from the agent (`≥0.65`) by a simple threshold. There is no
   object-recognition difficulty, occlusion, clutter, or distractor. Real beacon navigation
   often requires recognizing the goal object among many. This is a *reduction of perceptual
   difficulty*, not a coordinate cheat.
2. **Fixed-size blob** — apparent size does not encode distance (see N5 pitfall); only
   eccentricity does. Fine for an allocentric top-down render, but it is not an egocentric
   first-person view where looming would carry depth.
3. **Top-down allocentric render** — the "retina" sees a god's-eye map, not the agent's
   first-person egocentric view. The agent-centred offset is computed by the helper, which is
   a legitimate retinotopic re-centering, but a fully egocentric agent would render from its
   own viewpoint.

**What a richer (non-cheat) version would require** (optional enrichment, not a correction):
- Render the goal as a **textured/multi-pixel object** among **distractor objects / clutter**,
  so localization requires **learned object recognition** (the V2/IT plastic pathway already
  exists downstream of V1 — `visual_cortex.py` docstring `:14–16`), not a threshold split.
- Optionally an **egocentric first-person render** with size/looming depth cues.
- This turns N2 from "trivially perceptible beacon" into "recognize-then-localize a goal in a
  scene," which is a meaningful perception upgrade — but it is a *capability addition*, and the
  current form is **not a cheat that needs removing**.

### De-risk / cost
**No build required for the verdict** (CHARACTERIZE). If the owner wants the enrichment:
~1–2 weeks (new render with clutter + retrain V2/IT recognition + 6-seed nav). The cheap
de-risk would be to add 1–2 distractor blobs and confirm the SC reflex + learned circuit still
localize the *true* goal (anti-cheat: the agent must not just pick the brightest/nearest blob
if a distractor is closer). But this is optional scope.

---

## N7 — V1 Gabor RFs pre-initialized, not learned (CHEAT vs DEFENSIBLE?)

### Honest verdict: **DEFENSIBLE — a faithful model of INNATE / early V1, not a cheat. The activity-dependent version is an enrichment the catalog already flags.**

**Why pre-set Gabors are biologically faithful.** V1 orientation tuning is **largely innate
and present at eye-opening**, established by **genetics + spontaneous activity (retinal waves)
before any visual experience**:

- **Catalog L.05 (Spontaneous-activity-driven refinement — retinal waves, Kandel 6e Ch 49
  pp 1218–1222):** "even before eyes open … the developing nervous system generates
  spontaneous patterned activity (retinal waves …) that drives the refinement of downstream
  connections … The brain is *self-organizing* its sensory representations **before
  experience arrives**." The catalog explicitly notes this is "missing as a mechanism" and is
  "likely useful" as a **developmental-pretraining** for the project — i.e., the activity-
  dependent V1 is a *future build*, and pre-set Gabors stand in for its converged endpoint.
- **Catalog L.04 / L.19 (critical periods; ocular dominance):** orientation/OD tuning forms
  in an early window; the *initial* map is laid down by spontaneous activity, then *refined*
  by experience. The pre-set map is the pre-experience endpoint.
- **Literature:** **Orientation selectivity is present at eye-opening and is primarily
  vision-independent** — in ferret, "orientation selectivity is formed by spontaneous activity
  before eye-opening (~P14)" and broad tuning is observable at eye-opening (Chapman & Stryker;
  Crair, Gillespie & Stryker 1998, *Science* 279:566; White, Coppola & Fitzpatrick 2001).
  **Hubel & Wiesel** found oriented RFs in very young / visually inexperienced animals.
  **Retinal waves carry the structure that builds it:** "temporally asynchronous retinal waves
  from ON and OFF retinal ganglion cells can drive V1 neurons selectively by their orientation
  tuning" and **stage III ON-OFF-asynchronous waves are "a perfect candidate for early
  development of orientation preference"** (Gjorgjieva & Eglen; "Spontaneous Retinal Waves …
  generate long-range horizontal connectivity in visual cortex", *J. Neurosci.* 40:6584,
  2020). Orientation selectivity even **fails to develop without ON-center RGC activity**
  (Chapman, *J. Neurosci.* 20:1922, 2000) — i.e., it is *spontaneous-activity-built*, not
  experience-required. (Mouse/cat have more experience-dependence than ferret, but the early
  map is still substantially innate.)

So: a V1 whose Gabor RFs are **set at init** is a faithful model of **innate orientation
tuning present at eye-opening** — the genetically/wave-specified prior. It is **not a cheat**;
it is the project asserting "V1 is born tuned," which is what the biology says. (The
downstream V2/IT *are* learned via STDP — `visual_cortex.py` docstring `:18–19` — so the
*experience-dependent* part of the ventral stream is already plastic.)

**What an activity-dependent (developmental) version would require** (enrichment, not
correction):
- A **retinal-wave generator** — structured spontaneous activity (correlated traveling
  bursts, stage II cholinergic / stage III glutamatergic ON-OFF-asynchronous) injected into
  the `retina` region during a **developmental-pretraining gate-open phase**, with the
  `retina→cortex_v1_simple` pathway plastic (Hebbian/STDP) instead of fixed.
- The project already has the substrate: per-pathway plasticity gates +
  `set_pathway_weights`, and `apply_v1_gabor_weights` shows how to address the pathway. The
  build = generate wave patterns + make the V1 pathway plastic + run a pretraining phase +
  freeze (catalog L.05 behavioral-validation: "generate retinal-wave-like input → train
  sensory→cortex pathway … → verify cortex develops coherent receptive fields (analogue of
  orientation columns)").
- **This is a real, fundable enrichment** (it would let the project *demonstrate* L.05 — V1
  tuning emerging from spontaneous activity — which is a genuine biology-translatable result).
  But it is a **capability/faithfulness addition**, not a correction of a cheat. Pre-set
  Gabors remain a defensible innate-V1 model in the meantime.

### De-risk / cost
**No build required for the verdict** (CHARACTERIZE). The enrichment (retinal-wave
developmental pretraining of V1) is ~1–2 weeks: wave generator + plastic V1 pathway +
pretraining/freeze + validate that the *learned* RFs are orientation-tuned (HWHH ~30°, catalog
E.08 behavioral-validation) and that nav score with learned-V1 ≈ nav score with pre-set Gabors
(no regression). Anti-cheat: the wave generator must be **stimulus-blind** (structured noise,
not the gridworld image) — the whole point is *pre-experience* refinement; leaking the task
image into "spontaneous" activity would be the cheat.

---

## The "fully biologized navigation" definition (after N5 + N9)

After the two recommended BIOLOGIZE fixes, here is **exactly** what is biology-faithful and
what residual simplifications remain — so the owner can judge whether "fully biologized" is met.

### Biology-faithful (the four canonical axes)
| Axis | Mechanism | Status / grounding |
|---|---|---|
| **Action selection** | Spiking accumulate-then-commit: Wang-2002 NMDA decision attractor → Lo-Wang/SC commit burst under OPN gating, urgency (Cisek) | **DONE (N6)**; catalog H.24/H.25/A.07; `g11_bg_runner.py:184–221` |
| **Reward** | Perceived-approach (incentive-salience / phototaxis-like): `sign(Δ goal-eccentricity)` read from pixels, no coordinates | **N5 (recommended)**; catalog O.23/C.27; reuses `sc_salience_offset_from_image` (`:106`) |
| **Dopamine** | Spiking SNc computing RPE δ = r − V with a learned value/critic (actor-critic) | **N9 (recommended)**; catalog C.22/C.28/C.30/O.20/O.21; Schultz 1998; Frémaux 2013; Potjans 2011; reuses `snc` region (`:851`) + `dopamine` NM (`neuromodulators.py:914`) |
| **Perception** | (a) Innate V1 Gabors (born tuned, retinal-wave/genetic prior); (b) SC orienting reflex from retinotopic image; (c) learned dorsal/PPC vision circuit; (d) beacon localization of a visible goal | **DONE (N1, N7-defensible, N2-defensible)**; catalog L.05/L.04/E.08, A.07/H.25; `visual_cortex.py`, `g11_bg_runner.py:72,106` |

### Residual simplifications (each labeled cheat vs defensible)
| Simplification | Cheat or defensible? | Note |
|---|---|---|
| Goal is *trivially* visually distinct (single blob, no clutter/recognition) | **Defensible reduction** (N2) | Beacon navigation is real; the only reduction is perceptual *difficulty*. Enrichment = clutter + learned object recognition (V2/IT exists). |
| V1 RFs pre-set, not grown from spontaneous activity | **Defensible (innate V1)** (N7) | Orientation tuning is present at eye-opening (ferret); waves build it pre-experience. Enrichment = retinal-wave developmental pretraining (L.05). |
| Critic is a small scalar/striosome value (not a full TD(λ) with γ-bootstrap and cue-shift) | **Defensible first step** (N9 staged) | Step 1 = Rescorla-Wagner δ = r − R̄ (O.21); full bootstrap V(s′) (C.28) is the deeper version. Catalog gives the cue-shift falsifier. |
| Single broadcast DA pool (A9+A10 collapsed; no per-projection-target diversity) | **Defensible abstraction** | Catalog C.23 supplemental (Schultz 2016): the *phasic* RPE is "remarkably similar across the population … graded not categorical"; a single broadcast phasic DA is "plausibly the biologically faithful abstraction." Diversity is in tonic/projection-target, not the phasic teaching signal. |
| Top-down allocentric render (not egocentric first-person) | **Modeling choice** (N2) | The retinotopic re-centering (agent-relative offset) is legitimate; an egocentric render is an enrichment, not a correction. |
| Discrete grid + per-step reward (not continuous time) | **Modeling choice** | Frémaux 2013 is the continuous-time spiking actor-critic if the owner ever wants to remove the discreteness; the current per-step injection is a standard reduction. |

**Bottom line for the owner:** after N5 + N9, **all four canonical navigation axes (action,
reward, dopamine, perception) are biology-grounded and coordinate-free**, and **every residual
simplification is a defensible modeling choice or a documented enrichment opportunity, not a
hidden cheat.** That meets a principled bar for "fully biologized navigation." The two largest
*enrichments* still on the table (clutter+object-recognition for N2; retinal-wave V1 pretraining
for N7) are genuine biology-translatable results worth building later, but they are capability
additions, not cheat removals.

---

## Sources

### Project code (verified `file:line`)
- `research/runners/g11_bg_runner.py` — N5 reward (`:4710–4734`, `:4714–4727`,
  `manhattan` `:3562`/`:4010`/`:4704`); `current_reward_signal` (`:4836`, `:4870`); the
  perceived-offset helper `sc_salience_offset_from_image` (`:106–128`) and cardinal reflex
  `sc_orienting_cardinal_from_image` (`:72–103`); the **existing spiking SNc region**
  (`:851–860`, `IZH2007_DOPAMINE`); tonic-DA wiring (`:2631`, `:3185–3190`); N6 spiking
  decision (`:163–221`).
- `sim/neuromodulators.py` — `_default_dopamine_config` with `from_reward` rule that already
  subtracts `reward_baseline` (`:914–967`, esp. `:962–965`, `:634–635`); `from_region_firing`
  for a learned-value readout (`:902`).
- `sim/visual_cortex.py` — N7 `build_v1_simple_weights` (`:76–152`),
  `apply_v1_gabor_weights` (`:223–309`); N2 `render_gridworld_to_image` (`:155–207`); design
  note "V1 fixed Gabor at init; V2/IT learn via STDP" (`:14–19`).
- Prior nav-biologization findings:
  `research/findings/2026-06-07-N1-SC-orienting-reflex-GO.md`,
  `2026-06-07-perceptual-bootstrap-deep-research.md`,
  `2026-06-08-Rank2-learned-vision-circuit-and-teacher-correction.md`,
  `2026-06-07-learned-visuomotor-precision-research.md`.

### Project feature catalog (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`)
- **C.22** Dopamine Reward Prediction Error (Schultz RPE) — δ encoding; cue-shift/omission
  signatures; Schultz 1998 Eq. 6a (δ = r − P); Kandel 6e Ch 43 pp 1068–1069.
- **C.27** Wanting vs Liking (Berridge) — DA = incentive-salience/wanting = approach; Kandel
  6e Ch 43 p 1068, Ch 41 p 1038.
- **C.28** TD error = algorithmic phasic DA (δ = r + γV(s′) − V(s)); **C.30** Actor-critic
  mapping (SNc=δ, striosome=V(s), matrix=actor); **C.31** bootstrapping; **C.32** two-component
  DA — Sutton & Barto Ch 6/11; Schultz 1998 Fig. 9C; Houk-Adams-Barto 1995; Barto 1995.
- **O.20** Generalized Policy Iteration (actor-only without a critic), **O.21** average-reward
  R-learning (R̄ = learned baseline; maps `--adaptive-da` EMA), **O.22** striatal action-value,
  **O.23** three reward functions (reinforcer / approach-goal / emotion) — Sutton & Barto Ch
  4/11/15; Schultz 2016 NRN & J. Neural Transm.
- **C.23** DA-subpopulation diversity is graded-not-categorical; single broadcast phasic DA is
  a faithful abstraction (Schultz 2016 NRN pp 12–13).
- **A.07 / H.24 / H.25** Subcortical BG loops / saccade generator / SC topographic saccade-goal
  map — Kandel 6e Ch 35 pp 868–882.
- **E.08 / E.09 / E.10** V1 simple/complex cells, orientation columns (Hubel-Wiesel; Gabor RFs)
  — Kandel 6e Ch 22–23.
- **L.04** Critical periods (visual/language/social), **L.05** Spontaneous-activity-driven
  refinement (retinal waves; "self-organizing before experience"; flagged as a future
  developmental-pretraining build), **L.06** activity-dependent refinement is NMDAR-dependent,
  **L.19** ocular-dominance critical period — Kandel 6e Ch 48–49.

### Peer-reviewed literature
- Schultz W., Dayan P., Montague P.R. (1997) "A neural substrate of prediction and reward",
  *Science* 275:1593. (RPE; cited in catalog C.22.)
- Schultz W. (1998) "Predictive reward signal of dopamine neurons", *J. Neurophysiol.* 80:1.
  (δ = r − P; cue-shift; omission dip.)
- Hollerman J.R. & Schultz W. (1998) "Dopamine neurons report an error in the temporal
  prediction of reward during learning", *Nat. Neurosci.* 1:304. (Cue-shift graded with
  learning; omission dip — catalog C.22 supplemental validation criterion.)
- Joel D., Niv Y., Ruppin E. (2002) "Actor–critic models of the basal ganglia: new anatomical
  and computational perspectives", *Neural Networks* 15:535.
  https://www.sciencedirect.com/science/article/abs/pii/S0893608002000473
- **Frémaux N., Sprekeler H., Gerstner W. (2013)** "Reinforcement Learning Using a Continuous
  Time Actor-Critic Framework with Spiking Neurons", *PLoS Comput. Biol.* 9(4):e1003024.
  https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003024 — **the
  spiking actor-critic blueprint for N9.**
- Potjans W., Diesmann M., Morrison A. (2011) "An imperfect dopaminergic error signal can
  drive temporal-difference learning", *Front. Comput. Neurosci.* (PMC3093351).
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3093351/ — a realistic noisy spiking SNc RPE
  still drives TD learning (de-risks N9 step 3).
- Glimcher P.W. (2011) "Understanding dopamine and reinforcement learning: the dopamine reward
  prediction error hypothesis", *PNAS* 108(Suppl 3):15647.
  https://www.pnas.org/doi/pdf/10.1073/pnas.1014269108
- Berridge K.C. & Robinson T.E. — incentive-salience / "wanting vs liking" (motivational
  salience; DA = wanting = approach). "Disentangling pleasure from incentive salience and
  learning signals in brain reward circuitry", *PNAS* 108(Suppl 3):15647-ff
  https://www.pnas.org/doi/10.1073/pnas.1101920108
- Chan E., Baumann O., Bellgrove M.A., Mattingley J.B. (2012) "From Objects to Landmarks: The
  Function of Visual Location Information in Spatial Navigation", *Front. Psychol.* 3:304.
  https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2012.00304/full —
  **beacon navigation is a real strategy (N2 verdict).**
- Krauzlis R.J. / White B.J. et al. "Goal Representations Dominate Superior Colliculus Activity
  during Extrafoveal Tracking", *J. Neurosci.* 28:9426 (PMC2698013).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC2698013/ — SC retinotopic goal map (N2/N1).
- Hubel D.H. & Wiesel T.N. (1962) "Receptive fields, binocular interaction and functional
  architecture in the cat's visual cortex", *J. Physiol.* 160:106. (Oriented V1 RFs;
  catalog E.08.)
- Crair M.C., Gillespie D.C., Stryker M.P. (1998) "The role of visual experience in the
  development of columns in cat visual cortex", *Science* 279:566. (Orientation map present
  before/at eye-opening; N7.)
- Chapman B. (2000) "Cortical Cell Orientation Selectivity Fails to Develop in the Absence of
  ON-Center Retinal Ganglion Cell Activity", *J. Neurosci.* 20:1922.
  https://www.jneurosci.org/content/20/5/1922 — orientation selectivity is spontaneous-
  activity-built (N7).
- "Spontaneous Retinal Waves Can Generate Long-Range Horizontal Connectivity in Visual
  Cortex", *J. Neurosci.* 40:6584 (2020). https://www.jneurosci.org/content/40/34/6584 —
  stage III ON-OFF-asynchronous waves build orientation preference before eye-opening (N7).
- Sutton R.S. & Barto A.G. *Reinforcement Learning: An Introduction* (2nd ed.) — Ch 6 (TD),
  Ch 7 (eligibility traces / TD(λ)), Ch 11 (actor-critic, average-reward R-learning), Ch 15
  (GPI). (Catalog C.28/C.29/C.30/O.20/O.21.)
- Kandel et al. *Principles of Neural Science* 6th ed. — Ch 22–23 (visual cortex/V1), Ch 35
  (saccades/SC), Ch 43 (dopamine/reward), Ch 48–49 (synapse formation/elimination, critical
  periods, retinal waves).
