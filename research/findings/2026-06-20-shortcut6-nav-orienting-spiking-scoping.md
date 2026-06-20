# Shortcut #6 (nav SC orienting read-out) — re-opened: scoping the GENUINE point-neuron re-orient mechanism before any "closed" (2026-06-20)

**Type:** READ-ONLY deep-research scoping gate (the project's standing deep-research-first step before committing
build/GPU resources). NO code edits, NO experiments, NO GPU. Single deliverable = this doc. Stayed on `main`,
read-only.
**Why re-opened:** the owner re-opened #6. The status ledger marks it "✅ CLOSED (characterized honest-negative)"
(`2026-06-20-shortcut-burndown-status.md:33`), but an audit of the actual artifacts shows **the prescribed fix was
started and ABANDONED mid-build, not run to ground** — the population-vector build doc
(`2026-06-20-burndown-6-popvector-readout-build.md`) has its decisive grid-32 confirm section as an empty
`<!-- FILL -->` placeholder (`:73-76, :90, :99`) and only a grid-8 smoke at a **mis-calibrated** divisive operating
point (`gain=1`, which its own text says "OVER-ATTENUATES the SC drive", `:64-67`). The "closed" verdict rests on a
**different knob's** grid-32 sweep (`SC_CORTEX_W` drive strength, `2026-06-20-nav-sc-drive-reorient-derisk.md`), NOT
on the geometry fix. So the genuine point-neuron mechanism has NOT actually been attempted at faithful scale.
**The boundary under repair:** `2026-06-20-nav-sc-drive-reorient-derisk.md` (the stuck-N NEGATIVE).
**The prescribed fix (un-finished):** `2026-06-20-nav-readout-geometry-deep-research.md` (Option A pop-vector +
divnorm + the #4 WTA ring) → `2026-06-20-burndown-6-popvector-readout-build.md` (built, abandoned at the grid-8 stage).
**Owner standard:** BRAIN-BASED-ONLY. A spiking organ that underperforms the host scaffold IS a valid deliverable —
but only AFTER the real mechanism is attempted at faithful scale; an abandoned mid-build is not an "earned negative".
**Method:** verified the load-bearing read-out math + the render geometry + the drive-injection path line-by-line
against `g11_bg_runner.py`; cross-checked the prior deep-research doc's claims against the code; cross-checked the
canonical SC re-orient biology (catalog H.25/E.03/A.07 + Kandel 6e Ch 35 + current spiking-SC literature, sources at
end). Every load-bearing claim is anchored to file:line or a published source.

---

## TL;DR — diagnosis (1-2 sentences)

The deployed `sc_map → cortex_X` read-out is an **un-normalized, un-competitive half-plane linear ramp**
(`g11_bg_runner.py:299-300`) that mass-codes ("how much SC fires") instead of position-coding ("where the bump is"),
so it cannot track a moved goal — and the prescribed canonical fix (a population-VECTOR read-out + bump-mass divisive
normalization + the existing #4 WTA ring) was **built but abandoned at a mis-calibrated grid-8 smoke**, so it has
never been tested at faithful scale. **The geometry (egocentric retina → retinotopic bump) is verified CORRECT, the
fix is verified PURE POINT-NEURON, and the residual re-orient-specific risk (bump-attractor hysteresis on goal change)
has a named point-neuron remedy (collicular fixation-zone reset / inhibition-of-return).** ⇒ #6 is a genuine BUILD
to FINISH, not a closed negative.

---

## 1. DIAGNOSIS — why it orients statically but does not re-orient (verified against the code)

### 1a. The pipeline geometry is CORRECT (this is the first thing to rule out — and it is sound)

A bump that does not move with the goal would trivially explain a stuck read-out, so verify it first. It is correct:

1. **The SC has its OWN egocentric eye.** During the episode (`g11_bg_runner.py:6813-6822`) the SC retina region
   `sc_retina` is driven by `render_egocentric_goal((x,y),(gx,gy))` (`:6814`), which paints the goal as a single dim
   blob at `c + (goal − agent)·ppc` (`render_egocentric_goal:191-192`) — i.e. **the blob position directly encodes the
   goal's egocentric bearing**, and it MOVES when the goal moves. (The main `retina` region stays allocentric for the
   visual cortex / N5 reward — a deliberate split, `:6810-6811`.)
2. **The retinotopic pooling reads that egocentric eye** — `ret0 = sc_retina` (`:236`, "the SC's own egocentric eye"),
   and stage-1 pools each `sc_map` site's 2×2 ON block from it (`:252-264`). So `sc_map` is a faithful retinotopic map
   of the goal bearing.
3. **The bump is a clean single peak** — `sc_map↔sc_fs` Mexican-hat + `sc_map` recurrent (`:265-278` + framework-built)
   = a point-neuron continuous-attractor.
4. The read-out's `ddx, ddy = sx − sc_center, sy − sc_center` (`:286`) correctly measures the bump's offset **from the
   foveal centre** (`sc_center = (SCN−1)/2`, `:240`). When the goal is east, the bump sits east-of-fovea, `ddx > 0`.

**⇒ the geometry is sound; the bump tracks the goal.** The failure is downstream, in stage 3 (the bump→cardinal
decode). This rules out the "bump doesn't move" / "wrong render" class and the on-file "reentrant thal→cortex loop
holds the old orientation" hypothesis at the SC level — the SC bump itself is NOT stuck (it is fed a fresh egocentric
render every step); what is stuck is the *decode of* the bump into a cardinal.

### 1b. The read-out is mass-coding, not position-coding (the verified two structural flaws)

For each cardinal `a`, every `sc_map` site projects to `cortex_a` with weight (`:299-300`, the `popvector=False`
default):

```
ddx, ddy = sx − sc_center, sy − sc_center
wv = {"E": max(0, ddx), "W": max(0, −ddx), "N": max(0, ddy), "S": max(0, −ddy)}[a]
w  = w_sc_cortex * wv          # w_sc_cortex = SC_CORTEX_W, default 18 (:201, :4433)
```

So the East-pool drive is `Σ_sites max(0, sx−c)·sc_map_activity(sx,sy)` — a **signed half-plane LINEAR RAMP**, a
projection of the activity-weighted coordinate onto a fixed cardinal axis. This is NOT a position decode, for two
compounding reasons (these are the prior deep-research's flaws #i/#ii, re-verified):

- **(i) No normalization by bump mass.** The pool drives are an *un-normalized* weighted SUM, so they scale with the
  *total* `sc_map` activity, not the *location* of the mass. A centroid is `Σ(pos·act)/Σ(act)` — the `Σ(act)`
  denominator is what turns "weighted mass" into "where". The ramp omits it, so a brighter/bigger bump lifts all four
  half-plane sums together. This is exactly why raising `SC_CORTEX_W` 18→150 only over-drove all four pools toward
  uniform (the NEGATIVE's grid-8 `[121,117,105,107]`, `2026-06-20-nav-sc-drive-reorient-derisk.md:90`): a global gain
  on an un-normalized sum cannot sharpen a position read-out.
- **(ii) No competition between the four cardinals.** Each `sc_map_to_cortex_a` pathway (`:308`) is an INDEPENDENT,
  purely additive projection — there is no lateral inhibition *between* `cortex_N/E/S/W`. So the read-out has no way to
  turn "East's drive slightly exceeds the others" into "East wins, the rest suppressed". The four drives rise/fall
  together with bump size; the **winner's MARGIN does not widen**. The cascade's intrinsic structural N-bias (the agent
  pins to the top edge, pos-row 31) then dominates the tiny un-sharpened margin → the **stuck-N, goal-INVARIANT** action
  distribution the NEGATIVE documented (N ~0.45–0.52 in EVERY phase, every drive level, `…drive-reorient-derisk.md:133-147`).

The host positive control (`sc_orienting_cardinal_from_image:144-155`) does for free *both* halves the spiking
read-out is missing: it reads the goal-blob **centroid** (`goal_xs.mean()`, normalized position) and returns
`argmax(|dx|,|dy|)` (a hard competition). The spiking read-out has neither the normalization nor the argmax.

### 1c. WHY this presents as "orients statically, fails to re-orient" specifically

The static-vs-re-orient split (which the NEGATIVE measured cleanly: phase-0 acquire ≈ host at strong drive, post-change
catastrophic) is a direct consequence of (i)+(ii), NOT a separate loop-hysteresis cause:

- **Static acquisition** survives because, for a fixed goal, the cascade can be tuned (a stronger stable bump → a
  stronger consistent push) to hold ONE cardinal — sc_w150 phase-0 dropped to 1.54 (`…derisk.md:108`). Holding a fixed
  bias does not require a *position decode*; it only requires a consistent net drive.
- **Re-orient** fails because, when the goal moves, the bump moves correctly (§1a) but the **un-normalized,
  un-competitive read-out's output does not change enough to override the N-bias** — the margin between cardinals is
  set by bump *mass*, which barely changes, not by bump *position*, which is what moved. So the action distribution
  stays goal-invariant.

**⇒ the failure is a read-out-GEOMETRY problem (mass-coding + no competition), fully attributable to two missing
point-neuron mechanisms, exactly as the prior deep-research concluded — and NOT the on-file reentrant-loop hysteresis
hypothesis (the SC bump is re-rendered fresh every step; it is not the stuck element).** That said, §2/Option E below
flags a *genuine* hysteresis risk that appears only AFTER the geometry fix is in (the bump-attractor + a hard WTA can
themselves resist re-orient) — which is why the build must measure the re-orient metric, not just acquisition.

### 1d. The decisive new finding — the prescribed fix is UNFINISHED, not falsified

The status ledger's "closed honest-negative" (`burndown-status.md:33`) is **not supported by a faithful-scale test of
the geometry fix**. The evidence:

- The population-vector build (`burndown-6-popvector-readout-build.md`) **was implemented** (`install_spiking_sc_wiring
  (popvector=True)`, `g11_bg_runner.py:287-296`, verified present + correct: cosine projection
  `wv = max(0, û_a·u_site)`, bounded [0,1]) and committed (`048ea203`).
- But its **grid-32 faithful confirm is an empty placeholder** (`:73-76` `<!-- FILL: the per-phase action
  distribution … + the scramble lesion -->`; `:90` Verdict = `<!-- FILL -->`; `:99` "subsequent commits … FILL").
- Its only data is a **grid-8/480 smoke at the DEFAULT divisive op-point (`sigma=1, gain=1`)** where the pop-vector arm
  was still stuck-N (`:58, :61-62`) — and the doc's OWN mechanistic read says that default `gain=1` (calibrated for the
  conversational cortex's O(1) drives) **"OVER-ATTENUATES the SC drive … crushing the SC contribution so the cascade
  N-bias + OU win regardless of the (now-correct) cosine geometry"** (`:64-67`). It explicitly flags the next step as
  "calibrating it to the nav SC drive scale … within the prescribed A+B, not a config-search" — and never did it.
- The "closed" ledger row instead cites the grid-32 sweep of `SC_CORTEX_W` (the **drive-strength** knob), which the
  prior deep-research already proved is the WRONG lever (`…readout-geometry-deep-research.md:73-75`). The ledger
  conflates "the drive knob doesn't fix it at grid-32" (true, tested) with "the geometry fix doesn't fix it at
  grid-32" (NOT tested — only a mis-calibrated grid-8 smoke).

**⇒ #6 is a BUILD-IN-PROGRESS abandoned before its decisive test, mislabeled "closed".** The genuine point-neuron
mechanism (pop-vector + correctly-calibrated divnorm + the #4 WTA ring) must be FINISHED and run at faithful grid-32
before any honest "closed".

---

## 2. RANKED OPTIONS (biologically-grounded; ranked by leverage × cheapness for re-orient)

Each: mechanism · biology source (verified) · reusable machinery · point-neuron-vs-dendrite · expected cost/failure.

### Option A (TOP — FINISH the abandoned build) — calibrate the bump-mass divisive normalization to the nav SC drive scale, then run grid-32

- **Mechanism:** the pop-vector geometry (cosine-tuned weights, ALREADY built, `:287-296`) is necessary but inert
  unless the bump-mass normalizer is set to the SC drive scale. The default `gain=1` over-attenuates (§1d). Calibrate
  the `input_divisive_norm` `(sigma, gain)` on the four `cortex_X` pools so the normalized drive `drive/(σ+gain·mean)`
  lands in the cascade's responsive band (the SC drive is O(tens of pA); `gain=1·mean` crushes it). This is the
  remaining specified part of the prescribed Option A+B — a calibration of an existing primitive, not new mechanism.
- **Biology:** Goossens–Van Opstal SC spike-vector decode + Carandini–Heeger divisive normalization (catalog H.25 SC
  saccade map / E.03 population vector & vector averaging / E.05 lateral-inhibition–as-divisive). The normalizer is the
  algorithmic motif that makes a population code report direction not amplitude.
- **Reusable machinery:** `install_spiking_sc_wiring(popvector=True)` (built); `input_divisive_norm`
  (`sim/bridge.py:6048`, guarded no-op, already wired by `run_moving_goal_episode` when `sc_popvector_readout=True`,
  `:228-231` docstring); the `_nav_sc_popvector_readout_derisk.py` probe (built, has host + ramp + popvector + scramble
  arms). The calibration is a kwarg sweep on `sc_popvector_divnorm_sigma/gain` (`:37`).
- **Point-neuron vs dendrite:** PURE POINT-NEURON (feedforward cosine weighted sum + an existing divisive primitive).
- **Cost / failure:** a small gain/sigma grid (3–4 points) at grid-8 to find the responsive band, then ONE grid-32
  faithful confirm. Failure mode: if NO (σ,gain) makes the action distribution track the goal, the normalization is not
  the missing piece and Option B (competition) is load-bearing — escalate to B, don't keep sweeping A.

### Option B (TOP, COMPLEMENTARY — route the SC drive into the existing #4 WTA ring) — competition between the four cardinals

- **Mechanism:** add inter-cardinal competition so the cardinal with the largest normalized pop-vector drive suppresses
  the others (turn a small winner-margin into a decisive choice). The project ALREADY ships this: the #4
  `sel_X`/`commit_X` ring (Wang-2002 NMDA accumulator + cross-pool inhibition + Lo-Wang commit burst,
  `g11_bg_runner.py:446-479`, default-on, `2026-06-19-spiking-decision-default-on-GO.md`, 1.16× host). The faithful
  `--spiking-sc` config already routes `readout_source="spiking_wta"` so the SC drive flows
  `sc_map → cortex_X → str_D1_X → … → thal_X → sel_X` — meaning B's competition is ALREADY downstream. The question is
  whether the competition is far enough downstream that the un-sharpened `cortex_X` margin is already swamped by the
  N-bias before it reaches `sel_X`; if so, sharpen earlier (lateral inhibition directly between `cortex_N/E/S/W`, or
  feed the pop-vector drive closer to `sel_X`).
- **Biology:** the ring of action neurons (local excitation + global inhibition → a single bump) is the
  Frémaux–Sprekeler–Gerstner (2013) spiking-actor read-out; the SC superficial-layer "temporal WTA … chooses a winner"
  + deep-layer bump attractor is the canonical SC selection circuit (BMC Neurosci spiking-SC model; see Sources).
  Catalog A.04 (selective BG disinhibition WTA).
- **Reusable machinery:** the entire #4 ring (built, default-on). Worst case B is a routing/gain change, not a new
  build.
- **Point-neuron vs dendrite:** PURE POINT-NEURON (LIF Izhikevich ring, the deployed #4 default).
- **Cost / failure:** routing/gain tweak. Failure mode: a too-hard ring gain (`sel_recurrent_weight`) locks the first
  winner and resists re-orient — the #4 GO already tuned this to 0.3; if re-orient still sticks, that points at Option E.

### Option E (the re-orient-SPECIFIC remedy, only if A+B still stick) — a goal-change reset / inhibition-of-return on the bump + ring

- **Mechanism:** the deep SC is a **bump attractor** (short-range excitation + long-range inhibition) — it has
  *hysteresis by design*: the established bump self-sustains and resists moving. Biology re-orients by a **fixation-zone
  / omnipause reset**: rostral-SC fixation neurons inhibit the saccade generators ("Don't orient!") and their release
  permits a NEW bump ("Orient!"), and **inhibition-of-return** suppresses the just-attended location so attention can
  move on. A point-neuron analogue: when the perceived goal-bearing changes substantially (detectable on-substrate as a
  drop in `sc_rostral` foveation drive or a transient mismatch), inject a brief global inhibitory "reset" pulse into
  `sc_map` (and/or the `sel_X` ring) that collapses the old bump/winner so the fresh egocentric render can re-establish
  the new one. This is the spiking version of the host's implicit per-step recompute.
- **Biology (verified):** SC deep-layer bump attractor + rostral fixation neurons → brainstem omnipause neurons
  (Munoz; "the SC and its control of fixation via projections to omnipause neurons"); inhibition-of-return as a
  collicular/attentional reset (see Sources). Catalog A.07 (SNr→SC tonic inhibition — the disinhibition gate that, when
  re-applied, *is* a reset). The project ALREADY has a foveation read-out (`sc_map → sc_rostral` broad Gaussian,
  `:311-335`) that fires graded with how central/eccentric the bump is — a ready-made goal-change detector.
- **Reusable machinery:** `sc_rostral` (built, `:316-335`); the framework's GABAergic regions + `transmission_gate` /
  per-region drive injection (a goal-change-triggered inhibitory pulse is a runner-side `cp_external_input_current`
  write into `sc_map`/`sel_X`, the same injection pattern as the SC drive at `:6821`). The neuromodulator subsystem's
  `excitability_drive` (negative) could deliver a tonic reset window.
- **Point-neuron vs dendrite:** PURE POINT-NEURON (a transient inhibitory current + an existing graded foveation
  read-out as the trigger). NOT dendritic.
- **Cost / failure:** moderate — needs a goal-change detector calibrated on-substrate (the `sc_rostral` drop, NOT a
  host "goal moved" flag, to stay BRAIN-BASED). Failure mode: if the reset is too frequent/strong it destroys static
  hold (re-introduces chance behavior); too weak and the old bump persists. This is the option most likely to itself
  become a point-neuron BOUNDARY (calibrating an on-substrate reset trigger without a host goal-change signal is the
  hard part) — so it is ranked AFTER A+B, attempted only if the bump/ring hysteresis is the actual residual.

### Option C (ablation, lower rank) — sharper directional pooling KERNEL (Gaussian wedge per cardinal)

- **Mechanism:** keep the additive read-out but replace the broad ramp with a narrow Gaussian-weighted wedge centred on
  each cardinal axis, so off-axis sites stop leaking into the wrong cardinal. Treats the *symptom* (leak) not the
  *cause* (no normalization, no competition).
- **Biology:** cosine-tuning width (E.03/H.17) — a narrow tuning curve = sharper discrimination. Point-neuron (a
  weight-formula change).
- **Cost / failure:** cheap, but without A's normalization or B's competition it still mass-codes within the wedge →
  likely a partial improvement that doesn't reach host. Useful as an ABLATION to attribute the lift (does the cosine
  geometry alone, un-normalized, move the metric?), not as the primary fix.

**Ranking summary:** **A (finish + calibrate the divnorm) + B (the existing #4 WTA ring)** are the primary build —
both pure point-neuron, both with published precedents AND already-implemented in the codebase (A's geometry is built
but un-calibrated; B is built + default-on). **E** is the re-orient-specific remedy held in reserve for genuine
bump/ring hysteresis (the only option with a real chance of being a sub-boundary). **C** is an ablation.

---

## 3. REUSABLE PROJECT MACHINERY (most of the fix already exists)

- **`install_spiking_sc_wiring(popvector=True)`** (`g11_bg_runner.py:287-296`) — the pop-vector geometry, BUILT +
  verified correct (cosine projection, bounded [0,1]). Only its divnorm calibration is missing.
- **`input_divisive_norm`** (`sim/bridge.py:6048`, `cfg.enable_input_divisive_norm` + `BrainRegion.input_divisive_norm`)
  — the Carandini–Heeger normalizer, guarded no-op when off, already routed by `run_moving_goal_episode` when
  `sc_popvector_readout=True` (`g11_bg_runner.py:228-231`). The Option-A normalizer; needs only `(sigma, gain)`
  calibration to the nav SC drive scale.
- **The #4 spiking-WTA ring** — `enable_spiking_wta_readout` + `sel_X` (Wang attractor) + `commit_X` (Lo-Wang commit),
  `g11_bg_runner.py:446-479`, default-on (`2026-06-19-spiking-decision-default-on-GO.md`). The Option-B competition;
  already downstream of the SC drive in the faithful config.
- **`sc_rostral` foveation read-out** (`g11_bg_runner.py:311-335`) — `sc_map → sc_rostral` broad Gaussian; fires graded
  with bump centrality. A ready-made on-substrate goal-change detector for Option E.
- **The host positive control** — `sc_orienting_cardinal_from_image` (`:144-155`, centroid+argmax) and the graded
  sibling `sc_salience_offset_from_image` (`:158-180`, continuous (dx,dy)). The scaffold to approach + a ready "where"
  reference.
- **The retinotopy-scramble LESION** — `install_spiking_sc_wiring(scramble=True)` (`:244-249`) permutes the SC-site
  target assignment → the anti-cheat is built in.
- **The build-test probe** — `_nav_sc_popvector_readout_derisk.py` (built; arms host / sc_ramp / sc_popvector /
  sc_popvector_scr; reads per-phase action distribution + re-orient finalQ). Wired for `sc_popvector_divnorm_sigma/gain`
  (`burndown-6-popvector-readout-build.md:37`).
- **`run_moving_goal_episode` by import** — the whole episode harness, kwargs `sc_popvector_readout=`,
  `sc_popvector_divnorm_sigma=`, `sc_popvector_divnorm_gain=`, `--sc-popvector-readout` CLI + `SC_POPVECTOR` env.

⇒ **the build is ~90% done** — the missing 10% is the divnorm calibration + the grid-32 confirm + (conditionally) the
Option-E reset. No `sim/` edit anticipated (the read-out formula + the divnorm flag are runner-side / existing
primitive; Option E is a runner-side current injection).

---

## 4. RECOMMENDED CHEAP-FIRST DE-RISK (the smallest test that resolves the fork)

**The fork:** does the population-vector read-out, with its divisive normalizer CALIBRATED to the nav SC drive scale,
make `cortex_X` TRACK the bump position so the agent RE-ORIENTS after a goal change — or does a residual bump/ring
hysteresis still stick it (→ Option E)?

**The cheap-first shot (two stages):**

1. **Calibration micro-sweep (grid-8/480, seconds):** run `_nav_sc_popvector_readout_derisk.py` on the
   `sc_popvector` arm over a small `(divnorm_sigma, divnorm_gain)` grid (e.g. gain ∈ {0.05, 0.2, 1.0} to find the band
   where the normalized SC drive is NOT crushed — the build doc's diagnosis says `gain=1` over-attenuates, so the
   responsive band is gain ≪ 1). PASS-to-proceed = at least one (σ,gain) makes the grid-8 phase-1 action distribution
   shift toward the new goal's cardinal (vs the stuck-N at default). This is the missing step the build abandoned.
2. **The decisive faithful confirm (grid-32 / 1800 / warmup-600 — NOT a smoke):** at the best (σ,gain), run the full
   `--spiking-sc` config (the EXACT NEGATIVE config) and read the **per-goal-phase re-orient finalQ + the per-phase
   (N,E,S,W) action distribution** across all 4 phases (3 re-orients). **PASS = the pop-vector arm's post-change finalQ
   approaches the host control (the NEGATIVE's gap was ~73× on post-change) AND the action distribution TRACKS the goal
   (W-heavy for the far-west goal, E-heavy for the SE goal) instead of the stuck-N (N ~0.45–0.52 every phase).**

**Regime-fidelity is mandatory — grid-32, not grid-8.** The grid-8/480 smoke is the documented cautionary tale: it is
a weak read (only 2 goal phases complete; the cascade N-bias + OU dominate at small scale), and a grid-8 false-GO here
is precisely how the build mis-read its own state. Grid-8 is the calibration screen ONLY; the verdict is grid-32.

**If the calibrated pop-vector arm STILL fails to re-orient at grid-32** (action distribution still goal-invariant
despite a correct-and-normalized geometry), THAT is the trigger for Option E (the bump/ring hysteresis is the
residual) — and only then is a goal-change reset attempted, with its own on-substrate `sc_rostral` trigger (not a host
goal-moved flag).

---

## 5. THE ANTI-CHEATS (mandatory — all carried from the NEGATIVE + the build's own lesion)

1. **Host positive control** — `sc_orienting_cardinal_from_image` (centroid+argmax), SAME grid/schedule, anchors the
   pop-vector arm's residual gap (the NEGATIVE's host re-orients to post-change finalQ ~0.5, gate 2.19).
2. **The re-orient-after-goal-change metric (NOT static acquisition)** — the per-phase finalQ on phases 1..3
   (post-change). A read-out fix must move the *re-orient* metric; the static-hold metric is already movable by
   `SC_CORTEX_W` WITHOUT fixing re-orient (the NEGATIVE's whole point).
3. **The per-goal-phase action distribution** (the datum that diagnosed the NEGATIVE) — the (N,E,S,W) fraction per
   phase MUST track the goal's location (shift W-heavy↔E-heavy across phases), not stay goal-invariant. This is the
   direct read of "does the read-out track the bump's retinal position" and the clincher the NEGATIVE used.
4. **The static-vs-moving-goal contrast** — confirm the fix does NOT regress static acquisition (phase-0 finalQ must
   stay ≈ host) while it fixes re-orient. (Guards against an Option-E reset that fixes re-orient by destroying hold.)
5. **The retinotopy-scramble LESION** — `install_spiking_sc_wiring(scramble=True)` (built): a scrambled-retinotopy
   pop-vector read-out MUST regress to chance (proves the orienting is carried by the *retinotopic* decode, not a
   non-retinotopic leak / a cascade prior). This is the anti-cheat that the build doc left as an empty FILL.
6. **Drive non-confound (matched `SC_CORTEX_W`)** — run the pop-vector arm at the SAME `SC_CORTEX_W` as the host-pA
   equivalent so the improvement is attributable to the read-out GEOMETRY, not a covert drive increase (the NEGATIVE
   proved drive alone does not fix it; the build must beat that at matched drive).
7. **Perception NOT stripped** — `enable_visual_cortex` on, warmup honored (the actor keeps its vision drive), as the
   NEGATIVE did.
8. **Regime fidelity = faithful grid-32/1800/warmup-600 for the VERDICT** (grid-8 only for the calibration screen) —
   the explicit guard against the grid-8 false-GO that mis-led the abandoned build.

---

## 6. POINT-NEURON-ACHIEVABLE? — verdict

**The CORE read-out fix (Options A + B) is POINT-NEURON-ACHIEVABLE with HIGH confidence — and it is the on-file
hypothesis, correctly identified but NOT yet actually tested at faithful scale.** The mechanism is the SC's canonical
population-vector decode (a feedforward cosine-weighted sum on LIF point neurons) + Carandini–Heeger divisive
normalization (an existing `sim/` primitive) + the project's own already-deployed #4 WTA ring — all three pure
point-neuron, all three with published precedents AND in-codebase implementations. The geometry (egocentric retina →
retinotopic bump) is verified correct; the bump is a point-neuron attractor; nothing dendritic is implicated in the
decode. The "closed honest-negative" verdict is **premature** because the prescribed fix was abandoned at a
mis-calibrated grid-8 smoke (`gain=1`, self-documented as over-attenuating) and the decisive grid-32 confirm was never
run (it is a `<!-- FILL -->` placeholder); the "closed" ledger row instead cites a *different* knob's
(drive-strength's) grid-32 sweep.

**The ONE residual that could be a genuine sub-boundary is the re-orient-specific hysteresis (Option E):** the SC
deep-layer bump attractor + a hard WTA ring resist moving an established winner *by design* (this is correct biology —
fixation/omnipause exists precisely to override it). The point-neuron remedy (a goal-change-triggered
inhibition-of-return / fixation reset, triggered on-substrate via the existing `sc_rostral` foveation drop) is named
and point-neuron-feasible, but its on-substrate calibration (detecting "the goal moved" from neurons alone, without a
host flag, and resetting strongly enough to release the old bump yet not so strongly it destroys static hold) is the
part most likely to itself be a point-neuron operating-point wall. **That is the honest residual risk — but it is only
reached IF the calibrated A+B still sticks, and even then it is point-neuron-flavored (a transient inhibitory current),
NOT dendritic.**

**⇒ #6 should be RE-CLASSIFIED from "closed honest-negative" to "BUILD-IN-PROGRESS — finish the prescribed
point-neuron mechanism (calibrate divnorm → grid-32 confirm; Option E reset if A+B still sticks) before any honest
verdict".** The earned-negative bar requires the real mechanism to be attempted at faithful scale; that has not yet
happened.

---

## Sources

**Project code (load-bearing math/geometry verified line-by-line):**
- `research/runners/g11_bg_runner.py` — the deployed half-plane ramp read-out (`:299-300`); the pop-vector build
  (`:287-296`, cosine projection); `ddx,ddy` foveal-offset (`:286`, `sc_center` `:240`); `ret0 = sc_retina` (`:236`);
  the egocentric SC eye drive (`:6813-6822`) vs the allocentric main retina (`:6797-6806`); `render_egocentric_goal`
  (`:183-198`, blob = goal bearing); the host control `sc_orienting_cardinal_from_image` (`:144-155`); the graded
  sibling `sc_salience_offset_from_image` (`:158-180`); `sc_rostral` foveation read-out (`:311-335`); the scramble
  lesion (`:244-249`); the #4 `sel_X`/`commit_X` WTA ring (`:446-479`); `SC_CORTEX_W` env (`:4433`); the
  `popvector=True` + divnorm docstring (`:212-232`).
- `sim/bridge.py` — `input_divisive_norm` Carandini–Heeger gain control (`:6048`, guarded no-op).

**Project findings:**
- `research/findings/2026-06-20-nav-sc-drive-reorient-derisk.md` (the #6 NEGATIVE: drive sweep, stuck-N action
  distribution, operating-point-floor classification) — the boundary under repair.
- `research/findings/2026-06-20-nav-readout-geometry-deep-research.md` (the prior deep-research: Option A pop-vector +
  divnorm + #4 WTA ring; drive is the wrong lever) — the prescribed fix.
- `research/findings/2026-06-20-burndown-6-popvector-readout-build.md` (the BUILD, ABANDONED at grid-8: the grid-32
  FILL placeholders `:73-76/:90/:99`; the `gain=1` over-attenuation diagnosis `:64-67`) — the unfinished artifact.
- `research/findings/2026-06-20-shortcut-burndown-status.md:33` (the "closed honest-negative" ledger row, premature).
- `research/findings/2026-06-19-spiking-decision-default-on-GO.md` (the #4 WTA ring, default-on, 1.16× host — the
  Option-B machinery).

**Catalog (`sim-catalog/references/feature-catalog.md`, entries cited by the prior deep-research, verified there):**
- **H.25** Superior colliculus saccade map — topographic motor map (Kandel 6e Ch 35 p 875-882).
- **E.03** Population coding & vector averaging (Kandel 6e Ch 17 p ~458-464).
- **H.17** Georgopoulos population vector (Kandel 6e Ch 34 p 825-840) — the entry itself flags the project's pools as
  "categorical … could be tested by adding cosine-tuned input layer; would naturally yield population vector readout".
- **A.04** competitive WTA at GPi/SNr; **A.07** SNr→SC tonic inhibition (the disinhibition gate = the reset substrate);
  **E.05** lateral inhibition / center-surround (the divisive-normalization motif).

**Literature (WebSearch — the re-orient/reset + point-neuron-feasibility anchors):**
- Goossens & Van Opstal — spiking SC models; the cell "spike vector" summed/weighted-averaged over the bump = the SC
  population decode. [PMC5506246](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5506246/);
  [Sci Rep 2022](https://www.nature.com/articles/s41598-022-10991-6).
- A spiking model of SC for bottom-up saliency — superficial-layer temporal WTA ("chooses a winner") + deep-layer bump
  attractor (short-range AMPA/NMDA excitation + long-range inhibition). [PMC3704631](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3704631/);
  [BMC Neurosci P185](https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-14-S1-P185).
- Dynamic control of eye-head gaze shifts by a spiking-SNN SC model (the bump moves / re-targets dynamically).
  [PMC9714624](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9714624/).
- The SC's control of fixation via projections to brainstem omnipause neurons — rostral fixation neurons ("Don't
  orient!") gate the saccade generators; their release permits re-orienting (the Option-E reset biology).
  [PubMed 11702566](https://pubmed.ncbi.nlm.nih.gov/11702566/).
- Frémaux, Sprekeler & Gerstner (2013), PLOS Comput Biol — continuous-time spiking actor-critic; the actor = a ring of
  direction-coding neurons (local excitation + global inhibition → single bump), action = population vector.
  [PLOS CB 1003024](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1003024).

_Read-only deep-research deliverable. No code edits, no GPU runs. Load-bearing read-out math + render geometry +
drive-injection path verified line-by-line against `g11_bg_runner.py`; every "point-neuron-feasible" claim anchored to
a published point-neuron model + a catalog entry._
