# N5 — host-computed perceived-approach reward → a NEURAL (spiking) approach-reward signal

**Date:** 2026-06-09
**Type:** Deep-research / scoping review (READ-ONLY; no `sim/` edits, no GPU run, short CPU diagnostics only).
Written to the standing practice ("deep research + catalog review FIRST before committing build/GPU resources")
and the **2026-06-08 BRAIN-BASED-ONLY directive**: anything not done by neurons/synapses/their communication
is a shortcut, *even if the host calculation is biologically correct*.
**Predecessors read in full:**
- `2026-06-08-remaining-nav-cheats-full-biologization-research.md` (the N5/N9 reward research; N5 = perceived-approach reward).
- `2026-06-08-spiking-snc-N5-nav-derisk-NEGATIVE.md` → its own header: **RESOLVED GO** post bug-fix (brain-based reward+SNc beats the host shortcut 2.00 vs 4.24, 3/3).
- `2026-06-08-spiking-snc-stageB-Bprime-value-subtraction-circuit-research.md` (the B′ "signal gates a normal-reversal relay that drives the SNc" pattern — the load-bearing template here).
- `2026-06-09-navfaithful-afferent-critic-homeostasis-PASS.md` (the deterministic-nav-faithful de-risk that PASSED; its probe is the template for this N5 de-risk).
- `2026-06-08-nav-neural-value-critic-redesign-research.md` (the dense `vs_place_context` afferent design).

---

## EXECUTIVE SUMMARY (read this first)

**The precise residual.** N5 is *already coordinate-free* (it reads the goal's retinal eccentricity from pixels via
`sc_salience_offset_from_image`, not `(gx,gy)`). The 2026-06-08 directive re-classifies it on a *different* axis: the
reward `r` is still **computed by a Python formula** — `g11_bg_runner.py:5266–5273`:

```python
_eb = hypot(*_ob);  _ea = hypot(*_oa)          # offset magnitudes, pixels-only
reward = +1.0 if _ea < _eb - 1e-6 else (-1.0 if _ea > _eb + 1e-6 else 0.0)   # <-- HOST sign()
```

That scalar then becomes the SNc's **reward excitation** at `:5471` / `:5479`:
`I_snc = snc_tonic_pa + snc_reward_gain·max(0, reward) − …`. So today the brain computes the **prediction error**
(spiking SNc, Stage A GO) and — with the neural critic — the **value V** (Stage B GABA_B/GIRK subtraction, navfaithful
PASS), **but a host `sign(Δecc)` still computes the *primary reward* `r` and hands it to the SNc as a number.** That
last arithmetic term is the N5 shortcut. The brain is *not* detecting "I got closer"; the simulation's bookkeeping is.

**The biology question, answered.** A brain computes a primary approach/appetitive reward ("the goal is getting
closer in my sensory field") **from sensory neurons whose activity changes as the percept changes** — and emits it as
*spikes*, not a continuous scalar (catalog O.23 reward-fn-2 = "goal object for approach"; C.27 Berridge "wanting" =
DA-mediated incentive-salience/approach; the navigational homologue of **phototaxis** = reinforcement is the *change
in a perceived scalar*). The faithful minimal circuit is a small **spiking approach-detector** population that reads
the sensory-cortex code and fires when the perceived goal-eccentricity *decreases* — exactly the "reward signal
computed from the activity of the sensory layers, represented as spikes" pattern in the spiking-RL literature
(Kaiser/active-efficient-coding; the purely-spiking-RL principle that *reward must be a spike stream, not a
continuously-changing value*).

**Recommended option — `R5-APPROACH-CELL` (ranked #1):** a small spiking **`approach_reward` population** driven by a
*temporal-difference of the sensory eccentricity code* (the on-substrate analogue of `Δecc`), whose **firing is the
reward `r`** that drives the SNc — replacing the `snc_reward_gain·max(0, host_reward)` host term with
`snc_reward_gain·(approach_reward firing)`. It reuses the validated B′ "signal-drives-the-SNc-as-excitation" pattern,
the existing `sensory` Gaussian-bump region (already a spiking eccentricity code), per-region homeostasis (the committed
`89b8d909` edit), and the three-factor / SNc machinery. **Zero new protected `sim/` edit** (it is a new `BrainRegion` +
two `RegionPathway`s + a runner-side current-injection swap, exactly the surface the B′ critic used). The one genuine
design subtlety — computing a *temporal difference* of a spiking code on-substrate — has a clean biological realization
(a fast-excitatory / slow-feedforward-inhibitory "change/transient detector," the same ON-transient motif the retina
uses; "the sensory system is more responsive to changes than to static input").

**Honest-negative possibility, stated up front.** A spiking transient-detector approach-cell may not reproduce the host
`sign(Δecc)` cleanly enough on a discrete 8-grid (one-cell-per-step eccentricity changes are small and quantized → the
detector's spike-count difference may be too noisy to be reliably sign-correct). If a *faithful* spiking approach-reward
cannot track approach well enough to drive the SNc with the right sign at deterministic-nav fidelity, **that is a valid
finding** mapping a substrate limit (the same class of honest negative the BRAIN-BASED directive names as the
deliverable). The de-risk below is built to localize exactly that.

---

## 1. Diagnosis — what is host-computed today vs what must become neural

### 1.1 The exact host computation (verified `file:line`)
`research/runners/g11_bg_runner.py`, the `perceived_approach_reward` branch:
- **`:5258–5265`** renders the agent's visual input before/after the step (`render_gridworld_to_image`, legit host
  — *the environment rendering the sensory input*) and reads the **goal offset** from each image via
  `sc_salience_offset_from_image` (pixels only; coordinates structurally cannot enter — 6/6 unit tests pin this).
  **This part is legitimate** under the directive: it is "the environment + the body's sensory render," and the
  *percept* (the offset) is what a real retina/SC would deliver.
- **`:5266–5273`** — the **shortcut**: a Python `hypot()` + `sign()` turns the two percepts into `reward ∈ {−1,0,+1}`.
  This is **cognition done by host arithmetic**: "detecting that the goal got closer" is a computation the *brain*
  must perform (perception/salience/valuation are explicitly named brain-side in the directive). No neuron fires to
  produce this `r`; `numpy` does.
- **`:5471` / `:5479`** — `r` enters the brain as the SNc's reward excitation:
  `I_snc = snc_tonic_pa + snc_reward_gain·max(0, float(reward)) [− snc_value_gain·V]`. So the host scalar is *injected
  as a current*; the SNc then *fires* the prediction error δ = r − V. **The δ is neural; the `r` inside it is host.**

### 1.2 What is already neural (do NOT re-do)
- **Dopamine / RPE (N9 Stage A):** the **spiking `snc` pool fires δ** (`spiking_snc`, `:5467–5484`), read out as the
  `dopamine` concentration via the one merged protected edit (`from_region_firing_signed`,
  `sim/neuromodulators.py:774–817`). Header of `2026-06-08-spiking-snc-N5-nav-derisk-NEGATIVE.md`: **GO, 2.00 vs 4.24
  3/3** (the earlier NEGATIVE was a now-fixed bridge bug).
- **Value V (N9 Stage B):** the neural `striosome_value` critic learns V(s) from the perceived state and **subtracts it
  at the SNc membrane** via GABA_B/GIRK (`receptor="gaba_b"`, `E_K≈−90 mV`, now a committed conductance —
  `sim/bridge.py:1192–1199, 2271–2290`; `sim/regions.py:259–269`). The deterministic-nav-faithful de-risk
  **PASSED 3/3** (`2026-06-09-navfaithful-afferent-critic-homeostasis-PASS.md`) with the dense `vs_place_context`
  afferent + per-region homeostasis on both afferent and critic.

### 1.3 What must become neural (the N5 target)
Replace the **`snc_reward_gain·max(0, host_reward)` excitation term** with the **firing of a spiking population that
detects perceived approach**. After that, the SNc's reward drive, its value subtraction, and its δ readout are *all*
carried by neuron firing — the reward→value→dopamine loop is fully brain-based and coordinate-free. The plug-in point
is surgical: it is one additive current term at `:5471` (a one-region swap, mirroring how Stage B deleted the host
`−k_v·V` term and let the GABA_B critic carry it).

### 1.4 Anti-pattern to avoid (a literature correction, load-bearing)
The naive "lateral-hypothalamus glutamatergic *primary-reward* → excite the SNc" wiring is **wrong-signed**: the
**LH→VTA *glutamatergic*** arm drives **aversion / active avoidance**, while the **LH→VTA *GABAergic*** arm drives
**approach** (by inhibiting local VTA GABA interneurons → *disinhibiting* DA — Nieh/Stuber 2016 *Neuron*;
Barbano/Morales). So an approach-reward population must reach DA the way biology does — **as excitation onto the SNc's
burst-driver, or via disinhibition** (the B′-EXC / B′-SNr templates) — **not** as a direct LH-glut→DA excitation. The
recommended option below routes the approach signal through the *excitatory burst-driver* arm, which is sign-correct
and is the strongest route on this substrate.

---

## 2. Ranked biology-grounded options for a spiking approach-reward circuit

Goal restated precisely: **a spiking population whose firing rises when the perceived goal-eccentricity decreases
(approach) and whose firing is the reward `r` driving the SNc** — sign-correct, on-substrate, reusing the existing
perception (the `sensory` eccentricity code / the offset helper) and the SNc drive.

Each option is scored on **fidelity × P(works on this substrate) × surface (runner-side vs protected `sim/` edit)**.

### Option R5-APPROACH-CELL — a spiking transient/change-detector reads the sensory eccentricity code and *fires* the reward *(RECOMMENDED, ranked #1)*

**The circuit.**
1. **Perceived-eccentricity code (reuse).** The `sensory` region is *already* a spiking code of the perceived offset:
   each step the runner drives it with a Gaussian bump centred on the image-sourced `(dx,dy)`
   (`g11_bg_runner.py:5011–5021`, `learned_perception_from_vision`). Its **population firing is a monotone proxy for
   eccentricity** (near goal → bump near the "centre" preferred cells; far → bump on peripheral cells). For a reward we
   need a **scalar "closeness" code**, so add (or repurpose) a small **`ecc_now`** population whose total firing rate
   *increases as eccentricity decreases* — e.g. a "proximity cell" tuned so a *small* offset drives it hard (a single
   centre-preferring Gaussian cell group, or read the `sensory` bump through a fixed "centre-weighted" pooling
   pathway). This is the on-substrate "goal-proximity in the visual field" scalar — the phototaxis sensed-intensity
   analogue. *No coordinate enters; it reads the same image-sourced bump the learned dorsal circuit already uses.*
2. **Temporal-difference / approach detector (the one real design piece).** Approach = `ecc_now` *rising* across the
   step. Compute the on-substrate temporal difference with a **change-detector motif**: a small **`approach_reward`**
   excitatory population receives **fast excitation from `ecc_now`** and **slow (delayed) feed-forward inhibition from a
   one-step-lagged copy of `ecc_now`** (an interposed `ecc_prev` relay paced to hold the *previous* step's proximity).
   When proximity *rises* (approach), the fast excitation outruns the lagged inhibition → `approach_reward` **bursts**;
   when proximity *falls* (retreat), the lagged inhibition dominates → it is **silenced** (a rectified, sign-correct
   `[Δproximity]₊` detector). This is the canonical **ON-transient / change-detector** circuit ("the sensory system is
   more responsive to changes than to static input"; retinal transient cells; the active-efficient-coding reward read
   "from the activity of the sensory layers"). The lag is realized exactly like the existing **value-leads-reward
   eligibility window** the critic already uses (a one-step relay held across the nav integration), so the machinery is
   precedented.
3. **Drive the SNc with the *firing* (reuse B′-EXC).** Replace `snc_reward_gain·max(0, host_reward)` at `:5471` with
   the **`approach_reward → snc` excitatory pathway**: the population's windowed firing *is* `r`. SNc integrates
   `(approach_reward excitation) − (striosome_value GABA_B/GIRK) + tonic` and fires δ = r − V. **The reward, the value,
   and the error are now all neuron firing.**

**Sign analysis.** approach (ecc↓ ⇒ proximity↑) ⇒ fast exc > lagged inh ⇒ `approach_reward` bursts ⇒ SNc reward
excitation ⇒ δ>0 burst → LTP on the active cortico-striatal synapses (the actor that just produced the approach
move). Retreat ⇒ detector silent ⇒ no reward excitation ⇒ (with the value subtraction) δ<0 dip. **Sign-correct.**
(Negative reward can be sharpened later by a *mirror* "recede-detector" → a lateral-habenula-style negative-RPE arm;
not required for the first build — the `max(0, r)` rectification today already drops negative reward to "no excitation,"
and the value subtraction supplies the dip.)

**Fidelity.** High. It is the textbook **incentive-salience approach** signal (Berridge "wanting", C.27) realized as a
**sensory-change → appetitive-drive** transient — the navigational **phototaxis** homologue (reinforcement = change in a
perceived scalar), reaching DA through the **excitatory burst-driver** route (the sign-correct one; §1.4). The change-
detector is a real, ubiquitous sensory motif. *Honest abstraction:* the `ecc_now` "proximity cell" is an *abstraction*
of "goal-proximity in the visual field," and the single `approach_reward` pool stands for the appetitive
sensory→reward relay (LH-GABAergic-approach / PBN primary-reward → DA-disinhibition); defensible, not a literal nucleus.

**P(works).** Moderate-high for the *mechanism* (the SNc-drive and three-factor legs are already validated; this only
changes *what current the reward term carries*). The risk concentrates in step 2 on the **discrete 8-grid** (small,
quantized per-step `Δecc`), addressed by the de-risk in §4 and the honest-negative clause.

**Surface.** **Zero new protected `sim/` edit.** New regions (`ecc_now`, `ecc_prev`, `approach_reward`) = `BrainRegion`s;
the centre-weighted pooling, the fast-exc, the lagged-inh, and `approach_reward→snc` = `RegionPathway`s; the lag = the
existing one-step relay/eligibility-window pattern; per-region homeostasis on `approach_reward` (so it fires into a
useful range) = the **already-committed** `89b8d909` per-region mask, *global homeostasis stays OFF*. The runner-side
edit is the `:5471` current swap + a `--neural-approach-reward` flag — exactly the B′-critic surface.

### Option R5-DISINHIBIT — approach-cell drives DA via VTA-GABA disinhibition (the literal appetitive route) *(ranked #2; biology-faithful comparator)*

Same `approach_reward` detector, but it reaches DA by **inhibiting a tonic local SNc-GABA pool** (`snc_gaba_tonic`,
normal reversal), whose pause *disinhibits* the SNc — the **LH-GABAergic→VTA** appetitive route (Nieh 2016) and the
SNr→SNc disinhibition (B.15). Most literally the biology of "approach reward reaches DA by disinhibition," but (per the
B′ doc §2) the **final GABA hop lands on the depolarized SNc** (`E_GABA=−55 mV, no KCC2`) so the *delivered* effect is
GABA_A-weak — **unless** it uses the **now-available GABA_B/GIRK** conductance (`receptor="gaba_b"`), which makes the
disinhibitory pause genuinely strong. With GABA_B/GIRK this becomes a *strong, literal* appetitive-disinhibition route
and is a serious alternative. Zero protected edits (GABA_B already committed). Build as the fidelity comparator; expect
R5-APPROACH-CELL (direct excitation) to be the simplest-strongest first build.

### Option R5-BEACON-GRADIENT-NEURAL — neuralize the existing `--sensed-reward` beacon gradient *(ranked #3)*

Today `--sensed-reward` computes a beacon intensity `peak/(1+falloff·d)` and takes `sign(Δintensity)` — but `d` is the
**coordinate** distance (`:5276–5277`), so it is *not* coordinate-free (the prior research flagged this). A neural
version would (a) make the beacon a **rendered luminance the retina senses** (a real intensity gradient in the image,
not a coordinate formula) and (b) run that luminance through the same `approach_reward` change-detector. This is the
*purest* phototaxis (literal sensed-light gradient) but requires a **render change** (variable goal luminance with
distance — the world currently paints a *fixed-size, fixed-amplitude* blob, so apparent intensity does NOT encode
distance; the prior doc's N5 pitfall). Larger surface (render + retina) for the same end signal as R5-APPROACH-CELL,
which already gets approach from **eccentricity** (which the fixed-blob world *does* encode). Keep as the enrichment if
the owner later wants a literal luminance-gradient world.

### Option R5-LOOMING — a spiking looming/expansion detector reads the goal blob's growth *(ranked #4; NOT applicable here)*

Approach-detection famously *begins in the retina* (looming/expansion-selective RGCs → dorsal pathway; Kim 2020; the
locust LGMD collision-detector). But the gridworld render paints a **fixed-size** goal blob (`render_gridworld_to_image`
spreads the goal over a constant 3×3 at amplitude 0.5, distance-independent) — so **apparent size does not encode
distance**; there is no looming cue to detect. A looming detector is the *right* biology for an egocentric first-person
world with size-as-depth, and would be the matched mechanism *if* N2 were enriched to an egocentric render. **For the
current world it is inapplicable** (it would have nothing to fire on). Documented so the option space is complete; do
not build against this world.

### Option R5-HOST-SCALAR-AS-CURRENT — keep the host `sign(Δecc)`, just inject it as SNc current *(ranked last; explicitly NOT brain-based)*

This is the *status quo* (`:5471` already injects the host scalar as current). It is **not** a biologization — the
reward is still a host formula; only its *delivery* is a current. Listed to name it as the thing N5 exists to remove.
Useful **only** as the A/B baseline the neural reward must match.

**Ranking:** **R5-APPROACH-CELL ≻ R5-DISINHIBIT ≻ R5-BEACON-GRADIENT-NEURAL ≻ R5-LOOMING(n/a) ≻ R5-HOST-SCALAR.**

---

## 3. Reusable machinery vs genuinely new

| N5 needs… | Already exists (reuse) | file:line / source |
|---|---|---|
| A spiking **perceived-eccentricity** code | `sensory` region driven by a Gaussian bump on the image-sourced offset (the learned dorsal "where" read-out) | `g11_bg_runner.py:5011–5021`; helper `sc_salience_offset_from_image` `:106–128` |
| A spiking population **firing as the reward** into the SNc | The B′-EXC pattern: a signal-driven population drives the SNc as **excitation** (`snc_reward_gain·…` is *already* an excitatory drive term) | B′ doc §3 (B′-DISINHIBIT-EXC); current SNc swap `g11_bg_runner.py:5467–5484` |
| A **one-step lag** (for the temporal difference) | The value-leads-reward eligibility window / one-step relay held across the nav integration | `2026-06-09-navfaithful-…-PASS.md`; `critic_snc_window` `g11_bg_runner.py:5465–5466` |
| **Firing a region into a useful rate range** under the deterministic regime | Per-region homeostasis (intrinsic homeostatic plasticity), GLOBAL homeostasis OFF | committed edit `89b8d909`; `enable_critic_homeostasis` pattern `g11_bg_runner.py:167` |
| The **SNc** pool + δ readout + DA broadcast | `spiking_snc` Stage A (validated GO) | `g11_bg_runner.py:5467–5484`; `from_region_firing_signed` `sim/neuromodulators.py:774–817` |
| The **three-factor** LTP/LTD that the reward then drives | Global eligibility trace + signed `da_signal` + `Δw=lr·signal·elig` | `sim/bridge.py:616, 5894–5904, 5995–6013` |
| **Inhibitory→excitatory** routing, per-region reversals, **GABA_B/GIRK** (for R5-DISINHIBIT) | `exc_fraction` split; per-neuron `E_GABA`; committed GABA_B/GIRK conductance | `sim/regions.py:296–300, 259–269`; `sim/bridge.py:1086–1101, 1192–1199, 2271–2290` |
| **De-risk probe** (CPU, deterministic-faithful) | `snc_stageb_critic_probe_navfaithful.py` — builds the minimal SNc+critic bridge, calibrates DA threshold, runs the gates, has lesion anti-cheats, `SIM_BACKEND=numpy` | `research/runners/snc_stageb_critic_probe_navfaithful.py` |

**Genuinely new (small):** (1) the **`approach_reward` change-detector** wiring (fast-exc from `ecc_now` + lagged-inh
from `ecc_prev`) — a new motif, but built entirely from existing `BrainRegion`/`RegionPathway` vocabulary; (2) the
**`ecc_now` "proximity" pooling** of the sensory bump into a scalar closeness code; (3) the `:5471` current-term swap +
`--neural-approach-reward` flag. **No new kernel, no new `cp_*` array, no new protected edit.**

---

## 4. Cheap-first de-risk — REPLICATES THE DETERMINISTIC-NAV REGIME (load-bearing)

**Why this section is load-bearing.** This arc has hit **five "probe ≠ deployment" gaps + a homeostasis-timescale
gap** (every time a de-risk diverged from deployment — sparse vs dense afferent, GABA_A vs GABA_B, global vs per-region
homeostasis, lead-window timing). The de-risk MUST match nav's regime exactly, and state which deployment conditions it
replicates. The 2026-06-09 navfaithful PASS is the proof this discipline works; mirror it.

**Deployment regime to replicate (verbatim nav flags):** GLOBAL `cfg.enable_homeostasis` **OFF**, OU noise **OFF**,
conductance-noise **OFF**, heterogeneity **OFF**, `dt = 1.0` ms, deterministic (`CUBLAS_WORKSPACE_CONFIG=:4096:8`); the
**real perception** path (`--enable-visual-cortex`, the image-sourced `sensory` bump, `sc_salience_offset_from_image`);
`spiking_snc` ON; the reward-hold/eligibility window exactly as `:5500`. Only **per-region** homeostasis on
`approach_reward` (and `ecc_now`/`ecc_prev` if needed) is permitted — *global stays OFF*, preserving the deterministic
regime. **State explicitly in the probe header which of these it replicates** (it should replicate all of them).

**Probe (extend `snc_stageb_critic_probe_navfaithful.py`; CPU/numpy, no GPU, no nav build).** Add the `ecc_now`/
`ecc_prev`/`approach_reward` regions and the change-detector pathways; drive `ecc_now` with a **scripted eccentricity
trajectory** that alternates *approach* steps (proximity rising) and *retreat* steps (proximity falling) at the
realistic discrete-grid step size, under the deterministic regime above.

**PASS / FAIL gates (multi-seed ≥3: 42/43/44):**
1. **(i) APPROACH SELECTIVITY — the core gate.** `approach_reward` windowed firing on *approach* steps **>** on *retreat*
   steps by a **robust margin** (gate: approach-rate `> 1.5×` retreat-rate, **sign-consistent across ≥3 seeds**, robust
   over a small detector-gain sweep). This is the whole point: the spiking detector must track approach sign-correctly.
2. **(ii) SNc TRACKS THE NEURAL REWARD.** With `approach_reward → snc` live: SNc windowed rate on approach steps **>** on
   retreat steps (the δ inherits the reward sign), and the **omission/retreat case dips** (with the value subtraction
   on). I.e. the neural `r` drives the SNc the way the host `r` did.
3. **(iii) LABEL-AGREEMENT vs the host `sign(Δecc)` (the behavioral-equivalence guard).** On a *matched* eccentricity
   trajectory, the neural detector's per-step reward sign agrees with the host `sign(Δecc)` on a **high fraction** of
   steps (target ≥ 7/8, the same bar the original N5 met against Manhattan). Divergence localizes the discrete-grid
   quantization risk.
4. **(iv) REGIME FIDELITY.** Assert GLOBAL OU/cond-noise/homeostasis/heterogeneity all OFF, `dt=1.0` (mirror the
   navfaithful probe's gate (d)).

**Falsifier (the honest-negative trigger).** If gate (i) or (iii) **fails** multi-seed under the deterministic regime —
the spiking change-detector cannot track approach sign-correctly on the quantized 8-grid (small `Δecc`, transient-
detector spike-count noise) — that is the **valid negative**: it maps a substrate limit (a faithful spiking approach-
reward underperforming the host `sign()` on this discretization). Document it; do not paper over it by leaking the host
`r`.

**Anti-cheat controls (decisive — prove the reward is neuron firing, not a formula):**
- **(a) NO host `r` reaches the SNc.** Under `--neural-approach-reward`, assert the `snc_reward_gain·max(0, host_reward)`
  term is **removed** and the SNc reward excitation is the `approach_reward→snc` *synaptic* current only — no host
  `reward`/`Δecc` scalar enters `I_snc` (mirror Stage B's deletion of the host `−k_v·V` term, `:5468–5474`).
- **(b) The reward is a spiking-population READOUT, not a formula.** The reward magnitude logged each step is measured
  from `cp_firing_states[approach_reward]` (the same way `snc_rate_log`/`striov_rate_log` are measured, `:5506–5517`),
  **never** recomputed from the offset.
- **(c) LESION the reward population → the SNc reward response VANISHES.** Zero the `approach_reward→snc` edges (extend
  the probe's existing `_lesion_*`): the SNc must then **fail to burst on approach** (it should fire only tonic/dip,
  with no reward excitation). If a reward response survived the lesion, it would prove host arithmetic in disguise — it
  must not.
- **(d) Coordinate-freedom.** `ecc_now`'s only afferent is the image-sourced `sensory`/offset code; assert no coordinate
  enters the reward path (combined with the already-coord-free critic and δ, the *whole* reward→value→DA loop is
  coordinate-free).

**Nav-score gate (necessary, not sufficient — only AFTER the probe passes).** Flagship multi-goal-deterministic
**6-seed** (A+E+G v2.5 + `--enable-visual-cortex --sc-orienting-reflex --spiking-snc --neural-approach-reward`, and the
neural critic if stacking N9 Stage B): summed `sum_finalQ` **≥** the Stage-A host-reward baseline (the brain-based
reward+SNc already beat the host shortcut 2.00 vs 4.24, so the bar is "neural `r` ≥ host `r` inside that GO stack"). An
**honest nav regression is still a deliverable** (it would mean the spiking approach-cell is a worse `r` than the
host `sign()` in the full loop — a mapped substrate limit). Grids 8 and 32.

---

## 5. Anti-cheat controls + the honest-negative possibility (summary)

**Anti-cheats (collected):** (a) no host `r` in `I_snc`; (b) reward = `cp_firing_states[approach_reward]` readout, not a
formula; (c) **reward-population lesion → SNc reward response vanishes** (the decisive proof); (d) coordinate-freedom of
the whole loop; (e) regime-fidelity assertion (global OU/noise/homeostasis OFF, dt=1.0) so the de-risk = deployment.

**Honest-negative possibility (a valid deliverable).** Two faithful negatives are live and would each map a real
substrate limit:
1. **Quantization-limited detection:** on the discrete 8-grid, per-step `Δecc` is small and quantized; a spike-count
   change-detector may be too noisy to be reliably sign-correct (gate (i)/(iii) fail). *Finding:* a faithful spiking
   approach-reward needs a finer-grained world (grid-32, or sub-pixel/continuous motion) than the 8-grid affords — a
   mapped limit of rate-coded change-detection on coarse discretization.
2. **Loop-level regression:** the detector tracks approach in isolation (gates pass) but, deployed, the spiking `r`
   drives the SNc *worse* than the host `sign()` (nav regresses vs Stage A). *Finding:* the substrate's spiking reward
   is a noisier teacher than the clean host scalar in the full cascade — the exact "works-in-isolation / fails-in-the-
   whole-system" class the BRAIN-BASED directive says is the scientific deliverable.

Either negative is reported honestly and maps what the substrate can/can't do on its own — *not* worked around by
re-introducing the host `r`.

---

## 6. Sources

### Project code (verified `file:line` this session)
- N5 host reward formula: `g11_bg_runner.py:5258–5273` (render + `sc_salience_offset_from_image` + the `sign(Δecc)`);
  the SNc reward-excitation injection `:5467–5484` (`I_snc = tonic + snc_reward_gain·max(0,reward) [− k_v·V]`); the host
  `reward` branches `:5249–5294`; the reward-population firing-readout pattern to mirror `:5506–5517`.
- The perceived-eccentricity `sensory` bump (reuse): `:5011–5021`; offset helper `:106–128`; SC cardinal reflex `:72–103`.
- Spiking-SNc Stage A + neural critic Stage B wiring: builder `:131–167` (`enable_neural_critic`, `vs_place_context`,
  per-region homeostasis), runner `:5449–5484`; CLI/defaults `:2645–2736`, `:5983–5984`, `:6472`.
- The merged protected edit (the only one in the SNc arc): `from_region_firing_signed`, `sim/neuromodulators.py:774–817`.
- GABA_B/GIRK conductance (committed; enables R5-DISINHIBIT): `sim/bridge.py:1192–1199, 2271–2290`;
  `sim/regions.py:259–269` (`receptor="gaba_b"`, `E_K≈−90 mV`).
- Per-region homeostasis (committed `89b8d909`): used as `enable_critic_homeostasis` `:167`.
- Three-factor pipeline: `sim/bridge.py:616, 5894–5904, 5995–6013`; `exc_fraction` inh/exc split `sim/regions.py:296–300`.
- De-risk template: `research/runners/snc_stageb_critic_probe_navfaithful.py` (+ `snc_stageb_critic_probe.py`,
  `_place.py`).

### Project feature catalog (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`)
- **O.23** three reward functions — fn 1 (positive reinforcer, DA×eligibility) + **fn 2 (goal object for APPROACH)**;
  "project implements fn 1 fully, fn 2 partially" (`:553–566`).
- **C.27** Berridge wanting-vs-liking — **DA = incentive-salience / wanting = APPROACH** (not hedonic liking); the
  Schultz16 reconciliation (Component-1 detection/salience vs Component-2 value-RPE) (`:975–988`).
- **O.10** Incentive-motivation — deprivation *amplifies* the reward value of goal stimuli (Berridge/Toates) (`:4863–4873`).
- **B.15** SNc DA lacks KCC2 → depolarized `E_GABA≈−55 mV`; **disynaptic disinhibition (SNr→SNc) is the dominant
  phasic-DA route**; "DA cells resistant to direct GABA" (the R5-DISINHIBIT sign/strength logic).
- **B.14/B.16** MSN depolarized GABA reversal; SNr→SNc nigronigral collaterals; tonic rates.
- **C.22/C.28/C.30** Schultz RPE / TD error / actor-critic mapping (SNc=δ, striosome=V, matrix=actor) — the loop the
  N5 `r` feeds.
- **D.K / motion** PPC/MT optic-flow & MST self-motion (the looming/approach-in-retina pathway, R5-LOOMING context)
  (`:1492`).

### Peer-reviewed / current literature (verified via search this session)
- **Schultz W. (2016)** "Dopamine reward prediction error coding", *NRN* + *J. Neural Transm.* — reward functions 1/2;
  incentive-salience reconciliation. (Catalog O.23/C.27.)
- **Berridge K.C. & Robinson T.E.** — incentive-salience / "wanting = DA = approach" (catalog C.27).
- **Nieh E.H. … Stuber G.D. (2016)** "Inhibitory Input from the Lateral Hypothalamus to the VTA Disinhibits Dopamine
  Neurons and Promotes Behavioral Activation", *Neuron* 90:1286 — **the LH→VTA *GABAergic* arm drives APPROACH via
  DA-disinhibition** (the sign-correct appetitive route; R5-DISINHIBIT).
  https://www.cell.com/neuron/fulltext/S0896-6273(16)30122-2
- **Barbano M.F. / Morales M. et al.** — **LH→VTA *glutamatergic* arm = AVERSION / active avoidance** (the wrong-signed
  route to avoid; §1.4). https://pmc.ncbi.nlm.nih.gov/articles/PMC10776608/ ;
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8812999/
- **Kim T. et al. (2020)** "Selectivity to approaching motion in retinal inputs to the dorsal visual pathway",
  *eLife/PMC7080407* — **approach/looming detection begins in the retina** (R5-LOOMING biology; inapplicable to the
  fixed-blob world). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7080407/
- **(active efficient coding, 2025)** "A spiking neural network for active efficient coding", *PMC11775837* — a spiking
  RL learner whose **reward is computed from the activity of the sensory layers** (grounds "reward read from the sensory
  code"). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11775837/
- **"A purely spiking approach to reinforcement learning"**, *BioSystems 2024* — principle that **reward/punishment must
  be SPIKE streams, not continuously-changing values** (the core N5 requirement: `r` as spikes, not a scalar).
  https://www.sciencedirect.com/science/article/abs/pii/S1389041724001116
- **Friedrich J. & Lengyel M. (2016)** "Goal-Directed Decision Making with Spiking Neurons", *J. Neurosci.* 36:1529 /
  *PMC4737768* — near-optimal value estimation for goal-directed decisions in spiking neurons.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4737768/
- **Frémaux N., Sprekeler H., Gerstner W. (2013)** "Continuous-Time Actor-Critic with Spiking Neurons", *PLoS Comput.
  Biol.* 9:e1003024 — the spiking actor-critic the SNc/critic legs already follow.
- **Potjans, Diesmann & Morrison (2011)** "An imperfect dopaminergic error signal can drive TD learning",
  *PMC3093351* — de-risks small-pool rate-coded teaching-signal noise (relevant to gate (i)).
- **Eshel et al. (2015) *Nature* 527:398; Cohen et al. (2012) *Nature* 482:85** — local VTA-GABA carries/subtracts the
  expected value (the Stage-B critic the reward `r` now feeds).
- **Kandel et al.** *Principles of Neural Science* 6e — Ch 43 (dopamine/reward, Fig 43-2); Ch 35 (SC saccade/orienting);
  Ch 41 (incentive motivation).

---

**Deliverable path:** `E:\Documents\Projects\sim\research\findings\2026-06-09-N5-neural-reward-signal-research.md`
