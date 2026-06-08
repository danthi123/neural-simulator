# Nav neural value-critic redesign — deep-research findings (roadblock scope)

**Date:** 2026-06-08
**Type:** Read-only deep-research + catalog review (standing practice: research BEFORE committing build/GPU). Author: research subagent.
**Predecessors:**
- `2026-06-08-neural-critic-nav-smoke-NEGATIVE.md` (the roadblock: two blockers root-caused).
- `2026-06-08-gabab-girk-stageB-derisk-GO.md` (the GABA_B/GIRK protected `sim/` edit — validated Pavlovian, NOT in question).
- `snc_stageb_critic_probe.py` (the CPU de-risk harness the recommendation mirrors).
**New read-only diagnostics this session** (NO `sim/` edits, no commits, serial GPU):
`research/findings/raw/_gabab_flag_isolation_probe.py` (+`_result.json`),
`_gabab_navloop_isolation_probe.py` (+`_result.json`),
`_gabab_AvsB_reproduce_probe.py` (+`_result.json`).

---

## 0. TL;DR

- **Blocker 1 is now PINNED, and it is NOT a GABA_B-dynamics problem — it is a stale-mask shape bug.** Enabling
  the critic build raises a **silent broadcast error every nav step** (`operands could not be broadcast together
  with shapes (236160,) (169139,)`), which the bridge's per-step `try/except` swallows → the whole network emits
  nothing → SNc 0 Hz → no learning → nav collapses. The cause: `apply_v1_gabor_weights` (the Cluster-K visual
  pre-init) **grows `cp_connections` from 169139 → 236160 synapses AFTER `inject_explicit_wiring`**, but the
  `cp_gabab_synapse_mask` was built (sized to nnz) DURING `inject_explicit_wiring` and is never regrown. The
  per-step GABA_B block then multiplies the now-236160-long `effective_connections_matrix.data` by a stale
  169139-long mask → throw. The GABA_B *conductance* is provably inert (g=0 throughout); the flag only "destabilizes"
  the net because its block is the one place in the step that does element-wise `.data × mask` arithmetic (every
  other propagation block uses a matvec, which is length-agnostic). **This is a one-spot `sim/` fix (regrow/realign
  the mask to live nnz), not a recalibration of the value subtraction.**
- **Blocker 2 (the afferent) is a genuine biology error and the more important one.** The critic was wired to
  `cortex_it` — the position-**invariant** ventral "what" stream (IT) — which (a) never fires in nav and (b) cannot
  encode a value-of-**location** even if it did. The biology is unambiguous: the **striosome/patch critic V(s) reads
  LIMBIC input — vmPFC/amygdala/ventral hippocampus — NOT IT** (catalog C.30, B.07; Houk-Adams-Barto 1995). For a
  **spatial** value the faithful + ACTIVE afferent already in the runner is the **hippocampal place code
  `sensor_place_readout`** (the dorsal "where" / place-cell pathway), available with `--enable-place-goal-readout`.
  The hippocampus→ventral-striatum place-reward pathway is the canonical substrate for spatial value (Lansink 2009;
  van der Meer & Redish 2009).
- **Recommended next step:** a cheap-first **CPU place-cell critic de-risk** that mirrors the validated
  Pavlovian probe but swaps the cue for a **2-state place code** (state A near goal = high value, state B far = low),
  testing the one new claim ("a striosome critic driven by a place code learns a *state-of-location* value and
  subtracts it at the SNc via GABA_B") BEFORE any nav build. With the controls below. The brain-based-only nav
  value subtraction is **achievable** — the roadblock was a wiring/shape bug + a wrong afferent, not a substrate limit.

---

## 1. Diagnosis (grounded in code + biology)

### 1.1 Blocker 1 — `enable_gabab` "destabilizes full nav with g_gabab=0" → ROOT-CAUSED to a stale-mask broadcast throw

The NEGATIVE doc left this as "a subtle system-level interaction… mechanism not yet pinned." I pinned it.

**Evidence chain (all this session, read-only):**

| Probe | Build | Flag | Result |
|---|---|---|---|
| Free-run P0 | flagship regions via `build_bg_brain_regions`, critic OFF | gabab OFF | net active (snc 0.047, 135/150 steps fire), nnz 165610 |
| Free-run P1 | same, critic OFF | gabab ON, **mask None** | **identical to P0** (g=0) — flag inert with no GABA_B synapse |
| Free-run P2 | same, **critic ON** (407 GABA_B syn) | gabab ON | **also active** (snc 0.047, 136/150), nnz 169139 |
| Nav-loop ctrl | real `run_moving_goal_episode`, critic OFF | gabab OFF | nav **works** (sumFinalQ 0.50, snc 6.93 Hz) |
| Nav-loop D | real nav loop, critic OFF | gabab **forced ON** (conductance alloc, mask None) | nav **works** (sumFinalQ 0.50, snc 6.76 Hz), g=0 |
| Nav-loop A | real nav loop, **critic ON** | gabab ON (as built) | **SILENT** — `[CRITICAL] … broadcast … (236160,) (169139,)` every step; snc 0 Hz; sumFinalQ 50.87 |
| Nav-loop B | real nav loop, critic ON | gabab forced OFF | nav **works** (snc 8.11 Hz, sumFinalQ 0.50) |

Two things jump out: **(i)** the flag is completely inert in the constant-drive free-run (P1==P0, P2 active) AND in the
nav loop when the mask is empty (D works) — so the bare flag is harmless; **(ii)** the silence appears ONLY when the
critic's real GABA_B mask exists AND the full nav loop runs (A), and it comes with a per-step **exception**, not a
hyperpolarization.

**The exception is the whole story.** `(236160,)` vs `(169139,)`:
- At build time I measured `cp_connections.nnz = 169139`, `cp_connections.data.shape = (169139,)`,
  `cp_gabab_synapse_mask.shape = (169139,)` — all aligned.
- `apply_v1_gabor_weights` runs **after** `_initialize_simulation_data` (runner ~line 3447) and calls
  `bridge.set_pathway_weights(...)` (visual_cortex.py:303), which **grows `cp_connections`** (adds the dense Gabor
  retina→V1 synapses that the sparse `density=0.05` wiring did not include) → live nnz becomes 236160.
- The GABA_B per-step block (bridge.py:5504-5514) does
  `_gb_data = effective_connections_matrix.data * self.cp_gabab_synapse_mask[:cp_connections.nnz]`. With STP/neuromod
  gain off, `effective_connections_matrix is self.cp_connections`, so `.data` is length **236160**; the mask is only
  **169139** long, and `mask[:236160]` is still just 169139. → `(236160,) * (169139,)` broadcast error.
- The bridge catches it (the `[CRITICAL] Error during simulation step` line), the step is aborted, nothing
  propagates, `cp_firing_states` stays 0 everywhere → SNc 0 Hz → DA signal dead → `cortex_it` weight frozen, actor
  frozen → nav distance ~50 (never reaches goal). The "SNc silenced" symptom is downstream of the throw.

Why the free-run P2 *didn't* throw: it built the bridge via `build_bg_brain_regions` and stepped directly, **without
calling `apply_v1_gabor_weights`**, so nnz stayed 169139 and the mask matched. That is precisely the difference
between the isolated probe and the deployed nav stack — a textbook "probe must match deployment" trap
(`feedback_probes_match_deployed`).

**Falsified along the way** (so the doc doesn't relitigate them): the "slow GABA_B accumulates over the nav loop"
prime-suspect (g=0 throughout, confirmed in A's own trace); a CuPy CSR aliasing/`.T @` mutation of the shared
`indices/indptr` (I ran an isolated 5× transpose-matvec test on an aliasing CSR — indices/indptr/data all unchanged,
forward matvec still correct); a global `E_gabab=-90` hyperpolarizing the SNc (the SNc-tonic isolation in the prior
diag fires fine with the flag on).

**Mechanism class:** a **per-synapse-array capacity/growth gotcha** (one of the candidates the NEGATIVE doc listed).
The GABA_B mask is the third per-synapse array (after `cp_synapse_plastic_mask` and the plasticity-gate index map)
that must track `cp_connections` growth; the plasticity arrays are sized to **capacity** (253708) and sliced to nnz,
but `cp_gabab_synapse_mask` is sized to nnz-at-build and then read as `mask[:live_nnz]` — which silently under-slices
after growth. (Note bridge.py:2253 allocates `cp_gabab_synapse_mask` to `self._synapse_capacity` = 253708 in the
*capacity* branch — but the failing path shows a 169139-long mask, i.e. the realized array is nnz-sized in this build;
either way it is shorter than the post-Gabor 236160 and the element-wise multiply against the full `.data` throws.)

> **Fix shape (for the controller, NOT done here):** make the GABA_B block length-safe — slice
> `effective_connections_matrix.data` to `cp_connections.nnz` the same way the mask is sliced (so both are
> `[:nnz]`), OR rebuild/realign `cp_gabab_synapse_mask` whenever `cp_connections` grows (mirror how the
> plasticity-gate indices are handled), OR build the critic mask AFTER `apply_v1_gabor_weights`. This is a small,
> guarded, default-off-safe `sim/` (or runner-ordering) change. It does NOT touch the validated Pavlovian de-risk
> (that topology never grows nnz post-build, which is why it passed).

### 1.2 Blocker 2 — the critic afferent is the wrong stream (the load-bearing biology error)

`cortex_it` is IT (inferotemporal), the apex of the **ventral "what" stream** — explicitly **position-invariant**
(that is its computational job: recognize the object regardless of where it is). The runner even hard-asserts the
afferent is "perceived state, not a coordinate," which is satisfied — but **position-invariance is exactly wrong for a
value-of-location.** And empirically `it_mean = 0` over 16k steps in every condition (it is not driven in nav — the
retina/V1/V2/IT stack exists but the nav loop doesn't feed it object imagery that makes IT fire). So the afferent is
**doubly wrong: inactive AND position-invariant.** This is the same ventral-vs-dorsal root-cause CLAUDE.md already
records for the nav perceptual cold-start.

This is the blocker that actually requires a redesign (Blocker 1 is a bug fix). It is addressed in §2.

---

## 2. The right critic AFFERENT for a SPATIAL value in continuous nav

**What the striatal patch/striosome critic reads in the brain — catalog + literature:**

- **Catalog C.30 (Actor-critic; Houk-Adams-Barto 1995; Schultz98 Fig 9C):** "**striosome-patch (limbic striatum) =
  critic state-value**; striatal matrix (sensorimotor) = actor preferences." The critic's state input is **limbic**.
- **Catalog B.07 (patch/matrix):** "patch (striosome) ↔ ventral midbrain DA neurons (**limbic**) … identified by
  μ-opioid receptor." Striosomes **receive limbic-cortex input and project monosynaptically to midbrain DA**.
- **Web-confirmed (this session):** striosomes "receive inputs from the **limbic cortex** and project monosynaptically
  to midbrain dopaminergic neurons" and "could serve as the **critic** of the actor-critic model" (eNeuro 2018;
  ScienceDirect 2020). The **limbic** afferents to the value system are **vmPFC, amygdala, and (the spatial one)
  ventral hippocampus**.
- **The spatial-value pathway specifically:** the **hippocampus → ventral striatum** projection is the canonical
  substrate for *place*-value. "Forming a place–reward association depends critically on communication between the
  hippocampal formation and the ventral striatum"; hippocampal **place cells lead** ventral-striatal reward cells in
  replay (Lansink 2009, PLoS Biol); ventral-striatal neurons show **expectation-of-reward at spatial decision points**
  driven by hippocampal representations (van der Meer & Redish 2009). Kandel 6e Ch 43 lists hippocampus among the
  A10/VTA RPE targets (C.22).

**Translation to the nav runner — which EXISTING region is the faithful + ACTIVE afferent:**

The runner already builds the **dorsal/place machinery**: `sensor_place_readout` (the place-cell readout, a Gaussian
place code over agent `(x,y)`) and `ppc_goal_input` (a goal-vector code), enabled by **`--enable-place-goal-readout`**
(alias `--hippocampus`). Critically, `sensor_place_readout` is **driven every nav step** (runner ~line 4624-4626:
`place_drive = max_pA·exp(−‖pref−(x,y)‖²/2σ²)`), so it is **position-SENSITIVE and ACTIVE** — the exact opposite of
`cortex_it` on both axes. This is the project's stand-in for the hippocampal place code that, in biology, feeds the
ventral-striatal/striosomal critic.

> **Recommended afferent: `sensor_place_readout` → `striosome_value` (plastic, gate `value_input`).** It is the
> faithful biological analogue (place code → limbic/ventral-striatal value), it is active in nav, and it is
> position-sensitive so a *value-of-location* can actually be learned. (`ppc_goal_input` is the goal-vector; the
> value of a state in this task is "how close am I to the goal," so a *union* of `sensor_place_readout` +
> `ppc_goal_input` is defensible and richer — but start with `sensor_place_readout` alone to keep the de-risk clean
> and the anti-cheat tight. See the anti-cheat note in §5: the place code must be a *perceived-position* code, not a
> coordinate fed as a number — the Gaussian place-cell drive qualifies; passing raw `(x,y)` as a scalar would not.)

**Caveat the controller must weigh (honest):** `sensor_place_readout`'s drive is computed host-side from `(x,y)` via a
Gaussian (the project's standing place-cell encoding). Under the strict BRAIN-BASED-ONLY bar this is the *same*
defensible "sensory render of the world's state" the rest of the perception arc uses (it is a population code, not a
coordinate handed to a formula) — but it is on the boundary, and a fully spiking place-cell layer that *self-organizes*
from landmark sensors (`--enable-landmarks --landmarks-replace-place`, already in the runner) is the stricter,
preferred long-run afferent. For the de-risk, the Gaussian place code is the right cheap first step; flag the landmark
self-organized version as the follow-on hardening.

---

## 3. Ranked options for the neural value SUBTRACTION in a continuous loop

The Pavlovian probe subtracts on a clean cue→reward schedule (discrete trials, a reward-hold window). The nav loop is
continuous, and the value must be subtracted from the SNc at the reward moment. Ranked by **fidelity × likelihood ×
surface**, each with a cheap-first de-risk.

### Option A (RECOMMENDED) — keep the validated GABA_B subtraction, gate the critic→SNc current to the reward window, fix the mask bug
- **What:** Use the exact validated mechanism (`striosome_value → snc`, `receptor="gaba_b"`, host `−V` term dropped),
  but (1) fix Blocker 1's stale-mask bug, and (2) restrict the value subtraction to the **reward-hold window** where
  the SNc's `_I_snc = tonic + reward_gain·max(0,r)` is asserted — i.e. the value cancels reward *exactly when reward is
  delivered*, matching the discrete timing the Pavlovian de-risk validated. A `transmission_gate` on the
  `striosome_value → snc` route (open during the reward-hold sub-loop, closed otherwise) reproduces the validated
  timing inside the continuous loop without new biology. The critic still *learns* continuously (its afferent fires
  every step; STDP+δ run every step); only the *subtraction current* is windowed — biologically the phasic DA
  computation IS a brief event-locked window.
- **Fidelity:** high — it is the Eshel-2015/Tepper-Lee GABA_B→GIRK subtraction the de-risk already validated, with the
  reward-locked timing the de-risk used.
- **Likelihood:** high once Blocker 1 is fixed — the only nav-specific change is "when does the gate open," which
  reuses the shipped `transmission_gate` + `set_transmission_gate` machinery (CLAUDE.md, 2026-06-03).
- **Surface:** small. The `sim/` GABA_B support exists; the `transmission_gate` exists; the runner already has a
  reward-hold sub-loop to hang the gate-open on. The mask fix is the only genuinely new `sim/`/ordering change.
- **Cheap-first de-risk:** the CPU place-cell critic probe in §5 (it already exercises "windowed reward + GABA_B
  subtraction" if the probe drives reward in discrete holds — which the existing harness does).

### Option B — tonic place-value bias (slow), phasic reward unchanged
- **What:** Let the critic supply a **tonic** GABA_B bias proportional to V(s) continuously (no windowing); the SNc
  tonic floor is then state-dependent (higher value → lower baseline DA → smaller burst headroom), approximating
  `δ ≈ r − V` on average. This is the "accept value as a slow tonic bias" option.
- **Fidelity:** medium — tonic DA does encode motivational/value state (catalog C.22 "tonic DA encodes motivational
  state"), so a tonic value bias is biologically real; but the *phasic* RPE subtraction (the Schultz signature) is
  better captured by Option A's event-locked cancellation.
- **Likelihood:** medium — risk that a continuous GABA_B drag (τ=150 ms) over a busy nav loop with a *firing* critic
  silences or over-suppresses the SNc (the very failure mode the NEGATIVE doc feared — though we now know that fear was
  the bug, not the dynamics). Needs a careful `gabab_propagation_strength`/tonic rebalance.
- **Surface:** smallest (no gate; just fix the mask and pick the afferent).
- **Cheap-first de-risk:** same §5 probe but with the subtraction current left on continuously; compare burst headroom
  high-value vs low-value state.

### Option C — disynaptic disinhibition via SNr (the biology-literal route)
- **What:** value → (disinhibitor) → SNr-tonic-GABA → SNc, the textbook striosome→SNr/SNc disinhibition (the B'-SNr
  circuit the GABA_B de-risk explored). An odd inhibitory-link count makes "more V → less DA."
- **Fidelity:** highest anatomically (it is the literal BG→DA disinhibition, and the runner already has
  `gpi_X → snc` R3.10 + `str_striosome_X → snc` R3.11 as a scaffold).
- **Likelihood:** LOWER — the B'-SNr variant **already failed** the Pavlovian gap in the de-risk's predecessor work
  (`-Bprime-value-subtraction-circuit-research.md`); GABA_B superseded it. Re-opening it adds 2-3 relay populations and
  multi-hop tuning with no evidence it beats Option A.
- **Surface:** largest (new disinhib + SNr-tonic regions, multi-hop weight tuning).
- **Cheap-first de-risk:** not recommended ahead of A; only if A's GABA_B subtraction fails to port.

### Option D (honest fallback) — bank the Pavlovian GABA_B win + this honest negative; defer the nav critic
- **What:** If the §5 de-risk shows a place-cell critic can't learn a robust *state-of-location* value at nav scale
  (e.g. the place code is too sparse/aliased for STDP to carve a graded V), then the honest deliverable is: "the neural
  value subtraction validates Pavlovian but the *spatial* critic needs a richer state code (self-organized place cells
  / grid cells) before it ports to continuous nav." That maps a real substrate boundary and is a valid finding under
  the project's standard (honest negatives = deliverable).
- **When:** only after A's cheap de-risk is run, not instead of it.

**Ranking:** **A > B > C**, with **D** as the principled bail-out if A's de-risk is negative.

---

## 4. Reusable project machinery (what a redesign reuses vs what is genuinely new)

**Reusable (already shipped):**
- **GABA_B/GIRK `sim/` support** (commit a7370d49): the conductance, `cfg.enable_gabab` + the 3 gabab params, the
  per-`RegionPathway` `receptor` field, `fused_gabab_decay_and_current`, the per-synapse mask build. Validated,
  byte-identical-off. The subtraction primitive is done.
- **Spiking-SNc Stage-A drive** (runner ~5228): the `_I_snc = tonic + reward_gain·max(0,r) [− value]` membrane RPE,
  the `snc_rate_log` readout, the dopamine neuromodulator reading SNc firing (`from_region_firing_signed`). The actor
  side of actor-critic is in place.
- **The dorsal/place afferent**: `sensor_place_readout` + `ppc_goal_input` regions and their per-step Gaussian place/
  goal drive (`--enable-place-goal-readout`); the landmark self-organizing variant (`--enable-landmarks
  --landmarks-replace-place`). The right afferent *already exists and fires* — it just needs to be the critic's input.
- **The striosome scaffold**: `str_striosome_X → snc` (R3.11) and `gpi_X → snc` (R3.10) — the canonical striosome→SNc
  and SNr→SNc projections are already wired (Option C's substrate; also confirms the project already models the
  patch→DA route the critic rides).
- **The three-factor pipeline**: STDP eligibility × SNc-derived δ → weight update; `stdp_w_max` headroom; the
  `plasticity_gate` ("value_input") to freeze/thaw the critic afferent. The critic *learns* through existing rails.
- **`transmission_gate` + `set_transmission_gate`** (2026-06-03): the reward-window gating Option A needs — a per-route
  current gate, no new `sim/` code.
- **The de-risk harness**: `snc_stageb_critic_probe.py` (the CPU probe + its lesion/anti-cheat + auto-calibration of
  the DA threshold). The §5 de-risk is a small edit of this file (swap cue→place code), not new infrastructure.

**Genuinely new (small):**
1. **The Blocker-1 mask fix** (length-safe GABA_B block OR mask regrow-on-growth OR build-mask-after-Gabor) — the only
   new `sim/`/ordering change, and it is a bug fix, not a feature.
2. **Re-pointing the critic afferent** `cortex_it → striosome_value` ⇒ `sensor_place_readout → striosome_value`
   (runner-side; the assert + the `--enable-neural-critic` precondition change from "requires visual cortex" to
   "requires place-goal readout"). Pure runner edit.
3. **The reward-window gate** on `striosome_value → snc` (Option A) — runner wiring + a `set_transmission_gate` call in
   the reward-hold loop. Pure runner edit.

So the redesign is **~1 small `sim/` bug fix + ~3 runner edits**, reusing the entire validated subtraction + learning
substrate. It is not a from-scratch build.

---

## 5. Recommended cheap-first de-risk (CPU/small-bridge) + anti-cheat controls

**Goal:** before ANY nav build, validate the ONE new scientific claim — *"a GABAergic striosome critic driven by a
PLACE code learns a value-of-LOCATION (graded V(s) that is high near goal, low far from goal) and subtracts it at the
SNc through GABA_B, producing a state-specific RPE."* Mirror the Pavlovian probe exactly; swap the cue for a place code.

**Probe design (edit of `snc_stageb_critic_probe.py`, CPU `SIM_BACKEND=numpy`, ~minutes):**
- Replace the single `cue` region with a small **place-code input** of K cells with Gaussian tuning over a 1-D
  "corridor" position (the minimal spatial analogue): drive a **near-goal** state (state A) and a **far** state
  (state B) as two distinct K-of-N place-cell activations (not a scalar).
- Keep `striosome_value` (GABAergic MSN-D1, graded), `snc` (depolarized E_GABA), the plastic `place → striosome_value`
  afferent trained by the SNc δ, and the `striosome_value → snc` `receptor="gaba_b"` subtraction — all unchanged from
  the validated recipe.
- Training schedule = the place-reward analogue of CS→US: visiting state A is followed by reward; state B is not (or a
  smaller reward). The critic should learn `V(A) > V(B)`.
- **Acceptance gates (mirror the Pavlovian PRIMARY gate):**
  1. **V-learned-spatial:** `striosome_value` firing on state-A drive RISES across training and ends **higher for A
     than B** (a graded value-of-location, not a single cue).
  2. **State-specific RPE gap:** reward at state A (predicted) → **small** SNc burst; the SAME reward at state B
     (unpredicted-by-that-state) → **big** SNc burst (`gap_ratio > 1.30`). A host global-EMA value cannot produce a
     per-LOCATION gap → this is the discriminator that proves the value is neural AND spatial.
  3. **Omission dip:** state A with reward omitted → SNc dips below tonic.

**Anti-cheat controls (the probe must include all three):**
- **(a) Afferent provenance / perceived-position not coordinate.** The critic's input must be the **place-cell
  population code** (K Gaussian-tuned cells), NOT a scalar position handed to a formula. Assert the afferent region is
  the place code; verify driving it with a *different* position produces a *different* ensemble (so the value is read
  off a perceived spatial pattern, exactly as `sensor_place_readout` works in nav).
- **(b) Conductance/critic lesion.** After training, zero the GABA_B mask (`_lesion_gabab_mask`, already in the probe):
  the state-specific gap must **vanish** (SNc bursts to reward at both A and B) → proves the subtraction is carried by
  the critic's GABA_B current, not a host term. AND/OR lesion the `place → striosome_value` afferent: V must collapse
  to baseline → proves V is learned from the place code, not structural.
- **(c) A/B vs the host-value Stage A.** Run the same place schedule with `--gabab` OFF (GABA_A direct) — it should
  FAIL the gap (reproducing the depolarized-SNc wall), localizing the win to GABA_B. AND compare against Stage A's
  host `−V_scaffold` (the EMA critic): the host EMA gives the **same** V regardless of which state, so it **cannot**
  produce the per-state gap — the neural place critic must beat it on state-specificity (the whole point of going
  brain-based).

**Only after this CPU de-risk PASSES** (≥ the Pavlovian bar, multi-seed 42/43/44): apply the Blocker-1 mask fix + the
3 runner edits, then the nav A/B (flagship + `--spiking-snc --enable-neural-critic` with the place afferent), acceptance
= summed reward ≥ Stage A. An honest negative at the nav stage is still a valid deliverable (it would map "the spatial
critic learns in isolation but the place code is too aliased at nav scale," → the self-organized place-cell follow-on).

---

## 6. Honest verdict

The roadblock is **not** evidence that brain-based-only nav value subtraction is fundamentally hard. It decomposes
into **a fixable bug** (Blocker 1 = a stale per-synapse mask vs the Gabor-grown CSR, throwing a swallowed exception
every step — fully pinned this session, ~1 small `sim/`/ordering fix) and **a wrong afferent** (Blocker 2 = the
position-invariant ventral IT stream instead of the position-sensitive dorsal place code that biology actually uses
for spatial value). The validated GABA_B/GIRK subtraction and the entire learning substrate are reusable; the redesign
is small and well-scoped. The single load-bearing scientific uncertainty that justifies a cheap-first de-risk is
whether a striosome critic can learn a **graded value-of-location** from a place code (vs the trivial single-cue value
the Pavlovian probe already validated). Run that CPU probe (with the three anti-cheat controls) before spending GPU on
a nav build. If it passes, Option A (windowed GABA_B subtraction, place-cell afferent) is the recommended path; if it
fails, banking the Pavlovian GABA_B win + this honest negative (Option D) and moving to the N2/N7 characterizations is
the right call.

---

### Appendix — diagnostic artifacts (read-only, this session)
- `research/findings/raw/_gabab_flag_isolation_probe.py` + `_gabab_flag_isolation_result.json` — free-run P0/P1/P2
  (flag inert without the real nav loop).
- `research/findings/raw/_gabab_navloop_isolation_probe.py` + `_gabab_navloop_isolation_result.json` — nav-loop
  critic-OFF + flag-forced-ON works (g=0) → bare flag harmless in the live loop.
- `research/findings/raw/_gabab_AvsB_reproduce_probe.py` + `_gabab_AvsB_reproduce_result.json` — reproduces A=silent /
  B=works AND surfaces the decisive `[CRITICAL] … broadcast … (236160,) (169139,)` per-step exception.
- (inline) aliasing-CSR transpose-matvec mutation test — NEGATIVE (no shared-array corruption); build-time
  `cp_connections.data == nnz == mask == 169139` confirmed; the 236160 arises only after `apply_v1_gabor_weights`.
