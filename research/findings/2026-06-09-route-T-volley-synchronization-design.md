# Design — Route T: SYNCHRONIZING the sparse place ensemble into a COINCIDENT VOLLEY so Route D fires (theta-gamma volley ± conduction delays)

**Date:** 2026-06-09
**Type:** Deep-research + DESIGN pass (the SECOND substrate upgrade). **READ-ONLY — ZERO `sim/` edits made.** This is a *proposal* for the owner to byte-review before anything lands, per the protected-edit discipline (the GABA_B `receptor=` + per-region-NMDA + per-region-homeostasis + graded-lateral + `exc_receptor=nmda_slow` + the just-landed `coincidence_detector` precedents).
**Owner directive:** biologize everything, no banking, brain-based-only. The owner said "proceed" → start Route T. An honest negative IS the deliverable.
**Solves the ASYNCHRONY wall mapped by:** `2026-06-09-coincidence-substrate-upgrade-design.md` (Route D, landed `b980070a`), `2026-06-09-learned-graded-ca3-derisk-RESULT.md` (the rate-coding wall), and the Step-0 calibration `research/runners/coincidence_wall_probe.py` + `research/findings/raw/_coincidence_wall_probe.json` (the decisive `c_i ≤ 1` numbers).
**Builds on:** Route D ALREADY LANDED + VALIDATED INFRA (`enable_coincidence_detection` + `RegionPathway.coincidence_detector` + `fused_coincidence_plateau`; a no-op alone, banked for when T composes). The staged de-risk bed `research/runners/coincidence_n9_derisk.py` already carries the jitter/ablate/K-sweep anti-cheats AND explicitly notes "run only AFTER Route T (or with an externally-imposed synchronous volley)."

---

## 0. TL;DR (the recommendation, up front)

**The diagnosis is precise and not in dispute (Step-0 proved it across 3 CuPy seeds):** a sparse-distinct ~10 Hz place ensemble (≤5% active, each active cell **0.01 spk/step** — i.e. ~once per 100 steps) delivers, to any downstream target, **c_i ≤ 1 coincident routed input per step** (per-step max-over-targets c_i: p90 = **0–1**, max = **2**, across seeds 42/43/44), *even though each target's fan-in is ~200 source cells (density 0.5) and even at full density 1.0.* **The bottleneck is asynchronous EMISSION, not convergence:** the cells are wired to converge, but they fire at random, uncorrelated phases, so the probability that ≥2 of a target's afferents fire *in the same 1 ms step* is ~0. Route D's coincidence detector therefore has nothing to detect (`c_i < K` for any K>1, every step → the plateau never switches on). The instant you raise drive enough to get c_i≥2 (intensity ×1.5), the source goes **48% dense** and position-blind (`sparse_and_coincident=false` at every intensity, every seed). **This is the irreducible ASYNCHRONY half of the wall: a sparse-distinct code emits spikes too sparsely-in-time to coincide, and densifying it to coincide destroys distinctness.**

**The job (verbatim from the task):** make the sparse ensemble fire as a SYNCHRONIZED COINCIDENT VOLLEY (≥K cells in the same step) so Route D fires and the sparse-distinct code drives the downstream MSN — WITHOUT making the code dense/position-blind.

**The faithful fix — a THETA-GAMMA VOLLEY (a PING gamma rhythm), ± conduction delays:**

| | **(T-rhythm) gamma/PING synchronizing volley** | **(T-delay) per-synapse conduction delays (ring buffer)** |
|---|---|---|
| Mechanism | a population gamma rhythm (PING via FS interneurons, or a pacing drive) compresses the *currently-active* cells' spikes into one ~10–25 ms gamma window → the active cells fire together → c_i jumps ≥K **in the volley step**; inactive cells stay silent (distinctness preserved) | each synapse carries an axonal delay (1–10 ms) so an ensemble's spikes emitted at slightly different times *arrive in the same step* (Izhikevich-2006 polychrony) |
| Closes the ASYNCHRONY wall? | **YES directly** — a gamma cycle is the canonical biological device that turns a population of asynchronous ~10 Hz cells into a coincident packet; "neurons firing within the same gamma cycle are co-grouped" (catalog N.19, Buzsáki Cycle 9). It synchronizes a *stochastic* ensemble, which delays alone cannot. | **NO, not on a stochastic ensemble** — a *fixed* per-synapse delay aligns a *fixed* temporal pattern; it cannot align spikes whose emission times are RANDOM (a place cell's exact spike time within its ~10 Hz train is stochastic, so no constant Δt brings the ensemble into one step). Delays *fine-tune* an already-quasi-synchronized volley; they do not create synchrony from randomness. |
| Engine reuse | **VERY HIGH — possibly RUNNER-SIDE, NO `sim/` edit.** Gamma is ALREADY an emergent neural behavior on this engine (the `gamma-oscillations` benchmark: FS interneurons + drive → 30–80 Hz PING, no special config). Add an FS interneuron pool to the place pool (PING) and/or a sinusoidal `excitability_drive` theta/gamma pacing (the existing neuromodulator framework, catalog D.18/D.24 recipe) — both runner-side. | **LOW — a genuinely new hot-path state array.** Replaces the uniform 1-step `cp_prev_firing_states` (a single `(n,)` bool) with a `(max_delay_steps × n)` ring buffer + a per-synapse delay index + a delayed-gather propagation path. The catalog flags it explicitly missing (B.16). Touches the inner matvec. ~150–250 lines, harder byte-identity story. |
| Verdict | **THE FAITHFUL SYNCHRONIZER + the cheap first move** | the temporal *fine-tuning* half — needed ONLY if the rhythm alone can't tighten the volley enough; the proper SECOND-of-the-second upgrade |

**Recommendation: (T-rhythm) gamma-volley FIRST, almost certainly RUNNER-SIDE (NO `sim/` edit), tested on the EXISTING `coincidence_n9_derisk.py` bed + the already-landed Route D.** Because (a) a gamma volley is *the* biological mechanism that synchronizes a stochastic ensemble (delays cannot — see §1.2, the load-bearing subtlety), (b) the engine *already produces gamma* (FS-PING benchmark) and *already has a sinusoidal-excitability-drive path* (neuromodulator framework), so the first probe needs **zero `sim/` edits** — it adds an FS pool / pacing drive to the place pool in a runner and asks "does a gamma volley make a sparse ensemble fire the Route-D target, and does jitter collapse it?", and (c) conduction delays alone leave the stochastic-emission wall standing (they're the temporal *fine-tune*, not the synchronizer). **(T-delay) conduction delays are the proper follow-on** — they're independently on the roadmap (SH-3 / catalog B.16), they compose with the rhythm (delays align the gamma volley *tighter*, Izhikevich-2006), and they're the faithful fix for the *residual* temporal jitter — but they are NOT the first move and NOT obviously needed if the gamma window already supplies K coincident inputs.

**Net `sim/` surface for the recommended first move: NONE (runner-side).** The `sim/` edit (the conduction-delay ring buffer, §3.2) is **DEFERRED** to a second step, gated on the rhythm-only probe being insufficient. **Honest scope flag (the owner asked):** there is a real chance rhythm-alone suffices entirely runner-side (no second substrate upgrade needed for N9) — Step-0's own staged de-risk explicitly anticipates "an externally-imposed synchronous volley." If it does NOT suffice (the gamma window is too wide / the place cells won't entrain at <0.2 spk/step), the conduction-delay ring buffer is the genuine `sim/` upgrade, and §3.2 specifies it exactly.

**Build smallest-first (§5):** Step 0 (RUNNER-ONLY, no `sim/` edit) — add a gamma synchronizer (FS-PING pool and/or sinusoidal-excitability pacing) to the place `source` pool in `coincidence_n9_derisk.py`, turn Route D ON, and run the load-bearing pair: **does the volley fire the MSN (G_FIRE) AND does jitter/de-sync collapse it (the coincidence anti-cheat)?** Only if the rhythm can't supply K coincident inputs does Step A (the conduction-delay `sim/` ring buffer, byte-review) follow.

---

## 1. Diagnosis — why delays-alone fail on a stochastic ensemble; why a theta-gamma volley is the faithful synchronizer

### 1.1 The asynchrony wall, tied to the Step-0 `c_i ≤ 1` numbers

The Route-D design closed the *rate-coding* half (a supralinear coincidence detector — landed). Step-0 then measured whether the natural sparse-distinct dynamics actually *deliver* coincidence for that detector to read. They do not. From `_coincidence_wall_probe.json` (3 CuPy seeds, the validated Stage-1 place code, `source→target` density 0.5, fan-in mean ~200 per target):

```
                 source       active-rate   per-step max-over-targets c_i      sparse &
   seed  sparsity (≤5%)  (spk/step)   p90   p99   max    coincident?
    42      4.25%          0.010        1     1     2     NO
    43      3.96%          0.010        0     1     1     NO
    44      4.33%          0.010        0     1     2     NO
```

And the intensity sweep (seed 42; raising per-cell rate to force coincidence):

```
  intensity  source-sparsity  c_i p90  trigK2  sparse_and_coincident?
    ×1.0        ~4%             0       0.00      NO   (sparse, but c_i≤1 -> Route D never fires)
    ×1.5        48.3%           3       0.23      NO   (c_i≥2 appears, but source now 48% DENSE -> position-blind)
    ×2.0        76.8%           5       0.41      NO
    ×3.0        94.3%           9       0.59      NO
```

**`no_valid_K_above_1 = true`, `all_seeds_have_sparse_coincident_window = false`.** There is no operating point that is BOTH sparse-distinct (≤8% cells) AND delivers c_i≥2 per step. The mechanism is purely temporal: at 0.01 spk/step, a target's ~200 afferents collectively fire ~2 spikes per step *spread across the population*, but the chance that ≥2 land on the SAME target's afferent set in the SAME 1 ms step is ~0 because the cells' phases are uncorrelated. **Convergence (density 0.5→1.0) does NOT help** — Step-0 verified the bottleneck is the per-step *emission* count of the active ensemble, not how many of them wire to a given target. This is the **asynchrony wall**: Route D can fire a cell from K *coincident* inputs, but a sparse-distinct ensemble never emits K *coincident* inputs unless something synchronizes its spikes into the same step.

### 1.2 THE LOAD-BEARING SUBTLETY: fixed conduction delays CANNOT synchronize a *stochastic* ensemble

This is the design subtlety the task flags, and it is decisive for the route ranking. A per-synapse conduction delay `d_s` shifts synapse `s`'s contribution from "arrives 1 step after the presynaptic spike" to "arrives `d_s` steps after." That re-aligns a **deterministic** temporal pattern: if cell A always fires at phase φ_A and cell B at φ_B, a delay pair `(d_A, d_B)` can be chosen so both land in the same step. **But a place cell's spike time within its ~10 Hz train is STOCHASTIC** — it is set by the integrate-and-fire crossing of a noisy/asynchronous drive, not by a fixed phase. There is no constant `(d_A, d_B, …, d_K)` that brings K stochastically-timed spikes into one step on every cycle, because the emission times themselves jitter from cycle to cycle. Izhikevich-2006 polychrony works precisely because the *upstream* spikes are already time-locked (a "polychronous group" is a *learned, repeatable* spatiotemporal pattern, and STDP+delays select for it); it is not a method for synchronizing a free-running stochastic population.

**Biology's actual answer is a RHYTHM, not delays.** Hippocampal place cells do not fire at uncorrelated phases — they fire in **theta-gamma volleys** and **phase-precess** (O'Keefe-Recce 1993; Buzsáki Cycle 11): the active place cells are already quasi-synchronized *within a gamma cycle*, packed into a ~10–25 ms window nested in the theta cycle (Lisman-Idiart 1995, catalog N.15; "each theta cycle hosts 7–9 nested gamma cycles, each carrying a distinct cell-assembly"). Within one gamma cycle, "neurons firing within the same gamma cycle are co-grouped, neurons offset by a half-cycle are segregated" (catalog N.19, Buzsáki Cycle 9). **The gamma cycle and the STDP window are matched in duration** — the same ~10–40 ms window Route D needs for coincidence. So the synchronizer that makes a sparse ensemble's spikes coincide is the **gamma rhythm**; conduction delays then **fine-tune** the alignment within that already-narrow window (and, with STDP, select repeatable polychronous sub-groups). **Rhythm packages; delays sharpen.** Delays alone, on a free-running stochastic ensemble, leave the asynchrony wall standing.

### 1.3 The engine facts that ground the route choice

- **The 1-step delay (SH-3) is a single `(n,)` bool array.** `sim/bridge.py:1063` allocates `self.cp_prev_firing_states = cp.zeros(n, dtype=bool)`; `:6694` writes `self.cp_prev_firing_states[:] = fired_this_step`; every conductance matvec (`:5620`, `:5709`, `:5752`, `:5796`) reads it as "the previous step's firing." There is **one** step of history; all synapses share it. This is exactly the SH-3 shortcut.
- **`max_synaptic_delay_ms = 20.0` already exists** (`sim/config.py:189`) and `runtime_state.max_delay_steps` is set from it (`bridge.py:2081`), **but neither is used by propagation** — they exist for legacy/viz timing only. So the config knob for the ring-buffer depth is already present (Route T can reuse it).
- **Gamma is ALREADY an emergent neural behavior on this engine, with NO special machinery.** `run_benchmarks.py:613` `benchmark_gamma_oscillations` builds `CORTEX_GAMMA_FS_NETWORK` (FS interneurons + ~50 pA drive) and the network self-organizes into 30–80 Hz PING — "excitatory neurons fire → recruit FS interneurons → FS inhibition silences the network → inhibition decays → excitatory neurons fire again." This is the genuine neural rhythm (catalog N.19: gamma frequency set by the GABA_A decay time, ~10–25 ms → 40–100 Hz). **No `sim/` edit is needed to get a gamma rhythm — it falls out of FS interneurons + drive.**
- **A sinusoidal `excitability_drive` pacing path ALREADY exists** in the neuromodulator framework (`sim/neuromodulators.py`, `target_type="excitability_drive"`, scope=all / group:NAME / trait:N). The catalog's D.18/D.24 recipe for theta is literally "sinusoidal `excitability_drive` from a `septum` neuromodulator at 8 Hz" — a host-paced clock (a documented SH-5 shortcut: a global scalar), but a *legitimate-as-environment-pacemaker* one if framed as the septal/medial-septum theta generator. (Brain-based-only note in §4.)
- **The current place `source` pool has NO synchronizing machinery.** In both `coincidence_wall_probe.py` and `coincidence_n9_derisk.py` the `source` pool is built with `internal_density=0.0` and `exc_fraction=1.0` (NO FS interneurons, NO recurrence) — so there is *currently nothing* to make the ensemble synchronize. That is *why* c_i ≤ 1: the cells are pure independent integrators of their afferent drive. **Adding an FS-PING pool (or a gamma pacing drive) is the missing synchronizer, and it is a runner-side region/pathway addition — no `sim/` edit.**
- **Route D is landed and ready** (`sim/config.py:173-178`, `sim/regions.py:284-294`, `sim/bridge.py:5724-5768`, `sim/kernels.py:253` `fused_coincidence_plateau`). It reads `c_count = (binary routed mask).T @ prev_fired` per step and switches on a plateau when `c_count ≥ coincidence_k_threshold`. **A synchronized volley is exactly what makes `c_count` cross K.**
- **In-substrate "firing drives a gate" precedent exists** (`bridge.py:2958` `couple_gate_to_pool` / `:2981` `_apply_gate_couplings`) — a fully-neural option for a rhythm-gated route if a host pacing clock is judged too shortcut-y (§4 brain-based-only note).

---

## 2. The mechanism design (grounded in the literature)

### 2.1 (T-rhythm) the gamma/PING synchronizing volley — RECOMMENDED FIRST (runner-side)

**Concept (Buzsáki Cycle 9 PING; Lisman-Idiart theta-gamma multiplex):** give the sparse place `source` pool a gamma rhythm so its *currently-active* cells fire together inside one gamma cycle (a coincident packet), while inactive cells stay silent. Two compatible, runner-side ways to produce it (try the most-neural first):

- **(a) PING via an FS interneuron pool (MOST faithful, fully neural).** Add a small FS interneuron sub-pool to the place pool with reciprocal `place → place_FS` (EXC) and `place_FS → place` (GABA_A) connectivity, exactly as `CORTEX_GAMMA_FS_NETWORK` does. The active place cells excite the FS pool, the FS pool's GABA_A inhibition silences the population for ~10–25 ms (the GABA_A decay), then releases — so the active cells re-fire **together** on each gamma cycle. The coincidence is *generated by the network's own inhibition*, not imposed. This is the brain-based-only ideal: synchrony from neurons + synapses. (Catalog N.19; the existing gamma benchmark proves the engine does this.)
- **(b) A sinusoidal gamma/theta pacing `excitability_drive` (cheaper, partly host-shortcut).** Use the neuromodulator framework to add a sinusoidal `excitability_drive` to the place pool at gamma (~40 Hz) — a depolarizing pulse train that, on each cycle's peak, pushes the *already-depolarized* (active) cells over threshold together while leaving sub-threshold (inactive) cells silent. This is the catalog D.18/D.24 "septal theta `excitability_drive`" recipe extended to gamma. It is a host-paced clock (SH-5), so it is the *scaffold* version; (a) is the conversion to neural synchrony (the innate-pacemaker-teaches-the-network pattern the project already uses elsewhere). Either supplies the volley for the de-risk; (a) is the faithful endpoint.

**Why the volley fires Route D AND stays position-specific:** on the volley step, the K active cells (only the ones whose place field includes the current location) fire in the same step → for a target wired to those cells, `c_count` jumps from ≤1 to **≥K** → the Route-D sigmoid switches → the regenerative plateau fires the MSN. At a *different* location, a *different* sparse set is active → a *different* set of targets gets the volley → the code stays distinct (the threshold is on *which* cells coincide, not on total rate). **Crucially, the rhythm does NOT make the code dense:** it re-times the *same* sparse active set into one step; the inactive 95% of cells never fire (they're sub-threshold; the gamma window only synchronizes who was already going to fire). This is the exact property the densify-to-coincide approach destroyed.

**The minimal faithful form:** option (a), an FS-PING pool on the place source, is the minimal *faithful* synchronizer (neurons + synapses only). Option (b) is the minimal *cheap* synchronizer (one neuromodulator config). The de-risk can compare both; the recommendation is to validate the capability with (b) if (a) needs tuning, then land (a) as the brain-based-only version.

### 2.2 (T-delay) conduction delays — the temporal fine-tune (the `sim/` upgrade, DEFERRED)

**Concept (Izhikevich-2006 polychrony; catalog B.16; the MSO Jeffress delay-line):** replace the uniform 1-step `cp_prev_firing_states` with a `(max_delay_steps × n)` ring buffer of firing history, and give each synapse (or pathway) a conduction delay `d_s ∈ [1, max_delay_steps]`, so each synapse's contribution is gathered from `d_s` steps ago. **Within the gamma volley** this tightens alignment: cells that fire a step or two apart inside the gamma window can be delay-shifted into the *same* arrival step at the target, raising the per-step coincidence further and letting STDP+delays carve repeatable polychronous place-sequence groups (the theta-sequence compression of catalog D.24).

**Why it is the SECOND move, not the first (§1.2):** delays fine-tune an *already-synchronized* volley; on a free-running stochastic ensemble with no rhythm they cannot create the synchrony. So Route T's true payoff is **rhythm + delays composed**: the gamma cycle packages the active cells into a window, the delays align them within it. If the gamma window already supplies K coincident inputs (likely — a gamma cycle is ~10–25 ms ≈ many dt steps, but the *volley itself* is the synchronizing event), conduction delays may be unnecessary for N9 and are then a general fidelity upgrade for later (polychrony, theta sequences, the catalog B.16 BG three-phase response). **Build only if the rhythm-only probe is insufficient.**

### 2.3 Hybrid (the faithful endpoint, noted)

The biologically complete picture is **theta-gamma volley (rhythm) + conduction delays (fine-tune)**: the medial-septum theta carrier paces the gamma cycles (Lisman-Idiart nesting), the FS-PING generates the gamma volley, and per-synapse delays align the within-volley spikes and let STDP store phase-precessed sequences (catalog D.24). This is the full Route-T endpoint; the recommendation is to reach it incrementally — rhythm first (runner-side, closes N9), delays second (the `sim/` upgrade, the general polychrony capability) — rather than build the ring buffer before the cheaper rhythm probe settles whether it's even needed.

---

## 3. The exact byte-level `sim/` surface

### 3.1 RECOMMENDED FIRST MOVE — RUNNER-SIDE ONLY (NO `sim/` edit)

The gamma-volley probe is built ENTIRELY in the EXISTING staged runner `research/runners/coincidence_n9_derisk.py` (which already has the source→target topology, Route D wiring, and the jitter/ablate/K-sweep anti-cheats), by adding a synchronizer to the `source` pool. **No `sim/` change.** Concretely (runner edits only):

- **(a) FS-PING option:** in the runner's `_build`, add a `place_FS` `BrainRegion` (`exc_fraction=0.0`, FS Izhikevich preset, e.g. `IZH2007_FS_CORTICAL_INTERNEURON`) and two `RegionPathway`s: `source → place_FS` (EXC) and `place_FS → source` (GABA_A), mirroring `CORTEX_GAMMA_FS_NETWORK`. Tune the reciprocal weights so the place pool enters the PING regime (the gamma benchmark's parameters are the starting point). All existing fields; no new `sim/` surface.
- **(b) Pacing option:** set `cfg.enable_neuromodulator_subsystem = True` and declare a `NeuromodulatorConfig` whose `excitability_drive` target is scoped to the place pool (`scope="group:source"`, registered via `bridge.neuromodulator_manager.set_group_indices({"source": source_idx})`), driven by a `manual` rule the runner updates each step to a sinusoid `A·max(0, sin(2π f t))` at `f≈40 Hz` (gamma) — a per-step host write to the modulator concentration (the legitimate environment-pacemaker channel; see §4). All existing framework; no new `sim/` surface.
- **Turn Route D ON** (`cfg.enable_coincidence_detection = True`, `coincidence_detector=True` on `source→target`, `coincidence_k_threshold` from Step-0's calibration — start K≈3–5 and sweep). Run the de-risk's existing gates + anti-cheats (§4).

**This first move has ZERO `sim/` surface.** It is the cheap, decisive test of "does a gamma volley make a sparse ensemble fire a coincidence-detecting target, and does jitter collapse it" before any conduction-delay engine change.

### 3.2 DEFERRED `sim/` CHANGE — conduction-delay ring buffer (Route T-delay), ONLY if the rhythm-only probe is insufficient

Flagged here precisely for owner byte-review WHEN/IF needed; **NOT proposed for this pass.** All changes ADDITIVE, default-OFF, byte-identical-when-off, mirroring the precedent line. **This is the hot-path one — flag it.**

**(a) `sim/config.py`** — add (next to `max_synaptic_delay_ms`, line 189), default-OFF:
```python
# Per-synapse axonal CONDUCTION DELAYS (2026-06-09; Izhikevich-2006 polychronization;
# catalog B.16; the SH-3 uniform-1-step-delay shortcut). When True, the uniform 1-step
# firing-history (cp_prev_firing_states) is replaced by a (max_delay_steps x n) ring
# buffer, and each synapse gathers its presynaptic contribution from cp_conduction_delay_steps
# steps ago instead of always 1. Lets a gamma volley's within-window spikes be aligned into the
# SAME arrival step at the target (composes with Route T-rhythm), and lets STDP+delays carve
# polychronous place-sequence groups (theta compression, catalog D.24). Default False => the
# ring buffer is never allocated, propagation reads cp_prev_firing_states exactly as today, and
# every conductance matvec is BYTE-IDENTICAL (mirrors enable_coincidence_detection / enable_gabab).
enable_conduction_delays: bool = False
conduction_delay_default_ms: float = 1.0   # default per-synapse delay when a pathway sets none
```
(`max_synaptic_delay_ms = 20.0` and `max_delay_steps` already exist and are reused as the ring depth.)

**(b) `sim/regions.py`** — add to `RegionPathway` (next to `coincidence_detector`, line 294), default byte-identical:
```python
# conduction_delay_ms (2026-06-09): optional per-pathway axonal conduction delay (ms). When
# cfg.enable_conduction_delays=True, this pathway's synapses gather their presynaptic spike from
# round(conduction_delay_ms / dt) steps ago (clamped to [1, max_delay_steps]) instead of the
# uniform 1 step. None => use cfg.conduction_delay_default_ms. Biology: 1 ms pallidonigral ...
# 10 ms striatonigral (catalog B.16); the within-gamma-volley fine-tune (Route T-rhythm + delays).
# Default None = the uniform-1-step path (byte-identical when cfg.enable_conduction_delays is off).
conduction_delay_ms: Optional[float] = None
```
and plumb `round(delay_ms/dt)` per-synapse into `inject_explicit_wiring` as a per-synapse `cp_conduction_delay_steps` int array (mirroring how `coincidence_detector` extends the `keyed` tuple — append one more field, order-preserving).

**(c) `sim/bridge.py`** —
- **Allocate** (next to `cp_prev_firing_states`, ~line 1063), guarded: `self.cp_firing_history_ring = None` (a `(max_delay_steps × n)` bool array, allocated **only** when `cfg.enable_conduction_delays`), `self.cp_conduction_delay_steps = None` (per-synapse int), `self._ring_write_ptr = 0`. All None by default → ring path unreached.
- **Write the ring** alongside the existing `cp_prev_firing_states[:] = fired_this_step` (`:6694`): when the ring exists, `self.cp_firing_history_ring[self._ring_write_ptr] = fired_this_step; self._ring_write_ptr = (self._ring_write_ptr + 1) % max_delay_steps`. `cp_prev_firing_states` is STILL maintained (so the byte-identical default path and all the other matvecs that read it are unchanged).
- **Gather with delays** in the propagation matvec (`:5594-5621`): the byte-identical default keeps `prev_fired_float = cp_prev_firing_states.astype(float32)`. When the ring is active, the per-synapse delayed arrival is `cp_firing_history_ring[(write_ptr - delay_steps_s) % max_delay]` indexed per synapse — i.e. the matvec data is gathered per-synapse from the ring at each synapse's `delay_steps`. The clean way (avoids per-synapse Python loops): bucket synapses by integer delay `d ∈ {1..D}` and accumulate `Σ_d (W_d.T @ ring[(ptr-d)%D])`, where `W_d` is the CSR restricted to delay-`d` synapses (D small sub-matvecs; D≤20). **Byte-identity story:** with `enable_conduction_delays=False` the ring is None, every synapse is "delay 1," the single `W.T @ prev_fired` is taken unchanged → identical to today.
- **Cost flag (HOT PATH):** this turns 1 matvec into ≤D matvecs (one per distinct delay) and adds a `(D×n)` bool array. For D≈5–10 and the nav/probe scales this is modest, but it IS the inner loop — benchmark before/after, and keep the default path a literal `if self.cp_firing_history_ring is None:` short-circuit to the existing single matvec.
- **Save/load + capacity-growth** must handle the ring + the per-synapse delay array (mirror the per-synapse mask growth helper `_ensure_gate_capacity`).

**Diff size:** ~150–250 lines incl. docstrings + the bucketed-matvec + save/load. **Byte-identity proof (same as every precedent):** (i) `pytest tests/` green; (ii) a byte-diff harness running a g11 nav seed + a Tier-1 language seed with/without the patched bridge → identical spike rasters (the GABA_B / `nmda_slow` / Route-D changes all used this — `0b3c4b3f…` style stash-vs-edited sha compare); (iii) `git diff --stat sim/` shows only additive hunks. **This is the genuinely bigger, hot-path `sim/` edit the task warned about — land it ONLY after the rhythm-only probe shows it is needed.**

---

## 4. De-risk + the load-bearing anti-cheat

Reuse the EXISTING `research/runners/coincidence_n9_derisk.py` (already has the gates + jitter/ablate/K-sweep + the no-host-teacher audit + the CuPy-regime asserts), with the synchronizer (§3.1) added. 3 seeds (42/43/44), deterministic regime hard-asserted (OU/conductance-noise/global-homeostasis/heterogeneity/STP OFF, `backend=="cupy"` — numpy DISQUALIFIED). The gates are the EXACT inverse of the Step-0 result (the thing it proved impossible without synchronization).

**Load-bearing gates (the asynchrony wall becomes a pass):**
| Gate | Pass criterion | What it proves |
|---|---|---|
| **G_SPARSE** | the place ensemble stays sparse-distinct WITH the rhythm on: ≤5% active, diff-location cos < 0.30 | the volley did NOT densify the code (the property the densify-approach destroyed) |
| **G_VOLLEY** | the active ensemble's per-step max c_i now reaches **≥K** in the volley step (vs Step-0's ≤1) | the rhythm actually synchronizes the sparse ensemble into a coincident packet |
| **G_FIRE (the headline)** | with the synchronizer + Route D ON, the downstream MSN-D1 fires **≥5 Hz** from the sparse-distinct ensemble (vs Step-0's 0.00 spk/step) | rhythm + coincidence detection closes the wall |
| **G_DISTINCT** | downstream firing stays position-specific: near ≫ far drive **≥3×**, downstream diff-cos < 0.30 | it fired by *which* cells coincided, not by going dense |
| **G_MSN** | CA1/place effective drive ≥ ~420 pA → MSN ≥5 Hz | the place code can drive the striatal critic (the N9 read-out the chain is for) |

**Anti-cheats (each MUST behave consistently — this is what makes a pass honest):**
- **THE COINCIDENCE CONTROL (the decisive one, Branco-Häusser; already in the runner as `--jitter-inputs`): JITTER / DE-SYNCHRONIZE the volley → firing COLLAPSES.** Spread the ensemble's spikes back across several steps (same total spikes, same rate, just not coincident — i.e. *defeat the rhythm*) → G_FIRE must FAIL (MSN → ~0 Hz). If firing survives de-synchronization, the mechanism is reading RATE, not the volley → it's a cheat. This is the gate that proves the win is genuinely the synchronized volley.
- **REMOVE THE RHYTHM → COLLAPSE.** Turn the FS-PING pool / the pacing drive OFF (leave everything else, Route D still ON) → G_VOLLEY and G_FIRE must FAIL (back to c_i ≤ 1, MSN silent — reproducing Step-0). Confirms the rhythm is load-bearing, not Route D alone (Route D alone is a no-op here, as Step-0 showed).
- **ABLATE Route D (`enable_coincidence_detection=False`) with the rhythm ON → still NO firing at realistic weight.** A synchronized volley of K *sub-threshold* AMPA inputs without the supralinear plateau must NOT fire the MSN (the linear-summation wall) — confirms BOTH halves (rhythm + Route D) are needed together, and that the volley isn't just "more rate."
- **THE VOLLEY MUST NOT DENSIFY THE CODE (G_SPARSE + diff-cos).** The active set with the rhythm on must be the SAME sparse ≤5% set, just re-timed — assert sparsity and diff-location distinctness are preserved vs the rhythm-off code. (This is the property the densify-to-coincide approach failed; the rhythm passes it by construction, but it must be measured.)
- **K > 1 (not trivially low):** `coincidence_k_threshold` must be > 1 (a single coincident input must NOT fire the plateau, else it's a per-synapse gain). Swept (`--k-sweep`); G_DISTINCT must hold across the K that passes G_FIRE.
- **NO host teacher.** The ONLY `cp_external_input_current` write targets the sensory afferent (`src_sensors`) — and, for the pacing option (3.1b), the sinusoidal `excitability_drive` to the *place pool* (a population-pacing input, NOT a per-location pattern: it is location-INDEPENDENT, so it cannot encode *which* cells fire — it only sets *when* the already-selected cells fire). The downstream MSN fires from the brain's own routed synaptic coincidence, never a host-injected per-location pattern. Grep/audit-assert (the runner already does this). **Brain-based-only note:** the FS-PING option (3.1a) is the fully-neural synchronizer (no host pacing at all) — prefer it as the endpoint; the pacing option is the scaffold (the medial-septum theta generator is, biologically, an *external* pacemaker to the hippocampus, so a paced `excitability_drive` is a defensible environment/septal input, but the *coincidence selection* must remain neural — which it does, since pacing is location-blind).
- **CuPy regime fidelity.** `backend=="cupy"` (numpy DISQUALIFIED per `2026-06-09-N9-cupy-membrane-divergence-ROOT.md`). No per-region homeostasis on the MSN target (it must fire from the coincidence current, not threshold collapse). Deterministic knobs OFF, hard-asserted.

**Graceful-FAIL contract (names the next lever precisely):**
- If G_VOLLEY passes but G_FIRE fails → the volley supplies coincidence but the plateau strength/K is mis-tuned → adjust `coincidence_plateau_strength`/K (Route D operating point), still runner-side.
- If G_VOLLEY fails (the rhythm can't pack the sparse ensemble into one step at ≤5% sparsity — e.g. the gamma window is wider than dt and the cells still scatter within it) → **this is the precise signal that conduction delays (§3.2) are needed** to align the within-window spikes into one arrival step. That is the honest trigger for the `sim/` ring-buffer edit — and it tells the owner the second substrate upgrade is genuinely required, not speculative.
- If G_FIRE passes but G_DISTINCT fails → the volley recruited overlapping sets (the rhythm synchronized too many cells) → tighten the place code sparsity / the FS inhibition; if irreducible, it points at multi-subunit Poirazi-Mel or a compartmental CA1 (the deeper catalog D.04 fallback the Route-D design named).

Either way the de-risk produces a brain-based-only honest result that names the next lever.

---

## 5. Cheap-first build sequence (smallest first)

1. **Step 0 — RUNNER-ONLY (NO `sim/` edit). The gamma-volley probe on the existing bed + the already-landed Route D.**
   Edit `research/runners/coincidence_n9_derisk.py` to add a synchronizer to the `source` pool — start with **(b) the sinusoidal gamma `excitability_drive` pacing** (fastest to get a clean volley), turn Route D ON, and run the **load-bearing pair first**: **G_FIRE** (does the volley fire the MSN?) + the **`--jitter-inputs` anti-cheat** (does de-syncing it collapse firing?). Then the full gate set + G_SPARSE/G_DISTINCT + remove-the-rhythm + ablate-Route-D + K-sweep. **No `sim/` change.** This decisively answers "does a gamma volley make a sparse ensemble fire a coincidence-detecting target" before any engine edit — and Step-0's staged de-risk was explicitly written to anticipate exactly this "externally-imposed synchronous volley."

2. **Step 0b — RUNNER-ONLY. Convert the pacing to FS-PING (the brain-based-only synchronizer).**
   Replace the host pacing with an FS interneuron pool on the place source (PING, mirroring `CORTEX_GAMMA_FS_NETWORK`) so the volley is generated by the network's own inhibition (neurons + synapses only). Re-run the gates. This is the faithful endpoint of the rhythm route, still runner-side.

3. **Step A — `sim/` CHANGE (conduction-delay ring buffer), ONLY IF Step 0/0b show the rhythm alone can't pack the ensemble into one step (G_VOLLEY fails).**
   Land §3.2 (owner byte-review — the bigger, hot-path edit). Re-run with delays composing with the rhythm: delays align the within-gamma-window spikes into one arrival step (Izhikevich-2006), tightening c_i. Byte-identity proof (§3.2) BEFORE any behavioral run. **Flag to owner: this is the genuinely bigger substrate upgrade (a new hot-path state array); build it only when the cheap probe proves it's needed.**

4. **Step B — wire into the C1/N9 CA3→CA1→MSN chain.**
   Tag `ca3 → ca1` (and/or `landmark_sensors → ca3`) `coincidence_detector=True` AND give the CA3/place pool the gamma synchronizer (FS-PING), in the C1 harness (`learned_graded_ca3_derisk.py` / `c1_trisynaptic_ca1_place_code_derisk.py`, reused). Run ALL gates + anti-cheats. This is the real payoff: a distinct-AND-firing place code driving the MSN-D1 — exactly what C1/N9 proved impossible by rate.

5. **Then** re-run the **N9 place-graded critic** on the new CA1 code — a distinct-AND-high-rate place code is the input the value critic needed; the place-grading re-read is finally unblocked.

Each step is independently gated; nothing proceeds to the nav critic until G_FIRE + G_DISTINCT + the JITTER anti-cheat pass.

---

## 6. Honest risk assessment + scope flag

- **What's strongly de-risked:** (a) the **diagnosis** — Step-0's `c_i ≤ 1` across 3 CuPy seeds (even at full convergence) makes the asynchrony wall unambiguous, and the load-bearing subtlety (delays can't synchronize a *stochastic* ensemble; §1.2) is a solid argument from the IF-neuron stochastic-emission fact + the polychrony literature (Izhikevich-2006 selects *learned repeatable* patterns, not free-running ones); (b) the **mechanism** — a theta-gamma volley is THE canonical biological synchronizer of a stochastic ~10 Hz population (Lisman-Idiart N.15, Buzsáki Cycle 9 N.19, O'Keefe-Recce phase precession), and the gamma cycle == STDP window == Route D's coincidence window; (c) the **engine reuse** — gamma is already emergent on this engine (the FS-PING benchmark) and the pacing path already exists (neuromodulator `excitability_drive`), so the first decisive probe needs **ZERO `sim/` edits** and reuses the already-built `coincidence_n9_derisk.py` bed + the landed Route D.
- **The real residual risk (MEDIUM):** whether a gamma volley packs the sparse ensemble's spikes into a SINGLE dt step (not just a ~10–25 ms gamma window spanning several steps). If the active cells fire on the *same gamma cycle* but still scatter across a few dt steps within it, c_i per *step* may still fall short of K — and THAT is precisely the gap conduction delays (§3.2) close (align the within-window spikes into one arrival step). Step-0's G_VOLLEY measures this directly and is the honest trigger for the `sim/` edit. A secondary risk: at <0.2 spk/step the place cells may not entrain to a gamma rhythm cleanly (entrainment usually wants the cell near threshold) — the pacing-drive option (3.1b) mitigates this by depolarizing the active cells to the gamma peak.
- **Scope honesty (the owner asked to be told):** the RECOMMENDED first move is **runner-side, no `sim/` edit** — there is a real chance the second substrate upgrade for N9 is *just* "add a gamma synchronizer in a runner + the already-landed Route D," with NO new engine code. That is the cheapest possible outcome and should be tested first. The conduction-delay ring buffer (§3.2) IS a genuine, bigger, **hot-path** `sim/` upgrade (a new `(D×n)` state array + a bucketed delayed-gather matvec) — the task's "is the ring buffer genuinely needed + how big" question is answered by Step-0's G_VOLLEY gate: build it ONLY if the rhythm alone can't pack the ensemble into one step. So: **rhythm-only is the cheap, likely-sufficient first move; the conduction-delay `sim/` edit is the de-risk-gated, possibly-unnecessary follow-on.**
- **Net honest call:** the synchronizer is the textbook fix (theta-gamma volley), the engine already produces the rhythm, the first probe is zero-`sim/`-edit on an existing bed with the decisive jitter anti-cheat, and the `sim/` ring-buffer is held back behind a sharp falsifiable gate (G_VOLLEY). If the rhythm-only volley fires the MSN and collapses under jitter, N9's place-grading is unblocked with no new engine code; if G_VOLLEY shows the volley spans multiple steps, the conduction-delay ring buffer is the named, scoped `sim/` upgrade — composed with the rhythm, not as a substitute for it.

---

### Engine references (for the byte-review)
- The 1-step delay (SH-3, Route T-delay's target): `sim/bridge.py:1063` (`cp_prev_firing_states = cp.zeros(n, dtype=bool)`) + `:6694` (`cp_prev_firing_states[:] = fired_this_step`); read by every matvec at `:5620`, `:5709`, `:5752`, `:5796`. `max_synaptic_delay_ms = 20.0` `sim/config.py:189`; `max_delay_steps` set at `bridge.py:2081` (both currently UNUSED by propagation — reusable as the ring depth).
- Route D (landed, the detector the volley feeds): `sim/config.py:173-178` (`enable_coincidence_detection` + K/gain/strength/tau); `sim/regions.py:284-294` (`coincidence_detector`); `sim/bridge.py:5724-5768` (the guarded per-neuron coincidence block; `c_count = (binary routed mask).T @ prev_fired`); `sim/kernels.py:253` (`fused_coincidence_plateau`).
- Gamma already emergent (the synchronizer's engine basis): `run_benchmarks.py:613` (`benchmark_gamma_oscillations`, `CORTEX_GAMMA_FS_NETWORK`, FS-PING 30–80 Hz, no special config); catalog N.19 (gamma freq = GABA_A decay; gamma cycle == STDP window).
- The pacing path (the sinusoidal-`excitability_drive` synchronizer, runner-side): `sim/neuromodulators.py` (`target_type="excitability_drive"`, scope group:NAME; `compute_excitability_drive_per_neuron`); the group registration `bridge.neuromodulator_manager.set_group_indices(...)`. Catalog D.18/D.24 recipe (septal theta `excitability_drive`).
- In-substrate firing→gate precedent (a fully-neural rhythm-gate option): `bridge.py:2958` (`couple_gate_to_pool`) / `:2981` (`_apply_gate_couplings`).
- The de-risk bed (reuse; already has jitter/ablate/K-sweep + no-host-teacher + CuPy-regime): `research/runners/coincidence_n9_derisk.py` (header explicitly anticipates "an externally-imposed synchronous volley"). The Step-0 calibration: `research/runners/coincidence_wall_probe.py` + `research/findings/raw/_coincidence_wall_probe.json` (the `c_i ≤ 1`, `no_valid_K_above_1=true` numbers).
- The protected-edit precedents to mirror for §3.2: GABA_B `receptor=` (`regions.py:259-269`, `config.py:183-188`, `bridge.py:5770-5802`); `exc_receptor=nmda_slow` (`regions.py:271-282`, `config.py:160-178`, `bridge.py:5687-5722`); `coincidence_detector` (the just-landed sibling, the closest mirror). The per-synapse-mask growth helper `_ensure_gate_capacity` (`bridge.py:789`, `fill=False`).
- Catalog grounding: B.16 (no axonal conduction delays — Route T-delay, "no `conduction_delay_ms`; all one-step"; 1 ms pallidonigral / 10 ms striatonigral); N.15 (Lisman-Idiart theta-gamma multiplex; "single theta-gamma generator could drive both"; "missing — add via NM framework, sinusoidal `excitability_drive`"); N.19 (ING vs PING; gamma freq = GABA_A decay; gamma cycle = STDP window; "partial — FS gamma confirmed by the benchmark"); D.18 (theta — "missing; sinusoidal `excitability_drive` from a septum NM at 8 Hz"); D.24 (theta-paced sequence compression — phase precession + theta sequences bring distant positions into the STDP window).
- Literature: Lisman & Idiart 1995 (theta-gamma multiplexed buffer, 7±2); Buzsáki Cycle 9 (PING/ING, gamma binding-by-synchrony, gamma==STDP window) + Cycle 11 (theta sequences, phase precession compression); O'Keefe & Recce 1993 (phase precession); Fries (communication-through-coherence, gamma); Izhikevich 2006 "Polychronization" (conduction-delay coincidence selects *learned repeatable* patterns — why delays fine-tune but don't synchronize a stochastic ensemble).
- `git status --short sim/` byte-empty (this pass made ZERO `sim/` edits — verified).
