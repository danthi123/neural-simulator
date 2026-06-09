# Design — the COINCIDENCE-DETECTION substrate upgrade (let a SPARSE-distinct spiking code fire a downstream cell via clustered/temporal coincidence, not rate)

**Date:** 2026-06-09
**Type:** Deep-research + DESIGN pass. **READ-ONLY — ZERO `sim/` edits made.** This is a *proposal* for the owner to byte-review before anything lands, per the protected-edit discipline (the GABA_B `receptor=` + per-region-NMDA + per-region-homeostasis + graded-lateral + the just-shipped `nmda_slow` `exc_receptor=` precedents).
**Owner directive:** biologize everything, no banking, brain-based-only. The owner chose to **start the substrate upgrade now** ("proper point neurons" / "conduction delays" roadmap step). An honest negative IS the deliverable.
**Solves the blocker decisively mapped by:** `2026-06-09-learned-graded-ca3-derisk-RESULT.md` (the point-neuron RATE-CODING wall — NEGATIVE 0/3 with the `nmda_slow` recurrent fully exercised), `2026-06-09-C1-trisynaptic-ca1-place-code.md` (fire-vs-grade bifurcation, no middle), `2026-06-09-N9-place-graded-critic-stage2-derisk.md` (the single-hop version of the same wall).
**Supersedes the recommendation of:** `2026-06-09-learned-graded-ca3-design.md`. That design's `nmda_slow` recurrent **shipped** (`069d3023`, byte-reviewed: `enable_nmda_recurrent` + `exc_receptor="nmda_slow"`) and was then **tested NEGATIVE** — it amplifies (up to ~70% recurrent contribution at the dense point) but does NOT bridge the rate-coding wall. The wall is upstream of the recurrent: a sparse-distinct presynaptic *population* fires too few spikes/step to drive ANY downstream cell, with arbitrarily strong/dense projections. The fix is therefore not a better recurrent — it is a **coincidence-detecting postsynaptic mechanism**.

---

## 0. TL;DR (the recommendation, up front)

**The diagnosis is now precise and not in dispute (the RESULT's boundary sweep proved it):** on Izhikevich **point neurons** with **single-exponential AMPA** (τ≈5 ms) and a **rate-weighted linear matvec** (`g_e += Wᵀ·prev_fired`), a downstream cell fires only when its presynaptic population delivers enough *summed* conductance — which a SPARSE-distinct code (≤~5% active, each cell <0.2 spk/step) physically cannot do (CA1 = **0.00 spk/step** even at Schaffer weight 120 / density 0.9 / `ca1→msn` up to 500). Pushing the drive until cells DO fire forces the code dense (≥~29%) and **position-blind**. There is **no overlap**. This is a **linear-summation** wall: the soma sums inputs linearly, so "few synchronous inputs" and "many asynchronous inputs" are indistinguishable, and only the latter clears threshold.

**The faithful fix is COINCIDENCE DETECTION — make a handful of SYNCHRONOUS clustered inputs able to trigger a regenerative (supralinear) event that drives the soma, where the same inputs spread in time/space cannot.** Biology does exactly this with the **dendritic NMDA spike** (Major-Larkum-Schiller 2013, "Decade of the Dendritic NMDA Spike"; Poirazi-Mel 2003 two-layer model; Branco-Häusser 2010): **10–50 glutamatergic synapses clustered over ~20–50 µm, activated synchronously, trigger an all-or-none regenerative plateau (40–50 mV, 50–100 ms)** via the NMDA Mg²⁺ negative-slope conductance — *spread-out / asynchronous inputs are severely attenuated and do not fire the cell.* That is the precise inverse of the wall.

**Two candidate substrate routes (both on the SH-2 / SH-3 roadmap):**

| | **Route D — dendritic-coincidence subunit (NMDA spike)** | **Route T — conduction-delay temporal coincidence (polychrony)** |
|---|---|---|
| Mechanism | per-postsynaptic-neuron supralinear term over a *routed* (clustered) afferent: ≥K coincident inputs in one step → regenerative plateau current (Poirazi-Mel subunit / NMDA spike) | per-synapse axonal delays so a sparse ensemble's spikes *arrive in the same step* at the target → summate (Izhikevich 2006) |
| Closes the wall? | **YES directly** — supralinearity is exactly "few synchronous inputs fire the cell". Coincidence vs rate is *intrinsic* to the nonlinearity. | **Partially** — delays create temporal coincidence, but the postsynaptic integration is still LINEAR. It raises the *peak* of the summed transient but a sparse-distinct ensemble (≤5%, <0.2 spk/step) still delivers too few *simultaneous* quanta to clear an MSN rheobase. Delays help most when combined with a supralinear readout. |
| Engine reuse | **HIGH** — mirrors GABA_B / `nmda_slow` *exactly* (a per-synapse routing mask + a guarded per-neuron additive current). Reuses the existing per-neuron matvec + the Jahr-Stevens Mg block kernel. | **MEDIUM** — needs a per-synapse delay buffer (a ring buffer of `prev_firing` history) — a genuinely new state array + a changed propagation path; the catalog flags it "missing" (B.16). Bigger, touches the hot matvec. |
| Diff size | **~120–180 lines**, additive, default-OFF, byte-identical-when-off | **~150–250 lines** incl. a `(max_delay_steps × n)` history buffer; the propagation matvec changes shape; harder byte-identity story (the 1-step path must be the exact default) |
| Tractability winner | **✅ clear winner** | secondary / complementary |

**Recommendation: Route D (dendritic-coincidence subunit) first.** It (a) closes the wall *directly* (supralinearity is the mechanism, not a helper), (b) mirrors the GABA_B/`nmda_slow` precedent line-for-line (lowest byte-review risk, reuses the Mg-block kernel), (c) is the smaller diff, and (d) is the more biologically load-bearing fix for THIS specific failure (Larkum/Poirazi-Mel coincidence detection is *the* canonical answer to "sparse clustered inputs fire the soma"). **Route T (conduction delays) is the right SECOND substrate upgrade** — it is independently on the roadmap (SH-3), it composes with Route D (delays line up the cluster so the NMDA spike triggers), and it's the faithful fix for the *temporal* half — but on its own it leaves the linear-summation wall standing, so it is not the first move.

**Net `sim/` surface for Route D:** one additive per-pathway field (`RegionPathway.coincidence_detector: bool`) + a per-region opt-in flag + a guarded per-neuron supralinear current block, **mirroring the just-shipped `nmda_slow` change almost exactly** (same mask-build site, same guarded-additive-current pattern, same byte-identity story). **Risk: MEDIUM** — the *mechanism* is well-grounded and the *engine pattern* is proven (4 shipped precedents), but the empirical question — "does a sparse-distinct ensemble at <0.2 spk/step deliver ≥K *coincident* clustered inputs per step to trigger the plateau, AND does the plateau stay position-specific" — is exactly what the de-risk gates settle. The boundary sweep gives a sharp, falsifiable target.

**Build smallest-first (§5):** Step 0 is a **tiny isolated probe** ("can ONE downstream cell be fired by a sparse coincident ensemble, and does jittering the inputs collapse it?") on the EXISTING engine using a hand-built clustered projection — to *characterize the wall and the target* before any `sim/` edit. Step A lands the `sim/` coincidence term (byte-review). Step B re-runs the C1/N9 CA3→CA1→MSN gates. Step C (optional, later) adds conduction delays.

---

## 1. Diagnosis — why rate coding fails here, tied to the RESULT numbers

### 1.1 The wall is LINEAR SOMATIC SUMMATION on point neurons (not the recurrent, not NMDA dynamics)

The RESULT's recurrent-ablation anti-cheat is decisive: **zeroing the `ca3→ca3` `nmda_slow` recurrent changes NOTHING at the distinct point** (G1 0.135→0.138, G2 10.2→10.2, G6 0.41→0.41), and at the dense point ablation *improves* G4 (MSN 21→30 Hz). So the `nmda_slow` graded-attractor upgrade — which the prior design correctly built and which *works* as a graded amplifier — is **not** the blocker. The blocker is one stage earlier and is purely about firing a downstream cell from a sparse source.

The RESULT's boundary sweep maps it exactly:

```
 intensity  w   d  | CA3 diff-cos  CA3 sparsity  MSN max  | G1(distinct<0.30)  G4(MSN>=5Hz)
   450     20  0.3 |   0.135          4.7%        0.0 Hz  |       PASS              FAIL   <- distinct, MSN-silent
   500     22  .35 |   0.432         19.3%        0.0 Hz  |       FAIL              FAIL
   550     24  .35 |   0.689         29.0%       14.8 Hz  |       FAIL              PASS   <- MSN fires, position-blind
   600     26  0.4 |   0.942         44.2%       22.2 Hz  |       FAIL              PASS
```

The crossover sits between **4.7% sparsity (distinct, MSN-silent)** and **29% sparsity (MSN fires, position-blind)**, with **no point that is both**. The RESULT states the mechanism in one sentence: *"You cannot fire a downstream cell from a presynaptic population that itself fires at <0.2 spk/step"* — even with a near-fully-dense, very strong Schaffer projection (`ca3→ca1` w120, d0.9) and arbitrarily strong `ca1→msn` (up to 500), CA1 = 0.00 spk/step at the distinct point.

### 1.2 The engine code that IS the wall (exact lines)

`bridge.py:5557` — the entire excitatory synaptic drive is **one linear rate-weighted matvec**:
```python
g_e_increase = (effective_connections_matrix.T @ prev_fired_float) * cfg.propagation_strength
self.cp_conductance_g_e += g_e_increase
```
- `prev_fired_float` is the **previous step's** binary firing (the uniform 1-step delay, SH-3) — so all inputs to a cell are summed as if simultaneous, with NO mechanism to reward *actual* simultaneity vs spread.
- `effective_connections_matrix.T @ prev_fired_float` is a **linear sum of weights** of the presynaptic cells that fired. A sparse ensemble (say 19 cells of 400, each firing ~once per 100 steps) contributes, per step, ≈ `(0.19 expected active cells) × weight` of conductance — a tiny transient that decays at τ≈5 ms (`fused_conductance_decay_and_current`, `kernels.py:208`). The Izhikevich soma then integrates this linearly (`fused_izhikevich2007_dynamics_update`, `kernels.py:32`): `dv = (k(v-vr)(v-vt) - u + I)/C`. There is **no supralinearity, no regenerative dendritic event** — `I` is just the linear sum. So "a handful of synchronous inputs" produces a sub-threshold blip indistinguishable from noise.
- Even the existing global NMDA (`bridge.py:5603`) does NOT fix this: its conductance increment is `g_nmda_increase = g_e_increase * cfg.nmda_ratio` (`:5613`) — i.e. NMDA is driven by the *same linear rate-weighted sum*, scaled. The Mg-block kernel (`fused_nmda_update_and_current`, `kernels.py:228`) computes the voltage-dependent block per *neuron*, but it multiplies a conductance that was filled linearly by rate — so it amplifies an already-firing cell, it does not let a sparse coincident input *ignite* one. (The kernel's own docstring even says NMDA "is critical for coincidence detection in STDP" — but the *routing* never implements clustered-input coincidence; it implements rate-scaled global NMDA. That gap is the opportunity.)

**The catalog confirms this is the known, named gap:**
- **I.16 (membrane τ_m):** *"Larger τ_m → more temporal summation; smaller τ_m → more coincidence detection… FS neurons (low τ) should be coincidence detectors, RS (high τ) integrators."* — the substrate-level handle, but point-neuron coincidence-via-τ is far too weak for the sparse-ensemble regime (it only sharpens the summation window; it does not make it supralinear).
- **I.17 (cable / λ):** *"not-applicable — simulator is single-compartment. No dendrites… Synaptic inputs sum at the soma without distance-dependent attenuation."* — i.e. there are no dendritic subunits to make nonlinear.
- **D.04 (EC-III → distal CA1 apical dendrite):** *"missing — would require multi-compartment CA1 or distinct excitatory pathways with different dendritic-zone effects (currently CA1 single compartment can only sum inputs)."* — the exact CA1 stage in question, flagged as needing dendritic compartments.
- **SH-2 (point neurons)** + **SH-3 (uniform 1-step delay)** in the flowchart spec are precisely the two shortcuts this upgrade targets.

### 1.3 Why coincidence detection is the FAITHFUL fix (the literature)

The biological neuron does NOT sum linearly. The **dendritic NMDA spike** is the canonical mechanism by which a few clustered, synchronous inputs trigger the soma:

- **Major, Larkum & Schiller 2013, "Decade of the Dendritic NMDA Spike"** (Annu Rev Neurosci): *"Synchronous activation of 10–50 neighboring glutamatergic synapses, clustered over ~20–50 µm of dendritic length, triggers a local dendritic regenerative potential"* — **all-or-none, saturating at 40–50 mV, duration 50–100 ms.** The biophysical basis is the **NMDA receptor's negative-slope conductance due to relief of the Mg²⁺ block** (the same Jahr-Stevens block the engine already has). Decisively for our wall: *"Distal synaptic inputs are severely attenuated [when spread out]. By contrast, spatio-temporal clustering of glutamatergic inputs and the ensuing generation of dendritic spikes can dramatically increase the impact of distal synapses on the AP initiation process."* — a few *clustered* inputs fire the cell; the *same* inputs spread out do not. That is the inverse of the rate-coding wall.
- **Poirazi, Brannon & Mel 2003, "Pyramidal Neuron as Two-Layer Neural Network"** (Neuron): a pyramidal cell = a 2-layer net — *"synaptic inputs drive independent sigmoidal subunits corresponding to the cell's thin terminal dendrites; the subunit outputs are then summed within the main trunk and cell body prior to final thresholding."* This is the **reduced model**: per-subunit supralinear nonlinearity → sum at soma → threshold. The minimal version on a point neuron is a **single per-neuron supralinear term over a routed (clustered) afferent**, added to `I` before the spike test.
- **Branco, Clark & Häusser 2010, "Dendritic Discrimination of Temporal Input Sequences"** (Science): the dendritic NMDA nonlinearity is **temporally sensitive** — different temporal *sequences* of the same inputs give distinct dendritic responses, and *"pharmacological inactivation of NMDA receptors abolished"* the effect. This directly motivates the **jitter/desynchronize anti-cheat** (§4): a true coincidence mechanism must collapse when its inputs are de-synchronized; a rate mechanism would not.
- **Frontiers in Comp Neuro 2021, "Nonlinear Dendritic Coincidence Detection for Supervised Learning"** (the cleanest reduced template): a two-compartment rate model where the distal (clustered) term, once it crosses threshold θ_d, supralinearly switches the cell from an intermediate plateau (`y≈α`, α=0.3) to maximal firing (`y≈1`) — concrete params (θ_p0=0.25, θ_d=0, α=0.3). Confirms the minimal functional form: **a thresholded supralinear switch driven by the clustered subunit.**

**So the faithful fix is: give the postsynaptic neuron a coincidence-detecting dendritic subunit (an NMDA-spike / Poirazi-Mel term) over a designated clustered afferent, so ≥K synchronous inputs in a step trigger a regenerative plateau current — exactly what a sparse-distinct CA3/place ensemble needs to fire CA1→MSN while staying sparse-and-distinct.**

---

## 2. The mechanism design (grounded in the literature)

### 2.1 Route D — the dendritic-coincidence subunit (RECOMMENDED FIRST)

**Concept (Poirazi-Mel reduced to a point neuron + one nonlinear subunit; NMDA-spike biophysics):** designate a pathway (e.g. `ca3 → ca1`, or `landmark_sensors → ca3`, or `vs_place_context → striosome_value`) as a **coincidence-detector afferent**. Each postsynaptic neuron forms a *dendritic subunit* over its synapses from that afferent. Per step, compute over ONLY those routed synapses:
- **`c_i` = the number of presynaptic inputs to neuron i that fired this step** (the *coincidence count* — a matvec of the binary routed-connectivity mask with `prev_fired_float`), and/or
- **`s_i` = the summed routed conductance** (the existing weighted matvec restricted to the mask).

Then apply a **supralinear, saturating (NMDA-spike) nonlinearity** and inject the result as an additive plateau current into neuron i's `total_input_current_pA`:

```
plateau_i = g_plateau * sigmoid( gain * (c_i - K_thresh) )         # all-or-none switch (Poirazi-Mel subunit)
          * mg_block(V_i)                                          # NMDA voltage-dependence (Jahr-Stevens, reused)
          * (E_e - V_i)                                            # driving force (conductance, not fixed current)
```
- **`c_i ≥ K_thresh` (a handful of *coincident* clustered inputs) → the sigmoid switches ON → a large regenerative plateau current** (saturating, ≈ the 40–50 mV NMDA-spike, sustained by a slow decay so it lasts ~50–100 ms — reuse the slow-NMDA decay cache). `c_i < K_thresh` → ≈0. This is the all-or-none NMDA spike: **coincidence (many simultaneous clustered inputs), not summed rate, fires it.**
- **`K_thresh`** is the biological "10–50 synchronous synapses" scaled to the probe's fan-in (e.g. K=5–10 for a sparse ensemble where ~19 cells of 400 are the place ensemble and each target sees a fraction). It is a per-pathway/per-region config knob.
- **`mg_block(V_i)`** reuses the existing Jahr-Stevens block — so the subunit is genuinely NMDA-like (self-limiting, voltage-gated), AND so it composes with the cell already being slightly depolarized (the realistic NMDA-spike condition).
- **Why this stays position-specific:** the subunit fires ONLY when ≥K of *that target's specific* routed inputs coincide. A *distinct* place ensemble at location A drives the targets wired to ensemble-A's cells; location B drives different targets. The sparse-distinct code is *preserved* because the threshold is on *which* inputs coincide, not on total rate — there is no pressure to make the code dense (the opposite of the linear-summation regime, which only fired when dense).

**Coincidence-vs-rate is intrinsic:** with `c_i` = count-of-simultaneously-active inputs and a threshold K, a sparse ensemble whose K cells fire *in the same step* triggers the plateau, while the *same* K cells firing in *different* steps each contribute `c_i=1 < K` → no plateau ever. Jittering/desynchronizing the inputs (the anti-cheat) collapses firing — proving it's coincidence, not rate. (Branco-Häusser: NMDA-dependent, sequence-sensitive.)

**Minimal vs richer form (a design choice for the byte-review):**
- **Minimal (recommended): a per-neuron, single-subunit term over one routed afferent.** No new compartment state; `c_i`/`s_i` are matvecs (already the engine's idiom); the plateau is an additive current with a slow-decay accumulator (one new per-neuron conductance array, exactly like `cp_conductance_g_nmda_recurrent`). This is the smallest faithful version and the one I recommend landing first.
- **Richer (later, if needed): multiple subunits per neuron** (true Poirazi-Mel 2-layer, several dendritic branches each with its own K-of-N cluster). This needs per-(neuron×subunit) state and a grouping of synapses into clusters — a larger change, deferred. The minimal single-subunit version already closes the wall (one regenerative event is enough to fire the soma).

**This reuses the existing engine almost entirely:** the routed-synapse mask (mirror `cp_nmda_recurrent_synapse_mask`), the restricted matvec (the GABA_B/`nmda_slow` technique), the Mg-block kernel (`fused_nmda_update_and_current`, verbatim), a slow-decay per-neuron conductance accumulator (mirror `cp_conductance_g_nmda_recurrent`), and a guarded additive current into `total_input_current_pA` (mirror the `nmda_slow` block at `bridge.py:5624-5659`). The ONLY genuinely new piece is the **supralinear sigmoid over the coincidence count `c_i`** — a few element-wise lines in a new `@fuse()` kernel.

### 2.2 Route T — conduction delays / temporal coincidence (the SECOND upgrade, complementary)

**Concept (Izhikevich 2006 polychronization; the MSO Jeffress delay-line coincidence detector, catalog line 1552):** give each synapse (or pathway) an **axonal conduction delay** so a sparse ensemble's spikes, emitted at slightly different times, **arrive at the target in the same step** — temporal coincidence. The engine currently has SH-3: a uniform 1-step delay (`cp_prev_firing_states`, one step of history). Route T replaces that with a per-synapse delay drawn from a per-pathway distribution (1 ms pallidonigral … 10 ms striatonigral, per catalog B.16), implemented as a **ring buffer of the last `max_delay_steps` firing vectors**, indexing each synapse's contribution from the appropriate past step.

**Why it's the SECOND move, not the first:** delays create *temporal* coincidence, but the postsynaptic integration downstream of them is STILL the linear matvec (§1.2). Lining up a sparse-distinct ensemble's spikes into one step raises the *peak* of the summed transient, but ≤5%-active / <0.2-spk/step still means only a *handful* of quanta land per step — below an MSN rheobase unless a **supralinear** readout (Route D) converts that handful into a regenerative event. **Route T's true payoff is in COMBINATION with Route D:** delays guarantee the K coincident inputs arrive in the same step, *and* the dendritic subunit turns them into a plateau. Alone, Route D already works (a sparse ensemble that fires within the τ≈5 ms window — which the 1-step delay already approximates — supplies coincidence); delays make that coincidence *tighter and learnable* (STDP + delays → polychronous groups, Izhikevich 2006). So Route T is the natural follow-on substrate upgrade once Route D closes the wall.

**Engine cost (why it's bigger):** a `(max_delay_steps × n)` ring buffer is a new ~`200×n` state array; the propagation step must gather per-synapse from the right past row (a scatter/gather, not the clean single matvec); save/load + capacity-growth must handle it; and the byte-identity story ("uniform 1-step delay = the exact default") requires the buffer to degenerate to `prev_firing` when all delays = 1. Doable and on the roadmap (the catalog explicitly asks for it, B.16), but a larger, hotter-path change than Route D — hence second.

### 2.3 Hybrid / simplest-faithful option (noted, not recommended as primary)

A *very* small intermediate — **shorten the AMPA τ + add an FS-coincidence read-out** (catalog I.16: low-τ cells are coincidence detectors) — was effectively what the C1/Stage-2 negatives already explored via the FS-WTA `ca3_inh` / de Almeida E%-max selector, and it did NOT close the wall (it *selects* sparse winners; it does not make the readout supralinear, so the sparse winners still can't fire the MSN). So the τ/FS hybrid is insufficient on its own — it's the reason Route D (genuine supralinearity) is needed. **C4 (a sharper FS-WTA critic read-out on the distinct-regime sparse CA1)** — the lever C1/Stage-2 pointed at as the *cheap* alternative — is a legitimate, even cheaper, RUNNER-SIDE first probe; but it does not upgrade the substrate (the owner's chosen direction). If the owner wants the *substrate* upgrade (they do), Route D is the faithful one; C4 is the runner-side fallback if Route D's de-risk is marginal. Both can be tried; they are not mutually exclusive.

---

## 3. The exact byte-level `sim/` surface (Route D — proposal; owner byte-reviews)

All changes **ADDITIVE, default-OFF, byte-identical-when-off**, mirroring the just-shipped `nmda_slow` `exc_receptor=` change line-for-line (same author 2026-06-09, so the diff will be visually adjacent and the review pattern identical). **Separated into runner-side-only vs needs-`sim/`-edit.**

### 3.0 RUNNER-SIDE ONLY (no `sim/` edit) — Step 0

The isolated "can a sparse coincident ensemble fire one downstream cell" probe (§5 Step 0) is built ENTIRELY in a new probe runner (`research/runners/coincidence_substrate_probe.py`), reusing `build_biological_brain_regions` / a hand-built 2-region bridge. It **characterizes the wall on the existing engine** (confirms the RESULT: a hand-built clustered sparse projection → target silent) and defines the exact target the `sim/` term must hit. **No `sim/` change for Step 0.**

### 3.1 `sim/` CHANGE (load-bearing) — the per-pathway dendritic-coincidence subunit — Step A

**Purpose:** a coincidence-routed afferent → per-postsynaptic-neuron supralinear NMDA-spike plateau (§2.1). The fix.

**Mirror (the precedent to copy almost verbatim):** the `nmda_slow` `exc_receptor=` routing, all 2026-06-09:
- `sim/config.py:138-156` (`enable_nmda_recurrent` + params)
- `sim/regions.py:271-282` (`RegionPathway.exc_receptor`)
- `sim/bridge.py:242-251` (the `None`-init conductance + mask arrays)
- `sim/bridge.py:2182-2214` + `:2349-2372` (the per-synapse mask built in `inject_explicit_wiring` from `keyed[6]`)
- `sim/bridge.py:5486-5509` (the matvec-time routed-synapse capture + AMPA suppression — though Route D does NOT suppress AMPA; see (c))
- `sim/bridge.py:5624-5659` (the guarded additive-current block)

**(a) `sim/config.py`** — add (next to `enable_nmda_recurrent`, line 152), default-OFF:
```python
# Per-pathway dendritic COINCIDENCE detection (2026-06-09; Major-Larkum-Schiller
# NMDA spike + Poirazi-Mel two-layer subunit). When True AND a pathway sets
# coincidence_detector=True, each postsynaptic neuron forms a dendritic subunit
# over that pathway's synapses: if >= coincidence_k_threshold of its routed
# inputs fire in the SAME step (a clustered/synchronous volley), a regenerative
# all-or-none plateau current (saturating, Mg2+-self-limiting, slow ~80ms decay)
# is injected -- so a SPARSE-distinct ensemble (each cell <0.2 spk/step) can fire
# the cell by COINCIDENCE where the linear rate-weighted matvec cannot. Default
# False => the new per-neuron coincidence block is unreached and
# total_input_current_pA is byte-identical to today (mirrors enable_nmda_recurrent
# / enable_gabab). See 2026-06-09-coincidence-substrate-upgrade-design.md.
enable_coincidence_detection: bool = False
coincidence_k_threshold: float = 6.0        # # of synchronous clustered inputs to trigger the plateau (biology: 10-50 synapses; scaled to fan-in)
coincidence_gain: float = 2.0               # sigmoid slope of the all-or-none switch
coincidence_plateau_strength: float = 80.0  # peak plateau conductance scale (the regenerative NMDA-spike drive)
coincidence_tau_decay_ms: float = 80.0      # plateau duration (Major-Larkum-Schiller 50-100ms)
coincidence_tau_rise_ms: float = 2.0
```

**(b) `sim/regions.py`** — add to `RegionPathway` (next to `exc_receptor`, line 282), default byte-identical:
```python
# coincidence_detector (2026-06-09): when True, this EXCITATORY pathway is a
# dendritic-coincidence afferent -- each postsynaptic neuron forms an NMDA-spike
# subunit (Poirazi-Mel) over this pathway's synapses, firing a regenerative
# plateau when >= cfg.coincidence_k_threshold of its routed inputs coincide in one
# step. Lets a sparse-distinct ensemble drive the target by coincidence, not rate.
# Requires cfg.enable_coincidence_detection=True; the pathway's synapses are added
# to the per-synapse coincidence-routing mask. The fast-AMPA g_e component is KEPT
# (unlike nmda_slow which replaces it) -- the plateau is ADDITIVE on top, matching
# the NMDA-spike riding on the AMPA EPSP. Default False = byte-identical routing.
# See the exc_receptor/nmda_slow precedent above (this is its coincidence sibling).
coincidence_detector: bool = False
```
and plumb into `_build_pathway`'s returned dict next to `"exc_receptor"` (the line that emits `getattr(pw, "exc_receptor", "ampa")`):
```python
"coincidence_detector": bool(getattr(pw, "coincidence_detector", False)),
```

**(c) `sim/bridge.py`** —
- **Allocate** (next to `cp_nmda_recurrent_synapse_mask`, ~line 251): `self.cp_conductance_g_coincidence = None`, `self.cp_conductance_g_coincidence_rise = None`, `self.cp_coincidence_synapse_mask = None` (bool per-synapse, True for `coincidence_detector` synapses), `self._cached_decay_coincidence = None` / `_rise`. All None by default → block unreached.
- **Build the mask** in `inject_explicit_wiring` from `keyed[6]`-style metadata (extend the `keyed` tuple with `coincidence_detector` exactly as `exc_receptor` was appended 2026-06-09; build `cp_coincidence_synapse_mask` mirroring `cp_nmda_recurrent_synapse_mask` at `:2359-2367`). Guarded by `if cfg.enable_coincidence_detection and any_coincidence`.
- **The guarded per-neuron coincidence block** — a new block AFTER the existing `g_e_increase` matvec (so the routed mask is available), mirroring the `nmda_slow` block (`:5624-5659`), guarded `if getattr(cfg,"enable_coincidence_detection",False) and self.cp_coincidence_synapse_mask is not None:`. Inside:
  1. **Coincidence count `c_i`:** a restricted matvec of the *binary* routed-connectivity (the mask applied to a {0,1} copy of the connection structure) with `prev_fired_float` → `c_i` (shape `n_post`). (Reuse the masked-data technique at `:5497-5509`, but with `data = mask_f` instead of `data×mask_f` so it counts inputs, not sums weights.)
  2. **The plateau increment:** a new `@fuse()` kernel `fused_coincidence_plateau` computing `g_inc = plateau_strength * sigmoid(gain*(c_i - k_thresh))`, accumulate into `cp_conductance_g_coincidence` (+rise), then `I_coincidence = (g_slow - g_rise) * mg_block(V) * (E_e - V)` via the **reused** `fused_nmda_update_and_current` pattern (or a thin dedicated kernel that adds the sigmoid). Add `I_coincidence` to `total_input_current_pA`.
  - Byte-identical when the flag/mask is off (block short-circuits; AMPA matvec unmasked and unchanged — Route D does NOT touch the `g_e` matrix, unlike `nmda_slow`).
- **One new kernel in `sim/kernels.py`** (mirror `fused_nmda_update_and_current`):
```python
@fuse()
def fused_coincidence_plateau(g, g_rise, decay, decay_rise, v, E_e, mg_conc,
                              c_count, k_thresh, gain, plateau_strength):
    """Dendritic-coincidence (NMDA-spike) plateau. A per-neuron supralinear,
    all-or-none switch on the count of SYNCHRONOUS clustered inputs c_count:
    >= k_thresh coincident inputs -> regenerative plateau; fewer -> ~0. Mg2+-
    self-limiting (Jahr-Stevens) so it is genuinely NMDA-like. Reuses the dual-
    exp decay idiom of fused_nmda_update_and_current. (Poirazi-Mel 2003 subunit;
    Major-Larkum-Schiller 2013 NMDA spike.)"""
    g_inc = plateau_strength / (1.0 + cp.exp(-gain * (c_count - k_thresh)))   # sigmoid switch
    g_new = g * decay + g_inc
    g_rise_new = g_rise * decay_rise + g_inc
    g_eff = cp.maximum(g_new - g_rise_new, 0.0)
    mg_block = 1.0 / (1.0 + (mg_conc / 3.57) * cp.exp(-0.062 * v))
    I = g_eff * mg_block * (E_e - v)
    return g_new, g_rise_new, I
```

**Diff size:** ~120–180 lines incl. docstrings (most is the guarded block + the one kernel; the mask/conductance machinery is copy-of-precedent). **Byte-identity story:** with `enable_coincidence_detection=False` (default) AND no pathway setting `coincidence_detector=True`, every new array stays None, the new `if` blocks short-circuit, the `g_e`/`g_nmda`/`g_gabab`/`g_nmda_recurrent` paths are untouched, and the Izhikevich/HH/AdEx step is unreached-unchanged. Prove with: (i) `pytest tests/` green; (ii) a byte-diff harness running an existing g11 nav seed AND a Tier-1 language seed with/without the patched bridge → identical spike rasters (the GABA_B/`nmda_slow`/per-region-NMDA changes all used this); (iii) `git diff --stat sim/` shows only additive hunks adjacent to the `nmda_slow` ones.

### 3.2 `sim/` CHANGE (Route T — conduction delays) — DEFERRED to the second upgrade

Flagged here for completeness, NOT proposed for this pass. Adds `RegionPathway.conduction_delay_ms` (catalog B.16) + a `(max_delay_steps × n)` firing-history ring buffer + a delayed-gather propagation path that degenerates to `prev_firing` when all delays=1 (the byte-identical default). ~150–250 lines, touches the hot matvec. Land only after Route D closes the wall and the owner wants the temporal-coincidence half. (Izhikevich 2006; MSO Jeffress delay-line, catalog line 1552.)

---

## 4. De-risk + anti-cheats (the CuPy gates that confirm a sparse code now fires a downstream cell via COINCIDENCE and stays distinct)

A self-contained CuPy probe (`coincidence_substrate_probe.py` for Step 0; `ca3_coincidence_readout_derisk.py` reusing the C1/N9 harness for Step B), 3 seeds (42/43/44), deterministic regime hard-asserted (OU/conductance-noise/global-homeostasis/heterogeneity/STP OFF, `backend=="cupy"`). **The gates are built to be the EXACT inverse of the RESULT's boundary sweep — the thing it showed impossible.**

**Load-bearing gates (the wall becomes a pass):**
| Gate | Pass criterion | What it proves |
|---|---|---|
| **G_SPARSE** | the driving ensemble is genuinely sparse-distinct: ≤5% active, each active cell <0.2 spk/step, diff-location cos < 0.30 | the *input* is in the regime the RESULT proved cannot fire a cell by rate |
| **G_FIRE (the headline)** | with the coincidence subunit ON, the downstream cell (CA1, then MSN-D1) fires **≥5 Hz** from that sparse-distinct ensemble (vs the RESULT's **0.00 spk/step**) | coincidence detection closes the rate-coding wall |
| **G_DISTINCT** | the downstream firing stays position-specific: near-location ≫ far-location drive **≥3×**, downstream diff-cos < 0.30 | it did NOT have to go dense/position-blind to fire (the exact pair the RESULT proved impossible together) |
| **G_MSN** | CA1's effective drive to the `IZH2007_STRIATAL_MSN_D1` test cell ≥ ~420 pA → MSN ≥5 Hz | the place code can drive the striatal critic (the N9 read-out the whole chain is for) |

**Anti-cheats (each MUST behave consistently — this is what makes a pass honest):**
- **THE COINCIDENCE CONTROL (the decisive one, Branco-Häusser): JITTER / DESYNCHRONIZE the sparse inputs → firing COLLAPSES.** Spread the ensemble's spikes across several steps (same total spikes, same total rate, just not coincident) → G_FIRE must FAIL (downstream returns to ~0 Hz). If firing survives desynchronization, the mechanism is reading RATE, not coincidence → the upgrade is a cheat. This is the gate that proves it is genuinely coincidence detection and not a relabelled gain knob. (Mirror: shuffle the within-ensemble spike *times* but keep counts.)
- **ABLATE the coincidence subunit (`enable_coincidence_detection=False`) → reproduce the RESULT** (downstream 0.00 spk/step from the sparse code). Confirms the new term is load-bearing and the wall is real without it.
- **NO host teacher.** The ONLY `cp_external_input_current` write targets the sensory afferent (`landmark_sensors`); the downstream cells (CA3/CA1/MSN) fire from the brain's own routed synaptic coincidence, NOT a host-injected per-location pattern, NOT a `vs_place_context` Gaussian into the target, NOT a direct (x,y). Grep-assert.
- **K-threshold is real, not trivially low.** `coincidence_k_threshold` must be > 1 (a single input must NOT trigger the plateau — else it's just a gain on every synapse). Sweep K: the plateau should appear only when the *ensemble*'s clustered inputs coincide, and G_DISTINCT must hold across the K that passes G_FIRE.
- **CuPy regime fidelity.** `backend=="cupy"` (numpy DISQUALIFIED per `2026-06-09-N9-cupy-membrane-divergence-ROOT.md`). No per-region homeostasis on CA1/MSN (they must fire from the coincidence current, not threshold collapse). Deterministic knobs OFF, hard-asserted.

**Graceful-FAIL contract:** if G_FIRE passes but G_DISTINCT fails (the plateau fires but only by recruiting a dense/overlapping set), that is a *precise* negative — the coincidence threshold is too low / the routed clusters overlap — and it tells us whether the residual problem is the *subunit grouping* (→ multi-subunit Poirazi-Mel, §2.1 richer form) or genuinely the point-neuron substrate (→ a real compartmental CA1, the deeper fallback). Either way the de-risk produces a brain-based-only honest result that names the next lever.

---

## 5. Recommended cheap-first build sequence (smallest first protected edit + its de-risk)

1. **Step 0 — RUNNER-ONLY (NO `sim/` edit). The isolated "one downstream cell, sparse coincident ensemble" probe + wall characterization.**
   Build `coincidence_substrate_probe.py`: a tiny 2-region bridge — a 400-cell source pool clamped to fire a sparse-distinct ~5% ensemble per "location", a hand-built clustered projection to ONE downstream test cell. **First confirm the wall on the existing engine** (G_SPARSE holds; downstream = 0.00 spk/step — reproducing the RESULT in 30 lines). Then *characterize the target*: how many coincident inputs/step the ensemble actually delivers (this calibrates `coincidence_k_threshold` BEFORE the `sim/` edit). De-risk decides the exact K/gain/strength operating point. **No `sim/` change yet.**

2. **Step A — `sim/` CHANGE (the dendritic-coincidence subunit). The fix.**
   Land §3.1 (owner byte-review — adjacent to the `nmda_slow` hunks, identical pattern). Re-run Step 0's probe with `enable_coincidence_detection=True` + `coincidence_detector=True` on the projection. **Gate G_FIRE + the JITTER anti-cheat first** (the cheapest decisive pair: does a sparse coincident ensemble now fire the cell, and does desynchronizing it collapse firing?). Byte-identity proof (§3.1) BEFORE any behavioral run.

3. **Step B — wire into the C1/N9 CA3→CA1→MSN chain.**
   Tag `ca3 → ca1` (and/or `landmark_sensors → ca3`) `coincidence_detector=True` in the C1 harness (`c1_trisynaptic_ca1_place_code_derisk.py` / `learned_graded_ca3_derisk.py`, reused). Run ALL gates G_SPARSE/G_FIRE/G_DISTINCT/G_MSN + all anti-cheats. This is the real payoff: a distinct-AND-firing place code driving the MSN-D1 — exactly what C1/N9 proved impossible by rate.

4. **Step C (LATER) — conduction delays (Route T), the second substrate upgrade**, only after Route D closes the wall and the owner wants the temporal-coincidence half (§3.2). Compose: delays line up the cluster, the subunit fires the plateau (Izhikevich 2006 polychrony).

5. **Then** re-run the **N9 place-graded critic** on the new CA1 code — a distinct-AND-high-rate place code is the input the value critic needed; the place-grading re-read is finally unblocked.

Each step is independently gated; nothing proceeds to the nav critic until G_FIRE + G_DISTINCT + the JITTER anti-cheat pass.

---

## 6. Honest risk assessment + scope flag

- **What's strongly de-risked:** (a) the **diagnosis** — the RESULT's boundary sweep + recurrent-ablation make it unambiguous that the wall is linear somatic summation of a sparse source, not the recurrent/NMDA dynamics; (b) the **mechanism** — dendritic NMDA-spike coincidence detection is THE canonical biological answer (Major-Larkum-Schiller, Poirazi-Mel, Branco-Häusser), with concrete numbers (10–50 synapses, 20–50 µm, all-or-none 40–50 mV, 50–100 ms); (c) the **engine pattern** — the guarded-additive-current + per-synapse-routing-mask + reused-Mg-block idiom is proven by FOUR shipped precedents (GABA_B, per-region-NMDA, graded-lateral, `nmda_slow`), so the byte-review surface is low-novelty.
- **The real residual risk (MEDIUM):** whether a sparse-distinct ensemble at **<0.2 spk/step** actually delivers **≥K coincident clustered inputs in a single step** to a target. At ~5% active and <0.2 spk/step, the *expected* number of a target's routed inputs firing in any one step is small — the coincidence may be too rare to trigger the plateau reliably unless the projection is *clustered* (many ensemble cells → the same target) AND the ensemble fires *synchronously enough* (within the τ window / the delay-aligned step). Step 0 measures this directly BEFORE the edit. If the natural coincidence rate is too low, the levers are: tighter clustering (denser ensemble→target convergence), Route T delays to align the volley, or a lower K — but K must stay > 1 (anti-cheat). The de-risk gates settle it with a sharp, falsifiable target.
- **Where it could still fail on point neurons specifically:** a *single* per-neuron subunit pools ALL routed inputs — if two distinct ensembles share target cells, their inputs could co-count and merge (lower G_DISTINCT). The minimal single-subunit form bets that distinct ensembles route to *distinct* targets (the sparse-distinct property); if they overlap too much, the honest cause is "needs multiple per-neuron subunits (true Poirazi-Mel 2-layer)" or "needs a compartmental CA1" (catalog D.04) — a larger but well-scoped follow-on, which the graceful-FAIL gate will name precisely.
- **Scope honesty (the owner asked to be told):** this IS the substrate upgrade, and even the MINIMAL faithful version (Route D, single-subunit) is a **real new mechanism** (a supralinear per-neuron term — the first non-linear-summation element in the engine), not a tuning knob. But it is **the smaller of the two routes** and reuses the most machinery, so it is the tractable first move. **Dendritic (Route D) has a clear tractability AND faithfulness advantage over temporal (Route T) for THIS failure** — D closes the linear-summation wall *directly* and reuses the Mg-block kernel + the `nmda_slow` plumbing; T is bigger, touches the hot matvec, and alone leaves the wall standing. Recommend D first, T second.
- **Net honest call:** the mechanism is the textbook fix, the engine precedent is proven, the diff is bounded and default-OFF, and the de-risk has a decisive coincidence-vs-rate control (jitter the inputs → must collapse). The genuine unknown — does a <0.2-spk/step sparse ensemble supply enough per-step coincidence — is exactly what the cheap Step-0 runner-only probe answers BEFORE any `sim/` edit. If it doesn't yield even with clustering + delays, that is a precise, brain-based-only honest negative that maps the point-neuron coincidence floor and motivates a compartmental CA1 (the deeper, catalogued D.04 fallback).

---

### Engine references (for the byte-review)
- The rate-coding wall (the linear matvec): `sim/bridge.py:5530-5558` (`g_e_increase = (effective_connections_matrix.T @ prev_fired_float) * propagation_strength`); the soma integrator `sim/kernels.py:32-45` (`fused_izhikevich2007_dynamics_update`); the AMPA decay `sim/kernels.py:207-215`.
- The EXACT precedent to mirror (`nmda_slow` `exc_receptor=`, 2026-06-09): `sim/config.py:138-156`; `sim/regions.py:271-282`; `sim/bridge.py:242-251` (alloc), `:2182-2214` + `:2349-2372` (mask build in `inject_explicit_wiring`), `:5486-5509` (matvec-time routed capture), `:5624-5659` (guarded additive-current block).
- GABA_B `receptor=` precedent (the inhibitory sibling): `sim/regions.py:259-269`; `sim/config.py:157-164`; `sim/bridge.py:233-241` (alloc) + `:2326-2347` (mask) + `:5661-5693` (guarded block) + `fused_gabab_decay_and_current` `sim/kernels.py:217-226`.
- Per-region opt-in mask precedent (for a per-region coincidence flag if wanted instead of per-pathway): `sim/regions.py:100-112` (`enable_nmda`); `sim/bridge.py:1162-1176` (`_build_per_neuron_nmda_mask`); the per-region-NMDA masked current `:5616-5622`.
- The Mg-block kernel to REUSE: `sim/kernels.py:228-250` (`fused_nmda_update_and_current`).
- The per-synapse mask-capacity-growth helper (for the coincidence mask): `sim/bridge.py:799-822` (`_ensure_gate_capacity`, `fill=False, dtype=cp.bool_`).
- The 1-step delay (Route T's target, SH-3): `sim/bridge.py:5532` (`prev_fired_float`) + `:6585` (`cp_prev_firing_states[:] = fired_this_step`); the (viz-only, NOT conduction) pulse timers `:5960-5970`; `max_synaptic_delay_ms` `sim/config.py:167`, `max_delay_steps` `sim/bridge.py:2050`.
- Per-region AdEx override (a deeper fallback substrate, already present): `BrainRegion.adex_neuron_type` `sim/regions.py:89`; AdEx kernel `sim/kernels.py:183-205`.
- Catalog grounding: I.16 (τ_m coincidence-vs-integration), I.17 (single-compartment, no λ), D.04 (multi-compartment CA1 missing — the exact stage), B.16 (no axonal conduction delays — Route T), line 1552 (MSO Jeffress delay-line coincidence detector).
- Literature: Major/Larkum/Schiller 2013 "Decade of the Dendritic NMDA Spike" (10–50 synapses / 20–50 µm / all-or-none 40–50 mV / 50–100 ms / Mg²⁺ negative-slope coincidence); Poirazi/Brannon/Mel 2003 "Pyramidal Neuron as Two-Layer Neural Network" (per-subunit sigmoid summed at soma); Branco/Clark/Häusser 2010 "Dendritic Discrimination of Temporal Input Sequences" (NMDA-dependent temporal coincidence → the jitter anti-cheat); Izhikevich 2006 "Polychronization" (conduction-delay coincidence, Route T).
- `git status --short sim/` byte-empty (verified — this pass made ZERO `sim/` edits).
