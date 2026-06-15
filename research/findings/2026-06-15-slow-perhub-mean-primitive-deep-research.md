# Deep research — can a SLOW PER-HUB INPUT-MEAN subtraction be realized FAITHFULLY on the point-neuron bridge? (the Phase-2 fork)

**Date:** 2026-06-15. **Type:** read-only deep-research + reference-catalog review (standing directive: research before committing build/GPU). **NO `sim/` edits, NO GPU jobs.** **Deliverable:** the strategic fork verdict for the controller — cheap point-neuron primitive vs. months-scale dendrites — for the one load-bearing op (`x_h − slow_mean(x_h)`) that the L1 spiking learned cortex needs.

---

## 0. The exact op (restated, verified against the numpy probes)

From `_phaseB_perhub_adaptation_derisk.py:49-66` (read in full) and `_phaseB_whitening_axis_probe.py:47-70`, the **validated, 6-seed-GO** mechanism is:

```
m_h  ←  (1−α)·m_h + α·x_h           # per-hub EMA of the hub's OWN INPUT drive x_h, updated AFTER read (causal/lagged)
adapted_h  =  x_h − m_h              # per-FEATURE (axis-0) centering = subtract this hub's slow running mean
code  =  ON/OFF_spike( W @ adapted ) # then the random projection + ON/OFF readout
```
streamed over a **shuffled multi-epoch concept stream** (12 epochs in the probe), reading the converged last-epoch state. Numpy result (real corpus, 6 seeds, host PPMI+SVD ceiling +0.442): **best +0.311 at α=0.02–0.05** (96–108% of the batch axis-0 ideal +0.323; clears the +0.30 bar; gen ~0.70). **α=0.5 collapses to +0.17** — the slow time-constant is load-bearing: the mean must span ~20–50 concept presentations, NOT one. The common-mode POOL does axis-1 (per-concept) removal = +0.255 (below bar) — the wrong axis (`_phaseB_whitening_axis_probe.py:80-84`; matches the findings doc CYCLE-69 diagnosis exactly).

**Three properties of this op that are decisive for the fork:**
1. **PER-NEURON** — each hub subtracts a scalar computed from *its own* signal. No cross-neuron term. (Contrast: the common-mode pool and `graded_lateral` are cross-neuron.)
2. **SINGLE SCALAR** — the state is one number per hub (`m_h`), not a K×K matrix, not a per-dimension vector.
3. **INPUT-DRIVEN** — `m_h` integrates the hub's **drive** `x_h`, NOT its spike output. (This is the load-bearing distinction every existing bridge adaptation mechanism gets *wrong* — they are all output/spike-driven; see §5.)

---

## 1. Diagnosis — is this the Mikulasch-Priesemann analog-whitening WALL, or a SEPARABLE per-neuron predictive op point neurons CAN do?

**VERDICT: it is a SEPARABLE per-neuron predictive operation that a point neuron CAN do. It is NOT the Mikulasch-Priesemann wall.** Confidence: **high.**

The load-bearing reasoning, with the distinction made airtight:

- **What Mikulasch-Priesemann actually proves is hard** is **decorrelation/whitening across neurons** — removing the *correlation structure* between different input dimensions (the off-diagonal of the covariance), which requires a per-dimension, pre-spike, analog operation that mixes information *across* dimensions (a matrix operation in dendrites). Their fix is dendritic error compartments precisely because the error is a *per-dimension difference of a top-down prediction vector against a bottom-up input vector* — inherently multi-dimensional and cross-neuron. (Mikulasch, Rudelt, Wibral & Priesemann 2023, *Trends in Neurosciences* "Where is the error? Hierarchical predictive coding through dendritic error computation", PubMed 36577388; their *Local dendritic balance* PNAS 2021, doi 10.1073/pnas.2021925118, is the spiking-network realization.)

- **The op we need is strictly simpler and lives on the diagonal.** Subtracting each hub's *own* mean is **mean-centering**, not de-correlating. Mathematically it is the **diagonal/DC part** of whitening (removing each feature's first moment), and it is *separable* — it factorizes into N independent per-neuron scalar problems with **zero cross-neuron coupling**. The reason it nonetheless recovers most of the structure here (+0.31 of the +0.44 host) is that on this corpus the destructive common mode is *shared* (200 common hubs vs ~12 per category), so removing each feature's own DC offset already strips the bulk of it; the residual off-diagonal whitening is the +0.44−+0.31 gap. **Centering is the cheap, separable half of whitening; the cross-neuron de-correlation is the expensive half — and we only need the cheap half to clear the bar.**

- **A point neuron CAN compute its own slow input mean.** This is the canonical claim of single-compartment predictive-coding models. **VERIFIED (and a citation correction the controller needs): the paper the prior findings call "Jang et al. 2024, PMC11045951" is actually Lee, Dora, Mejias, Bohte & Pennartz 2024, "Predictive coding with spiking neurons and feedforward gist signaling"** (J. Neurosci., PubMed 38680678, PMC11045951). I fetched it: it implements predictive coding in **single-compartment AdEx point neurons** (no dendrites), with prediction-error computed by **separate positive/negative error populations** wired with opposite E/I arrangements, reaching **ρ > 0.8** RSA correlation. So the *substance* the prior findings invoked (point neurons CAN do predictive prediction-subtraction, no dendrites required, ρ>0.8) is **correct and load-bearing**; only the author name "Jang" is wrong — flag this in any doc that cites it. *(Caveat: that paper does a richer hierarchical +/− error microcircuit; our op is the much simpler degenerate case — a per-neuron self-prediction = subtract own slow mean — which is a strict subset of what they demonstrate point neurons can do. It does not need their two-population error coding; a single per-hub adaptive subtraction suffices because the "prediction" is just the neuron's own DC level.)*

- **The biology of "subtract a slow running mean" is the textbook function of spike-frequency adaptation, AND it is subtractive (not divisive).** VERIFIED from Benda & Herz 2003 / the electroreceptor result (Benda, Longtin & Maler 2005, *J. Neurosci.* 25(9):2312, fetched): adaptation "acts purely subtractively on the input" — a **subtractive shift of the f-I curve** = a high-pass filter that "separates fast signals from slower changes in input." That is *exactly* `x − slow_mean(x)`. Catalog **I.08** (M-current/Kv7) and **I.13** (SK/sAHP) are the channels; the project models adaptation phenomenologically via Izhikevich `u` / AdEx `w` / homeostatic threshold (`feature-catalog.md:3385`).

**So the op is biologically canonical (subtractive adaptation = a per-neuron high-pass that removes the DC/slow-mean), it is the separable diagonal half of whitening (not the cross-neuron de-correlation MP proves hard), and point neurons are explicitly shown to do the predictive-subtraction it generalizes (Lee/Pennartz 2024). The two-axis mismatch between textbook adaptation and our op (next paragraph) is the ONLY real obstacle — and it is an engineering parameterization, not a substrate impossibility.**

**The ONE genuine subtlety (the real risk, not the wall).** Textbook spike-frequency adaptation differs from our op on **two axes**, both verified:
- **(a) Driver:** biological adaptation is **OUTPUT/spike-driven** (the adaptation current is "activated by the generated spikes" — Benda-Herz; SK is Ca-from-spikes; AdEx `w` gets a spike-triggered `+b`). Our op is **INPUT-driven** (integrates `x_h`, the drive, even when the hub doesn't fire). A pure output-driven adapter tracks a *percentile of firing*, not the *input mean* — which is precisely why the bridge's homeostasis (output EMA → threshold) caps at +0.290 and is RULED OUT (findings doc; `kernels.py:296-309`).
- **(b) Timescale:** biological SFA is **fast** (τ ~ 42 ms, cutoff ~23 Hz — Benda-Herz). Our op is **slow** (α 0.02–0.05 ≈ tens of presentations; α=0.5 fails). Real *slow* adaptation exists (sAHP over many spikes; light adaptation over seconds; synaptic depression τ up to seconds) but the project's within-presentation depression τ (~100ms–1s, `fused_stp_decay_recovery`) is still too fast by ~1–2 orders.

Both (a) and (b) are **parameterization/wiring choices on a point substrate**, not analog-pre-spike impossibilities. A per-neuron scalar EMA of a continuous drive signal, with a slow τ, is a one-line state update — the kind of thing AdEx's `w` already is, just (i) fed from the *input* not the voltage-after-spikes, and (ii) given a slow τ. **This is the crux: the op the wall seemed to demand (cross-neuron analog whitening) is NOT what L1 actually needs; what it needs is a per-hub slow input-mean — separable, scalar, and within point-neuron reach.**

---

## 2. Ranked brain-grounded options to realize it on the point-neuron bridge (cheapest-first)

For each: biology + citation · bridge realization · edit class · risk.

### Option A (LEAD) — a slow per-hub INPUT-mean adaptation primitive (a small, guarded, default-off `sim/` state variable)
- **Biology:** intrinsic spike-frequency adaptation as a **subtractive high-pass** (catalog **I.08** M-current; **I.13** sAHP; Benda-Longtin-Maler 2005 J.Neurosci 25:2312 — "purely subtractive", separates transients from slow background). The *input-driven, slow* variant is the predictive-coding "subtract your own expected drive" form (Lee/Pennartz 2024 point-neuron PC, PMC11045951).
- **Bridge realization:** add one per-neuron array `cp_input_mean_ema` (length N, default unused). Each step, for flagged hubs only: `m ← (1−α)·m + α·drive_h`, and subtract `gain·m` from those neurons' input current (`total_input_current_pA`) BEFORE the threshold. `drive_h` = the hub's pre-threshold excitatory drive (`g_e·(E_e−V)` or the injected stimulus current — the *input*, not spikes). α exposed as `cfg.input_mean_adapt_alpha` (slow, ~0.02–0.05 per presentation → convert to per-step via the presentation length), flagged on a region/pathway (e.g. `BrainRegion.input_mean_adapt=True`). **This is the faithful axis-0 op, done per-neuron, input-driven, with a slow τ — exactly the validated numpy mechanism.**
- **Edit class:** **guarded default-off `sim/` primitive.** Byte-identical when the flag is unset (array stays `None`/the block unreached, mirroring the `cp_graded_synapse_mask is None` guard pattern at `bridge.py:5676-5680` and the `cp_graded_lateral_M is None` guard at `:5833`). Smallest possible core edit: one array + one EMA update + one subtraction, all behind a flag.
- **Risk:** **LOW-MEDIUM.** (i) the input-mean is computed from the bridge's *spiking* hub drive (Poisson `g_e`), which is noisier than the numpy `x_h` — the EMA must average enough presentations (the slow α helps); (ii) "presentation length → per-step α" conversion needs care (the mean must span concepts, not steps); (iii) per BRAIN-BASED-ONLY, computing `drive_h` must read the neuron's *own synaptic conductance/injected current* (legitimate substrate state), not a host-precomputed `x_h` (that would be a shortcut — see §4 anti-cheat). The numpy GO de-risks the *mechanism*; the open question is whether the *spiking* input-mean is clean enough — which the §4 probe settles before the edit.

### Option B (config-only fallback, likely insufficient alone) — repurpose AdEx `w` as a slow self-adapting subtractor
- **Biology:** AdEx adaptation current `w` (Brette-Gerstner 2005) IS a subtractive adaptation. `kernels.py:201`: `dw/dt = (a·(V−E_L) − w)/tau_w` — the **subthreshold** term `a·(V−E_L)` is **already input/voltage-driven** (tracks depolarization = a drive proxy), independent of spikes; the spike-triggered `+b` is the output-driven part.
- **Bridge realization:** put hubs on AdEx with **large `a`** (strong subthreshold coupling = `w` tracks the mean drive), **`b≈0`** (kill the spike-driven part so it's input-driven), and **very large `tau_w`** (slow). `w` then ≈ a slow EMA of the hub's depolarization and is subtracted in the membrane equation (`−w` term, `kernels.py:200`) — a built-in per-neuron subtractive high-pass. **No `sim/` edit** (AdEx presets + per-region `adex_*` already exist).
- **Edit class:** **CONFIG-ONLY** (new AdEx preset + per-region assignment).
- **Risk:** **MEDIUM-HIGH that it's insufficient.** `tau_w` is a membrane-equation time constant in ms; reaching a *cross-presentation* slow mean (seconds–tens of seconds of sim time) needs `tau_w` orders larger than any biological value, and `w` is voltage-driven not input-current-driven (voltage saturates at spike threshold → `w` tracks a *percentile-ish* level, the same failure mode as homeostasis). Worth a **zero-edit numpy/bridge smoke** because it's free, but the timescale + voltage-vs-input mismatch make it a long shot. **Most useful as the cheap control that motivates Option A** (if `w` could do it, no `sim/` edit needed; the probe will likely show it can't, justifying A).

### Option C — a slow per-hub feedback-inhibition "shadow" using the EXISTING graded transmission mode
- **Biology:** retinal **horizontal-cell graded feedback** (catalog **E.05** center-surround, "decorrelates output"; Kandel retina) — but here a *1-to-1* graded shadow per hub, not a pool, so it removes the hub's *own* slow level (per-feature), not the neighbourhood mean.
- **Bridge realization:** one "shadow" interneuron per hub, 1-to-1, on a `RegionPathway(graded=True)` (the CYCLE-68 edit, `bridge.py:5775-5827`) so the shadow's *continuous membrane* drives the hub's `g_i`. Give the shadow a slow integrating membrane (large leak τ) so it tracks the hub's slow mean drive; its graded inhibition = `−slow_mean`. **Reuses the shipped graded-transmission `sim/` machinery — no new core edit, only wiring + a slow-membrane preset.**
- **Edit class:** **CONFIG-ONLY** (wiring + preset), given graded transmission already ships.
- **Risk:** **MEDIUM-HIGH.** (i) doubles the hub count (one shadow each) — fine at n_hub 500; (ii) the shadow's "slow membrane mean" is again a τ-on-a-point-neuron-membrane challenge (same timescale wall as B — a passive membrane's slow mean needs an implausibly large τ); (iii) the graded drive is `clip((v−rest)/scale,0,1)` (`bridge.py:1790-1803`) — *saturating/rectifying*, so it tracks a clipped level, not a clean linear mean. A neat reuse, but the slow-mean-on-a-membrane problem makes it likely to lose like the cm-pool did (host axis-1 +0.246 → neural +0.138). **Good as the "no-core-edit" attempt to try BEFORE Option A**, with A as the fallback if the membrane can't hold a slow-enough mean.

### Option D (NOT recommended unless A–C all fail) — full dendritic / two-population predictive-error microcircuit
- **Biology:** Mikulasch-Priesemann dendritic error compartments (Trends Neurosci 2023) OR the Lee/Pennartz 2024 separate +/− error populations.
- **Why deprioritized:** this is the machinery for **cross-neuron** prediction-error (the expensive off-diagonal whitening). Our op is the **diagonal** (per-neuron self-mean) — using a dendritic/two-population microcircuit for it is massive overkill and re-opens the months-scale core-substrate rewrite the whole arc was trying to avoid. Only justified if A–C prove a point neuron genuinely can't hold a *slow input mean* in spikes (which §1 argues it can).
- **Edit class:** **months-scale core `sim/` rewrite (multi-compartment).** **Risk:** HIGH cost; the owner-gated piece.

---

## 3. The fork verdict

**There IS a cheap point-neuron primitive. Recommend Option A (a guarded, default-off, slow per-hub INPUT-mean adaptation `sim/` primitive) as the next edit. Faithful axis-0 does NOT genuinely need dendrites.** Confidence: **high** that the op is point-neuron-realizable; **medium** that the *first* cheap realization clears the bar in spikes on the first try (the spiking-input-mean cleanliness + slow-τ-on-a-point-substrate are the residual risks the §4 de-risk exists to retire).

**Load-bearing evidence:**
1. The op is **separable and per-neuron** (subtract own scalar mean) — it is the **diagonal/DC half of whitening**, NOT the cross-neuron de-correlation the Mikulasch-Priesemann limit is *about*. The MP wall does not apply to a per-neuron self-prediction. *(§1; MP 2023 PubMed 36577388.)*
2. Point neurons are **explicitly demonstrated** to do predictive prediction-subtraction without dendrites at ρ>0.8 — and our op is a *strict simplification* of that (a per-neuron self-mean, not a hierarchical +/− error code). *(Lee/Pennartz 2024, PMC11045951 — verified, and the "Jang" attribution corrected.)*
3. The biology of the op is **textbook subtractive spike-frequency adaptation** = a per-neuron high-pass that removes the slow/DC mean. *(Benda-Longtin-Maler 2005 J.Neurosci 25:2312 — "purely subtractive"; catalog I.08, I.13.)*
4. The numpy mechanism is **already 6-seed GO** at +0.311 (clears +0.30). *(`_phaseB_perhub_adaptation_derisk.py`.)*
5. The realization is a **one-array, one-EMA, one-subtraction** guarded edit — strictly smaller than the graded-transmission edit already shipped and reviewed (CYCLE-68). *(§2 Option A; guard pattern at `bridge.py:5676`, `:5833`.)*

**The only thing the prior arc got *directionally* wrong:** it filed "faithful axis-0 in spikes" under the Mikulasch-Priesemann *cross-neuron whitening* wall and jumped to "needs dendrites (months-scale)." But the corrected op (CYCLE-69's own diagnosis) is **per-feature = per-neuron = the separable diagonal**, which MP does not forbid. The dendritic substrate (Option D) is the wrong tool for a diagonal operation. **The fork resolves to: build the cheap point-neuron primitive (A); reserve dendrites only if A–C empirically fail the slow-input-mean-in-spikes test.**

---

## 4. The cheap-first de-risk to run BEFORE the `sim/` edit (+ anti-cheat controls)

**The decisive question Option A's edit rides on:** *can a per-hub EMA of the hub's own SPIKING drive (noisy Poisson `g_e`), with a slow τ, recover axis-0 (~+0.31) — i.e. is the spiking input-mean clean enough, and does a point-membrane / step-EMA hold a slow-enough mean?* The numpy GO used the *clean* `x_h`; the gap is the spiking input-mean.

**De-risk D0 (numpy, ~free, run first):** extend `_phaseB_perhub_adaptation_derisk.py` to replace the clean `x_h` in the EMA with a **Poisson-sampled spiking estimate** of the hub drive (`poisson(x_h·gain)/gain`, the same noise model already in `poisson_spk`, `:39`), keep the slow streaming EMA, and confirm best-α still clears +0.30. **GO ⇒ the spiking input-mean is clean enough; proceed to a tiny bridge probe / the edit.** NEGATIVE ⇒ the noise floor is the problem (raises the spike budget / slows α), localizing before any core edit.

**De-risk D1 (zero-`sim/`-edit bridge smoke for Option B/C, optional, run if D0 GO):** on a small bridge, put the hubs on the **AdEx large-`a`/`b≈0`/large-`tau_w`** preset (Option B) OR a 1-to-1 **graded shadow** (Option C, reusing the shipped graded mode) — both config-only — and read the cortex code. If either clears ~+0.31, **no `sim/` edit is needed at all.** Expected to under-perform (the slow-τ-on-a-membrane wall, §2) — but free, and a clean NEGATIVE here is the empirical justification for the Option-A primitive.

**Anti-cheat controls (mandatory — BRAIN-BASED-ONLY + the project standards):**
- **Permuted-label control:** shuffle concept→category labels; Pearson(cos, S_true) must collapse to ~0 (the +0.31 must be real structure, not an artifact). The probe already imports `heldout_generalization`; add the permuted run.
- **Beats-point control:** the per-hub-adapted code must beat the **no-adaptation point control** (the findings doc's +0.065 cm-pool-off baseline) AND the cm-pool axis-1 (+0.246). Already in `_phaseB_perhub_adaptation_derisk.py` (`axis1_batch`, `none`).
- **THE host-shortcut to avoid (load-bearing per BRAIN-BASED-ONLY):** the EMA `m_h` must be computed from the **bridge's own neuronal state** — the hub's synaptic conductance `g_e` / injected stimulus current (legitimate substrate input) — and subtracted as a *neuronal current*. Computing `x_h − mean` in host numpy and writing it into the bridge is a **shortcut** (the brain isn't doing the subtraction; the bookkeeping is) and is exactly what the prior cm-pool "neural ≈ host" gap (+0.246 → +0.138) was honestly measuring. The Option-A primitive keeps it on-substrate: `cp_input_mean_ema` is a per-neuron state the bridge updates and the bridge subtracts.
- **Slow-α is load-bearing, assert it:** re-confirm α=0.5 fails (+0.17) and α=0.02–0.05 passes — the gate must verify the *slow* timescale, not just "some adaptation," so a fast/wrong-τ implementation can't pass by accident.
- **6 seeds** (project standard) for any GO claim; the probe already uses 42–47.

**Gate for the Option-A `sim/` edit (after D0 GO):** the bridge per-hub-adapted cortex code recovers axis-0 (≈ numpy +0.31) on the real corpus, beats point + cm-pool + permuted-clean, at a slow α, 6 seeds — AND the edit is byte-identical with the flag off (a true pre/post A/B, as done for the graded edit in CYCLE-68). A NEGATIVE (spiking realization loses the slow mean despite D0) maps the wall one level deeper (it would mean the slow-input-mean-in-spikes itself is the obstacle — *then* Option D / dendrites is back on the table) and is itself the deliverable.

---

## 5. Existing project machinery that is reusable

- **The guarded-`None`-mask edit pattern** (the template for Option A's guard): `cp_graded_synapse_mask is None` ⇒ block unreached, byte-identical (`bridge.py:5676-5684`); `cp_graded_lateral_M is None` ⇒ no-op (`:5833`). Option A's `cp_input_mean_ema` follows this exactly.
- **Graded transmission mode** (`RegionPathway(graded=True)`, shipped CYCLE-68, `bridge.py:5775-5827`; `regions.py:296-313`): drives a target's `g_i` from the source's **continuous membrane** `clip((v−rest)/scale,0,1)` — the ready-made analog routing for Option C's slow per-hub shadow. **No new core edit needed for C.**
- **`graded_lateral` learned decorrelation** (`bridge.py:1789-1846`, `regions.py:188`): a K×K anti-Hebbian `ΔM = lr(⟨aaᵀ⟩−I)−λM` in analog membrane space — this is the *cross-neuron* (off-diagonal) whitening machinery. **Not for the per-hub op** (it's the expensive half we don't need), but it's the on-substrate tool if the residual +0.31→+0.44 gap (the de-correlation half) is ever pursued.
- **AdEx adaptation `w`** (`kernels.py:184-205`): subthreshold-input-driven subtractive adaptation already in-kernel — the basis of Option B (config-only). Per-region AdEx presets already overlay (`DefaultAdExParamsManager`, `sim/enums.py`).
- **The numpy probes** (`_phaseB_perhub_adaptation_derisk.py`, `_phaseB_whitening_axis_probe.py`, `_phaseB_input_centering_derisk.py`) + their helpers (`build_real_corpus`, `ppmi_svd_sim`/`score`, `_cos_sim`/`_pearson_vs_Strue`/`heldout_generalization`): the D0 de-risk extends these directly (add the Poisson-spiking-mean EMA + the permuted control).
- **The ON/OFF retinal bridge** (`_phaseB_retinal_cortex.py`) + the dense-firing regime + `enable_homeostasis` kwarg + the bridge-STDP clock fix (`_step_with_time`): the bridge-side scaffolding for the eventual Option-A bridge gate (a NEGATIVE-banked dense-readout regime that the per-hub-adapted code plugs into).
- **Homeostasis** (`fused_homeostasis_update`, `kernels.py:296`) — for the record, NOT reusable for this op: it EMAs **output** firing and moves the **threshold** to a target *rate* (`feature-catalog.md:3449` "adapts to mean activity"), i.e. percentile-not-mean, input-blind — the exact failure the findings doc RULED OUT. Documented here so it isn't re-attempted.

---

## 6. Flags / things I could NOT fully verify (trust-but-verify for the controller)

- **Citation correction (verified):** the prior findings' **"Jang et al. 2024, PMC11045951"** is **misattributed**. PMC11045951 = **Lee, Dora, Mejias, Bohte & Pennartz 2024**, "Predictive coding with spiking neurons and feedforward gist signaling" (J. Neurosci.; PubMed 38680678). The *claim* (point-neuron AdEx predictive coding, separate +/− error units, ρ>0.8, no dendrites) is correct and verified by fetching the paper; only the author name is wrong. Fix wherever it's cited.
- **Lee/Pennartz 2024 does MORE than our op:** it's a hierarchical +/− error microcircuit, not a bare per-neuron self-mean. I assert (high confidence) our op is a strict simplification — but the paper does not *itself* isolate the "subtract your own slow input mean" degenerate case, so the point-neuron-feasibility argument leans on the *simplification* logic (§1) + the adaptation literature, not a direct demonstration of exactly our op in spikes. The D0 de-risk is what actually closes this.
- **Scholarpedia "Spike frequency adaptation"** (the canonical subtractive-vs-divisive reference) was unreachable (ECONNREFUSED); the subtractive-high-pass fact is instead sourced to the **Benda-Longtin-Maler 2005 J.Neurosci 25(9):2312** paper (fetched, explicit "purely subtractive") + the Benda-Herz 2003 model it builds on. Solid, but if the controller wants the primary methods source, Benda & Herz 2003 *Neural Computation* "A universal model for spike-frequency adaptation" is the canonical cite (not fetched here).
- **The slow-τ feasibility on a point substrate is the genuine open empirical risk** (Options B/C explicitly, and Option A's per-step→per-presentation α conversion). §1 argues a per-neuron *scalar EMA* (Option A) sidesteps the membrane-τ problem (it's an explicit state update, not a passive RC decay), but the bridge gate is what proves it. This is the deliberate residual the de-risk + gate exist to settle; a NEGATIVE there is the deliverable that *would* re-open dendrites.
- **MP "point neuron fundamentally cannot whiten" framing:** the catalog has no single "Mikulasch-Priesemann" entry to cite; the claim is sourced to their two papers (Trends Neurosci 2023 PubMed 36577388; PNAS 2021 doi 10.1073/pnas.2021925118) as surfaced by search, not read in full. The *scope* distinction I draw (cross-neuron decorrelation = hard; per-neuron mean = not what they forbid) is my reading of what their dendritic-error mechanism is *for*; the controller may want to spot-check the PNAS paper's exact claim about what a point neuron cannot do.
