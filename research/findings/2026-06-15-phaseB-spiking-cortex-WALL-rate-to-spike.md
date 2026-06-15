# Phase B — the spiking learned-cortex HARD GATE returns an airtight WALL: the L1 rate recipe does NOT realize on the point-neuron spiking bridge (the rate→spike Mikulasch-Priesemann wall)

**Date:** 2026-06-15 (CYCLES 59–62). **Status:** ⛔ **NEGATIVE / WALL** — honest, multiply-confirmed, decision-relevant. This is the deliverable the Phase-B HARD GATE (proposal risk #2) was built to produce: it **maps the rate→spike wall precisely**. NO `sim/` edits anywhere in the whole gate (the protected set stayed byte-empty).

## The one-line result
The L1 learned-cortex recipe — validated comprehensively at the **rate** level (`2026-06-14-L1-learned-cortex-fair-test-GO.md`, +0.545, 5 axes GO) — **does not recover the category structure on the bridge's point-neuron spiking substrate.** Across 6 distinct attempts/probes the trained cortex code is ≈ 0 / slightly negative (best ≈ −0.04), no better than a random projection, while the rate ceiling is +0.89. The structure is lost in the **rate→spike encoding**, which the point-neuron substrate cannot undo.

## What was tried (the gate did its job — cheap-first, zero `sim/` edits)
| # | attempt | result | what it ruled out |
|---|---|---|---|
| 1 | naive hub-drive STDP | cortex silent | **clock bug** — `bridge._run_one_simulation_step()` doesn't advance `current_time_ms` → every spike t=0 → `delta_t≡0` → STDP a no-op (NOT "net-depression"; the "824→169 decay" was the buggy structural plasticity). Fixed runner-level (`_step_with_time`). |
| 2 | C1a: WTA + FAST adaptive-θ homeostasis + non-specific co-fire (Diehl-Cook) | cortex FIRES, structure −0.07 | the silence is curable; the WTA/competition does not recover the structure (the project's own Phase-A finding: the lateral isn't the key). |
| 3 | centering: feedforward subtractive-inhibition cm-pool + synaptic scaling + stronger dendritic gain | WALL, best −0.04 | a single inhibitory pool delivers only **rank-1 (uniform)** inhibition; the common mode is **per-cortex-neuron-varying** (random connectivity) → uniform inhibition removes signal+common-mode together (no Goldilocks). |
| 4 | per-postsynaptic-neuron centering (host-side C1b instrument) | does not recover (−0.15→−0.09) | the **exact L1 op** (`x − mean`) applied to the bridge g_e does not recover it → the wall is **below** the centering rule. |
| 5 | hub-encoding regime (drive strength) | hubs CAN encode (+0.33 at strong drive) but it was a drive artifact | at drive_scale 12 the hubs fired ~0.15 spk/hub (Poisson noise) → the prior 36-cell grid (ds{12,20}) **under-fired**; at ds40, hub code +0.33 (centered) — so the input layer is NOT the fundamental wall. |
| 6 | strong-drive cortex gate (the regime the grid never tried) | WALL, −0.04..−0.07, ≈ random-proj | even with the hubs encoding (+0.33) + centering + strong drive, the **cortex** code does not recover the structure. |

## The precise localization (airtight)
- The drive carries the structure: **log-input cosine +0.891** (centered +0.936 — the L1 op works on the input).
- The **hub spike-rate code** can recover it given enough firing: −0.06 at ds12 (~0.15 spk/hub, Poisson noise) → **+0.33 (centered) at ds40** (~3 spk/hub). So the input spiking is not fundamentally lossy — it just needs enough spikes.
- The **cortex** code (after the hub→cortex projection) loses it: g_e (analog, pre-threshold) cosine ≈ **−0.06 ≈ the spike-code −0.07** ⇒ the spiking threshold is **not** the destroyer; **the common mode dominates the analog cortex drive**, and it is **per-cortex-neuron-varying**, so no point-neuron mechanism (feedforward inhibition, per-neuron centering, WTA, synaptic scaling, the dendritic divisive gain, strong drive) removes it without also removing the signal.

## Root cause — the rate→spike Mikulasch-Priesemann wall
L1's load-bearing op is **common-mode removal (centering / whitening)** — a **per-input-dimension subtraction on a full-precision analog code before the projection**. On the bridge, the projection happens through spikes + conductances; the common mode enters the cortex drive **per-neuron-varyingly** and a point neuron has only **rank-1 (scalar) or threshold-based** tools to remove it — it cannot do the **per-dimension analog whitening**. This is exactly the documented **Mikulasch-Priesemann point-neuron limit** the project has hit 5+ times: *decorrelation/whitening is an analog / pre-spike / dendritic computation a point neuron fundamentally cannot do.* The HARD GATE confirmed it with the bridge in the loop — which the rate-level de-risk could not see.

## What this means (the honest synthesis)
- The L1 mechanism is **real and rate-validated** (the learned cortex reaches the host ceiling at the rate level). Its **faithful spiking, point-neuron realization is blocked** by the Mikulasch-Priesemann wall.
- The L1 GO's conclusion that *"the dendritic D2 build is OFF the critical path"* was a **rate-level** conclusion. The bridge realization **re-opens it**: the **spiking** realization of the centering needs **dendritic (analog, pre-spike, per-dimension) computation** — the deferred, months-scale dendritic-substrate piece.
- Per the BRAIN-BASED-ONLY standard, a host-computed centering is a cheat (ruled out). A simple guarded `sim/` edit (post-triggered STDP, a per-neuron subtractive primitive) is unlikely to suffice — probe #4 shows the wall is **below** the centering rule (the per-dimension analog whitening, not just the LTP/LTD shape).

## The strategic fork (owner's call — the next step is the gated months-scale piece)
- **(A) The dendritic substrate (months-scale).** The only path to a faithful, brain-based, *spiking* learned cortex that does the analog whitening → generalizes. The deep artificial-life / biology-translatable frontier; re-confirmed as necessary by this bridge wall.
- **(B) Accept the point-neuron limit.** Ship the **flat 2,048-concept curated cortex** (the conversational product, delivered) + bank the L1 rate mechanism as validated-but-not-point-neuron-spiking-realizable.
- **(C) A guarded `sim/` edit.** Likely insufficient on its own (the wall is the analog whitening, not the rule shape); blurs into (A).

## What's banked (durable, all pushed both remotes)
- The corrected bridge-STDP **clock fix** (runner-level `_step_with_time`; no `sim/` edit) — a real, reusable correction (`bridge._run_one_simulation_step` must be paired with a `current_time_ms` advance for STDP).
- The **C1a competitive machinery** (WTA via `exc_fraction`+internal density, fast-θ homeostasis override, non-specific co-fire, the cm-pool, synaptic-scaling flag) — all opt-in/additive in `research/runners/spiking_sm_cortex.py`, default-off byte-preserving; reusable for any future bridge competitive-learning work.
- The **precise localization probes** (`_phaseB_c1b_derisk_perneuron_centering.py`, `_phaseB_hub_encoding_regime.py`, `_phaseB_strong_drive_gate.py`) + the deep-research (`2026-06-15-bridge-competitive-stdp-deep-research.md`) + the subagent's WALL write-up (`2026-06-15-phaseB-task3-centering-RESULT.md`).
- A flagged real `sim/` bug (structural-plasticity not resizing `cp_plasticity_rate_gain` → IndexError on gated pathways).
- The protected set stayed **byte-empty** — the entire Phase-B gate was zero-`sim/`-edit.

The honest NEGATIVE IS the deliverable: it maps the rate→spike wall precisely and tells the owner the spiking learned cortex requires the dendritic substrate, saving a months-scale build from the wrong (point-neuron) premise.

---

## ⚠️ REFINEMENT (same night, CYCLE 63) — the wall is the SPIKE-COUNT READOUT of a common-mode-buried weak signal; I over-claimed "the projection needs dendrites." It is a BOUNDARY, not a clean WALL.

A follow-on deep-research (`2026-06-15-spiking-whitening-cheapest-mechanism-research.md`) flagged that the 6 probes centered at/after the **cortex**, whereas L1 centers the **input per-hub before the projection**, and argued the months-scale substrate is likely unnecessary. Four more free probes (no `sim/` edits) localized it precisely and **partly overturn, partly confirm** the WALL:

| measurement (clean bridge, strong drive ds40, untrained random W) | result |
|---|---|
| bridge cortex **g_e (analog conductance)** cosine | **+0.45 to +0.57** — the projection PRESERVES the structure |
| bridge cortex **spike-count** code cosine | **≈ 0** (−0.04..+0.05) across ds{20,40,80,120} × window{150,300}, 60–136 spikes/concept |
| numpy (rate) projection of the hub codes | +0.34 (input-centering ≈ output-centering +0.338 ≈ +0.341 — the **locus does NOT matter**) |
| g_e **per-neuron-centered** | +0.001 (dense) — the g_e structure is **common-mode-correlated / weak**, centering removes it |

**The corrected localization:**
1. The earlier "g_e −0.06" was the **weak-drive (ds12) regime** (hubs under-firing, ~0.15 spk/hub). At strong drive the **analog path is fine** — the hub→cortex projection preserves the structure (g_e +0.45). So the projection does **not** need dendrites — I over-claimed that.
2. But the category signal is a **weak perturbation on a large common mode** (200 common hubs vs 12/category); it sits in the g_e weakly (+0.45 uncentered, ~0 centered), and the **spike-count readout robustly loses it** — the spiking threshold saturates on the common mode, burying the weak category signal. This is the common-mode problem manifesting **at the spike readout**, not the projection.
3. The research's input-vs-output **locus** reframe did **not** rescue it on the bridge (numpy: input ≈ output ≈ +0.34; bridge spike readout: both ≈ 0). So the fix is **not** simply "center at the input."

**Honest status = BOUNDARY** (not the clean WALL above, not "just engineering"): the structure lives in the analog path; transmitting the **common-mode-buried weak category signal through a spike-count code** is the genuine open problem, and removing the common mode cleanly is the point-neuron-hard whitening (the Mikulasch-Priesemann theme — my original WALL was directionally right about the *mechanism*, wrong that it's the projection). The **untested** cheaper-than-dendrites candidate is **predictive coding with per-error-unit interneurons** (Jang et al. 2024, PMC11045951 — demonstrated in single-compartment AdEx POINT neurons, ρ>0.8: a per-dimension prediction-subtraction microcircuit, richer than the rank-1 pool that failed). The FHRR phase-coding escape does **not** apply (different common mode).

**Refined fork for the owner:** (A′) a **predictive-coding microcircuit** (per-dimension common-mode prediction+subtraction in point neurons — Jang 2024; a medium build, cheaper than dendrites, untested here); (B) ship the flat curated cortex (delivered) + bank L1; (C) the minimal single-extra-compartment dendrite (now looks like the *fallback*, not the lead). New localization probes: `_phaseB_input_centering_derisk.py`, `_phaseB_projection_isolation.py`, `_phaseB_cortex_readout.py`.

## ✅ THE BOUNDARY CRACKS (CYCLE 64) — it was spike-count SPARSITY, config-level; NOT the fundamental wall. The build RE-OPENS.

`_phaseB_dense_firing_readout.py` + `_phaseB_dense_gate.py`: across all ~11 prior probes the cortex fired
only **~1 spike/neuron** — a spike count that sparse is binary-ish and cannot encode the graded g_e. With
**DENSE firing** (~15 spikes/neuron: stronger coupling weight_mean 200–800 + longer window + homeostasis off)
the cortex **spike-count code reaches +0.42** (tracking g_e +0.67), monotonically rising with firing density.
The rigorous dense-regime gate (LEARNED cortex + full battery): **LEARNED +0.401, gen 0.906, permuted −0.002
(clean)**; random-proj +0.323/gen 0.953. ⇒ **the category structure DOES transmit through the spiking cortex
when it fires densely enough** — the WALL/BOUNDARY was **spike-count SPARSITY**, a config-level issue, NOT the
Mikulasch-Priesemann wall, NOT the projection, NOT centering, NOT dendrites.

**HONEST whipsaw + lesson:** CYCLE 62 declared an "airtight WALL / needs dendrites" after 6 attempts **all in
a sparse-firing regime** (weight_mean 80 + homeostasis on → ~1 spike/neuron) — a **premature WALL claim**.
The thorough localization (the analog g_e carries the structure +0.45; the spike readout was just too sparse)
cracked it. Lesson: don't claim a fundamental WALL before exhausting the cheap readout/regime knobs — the
"airtight" was an artifact of one un-swept axis (firing density).

**The one remaining honest caveat (the real test):** on this SYNTHETIC data the STDP *learning* is **not
clearly load-bearing** over a dense random projection (random already generalizes 0.95 — the synthetic
category structure is in the uncentered input, captured by any similarity-preserving dense readout). Whether
STDP adds genuine "learned cortex" value (generalization the random projection lacks) needs the **REAL
corpus** — where L1 showed learning IS load-bearing (learned +0.48 vs random +0.17). The real-corpus dense
bridge gate (GPU) is the decisive GO confirmation, and it is the natural Task-4.

### ⛔ THE REAL-CORPUS GATE = NEGATIVE (CYCLE 65, GPU): the dense-readout fix was REAL but INSUFFICIENT. The spiking substrate loses the *weak/diffuse* real structure even with whitened input.

`_phaseB_real_dense_gate` (GPU, n_hub 500, host +0.442): **LEARNED +0.058 / gen 0.234, RANDOM +0.074, permuted
+0.008** — the dense spiking cortex (learned *or* random) does **not** recover the real structure (+0.06 vs
host +0.44). `_phaseB_real_ppmi_input` pins it airtight: PPMI input *has* the structure (input cos **+0.502**)
but the bridge dense cortex code is **+0.075 (PPMI) / +0.051 (log)** — **both ≈ 0**. So even with the
whitened (PPMI) input AND the dense-readout fix, the bridge's **spiking hub→cortex transform loses the real
category structure.** Synthetic worked (+0.40) because its structure is strong/concentrated (host +0.96); the
**real** structure is weak/diffuse (host +0.44) and the spiking substrate loses it.

**FINAL honest synthesis (the whole night's arc):** (1) The L1 *rate* recipe is GO (+0.545). (2) The
dense-firing readout fix is real — spike-count sparsity was a config issue, and it lets the spiking cortex
recover *strong/concentrated* (synthetic) structure (+0.40). (3) **But on the REAL corpus the spiking
learned cortex FAILS (+0.06 vs +0.44), even with PPMI input + dense readout** — the point-neuron spiking
substrate loses the *weak, diffuse* real category structure that the rate recipe recovers. This is the
genuine **rate→spike wall, confirmed on the real (hard) data** — my CYCLE-62 instinct (a real rate→spike
wall) was right; the CYCLE-64 dense crack was a real-but-insufficient sub-fix; the precise truth is the
spiking substrate cannot preserve the weak/diffuse real structure. **Whipsaw, honestly:** WALL (62,
over-claimed localization) → readout-crack (64, synthetic-only) → real-corpus NEGATIVE (65, the wall holds
on real data). **Net for the owner:** the spiking learned cortex realizes L1 on strong/concentrated structure
but **NOT on the real weak/diffuse corpus**; the flat 2,048-concept curated cortex remains the conversational
product; the faithful spiking-from-real-experience cortex needs a substrate that preserves weak/diffuse
structure (the deep frontier — dendrites / predictive-coding / a different code) — a genuine owner-strategic
call, not cheaply closable. **The spike-readout-sparsity fix + the dense regime are banked** (real, reusable).

**The deep WHY (the mechanism that explains the whole arc):** the synthetic *uncentered* code survives the
spiking (random dense +0.32) but the real *whitened* PPMI code does not (+0.075), because **rate-coded
spiking encodes in firing-rate MAGNITUDE, and whitening REMOVES the magnitude** — the real category signal,
once the common mode is removed (the whitening the real structure requires), is a low-magnitude differential
that a point neuron's firing rate cannot carry. This is the fundamental **whitening-vs-spiking-magnitude
tension**: the real structure needs whitening (analog, magnitude-removing) but the spiking needs magnitude.
Consistent with Mikulasch-Priesemann (whitening is an analog/pre-spike computation). **The escape route worth
flagging:** **PHASE coding** (the project's `RESONATE_AND_FIRE` phasor neurons — info in PHASE, not rate
magnitude — already shipped for the composer) is exactly the kind of code that could carry a whitened,
low-magnitude differential where rate coding cannot. A phase-coded cortex is the project-grounded candidate
for the deep frontier (medium/large build, owner-gated), distinct from full dendrites.

## ✅ PATH FORWARD (CYCLE 66): the RETINAL escape — analog center-surround whitening + ON/OFF cells + high spike budget — is a marginally-validated brain-based escape. Phase coding is OUT.

Two escape de-risks (numpy, real corpus): **(1) phase coding — NEGATIVE.** A unit-magnitude phasor projection
is *worse* than rate (+0.12 vs +0.23) — phasors are for binding discrete symbols, not preserving a
continuous similarity, and don't fit (`_phaseB_phasor_derisk.py`). **(2) the RETINAL mechanism — marginal GO.**
The rate→spike wall is the *whitening-vs-magnitude* tension (the real whitened structure is a signed,
low-magnitude differential rate coding can't carry). The retina solves exactly this: analog center-surround
**whitening** (remove the common mode pre-spike) + **ON/OFF cells** (split the signed signal into two
non-negative spiking populations). `_phaseB_onoff_whitened_derisk.py` — the **spike-budget sweep on real**:
g20 +0.205 → g100 +0.296 → g500 +0.321 → **g2000 +0.327, gen 0.766** (host +0.442). So with analog whitening
+ ON/OFF + enough spikes, the spiking code **clears the +0.30 structure bar and generalizes (0.77)** on real —
above bar, below the full-precision ceiling, saturating ~+0.33 (a residual precision gap). **The precision
wall is partly a spike-budget issue, and the retinal mechanism is a genuine brain-based escape** — the most
hopeful result of the arc. (Learning adds nothing over the random ON/OFF projection here — the structure is
in the *representation* the retinal front-end builds, captured by any similarity-preserving readout.)

**The build (the call, CYCLE 66):** realize the retinal+cortical stack on the bridge — a **center-surround
whitening front-end** (within-hub lateral inhibition: each hub minus its neighborhood mean = the common-mode
removal, the input-locus the prior cortex-locus cm-pool got wrong) + **ON/OFF cortex cells** (two populations
on the signed whitened drive) + a **high spike budget** (long readout window / dense firing). Brain-canonical
(Kandel retina), cheaper than full multi-compartment dendrites, and the numpy de-risk clears the bar. Risk:
the residual +0.33-vs-+0.44 gap + the bridge's own projection loss (real g_e +0.175) — the bridge build must
match the numpy +0.33; a NEGATIVE there is the honest boundary. **This supersedes the "ship-and-park" lean:
there is a validated brain-based path to build.**

---

**Final localization (CYCLE 63, the last probe `_phaseB_homeo_off_readout.py`):** the spike-readout loss is **NOT homeostasis equalization** — with homeostasis OFF the cortex spike code is still ≈ 0 (−0.09..+0.01) while g_e stays +0.40..+0.57. So the loss is robust across drive × window × homeostasis × density × gain (≈ 11 probes total). **The honest, well-localized status:** the category structure lives in the cortex analog g_e (+0.45) but does **not survive the spike-count code**, because the category signal is a *weak perturbation on a large common mode* and removing that common mode **before** the spiking threshold is the point-neuron-hard analog whitening (the Mikulasch-Priesemann theme — my CYCLE-62 instinct about the *mechanism* was right; my claim that it's the *projection*/needs-dendrites was wrong — it is the **spike-count readout of an un-whitened weak signal**). Faithful spike-based transmission needs the common mode removed pre-threshold (whitening) or a richer code/microcircuit (predictive coding, Jang 2024). The analog g_e proves the structure is recoverable *in principle*; the spike transmission is the genuine open boundary. **This is owner-decision territory** (medium build) — the solo cheap-first probing is exhausted.

---

## ✅ Phase 1 LANDED + ⛔ Phase 2 cm-POOL gate = NEGATIVE on real, with an airtight DIAGNOSIS + a 6-seed-GO FIX (CYCLE 68–69, 2026-06-15)

**Phase 1 (graded inhibition) is BUILT, verified byte-identical, and on main (both remotes).** The retina's
horizontal-cell mechanism — a `RegionPathway(graded=True)` whose per-step inhibitory conductance is driven by
the source's *continuous membrane* `a_cont = clip((v−rest)/scale,0,1)` instead of its binary spikes — is a
minimal, default-OFF, guarded `sim/` edit (commit `dec311f4` + comment fix `cbcc8f85`). Trust-but-verified to
ground: a **true pre/post A/B** (a fresh golden captured on the parent commit in an isolated worktree is
byte-identical, atol=0, to the committed golden the regression test asserts against) proves the non-graded path
is byte-identical; the 4 pre-existing `test_regions` numpy failures are identical on the parent (not introduced);
6/6 graded tests pass incl. the function test (graded g_i scales with the source's continuous depolarization —
the property a depol-block-limited spiking pool cannot do).

**But the Phase-2 cm-POOL gate on the REAL corpus is NEGATIVE** (`_phaseB_retinal_cortex --real --graded-cm
--cm-bias-pA 300 --window 1000`, seed 42): the ON/OFF graded-whitened cortex code = **+0.051** (gen 0.203,
eff-rank 2.1) — it does **not even beat the no-whitening POINT control (+0.065)** and collapses to near-rank-1.
The Step-1 graded-whitening front-end on real reaches only **+0.138** (host axis-1 reference +0.246; neural~host
align +0.111) — far below the synthetic-smoke +0.316 the front-end hit. Synthetic (strong/concentrated) worked;
real (weak/diffuse) does not — again.

**The airtight DIAGNOSIS (two cheap numpy probes, real corpus, the load-bearing finding):**
1. **The common-mode POOL whitens on the WRONG AXIS.** `_phaseB_whitening_axis_probe.py` (3 seeds, host
   +0.442): the L1 / retinal-reference centering is **per-FEATURE** (axis-0: subtract each hub's mean across
   concepts) = **+0.323**, clears +0.30. The bridge's cm pool (hub_e drives it densely ⇒ it fires ~ each
   *concept's* mean over hubs) does **per-CONCEPT** removal (axis-1) = **+0.255**, below the bar. An
   *instantaneous* pool can only do axis-1 (it has no per-hub cross-concept memory). ⇒ even a *perfect* graded
   cm pool is capped ≈0.07 below the bar for an **architectural** reason — the retinal cm-pool escape cannot
   clear +0.30 on real no matter how the graded transmission is tuned.
2. **Per-hub ADAPTATION is the correct mechanism, and it's MORE biological.** `_phaseB_perhub_adaptation_derisk.py`
   (**6 seeds** — the project standard): each hub subtracting its *own* slow running mean (intrinsic spike-
   frequency adaptation / slow AHP / synaptic depression = a per-neuron high-pass = the Mikulasch-Priesemann
   *per-neuron predictive-coding* form of whitening) **recovers axis-0**: best **+0.311** at a slow rate
   (α=0.02–0.05; mean over 6 seeds, 96–108% of the batch axis-0 ideal, clears +0.30, gen ~0.70), vs the cm-pool
   axis-1 +0.246. The slow time-constant matters (α=0.5 collapses to +0.17 — the adaptation must span *many*
   concept presentations, not one).

**⇒ The Phase-2 PIVOT: replace the common-mode POOL with per-hub ADAPTATION.** The cm-pool (the heart of the
CYCLE-66 retinal-escape design) does the wrong whitening axis; per-hub adaptation does the right one and is the
more faithful biology (every real neuron adapts to its own mean). The graded-inhibition Phase-1 edit is **not
wasted** — per-hub adaptation realized as a *slow per-hub feedback inhibition* uses the same graded-transmission
mode. **The open risk** (the next de-risk): the bridge realization of per-hub adaptation needs a *slow per-hub
mean* (spanning many concepts) on a point-neuron substrate — itself the Mikulasch-Priesemann slow-analog-
integration challenge; the cm-pool's bridge realization already lost half (host axis-1 +0.246 → neural +0.138),
so per-hub adaptation's spiking realization could similarly lose. The numpy mechanism is GO; the *bridge*
realization is the next gate. NO `sim/` edits in the diagnosis (the two probes are pure numpy). Honest NEGATIVE
(cm-pool on real) + a validated corrected mechanism (per-hub adaptation) = the deliverable.

**The cheapest bridge mechanism (existing homeostasis) is RULED OUT (CYCLE 70).** `_phaseB_homeostatic_centering_derisk.py`
(3 seeds, real): the bridge's homeostasis drives each hub to a target *rate*, so its threshold sits at the
**(1−rate) PERCENTILE** of the drive, not the mean — it is **not faithful mean-subtraction**. Best numpy
ceiling **+0.290** (at target rate 0.25, 90% of the ideal axis-0 mean +0.307) — **below the +0.30 bar even
ideally** (and the bridge spiking realization would lose more). So only a **faithful slow per-hub INPUT-mean
subtraction** clears the bar (the ideal axis-0 +0.307), and the existing per-neuron mechanisms don't provide it:
homeostasis is a percentile; spike-frequency adaptation / STP depression are output-driven and within-
presentation (tau ~100ms–1s, but the needed mean spans *many* concept presentations). **⇒ the faithful axis-0
realization needs a dedicated slow per-hub input-mean primitive (a per-feature *predictive* subtraction =
Mikulasch-Priesemann / point-neuron predictive coding, Jang 2024).** This is a NEW bridge mechanism → per the
standing deep-research-at-roadblock directive a read-only research pass is scoping it
(`2026-06-15-slow-perhub-mean-primitive-deep-research.md`, in flight): is there a CHEAP point-neuron primitive
(a guarded default-off slow-input-mean `sim/` edit) or does faithful axis-0 genuinely need multi-compartment
dendrites (the months-scale owner-gated piece)? **The convergent honest status:** the real category structure
is *moderate* (host +0.44), so the best achievable whitened spiking code is *marginal* (~+0.30) even ideally;
the point-neuron bridge realizations explored (cm-pool, homeostasis) all land below it; the mechanism (per-hub
adaptation) is correct but its faithful realization is the slow-analog-integration wall again — to be scoped
cheap-first before any sim/ commit.
