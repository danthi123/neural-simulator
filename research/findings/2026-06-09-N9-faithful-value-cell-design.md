# N9 — the FAITHFUL way to make the MSN-D1 value critic fire + learn V(s) on CuPy (convergent-excitation up-state, post-aliasing-fix design)

**Date:** 2026-06-09
**Type:** read-only deep-research + catalog/Kandel/literature review + code-grounded buildable spec (standing practice: research BEFORE committing build/GPU). NO `sim/` edits, no commits.
**Owner directive (load-bearing):** *"I want everything biologized before we move on — NO banking any cheats."* + BRAIN-BASED-ONLY (host code legit ONLY for environment sensory render + body action; the up-state + value learning must be neural).
**Designs around the SETTLED facts** (post-`36f15b25` aliasing fix; do NOT re-derive): MSN-D1 critic rheobase **≈339 pA** (both backends); the current `vs_place_context→striosome_value` afferent (200 cells, density 0.5, **weight 0.2**) delivers **~5 pA** → critic correctly silent; the CuPy convergent-drive sweep (`59369146`) **fires it: 25×→0, 70×→3, 150×→193 spikes/120 steps ≈ 20 Hz up-state.**
**Supersedes for the build:** the `2026-06-09-navfaithful-afferent-critic-homeostasis-PASS.md` "PASS 3/3" — that was a **numpy aliasing artifact** (`v≡vr` made the critic fire spuriously). The homeostasis path is NOT the recommended mechanism (see §1.3). References, does not restate, the two 2026-06-08 deep-research docs + the N9-ROOT resolution.

---

## 0. TL;DR (the decision this doc forces)

The critic doesn't fire because **one weak afferent (w=0.2 → ~5 pA) cannot lift an MSN over its ~339 pA rheobase** — and that is *correct, faithful* biology, not a bug to patch with a homeostasis threshold-drop. The real MSN value cell fires from **massive convergent cortico/limbic/thalamic excitation → the KIR2 up-state** (catalog B.02: an MSN receives **~10,000 glutamatergic synapses**, E/I **2–5× excitation-dominant** in the up-state; Wilson & Kawaguchi 1996; the up-state is *excitation-driven*, not noise- or threshold-driven). The CuPy sweep already proved the production substrate fires the critic **at ~150× the present drive** (~339 pA in, ~20 Hz out), place-graded by construction (the afferent is a Gaussian place code, so far-from-goal cells stay sub-threshold). **So the fix is purely a faithful re-parameterization of the existing `vs_place_context → striosome_value` projection to deliver up-state-class convergent current — denser, with many realistic per-synapse weights summing past rheobase, NOT one giant synapse — plus the up-state-stabilizing per-region NMDA the sim already supports.** The bootstrap is escaped by making the **convergent drive non-plastic and strong enough to clear rheobase from init** (the cell is in a location-gated up-state on step 1, before any learning), while a **separate, plastic, place-specific component** is what DA-gated three-factor STDP sculpts into the graded V(s). This is runner-side on top of the shipped GABA_B substrate; the only `sim/` touch considered is **opt-in per-region NMDA on the critic** (already-existing machinery — the `--enable-pfc-nmda` mask), and it is *optional* (Option A works without it). The recommended cheap-first de-risk is the CuPy `_n9_critic_current_diag.py` harness re-run at the proposed weights with explicit near≫far place-grading + actor-not-perturbed + LTP + GABA_B-lesion gates. An honest negative remains possible and the de-risk fails gracefully.

---

## 1. Diagnosis — the cleanest faithful mechanism for a firing, learning, location-specific MSN value cell

### 1.1 Why the present afferent can't fire it (and why that's the *right* biology)

The `striosome_value` critic is `IZH2007_STRIATAL_MSN_D1` (`C=50, k=1.0, vr=−80, vt=−25, b=−20`; `sim/enums.py:671`): a **55 mV rest-to-threshold gap** with a KIR2-mimicking hyperpolarized clamp (`b=−20`). That is the B.02 design intent — the MSN is *silent at rest and thresholds only strong, consensual input* (Kandel 6e Ch 38 pp 933–938). Its measured rheobase is **≈339 pA** on both backends. The current afferent (`n_vs_place_context=200`, `vs_place_to_value_density=0.5`, `vs_place_to_value_weight=0.2`; runner `:164–166`) delivers only **~5 pA** of effective excitatory current → the cell never leaves the down-state. **This is not a wall to hack around; it is the cell correctly refusing weak input.** The faithful question is therefore "how does a real MSN value cell get the ~10× rheobase of convergent drive it needs?" — and the answer (B.02 + the in-vivo literature) is **many excitatory synapses firing in spatial+temporal synchrony**, not one strong one and not a lowered threshold.

### 1.2 The faithful mechanism: a convergent-excitation up-state, place-gated

Catalog B.02 quantitative budget (PBR-160 ch 6 Wilson pp 92–95): a single MSN receives **~10,000 asymmetric (glutamatergic) synapses** (cortical + intralaminar thalamic), each individually *weak* (small bouton counts), and the **up-state is reached when cortical+thalamic synchrony lifts the cell over KIR2's voltage barrier** — E/I during the up-state is **2–5× excitation-dominant** (Wilson & Kawaguchi 1996, ch 6 Fig 5). Inhibition is *not* the up-state gate; it shapes dendritic electrotonus and AP timing. The in-vivo literature (this session's confirmation search) sharpens it: MSN action potentials occur *only* during these plateau up-states; **NMDA receptors at corticostriatal synapses sustain the membrane near threshold during up-states**, and **spatially clustered + temporally synchronized excitatory inputs trigger dendritic plateau potentials** ([Pomata 2008 J Neurosci](https://www.jneurosci.org/content/28/50/13384); [Carter & Sabatini; PNAS plateau-potential work](https://www.pnas.org/content/114/36/E7612)).

For a *value-of-location*, this convergent drive must be **place-graded**: strong when the agent is near a rewarding location (up-state → critic fires → V high), weak when far (down-state → silent → V low). The runner's `vs_place_context` is already a dense grid-32 Gaussian place code (σ=`grid_size/8`=4.0; `:3381`) — at the goal ~30–80 cells fire, far from it ~0 fire. **So the place-grading is built-in to the afferent code: the same up-state mechanism that fires the cell at the goal leaves it sub-threshold elsewhere.** The CuPy sweep confirms exactly this two-regime behavior on the production backend (150× → ~20 Hz; far cells contribute ~0). The dense HC/subiculum → ventral-striatum value pathway is the canonical anatomy for this (van der Meer & Redish 2009; Lansink 2009 — hippocampal place cells *lead* ventral-striatal reward cells; ramping/anticipatory value at decision points). Note these are *ventral-striatal value cells*, which fire at appreciably higher rates than dorsal sensorimotor MSNs — so a ~20 Hz up-state is physiologically appropriate for a value cell, not a regime violation.

### 1.3 How the cell escapes the LTP-bootstrap deadlock (the load-bearing design choice)

The deadlock: at low plastic weight the cell can't fire → no post-spike → no STDP eligibility → weight frozen (the arc's repeatedly-confirmed blocker). A real MSN value cell never faces this because **its up-state drive is largely non-plastic, anatomically pre-wired convergent excitation** — the ~10,000 corticostriatal synapses are *already there*, putting the cell in a location-gated up-state independent of value learning; what *learning* does is **adjust the gain/selectivity of the place→value mapping on top of an already-firing cell** (DA-gated three-factor STDP, Schultz 1998; catalog O.03/C.30). The faithful escape is therefore a **two-component afferent**:

1. **A strong, NON-PLASTIC, convergent place drive** (`plastic=False`) sized to clear rheobase from init at the goal → the cell is in a **location-gated up-state on step 1**, firing near rewarding locations *before any learning*. This is the pre-wired corticostriatal up-state drive (B.02). It is place-graded by the Gaussian code, so it is not a constant blob — it fires near, stays silent far.
2. **A PLASTIC, place-specific component** (`plastic=True`, DA-δ-gated, init small) that STDP sculpts into the graded V(s): because the cell *already fires* (component 1), the plastic synapses now have a post-spike to pair with → eligibility forms → the SNc-derived δ converts near-the-goal coincidences to LTP (value-leads-reward). **The deadlock is broken structurally, by construction.**

This is strictly more faithful than the committed-but-numpy-validated **per-region homeostasis** path (lower the MSN threshold so weak input fires it): homeostatic intrinsic plasticity *is* real biology (Desai 1999; Turrigiano), but (a) it fires the cell by *threshold collapse*, not the *convergent-excitation up-state* B.02 actually specifies; (b) it is slow (τ≈5000 steps; config.py:167-171) — the 1800-step nav can't wait for it (the `…-homeostasis-fix HONEST NEGATIVE`); and (c) its only "PASS" was the numpy aliasing artifact. **The convergent-drive escape needs no threshold edit, fires on step 1, and is the textbook B.02 mechanism.** (Optionally, per-region NMDA on the critic — §2 Option B — adds the literature-confirmed up-state-sustaining current, deepening fidelity and widening the firing margin, but is not required for Option A.)

### 1.4 Residual honesty (carry-overs, not re-litigated)
- **The place code is host-rendered** (a Gaussian over `(x,y)` injected as `cp_external_input_current`). Under BRAIN-BASED-ONLY this is **legitimate sensory rendering of the world's state into the neural place input** — identical in kind to the actor's `sensor_place_readout` injection — but it is a **separate perception shortcut** (the dorsal place code itself isn't yet a self-organized spiking place-cell layer). N9's scope is the neural **value learning + subtraction**; the place-code-as-input shortcut is a *different* item to biologize later (self-organized place cells from landmark sensors; `--enable-landmarks`). Flag it, don't conflate it.
- **The GABA_B value subtraction is already neural + validated** (`enable_gabab`, `striosome_value→snc receptor="gaba_b"`, the `critic_snc_window` lead gate) — SHIPPED, byte-identity-verified, NOT in question. This design only makes the *critic that drives it* fire and learn.
- **The SNc δ (the actor's teaching signal) is already neural in deployment** via `--spiking-snc`. N9's open piece is solely the learned-V critic.

---

## 2. Ranked design options

Notation: **fidelity** · **P(works on CuPy)** (given the sweep) · **surface** (runner-side vs protected `sim/` edit) · **brain-based honesty**.

### ★ Option A (RECOMMENDED) — two-component convergent place afferent (non-plastic up-state drive + plastic place-specific learner), runner-side only

- **Mechanism:** split `vs_place_context → striosome_value` into TWO pathways from the SAME dense place-context region:
  - **A1 (up-state, NON-plastic):** dense (`density≈0.8`), many realistic per-synapse weights (`weight_mean≈6.0`, jitter so individual synapses are weak-but-numerous, summing to ~350–450 pA at the goal — *past* the 339 pA rheobase with margin, NOT one giant synapse) → fires the location-gated up-state from init. `plastic=False`.
  - **A2 (value learner, PLASTIC):** sparser (`density≈0.4`), small init (`weight_mean≈0.2`), `plastic=True, plasticity_gate="value_input"`, DA-δ-gated → STDP sculpts the graded V(s) on top of the already-firing cell.
- **What it reuses:** the existing `vs_place_context` region + drive injection (`:5036–5043`, the per-step Gaussian render — unchanged); the existing `vs_place_context→striosome_value` plastic pathway becomes A2; A1 is a second `RegionPathway` (same machinery); the three-factor pipeline (STDP eligibility × SNc-δ); the `critic_warmup` (now it will *work*, because the cell fires); the GABA_B `critic_snc_window` subtraction; per-region homeostasis NOT used (revert `enable_critic_homeostasis` to default-off for the build, or leave wired but unused).
- **`sim/` edit?** **NONE.** Pure `build_bg_brain_regions` parameter/pathway change (§4). The convergent budget is realized by `density × n_presynaptic × per-synapse-weight` summing past rheobase — exactly how every other strong pathway in the runner works (e.g. cortex→D1 weight ~125 across many cells).
- **Brain-based honesty:** the up-state + the value learning are **fully neural** (the cell fires from convergent synaptic current and learns via DA-gated STDP). A1's *non-plasticity* is the faithful pre-wired corticostriatal up-state drive (B.02 — those synapses exist innately, the cell doesn't *learn* to enter the up-state). The only shortcut is the (separate, acceptable) host-rendered place code (§1.4).
- **fidelity High · P(works) High** (the CuPy sweep already fired the cell at this current class) **· surface runner-side · honesty clean.**

### Option B — Option A + opt-in per-region NMDA on the critic (deepest fidelity; small protected `sim/` reuse, NOT new code)

- **Mechanism:** Option A, plus enable the **already-existing per-region NMDA mask** on `striosome_value` (the same mechanism `--enable-pfc-nmda` uses to scope NMDA to the dlPFC/cortex/motor slices). NMDA at corticostriatal synapses is the literature-confirmed current that **sustains the MSN near threshold during up-states** and lets *bursts* of convergent input produce prolonged plateau depolarization ([Pomata 2008](https://www.jneurosci.org/content/28/50/13384)). This widens the firing margin (the down→up transition is more robust to seed/numerics) and is the most biologically complete up-state.
- **What it reuses:** the per-region NMDA mask infrastructure (already shipped + validated for PFC; the `enable_nmda` per-region path). The voltage-dependent Mg²⁺-block NMDA kernel (`fused_nmda_update_and_current`) already exists.
- **`sim/` edit?** **NONE — confirmed.** `BrainRegion.enable_nmda` is already a per-region field (`sim/regions.py:112`) wired through the bridge's per-region `cp_nmda_neuron_mask` (the exact mechanism `--enable-pfc-nmda` uses; runner `:226-227, :591`). Enabling NMDA on the critic is therefore a **runner-side** `BrainRegion(name="striosome_value", ..., enable_nmda=True)` on the critic region (composes with global `cfg.enable_nmda` via the mask, scoped to the critic slice only). No new `sim/` code path. (Verified this session — the per-region NMDA mask accepts an arbitrary region, so no hard-coded PFC/cortex/motor set blocks it.)
- **Brain-based honesty:** identical to A; NMDA is a faithful corticostriatal receptor, fully neural.
- **fidelity Highest · P(works) High (and most robust margin) · surface runner-side-or-tiny-protected-reuse · honesty clean.** Recommend as the **fidelity upgrade once A passes** (don't gate the first de-risk on it).

### Option C — single plastic afferent + per-region homeostasis (the committed path; NOT recommended as the mechanism)

- **Mechanism:** the existing `enable_critic_homeostasis` on `vs_place_context`+`striosome_value` (committed edit `89b8d909`) — lower the MSN/afferent thresholds so the weak (w=0.2) afferent fires it.
- **Why not:** (a) its only PASS was the **numpy aliasing artifact** — it has *never* been shown to fire the critic on CuPy at the nav timescale; (b) homeostasis is **too slow** for 1800 steps (the committed `…-homeostasis-fix HONEST NEGATIVE`); (c) it fires by **threshold collapse**, not the convergent-excitation up-state B.02 specifies (lower fidelity). Homeostatic intrinsic plasticity *is* real and the per-region edit is a legitimate, byte-verified piece of infrastructure to keep — but it is the *wrong primary mechanism* for the MSN up-state. **Keep the edit available; do not rely on it for N9 firing.**
- **fidelity Medium · P(works on CuPy at nav timescale) Low · surface protected-edit-already-landed · honesty OK-but-wrong-mechanism.**

### Option D (honest fallback) — bank the deeply-mapped negative + move to N5
- If Option A's CuPy de-risk shows the convergent up-state can't be both firing AND place-graded AND non-actor-perturbing at the nav scale (e.g. the up-state bleeds into far locations, or the plastic component can't carve selectivity on top of the strong fixed drive), the honest deliverable is: "the neural value subtraction validates Pavlovian + the up-state fires the critic, but a *graded location-value* needs a richer/self-organized place code before it ports." A valid BRAIN-BASED-ONLY finding (maps a real substrate boundary). **Only after A's de-risk runs.**

**Ranking: A > B > C**, with **D** the principled bail-out. (A first because it's runner-side, needs no edit, and the CuPy sweep already proves the firing; B is the fidelity upgrade once A is green; C is kept-but-not-relied-on; D if A genuinely fails on CuPy.)

---

## 3. Recommended cheap-first de-risk — the smallest CuPy test that confirms fire + place-graded + learns + actor-not-perturbed

**This is the load-bearing methodological requirement** — the arc has hit **six** probe-vs-deployment gaps, every one because the de-risk diverged from deployment (the latest: the whole homeostasis "PASS" was a numpy artifact). **This de-risk MUST run on CuPy** (`SIM_BACKEND=cupy`) — numpy is disqualified for this critic (the aliasing class lived exactly in the weak-drive/near-rest MSN regime). It must replicate the deterministic-nav regime (OU/conductance-noise/global-homeostasis/heterogeneity/STP all OFF, the `g11_bg_runner.py:3340-3344` knobs) and test the **actual proposed two-component afferent** at the proposed weights.

### Harness
Reuse `research/findings/raw/_n9_critic_current_diag.py` (it already builds the deployed-vs-isolation critic on either backend and ran the convergent sweep that produced 25×/70×/150×). Extend it to build the **two-component A1+A2 afferent** at the §4 weights and add the gates below. Keep it tiny (CPU-sized region counts) but **run it `SIM_BACKEND=cupy`** (this is the whole point). Serial; do not disturb the webapp GPU PID.

### Gates (the build proceeds only if ALL pass at ≥3 seeds on CuPy)
1. **(FIRE)** With the A1 non-plastic convergent drive at init (no learning yet), the critic fires at the **goal** at a useful rate (≥ ~5 Hz; the sweep's 150× gave ~20 Hz) on **CuPy**. *Directly refutes the silent-on-CuPy result that the weak afferent gave.*
2. **(PLACE-GRADED)** Same A1 drive: critic firing at the **goal (near)** ≫ at a **far** location (ratio ≥ ~3×, and far ≈ 0 Hz). This is the load-bearing "it's a value signal V(s), not a constant blob." Use ≥2 far locations; assert the dense place code's NEAR/FAR ensembles are distinct (Jaccard < 0.5 — anti-cheat (a), carried).
3. **(LEARNS V — LTP not LTD, bootstraps from realistic init)** Over the value-leads-reward warm-up (the existing `_run_critic_warmup` protocol: ITI floor → clear eligibility → place-drive + SNc reward burst), the **A2 plastic** `vs_place_context→striosome_value` near-ensemble weight **grows from its w=0.2 init AND grows more than the held-out far ensemble** (w_near/w_far ≥ ~2×). Since A1 makes the cell fire from step 1, A2 has a post-spike to pair with → the bootstrap is broken *on CuPy*. *This is the gate the whole arc has been unable to satisfy on the production backend; it is the decisive one.*
4. **(ACTOR-NOT-PERTURBED)** The actor's `sensor_place_readout → cortex_X` firing with the critic present is within ±10% of a critic-absent twin. By construction A1/A2 feed ONLY `striosome_value` (no edge to actor cortex), so this should hold — but assert it (this is the Layer-3 collateral that degraded nav 2.16→3.24 when the critic was fired by over-driving the *shared* afferent).
5. **(VALUE IS NEURAL + GABA_B-carried — anti-cheat lesion)** After training, **zero the GABA_B synapse mask** (`_lesion_gabab_mask`): the state-specific SNc gap must **vanish** (SNc bursts to reward at both near and far). Proves the r−V subtraction is carried by the GABA_B conductance, not host arithmetic. Keep `current_reward_signal=0.0` throughout the subtraction (no host reward reaches the SNc).

### Anti-cheat controls (carry all; the CuPy + grading ones are the new load-bearers)
- **(a) Population code, not a coordinate:** the afferent is the dense Gaussian place ensemble; different locations → different ensembles (Jaccard < 0.5). Rendering `(x,y)→place current` is legitimate sensory rendering (BRAIN-BASED-ONLY), identical to the actor place injection.
- **(b) Place-shuffled control (NEW, decisive against "fired a blob"):** train with the **place-code labels permuted** (the near drive paired with reward but the *ensemble identity* shuffled relative to position) → V must NOT come out place-graded (gate 2 must FAIL under the shuffle). This separates "learned a value-of-*location*" from "learned to fire on any strong drive." *This is the anti-cheat the homeostasis path never had — it directly tests the place-specificity of V.*
- **(c) GABA_B lesion → gap vanishes** (gate 5) + **GABA_A-direct A/B** (`receptor="gaba_a"` must FAIL the gap — the depolarized-SNc wall) + **host-EMA is place-blind** (a global reward-EMA V is identical near vs far → can't produce the gap; the neural place-graded gap is the discriminator).
- **(d) CuPy regime fidelity (NEW, load-bearing):** assert backend==cupy AND OU/conductance-noise/global-homeostasis OFF. If a future run silently falls back to numpy, the gate is VOID (the numpy MSN is the artifact). Hard-fail otherwise.
- **(e) No-threshold-collapse provenance:** assert the critic fires with **global homeostasis OFF and no per-region homeostasis mask on the critic** (Option A fires by convergent current, not threshold drop). If the cell only fires with the homeostasis mask on, that's Option C, not A — flag it.

### Graceful-FAIL contract
If, on CuPy with the two-component convergent afferent, gate 1 (fire) or gate 3 (learn) fails, or gate 2 holds but the **shuffle control (b) also produces grading** (i.e. it fired a blob, not a value), the verdict is **FAIL** → escalate to Option B (add per-region NMDA for a deeper/robuster up-state) and re-de-risk; if B also fails → Option D (bank the negative; the graded-location-value needs a richer place code). The de-risk **must not** rescue a failure by re-enabling OU, lowering the threshold (homeostasis), over-driving the actor's shared pathway, or running numpy.

### Decision rule
PASS (≥3 CuPy seeds, gates 1–5 + anti-cheats) → proceed to the nav 6-seed A/B (flagship A+E+G v2.5 + `--spiking-snc --enable-neural-critic` with the two-component afferent; acceptance = summed reward ≥ Stage-A host scaffold; an honest nav regression is still a deliverable). FAIL → Option B then D as above.

---

## 4. The concrete buildable spec (Option A) — exact `build_bg_brain_regions` changes

All changes are **runner-side** in `research/runners/g11_bg_runner.py`. No `sim/` edit for Option A. Realistic per-synapse weights (many weak synapses summing past rheobase), NOT one giant synapse.

### 4.1 Region (UNCHANGED dense place-context afferent; revert homeostasis to off for the build)
`vs_place_context` stays the dense grid-32 place code (`n_vs_place_context=200`, `IZH2007_RS_CORTICAL_PYRAMIDAL`, drive-injected each step at `:5036–5043`, σ=`grid_size/8`=4.0, `vs_place_drive_max_pA=800.0`). **Set `enable_critic_homeostasis=False`** for the Option-A build (Option A fires by convergent current; the homeostasis edit stays in the codebase, unused — anti-cheat (e)). `striosome_value` stays the fully-GABAergic MSN-D1 critic (`n_striosome_value=80`, `exc_fraction=0.0`, `internal_density=0.0`, `syn_reversal_potential_i_override=-60.0`), `enable_homeostasis=False`.

### 4.2 The TWO afferent pathways (replace the single `vs_place_context→striosome_value` pathway at `:1457–1462`)

```python
# A1 — UP-STATE DRIVE: dense, NON-plastic convergent excitation (the B.02 pre-wired
#      corticostriatal up-state drive). Many weak per-synapse weights summing PAST the
#      ~339 pA rheobase at the goal (NOT one giant synapse). Fires the location-gated
#      up-state from INIT (breaks the bootstrap structurally). Place-graded by the
#      Gaussian code: far-from-goal presynaptic cells are silent, so their summed
#      drive stays sub-threshold (down-state).
pathways.append(RegionPathway(
    from_region="vs_place_context", to_region="striosome_value",
    density=0.8,                # dense convergence (vs 0.5) — ~0.8*200*0.6 active-at-goal ≈ 96 inputs
    weight_mean=6.0,            # weak-per-synapse; 96 inputs * ~6 * decay ≈ 350-450 pA at goal (> 339 rheobase, margin)
    weight_jitter=0.5,          # realistic heterogeneity (some synapses weaker/stronger)
    plastic=False,              # the pre-wired up-state drive is NOT learned (escapes the deadlock)
))
# A2 — VALUE LEARNER: sparser, PLASTIC, DA-delta-gated. STDP sculpts the graded V(s)
#      on top of the already-firing cell (now there IS a post-spike to pair with).
pathways.append(RegionPathway(
    from_region="vs_place_context", to_region="striosome_value",
    density=0.4,                # the learnable place-specific component
    weight_mean=0.2,            # small init; STDP grows the near-ensemble (value-leads-reward)
    weight_jitter=0.1,
    plastic=True, plasticity_gate="value_input",
))
```

**Sizing rationale (must be tuned in the de-risk, these are the starting point):** the CuPy sweep established the firing curve in terms of `CRITIC_MARGIN_MULT` (25×→0, 70×→3, 150×→~20 Hz) where 1× = the present ~5 pA. 150× ≈ ~750 pA gave ~20 Hz; the goal is ~5–20 Hz, so target ~350–500 pA of *summed* A1 current at the goal — i.e. ~70–100× the present drive. With ~96 active presynaptic cells at the goal (density 0.8 × 200 × the ~0.6 goal-active fraction), per-synapse `weight_mean≈6.0` (× the AMPA decay factor) lands in that band. **The de-risk's gate-1 firing-rate sweep tunes `weight_mean` to put the goal up-state at ~10–20 Hz and the far location at ~0 Hz (gate 2).** Keep per-synapse weights small (≤ ~8) and reach the budget via *count* (density), per B.02's "many weak synapses" — do NOT use a single high-weight synapse (that would be a non-faithful giant-synapse shortcut).

### 4.3 critic → SNc (UNCHANGED — the shipped GABA_B subtraction)
`striosome_value → snc`, `receptor="gaba_b"`, `transmission_gate="critic_snc_window"`, `plastic=False` (runner `:1470–1477`) — no change. The critic now *fires*, so this route carries a real learned V.

### 4.4 Warm-up (UNCHANGED protocol, now functional)
`_run_critic_warmup` (`:4292`) already runs the value-leads-reward protocol (ITI floor → clear eligibility → goal place-drive + SNc reward burst → DA-gated LTP). It produced **zero** critic spikes before *only because the cell couldn't fire*; with A1 the cell fires from init, so the warm-up now seeds A2's LTP. Keep `--critic-warmup-trials 20`. (The DA-threshold calibration inside it is unchanged.)

### 4.5 Option B add-on (fidelity upgrade, after A passes — NO `sim/` edit, confirmed)
Set `enable_nmda=True` on the `striosome_value` `BrainRegion` (per-region field, `sim/regions.py:112`, wired through `cp_nmda_neuron_mask` exactly as `--enable-pfc-nmda` does; requires global `cfg.enable_nmda=True`, which the flagship NMDA configs already set — and the mask scopes it to the critic slice only, so it does NOT turn on NMDA elsewhere). Runner-side, no protected edit. NMDA sustains the up-state near threshold ([Pomata 2008](https://www.jneurosci.org/content/28/50/13384)) → robuster down→up transition, wider firing margin.

### 4.6 What does NOT change (reuse, don't rebuild)
The GABA_B/GIRK `sim/` substrate; the three-factor STDP×δ pipeline; the dopamine `from_region_firing_signed` rule; the `critic_snc_window` gate + reward-window open logic (`:5627–5628`); the place-drive injection (`:5036–5043`); the per-region homeostasis edit (kept, unused by Option A); the per-region NMDA mask (reused by Option B). Net new for Option A = **one extra `RegionPathway` + two parameter changes**; for Option B = **+ one per-region NMDA flag (runner-side or a tiny mask extension)**.

---

## 5. Honest bottom line

The MSN-D1 critic is silent because the present afferent (w=0.2 → ~5 pA) is **~70× too weak to clear the cell's faithful ~339 pA rheobase** — and the post-aliasing-fix CuPy sweep proves the production substrate *does* fire the cell (~20 Hz up-state) once given convergent-excitation-class drive. The faithful fix is therefore **not** a threshold hack (homeostasis — the wrong mechanism, and its only "PASS" was a numpy artifact) but a **convergent-excitation up-state**: a dense, non-plastic, place-graded place drive that puts the cell in a location-gated up-state from init (breaking the LTP-bootstrap deadlock *structurally*), plus a plastic, place-specific component that DA-gated STDP sculpts into V(s) — exactly the B.02 / Wilson 1996 / corticostriatal-NMDA-plateau biology, and exactly the anatomy of dense HC→ventral-striatum value cells. It is **runner-side only** (Option A, no `sim/` edit), with an optional per-region-NMDA fidelity upgrade (Option B, reusing the shipped `--enable-pfc-nmda` mask). The decisive de-risk is the existing `_n9_critic_current_diag.py` re-run **on CuPy** at the proposed weights, gated on fire + **place-graded (near≫far)** + **LTP-from-realistic-init** + actor-not-perturbed + GABA_B-lesion, with a **place-shuffle anti-cheat** that the homeostasis path never had and a **CuPy-regime-fidelity** assertion that closes the numpy-artifact gap. The residual shortcut is the (separate, acceptable) host-rendered place code; the up-state and the value learning are fully neural. If the CuPy de-risk passes, Option A → nav 6-seed A/B; if not, Option B (NMDA) then the honest negative (Option D). Recommend the controller present Option A and run the CuPy de-risk before any nav build.

---

### Sources
- Catalog `E:\Documents\Projects\sim-catalog\references\feature-catalog.md`: **B.02** (MSN bistability, ~10,000 glutamatergic synapses, E/I 2–5× excitation-dominant in the up-state, KIR2 up/down mechanism, GABA_A E≈−60 mV; Wilson & Kawaguchi 1996; Kandel 6e Ch 38 pp 933–938); **B.07** (patch/striosome ↔ ventral-midbrain DA = limbic); **C.28** (TD-error = phasic DA, the missing V(s) bootstrap); **C.30** (actor-critic: striosome-patch = critic V(s), Schultz98 Fig 9C, Houk-Adams-Barto 1995, Barto 1995); **C.31** (bootstrapping vs Monte Carlo).
- Existing N9 research (referenced, not restated): `2026-06-08-striatal-value-critic-firing-research.md`, `2026-06-08-nav-neural-value-critic-redesign-research.md`, `2026-06-09-N9-cupy-membrane-divergence-ROOT.md` (✅ RESOLVED banner = the aliasing fix), `2026-06-09-N9-forensic-substrate-is-NOT-the-wall…`, `2026-06-09-navfaithful-afferent-critic-homeostasis-PASS.md` (numpy artifact — re-read critically).
- Diagnostic artifacts: `research/findings/raw/_n9_critic_current_diag.py` (the CuPy harness + convergent sweep), `_n9_isolated_msnd1_step.py` (rheobase scan), `_n9_fix_verify.py` (aliasing fix verification). Commit `59369146` (CuPy sweep: 25×→0/70×→3/150×→193 spikes), `36f15b25` (aliasing fix).
- Literature (this session's confirmation + the prior docs' citations): MSN UP-state NMDA plateau / convergent corticostriatal input — [Pomata et al. 2008, NMDA Receptor Gating of Information Flow through the Striatum In Vivo, J Neurosci](https://www.jneurosci.org/content/28/50/13384); [Cell-type-specific inhibition of the dendritic plateau potential in striatal SPNs, PNAS](https://www.pnas.org/content/114/36/E7612); [Action Potential Timing Determines Dendritic Calcium during Striatal Up-States, J Neurosci](https://www.jneurosci.org/content/24/4/877). Ventral-striatum place-value: van der Meer & Redish 2009 (expectation-of-reward at decision points); Lansink 2009 (hippocampus leads ventral striatum in place-reward replay).
