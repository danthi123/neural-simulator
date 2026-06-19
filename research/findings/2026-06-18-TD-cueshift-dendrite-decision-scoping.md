# TD cue-shift (roadmap #3) — the dendrite-decision scoping: the cue-shift is ALREADY point-neuron-feasible (multi-seed GO, no dendrites); the genuine open work is CONSOLIDATION onto the merged "one brain," NOT a dendritic substrate. The D2 two-compartment machinery is the WRONG dendrite for this problem.

**Status:** READ-ONLY deep-research + design-first scoping (the standing "deep research + catalog review FIRST at a new direction / multiply-confirmed roadblock" move, CLAUDE.md). NO `sim/` edits, NO build, NO GPU. Single deliverable = this doc.
**Date:** 2026-06-18.
**Author role:** read-only research subagent. Every load-bearing project claim below was re-verified against the repo (file/finding cited) and the surprising ones (the prior TD GO; the D2 dendrite's actual function) were read in full, not trusted from a summary. Literature was checked against the project's last pass.
**This is a scoping/decision doc, NOT a brain-based result and NOT a commitment to build.**

---

## 0. The one-paragraph answer (the rest is the evidence)

The prompt's premise — *"the TD cue-shift is the LAST roadmap item and THE prime dendrite candidate; the point-neuron temporal bootstrap may hit the same lag/SNR limit the prior N5 TD attempt did, so this is where the deferred dendritic substrate may finally earn its keep"* — **is overtaken by the project's own work and is, on the evidence, the opposite of the case.** The TD cue-shift was **already built on POINT neurons and is a multi-seed GO** (`2026-06-10-N9-TD-cue-shift-A-CSC-GO.md`: the SNc dopamine burst migrates from the reward onto the predictive cue, **r = −0.802 / −0.765 / −0.891**, 3/3 < −0.7, full Schultz signature, both anti-cheats decisive, the TD error computed entirely by neurons). The temporal credit / value-derivative the bootstrap needs was supplied **without dendrites** by three reuse-heavy ingredients: (a) the **A-CSC tapped-delay cue** (the literal Montague-Dayan-Sejnowski complete-serial-compound — the state representation, not a neuron model), (b) the **B-2 conductance-derivative** (a slow GABA_B/GIRK EMA read at the SNc *membrane* = `+dV/dt` in conductance — a genuine multi-timescale temporal filter that is a *point-neuron membrane* mechanism, the protected `sim/` edit), and (c) a **short eligibility tau** (~40 ms ≈ tap-local credit, the already-implemented `fused_eligibility_trace_decay`, catalog C.29). So the prompt's worry — "the point-neuron temporal bootstrap may hit a lag/SNR wall" — was the B-3 *zero-edit* route's failure, and it was **escalated through and SURPASSED on point neurons** by the A-CSC build; it is a documented GO, not an open dendrite candidate. The **genuine residual work** for roadmap #3 is therefore **CONSOLIDATION, not biology**: the cue-shift lives on the *standalone* CPU `snc_stageb_critic_probe`, while the merged "one brain" currently has only the **Rescorla-Wagner** δ=r−V limbic core (GO 6/6, `2026-06-18-limbic-core-rpe-battery-GO.md`) and a **value-train BOUNDARY** (the learned-V δ is graded-but-weak ~1.3×, capped by the position-blind up-state floor, `2026-06-18-merged-navcritic-valuetrain-BOUNDARY.md`). On the dendrite question specifically: the project's existing two-compartment machinery (`sim/dendritic_neuron.py`, D2 Phase 0-2) is the **WRONG dendrite for this problem** — it is a **SPATIAL** (cross-neuron / off-diagonal) decorrelation + feedback-alignment credit machine for the *generalizing cortex* (basal-forward + fixed-random apical + BAC threshold, with a single leak and NO multi-timescale temporal state), whereas the TD cue-shift needs a **TEMPORAL** (multi-timescale eligibility / value-derivative) mechanism — and the project already realizes that temporal mechanism *functionally* in conductance (the GABA_B EMA). My top recommendation: **Option C-first — accept the cue-shift as a point-neuron-DELIVERED capability and make the next increment the CONSOLIDATION (lift the A-CSC TD machinery onto the merged limbic core)**, run the cheap CPU re-validation de-risk that this scoping specifies, and **do NOT open a dendritic substrate for the TD cue-shift** (the dendrite candidate for the TD problem is not supported by the evidence). The dendrite, if ever, is reserved for a *different* capability (the off-diagonal cortex, which is itself de-prioritized post-PPMI — `2026-06-17-dendritic-substrate-frontier-scoping.md`).

---

## 1. DIAGNOSIS — the TD cue-shift requirement, and why the "dendrite candidate" framing does not hold

### 1.1 What the cue-shift IS, and the precise computational requirement (catalog C.28 / C.30 / C.31 / C.22)

The temporal-difference error is δ_t = r_t + γ·V(s_{t+1}) − V(s_t). The bracketed `γ·V(s_{t+1}) − V(s_t)` is the **bootstrap / value-derivative** term that Rescorla-Wagner (δ = r − V) lacks. The observable it produces is the iconic **Schultz 1997 cue-shift**: across cue→reward trials the SNc phasic dopamine burst **migrates** from the reward (US) onto the **earliest predictive cue** (CS), the US burst vacates, and a reward-omission produces a **dip at the expected-reward time** (HS98). The catalog is explicit about *why* this is hard and *what* it requires:

- **C.28** TD error — *"partial — gap is measurable."* The project EMAs `r` to subtract a baseline (≈ R-W one-step) but *"never bootstraps from a learned V(s′), so it cannot produce the cue-shift signature. Closing this requires a critic population (see C.30)."*
- **C.30** Actor-critic — *"actor implemented, critic missing."* Acceptance = cue-shift + omission dip. **The named requirement is a separable VALUE population (the striosome critic), NOT a dendrite.**
- **C.31** Bootstrapping vs Monte Carlo — why phasic DA *must* bootstrap (it shifts on a *single trial*, no episode-end wait); the project is *"a windowed Monte Carlo in which the window is the eligibility-trace decay length."* This is the **one** thing that distinguishes TD from the project's current windowed-MC eligibility.
- **C.29** Eligibility traces / TD(λ) — **"implemented"** (TD(λ) in all but name; `fused_eligibility_trace_decay`). The temporal-credit *substrate* already exists.
- **C.22** Schultz RPE — the migration-r > 0.7 + omission-dip quantitative acceptance criteria.

**The computational requirement, stated precisely:** (i) a **temporally-extended cue state** (so V(cue) is non-zero across the CS→US interval), and (ii) the **bootstrap** `γV(s_{t+1}) − V(s_t)` delivered to the SNc *as a temporal change in value*, not just `−V`. Crucially — **the catalog nowhere names a dendrite as a prerequisite for the cue-shift.** The named prerequisites are C.28 (TD error) ← C.30 (critic population) + C.29 (eligibility, already implemented) + C.33 (PPN reward driver, already built as `reward_us`). This matters: the prompt's "prime dendrite candidate" framing is not the catalog's framing of the gap.

### 1.2 Why the "point neurons may hit the lag/SNR wall" worry was real — and was ALREADY beaten on point neurons

The prompt's exact worry traces to the genuine point-neuron risk the TD design flagged up front (`docs/plans/2026-06-10-N9-TD-cue-shift-design.md` §6.1, citing Potjans-Diesmann-Morrison 2011 "an imperfect dopaminergic error signal *can* drive TD but is fidelity-sensitive"): a rate-coded spiking critic estimates V noisily, and the bootstrap is a *difference of two noisy value estimates* — the classic TD-instability regime. The project ran this to ground cheapest-first, and the arc is fully documented:

| Attempt | What | Result | Why |
|---|---|---|---|
| **B-3** (zero `sim/` edit) | value-derivative via a disinhibition relay (the cheapest route) | **NEGATIVE 3/3** (migration r = +0.000) | the relay forces DENSE critic firing → STDP eligibility net-negative → **the cue value SHRANK** (70→13 Hz), so migration was structurally impossible. *This is the "point-neuron lag/SNR wall" the prompt worries about — and it was the CHEAP route's wall, not the substrate's.* |
| **B-2** (protected conductance-derivative edit) | the value-derivative read from the GABA_B *conductance* (a slow leaky-EMA of `g_gabab`), decoupled from firing density | **PARTIAL 3/3** — RESOLVES the value-growth blocker (V grows 75→103 Hz, 3/3) but the **burst does not migrate** (single-channel edge-vs-level + floor-ceiling conflict) | the single-channel rate critic cannot simultaneously keep the SNc alive at tonic, grow the value, and translate a growing value into a *transferring* cue burst |
| **A-CSC** (+ multi-channel reward relay, re-applying the B-2 edit) | the tapped-delay cue (K=8 time-tagged sub-states, each its own plastic critic synapse) + the reward routed through an excitatory relay the critic inhibits (so `r−V` localizes to the reward window) + FS-clamp + short eligibility tau | **GO 3/3** — **migration r = −0.802 / −0.765 / −0.891**, early-burst-at-US → late-burst-at-CS (genuine transfer), omission-dip-at-reward, value grows, both anti-cheats decisive | the multiple sub-channels decouple the single-channel conflict; the relay decouples the floor-ceiling; the short tau gives one-tap-per-trial back-propagation |

**The decisive fact for the dendrite decision: the migration was achieved on POINT neurons.** No two-compartment neuron, no dendritic plateau. The temporal credit was carried by (a) a *state representation* (A-CSC — a tapped-delay chain of point-neuron relay populations; the cue's time-tagging is the world's stimulus presentation, exactly the legitimate environment boundary the B-3 sustained cue also used), (b) a *membrane conductance filter* (the B-2 slow GABA_B EMA → `+dV/dt`; the slow GIRK channel IS the multi-timescale element, and it is a *somatic-membrane* conductance, not a dendrite), and (c) the *already-implemented eligibility trace* at a short tau. The A-CSC ablation proves the conductance-derivative is load-bearing (off → r drops to −0.624, below the bar, and the early-US burst is lost), so the multi-timescale conductance filter is the genuine new mechanism — and it is a point-neuron membrane mechanism.

> **⇒ The prompt's hypothesis "the point-neuron temporal bootstrap may hit the same lag/SNR limit → the dendrite earns its keep here" is empirically FALSIFIED for the cue-shift.** The point-neuron substrate produces the full Schultz cue-shift, multi-seed, with the anti-cheats. The "N5 TD attempt that failed" the prompt references is the **B-3** zero-edit route (it failed exactly as the prompt predicts — a compound lag + the dense-firing/eligibility collision), but the project did not stop there: it escalated through B-2 → A-CSC and **surpassed the wall on point neurons.** This is the single most important correction to the prompt's framing.

### 1.3 So what IS the genuine open work for roadmap #3? (CONSOLIDATION, not biology)

The roadmap audit (`2026-06-18-full-spikeification-shared-substrate-roadmap.md` §3 #3) is *correct that there is residual work*, but the residual is **consolidation onto the merged "one brain," not a new biological mechanism.** Three facts pin it:

1. **The cue-shift lives on the STANDALONE CPU probe** (`snc_stageb_critic_probe.py --td-csc`), not on the merged bridge. The merged "one brain" (`nav_conv_merged_bridge.py`) has the BG actor + the conversational cortex co-resident, and — as of this week — a **Rescorla-Wagner** limbic core (δ=r−V, the limbic-core RPE battery GO 6/6, lifted/validated `2026-06-18-limbic-core-rpe-battery-GO.md`). It does **NOT** have the TD cue-shift machinery (the A-CSC chain, the B-2 derivative, the multi-channel relay).
2. **The merged value critic is a documented BOUNDARY, and it is the δ=r−V (R-W), not the TD cue-shift.** `2026-06-18-merged-navcritic-valuetrain-BOUNDARY.md`: V *is* learned co-resident (20× weight growth, critic-grade flip, lesion-confirmed), but the afferent-driven δ gap is **graded-but-weak (~1.3×)**, capped by the position-blind non-plastic up-state floor. This is the *spatial value-grading* boundary (V(goal) vs V(far)), a different axis from the *temporal cue-shift* (burst migrating US→CS over trials). Neither this BOUNDARY nor the limbic-core GO is the cue-shift.
3. **The A-CSC GO has documented residuals that are consolidation/scale, not substrate:** the full-vacating gate is graded on 2/3 seeds (the HS98 slow-learning regime, a defensible PASS); the cue's time-tagging is world-clocked (a faithful enrichment would be a self-propagating neural delay chain, deferred because a tiny CPU bridge can't reliably space taps one bin apart — a *scale* limitation, not a substrate wall); and the nav gridworld is orient-solvable / reward-insensitive (so the probe is the sensitive test; an in-vivo cue-shift demonstration needs a reward-load-bearing task — a separate, larger arc).

**⇒ Roadmap #3's honest target is: lift the validated A-CSC TD machinery onto the merged limbic core, so the "one brain" computes δ = r + γV(s′) − V(s) (TD) and shows the cue-shift co-resident with the R-W δ=r−V it already has** — the same *consolidation* pattern as roadmap #1/#2, not a new organ. The point-neuron substrate is sufficient (proven). The wall, if any, is the merged-bridge engineering (the value-train BOUNDARY's up-state floor; reproducibility; the orient-solvable task), not biology.

---

## 2. Is the existing D2 two-compartment machinery REUSABLE for the TD cue-shift? — NO (it is the wrong dendrite)

The prompt asks the right question: D2 Phase 0-2 already built a two-compartment neuron on the bridge + a learned graded cortex; could it supply the dendritic eligibility / multi-timescale credit the TD cue-shift needs? I read the module and the D2 findings in full. **The answer is no, and the reason is sharp: the D2 dendrite is a SPATIAL decorrelation machine, the TD cue-shift needs a TEMPORAL credit machine — orthogonal dendritic functions.**

### 2.1 What the existing two-compartment neuron actually does (`sim/dendritic_neuron.py`, read in full)

`DendriticLayer` (58 lines) is the **Larkum BAC / Guerguiev-Lillicrap-Richards 2017 segregated-dendrites** model, built for **credit assignment + per-compartment divisive normalization**:
- `W_basal` — bottom-up forward drive (the feedforward representation).
- `B_apical` — **FIXED RANDOM** top-down feedback (feedback alignment; *never learned, no weight transport*) — this is a *teaching-signal* / *spatial-context* channel, not a temporal one.
- `v_basal = leak·v_basal + x@W_basal` — a **single leak** (one time constant). The only temporal state is this one-pole low-pass; there is **no multi-timescale eligibility, no plateau-as-trace, no second slow channel.**
- `effective_threshold = theta_high − apical_gain·|apical_depol|` — BAC: apical depolarization *lowers the somatic threshold* (a spatial gain/coincidence operation: basal-apical *co-location*, not temporal credit-over-time).

And the D2 *cortex* purpose (`docs/plans/2026-06-14-D2-dendritic-cortex-build-plan.md`, `2026-06-14-D2-phase1-DONE-phase2-frontier.md`) is explicitly **per-input divisive normalization for the generalizing cortex** — the `enable_dendritic_divisive_gain` per-presynaptic-source gain `g_i = σ/(σ + a_i)` that suppresses high-frequency common hubs. That is the **off-diagonal / cross-neuron (spatial)** decorrelation problem (Mikulasch-Priesemann whitening), not temporal credit assignment.

### 2.2 The TD cue-shift needs the OTHER dendritic function (and the literature draws this line cleanly)

The dendritic literature has two largely separate "dendrites help X" stories, and they map to *different* compartments and *different* mechanisms:

- **SPATIAL / decorrelation / credit-routing** (what D2 built): Mikulasch-Priesemann 2021 (local dendritic balance whitens correlated inputs *across neurons*); Guerguiev-Lillicrap-Richards 2017 / Richards-Lillicrap 2019 (segregated apical/basal dendrites solve the *spatial* credit-assignment / weight-transport problem). These are about *which synapse / which neuron* gets credit and *removing cross-neuron correlation*. **One time constant suffices; the dendrite is a spatial-routing / normalization device.**
- **TEMPORAL / eligibility-over-time** (what the TD cue-shift needs): the **behavioral-timescale-plasticity (BTSP)** family — dendritic *plateau potentials* as a **seconds-scale eligibility trace** (Bittner-Magee 2017; the J Neurosci review confirms verbatim: *"dendritic plateau potentials may help solve the temporal credit assignment problem through an interaction of two signals: a slowly decaying eligibility trace produced by synaptic input and a more global, faster-decaying signal associated with the plateau potential"*), and the **e-prop** factorization (Bellec et al. 2020: gradients = local *eligibility traces* × a global learning signal). **Here the dendrite's value is a slow temporal state (the plateau as a multi-second trace).**

The TD cue-shift's hard part is **(ii) the bootstrap = a temporal change in value** (§1.1). That is a *temporal* mechanism. The D2 two-compartment neuron has no multi-timescale temporal state (a single leak), and its apical channel is a fixed-random *spatial* teaching signal — so **it does not supply the value-derivative / multi-timescale eligibility the cue-shift needs.** Wiring the D2 dendrite into the TD circuit would not produce the bootstrap; the bootstrap is supplied by the **B-2 slow GABA_B conductance EMA** (the functional temporal filter the project already built and validated for exactly this) + the A-CSC tapped-delay state + the short eligibility tau.

### 2.3 The honest corollary — the project ALREADY realizes the "temporal dendrite" function functionally, on point neurons

The deep point: the *functional role* a temporal dendritic plateau would play (a slow multi-timescale trace that lets value-onset register as a burst and value-decay register as a dip) **is already realized in conductance on the point-neuron substrate** by the B-2 slow GABA_B/GIRK EMA at the SNc membrane (`(g_gabab − g_gabab_slow)·(E_exc − V)` = `+dV/dt`). A slow metabotropic conductance with a configurable time constant IS a multi-timescale membrane filter — and it is exactly the kind of slow process a dendritic plateau would otherwise provide. The A-CSC GO is the proof that this functional substitution works. So even on the *temporal*-dendrite reading, the dendrite is not needed for the cue-shift: the project found the cheaper point-neuron-membrane realization and validated it.

> **⇒ D2 two-compartment is NOT reusable for the TD cue-shift** (it is a spatial-decorrelation/credit-routing machine; the cue-shift needs temporal-eligibility, which the project realizes functionally via the B-2 conductance EMA). And the D2 build is *itself* de-prioritized for its *own* (cortex) purpose post-PPMI (`2026-06-17-dendritic-substrate-frontier-scoping.md`: Phase 2's clean-readout control inverted the "gain load-bearing" claim; the generalizing cortex ships on point neurons). There is no version of "reuse D2 for TD" that the evidence supports.

---

## 3. The recommended cheap-first de-risk (the decision-informing experiment)

Because the cue-shift is *already* a point-neuron GO on the standalone probe, the decision-informing experiment is **not** "can point neurons do the cue-shift" (answered: yes) — it is **"does the validated A-CSC cue-shift SURVIVE co-resident on the merged 'one brain,' alongside the existing R-W limbic core, without dendrites and without perturbing the conversational moat / nav byte-identity?"** That is the genuine open question for roadmap #3, it is the cheapest thing that decides the dendrite question (a GO closes it: no dendrite needed; a NEGATIVE localizes whether the merge is the wall vs. a substrate limit), and it is reuse-heavy.

### 3.1 The probe (CPU/numpy first, `SIM_BACKEND=numpy`, NO new `sim/` edit beyond the already-validated B-2 edit)

A cheap-first CPU build that **lifts the A-CSC TD machinery onto a merged-bridge limbic slice** (mirroring the limbic-core lift pattern, `2026-06-18-limbic-core-rpe-battery-GO.md`):
- Reuse `snc_stageb_critic_probe.py --td-csc` *verbatim* as the standalone reference (the validated GO recipe is locked in that finding).
- Build the A-CSC TD slice (the `csc_0..csc_{K-1}` tapped-delay cue + `striosome_value` critic + `csc_fs` FS-clamp + `reward_us` relay + `snc`, the B-2 conductance-derivative on the GABA_B route) as an **additive, default-off opt-in** on `build_merged_nav_conv_bridge` (append after the existing slices, index bases preserved — the exact `co_resident_*` masked-region pattern the merge already uses for `co_resident_rf` / `co_resident_perception` / `co_resident_nav_critic`).
- Run the **same Pavlovian cue→reward protocol** + the **migration-r time-of-peak metric** + the two anti-cheats (cue-pathway lesion; unpaired-timing control) — *the identical battery the standalone A-CSC passed.*

### 3.2 The frozen, pre-registered GO bar (inherited from the A-CSC GO so the bar is not tuned on the test)

- **(headline) Migration:** migration r **< −0.7**, sign-consistent (cue-ward), on **≥ 5/6 seeds** (the standing 6-seed rule for the variable effect; the standalone got −0.80/−0.77/−0.89 at 3 seeds).
- **Early-burst-at-US → late-burst-at-CS** (genuine transfer, not mere shrink): late CS-rate > tonic AND late US-rate substantially shrunk (graded transfer is HS98-faithful and a defensible PASS, per the A-CSC scope).
- **Omission dip at the expected-reward time** (not at the cue): ≥ 5/6.
- **No burst in the CS→US gap** (value flat → derivative ≈ 0): ≥ 5/6.
- **Cue value grows** across trials (the prerequisite B-3 failed): ≥ 5/6.
- **Co-residence preserved (the consolidation gates — decisive for "one brain"):**
  - **MOAT:** the conversational no-confab moat is byte-intact — `what_does('dog','go')=='north'` AND `what_does('river','look') is None`, **6/6** (the dopamine `scope=all` broadcast must not perturb the frozen conv slice; the limbic-core and value-train builds both hold this).
  - **NAV byte-identity:** the nav score is byte-identical to the pre-TD-slice merged bridge (the TD slice is additive/default-off → the existing nav/conv slices are untouched).

### 3.3 Anti-cheats (the project's standard — the TD error must be NEURAL, the migration not a host/co-residence artifact)

- **Cue-pathway lesion → migration vanishes, US reflex survives** (decisive — the migration rides the synaptic `csc → critic` conduit; the standalone got V→0 + US reflex 178–231 Hz, 3/3).
- **Unpaired-timing control → no migration** (DISCRIMINATING — the standalone got r = −0.28/−0.25/−0.28 paired-vs-migrating; this is the control B-2/B-3 could not show, and it is the one that proves the migration is the genuine CS→US contingency).
- **Provenance assertion:** under the TD mode, the SNc drive is `tonic + reward_us(synaptic relay; critic inhibits = r−V) + synaptic GABA_B(−V) + synaptic conductance-derivative(+dV/dt)` ONLY — `current_reward_signal == 0`, no host δ / γV′−V / value-EMA. (The standalone asserts this; the merged version must re-assert it co-resident.)
- **B-2 edit byte-identity-when-off** (already proven, COMBO `e728d7f19d99b5b4` pre==post) — the merged run uses the edit ON for the TD slice; every non-TD merged run is byte-identical (the conv moat + nav byte-identity gates above re-confirm this co-resident).
- **The cue's time-tagging is world-presented** (the apparatus activates sub-state k in bin k) — the legitimate environment boundary; the brain's job (value, derivative, burst, dip, credit) is 100% neural.

### 3.4 Pre-registered three-state outcome (the decision-informing payoff)

- **GO** (the cue-shift survives co-resident, gates + anti-cheats hold) ⇒ **roadmap #3 is DONE on the merged "one brain," on point neurons** — the TD cue-shift is the THIRD canonical dopamine signature consolidated onto the one brain (alongside δ=r−V), **and the dendrite question for the TD problem is closed NEGATIVE** (no dendrite needed; the point-neuron substrate + the B-2 conductance filter suffice). This is the expected outcome given the standalone GO; the only new risk is co-residence (the dopamine `scope=all` broadcast / the masked-slice isolation), which the limbic-core + value-train builds already cleared for the R-W core.
- **BOUNDARY** (the migration degrades co-resident — e.g. the shared dopamine broadcast or the up-state-floor interaction weakens the transfer to a partial) ⇒ the deliverable is the precise *co-residence* boundary (which shared signal interferes), localized to the merge engineering, with the cheap fix named (mask isolation / a per-slice dopamine route — the value-train build already needed a re-pointed `dopamine` over `["snc"]`). **Still not a dendrite finding** — a consolidation finding.
- **NEGATIVE** (the cue-shift cannot be made to survive co-resident on the point-neuron merged bridge at all, despite the standalone GO) ⇒ *this* would be the only outcome that re-opens a substrate question — and even then the honest reading would be "the point-neuron *merged-bridge* engineering (reproducibility / SNR at the 10-neuron SNc pool / the orient-solvable-task interaction) is the wall," not "the cue-shift needs a dendrite" (the standalone GO already disproves that). A dendritic *temporal-eligibility* mechanism would be a *candidate enrichment* only if a careful localization showed the multi-timescale conductance filter is the specific failing piece co-resident — which the standalone GO makes unlikely.

### 3.5 Reusable machinery (the de-risk is almost entirely reuse)

- `research/runners/snc_stageb_critic_probe.py` — `--td-csc` (`run_td_csc`, `run_td_csc_lesion`, `_run_td_csc_mode`, `_csc_substate_weights`, `_drive_timecourse`, `_calibrate_da_baseline`) — the **validated GO recipe, lifted verbatim**.
- `research/runners/nav_conv_merged_bridge.py` — `build_merged_nav_conv_bridge` + the `co_resident_*` masked-region pattern (the additive-slice template); the merged `dopamine` modulator over `["snc"]` (`:693-706`); the GABA_B route + `value_input` gate.
- `research/runners/_limbic_core_rpe_battery_derisk.py` — the standalone-organ → merged-lift pattern + the Schultz RPE battery + the MOAT/lesion anti-cheats (the *exact* harness to extend from R-W to TD).
- `research/runners/_merged_navcritic_valuetrain.py` — the merged-bridge value-train + the MOAT/UNTRAINED/LESION anti-cheats + the GIRK-cap op-point correction (the co-residence wiring to reuse).
- `sim/kernels.py::fused_eligibility_trace_decay` — the eligibility trace (C.29, the short-tau temporal credit).
- The **B-2 protected edit** (`sim/config.py` `enable_td_value_derivative` / `td_slow_tau_ms` / `td_derivative_gain`; `sim/bridge.py` the slow-EMA + `I_td_deriv`) — byte-identical-when-off, already byte-reviewed; the only `sim/` surface, and it is the genuine new mechanism (the multi-timescale conductance filter), NOT a dendrite.

---

## 4. The DENDRITE DECISION framing for the owner (the clean fork)

The prompt frames this as a clean owner call between (A) point-neuron cue-shift, (B) dendritic via the existing D2 substrate, (C) accept the R-W-vs-TD boundary + defer. **On the evidence, the fork has already largely resolved itself, and the honest recommendation reshapes the options:**

### Option A — point-neuron cue-shift (the prompt's "if the de-risk surprises GO")
**This is not a hypothetical surprise — it is the DELIVERED reality.** The cue-shift is a multi-seed point-neuron GO (`2026-06-10-N9-TD-cue-shift-A-CSC-GO.md`). The only open piece is **consolidation** (lift it onto the merged "one brain"), which §3's cheap de-risk tests.
- **Cost:** LOW. Reuse-by-import of the validated A-CSC recipe + the merge's `co_resident_*` pattern + the already-byte-reviewed B-2 edit. CPU de-risk first (afternoon-scale), then a GPU 6-seed merged validation. No new biology, no new `sim/` mechanism.
- **Risk:** LOW-MODERATE — entirely *co-residence engineering* (the shared dopamine broadcast / masked-slice isolation), which the R-W limbic-core + value-train builds already cleared for the merged bridge. The documented A-CSC residuals (graded transfer on 2/3 seeds; world-clocked CSC; orient-solvable in-vivo task) are scale/task, not substrate.
- **⇒ This is my recommended path.** It completes roadmap #3 (the last canonical dopamine signature on the one brain) at low cost, reuse-heavy, and *closes the dendrite question NEGATIVE for the TD problem.*

### Option B — dendritic via the existing D2 two-compartment substrate
**Not supported by the evidence; do NOT pursue for the TD cue-shift.** Three independent reasons:
1. **The dendrite is not needed** — the cue-shift is a point-neuron GO (Option A). The prompt's premise ("the point-neuron temporal bootstrap may hit the lag/SNR wall") was the B-3 *cheap-route* wall; it was *escalated through* on point neurons (B-2 → A-CSC). There is no demonstrated TD wall for a dendrite to break.
2. **The D2 dendrite is the WRONG dendrite** (§2) — it is a *spatial* decorrelation / feedback-alignment machine (single leak, fixed-random apical), with no multi-timescale temporal eligibility. It does not supply the value-derivative the cue-shift needs. The *temporal*-dendrite function (BTSP plateau-as-eligibility) is a *different* mechanism the D2 build did not build — and the project already realizes that function *functionally* via the B-2 conductance EMA.
3. **D2 is de-prioritized for its OWN purpose** — the generalizing cortex it was built for ships on point neurons (PPMI), and Phase 2's clean-readout control inverted the "gain load-bearing" claim (`2026-06-17-dendritic-substrate-frontier-scoping.md`). Opening D2 for the TD problem would be opening a months-scale, highest-variance, hot-path protected edit (a prior dendritic arc VOIDed) for a capability the point-neuron substrate already delivers.
- **Cost (if pursued anyway):** HIGH (months; a new temporal-eligibility dendritic mechanism — *not* the existing D2 — on the hot path). **Risk:** HIGH. **Payoff over Option A:** none demonstrated.
- **The ONLY scenario where a (temporal) dendrite becomes a candidate:** if §3's de-risk returns a *substrate*-NEGATIVE (the cue-shift cannot survive co-resident even after careful merge engineering) AND a localization shows the multi-timescale conductance filter is the *specific* failing piece. The standalone GO makes this unlikely; even then, the mechanism would be a *new* slow-plateau-as-eligibility (BTSP), not the existing D2 spatial dendrite.

### Option C — accept the documented R-W-vs-TD boundary as the deliverable + defer
**Partially superseded** — there is no R-W-vs-TD *boundary* to accept on the standalone substrate (the TD cue-shift is a GO there, not a boundary). The only place an R-W-vs-TD distinction currently stands is **on the merged "one brain"** (it has R-W δ=r−V; it does not yet have TD) — and that is exactly the *consolidation gap* Option A closes cheaply. So "accept the boundary + defer" amounts to "ship the R-W limbic core on the one brain and defer the TD lift" — a legitimate *de-prioritization* choice (if the owner judges the merged-bridge TD lift lower-leverage than other frontiers, e.g. the conversational consolidation or reasoning frontiers ranked in `2026-06-17-dendritic-substrate-frontier-scoping.md` §5), **but NOT a substrate/biology finding and NOT a dendrite justification.**

### The clean recommendation for the owner
**Do Option A: run §3's cheap CPU de-risk (lift the validated A-CSC TD machinery onto the merged limbic core), then a GPU 6-seed merged validation.** It completes roadmap #3 on the one brain at low cost, and **its GO closes the dendrite question NEGATIVE for the TD problem** (the prompt's "prime dendrite candidate" is, on the evidence, not one). **Do NOT open a dendritic substrate for the TD cue-shift** — the dendrite is the wrong tool here (wrong function, and not needed), and the existing D2 machinery is doubly inapplicable (wrong dendrite, and de-prioritized). Reserve any future dendrite spend for the *off-diagonal cortex* question (its own cheap de-risk, `2026-06-17` Option 1) or a *new* capability — never as the TD-cue-shift's answer.

---

## 5. Honest scope / non-claims

- I did **not** re-run the A-CSC probe; I read the GO finding + the B-2/B-3 findings + the design doc in full and the migration numbers are internally consistent + anti-cheat-backed. The §3 de-risk re-measures everything on the merged bridge (the bar is not trusted from the standalone).
- I did **not** relitigate the FHRR bind, the generalizing cortex (PPMI, closed), or the off-diagonal dendrite (de-prioritized, its own scoping). They are out of scope for the TD-cue-shift dendrite question.
- The claim "D2 two-compartment is the wrong dendrite" is a *functional* claim (spatial-decorrelation vs temporal-eligibility), grounded in reading `sim/dendritic_neuron.py` (single leak, fixed-random apical) + the D2 build docs (per-input divisive normalization for the cortex) + the literature's clean split (Mikulasch/GLR spatial vs BTSP/e-prop temporal). It is high-confidence on the functional mismatch; a builder who wanted a *temporal* dendrite would build a *new* slow-plateau mechanism, not reuse D2.
- The honest uncertainty I did **not** resolve: whether §3's co-resident de-risk returns GO / BOUNDARY / NEGATIVE. The standalone GO + the R-W-core co-residence GO make GO the high-probability outcome, but the dopamine-broadcast / SNc-SNR-at-10-neurons / orient-solvable-task interactions are genuine co-residence unknowns the de-risk exists to measure. A *substrate*-NEGATIVE is the only outcome that would re-open even a *temporal*-dendrite candidacy, and it is unlikely.

---

## 6. Sources

### Project record (re-verified this pass, file/finding cited)
- **The TD cue-shift is a point-neuron GO (the load-bearing correction):** `research/findings/2026-06-10-N9-TD-cue-shift-A-CSC-GO.md` (migration r −0.80/−0.77/−0.89, full signature, anti-cheats); `2026-06-10-N9-TD-cue-shift-B2-conductance-derivative-PARTIAL.md` (the conductance-derivative resolves value-growth; single-channel migration wall); `2026-06-10-N9-TD-cue-shift-B3-cheap-first-derisk-NEGATIVE.md` (the zero-edit "point-neuron wall" the prompt references — and the route that was escalated past).
- **The TD design (options A-CSC / B-1 / B-2 / B-3, the three-outcome framing, the Potjans-Diesmann SNR risk):** `docs/plans/2026-06-10-N9-TD-cue-shift-design.md`.
- **The roadmap audit framing #3 as the dendrite candidate (the prompt's source):** `research/findings/2026-06-18-full-spikeification-shared-substrate-roadmap.md` §3 #3, §1a.
- **The merged "one brain" limbic state (R-W, not TD):** `2026-06-18-limbic-core-rpe-battery-GO.md` (δ=r−V GO 6/6, lift pattern); `2026-06-18-merged-navcritic-valuetrain-BOUNDARY.md` (learned-V δ graded-but-weak ~1.3×, up-state floor); `2026-06-18-navcritic-valuetrain-opmap-derisk.md` (the GRADED-δ op-point; the gap was a silent critic, not SNc saturation).
- **The D2 dendrite is the wrong dendrite + de-prioritized:** `sim/dendritic_neuron.py` (read in full — Larkum BAC / GLR2017 segregated dendrites, single leak, fixed-random apical, spatial); `sim/dendritic_plasticity.py` (`urbanczik_senn_update`), `sim/dendritic_mlp.py` (GLR2017 feedback alignment); `docs/plans/2026-06-14-D2-dendritic-cortex-build-plan.md` (the per-input divisive-normalization cortex purpose); `2026-06-14-D2-phase1-DONE-phase2-frontier.md` (the gain delivered + the clean-readout control inverting "load-bearing"); `2026-06-17-dendritic-substrate-frontier-scoping.md` (D2 de-prioritized post-PPMI; the off-diagonal residual is the only narrow dendrite question, itself cheap-de-riskable).
- **Eligibility already implemented:** `sim/kernels.py::fused_eligibility_trace_decay` (C.29).

### Feature catalog (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`)
- **C.28** TD error δ = r + γV(s′) − V(s) — "partial — gap is measurable"; closing it requires a *critic population* (C.30), **not a dendrite** (`:574-575`).
- **C.29** Eligibility traces / TD(λ) — "implemented" (TD(λ) in all but name) (`:583-590`).
- **C.30** Actor-critic — "actor implemented, critic missing"; acceptance = cue-shift + omission dip; striosome=V, SNc=δ, matrix=actor (`:592-599`).
- **C.31** Bootstrapping vs Monte Carlo — why phasic DA must bootstrap (single-trial shift); the project is "windowed Monte Carlo" (`:601-611`).
- **C.33** PPN → DA reward driver (the cue-shift driver, built as `reward_us`) (`:624-637`).
- **C.22** Schultz RPE — the migration-r > 0.7 + omission-dip acceptance numbers (`:907-921`).
- **G.02** Active dendrites — MISSING (~10× compute/neuron); behavioral validation = the Larkum BAC firing experiment (`:2644-2652`) — i.e. the catalog's dendrite entry is the *spatial* BAC coincidence/nonlinearity, not a TD-temporal mechanism.

### Peer-reviewed literature (re-confirmed this pass)
- **Schultz, Dayan, Montague (1997)** "A neural substrate of prediction and reward", *Science* 275:1593 — the cue-shift / TD-dopamine result.
- **Montague, Dayan, Sejnowski (1996)** *J. Neurosci.* 16:1936 — the TD model of dopamine + the complete-serial-compound (CSC) that makes TD reproduce the cue-shift (the A-CSC's basis).
- **Hollerman & Schultz (1998)** *Nat. Neurosci.* 1:304 — the *graded* cue-shift + omission dip (the HS98 slow-learning regime = the A-CSC's defensible graded-transfer PASS).
- **Potjans, Diesmann, Morrison (2011)** "An imperfect dopaminergic error signal can drive temporal-difference learning", *Front. Comput. Neurosci.* (PMC3093351) — a noisy spiking RPE *can* drive TD but is fidelity-sensitive (the §6.1 risk the project ran to ground).
- **Frémaux, Sprekeler, Gerstner (2013)** "Reinforcement Learning Using a Continuous Time Actor-Critic Framework with Spiking Neurons", *PLoS Comput. Biol.* 9:e1003024 — the canonical spiking actor-critic (a spiking critic estimates V; the TD error modulates reward-STDP) the project's circuit follows — *point-neuron, no dendrite.*
- **A spiking temporal-difference learning model based on dopamine-modulated plasticity**, *BMC Neuroscience* 2009 (10(S1):P140) — spiking TD with a striatal/VTA critic, point neurons.
- **Bittner, Milstein, Grienberger, Romani, Magee (2017)** "Behavioral time scale synaptic plasticity underlies CA1 place fields", *Science* — BTSP: dendritic *plateau potentials* as a seconds-scale eligibility (the *temporal*-dendrite mechanism, distinct from the D2 spatial dendrite).
- **Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass (2020)** "A solution to the learning dilemma for recurrent networks of spiking neurons (e-prop)", *Nat. Commun.* 11:3625 — eligibility traces × a learning signal (the temporal-credit factorization).
- **Guerguiev, Lillicrap, Richards (2017)** "Towards deep learning with segregated dendrites", *eLife* 6:e22901 + **Richards & Lillicrap (2019)** "Dendritic solutions to the credit assignment problem", *Curr. Opin. Neurobiol.* — the *spatial* segregated-dendrite credit-assignment story (= the D2 dendrite's lineage; confirms D2 is the spatial, not temporal, dendrite).
- **Mikulasch, Rudelt, Priesemann (2021)** "Local dendritic balance enables learning of efficient representations in networks of spiking neurons", *PNAS* (PMC8685685) — the *spatial* cross-neuron decorrelation dendrite (the D2 cortex motivation; not the TD-temporal mechanism).
- **Sutton & Barto** *Reinforcement Learning* 2e — Ch 6 (TD/bootstrapping), Ch 7 (eligibility/TD(λ)), Ch 11 (actor-critic), Ch 12 (the stimulus-trace/CSC for the cue-shift).
- **Kandel et al.** *Principles of Neural Science* 6e — Ch 43 (dopamine/reward, the Schultz cue-shift figure).
