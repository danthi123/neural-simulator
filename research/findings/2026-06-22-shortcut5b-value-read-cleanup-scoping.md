# Shortcut #5b — the value-read cleanup (separate LEARNED value from STRUCTURAL place-code salience): deep-research scoping (2026-06-21)

**Type:** READ-ONLY deep-research scoping (the scope-before-build step). NO code edits, NO experiments. The deliverable is this doc + a single recommended cheap-first de-risk for the controller to run.

**Goal restated.** The last navigation host shortcut to close by default is the place-code host-Gaussian (`vs_place_context`). Its neural replacement — the spatial-phase grid-cell front end — is BUILT and CLOSES the core job (R1 afferent selectivity, the value grades 3/3) but ships OPT-IN because of a characterized downstream boundary: **the value read-out conflates the place code's intrinsic STRUCTURAL near/far magnitude asymmetry with the LEARNED near/far value**, so the dopamine-RPE δ is not a clean learned RPE. This scoping diagnoses that boundary and ranks biology-grounded, cheap-first mechanisms to surpass it, so the grid front end can become the honest production default.

---

## VERDICT (one line)

**SURPASSABLE, and cheaply — the boundary is the READ mechanism, not the place code or the learned value.** The probe currently reads the dopamine δ as a **raw single-state magnitude** (the reward burst at NEAR, where value is predicted/subtracted, vs at FAR, where it is not), which by construction reads TOTAL afferent magnitude = structural + learned. The biologically-correct dopamine signal is a **temporal-difference error δ = r + γV(s′) − V(s)** (catalog C.28/C.30/C.31; Schultz–Dayan–Montague 1997; Schultz 1998), which is a DIFFERENCE between successive states and therefore **cancels any baseline that is approximately consistent across adjacent states — exactly the structural place-code magnitude asymmetry**. The reusable machinery already exists and is decisive: `sim/td_value_critic.py` implements `delta = r + GAMMA*v_tp1 - v_t` AND carries a `no_bootstrap` mode `delta = r - v_t` that is *exactly* the probe's current single-state read — so the TD-vs-no-bootstrap contrast is a ready-made, frozen-bar anti-cheat. **Recommended cheap-first de-risk (RANK 1): a TD-difference δ read** — read the δ between an adjacent FAR→NEAR transition (`r + γV(near) − V(far)`) instead of the raw burst at a single location — with the anti-cheat that this δ must SURVIVE the learned-value controls (shuffle_v / no_learn / lesion) collapsing it AND must NOT survive on a structurally-asymmetric-but-unlearned code (the magnitude-matched `shuffle_v` that the raw read currently passes). If the TD δ holds 3/3 AND collapses the magnitude-matched shuffle_v, the host-Gaussian scaffold retires as the production default. The genuinely-irreducible residual, if any, is small and named in MOVE 1 below.

---

## MOVE 1 — ISOLATE + QUANTIFY the genuine residual

### What the read actually computes (verified against the probe code, NOT the doc prose)

The δ read is `_snc_burst_rate_graded(px, py)` in `research/runners/_n5_grid_frontend_onbridge_probe.py` (lines 944–990). For ONE location it:

1. Resets the slow plateau / GABA_B / SNc membrane (`_reset()`, lines 924–942).
2. Drives `place_sensors` with the grid code at `(px,py)` + SNc tonic; runs a LEAD window (`lead_steps≈120`) during which the place→critic→GABA_B path builds a GABA_B conductance ONTO the SNc (the "−V" subtraction term).
3. Fires the spiking reward US into the SNc and counts SNc spikes over `hold_steps≈40` → the SNc burst rate at that location.

The authoritative δ is then computed in `_read_graded_v_delta` (lines 993–996):

```
snc_pred,   _ = _snc_burst_rate_graded(near)   # NEAR: V is high → GABA_B subtracts → SNc burst suppressed
snc_unpred, _ = _snc_burst_rate_graded(far)    # FAR:  V is low  → little GABA_B → SNc burst full
gabab_gap = (snc_unpred > 1.30 * snc_pred)      # the δ-gap
```

**This is a raw-magnitude, single-state read.** Each location is read in isolation; the quantity is `burst(FAR) / burst(NEAR)`. There is **no `V(s′)` term and no difference between successive states** — it is the `no_bootstrap` form `δ ≈ r − V(s)` evaluated separately at two places, not the TD form `δ = r + γV(s′) − V(s)`. (Catalog C.28 names this gap precisely for the project as a whole: "Project's `current_reward_signal = r(t)` *not* `r(t) + γV(s′) − V(s)` … never bootstraps from a learned `V(s′)`.")

### How much of the δ is structural vs learned (the numbers, pinned from the finding docs)

| quantity | value | source | what it shows |
|---|---|---|---|
| `w_n/f` after value-train (grid arm) | **1.43 / 1.07 / 1.91×** (seeds 42/43/44) | R1-grid-frontend-derisk | the value-train DOES learn a real near/far ratio (the learning is genuine, ~1.0 → ~1.5–1.9×) |
| `w_n/f` with no value-train (`no_learn`) | **≈1.0** | volley-normalization-close, control battery | the bare structural code has a FLAT learned ratio (init weights) |
| `w_n/f` after magnitude-matched `shuffle_v` | **1.005 / 0.817 / 1.049×** (flat, **inverted on seed 43**) | volley-normalization-close, shuffle_v row | the LEARNED spatial correspondence is destroyed (flat / far>near) |
| δ (`gabab_gap`) on magnitude-matched `shuffle_v` | **True 3/3** (snc 0/50) | volley-normalization-close, shuffle_v row | **the δ SURVIVES with flat/inverted learned weights → the δ does NOT require the learned ratio → it is reading STRUCTURAL magnitude** |
| δ on `no_learn` | **False 3/3** (collapses) | control battery | BUT a δ at INIT weights collapses — the magnitude must be raised (by value-train OR normalization) for the structural δ to appear |
| `w_n/f` on grid arm WITH place-drive normalization | **0.97 / 1.00 / 0.84** (flat) AND δ collapses | volley-normalization-close, normalization result | removing the structural asymmetry leaves the value-train NOTHING to grow from → flat ratio → no δ |

**The precise residual.** The genuine learning is real (`w_n/f` 1.0→1.43–1.95) but the READ is a raw-magnitude read that reflects TOTAL afferent magnitude (structural baseline + learned increment), not the learned increment selectively. Three facts triangulate it exactly:

1. The δ survives at flat/inverted learned weights (magnitude-matched shuffle_v) → it is reading the structural magnitude asymmetry, not the learned ratio.
2. The δ collapses at INIT weights (no_learn) → the structural asymmetry must be *amplified* (by weight magnitude) to be read; it is latent in the bare code.
3. Killing the structural asymmetry (place-drive normalization) also kills the learning substrate → on this substrate the value-train learns the near/far V *by amplifying* the structural magnitude differences, so structural and learned magnitude are inseparable AT THE INPUT.

**⇒ The irreducible part is NOT the place code and NOT the value-train. It is the read operator: a single-state magnitude read cannot separate a learned increment from a co-located structural baseline.** This is a ~one-operator residual (the `snc_unpred(FAR) / snc_pred(NEAR)` ratio in `_read_graded_v_delta`), and it is the *kind* of residual the SURPASS doctrine targets: small, localized, and (per MOVE 2) the result of testing the wrong read. The place code's structural magnitude asymmetry is a legitimate, biology-faithful property of grid/place codes (grid cells genuinely fire more densely in some locations); the fix is not to flatten the code but to read it correctly.

---

## MOVE 2 — REFRAME via "how does REAL biology separate learned value from structural salience?"

This is a classical, well-characterized neuroscience problem. The literature gives a clear ranking, and the leading answer is the one the catalog already names as the project's missing piece.

### (LEADING) The dopamine signal is a TEMPORAL-DIFFERENCE error, and the difference cancels a consistent structural baseline

The canonical result (Schultz, Dayan & Montague 1997; Schultz 1998; catalog C.28/C.30/C.31; Sutton & Barto §6.1, §11.1): phasic midbrain dopamine encodes

```
δ(t) = r(t) + γ·V(s(t+1)) − V(s(t))
```

— a **difference between the value of successive states**, not a raw value read. The web-confirmed framing: "the goal of TD learning is to learn a value function … with learning driven by the difference between value at two subsequent states", and "when the reward prediction is correct, the actual reward value is cancelled out by the prediction" (introduction-to-RL-for-neuroscience; O'Reilly–Munakata CCN 3e §7.3). Dopamine increases *above* and decreases *below* a baseline for positive/negative δ.

**Why this dissolves the boundary.** The structural place-code magnitude asymmetry is a property of the place representation that is present in `V(s)` AND `V(s′)` alike — it is approximately consistent across adjacent states (a near location and its neighbour have similar structural drive density). A *difference* `γV(s′) − V(s)` between adjacent states therefore **cancels the common structural baseline** and leaves only the part of V that genuinely DIFFERS between the states — which, after learning, is the learned near/far value gradient. The raw single-state read keeps the baseline (it has nothing to subtract it against); the TD difference removes it for free. This is the same mathematical move that makes the RPE zero for a fully-predicted reward.

**This is precisely the project's named gap.** Catalog C.28: the project uses `r(t)` not `r(t) + γV(s′) − V(s)`; "`--adaptive-da` EMAs `r(t)` to subtract a baseline … but never bootstraps from a learned `V(s′)`". Catalog C.31: the project is "a windowed Monte Carlo … without any predictive value computation." The grid-frontend δ probe inherits exactly this: it reads a raw burst magnitude, never a bootstrapped difference. **We have been testing the `no_bootstrap` read.**

### (SECONDARY) Adaptive / reference-dependent / divisive coding normalizes DA to the value RANGE

Tobler, Fiorillo & Schultz 2005 ("Adaptive coding of reward value by dopamine neurons", *Science*): dopamine responses adapt to the expected reward and **rescale (gain-adjust) to the variance/range of reward value** — "a normalization process that brings different magnitudes onto the same coding scale." This is a contrast/divisive normalization of the value read against a reference (the expected value / the range). A read that divided the burst by a running reference of total afferent drive would partially cancel the structural baseline (which is a roughly constant component of the total). Weaker than the TD difference for THIS problem because the structural component is location-DEPENDENT (near has more than far), so a single scalar reference under-corrects; but it composes with the TD read and the project already has the substrate (`--adaptive-da`'s EMA baseline `R̄`, catalog line 536).

### (SECONDARY) Baseline subtraction / actor–critic baseline / average-reward `R̄`

The actor–critic and average-reward RL formalisms subtract a learned baseline before crediting (catalog C.30; the `R̄` EMA already implicit in `--adaptive-da`, catalog line 536). A read that subtracts the PRE-LEARNING structural baseline (measured once at init, before any value-train) from the post-learning read isolates the learned increment directly. This is the most literal "remove the structural part" and is a cheap, defensible control — but it requires a stored pre-learning reference per location and is less biologically autonomous than the TD difference (the brain does not store an init snapshot; it bootstraps).

### (WEAKEST for this problem) Separate value vs salience dopamine pathways

Bromberg-Martin, Matsumoto & Hikosaka 2010 ("Dopamine in Motivational Control", *Neuron* 68:815–834) and Matsumoto & Hikosaka 2009: distinct DA populations code motivational VALUE (excited by reward, inhibited by aversive) vs motivational SALIENCE/intensity (excited by both) — biology DOES, anatomically, separate value from salience/intensity. **However**, the project's own catalog flags the strong counter-evidence (C.23 Supplemental, Schultz 2016): the *phasic* RPE is "remarkably similar across the population, with only graded — not categorical — differences", and the apparent value-vs-salience split is reinterpreted as varying Component-1 (physical-intensity/detection) sensitivities, not anatomically distinct value vs salience encoders. So a separate structural-salience pathway is biologically defensible but is the *weaker* hypothesis AND the most expensive to build (a second population + its own learning). It is the wrong place to spend first.

**Reframe conclusion.** We tested the wrong read (a raw single-state magnitude = the `no_bootstrap` form). Real biology reads dopamine as a temporal-difference error, and that difference is exactly what cancels a structural baseline that is consistent across successive states. The leading fix is to read the δ as a difference, not a magnitude.

---

## MOVE 3 — RANK cheap-first SURPASS mechanisms

The target: a value read whose δ reflects the LEARNED near/far value increment, not the structural place-code magnitude. Ranked cheapest-first.

| # | mechanism | biology | reusable project machinery | de-risk cost | anti-cheat it needs |
|---|---|---|---|---|---|
| **1** | **TD-difference δ read** — read `δ = r + γV(near) − V(far)` across an adjacent FAR→NEAR transition (a bootstrapped difference of successive-state values), instead of the raw single-state burst at NEAR. The structural baseline (common to V(near) and V(far)) cancels; only the learned increment survives. | Schultz–Dayan–Montague 1997; Schultz 1998; **catalog C.28 / C.30 / C.31** (the project's explicitly-named missing piece — phasic DA = `r + γV(s′) − V(s)`). | **`sim/td_value_critic.py` — DIRECTLY: it computes `delta = r + GAMMA*v_tp1 - v_t` AND has a `no_bootstrap` mode `delta = r - v_t` = the probe's CURRENT read.** The GABA_B/SNc subtraction path already reads V at a location; the change is to read it at TWO adjacent locations and take the (γ-discounted) difference, reusing the existing `_snc_burst_rate_graded` per-location read + the existing `cp_conductance_g_gabab` "−V" term. The existing controls (render / scramble / no_learn / lesion / shuffle_v) port unchanged. | **LOW** — probe-only, reuse-by-import, NO `sim/` edit (the TD difference is computed in the probe from two existing per-location reads). ~1–2 runs × 3 seeds. The deterministic-read + volley-normalization levers already stabilize the per-location reads. | (i) δ SURVIVES while no_learn / lesion / **magnitude-matched shuffle_v** COLLAPSE it (the shuffle_v is the make-or-break: the raw read currently PASSES it — the TD read must FAIL it, because shuffle_v destroys the learned near/far difference the TD read depends on). (ii) δ does NOT appear on a structurally-asymmetric-but-unlearned code. (iii) the ready-made TD-vs-`no_bootstrap` frozen-bar contrast in `td_critic_core` as the algorithmic positive control. |
| **2** | **Baseline-subtracted read** — store the PRE-learning structural read per location (measured at init, value-train OFF), subtract it from the post-learning read: `δ_learned = burst_post(loc) − burst_init(loc)`. | Actor–critic baseline; average-reward RL `R̄` (catalog C.30; the EMA baseline already in `--adaptive-da`, catalog line 536). | The probe already runs a `no_learn` arm (value_train_trials=0) that IS the per-location init read — reuse it as the stored baseline. The graded-V read (`_read_graded_v_near_far`) gives the per-location V directly. | **LOW–MEDIUM** — probe-only; needs a stored init reference per location (one extra `no_learn` pass) + the subtraction. NO `sim/` edit. | δ_learned SURVIVES the value-train, COLLAPSES for shuffle_v / no_learn (subtracting init from init = 0) / lesion. Must confirm the init read is on the IDENTICAL frozen place code (same grid draw) so the subtraction is apples-to-apples. |
| **3** | **Divisive / reference normalization of the value read** — divide the burst by a running reference of total afferent drive (or the expected-value EMA), rescaling out the roughly-constant structural component. | Tobler–Fiorillo–Schultz 2005 (adaptive/divisive coding to the value range); Carandini–Heeger divisive normalization. | The place-drive normalization lever (`--normalize-place-drive`) already exists but normalizes the INPUT (which kills the learning substrate); the fix is to normalize the READ (the output), not the input — divide the SNc burst by the total place→critic drive. `--adaptive-da`'s `R̄` EMA is the reference substrate. | **MEDIUM** — the structural component is location-DEPENDENT, so a single scalar reference under-corrects; likely needs a per-location or graded reference → more tuning. NO `sim/` edit if done on the read. | δ SURVIVES the value-train, COLLAPSES shuffle_v / no_learn / lesion. Critically must show it does NOT merely re-introduce the input-normalization collapse (it must leave the learned increment intact while removing only the structural baseline). |
| **4** | **Separate structural-salience vs value pathway** — a second read population coding total-intensity/salience, subtracted from the value read so only the learned-value-specific part drives δ. | Bromberg-Martin–Matsumoto–Hikosaka 2010; Matsumoto–Hikosaka 2009 (value vs salience DA populations). **Caveat:** catalog C.23 Supplemental (Schultz 2016) argues the phasic RPE is uniform across DA neurons and the split is graded-not-categorical — the WEAKER biology here. | Would need a new population + its own learning (`td_value_critic` is value-only). The regions/neuromodulator framework could host it. | **HIGH** — a new mechanism class (a second learned population + subtraction), not a composition of de-risked pieces. Would itself trigger the research gate for a new mechanism. | Full battery + must show the salience pathway tracks STRUCTURAL magnitude (fires for the unlearned structurally-asymmetric code) while the value pathway tracks the LEARNED increment — i.e. the two pathways must dissociate on the shuffle_v / no_learn controls. |

**Why #1 first.** It is the cheapest (probe-only, the machinery already computes the exact δ and the exact `no_bootstrap` foil), it is the biologically-correct dopamine signal (not a workaround), it is the project's explicitly-named missing piece (catalog C.28/C.30/C.31), and the mathematics guarantees the structural-baseline cancellation that the raw read cannot achieve. #2 is a clean, even-cheaper *sanity* control to run alongside (it literally subtracts the structural part) but is less biologically autonomous (stores an init snapshot the brain wouldn't). #3 composes with #1 (range normalization on top of a difference) but is weaker alone. #4 is a new mechanism class and the weaker biology — reserve it only if #1–#3 all fail.

---

## MOVE 4 — VERDICT: surpassable-and-how-cheaply vs irreducible-and-why

**SURPASSABLE, cheaply (RANK 1, probe-only, NO `sim/` edit).** The boundary is the READ operator, not the place code or the learned value. The grid front end gives a genuinely selective, genuinely learned near/far value (`w_n/f` 1.0→1.43–1.95, R1 closed 3/3). The defect is that the probe reads the dopamine δ as a raw single-state magnitude (the `no_bootstrap` form), which keeps the structural baseline. Reading it as the biologically-correct temporal-difference `δ = r + γV(near) − V(far)` cancels the structural baseline (common to both states) and leaves the learned increment — and the reusable machinery (`sim/td_value_critic.py`) already computes exactly this δ AND the exact raw-read foil. The host-Gaussian `vs_place_context` scaffold retires as the production default IF the TD δ holds 3/3 AND collapses the magnitude-matched shuffle_v.

**The genuinely-irreducible part, precisely (the honest fallback).** IF the TD-difference read does NOT collapse the magnitude-matched shuffle_v — i.e. if the structural magnitude asymmetry is so location-correlated that it survives even the adjacent-state difference — then the irreducible residual is real and is exactly this: *on a point-neuron substrate where the value-train learns the near/far V by amplifying the place code's intrinsic structural magnitude differences (place-drive normalization confirms the two are inseparable AT THE INPUT), a value read cannot fully separate the learned increment from the structural baseline at the same location, because they are the same physical quantity (afferent drive magnitude) differing only in how it was set (developmentally vs by learning).* That would be a substrate-honest negative (an instance of the documented point-neuron limit family), and the correct disposition would be: keep the grid front end as the production place code (R1 is genuinely closed), and document the value-read's structural contamination as a characterized boundary that a dendritic substrate (separate apical/basal compartments for structural drive vs learned value) could separate — the deferred deep-frontier, not a blocker for retiring the host-Gaussian on R1 grounds. **But the TD difference is the cheap test that decides this, and the mathematics strongly predicts it cancels the baseline — so run it before accepting any boundary.**

---

## RECOMMENDED CHEAP-FIRST DE-RISK (for the controller to run)

**RANK 1 — the TD-difference δ read.** Add a probe-level δ read that computes the bootstrapped difference across an adjacent FAR→NEAR transition rather than the raw single-state burst:

```
δ_TD = r + γ · V(near) − V(far)
```

where `V(near)`, `V(far)` are the existing per-location graded-V / GABA_B reads (reuse `_snc_burst_rate_graded` / `_read_graded_v_near_far`), `γ` is the discount (the `td_value_critic` GAMMA=0.95 default), and `r` is the spiking reward US already in the read. Compute it in the probe (NO `sim/` edit). Hold WITH the already-validated `--deterministic-read` (+ the volley-normalization `--synaptic-scaling --synscale-mode freeze_seam --synscale-fs-target-wnear 0.5`) so the per-location reads are seed-stable. Run 3 seeds (42/43/44) first; if GO, confirm at 6 seeds.

**The exact anti-cheat controls (the read must show δ tracks LEARNED value, not structural magnitude):**

1. **δ_TD SURVIVES on the grid arm (TEST)** — the learned near/far value produces a positive TD δ, 3/3 seeds.
2. **δ_TD COLLAPSES on `no_learn`** (value_train_trials=0) — no learning → no learned increment → no TD δ.
3. **δ_TD COLLAPSES on `lesion`** (graded_plateau_strength=0) — no graded read-out → no V → no δ.
4. **δ_TD COLLAPSES on the MAGNITUDE-MATCHED `shuffle_v`** (the make-or-break, the one the RAW read currently PASSES): permute the learned place→value weights across place neurons, then normalize to the same magnitude as the grid arm. The raw read survives this (it reads total magnitude); **the TD read MUST fail it** (shuffle_v destroys the learned near/far *difference* the TD read depends on, while leaving the magnitude matched). This is the decisive discriminator between "δ tracks learned value" and "δ tracks structural magnitude."
5. **δ_TD does NOT appear on a structurally-asymmetric-but-unlearned code** — i.e. the structural baseline alone (no learning) must not manufacture a TD δ. (Controls 2 and 4 jointly establish this.)
6. **Algorithmic positive control (free):** the existing `td_critic_core` frozen-bar TD-vs-`no_bootstrap` contrast in `sim/td_value_critic.py` confirms, in pure array math, that the TD form produces the learned-value signal where the raw read does not — the same `no_bootstrap = r − V` form the probe currently uses.

**GO = δ_TD survives 3/3 on grid AND collapses on no_learn / lesion / magnitude-matched shuffle_v → the δ is the genuine LEARNED RPE, the structural contamination is removed → the grid front end becomes the honest production default and the host-Gaussian `vs_place_context` scaffold retires (R1 closed 3/3 + the value read now learned-clean).** NO-GO (δ_TD survives the magnitude-matched shuffle_v too) = the structural/learned inseparability is irreducible on the point-neuron substrate (MOVE 4 fallback) → keep the grid as the production place code, document the value-read contamination as a dendritic-frontier boundary, and the host-Gaussian retires on R1 grounds with the value-read residual characterized.

---

## Reusable machinery inventory (for the build)

- `research/runners/_n5_grid_frontend_onbridge_probe.py` — the grid front end, the per-location V/δ reads (`_snc_burst_rate_graded`, `_read_graded_v_delta`, `_read_graded_v_near_far`), the full control battery (`grid` / `render` / `scramble` / `no_learn` / `lesion` / `shuffle_v`), `--deterministic-read`, `--synaptic-scaling`, `--normalize-place-drive`. The TD read is a new δ computed from the existing two per-location reads.
- `sim/td_value_critic.py` — `delta = r + GAMMA*v_tp1 - v_t` (the TD form) AND `no_bootstrap` mode `delta = r - v_t` (the probe's current read); GAMMA=0.95, LAMBDA=0.9; reuses `fused_eligibility_trace_decay` unmodified; the `td_critic_core` frozen science bars (the ready-made TD-vs-no-bootstrap anti-cheat).
- `sim/bridge.py` — the GABA_B/graded-plateau critic path (`cp_conductance_g_gabab`, `cp_conductance_g_graded_plateau`); the deterministic-transpose-matvec path (`cfg.deterministic_transpose_matvec`, `sim/config.py:293-300`); the synaptic-scaling path (`cfg.enable_synaptic_scaling`, `sim/bridge.py:7402`). All UNCHANGED by this scoping.
- `--adaptive-da` EMA baseline `R̄` (catalog line 536) — the substrate for the RANK-3 reference normalization if RANK-1 needs composing.

## Moat confirmation

A nav-only probe (no conversational regions). The place/critic/SNc state (`cp_connections` / `cp_firing_states` / `cp_conductance_g_graded_plateau` / `cp_conductance_g_gabab`) is array-disjoint from the composer's complex `cp_rf_w_*` synapses. The no-confab moat is preserved by construction and is untouched by any read-mechanism change.

## Sources (literature)

- [An introduction to reinforcement learning for neuroscience (arXiv 2311.07315)](https://arxiv.org/pdf/2311.07315) — δ = −V(s) + r + γV(s′); learning driven by the difference between successive-state values.
- [O'Reilly & Munakata, Computational Cognitive Neuroscience 3e §7.3 — Dopamine and Temporal Difference Reinforcement Learning](https://med.libretexts.org/Bookshelves/Pharmacology_and_Neuroscience/Computational_Cognitive_Neuroscience_3e_(O'Reilly_and_Munakata)/07:_Motor_Control_and_Reinforcement_Learning/7.03:_Dopamine_and_Temporal_Difference_Reinforcement_Learning) — "when the reward prediction is correct, the actual reward value is cancelled out by the prediction."
- [Schultz, Dopamine reward prediction error coding (PubMed 27069377)](https://pubmed.ncbi.nlm.nih.gov/27069377/) — phasic DA = RPE; increases above / decreases below baseline.
- [Tobler, Fiorillo & Schultz 2005, Adaptive Coding of Reward Value by Dopamine Neurons (Science)](https://www.pdn.cam.ac.uk/system/files/documents/2005-tobler-science.pdf) — adaptive/divisive normalization of DA to the reward range.
- [Bromberg-Martin, Matsumoto & Hikosaka 2010, Dopamine in Motivational Control: Rewarding, Aversive, and Alerting (Neuron 68:815-834)](https://www.cell.com/neuron/fulltext/S0896-6273(10)00938-4) — distinct value-coding vs salience-coding DA populations.
- Catalog (`sim-catalog/references/feature-catalog.md`): **C.28** (TD error δ = r + γV(s′) − V(s); the project uses r(t) not the TD form), **C.30** (actor-critic, the td_value_critic mapping), **C.31** (bootstrapping vs Monte Carlo), **C.23/C.24** (value vs salience DA; Schultz16 graded-not-categorical caveat), line 536 (the `R̄` EMA baseline in `--adaptive-da`).

## Disposition

- **This scoping:** the value-read boundary is diagnosed (a raw single-state magnitude read = the `no_bootstrap` form) and the cheap-first surpass ranked (RANK 1: a TD-difference δ read, reusing `td_value_critic`'s exact δ + foil, NO `sim/` edit).
- **Next (the controller's de-risk):** run RANK 1 — the TD-difference δ read with the 6-control anti-cheat (the magnitude-matched shuffle_v is the make-or-break). GO → the grid front end becomes the honest production default; the host-Gaussian `vs_place_context` retires (R1 closed 3/3 + value-read learned-clean). NO-GO → the structural/learned inseparability is the documented point-neuron/dendritic-frontier boundary; the host-Gaussian still retires on R1 grounds with the value-read residual precisely characterized.
