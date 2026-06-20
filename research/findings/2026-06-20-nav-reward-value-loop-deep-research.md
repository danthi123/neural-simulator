# Nav reward/value/sustained-control loop (shortcut burndown #6-#9) — deep-research + catalog gate

**Date:** 2026-06-20
**Type:** READ-ONLY deep-research + reference-catalog review (the project's standing "research-first at a roadblock" gate). NO code edited, NO experiments run. Load-bearing claims trust-but-verified against the actual project findings / catalog / code text.
**Scope:** the FOUR host (non-spiking) cognitive shortcuts that together form the navigation reward/value/perception loop — treated as ONE closed-loop family per the burndown:
- **#6** host Manhattan orienting heuristic (`g11_bg_runner.py:6385-6411`, `h_drive = 800·h_strength` injected into the goal-reducing `cortex_X` pools)
- **#7** host distance reward (`sign(Δ eccentricity)` / Manhattan, written to the SNc as a current)
- **#8** host EMA value baseline (`reward_ema` / `_V_scaffold` — the host critic)
- **#9** host Gaussian place/goal code (`vs_place_context`, the tuned-Gaussian critic afferent)

**The blocker being researched:** the fully-NEURAL version of this loop is a documented NO-GO
(`2026-06-19-nav-spiking-sc-deploy-NO-GO.md`): the spiking-superior-colliculus + neural-reward → SNc → critic → actor-drive closed loop navigates **~58× WORSE** than the host scaffold, the **actor goes silent** after warmup, and the scramble control localized the failure to the **reward/actor-drive half** (NOT the orienting). The related **#5** self-org place-code value-train δ also underperforms the host Gaussian (`2026-06-18-merged-neural-place-code-delta-probe-NEGATIVE.md`; `2026-06-19-place-code-sparsify-default-BOUNDARY.md`).

**The pre-registered question (from the dispatch):** is this a **dendritic / credit-assignment** boundary (apical-basal, multi-layer credit routing the point-neuron substrate cannot do), **OR** a **point-neuron loop-stability / operating-point** problem (the organs work in isolation but the closed loop does not sustain — the same KIND of issue as **#4** the motor read-out, which looked like a wall at ~1.7× and turned out point-neuron-tunable to 1.16×)?

---

## 0. TL;DR / headline verdict

**This is NOT a dendritic / credit-assignment boundary. It is a point-neuron LOOP-STABILITY / OPERATING-POINT problem, and it is the same family as #4 (the action-decision read-out that was point-neuron-tunable).** The diagnosis is decisive and reuses the project's own evidence:

1. **The actor-silence is an OPEN-LOOP starvation, not a credit-assignment failure.** In the SC-deploy config the cortex (the actor's input stage) has **no recurrent self-sustain** — the `thal_X → cortex_X` reentrant closure (catalog **A.05**) is gated behind `enable_cluster_a_closed_loop` (default **OFF**, and NOT set by `--spiking-sc`). With the host heuristic's 800 pA cortex drive removed (`heuristic_strength=0`) AND the place-goal-readout perception that actually carried nav also removed, the ONLY thing left driving the actor toward the goal is the spiking-SC `sc_map → cortex_X` orienting current — and the scramble control proves that arm is **not load-bearing for the failure** (scramble ≈ SC-on within 1%). So the actor is starved of drive and the open cortico-BG-thalamic loop has nothing to keep it firing → it goes silent. This is loop *gain/closure*, not a dendrite.

2. **The organs all VALIDATE in isolation** — the spiking SC bump (peak/mean 35.7×, N1 8/8), the neural reward (corr(eccentricity, reward_us) = −0.989, omission dip, lesion-clean), the spiking SNc + GABA_B critic (the full Stage-B δ=r−V gates pass, lesion-confirmed synaptic). The failure is exactly the project's recurring **"works-in-isolation / fails-in-the-whole-loop"** signature, which is a systems-integration / operating-point class, not a substrate-cannot-compute-it class.

3. **The earlier, milder nav A/B (Stage C) ALREADY showed the spiking loop is COMPETITIVE** when the perception scaffold is retained: the neural δ=r−V vs the host δ=r−V_scaffold was **1.34× → 1.23× host** (and *beat* host on moderate-critic draws), NOT 58×. The 58× of the SC-deploy is a *different, more severe* failure (it additionally removed the load-bearing perception), not a deeper wall in the same place.

4. **The one shortcut with a genuine dendritic *flavor* is #9 (the place code), and even there the operative blocker is a point-neuron READ-OUT-regime problem, not the dendrite per se.** The `place_sensors → place` self-org cannot make many sparse, location-selective fields at nav scale (a real, Mikulasch-Priesemann-adjacent point-neuron limit), AND the all-or-none coincidence-plateau critic read-out has only two reachable regimes (under-discriminating vs over-clamping). The *named fix* for the read-out half is a **graded rate read-out** (point-neuron-feasible), not a dendrite. The selective-place-code half is the only thing in the whole family where the dendritic substrate could plausibly earn its keep — and it is NOT what makes the actor go silent.

**The load-bearing shortcut is the actor-drive half (#6 orienting AS the actor's drive, compounded by the missing loop closure), NOT the reward/value computation (#7/#8) and NOT the place code (#9).** The scramble localization said "reward/drive, not orienting," and the mechanism is: with the host orienting removed, the actor has no sustained drive because (a) the spiking-SC orienting current is too weak to fire the open cortico-BG-thalamic cascade on its own, and (b) the reentrant `thal→cortex` loop that would let the action self-sustain is OFF.

**Honest top-line (answering the pre-registered fork):** the nav loop is **closable on point neurons** — by closing the reentrant loop (turn `thal→cortex` on as the sustain), keeping a neural perception drive into the actor, and tuning the operating point (the exact #4 playbook: leak + N-scaling + the homeostasis f-I enabler). It is **NOT** the one place a real substrate wall appears. The only residual with a dendritic flavor is the *selective place code* (#9), and that is a SEPARATE, well-localized boundary that does not gate the actor-silence. **Expect a documented honest-negative on absolute parity with the host** (a clean spiking closed-loop will pay a finite operating-point cost vs a zero-noise host scaffold, exactly as #4 ended at 1.16× and Stage C ended at ~1.23×) — and per BRAIN-BASED-ONLY that characterized cost IS the deliverable.

---

## 1. DIAGNOSIS — credit-assignment-dendritic vs loop-stability-point-neuron

### 1.1 The decisive evidence: WHERE the loop is open

The SC-deploy NO-GO's own root-cause section says the actor "fires in the warmup window then drops to ~zero" and reaches the goal **8/1800** steps (host: 822). To localize *why*, trace what drives the actor (`cortex_X`) in each config:

| config | cortex drive source(s) | reentrant `thal→cortex`? | result |
|---|---|---|---|
| host flagship | host heuristic **800 pA** into goal-reducing `cortex_X` (`:6385`) | OFF (default) | navigates (host floor 2.0) |
| prior "passing" neural A/Bs (Stage C, N9 milestone) | **place-goal-readout perception** + (sometimes) host heuristic | OFF | navigates (~1.23–1.34× host) |
| **SC-deploy NO-GO** (`--spiking-sc`) | **only** `sc_map → cortex_X` (`heuristic_strength=0`, host reward zeroed, **no place-goal-readout**) | OFF | **actor silent, 58×** |

Three facts, all verified in code, make the diagnosis unambiguous:

1. **The reentrant `thal_X → cortex_X` closure is OFF.** It exists (`g11_bg_runner.py:1985-1989`, weight 5.0, density 0.5, gated by `enable_cluster_a_closed_loop`) but is default-OFF and NOT enabled by `--spiking-sc`. Catalog **A.05** ("Reentrant cortico-BG-thalamo-cortical loops"): *"Selection is an emergent property of the entire reentrant network."* Biology SUSTAINS an action through cortex→BG→thalamus→**back to cortex**. With that return arc open, the cortex must be driven externally **every step** — it has no way to hold an action once the external drive drops.

2. **The host heuristic IS the actor's drive (#6 is the load-bearing shortcut, not #7/#8).** `h_drive = 800·h_strength` injected straight into the goal-reducing `cortex_X` pools (`:6385-6411`) is what fires the actor in the host config. Removing it (`heuristic_strength=0`) removes the actor's drive. The reward/value loop (#7/#8) modulates *plasticity*; it does not itself drive the cortex. So when #6 is retired, the actor has no replacement drive unless the spiking SC (or a neural perception) supplies it.

3. **The scramble control proves the spiking-SC orienting is NOT carrying the failure.** scramble (scrambled retinotopy → orienting meaningless) ≈ SC-on (116.7 vs 117.5, within 1%). If the orienting were the load-bearing actor-drive, scrambling it would change the outcome. It does not — because the orienting arm is too weak to drive the open cascade either way. The dominant variable is the absence of a sustained actor drive, which the scramble (which leaves the reward/SNc/critic loop intact but the orienting meaningless) does not restore.

**This is a loop-CLOSURE + operating-point failure.** Credit assignment (dendritic) is about *routing error to the right synapse across layers*. Here the agent is not failing to learn the right weights — the agent's **actor is not firing at all**, so there is no behavior to credit. A silent actor is an upstream drive/loop-gain problem, categorically before the credit-assignment question.

### 1.2 Why it is NOT dendritic / credit-assignment

- **The corticostriatal credit assignment is single-layer and already works.** The actor is `cortex_X → str_D1_X → gpi_X → thal_X → motor_X` (catalog **A.04**, "implemented... aligns with textbook"). The DA-gated three-factor rule updates ONE synapse class (`cortex_X → str_D1_X`) from a global δ — this is exactly the shallow, biologically-faithful credit assignment (Schultz RPE → cortico-striatal LTP, catalog **C.30**) that the point-neuron substrate does natively. The boundary ledger audit (`2026-06-20-boundary-ledger-dendritic-audit.md`) tested a dendritic credit-assignment toy (`2026-06-19-dendrite-credit-assignment-toy-stage1`) and found the **single-layer actor has nothing to route** — i.e. there is no deep credit-assignment problem here for a dendrite to solve.

- **The reward/value organs do not need a dendrite.** The reward burst (#7) is an excitatory PPN→SNc afferent (catalog **C.33**, "small PPN region projecting to the dopamine pool" — point-neuron). The value subtraction (#8) is a striosome GABA_B/GIRK conductance at the SNc membrane (catalog **B.07**, **C.30**) — point-neuron, owner-approved, validated. The δ=r−V is computed by two spiking populations and passes the Stage-B gates. None of this is a dendritic computation.

- **The literature shows spiking point-neuron CBGT loops DO sustain reward-driven navigation/decision policies.** Dunovan–Verstynen-style biologically-constrained spiking cortico-basal-ganglia-thalamic models learn action selection via dopamine-dependent plasticity at cortico-striatal synapses driven by RPE, and **tune the speed-accuracy tradeoff to maximize reward rate** (biorxiv 2024.05.21.595174). Point-neuron CBGT loops are a working substrate for exactly this loop. The project's failure is its specific *open-loop, perception-stripped, untuned* configuration, not the substrate.

### 1.3 The one dendritic-flavored residual: #9 the place code (but it is a read-out regime, not the actor-silence)

`#9` (the host Gaussian `vs_place_context`) is the only shortcut whose neural replacement hits something that *looks* like the project's dendritic family. The sparsify BOUNDARY (`2026-06-19-place-code-sparsify-default-BOUNDARY.md`) is precise: sparsification FIXES the value-LEARNING (`w_near/w_far` 1.01 → 1.91×) but the δ does not cross the 1.3 bar because (a) the self-org `place` pool cannot form **many** distinct, location-selective sparse codes from heavily-overlapping egocentric landmark sensors at nav scale (the Mikulasch-Priesemann point-neuron limit — a genuine substrate boundary), and (b) the **all-or-none coincidence-plateau read-out** has only two reachable regimes (under-discriminating at low weights / over-clamping the SNc at high weights), neither of which grades.

But two things keep this from being the answer to the actor-silence:
- **It is the CRITIC's afferent, not the ACTOR's drive.** A weak/flat critic degrades the *quality* of the δ (the value baseline #8); it does not silence the actor. Stage C showed the neural critic δ is *competitive* (1.23×) with the host scaffold even when the place code is imperfect — i.e. an imperfect place code costs a fraction, not 58×.
- **The named fix is point-neuron, not dendritic.** The BOUNDARY doc itself prescribes a **graded rate read-out** (so a modest near>far weight gradient → a modest near>far critic rate → a graded GABA_B δ, without the over-clamp) as the read-out fix — that is a point-neuron operating-point change. The dendrite would only be needed for the *selective-place-code* half (carving many selective fields), which is a separate, deferred, owner-call increment and is NOT on the actor-silence critical path.

⇒ **#9 is a real but separate boundary (a place-code-selectivity + read-out-regime problem), and even it is mostly point-neuron-closable. It does not explain the actor going silent.**

### 1.4 Separating the four shortcuts by load-bearing-ness

| # | shortcut | role in the loop | is it the actor-silence cause? | dendritic? | point-neuron-closable? |
|---|---|---|---|---|---|
| **#6** | host Manhattan orienting heuristic | **the actor's drive** (800 pA into cortex) | **YES — load-bearing** (its removal starves the actor; compounded by the open `thal→cortex` loop) | no | **yes** (close the reentrant loop + a neural perception drive + op-point) |
| **#7** | host distance reward | the reward burst `r` onto SNc | no (modulates plasticity, not drive; organ validates) | no | **yes** (PPN→SNc, catalog C.33; already built + Stage-B GO) |
| **#8** | host EMA value baseline | the value `V` subtracted at SNc | no (Stage C competitive; modulates δ quality) | no | **yes** (striosome GABA_B critic; built; δ graded on moderate draws) |
| **#9** | host Gaussian place code | the critic's afferent | no (degrades δ quality, not actor drive) | **flavor only** (selective-field carving is the dendritic part) | **partly** (read-out fix is point-neuron; selective code is the deferred dendritic call) |

---

## 2. RANKED OPTIONS per shortcut (mechanism · biology · point-neuron vs dendritic · convert-vs-honest-negative · failure mode)

### #6 — Replace the host orienting heuristic AS the actor's drive

The host heuristic does two jobs the neural replacement must both cover: it (a) **computes the goal direction** (perception/orienting) and (b) **delivers a strong sustained drive into the actor cortex** that keeps the cascade firing. The SC-deploy failed because the spiking SC covered (a) at the de-risked op-point but did not deliver enough of (b), and the loop had no self-sustain to compensate.

| Option | Mechanism | Biology / catalog | Point-neuron vs dendritic | Convert vs honest-negative |
|---|---|---|---|---|
| **★ 6-A (RECOMMENDED first) — close the reentrant loop + boost the SC→cortex drive** | turn ON `enable_cluster_a_closed_loop` (the `thal_X → cortex_X` return arc, A.05) so a selected action SELF-SUSTAINS via the cortico-BG-thalamic loop, AND raise the `sc_map → cortex_X` weight (currently merged-tuned 18) / SC-bump gain so the orienting current actually fires the cascade | catalog **A.05** (reentrant loops, "selection is an emergent property of the entire reentrant network"); **A.04** (GPi disinhibition WTA, implemented); Dunovan-Verstynen spiking CBGT (loops sustain policy) | **point-neuron** (both arcs exist; runner-flag + a weight tune) | **CONVERTS (expected).** The single most likely fix — the loop is *open* in the failing config. Failure mode: an over-strong `thal→cortex` loop latches an action (won't re-select on goal change) → tune the loop weight + lean on the #4 leak (`sel_recurrent_weight` already a knob) |
| **6-B — keep a neural PERCEPTION drive into the actor (don't strip the place-goal-readout)** | the prior "passing" A/Bs navigated because `place-goal-readout` (the learned goal-direction perception) drove the actor; deploy the spiking SC ALONGSIDE it (orienting), not as a replacement for the actor's perception drive | the project's own re-classification: "the agent navigates via its goal-direction perception (place-goal-readout / the spiking SC)" (N9 assessment); biology has BOTH SC orienting AND cortical/hippocampal goal-direction drive | **point-neuron** (the place-goal-readout is an existing learned plastic pathway) | **CONVERTS.** This is what made Stage C competitive (1.23×). The SC-deploy's mistake was treating the SC as the *only* actor drive. Failure mode: the place-goal-readout is itself a perception scaffold (#9-adjacent) — honest about which perception stays host |
| 6-C — graded urgency / tonic exploration drive on the actor (the #4 lever) | add the validated Cisek urgency ramp + finite-size N-scaling to the actor pools so a weak orienting current still crosses the commit bound | Cisek-Thura urgency-gating (PMC3042674); the #4 cost-reduction plan (LEVER 2 urgency, LEVER 3 N-scaling) | **point-neuron** (the spiking_wta decision layer already has these knobs, default-on since #4) | **CONVERTS (compounding).** Lifts the weak-drive silent-commit. Failure mode: over-urgency → premature/wrong commits (guard with thal-winner alignment, as #4 did) |
| 6-D — fully-spiking SC salience map replacing the host centroid read | route the orienting off a spiking SC salience map (not the numpy `sc_salience_offset_from_image` centroid) | catalog **H.24/H.25** (SC saccade burst / OPN); the N1 SC is already spiking for the bump | **point-neuron** | partial-CONVERT; the perception centroid is a separately-scoped shortcut. Failure mode: the documented co-residence starvation (organ fires weaker co-resident) — already cleared at Step-0 |

**Recommendation for #6: 6-A + 6-B + 6-C composed** — close the reentrant loop (the structural fix), keep a neural perception drive into the actor (don't strip it), and apply the #4 urgency/N-scaling so the orienting actually crosses the bound. Expect this to recover navigation; expect a finite residual cost vs host (honest-negative on absolute parity).

### #7 — Host distance reward → spiking reward burst (already largely converted)

| Option | Mechanism | Biology / catalog | Point-neuron vs dendritic | Convert vs honest-negative |
|---|---|---|---|---|
| **★ 7-A (already built + GO in isolation) — `reward_us` PPN-like excitatory US→SNc** | the spiking `reward_us` pool (40 cells) fires on perceived goal-proximity/contact and bursts the SNc, replacing the host `snc_reward_gain·max(0,reward)` write | catalog **C.33** ("small PPN region... projecting to the dopamine pool, receiving sensory cue inputs") — verbatim prescription; Watabe-Uchida 2012 (SNc gets strong excitatory drive); Eshel 2015 (the subtraction is separate) | **point-neuron** | **CONVERTED (organ-level).** RPE battery PASS: corr(ecc, reward_us) −0.989, omission dip, lesion-clean. Failure mode: NONE at the organ level. The honest gap is **behavioral load-bearing** — the gridworld is orient-solvable so the reward doesn't *change* nav (documented). Closing that needs a harder, reward-load-bearing task (a separate arc), NOT a substrate fix |

**Recommendation for #7: it is converted as a mechanism.** The residual is the orient-solvable task not stressing it — a task-design item, not a substrate boundary. Do not spend substrate effort here; the deliverable is the validated mechanism + the honest "not behaviorally load-bearing in this gridworld" note.

### #8 — Host EMA value baseline → spiking striosome critic (built; competitive)

| Option | Mechanism | Biology / catalog | Point-neuron vs dendritic | Convert vs honest-negative |
|---|---|---|---|---|
| **★ 8-A (built) — striosome GABA_B/GIRK critic subtracting V at the SNc membrane** | the `striosome_value` MSN critic learns V via the coincidence plateau on the place volley and subtracts it through a real GABA_B conductance (E_K −90 mV) at the KCC2-lacking SNc | catalog **B.07** (patch/striosome→SNc), **C.30** (actor-critic: striosome = state-value), Eshel 2015 (VTA GABA = the subtraction); owner-approved sim/ edits | **point-neuron** | **CONVERTS / competitive.** Stage C: neural δ vs host scaffold **1.23×**, BEATS host on moderate-critic draws. Failure mode: the GABA_B operating point — a FIXED prop can't serve the draw-variable critic rate (binary-clamp on hot draws). Fix = the homeostasis/normalization op-point (below) |
| 8-B — critic rate normalization (the f-I enabler + FS inhibition + divisive norm) | hold the critic in the physiological 1-20 Hz band across draws so a fixed GABA_B prop gives a stable graded δ | catalog **B.06** (PV-FSI feedforward inhibition); Carandini-Heeger divisive normalization; the **homeostasis enabler** (CYCLE 208-209) that restored the SNc f-I; the **`input_divisive_norm` primitive** (CYCLE 92/294) | **point-neuron** (all existing) | **CONVERTS (op-point).** The named fix for the binary-δ. Failure mode: an all-or-none plateau is a poor target for *dividing* (B-4 honest analysis) — the grading lives in the WEIGHTED plateau + rate clamp, not in dividing the plateau |
| 8-C — a proper bootstrapping TD critic (γV(s′)−V(s)) | the full TD critic so V is a learned bootstrap, not a one-step EMA | catalog **C.30/C.31** (bootstrapping vs Monte Carlo); the standalone A-CSC TD cue-shift GO (`2026-06-10-N9-TD-cue-shift-A-CSC-GO.md`) | **point-neuron** (standalone GO, lesion-clean 3/3) | **CONVERTS standalone; merged is a BOUNDARY** (`2026-06-19-merged-TD-cueshift-opsearch-BOUNDARY.md`: r=-0.719 reached but the cue-pathway lesion doesn't discriminate on the merged bridge — a merge-engineering issue, a ~66 Hz SNc cue-onset transient; named fix = SNc onset-recovery/adaptation, config-level). Failure mode: the merged SNc excitability |

**Recommendation for #8: it is converted and competitive; the residual is the GABA_B operating point** (binary-clamp on hot draws), closable by the rate-normalization op-point (8-B, all existing primitives). Expect a documented small residual cost vs the host's smooth-analog V.

### #9 — Host Gaussian place code → self-org spiking place code (the dendritic-flavored one)

| Option | Mechanism | Biology / catalog | Point-neuron vs dendritic | Convert vs honest-negative |
|---|---|---|---|---|
| **★ 9-A (RECOMMENDED) — graded rate read-out of the critic (replace the all-or-none plateau)** | a graded rate read-out that scales smoothly with V, so a modest near>far weight gradient → a modest near>far critic rate → a graded GABA_B δ, without the over-clamp | the BOUNDARY doc's own named fix; Carandini-Heeger; rate-coded value (vs the binary plateau) | **point-neuron** | **CONVERTS the read-out half.** Failure mode: a graded rate read-out is noisier than the all-or-none plateau (the project's recurring rate-code-wall — possibly liftable by the population-code lever that lifted the conversational read-out 47%→100%) |
| 9-B — sparser self-org target (separable fields) | drive the place self-org to a lower sparsity so fields are separable | Hollup 2001 / Dupret 2010 (goal over-representation); place-cell sparsity ~1-5% | **point-neuron** (an op-point on the afferent threshold-WTA) | partial; the sparsify probe showed W=10 hits 6% sparsity but the FS-PING-open *read* regime is still non-selective (a few dominant cells fire everywhere) — the deeper limit |
| 9-C — selective place fields via the dendritic substrate | per-cell nonlinear input integration to carve many selective fields from overlapping egocentric sensors | the Mikulasch-Priesemann point-neuron limit (the conversational whitening wall); the deferred D2 dendritic substrate | **DENDRITIC** (the one genuinely-dendritic option in the whole family) | **HONEST-NEGATIVE candidate.** This is where the deferred dendritic substrate could earn its keep — but it is a SEPARATE, deferred owner-call, NOT on the actor-silence path. Failure mode: months-scale; the D2 dendritic toys to date were NEGATIVE on their two named jobs |

**Recommendation for #9: the read-out fix (9-A) is point-neuron and closes the δ-quality residual; the selective-place-code (9-C) is the only dendritic call, and it is deferred and off the critical path.** Honest-negative framing: a self-org place code that does not match the hand-tuned Gaussian's position-specificity at nav scale IS a documented substrate cost (the host Gaussian stays the better-δ scaffold), exactly as the BOUNDARY doc already concluded.

---

## 3. Reusable project machinery (point the de-risk at these proven primitives)

| Primitive | What it gives the nav loop | Where / status |
|---|---|---|
| **Reentrant `thal→cortex` closure** | the actor SELF-SUSTAIN (the missing loop arc — the #1 fix) | `g11_bg_runner.py:1985-1989`, `enable_cluster_a_closed_loop` (built, default-OFF) |
| **Spiking BG WTA / commit-burst (#4)** | the action EMERGES from spiking competition (default-on); the leak + N-scaling + urgency op-point knobs | `g11_bg_runner.py:2094-2203`; `2026-06-19-spiking-decision-default-on-GO.md` (1.16× host) |
| **The #4 cost-reduction PLAYBOOK** | the closable-vs-fundamental decomposition + the cheap-first lever sequence (leak → urgency → N-scaling) + the STOP/DEPLOY criterion | `2026-06-19-spiking-decision-cost-reduction-plan.md` — directly transferable template |
| **`reward_us` PPN-like US→SNc (#7)** | the spiking reward burst (host write dropped) | `build_bg_brain_regions(spiking_reward_us=True)`; RPE-battery GO |
| **Striosome GABA_B/GIRK critic (#8)** | the spiking value subtraction at the SNc membrane | `enable_neural_critic`; Stage-B GO; owner-approved sim/ edits |
| **The homeostasis f-I enabler (CYCLE 208-209)** | restores the SNc reward-burst f-I (446 Hz / 5.47× vs broken 111 Hz) — the critic-rate normalization | post-hoc `enable_homeostasis=True` on `snc`/`reward_us`; `2026-06-18-organ-lift-homeo-generalize-derisk.md` |
| **`input_divisive_norm` (CYCLE 92/294)** | Carandini-Heeger divisive normalization on a flagged pool — the critic-rate-band fix; just proved as the S5 read-out fix | `sim/regions.py:240` + `config.py:440` + `bridge.py:6048` (guarded, byte-clean off) |
| **NEF thresholded cleanup** | a placed-threshold readout that discretizes a rate to a decision (the conversational cleanup precedent for a graded-rate critic read-out) | `2026-06-05-composer-cleanup-NEF-GO.md` |
| **`td_value_critic` / the standalone A-CSC TD cue-shift** | the bootstrapping TD critic (#8-C), lesion-clean standalone | `2026-06-10-N9-TD-cue-shift-A-CSC-GO.md` |
| **Neuromodulator subsystem (`from_reward` / `from_region_firing_signed`)** | the DA broadcast derived from SNc firing (the three-factor gate) | `sim/neuromodulators.py:736` |
| **Eligibility-trace machinery (C.29)** | the actor's three-factor (eligibility × dopamine) plasticity | `g11_bg_runner.py` actor STDP path |
| **The merged-bridge `co_resident_nav_critic` integration** | the nav critic already wired onto the merged "one brain" with the homeostasis enabler | `nav_conv_merged_bridge.py:508` (CYCLE 209) |
| **`--deterministic-selforg`** | reproducible place-code draws (so a multi-seed A/B is attributable) | `sim/config.py` `deterministic_transpose_matvec`; owner byte-reviewed |

**The de-risk needs essentially NO new sim/ machinery** — every lever is an existing flag, an existing primitive, or a runner-side weight tune. The one structural change (turning `enable_cluster_a_closed_loop` ON) is an existing builder flag.

---

## 4. The recommended cheap-first de-risk

**Goal:** decisively separate "the loop is open / under-driven (closable)" from "the substrate cannot sustain the loop (a wall)" — the exact fork the dispatch asks. The #4 playbook is the template: decompose closable-vs-fundamental, then a cheap lever sequence.

**Phase 0 — CPU/numpy, the decisive control (~minutes, leave the GPU alone):**
The single cheapest experiment that answers the fork is the **loop-closure A/B on the failing config**: re-run the SC-deploy `--spiking-sc` config but with `enable_cluster_a_closed_loop=True` (close the `thal→cortex` reentrant arc) at grid-8, seed 42, ~120-step smoke.
- **If the actor stops going silent** (motor counts stay non-zero past warmup, distance drops) → the failure was the OPEN LOOP (loop-stability, point-neuron). This is the expected outcome.
- **If the actor still goes silent** → the loop closure is not the cause; escalate to Phase 1.
This one A/B distinguishes the two hypotheses at the lowest possible cost, before any GPU spend.

**Phase 1 — CPU, the drive-decomposition probe (the #4 "failing-decision profile" analogue):**
Instrument the actor drive at each cortex pool over the SC-deploy run (the SC orienting current vs the threshold to fire the cascade), and the cortex/thal/motor firing decomposition, in four arms: (a) SC-only (the NO-GO config), (b) +closed-loop, (c) +closed-loop +place-goal-readout (keep a neural perception drive), (d) +closed-loop +place-goal-readout +urgency/N-scaling (#4 levers). Report, per arm, the per-phase distance SUM + the fraction of steps the actor fired. This is the direct port of the `_n6_refine_analyze.py` decision-profile diagnosis to the nav-loop and tells you which lever recovers the drive.

**Phase 2 — GPU, the cheap lever sequence (only the arms Phase 1 says are alive):**
Apply the #4 STOP/DEPLOY discipline: tune the closed-loop weight (avoid latching), the `sc_map→cortex_X` gain, the urgency peak, and N-scaling, grid-8 seeds 42/43/44, stopping when two consecutive levers each yield <0.15 absolute SUM improvement OR the SUM is within ~25% of the host floor with the actor firing ≥90% of steps. Then a 6-seed grid-32 A/B vs the host scaffold (the deploy gate).

**The honest-negative contract (BRAIN-BASED-ONLY):** if, after the lever sequence, the closed-loop spiking nav still underperforms the host scaffold by >25% (the deploy bar), that characterized cost IS the deliverable — a clean, biology-faithful, fully-spiking nav loop that pays a finite operating-point price vs a zero-noise host scaffold (the same class of result as #4's 1.16× and Stage C's 1.23×). Do NOT fake a conversion or strip the perception to hit a number. The moat is N/A for nav, but the brain-based-only honest-negative framing is the bar.

**Why CPU-first:** the decisive loop-closure A/B (Phase 0) is a tiny smoke that runs on numpy in minutes and answers the fork; only the lever-tuning + the 6-seed deploy gate need the GPU. This respects the standing "research/cheap-first before GPU resources" gate.

---

## 5. The anti-cheat controls (the localization the SC NO-GO already used)

The SC NO-GO modeled the right rigor; carry it forward and extend it:

1. **Scramble (retinotopy)** — the NO-GO's own control: scrambling the SC map must change the outcome IF the orienting is load-bearing. (In the NO-GO it did NOT change it — correctly localizing the failure away from orienting. In a *recovered* config, a scramble should regress, proving the recovered nav uses the orienting.)
2. **Lesion the reentrant arc** — if Phase 0 recovers the actor by closing `thal→cortex`, lesion that arc (zero the `thal_X→cortex_X` weights) post-recovery: the actor must go silent again. This proves the loop closure (not something else) is carrying the sustain — the decisive control for the "loop-stability" claim.
3. **Lesion the perception drive** — zero the `place-goal-readout → cortex` (or the SC→cortex) edges: nav must collapse. Proves the actor's drive is the neural perception, not a back-channel.
4. **Host positive control** — the host scaffold arm (heuristic + host reward) is the upper-bound reference (the floor ≈ 2.0); every neural arm is measured against it. (Already in the gate as the `motor`/`thal` host-argmax oracle + the host-reward control.)
5. **Provenance / coord-free** — assert no `(x,y)/(gx,gy)/Manhattan` enters the SC drive or the reward (the SC reads only the egocentric render; the reward rides on pixels). Already an established bar (N5 meets it).
6. **Decision-path primary ≥90% (the #4 anti-cheat)** — the recovered nav's decision must come from the `commit_X` burst, NOT the sel-lean argmax fallback (a recovery that just raises the fallback% is rejected — it would re-introduce the host-argmax shortcut).
7. **Per-phase localization (the #4 P0-P1 vs P2-P3 split)** — report the stable-goal vs post-goal-change phases separately; a closed loop that latches will show P2-P3 blow-out (the hysteresis the #4 leak addresses), distinguishing "can't sustain" from "can't re-select."

---

## 6. Honest top-line

**The nav reward/value/sustained-control loop is closable on point neurons. It is NOT the one place a real substrate wall appears, and the wall it does have a flavor of (the selective place code, #9) is (a) NOT what makes the actor go silent and (b) mostly point-neuron-closable via a graded-rate read-out, with the genuinely-dendritic selective-field-carving a separate, deferred, off-critical-path call.**

- **The fork's answer: LOOP-STABILITY / OPERATING-POINT (point-neuron), the same family as #4** — not credit-assignment-dendritic. The actor goes silent because the SC-deploy config (i) removed the host heuristic that WAS the actor's drive (#6), (ii) removed the place-goal-readout perception that the prior "passing" runs actually navigated on, and (iii) left the `thal→cortex` reentrant self-sustain arc OFF, so the open cortico-BG-thalamic cascade had nothing to keep the actor firing. The scramble control already localized the failure to the reward/drive half and away from the orienting; the mechanism is the open + under-driven actor loop.
- **The load-bearing shortcut is #6 (the orienting heuristic AS the actor's drive), compounded by the missing loop closure — NOT #7/#8 (the reward/value organs, which validate in isolation and are competitive when the perception is retained).** #9 (the place code) is a separate δ-quality boundary with a real dendritic flavor on its selective-field half, but it degrades quality, not actor drive.
- **The cheapest decisive test** is the Phase-0 loop-closure A/B (turn `enable_cluster_a_closed_loop` on in the failing config, CPU smoke): if the actor stops going silent, the diagnosis is confirmed and the fix path (close the loop + keep a neural perception drive + apply the #4 urgency/N-scaling op-point) is the conversion.
- **Expect a documented honest-negative on absolute PARITY** (a fully-spiking closed nav loop will pay a finite operating-point cost vs a zero-noise host scaffold — Stage C's ~1.23× and #4's 1.16× are the precedents) — and per BRAIN-BASED-ONLY that characterized cost is the scientific deliverable. This is NOT a "the substrate cannot do it" wall; it is a "the substrate pays a small, biology-faithful price, and the loop was misconfigured (open + perception-stripped) in the test that read as 58×."

---

## Citations

**Project findings (read in full):**
- `research/findings/2026-06-19-nav-spiking-sc-deploy-NO-GO.md` (the 58× NO-GO + scramble localization)
- `research/findings/2026-06-19-nav-spiking-sc-deploy-prep.md` (the deploy config: `heuristic_strength=0`, host reward zeroed, no place-goal-readout)
- `research/findings/2026-06-18-merged-neural-place-code-delta-probe-NEGATIVE.md` + `2026-06-19-place-code-sparsify-default-BOUNDARY.md` (#5/#9: the dense-field root cause + the all-or-none read-out regime + the graded-rate-read-out named fix)
- `research/findings/2026-06-10-N9-fully-spiking-reward-loop-MILESTONE.md` (the reward loop made spiking + the "nav A/B insensitive to the reward pathway" finding)
- `research/findings/2026-06-10-N9-nav-deployment-stageB-PASS-seed42.md` (Stage C: neural δ vs host scaffold 1.34×→1.23×, competitive; the GABA_B operating-point diagnosis)
- `research/findings/2026-06-10-N5-reward-CLOSED-and-navigation-fully-biologized.md` (the "orient-solvable task can't validate the reward" lesson; the RPE battery)
- `research/findings/2026-06-10-N9-spiking-snc-current-state-assessment.md` (the full N9 map: organs validate, nav A/B insensitive, the behavioral-load-bearing residual)
- `research/findings/2026-06-10-N9-spiking-reward-and-critic-normalization-research.md` (C.33 reward afferent + the FS/divisive critic-rate analysis)
- `research/findings/2026-06-19-spiking-decision-default-on-GO.md` + `2026-06-19-spiking-decision-cost-reduction-plan.md` (#4: the "wall→point-neuron-tunable" precedent + the closable-vs-fundamental playbook)
- `research/findings/2026-06-19-merged-TD-cueshift-opsearch-BOUNDARY.md` (#8-C: standalone TD GO, merged a merge-engineering boundary not a substrate one)
- `research/findings/2026-06-20-boundary-ledger-dendritic-audit.md` (the dendritic-debt audit: 0 dendritic boundaries block a shipped capability; the single-layer actor has nothing to route)
- `research/findings/2026-06-18-organ-lift-homeo-generalize-derisk.md` (CYCLE 208-209: the homeostasis f-I enabler restoring the SNc burst)

**Catalog (`sim-catalog/references/feature-catalog.md`):**
- **A.04** (BG output disinhibition WTA — implemented, aligns with textbook); **A.05** (reentrant cortico-BG-thalamo-cortical loops — "selection is an emergent property of the entire reentrant network"; the project's `thal→cortex` closure is the default-OFF realization)
- **C.30** (actor-critic: the project is a "two-actor, no-critic" architecture, policy-improvement-only, can converge to local optima; the critic is the bolt-on); **C.31** (bootstrapping vs Monte Carlo); **C.22** (Schultz RPE); **C.33** (PPN→DA reward afferent — the #7 prescription)
- **B.06** (PV-FSI feedforward inhibition — critic rate control); **B.07** (striosome→SNc — the #8 value subtraction)
- **G.16/G.17** (drift-diffusion bound / LIP accumulator — the #4 decision layer); **H.24/H.25** (SC saccade burst / OPN — the orienting)

**Literature (WebSearch):**
- Dunovan, Verstynen et al. — biologically-constrained spiking cortico-basal-ganglia-thalamic model learning action selection via dopamine-dependent cortico-striatal plasticity, tuning the speed-accuracy tradeoff to maximize reward rate (biorxiv 2024.05.21.595174) — point-neuron CBGT loops DO sustain reward-driven policy.
- Actor-Critic RL with stability guarantee (arXiv 2004.14288) — "errors in the critic's policy can cause destructive feedback and divergence," addressed with a slower-timescale target — the loop-stability framing for a spiking actor-critic.
- Spiking SNN actor-critic with temporal coding + reward-modulated plasticity (Moscow Univ. Phys. Bull. 2024) — the critic computes value change from spike timing, weights via reward-modulated STDP — point-neuron-feasible.

**Source URLs:**
- https://www.biorxiv.org/content/10.1101/2024.05.21.595174.full.pdf
- https://arxiv.org/abs/2004.14288
- https://link.springer.com/content/pdf/10.3103/S0027134924702400.pdf
- https://www.sciencedirect.com/science/article/pii/S0166223625001924 (interacting CBGT loops shape behavioral control)
