# Navigation reward / value / SC-orienting / place CLOSED LOOP — the deep-research gate (2026-06-27)

**Type:** READ-ONLY deep-research + reference-catalog gate (the project's standing "research-first at a roadblock /
new direction" move). NO code written, NO `sim/` edit, NO experiments run. Single deliverable = this doc. Every
load-bearing claim was trust-but-verified against the actual finding text + the current `g11_bg_runner.py` / merged-gate
defaults + the catalog. Where a prior verdict is stale or was a false localization, that is flagged.

**Scope (per the dispatch):** the navigation reward / value / SC-orienting / place CLOSED loop — the one cluster the
comprehensive close-out plan (`2026-06-27-comprehensive-shortcut-inventory-burndown-plan.md`, SHA 5ad2f453) flagged as
RESEARCH-GATE-NEEDED (its N-2 value / N-3 reward / N-4 SC-orienting / N-5 place, treated as ONE closed-loop family). The
mandate: scope whether/how the loop closes to spiking, separating the closeable pieces from any genuinely-irreducible
dendritic-months wall — defaulting to closeable + a SURPASS (the owner believes they are closable), accepting a boundary
ONLY if it truly survives the 4-move SURPASS round.

---

## 0. TL;DR — the headline reconciliation, then the four-piece verdict

**The comprehensive plan's framing of this cluster as "the documented NO-GO (~58×, the actor goes silent)" is STALE,
and the staleness is decisive.** That ~58× number is from `2026-06-19-nav-spiking-sc-deploy-NO-GO.md`. Two arcs since
then have overturned it, and BOTH have LANDED in current code:

1. **CYCLE 1B (2026-06-24, commit in the 1B GO finding) flipped the deployed merged-nav episode loop to the SPIKING
   limbic δ=r−V by default** — `perceived_approach_reward=True` (`g11_bg_runner.py:3346`), `spiking_snc=True` (`:3480`),
   `enable_neural_critic=True` (`:3506`), the host `_V_scaffold` + `snc_reward_gain` write DROPPED. Validated BY FUNCTION
   (the Schultz RPE battery: corr(ecc, reward) = −0.81, SNc-burst 1.45×, reward-lesion vanishes, critic-GABA_B-lesion
   collapses), nav NOT regressed, the conversational no-confab moat byte-frozen under the live spiking-reward + DA
   stressor. **⇒ the nav REWARD (N-3) + VALUE-subtraction (N-2's δ machinery) are SPIKING-DEFAULT-ON on the merged
   "one brain" the production agent builds.**

2. **Burndown-3F (2026-06-24, commit `853599cb`) SURPASSED the SC-orienting closed-loop NO-GO.** The ~58× was a FALSE
   LOCALIZATION: the original scramble control had nothing to scramble because at grid-32 the four corner goals (30+
   cells away) rendered ENTIRELY OFF the truncated 32-px egocentric retina (retina mass = 0.0 → the SC bump was ABSENT),
   AND the host `max()` tie-break resolved the saturated `[40,40,40,40]` accumulator ties to N (a stuck-N policy). With
   both non-biological modeling choices fixed — the **log-polar foveal-magnified render** (`log_polar_retina=True`
   **default**, `g11_bg_runner.py:3866`, 4/4 seeds GO) + the **FIX1 stochastic tie-break** (3/3 seeds GO) — the EXACT
   SAME closed neural-reward→critic→actor loop now SUSTAINS navigation at **~2.4× host (down from ~58×)**, with the actor
   firing through (late-sustain ~0.97) and a 21.8× lesion contrast proving the surpass is load-bearing.

**So the cluster is far closer to closed than 5ad2f453 reads.** The four-piece verdict:

| piece | what is host vs spiking NOW | genuine residual | class |
|---|---|---|---|
| **N-3 REWARD** | spiking by default on the merged agent (CYCLE 1B); CLI standalone defaults to the host oracle | none at the organ level; the gridworld is orient-solvable so the reward is not *behaviorally* load-bearing (a task-design item) | **CLOSEABLE — largely CLOSED** (CLI deployment flip + a load-bearing task) |
| **N-2 VALUE / RPE baseline** | δ=r−V spiking by default on the merged agent; V learned co-resident | the afferent-driven δ is GRADED but WEAK (~1.3×) capped by the position-blind up-state floor; the dendrite Stage-1 graded plateau already reaches δ=1.33 = host ceiling on a critic bridge | **CLOSEABLE** (the graded-plateau read-out is validated; deploy + a value-load-bearing task) |
| **N-4 SC-ORIENTING** | SURPASSED in the LOOP (~2.4× host); but the *spiking SC* itself is `enable_spiking_sc=False` library-default, so the host Manhattan heuristic still ships as the deployed orienting | the residual ~2.4× is the finite-size margin-SNR floor (B-4 family); the named next mechanism is the opponent-axis (FIX3); deploying the spiking SC as the default orienting is a flip | **CLOSEABLE — loop SURPASSED, deployment + margin-SNR opponent-axis remain** |
| **N-5 PLACE code** | host Gaussian `vs_place_context` is the deployed critic afferent; the self-org spiking place code is opt-in | the self-org place fields are NOT location-selective in the read regime (a few cells fire everywhere) → the value δ does not grade; the **graded dendritic plateau read-out CLOSES the read-out half** (δ=1.33 on-bridge); the deeper SELECTIVE-FIELD CARVING is the one genuinely dendritic-flavored residual | **MOSTLY CLOSEABLE** (read-out solved on-substrate); the selective-field carving is the **one accepted-dendritic candidate — but it has a validated point-neuron WORKAROUND, so it does not block** |

**The bottom line:** there is **no genuinely-irreducible dendritic-months wall that BLOCKS closing this loop to
default-on spiking.** The one piece with a real dendritic flavor — carving many sparse, location-selective place fields
from overlapping egocentric landmark sensors (the Mikulasch-Priesemann point-neuron limit) — is (a) NOT on the
actor/sustain critical path, and (b) already has a validated point-neuron-substrate workaround (the host-Gaussian place
code stays the better-δ scaffold; the graded dendritic plateau supplies the graded value read-out the soma can't). Per
BRAIN-BASED-ONLY, the residual ~2.4× nav cost and the ~1.3× value-grade ARE the scientific deliverables (the finite-size
/ position-blind-floor cost the point-neuron substrate pays). The work is dominated by **deployment flips** (make the
spiking forms the SHIPPED default, keeping the host as the documented oracle) + **two small validated builds** (the
opponent-axis margin-SNR remedy; the delayed-reward task that makes the value load-bearing).

---

## MOVE 1 — ISOLATE + QUANTIFY: the precise residual of each of the four

### N-3 REWARD — host distance formula → synaptic SC-proximity approach-reward

- **Host now:** the standalone CLI (`g11_bg_runner.py` argparse, `_nav_gate_merged_run.py:71` `--perceived-approach-reward`
  is `store_true` = off) computes `current_reward_signal = delivered_reward` from a host distance/eccentricity formula
  (`g11_bg_runner.py:~7616-7661`).
- **Spiking now:** the LIBRARY default `perceived_approach_reward=True` (`:3346`) routes the reward synaptically through
  the `sc_rostral → reward_us` PPN-like US→SNc afferent (catalog **C.33**), and the merged production agent inherits it
  (CYCLE 1B). The organ is QUALIFIED-GO in isolation (`2026-06-18-merged-neural-reward-QUALIFIED-GO.md`): corr(ecc,
  reward_us) = −0.989 / −0.81 deployed, omission dip present, lesion-clean.
- **The genuine residual:** NOT a substrate limit — it is (a) a **CLI deployment flip** (the shipped standalone benchmark
  still defaults to the host reward), and (b) a **behavioral-load-bearing gap**: the moving-goal gridworld is
  **orient-solvable** (the SC/orienting/place machinery reaches a per-step-rewarded goal without *predictive* value), so
  the reward does not *change* navigation and the lesion barely moves the score. This is the documented
  `feedback_validate_signal_by_its_function` lesson (N5 reward "passed" a task that didn't exercise it).
- **The truly-irreducible part:** essentially nil. Reward FROM the environment is legit-host (the world delivers it);
  reward COMPUTED by a host distance formula (the brain should appraise proximity) is the shortcut, and the synaptic
  appraisal is built + validated.

### N-2 VALUE / RPE baseline — host EMA `_V_scaffold` → spiking striosome critic

- **Host now (CLI):** `reward_ema` + `_V_scaffold = max(0, reward_ema_pre)` in standalone runs.
- **Spiking now (merged):** `enable_neural_critic=True` default — the striosome GABA_B/GIRK critic (catalog **B.07**,
  **C.30**) learns V via DA-gated STDP co-resident and subtracts it at the SNc membrane. **V IS learned on the one brain**
  (`2026-06-18-merged-navcritic-valuetrain-BOUNDARY.md`): the plastic `vs_place_context→striosome_value` weight grows ~20×
  every seed and FLIPS the critic from far-dominant (0.5) to goal-dominant (~2.0), 6/6, lesion-confirmed.
- **The genuine residual (quantified, a BOUNDARY):** the afferent-driven δ=r−V is GRADED in the right direction
  (predicted@goal < unpredicted@far) but the gap is **WEAK (~1.32 mean, 6/6, σ~0.04)** — RIGHT AT the 1.3 bar, not the
  4–19× direct-drive ceiling. The precise cause is STRUCTURAL, not tuning: the dense, NON-plastic convergent up-state arm
  (`vs_place_drive`, needed to fire the cold MSN-D1 so the STDP has a post-spike) is **position-blind** — it fires the
  critic ~19 Hz at the FAR location too, delivering GABA_B onto the SNc at far, compressing the (V_goal − V_far) contrast
  the subtraction reads. The cap is the position-blind up-state floor, not a learnable quantity (the far cells stay at
  init; the up-state arm is `plastic=False`).
- **The point-neuron READ-OUT sub-problem (CHARACTERIZED):** the graded-RATE point-neuron read-out is NOT realizable for
  the MSN-D1 critic (`2026-06-20-burndown-9-critic-graded-readout.md`): a linear-summation place input is sub-rheobase at
  any weight (the MSN-D1 membrane reaches only −72 mV vs ~−40 mV rheobase, tested to 30× init weight → 0 Hz critic), while
  the all-or-none coincidence plateau that DOES fire it over-clamps the SNc to 0 (176–239 Hz critic → δ=0.00). The two
  reachable regimes (under-discriminating / over-clamping) bracket the graded window.
- **The truly-irreducible part — already SURPASSED on-substrate:** the graded analog read-out the point-neuron soma
  provably cannot produce (Mikulasch-Priesemann) is realized by the **graded dendritic plateau**
  (`2026-06-20-dendrite-stage1-snc-calibration.md`): on a critic bridge the SNc burst δ reaches **1.33 = the host
  ceiling, 6/6 seeds**, V still a 3-level graded continuum (near 0.13 > mid 0.08 > far 0.014, ~9× near/far, learned by the
  bridge's own reward-STDP), all anti-cheats green, behind a default-OFF byte-reviewed `sim/` flag
  (`enable_graded_dendritic_plateau`, commits `d69cc0ab`/`52dafaeb`). So the value read-out is NOT an open wall — the
  dendritic-plateau form closes it on the substrate. The residual is (a) DEPLOY the graded plateau into the merged critic
  (the `--dendrite-critic` wiring exists, `2026-06-20-shortcut9-dendrite-critic-deploy.md`, deploy table was pending), and
  (b) a task where the value is PROVABLY load-bearing so the deploy is not a confounded NEGATIVE (the deploy into nav was
  a QUALIFIED NEGATIVE — value-lesion ≈ deploy, Δ7.2% — because the task is immediate-reward-solvable; the whole gain over
  the point-neuron baseline was the NMDA on the critic slice, not the value).

### N-4 SC-ORIENTING — host Manhattan heuristic → spiking superior colliculus

- **Host now:** the host Manhattan compare (`if gx > x: cortex_E += HEURISTIC_DRIVE_PA`, `heuristic_strength=1.0`,
  `g11_bg_runner.py:~4003`) is the DEPLOYED orienting; `enable_spiking_sc=False` library-default (`:3821`),
  `sc_tie_break_stochastic=False` library-default (`:3882`).
- **The original ~58× NO-GO — now corrected.** `2026-06-19-nav-spiking-sc-deploy-NO-GO.md` read the spiking-SC closed
  loop as ~58× worse with the actor silent, and the scramble control localized the failure to the "reward/drive half,
  not orienting." **Both readings were artifacts of two non-biological runner-side modeling choices, since fixed:**
  - the **egocentric retina TRUNCATION** (a flat linear `ppc=4` over a fixed 32-px field clips far goals off-image → the
    SC bump is ABSENT → the scramble had nothing to scramble → "scramble ≈ SC-on" was trivially true, NOT evidence the
    reward/drive loop is the cause);
  - the **host N-first tie-break degeneracy** (`max()` resolves `[40,40,40,40]` accumulator-saturation ties to N → a
    stuck-N policy regardless of the goal; `2026-06-20-cascade-north-bias-FIX.md`).
- **Spiking now — the loop SUSTAINS.** With the log-polar foveal render + FIX1 (both default-on at the library level), the
  exact same closed neural-reward→critic→actor loop tracks the moving goal on all phases and holds within ~1 cell at
  **~2.4–3.0× host** (`2026-06-24-burndown-3F-sc-sustained-orienting-surpass.md`, 4/4 seeds; the 21.8× retina-lesion
  contrast + the SCRAM-collapse prove the orienting decode is load-bearing). The reentrant `thal→cortex` self-sustain arc
  (catalog **A.05**) was found to be ALREADY ON in the NO-GO config and demonstrably NOT the silence's cause or fix
  (`2026-06-20-nav-loop-closure-derisk.md` — the deep-research "open loop" premise was falsified by a 490-vs-0 synapse
  count + a grid-32 A/B with <3% sustain difference ON vs OFF).
- **The genuine residual:** (a) the **deployment flip** — `enable_spiking_sc` is still library-default-OFF, so the host
  Manhattan heuristic is the SHIPPED orienting; the spiking SC is validated-opt-in; and (b) the residual ~2.4× is the
  **finite-size margin-SNR floor** (tie-fraction ~0.18–0.20 = ~18–20% of decisions resolved by the fair tie-break draw on
  weak diagonal margins) — the SAME family as B-4's ~16% spiking-decision cost. The named next mechanism is the
  **opponent-axis push-pull** (FIX3: organize the four cardinals as two balanced N↔S / E↔W competitions so a clean 1-D
  margin is extracted per axis); the margin-amplification screen (stronger SC drive) was NEGATIVE (it re-biases, not
  sharpens — `2026-06-20-nav-sc-drive-reorient-derisk.md` + the FIX doc's amplification probe).
- **The truly-irreducible part:** the finite-size margin-SNR floor on an orient-solvable task is irreducible AND honest
  (the same class as B-4 1.16×), but it is small (~2.4×) and not a substrate-cannot-do-it wall — there is no dendritic
  frontier here.

### N-5 PLACE code — host Gaussian field → self-org spiking place code

- **Host now:** `place_drive = ...np.exp(-place_dsq/...)` — a tuned Gaussian over the true (x,y)/(gx,gy)
  (`g11_bg_runner.py:~6563`); `neural_place_selforg=False` library-default.
- **Spiking now (opt-in):** the self-org `place_sensors→place` code COMPOSES on the merged bridge (`neural_place_selforg`,
  SCOPE-GO) and LEARNS a real V gradient when sparsified, but the value δ does NOT cross 1.3 (a BOUNDARY).
- **The genuine residual (a BOUNDARY, precisely localized):** `2026-06-19-place-code-sparsify-default-BOUNDARY.md` —
  sparsification (afferent weight 28→10) FIXES the value-learning root cause (`w_near/w_far` 1.01 → 1.91×, sparsity
  0.46→0.06), **but** in the FS-PING-open operating regime that the value-train and critic read in, the sparse cells are
  **NOT location-selective** (a few dominant cells fire at MANY locations → near/far ensemble overlap cos ≈ 0.42–0.78
  regardless of self-org sparsity), and the all-or-none coincidence-plateau read-out over-clamps when driven hard. Every
  regime lever was exhausted (afferent weight {8..28}, FS-during-self-org, fs→place {8..40}, init V, trials, k {4..20},
  GIRK cap) — none lowered the operative read cos below ~0.42.
- **The truly-irreducible (dendritic-flavored) part — and its workaround:** the DEEPER cause is the
  Mikulasch-Priesemann point-neuron limit: a point-neuron `place` pool cannot form MANY distinct, location-selective
  sparse codes from heavily-overlapping egocentric landmark sensors — a genuinely sparse+selective place code would
  plausibly need per-cell nonlinear dendritic input integration. **This is the ONE place in the whole cluster with a real
  dendritic flavor.** BUT: (a) it is the CRITIC's afferent, NOT the actor's drive — it degrades δ-quality, not actor
  sustain; (b) the host-Gaussian place code (position-specific by construction) stays the validated better-δ scaffold; and
  (c) the read-out half (the all-or-none over-clamp) has a validated point-neuron-substrate workaround — the **graded
  dendritic plateau** reaches δ=1.33 on the substrate (N-2 above). So even the dendritic-flavored piece does NOT block —
  it is a δ-quality / breadth refinement with two existing fallbacks (host Gaussian; graded plateau), not a wall on the
  path to a closed loop.

---

## MOVE 2 — REFRAME via biology: how the brain does each, and where the wrong hypothesis was tested

### N-4 SC-orienting — the SURPASS reframe (the wrong hypothesis WAS tested, and was corrected)

The NO-GO tested *"can a closed neural reward/critic/actor loop SUSTAIN nav?"* and concluded NO — but the loop was being
fed a **dead orienting input** (absent bump) and read out through a **stuck-N** policy. The right question is one stage
upstream: **does the SC's INPUT REPRESENTATION guarantee an eccentric target is still represented?** The decisive biology
(catalog + literature, verified): the intermediate/deep superior colliculus holds a **log-polar / foveal-magnified
retinotopic saccade map** — eccentricity mapped along the rostral-caudal axis with strong foveal magnification; the
periphery is COMPRESSED but ALWAYS represented, never clipped (Ottes–Van Gisbergen–Eggermont logarithmic afferent map;
Hafed lab 2019; human-SC eccentricity work). Catalog **E.04**: topographic maps are *"warped by behavioral importance —
cortical magnification — fovea"* (verified at `feature-catalog.md:1384`). Catalog **H.25**: the SC is the full-hemifield
"where to look next" saccade map; **A.07**: SNr→SC disinhibition gates the saccade out. **A linear, truncated retina is
the non-biological special case** — the biology-faithful SC never truncates. The "sustain" the loop needed was never a
reward/value problem; it was the missing foveal-magnified input representation (+ a fair tie-break = Wang-2002 finite-size
decision noise breaking genuine ties, not the host N-first ordering). The residual margin-SNR is then the second-order
fix (opponent-axis competition / SC-margin amplification).

### N-2 VALUE — the SURPASS reframe (the graded analog read-out is a dendrite the soma can't be)

The point neuron's failure to grade a distributed value is the documented analog/dendritic-computation limit
(Mikulasch-Priesemann). Biology grades value in the DENDRITE: the regenerative NMDA-spike-like plateau is the analog
read-out (Poirazi-Mel; Larkum BAC). The project's **graded dendritic plateau** (`fused_graded_dendritic_plateau`, a
smooth non-saturating logistic) is exactly this, and it reaches the host δ ceiling on-substrate. This is NOT a months-long
dendritic rewrite (that is the full `NeuronModel.TWO_COMPARTMENT` neuron, catalog T3.A) — it is a guarded, byte-reviewed,
already-shipped read-out primitive. **The right reframe for the WEAK merged δ:** the residual is not the read-out (the
plateau grades it) — it is the position-blind up-state FLOOR + the orient-solvable task that doesn't make the value
load-bearing. The biology for the latter is **trace conditioning** (catalog **F.22/F.23**; Hesslow-Yeo 2002; the
eNeuro-2025 NAc-DA-encodes-the-trace-period result): a reward separated from its predictive cue by a CS-free gap, where
the ONLY way to act correctly is to carry a learned value across the gap. The catalog's own 2×2 factorial (TRACE vs DELAY
× value-ON vs value-LESION) is the discriminating design: the TRACE arm needs the value (lesion collapses it); the DELAY
arm (no gap) does not (the immediate-reward control that answers the orient-solvable confound).

### N-5 PLACE — the reframe (selective fields are the dendrite's genuine job, but two fallbacks exist)

Biology carves sparse, selective place fields via per-cell nonlinear dendritic integration (BTSP plateau potentiation;
Bittner-Magee 2017). On the point-neuron substrate this is the genuine Mikulasch-Priesemann limit — the one piece where a
dendrite would earn its keep. But the brain ALSO has hand-developed position-specificity (the host Gaussian is a
legitimate stand-in for a developed receptive field), and the graded value read-out the field would feed is already
solved by the plateau. So the reframe is: **the selective-field carving is a deferred breadth/depth refinement with a
clear dendritic home AND two existing point-neuron fallbacks — it is the only accepted-dendritic candidate, and it does
not gate closing the loop.**

### N-3 REWARD — the reframe (already converted; the task must make it matter)

The synaptic SC-proximity approach-reward (catalog C.33, PPN→DA) is the converted organ. The reframe is purely
task-design: the orient-solvable gridworld never exercises the reward's predictive function; the trace-conditioning task
(shared with N-2) is the paradigm that makes a reward-across-a-gap load-bearing.

---

## MOVE 3 — RANK cheap-first: the closeable pieces (each with a de-risk + anti-cheat)

Ranked cheapest-first. Anti-cheats reuse the cluster's established battery: HOST positive control (the ceiling), the
SCRAM / lesion control (the surpass must be load-bearing), validate-by-FUNCTION (a task that exercises the signal),
6-seed for variable effects (1–3 seeds only for exact/mechanistic byte effects, per the standing rule), grid-32 NEVER
grid-8 (the documented false-GO scale), regime fidelity (OU/conductance-noise/homeostasis OFF), and the no-confab MOAT
(the nav cascade `cp_*` state is array-disjoint from the composer's complex `cp_rf_w_*` synapses → preserved by
construction + re-asserted).

### RANK 1 (cheapest, highest value) — DEPLOYMENT FLIPS: make the validated spiking forms the SHIPPED default

The spiking reward + value + decision are already the LIBRARY default the merged production agent inherits (CYCLE 1B);
the standalone CLI / merged-gate still defaults to host so documented benchmarks reproduce. The cheap close is to make the
SHIPPED orienting/reward/decision spiking, keeping host as the explicit `--readout-source motor` / no-`--spiking-sc`
oracle.

- **R1-a — flip the merged-gate / demo CLI orienting + decision to spiking** (`_nav_gate_merged_run.py`: `--readout-source`
  default `motor`→`spiking_wta` + `--urgency-max-pa 180`; expose + default-on `--spiking-sc` with `log_polar_retina` +
  `sc_tie_break_stochastic`). **De-risk:** the merged-gate score == the library run with the spiking config; grid-32, the
  spiking decision via the commit-burst with `decision_path_counts` primary ≥90% (NOT the argmax fallback). **Anti-cheat:**
  conversational moat byte-frozen (array-disjoint); host arm retained as the explicit oracle. *(Note: the merged-gate
  runner does not currently expose `--log-polar`/`--fix1`; those live on `_nav_sc_popvector_readout_derisk.py` — wiring
  them onto the deployed gate is part of this flip.)*

### RANK 2 — the SC margin-SNR opponent-axis (FIX3): the named remedy for the residual ~2.4×

- **R2 — opponent-axis push-pull at the cortex read-out.** Organize the four cardinals as two balanced N↔S / E↔W
  competitions so a clean 1-D margin is extracted per axis (vs the current 4-way WTA that the weak diagonal margin can't
  win). **De-risk:** grid-32, the post-change Σ drops toward host as the tie-fraction drops below ~0.18; SCRAM collapses
  (the decode is load-bearing). **Anti-cheat:** matched-drive (any lift is from the geometry, not a covert drive
  increase — the amplification screen already showed stronger drive only re-biases); 6-seed for the effect (it is a
  variable effect, not a byte-identity gate). **Reusable machinery:** the cortex-WTA flags (`enable_cortex_lateral_inhibition`),
  the `input_divisive_norm` primitive. NO `sim/` edit expected (a read-out formula + existing flags).

### RANK 3 — deploy the graded dendritic plateau into the merged critic (closes the value read-out on-substrate)

- **R3 — wire `--dendrite-critic` (the graded plateau) into the merged value critic.** The mechanism is validated on a
  critic bridge (δ=1.33 = host ceiling, 6/6). **De-risk:** the merged value-train δ lifts from ~1.3× toward the host
  ceiling with the graded plateau as the V read-out; V still graded (not saturated). **Anti-cheat:** the two point-neuron
  controls fail (LINEAR 0 Hz flat; all-or-none over-clamp); plateau-lesion collapses the δ; MOAT byte-frozen (the plateau
  flag is default-OFF for the conversational slices). NO new `sim/` edit (the `enable_graded_dendritic_plateau` flag +
  params already ship, byte-reviewed). **Caveat:** the merged δ is also capped by the position-blind up-state floor (N-2),
  so the plateau lifts the read-out half but the floor is the residual — which leads to R4.

### RANK 4 — the delayed-reward (trace-conditioning) task: make the value PROVABLY load-bearing

- **R4 — build ONE trace-conditioning harness (Pavlovian-first) on the limbic core** (scoped in full,
  `2026-06-21-shortcut9-B4-delayed-reward-value-task-scoping.md`; ~90% reuse-by-import, NO new `sim/` edit). It serves
  BOTH the #9 value-load-bearing gate AND the B4 TD-cue-shift consolidation. **De-risk (the 2×2 factorial):** (G1) TRACE
  acquisition at gap ≥300 ms-equiv; (G2 headline) lesion the graded value → the TRACE-arm behaviour COLLAPSES; (G3
  discriminator) the SAME lesion on the DELAY arm (gap=0) does NOT collapse it (the direct answer to the orient-solvable
  confound); (G4) the value GRADES the burst (not the flat-50-Hz saturation); ≥5/6 seeds. **Anti-cheat:** no-learning
  control (freeze the cue→critic STDP → no acquisition); unpaired-timing control; GABA_B-subtraction lesion; MOAT
  byte-intact; regime fidelity. **Honest scope:** the Pavlovian (PREDICT-over-gap) arm is unlikely to wall (the cue-shift
  is a point-neuron GO, eligibility traces are built for the gap); the INSTRUMENTAL (ACT-over-gap) arm is where a real
  substrate wall MIGHT appear — and that is the SPATIAL actor-critic credit family (the hidden-goal 3× NEGATIVE), a
  SEPARATE problem the trace task deliberately sidesteps; a V-B NEGATIVE would be the honest characterized deliverable,
  the legitimate juncture for the deferred dendritic substrate question — NOT the trace close.

### RANK 5 (lowest priority / accepted) — the SELECTIVE place-field carving

- **R5 — selective sparse place fields via the dendritic substrate.** This is the ONE genuinely dendritic-flavored
  residual (per-cell nonlinear integration to carve many selective fields). **It is NOT ranked for a build now:** it is
  off the actor/sustain critical path, the host Gaussian + the graded plateau are validated fallbacks, and the broader
  dendrite has been COMPREHENSIVELY ruled out for the project's other named jobs (multiplicative binding 3× NEGATIVE,
  apical-basal credit NEGATIVE — `2026-06-20-boundary-ledger-dendritic-audit.md`). The full `NeuronModel.TWO_COMPARTMENT`
  is months-scale (catalog T3.A, ~10× compute). **Classification:** the only accepted-dendritic candidate in the cluster —
  but with validated point-neuron fallbacks, so it does NOT block closing the loop; it is a deferred breadth/depth
  refinement, not a wall.

---

## MOVE 4 — VERDICT: which pieces close to spiking, and the one accepted boundary

**The navigation reward / value / SC-orienting / place CLOSED LOOP is CLOSEABLE to default-on spiking. There is no
genuinely-irreducible dendritic-months wall that blocks it.** The comprehensive plan's "documented NO-GO (~58×)" framing
is stale: the closed loop already SUSTAINS at ~2.4× host (the log-polar render + FIX1 tie-break, default-on at the library
level) and the merged production agent already runs reward + value-subtraction as spikes by default (CYCLE 1B). The
residual is dominated by DEPLOYMENT FLIPS + two small validated builds, plus one honest finite-size cost and one deferred
dendritic-flavored refinement that does not block.

### What closes to spiking (the build sequence + de-risks)

1. **R1 — the deployment flips (FIRST, cheapest).** Make the SHIPPED merged-gate / demo orienting + decision + reward
   spiking (`--readout-source spiking_wta` + `--urgency-max-pa 180`; expose + default-on `--spiking-sc` with
   `log_polar_retina` + `sc_tie_break_stochastic`), keeping host as the explicit `--readout-source motor` / no-`--spiking-sc`
   oracle. De-risk: gate score == library run, decision primary ≥90%; moat byte-frozen.
2. **R2 — the SC opponent-axis (FIX3).** The named remedy for the residual ~2.4× margin-SNR floor; 6-seed, SCRAM-collapse,
   matched-drive anti-cheat. Closes the orienting toward host on the weak-margin phases.
3. **R3 — deploy the graded dendritic plateau into the merged critic.** Lifts the value read-out half toward the host
   δ ceiling on-substrate; default-OFF flag already byte-reviewed; the two point-neuron controls fail + plateau-lesion
   collapses.
4. **R4 — the delayed-reward (trace-conditioning) task.** Makes the value PROVABLY load-bearing (the 2×2 factorial:
   TRACE-needs-value, DELAY-doesn't) — the genuine close for the value, and B4's consolidation in one harness. ~90%
   reuse, NO new `sim/` edit.

### The recommended FIRST nav close-out

**R1-a — flip the merged-gate / demo CLI to the spiking decision + spiking SC orienting (with `log_polar_retina` +
`sc_tie_break_stochastic`), keeping the host arm as the explicit oracle.** It is the single cheapest, highest-value,
lowest-risk close-out: every organ is validated (the spiking decision at 1.16× host; the SC closed loop at ~2.4× host
default-on at the library level; the spiking reward/value at CYCLE 1B), and this one flip makes the SHIPPED navigation
benchmark fully-spiking-on-one-brain — exactly the proven CYCLE-1B / consolidated-320 default-flip pattern (flip the
deployed default; keep host for reproduction). Mirror it onto the standalone `g11_bg_runner.py` CLI in the same pass.

### The precise accepted-deep boundary (the ONE, and exactly why — but it does NOT block)

**The carving of many sparse, location-selective PLACE FIELDS from heavily-overlapping egocentric landmark sensors at
nav scale (N-5's deep half).** Why it is dendritic-flavored and months-long if pursued AS a faithful self-org code: it is
the Mikulasch-Priesemann point-neuron limit — a point neuron cannot perform the per-cell nonlinear input integration that
carves selective fields from overlapping inputs (the same analog/pre-spike computation as the conversational
decorrelation/whitening wall); the faithful realization is the full `NeuronModel.TWO_COMPARTMENT` dendritic neuron
(catalog T3.A, ~10× compute, months-scale, high variance, and the broader dendrite is already 3× NEGATIVE on its other
named jobs). **Why it does NOT block closing the loop:** (a) it is the CRITIC's afferent, not the actor's drive — it
caps δ-QUALITY, not actor sustain (the loop sustains at ~2.4× with the host-Gaussian place code); (b) the host-Gaussian
place code (position-specific by construction) is a legitimate developed-receptive-field stand-in and stays the better-δ
scaffold; and (c) the graded value read-out the selective field would feed is ALREADY solved on the point-neuron
substrate by the graded dendritic plateau (δ=1.33 = host ceiling). So the honest classification is: **the selective-field
carving is the one accepted-dendritic candidate in the cluster, deferred deliberately (the artificial-life breadth
horizon), but it has two validated point-neuron fallbacks and is NOT on the critical path to a closed, default-on,
fully-spiking nav loop.**

### The honest-negatives that ARE the deliverable (BRAIN-BASED-ONLY)

Per the owner standard, two characterized costs are the scientific deliverable, NOT failures to chase: the residual
**~2.4× nav cost** (the finite-size margin-SNR floor of an orient-solvable task, the B-4 family) and the **~1.3× merged
value-grade** (the position-blind non-plastic up-state floor — a structural property of the A1+A2 critic needed to fire
the cold MSN-D1). Both map exactly what the point-neuron substrate pays for sustained reward-driven control; both are
small, characterized, and biology-faithful — not walls.

---

## Reusable machinery (point any build at these proven primitives — NO new `sim/` edit expected for R1–R4)

| Primitive | What it gives the loop | Where / status |
|---|---|---|
| Spiking commit-burst decision (#4) | the action EMERGES from spiking competition (default-on lib); leak + N-scaling + urgency knobs | `g11_bg_runner.py:2094-2203`; `2026-06-19-spiking-decision-default-on-GO.md` (1.16× host) |
| Log-polar foveal SC render + FIX1 tie-break | the SC bump exists for far goals + a fair tie read → the loop SUSTAINS | `render_egocentric_goal(log_polar=True)` `:229`; `log_polar_retina=True` default `:3866`; `sc_tie_break_stochastic` (FIX1); `2026-06-24-burndown-3F-...` (4/4 + 21.8× lesion) |
| `reward_us` PPN-like US→SNc (#7) | the spiking reward burst (host write dropped) | `perceived_approach_reward=True` default `:3346`; CYCLE 1B; RPE-battery GO |
| Striosome GABA_B/GIRK critic (#8) | the spiking value subtraction at the SNc membrane | `enable_neural_critic=True` default `:3506`; value-train GO (V learned ~20×, lesion-confirmed) |
| Graded dendritic plateau read-out | the graded analog value the soma can't be (δ=1.33 = host ceiling on-substrate) | `enable_graded_dendritic_plateau` (default-OFF, byte-reviewed `d69cc0ab`/`52dafaeb`); `--dendrite-critic` deploy wiring |
| Trace-conditioning reuse stack | the value-load-bearing task (~90% reuse) | `sim/td_value_critic.py`, `_limbic_core_rpe_battery_derisk.py`, `_merged_td_cueshift_consolidation_derisk.py`, `_merged_navcritic_valuetrain.py`; scoped `2026-06-21-shortcut9-B4-...` |
| Opponent-axis / cortex-WTA | the SC margin-SNR remedy (FIX3) | `enable_cortex_lateral_inhibition` + `input_divisive_norm` (`sim/bridge.py`) |
| `check_moat` / array-disjoint nav-vs-conv | the no-confab moat preserved by construction + re-asserted | `_merged_navcritic_valuetrain.check_moat`; `cp_rf_w_re/im` disjoint from `cp_connections` |

---

## Citations

**Project findings (read in full, trust-but-verified against the actual text + current code):**
- `2026-06-19-nav-spiking-sc-deploy-NO-GO.md` (the ~58×, now corrected as a false localization)
- `2026-06-24-burndown-3F-sc-sustained-orienting-surpass.md` (B-2 SURPASSED → ~2.4×; log-polar + FIX1 default-on; commit `853599cb`)
- `2026-06-20-nav-loop-closure-derisk.md` (the reentrant-arc "open loop" premise FALSIFIED — arc already ON)
- `2026-06-20-cascade-north-bias-FIX.md` (the host N-first tie-break shortcut; FIX1 3/3, SCRAM collapses)
- `2026-06-20-shortcut6-nav-orienting-CLOSE.md` + `2026-06-20-nav-sc-drive-reorient-derisk.md` (the pre-log-polar read-out exhaustion; stronger drive only re-biases)
- `2026-06-18-merged-navcritic-valuetrain-BOUNDARY.md` (V learned co-resident ~20×; δ ~1.32 capped by the position-blind up-state floor)
- `2026-06-20-burndown-9-critic-graded-readout.md` (the point-neuron MSN-D1 can't grade; the all-or-none over-clamps — the read-out FORK)
- `2026-06-20-dendrite-stage1-snc-calibration.md` (the graded dendritic plateau → δ=1.33 = host ceiling on-substrate, 6/6; default-OFF byte-reviewed `sim/` flag)
- `2026-06-20-shortcut9-dendrite-critic-deploy.md` (the `--dendrite-critic` deploy wiring; deploy table pending)
- `2026-06-21-shortcut9-B4-delayed-reward-value-task-scoping.md` (the trace-conditioning value-load-bearing task; the #9 deploy qualified-NEGATIVE Δ7.2%; ~90% reuse)
- `2026-06-19-place-code-sparsify-default-BOUNDARY.md` (the self-org place read-out non-selectivity; the dendritic-flavored deeper cause)
- `2026-06-18-merged-neural-reward-QUALIFIED-GO.md` (the synaptic SC-approach reward organ)
- `2026-06-20-boundary-ledger-dendritic-audit.md` (the dendrite ruled out for binding + credit; 0 dendritic boundaries block a shipped capability)
- `2026-06-19-spiking-decision-default-on-GO.md` (the spiking decision at 1.16× host, default-on lib)
- `research/findings/AUTONOMOUS_STATE.md` CYCLE 1B (the merged-nav spiking-limbic δ=r−V default flip, validated by function)
- `2026-06-27-comprehensive-shortcut-inventory-burndown-plan.md` (SHA 5ad2f453 — the inventory this gate refines; its N-2/N-3/N-4/N-5 NO-GO framing is reconciled here as STALE)

**Code (current defaults, verified this pass):** `g11_bg_runner.py` `perceived_approach_reward=True:3346` · `spiking_snc=True:3480`
· `enable_neural_critic=True:3506` · `log_polar_retina=True:3866` · `readout_source="spiking_wta":4124` · `enable_spiking_sc=False:3821`
· `sc_tie_break_stochastic=False:3882`. `_nav_gate_merged_run.py` `--readout-source` default `"motor"` · `--spiking-sc` store_true (off).

**Catalog (`sim-catalog/references/feature-catalog.md`, verified):** **A.05** reentrant cortico-BG-thalamo-cortical loops
(`:143`) · **A.07** subcortical BG loops / SC / SNr→SC disinhibition (`:169`) · **C.29** eligibility traces / TD(λ) (`:583`)
· **C.30** actor-critic (`:592`) · **C.31** bootstrapping vs Monte Carlo (`:601`) · **C.33** PPN→DA reward afferent (`:624`)
· **C.22** Schultz RPE (`:907`) · **E.04** topographic maps warped by cortical magnification / fovea (`:1384`) · **F.22**
trace conditioning + the delay-vs-trace × lesion 2×2 factorial (`:1922`) · **G.16** drift-diffusion bound (`:2826`) ·
**H.25** SC saccade map (`:3209`).

**Literature (from the prior nav deep-research, verified citations):** Ottes–Van Gisbergen–Eggermont (SC logarithmic
afferent map); Hafed lab 2019 (SC eccentricity model); Dunovan-Verstynen (biologically-constrained spiking CBGT loops
sustain reward-driven policy, biorxiv 2024.05.21.595174); Hesslow-Yeo 2002 + Moyer-Deyo-Disterhoft 1990 (trace vs delay
dissociation); NAc DA encodes the trace period (eNeuro 2025 ENEURO.0016-25); Frémaux-Sprekeler-Gerstner 2013 (spiking
actor-critic, the V-B path); Bittner-Magee 2017 (BTSP plateau place-field formation); Sutton & Barto 2e (TD, eligibility,
actor-critic).

_READ-ONLY deep-research gate. No code, no `sim/` edit, no experiments. The no-confab moat is array-disjoint from the nav
cascade and untouched. grid-32 is the verdict scale (never grid-8). Load-bearing "surpassed" / "default-on" claims
verified against the actual finding text + the current code defaults._
