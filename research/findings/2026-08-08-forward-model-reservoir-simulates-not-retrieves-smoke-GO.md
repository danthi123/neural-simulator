---
type: finding
status: contributing
date: 2026-08-08
mechanism: forward-model-spiking-reservoir-gain-field-conjunction
lane: world-model
---

# FORWARD MODEL on the SPIKING RESERVOIR — a LOCAL read-out over an Izhikevich reservoir predicts (s,a)->s' and GENERALIZES to HELD-OUT (s,a), beating retrieval (single-seed SMOKE GO, 2026-08-08)

The one missing cognitive organ named by the faculty map — a **learned forward model** that SIMULATES a novel transition rather than RETRIEVING a stored fact — is built here on the UNBLOCKED spiking reservoir substrate. Driving `OnBridgeLSM`'s recurrent Izhikevich region with a (state, action) input and training a **LOCAL delta-rule read-out** over its real `cp_firing_states` spike-counts predicts the next state AND generalizes to (s,a) combinations **never trained** (held-out 0.720 vs marginal-prior lookup 0.000 / nearest-neighbour retrieval 0.000, chance 0.040). This is a **1-seed SMOKE**; the parent runs the 6-seed validation before any generalization claim.

**Runner:** `research/runners/_forward_model_reservoir_derisk.py` (reuse-by-import of `OnBridgeLSM` from `_emerge82_onbridge_lsm_derisk`; numpy; NO `sim/` edit). **Artifact:** `research/findings/raw/_forward_model_reservoir_smoke.json`.

## Research gate — what the record had, and the genuine un-built organ
`bash tools/before_you_build.sh` + RAG surfaced the adjacent priors; I READ each. The world-model already inherits/completes/reasons transitively (spreading-activation completion GO 2026-07-08) but RETRIEVES. Two adjacent forward-model priors exist and this is DISTINCT from both:
- **BORN learned bodily self-model (2026-08-07 GO):** a LEARNED forward model, but efference->predicted-SENSORY-feedback for the AGENCY/self signal (Hebbian/Oja SELECTIVE DIAGONAL, K=4 identities). It is NOT a general compositional (s,a)->s' world-model and has NO held-out (s,a) generalization probe.
- **D3 spiking transition (2026-07-09 GO):** a DFA transition learned THROUGH a spiking pool, but by **surrogate gradient** (a gradient method, teacher-forced per-step); its generalization probe is sequence DEPTH, not novel (s,a) combinations.

The genuine un-built organ: a forward model on the **reservoir** substrate, trained by a **LOCAL rule** (delta/LMS at the read-out — not BPTT, not surrogate gradient), whose prediction on **held-out (s,a)** beats retrieval — the decisive "simulate, don't look up" test. Built ON `OnBridgeLSM` (EMERGE-82, the spiking reservoir on a real `SimulationBridge`).

## The mechanism (brain-based; reuse-by-import; NO `sim/` edit)
- **World (legit host):** a 5x5 toroidal grid. State s=(x,y); action a in {E,W,N,S}; the world's transition is a factored SHIFT s'=(x+-1,y) or (x,y+-1) mod 5. The world renders the sensory code and supplies the action drive; it does NOT compute s' for the brain.
- **Encoding (sensory rendering):** state -> factored code (x one-hot ++ y one-hot); action -> one-hot. Tokens present state AND action **SIMULTANEOUSLY** (reps x [state ++ action]).
- **Reservoir (NEURAL):** `OnBridgeLSM` drives its recurrent Izhikevich region through the bridge's real step loop; the read-out feature = the region's per-neuron SPIKE-COUNT (`cp_firing_states`).
- **Read-out (LOCAL RULE):** normalized-LMS delta rule over the frozen reservoir features -> predicted s' code; W += eta/(1+||x||^2)*(target-pred) outer x. Local (post-synaptic error x pre-synaptic activity); no BPTT, no surrogate gradient (the reservoir is fixed-random). Converges to the ridge least-squares read-out.

## The load-bearing insight — SIMULATION needs a GAIN-FIELD (state x action) CONJUNCTION, on spikes
The decisive diagnostic is now three REAL mechanism-control arms in the runner (`_feat_additive`, `_feat_conjunctive`, `_encode_seq_separate`), each trained by the SAME local delta read-out on the SAME train/held-out split — the numbers below trace to the committed artifact, not to a comment (the derived table follows; earlier drafts stated the separate-presentation number as 0.04, the true measured value is 0.12). <!--derived--> A purely ADDITIVE linear read-out over the raw one-hot input generalizes to held-out (s,a) at **0.04** (chance) — and cannot even FIT the trained shifts (train 0.227) — because the action-conditional shift is MULTIPLICATIVE, not additive. Given an explicit (state (x) action) CONJUNCTIVE basis it generalizes at **1.00**. This is the parietal **gain-field** primitive (Andersen; Salinas & Abbott 1996; Pouget & Sejnowski basis functions) — the brain's coordinate-transform mechanism. (Arms 1 and 2 are HOST reference bases — declared shortcuts, not neural mechanisms; arm 1's FAILURE and arm 2's CEILING bracket the reservoir's neural held-out of 0.72.) The reservoir's job is to SUPPLY that conjunctive basis on spikes, and it does so ONLY when state and action are **co-present** at a neuron so the Izhikevich threshold acts as a coincidence detector: separate state-then-action presentation MEMORIZES the trained cells (train 1.00) but collapses held-out to **0.12** (comparator with teeth — had it survived, the coincidence story would be refuted); simultaneous presentation reaches **0.72**. The companion process real cortex runs that we had proxied away was the *coincidence*: presenting the factors serially removed it.

## Result — single-seed SMOKE (seed 42, G=5, n_pool=400, held-out 25 of 100 (s,a) cells)
<!--derived-->

| quantity | value | meaning |
|---|---|---|
| mean spikes/neuron | 4.07 | reservoir genuinely active (read from `cp_firing_states`) |
| train_acc | 1.000 | read-out fits the trained transitions |
| **held-out_acc** | **0.720** | **generalizes to (s,a) NEVER trained -> simulation** |
| prior-lookup baseline | 0.000 | "has the fact else prior" store fails on novel cells |
| NN-retrieval baseline | 0.000 | strongest soft retrieval fails on novel cells |
| chance | 0.040 | 1/25 |
| lesion read-out (zero W) | 0.040 | collapses to prior -> neural read load-bearing |
| lesion reservoir-silence | 0.040 | collapses to prior -> spikes load-bearing |
| seeded (byte-identical) | True | two same-seed builds hash identically |

## Mechanism controls — the reservoir is COMPOSITIONAL, not additive/lookup (seed 42, committed code paths)
<!--derived-->

All three arms are REAL code in the runner, trained by the SAME local delta read-out on the SAME train/held-out split; every value traces to `research/findings/raw/_forward_model_reservoir_smoke.json` (keys `ctrl_*`).

| arm (feature) | code path | train_acc | held-out_acc | reads |
|---|---|---|---|---|
| reservoir, SIMULTANEOUS (the mechanism) | `_encode_seq` → `OnBridgeLSM` | 1.000 | **0.720** | neural spikes, factors co-present |
| (1) additive raw-input read-out | `_feat_additive` (host, declared shortcut) | 0.227 | **0.040** | linear/additive code CANNOT do the multiplicative shift (fails even on train) |
| (2) explicit conjunctive gain-field basis | `_feat_conjunctive` (host idealization/CEILING) | 1.000 | **1.000** | a perfect (state x action) conjunction; the ceiling the reservoir approximates |
| (3) separate (non-simultaneous) presentation | `_encode_seq_separate` → `OnBridgeLSM` | 1.000 | **0.120** | reservoir MEMORIZES trained cells but no coincidence → no held-out |

Reading: additive (0.04, chance) < separate-presentation (0.12) << reservoir-simultaneous (0.72) < conjunctive ceiling (1.00). The reservoir's neural held-out sits far above the additive floor and the co-presence-broken control, and below the explicit-conjunction ceiling — i.e. the plain reservoir CONSTRUCTS most (not all) of the (state x action) conjunctions on spikes, and ONLY when the factors are co-present. `compositional_not_additive=True` (held-out beats both additive and separate by ≥0.30 and sits at/below the conjunctive ceiling).

## Anti-cheats — all pass at seed 42
- **(a) GENERALIZATION > RETRIEVAL** — held-out 0.720 vs prior-lookup 0.000 AND nearest-neighbour retrieval 0.000 (gap 0.72 >> 0.30). NN retrieval is BELOW chance because a held-out cell's nearest trained feature carries a systematically-wrong s' (the shift offsets it) — retrieval genuinely cannot simulate.
- **(b) LESION -> PRIOR** — zeroing the read-out W (0.040) and silencing the reservoir input (0.040) both collapse held-out to chance: the prediction is genuinely read off the reservoir's spikes, not a static bias.
- **(c) NEURAL SOURCE** — the read-out input is the region's real `cp_firing_states` spike-counts; 4.07 spikes/neuron.
- **(d) DEFAULT-OFF / BYTE-IDENTICAL** — a separate bridge instantiated in the runner (NO `sim/` edit; shared substrate untouched); `cfg.seed` produces byte-identical `cp_neuron_firing_thresholds` across two builds (seeded=True).

## Honest scope
- **SMOKE, one seed.** GO is declared for seed 42 only; the parent runs 6 seeds. Held-out 0.720 clears the 0.60 floor with margin but is NOT perfect — the plain reservoir supplies MOST but not ALL of the (state x action) conjunctions (the conjunctive-basis ceiling is 1.00). Absolute accuracy is secondary; the DECISIVE result is the gap over retrieval (0.72 vs 0.00).
- **The world is a toy factored-shift grid** — a clean, controlled transition where held-out (s,a) is predictable from the factored operation, so generalization = compositional simulation (not lookup). It is a mechanism proof, not a claim about arbitrary world dynamics.
- **The gain-field conjunction is currently supplied by the reservoir's random coincidence neurons.** The named next lever to raise held-out toward the 1.00 conjunctive ceiling: a dedicated spiking coincidence/gain-field layer (state and action projections onto a high-threshold AND layer) read alongside the reservoir, still local at the read-out.
- Wider reservoirs (n_pool 600/800) hit a pre-existing bridge synapse-capacity broadcast error at build; capped at 400. (Noticed failure -> logged.)

## What this establishes
The first **learned forward model that SIMULATES rather than retrieves** on the spiking reservoir substrate: a local-rule read-out over an Izhikevich reservoir predicts (s,a)->s' and generalizes to novel (s,a) far above every retrieval baseline, with the neural read shown load-bearing by two lesions — and it names WHY (the gain-field state x action coincidence conjunction, supplied on spikes only when the factors are co-present). Follow-ons: the 6-seed validation; a dedicated spiking gain-field layer toward the conjunctive ceiling; multi-step roll-out (feed s'_pred back as the next state to imagine a trajectory).

## Files
`research/runners/_forward_model_reservoir_derisk.py`; artifact `research/findings/raw/_forward_model_reservoir_smoke.json`. Reuses `OnBridgeLSM` (`_emerge82_onbridge_lsm_derisk`). NO `sim/` edit.
