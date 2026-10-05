# Selection seam with no host authority: an interoceptive need reverses the basal-ganglia winner

*2026-09-29. Plan step 3 of `docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md` (sections 2, 3, 6).
Branch `research/selection-seam-need-reversal`, lane A. Runner `research/runners/_need_reversal_selection_seam.py`,
CI guard `tests/test_need_reversal_selection_seam.py`. Preregistration: `research/findings/2026-09-29-need-reversal-selection-seam-PREREGISTRATION.md`.*

## 1. Which existing machinery, and where its host authority sits

The plan says to use the existing ignition bus and basal-ganglia selection, not a new selector. I read both.

- **Basal-ganglia selection** (production, default-ON as `BRAIN_BG_SELECT`):
  `research/runners/bg_action_selection_production_organ.py` over the Gate-A v2 topology
  (`research/runners/_vocal_action_selector_gate.py::build_selector_bridge`; finding
  `2026-08-03-neural-vocal-selector-gateA-v2-4seed-GO.md`). The race is spiking end to end: shared arousal ->
  cortical proposals -> striatal D1/D2 -> GPi (direct path) / GPe-STN (indirect) -> thalamus -> commit burst with
  cross-channel FS inhibition -> motor pool. The first motor pool to cross 12 spikes with the other at most 25% of it
  is the winner. This is where selection AND commitment are neural.
- **The GNW ignition bus** (`webapp/gnw_bus_shadow.py`, `gnw_two_organ_bus.py`, `gnw_three_organ_bus.py`,
  `gnw_deliberation.py`; workspace builder `research/runners/_p1_2_workspace_deliberation_loop_derisk.py`) commits a
  winner by NMDA-attractor ignition, but every slot's drive is set by the host before ignition: the target slot at
  `IGNITE_PA`, distractors at 0.3x, or all candidates equal (seam-map rows C10-C12 on `research/draft-seam-map`). The
  workspace makes the commitment all-or-none; it does not score candidates. Selection among candidates of different
  value therefore has to happen upstream of it, which in this brain is the basal ganglia (the PBWM picture: striatal
  output gating opens the thalamic drive into the workspace).

So the seam with the most host authority over *which candidate wins* is the salience input to the BG race, and this
build targets it. Routing the BG commit into a workspace slot (so ignition broadcasts the neural winner) is the next
rung; it does not change who owns the choice, which is what this step measures.

### Seam-map rows for this seam (plan section 1 fields)

| field | production `bg_action_selection_production_organ` (today) |
|---|---|
| host operation | `salience(msg)` -> `(speak, silent)` scalars from a content-token count; `+= salience * 600 pA` onto each channel's D1 pool; plus the pre-filter `if not (silent_sal > speak_sal): return None` |
| neural state before it | none: the scores are computed from the message text, not read from the substrate |
| decision the host makes | candidate scoring (which action is more valuable) AND admission (whether the race is consulted at all) |
| does the substrate already represent the choice | the race represents the winner-take-all, not the value; there is no neural value input |
| host override effect | total by construction: the salience is the only candidate-specific input to the race |
| replacement difficulty | moderate: needs a neural value source wired into D1; this build supplies one for need-based value |
| research value | high: the same host-scoring pattern feeds the tone seam (`congruence_saliences`) and value choice (`VALUE_HZ_GAIN_PA` x Hz) |

## 2. The build

Everything runs on ONE `SimulationBridge`: the unchanged Gate-A v2 selector plus two regions and four pathways added
through the builder's own `extra_regions` / `extra_pathways` arguments. No `sim/` or `webapp/` edit.

- **Candidates.** Two actions = the selector's two channels. What each action *is* ("rest", "forage") lives only in
  the world-side table `WorldCandidates`, which logs every read with its phase. The world applies the consequence
  (foraging reduces the deficit) after the commit.
- **Need state.** `intero_hunger` (AgRP-like) and `intero_satiety` (POMC-like), 24 Izhikevich RS relay neurons each,
  driven by the body's energy deficit h in [0, 1] through an afferent current (`INTERO_MAX_PA * h` and
  `INTERO_MAX_PA * (1 - h)`). This is the body -> sensory boundary already used by the homeostatic-drive GO
  (`2026-06-17-homeostatic-spiking-drive-mechanism-GO.md`) and the embodied-affect GO
  (`2026-08-19-embodied-affect-interoception-GO.md`). h is body state; the host never converts it into a value.
- **Need -> value, through synapses.** `intero_hunger -> str_d1_1` and `intero_satiety -> str_d1_0`, excitatory
  slow NMDA (Mg2+-blocked, `exc_receptor="nmda_slow"`), frozen, behind the transmission gate `need_value`. The NMDA
  receptor makes the need act on MSNs the cortical proposal barrage has already depolarized (the striatal up-state),
  so it biases the race without driving it: on 12 dev seeds an AMPA route of matched selectivity let the need commit
  an action on its own more often (DEV tune in `research/findings/raw/_need_reversal_seam/dev_tune*/`). Biology: hypothalamic need signals reach the ventral striatum
  through glutamatergic relays (paraventricular thalamus -> nucleus accumbens) and bias action value by physiological
  state (incentive value depends on the current need). The mapping is wired, not learned (see section 6).
- **What the host does.** Equal tonic currents, the shared arousal onset, the inter-trial reset, the body current.
  It receives only the index of the committed motor pool. In the synaptic arm the host salience is exactly (0, 0).
- **The trial function** `run_need_trial` mirrors the production `_run_biased_trial` step for step; the CI guard shows
  it reproduces the production race spike for spike when the body and clamps are off. It takes no candidate identity.
  Every trial records the membrane range; `tools/lab.py::membrane_bound_check` raises outside [-150, +100] mV (the
  review found the local model's N=5 build ran at -377 mV).

## 3. Arms and measurements

**Arms.** BEFORE = `host_salience`: need synapses closed (gate 0); the host scores each candidate from its label and
the need (`host_value_rule`: forage gets h, rest gets 1 - h) and injects the production 600 pA gain. This is the
production pattern with a need-based score. AFTER = `synaptic_need`: need synapses open, host salience (0, 0).
Both arms use the same build, so they differ only in the gate and the salience.

**Key test.** State-dependent reversal of the winner in the AFTER arm: R = P(forage | committed, h=1) -
P(forage | committed, h=0).

**Controls (AFTER arm).**
1. Need-synapse lesion (gate `need_value` = 0): the reversal should vanish while the race still commits.
2. Shuffled need mapping, two forms: (a) each interoceptive population projects to both D1 pools at half density
   (same expected synapse count, mapping destroyed): the reversal should vanish; (b) swapped interoception (the body's
   deficit drives the satiety relay and vice versa): the winner should follow the neural state, so R flips sign.
3. Code path: `WorldCandidates` counts reads before the commit (must be 0 in the AFTER arm, and is > 0 in the BEFORE
   arm, which is the instrument's positive control); a static AST audit of `run_need_trial`; and flipping every host
   decision variable (swapped label table, inverted host belief about the need) must leave the AFTER arm's spike
   record byte-identical.
4. The selector's own anti-cheats: arousal lesion and direct-path lesion floor the commit.

**Host share (plan section 2), BEFORE vs AFTER.**
- Decoding: sign-rule prediction of each committed winner from the host variable, the interoceptive differential and
  the D1 differential. A screen only.
- Substitution: the host scoring replaced by the neural route at the same body state; the change in P(forage).
- Intervention (primary): neural need fixed, host decision variables flipped. HS = (P(H | intervened) -
  P(H | baseline)) / (1 - P(H | baseline)), H = the option the intervention favors. The converse (neural need
  flipped, host variables fixed) gives the neural share NS. Expected: BEFORE HS ~ 1, NS ~ 0; AFTER HS = 0, NS ~ 1.
- Host override effect (seam-map field): in the AFTER arm the host injects the production salience for the
  non-preferred option; how often does the host win against the need?

**Mechanism necessity (non-negotiable 4).** Each ablated alone in the AFTER arm, R and commit rate at both extremes:
need synapses; interoceptive relays silenced; shared arousal; direct path; D2 MSNs (indirect path) inactivated; GPe
inactivated; STN inactivated; commit cross-inhibition (commit FS pools inactivated); thalamus inactivated; commit
NMDA off; OU noise off. Inactivation is a constant -1000 pA on the region (silences every target with V >= -95 mV), and the
runner records the silenced region's spikes to show it worked.

**Secondary (reported, not gated):** the need curve h = 0, .25, .5, .75, 1 in both arms.

## 4. Operating point and seeds

`NEED_WEIGHT`, `NEED_DENSITY`, `INTERO_MAX_PA` are tuned on dev seeds 1-12 only and frozen in the preregistration
before any gate seed runs. Gate seeds 42/43/44/100/101/102 run only after the preregistration is committed; the
runner refuses them without `--allow-gate-seeds`. Backend: numpy (deterministic; the CI guard and the byte-identity
check depend on it). A cupy replicate on the GPU is a DEV cross-backend check, not part of the verdict.

## 5. What a GO would and would not show

It would show that for this seam the choice between two candidates is owned by the substrate: the winner follows a
neural state that reaches the striatum through synapses, flipping every host variable changes nothing, and the
host-share measure drops from about 1 (production pattern) to 0. It would not show that the need -> value mapping was
learned (it is wired), that the ignition bus broadcasts the winner (next rung), or anything about production chat
turns (the production organ still takes host salience; wiring this route into it is a flag-gated follow-on).

## 6. Next rungs (no-defer)

1. Learn the mapping: DA-gated plasticity on `intero_* -> str_d1_*` from the drive-reduction reward (the
   homeostatic-drive GO already reads that reward off spikes), so the association self-organizes. Couples to lane B's
   temporal-credit work; the Gate-B vocal credit record (v1-v14) shows reward credit on this selector is the hard part.
2. Route the BG commit into the GNW workspace slot, so the ignited broadcast carries the neural winner.
3. Replace the production organ's host salience with a neural value source behind a default-off flag, with a
   byte-identical-off test, and re-measure its host share.
