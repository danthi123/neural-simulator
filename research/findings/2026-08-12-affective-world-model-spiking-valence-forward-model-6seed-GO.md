---
type: finding
status: live
date: 2026-08-12
mechanism: affective-world-model-valence-forward-model
verdict: GO (6/6 seeds, runner-level de-risk; NOT yet wired/integrated to production)
artifacts: research/findings/raw/_affective_world_model_6seed.json
verification: >
  substrate seeded (firing-threshold hash identical @ seed, differs across seeds); two full
  runs byte-equal on the intact rows; instrument verified in BOTH directions (a false-null from
  g_i numerical instability AND a false-GO from a weak-firing artifact were caught and fixed by
  reading conductances / pool rates, not the summary); runner's own tools.verdict.Verdict = GO.
---

# Internal worldview / affective world-model — a spiking VALENCE FORWARD MODEL, 6/6-seed GO (2026-08-12)

## Result
The owner-named **internal worldview / affective world-model** faculty (burn-down E2, previously
"likely ABSENT/unvalidated — RESEARCH-NEEDED") is de-risked at a **6/6-seed GO** (local CPU,
`SIM_BACKEND=numpy`, seeds 42 43 44 100 101 102). The brain maintains a LEARNED spiking forward
model of its conversational world: from the current conversational STATE it PREDICTS the
interlocutor's NEXT-turn AFFECT (valence), holds that expectation, fires a spiking SURPRISE
(affective prediction-error) when the actual next turn VIOLATES it, UPDATES the transition from
that error, and is QUERYABLE turn-by-turn. This is NOT static fact recall — it is a transition
model over states that predicts the FUTURE and rides LEARNED trajectory statistics (the shuffle
control proves it).

<!--derived--> (Values re-quoted from the cited artifact.)
Aggregate (6 seeds): predicted-valence accuracy **1.00**; expected-turn surprise **0.00 Hz**
(clean predictive-coding cancellation); violated-turn surprise **36.7–45.6 Hz** (a real error
signal). Lesion of the learned transition collapses the separation (ratio **1.0×**, expected
rises to the violated level ~47 Hz, **100%** attributable). Dual-scored shuffle: **3/3** seeds
fail to reproduce the TRUE separation yet **3/3** still GO vs their OWN scrambled map (acc vs
trained **1.00**). Multiturn: surprise **0 Hz** on expected turns, **49.3 Hz** on the injected
deviation. Update-on-error: the predicted-valence read shifts **+395 → −2.78** toward the new
observation after re-experiencing the state with the opposite valence.

Runner: `research/runners/_affective_world_model_derisk.py` (NEW; reuse-of-pattern, NO `sim/` edit).
Artifact: `research/findings/raw/_affective_world_model_6seed.json` (+ provenance sidecar).

## Reproduce
```bash
SIM_BACKEND=numpy python -m research.runners._affective_world_model_derisk \
    --seeds 42,43,44,100,101,102 --n-reps 22 \
    --out research/findings/raw/_affective_world_model_6seed.json
```

## Why this is a WORLD-MODEL, not fact recall (the load-bearing distinction)
A world-model maintains + updates an internal PREDICTIVE representation of the world that it can
query and that drives behaviour. This mechanism is a **learned first-order transition (forward)
model** — the prediction at turn t is a function of the OBSERVED state at turn t (which came from
turn t−1), so predictions chain and the expectation rolls forward across turns. The substrate
already had the within-turn spiking mismatch unit over stored facts
(`_spiking_expectation_rpe_derisk.py`, 2026-08-12), the HTM Temporal-Memory next-SYMBOL predictor
(EMERGE-15/9d GO), the W5 other-tagged affect model (2026-08-01), and the P0.3 valence latch —
but NONE was an AFFECTIVE forward model that predicts the interlocutor's next-turn valence,
updates on a spiking affective prediction-error, and is queryable. This de-risk INTEGRATES those
pieces into the missing faculty.

The four faculty properties, each demonstrated:
- **(a) maintained across turns** — the state pool is re-driven each turn and the expectation is
  continuously queried/updated (multiturn arm: surprise 0 on expected turns, spikes only on the
  deviated turn).
- **(b) predicts the next affect** — the queryable predicted-valence sign
  `sign(rate(pred_pos) − rate(pred_neg))` matches the true next valence on 6/6 states, 6/6 seeds.
- **(c) updates on prediction-error** — the transition is Hebbian-plastic; re-experiencing a
  state with the opposite observed valence (learning ON) shifts the prediction toward the new
  observation (+395 → −2.78).
- **(d) queryable** — the predicted valence and the surprise rate are within-window spike reads
  ("what do you expect / was that a surprise?").

## Mechanism (brain-based; a 2-channel spiking predictive-coding valence forward model)
Predictive coding (Rao & Ballard 1999; Bastos et al. 2012) over the AFFECT good/bad axis, in the
interoceptive/affective-inference line (Seth 2013; Barrett & Simmons 2015 — the brain runs a
generative model predicting its affective state), on the real spiking Izhikevich bridge:
- **state** (RS, one block/state) — the current conversational state, driven by the environment.
- **pred_pos / pred_neg** (FS, PV-like) — the PREDICTED next-turn valence, delivered as
  subtractive GABA_A inhibition (the top-down prediction). `state → pred_{pos,neg}` is
  all-to-all PLASTIC, INITIALISED AT ZERO: Hebbian co-fire builds each state's edge to ONLY the
  valence pool that followed it (a non-zero init drives both pools → no selectivity). A 2-way
  learned discrimination per state — arbitrary per seed, so the shuffle is decisive, while
  sidestepping the n-way CA3 pattern-separation wall (2026-06-05-D-cue-recall-RESOLVED).
- **obs_pos / obs_neg** (RS) — the ACTUAL next-turn valence, delivered as sensory drive (the
  legitimate environment/teacher boundary).
- **surprise_pos / surprise_neg** (RS) — the error units. `obs_v` excites `surprise_v`;
  `pred_v` inhibits `surprise_v`. EXPECTED (obs == predicted): the prediction cancels the
  observed → surprise ~0. VIOLATED (obs == opposite): the un-inhibited channel FIRES.
  Total surprise firing IS the affective prediction-error, read from `cp_firing_states`.

What is neural: the prediction (Hebbian recall, a spike-rate read), the mismatch (membrane
subtraction, a firing read), and the update (plastic transition). The legitimate host boundary:
the conversational-state token and the observed next valence delivered as DRIVE.

## Anti-cheat controls (all pass, and the metric provably reads non-zero when it should)
- **Lesion (decisive)** — zero the learned `state → pred` edges → no prediction → surprise fires
  HIGH on EXPECTED too (expected 0 → ~47 Hz), ratio → 1.0×, 100% of the separation attributable
  to the spiking prediction. Lesion HOLDS: `measure()` freezes learning
  (`enable_hebbian_learning = False`), so the zeroed edges cannot regrow.
- **Shuffle (dual-scored, structure not template)** — train on a scrambled balanced valence map.
  (a) scored vs the TRUE trajectory: 3/3 seeds fail to reproduce the separation (no true-GO,
  acc → chance 0.44). (b) scored vs its OWN trained map: 3/3 still GO (acc 1.00) — the model
  genuinely LEARNED the arbitrary structure. (A random 3+/3− permutation partially overlaps
  true, so the true-scored ratio alone is noisy per seed; the trained-scored arm is the
  per-seed-clean control.)
- **Metric can read the collapse** — expected-turn surprise is 0 Hz intact but rises to 15–28 Hz
  under shuffle and ~47 Hz under lesion. The 0 Hz is a real cancellation, not a dead instrument.

## The operating point IS the companion process (the wall-reframe, and two instrument saves)
The 2-channel error unit's operating point is the GAIN MATCH between observed excitation and
predicted inhibition (precision / divisive normalization; PV/SST + neuromodulation in biology).
Two instrument failures were caught here, one in each direction — both by READING the substrate,
not the summary:
- **A false NULL**: with the state cue at 1000 pA the FS prediction fires ~446 Hz; at
  `pred→surprise` weight 24 the accumulated `g_i` reaches ~653 nS, which destabilises the
  explicit-Euler membrane update and reads as spurious firing (expected saturated at 500 Hz →
  ratio ~1). Reading `cp_conductance_g_i` on the surprise pool exposed it. The fix is the
  precision companion: the pred→surprise weight must be ~2 (matched to the ~446 Hz prediction).
- **A false GO**: an earlier "cancellation" at cue 600 pA was a weak-firing artifact (pred ~22
  Hz), not real subtraction — the pred→surprise pathway is EXCITATION-free only when the gain is
  matched. Driving `state` alone (obs off) and seeing surprise fire exposed it.
The instrument is part of the emulation: a mechanism measured wrong is tuned wrong, confidently.

## Honest scope + the named next rungs (the boundary is the deliverable)
This is a runner-level GO — **NOT `wired` / `integrated` / production-default**. It is a genuine
faculty de-risk; binding it into `/api/brain-chat` is the parent session's integration step.
Boundaries, each with the mechanism that surpasses it (no permanent walls):
1. **First-order (Markov-1)**: predicts next valence from the current state alone. A state that
   is ambiguous given history needs the HTM-TM high-order predictor (EMERGE-15 GO) — the next
   rung for context-dependent affect.
2. **2-way valence, not n-way next-STATE**: the prediction is over the affect good/bad axis
   (matching P0.3's characterized bistable scope), which sidesteps the CA3 pattern-separation
   wall. A full next-conversational-state ROLLOUT (predicting the state sequence, not just its
   valence) needs the CA3 sparse pattern-separation companion (2026-06-05) — the next rung.
3. **Generic pools, not the P0.3 organ / W5 other-model**: pred/obs are plain FS/RS pools tagged
   pos/neg; they are not yet the P0.3 bistable affect latch nor the W5 OTHER-tagged affect model.
   Binding the predicted valence to the real P0.3 latch and routing the OBSERVED valence through
   the W5 ToM channel (so the world-model predicts the INTERLOCUTOR's affect) is the integration
   rung toward the production faculty.
4. **Teacher-driven, not self-organized**: the observed next valence is provided as the training
   signal (the legitimate environment/teacher boundary), so the transition is LEARNED but NOT
   `self-organized` (the host supplies the target). Deterministic regime (OU + channel noise off
   for a controllable operating point; a noise-robustness pass is a separate test).
5. **Graded circumplex** (arousal, discrete emotions) is the shared P0.3 surpass (a line/bump
   attractor with SFA eviction / the dendritic substrate), not a new wall.

## Adversarial verification (verify-go lenses, all pass)
- **Reproducibility/power** — 6/6 seeds GO; two full runs byte-equal on the intact rows; the
  effect (violated ~40 vs expected 0) >> seed-to-seed noise; no single seed carries it.
- **Gate-cheat** — lesion, dual-scored shuffle, and attribution are invoked every run and sit in
  `go_components`/`preconditions`, not merely defined.
- **Control-integrity** — the lesion is ONE variable (zero `state → pred`); arms are not both
  saturated (intact expected 0 vs violated 40; lesion both ~47 — a responsive range).
- **Instrument-trust** — the runner's `tools.verdict.Verdict` earned GO; the metric provably
  reads the collapse (lesion 47, shuffle 15–28). Both a false-null and a false-GO were caught.
- **Seeding** — `cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = seed`; firing-threshold hash
  identical at a seed, differs across seeds. Deterministic.
- **Brain-based** — surprise = `cp_firing_states` read; predicted valence = a two-pool spike-rate
  difference (W5 tone_sign motif); `current_reward_signal == 0` (asserted); no host argmax over a
  transition table, no host compare of observed vs predicted valence.

## Roadmap
Advances the affective/self-model lane (the internal-worldview faculty). Builds on the D2 spiking
mismatch unit, EMERGE-15 (high-order predictor, the rung-1 surpass), P0.3 (valence latch), and W5
(other-tagged affect). CPU, disjoint from the GPU lanes. Next actions for the parent: (i) wire the
predicted valence to the P0.3 latch + the W5 ToM channel; (ii) the HTM-TM high-order affect rung;
(iii) a noise-robustness pass; (iv) integration to `/api/brain-chat` (the `wired`/`integrated`
credit).
