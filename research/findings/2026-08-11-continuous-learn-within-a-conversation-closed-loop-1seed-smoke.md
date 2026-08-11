---
type: finding
status: contributing
date: 2026-08-11
mechanism: a fact taught MID-CONVERSATION (turn K) by the brain's own co-resident e-prop plasticity changes the brain's LATER reply (turn K+M) on ONE persistent spiking bridge — a minimal closed learn-within-the-conversation loop
lane: E-language / INTEGRATION (THE CONTINUOUS LOOP — perception→internal-state→speech→consequence→LEARNING)
seeds: [42, 43, 44, 100, 101, 102]
verdict: 6-SEED GO (6/6, coordinator-run) — GO_6of6=True, verdict_earned=GO. The 1-seed smoke's closed learn-within-a-conversation loop holds across all six seeds.
artifacts:
  - research/findings/raw/lanes/stageA/continuous_learn_in_conversation_6seed.json
  - research/findings/raw/lanes/stageA/continuous_learn_in_conversation_s42_smoke.json
runner: research/runners/_continuous_learn_in_conversation_derisk.py
instrument: build_one_brain(co_resident_eprop=True) makes ONE persistent SimulationBridge; ONE CoResidentEpropNet (`_mk_merged_net`) + ONE ConjunctiveFamiliarityGate are constructed ONCE and never rebuilt; the conversation runs as SEGMENTS on the SAME bridge/shim/snap (pre-teach → learn → M intervening turns → query) via CF.run_chat; the mid-conversation learning event is I7._train_eprop on the persistent net (a real e-prop weight change in the shared cp_connections); the reply is read by AcquiredReadComposer.query_patient consulting the LIVE net. SIM_BACKEND=numpy, cfg.seed-controlled, NO sim/ edit, reuse-by-import.
---
<!--derived-->
**⭐ 6-SEED CONFIRMATION (coordinator-run, `research/findings/raw/lanes/stageA/continuous_learn_in_conversation_6seed.json`): GO 6/6 (`GO_6of6=True`, `verdict_earned=GO`).** The closed learn-within-a-conversation loop — a fact taught mid-conversation, via an on-substrate e-prop weight change, causally changing a LATER reply on ONE persistent spiking bridge, with the moat intact — holds on all six seeds (42/43/44/100/101/102), not just the smoke seed.


# The continuous loop — a fact learned mid-conversation changes a later reply on ONE persistent brain (1-seed smoke)

INTEGRATION #7 (+ the one-brain merge) already gave the chat a co-resident e-prop acquisition net that learns facts
into the shared weights — but its eval TEACHES every fact UP FRONT and then reads a FIXED 14-turn script. It is not a
loop where a fact told at turn K changes the brain's reply at turn K+M via the on-substrate weight change. This
de-risk closes that specific piece: the learning event happens INSIDE a persistent conversation and its causal effect
on a LATER reply — on the SAME bridge, NO rebuild between K and K+M — is measured and lesioned.

## What runs

<!--derived-->

ONE persistent brain (`build_one_brain(co_resident_eprop=True)`, 26403 neurons, seed 42, SIM_BACKEND=numpy, backend
module `builtins` = the numpy backend). ONE co-resident e-prop net + ONE familiarity gate, constructed once and never
rebuilt. The conversation is run as segments on the SAME `SimulationBridge`, so the only thing that carries across a
turn is what the substrate itself holds:

- SEGMENT 1 (turns 1..K-1, pre-teach): the NEW referent "dax" (absent from the curated kb) is queried — the brain is
  honestly ignorant → SILENCE (net untrained on it + the gate empty → abstain).
- LEARNING EVENT (turn K): the teacher tells dax→grass (+ dog→bone, cat→fish contrastive background). `_train_eprop`
  moves the persistent net's readout — a real e-prop weight change in the SAME `cp_connections` as every
  conversational synapse.
- SEGMENT 2 (M=7 intervening OOD/small-talk turns, then the query at K+M): the same "Tell me about the dax." now
  answers grounded prose from the LEARNED weight change.

The mechanism that makes it continuous: `run_chat`'s per-turn `_restore_state` washes the neuron dynamical state
(v/u/firing/conductances) but NOT `cp_connections.data`, so the mid-conversation weight change survives the washing
turns.

## Result (seed 42) — the closed loop holds, 8/8 GO flags

<!--derived-->

From `research/findings/raw/lanes/stageA/continuous_learn_in_conversation_s42_smoke.json`:

- CLOSED LOOP: pre-teach "dax" reply is empty (taught-recall 0); after the mid-conversation learning event and 7
  intervening turns, the same probe answers "warmly, gladly Dax eats grass." with taught-recall 3/3 — the reply at
  turn K+M reflects the learning at turn K, on ONE persistent bridge.
- PERSISTENCE: the readout weight moved by 41.471 at the teach and was byte-invariant from K to K+M (it survived the 7
  intervening washing turns) — the loop is genuinely continuous, not a re-teach at query time.
- LESION (load-bearing): restoring the pre-teach FF weights (undo ONLY the turn-K e-prop change) reverts the K+M reply
  to the untrained zero-init attractor ("Dax eats apple.") — taught-recall drops to 0. The reply RODE the weight
  change, not a host buffer. (The surface still emits the attractor patient "apple" rather than literal silence, but
  the TAUGHT content "grass" is gone; this is the design's off-index-0 attractor, self-consistent through the shim so
  the post-hoc moat does not flag it.)
- FROZEN control (no e-prop): a second persistent run, identical teaching but eprop_lr=0 (readout moved 0.0000, gate
  still imprinted) — the K+M query does not recall the taught fact (recall 0, reply "Dax eats apple."). The CONTENT
  rode the weight change.
- MOAT intact: 0 false-accepts on the untaught-cue battery; every OOD turn abstains; 0 confabulations across both
  segments; the composer kb is unchanged by teaching (acquisition is a weight change, not a host store append).
- ATTRIBUTION (tools.lab): the K+M taught-recall is 100% attributable to the mid-conversation weight change — vs the
  frozen-readout control, vs the weight-lesion, and vs the pre-teach baseline (each control recall = 0).

## Scope / honesty

<!--derived-->

- This is a 1-seed SMOKE. The exact 6-seed self-sweep (42/43/44/100/101/102) is returned but NOT run this pass. The
  runner is self-sweeping (`--seeds ...` aggregates + earns a verdict) so the parent runs one command.
- What this adds over INTEGRATION #7: the learning event is INSIDE the persistent conversation (not before it), and
  its causal effect on a later reply — same bridge, no rebuild — is measured, persisted-checked, and lesioned. It is a
  PIECE of the closed perception→internal-state→speech→consequence→LEARNING loop, not the whole loop.
- Learned-fact SET = the reliable K=3 joint-contrastive regime taught in ONE presentation at turn K. Sequential /
  continual breadth across many separate teacher-turns (frac_recalled ~ 1/N) is the OPEN continual-learning arc and is
  NOT re-litigated here — a genuinely incremental multi-turn teacher is the named next step for the continuous loop.
- Named scaffolds inherited unchanged: the familiarity gate is a numpy anti-Hebbian projector (host-idealized; the
  spiking v320 gate is the swap-in, burn-down #2); the conjunctive cue codebook + patient argmax are the
  composer-idealization / neural-motor-readout targets; the teacher/curriculum + per-turn appraisal are the legitimate
  AI-teacher host social environment; the generator mouth is OFF (the grounded content is the learned read).
- No `sim/` edit was needed. Everything is runner-side reuse-by-import of `build_one_brain` / `CoResidentEpropNet` /
  the INTEGRATION #7 teach/chat/moat/recall machinery. cfg.seed is set by `build_one_brain` (not `actual_seed_used`).
