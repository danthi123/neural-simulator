# Dendritic credit-assignment — FAIR-SCALE, literature-faithful GLR-2017 discriminating run (OWNER-AUTHORIZED, eyes-open) — Design (ACTIVE)

> **For Claude:** REQUIRED NEXT SKILL: superpowers:writing-plans (then
> superpowers:subagent-driven-development). Continuous autonomous arc;
> documented design calls; no stopping/asking. OWNER explicitly chose
> Option 2 (2026-05-17): deliberate, pre-registered, eyes-open
> large-scale investment in the literature's discriminating regime.
> Week-scale local GPU authorized; kill-safe/pausable is a HARD
> requirement. Self-contained at RUNTIME; public dataset authorized;
> no external LLM; no cheats.

## Why this is owner-authorized, NOT config-cranking

Five independent directions (generation, realization, grounded memory,
the faithful dendritic test, the LLM-teach-then-wean steelman) all
triangulated the same conclusion: the dendritic credit-assignment
rule's FORM is correct (weight-transport cos = +1.0) but its genuine
contribution is **not discriminable at feasible CHEAP local scale**
because a readout over rich hidden features rescues any hidden rule.
The honest options were handed to the owner. The owner chose Option 2:
run the literal Lillicrap-2016 / Guerguiev-Lillicrap-Richards-2017
regime on a REAL dataset where the readout confound is genuinely
defeated, as a deliberate eyes-open investment despite no cheap
positive precursor. This is a legitimate strategic override of the
cheap gate, not autonomous config-cranking. Outcome decision-relevant
either way; propagated without spin.

## Thesis

On MNIST (the canonical feedback-alignment benchmark, where a linear
readout over random/wrong hidden features provably cannot reach high
accuracy -- the confound is defeated by task structure), train a deep
MLP whose hidden layers learn ONLY by the committed sign-correct LOCAL
Urbanczik-Senn rule (`sim.dendritic_plasticity`) driven by per-layer
FIXED-RANDOM feedback (feedback alignment: no weight transport, no
autograd at runtime), output layer by a local delta. The genuinely
discriminating question the cheap probes could not ask: does the
biologically-LOCAL rule genuinely learn MNIST (>= a fixed high bar,
multi-seed) the way GLR-2017 says it should, AND does a WRONG-SIGN
control genuinely FAIL (proving it is the local credit assignment, not
the readout, doing the work)?

## Pre-registered gate (FIXED bars in a NEW `dendritic_fair_core`, never tuned)

Own frozen module constants (do NOT mutate `dendritic_core`'s `_DEND_*`
or any other frozen core). Concrete FIXED bars (justified; never
tuned), MNIST 10-class (chance 0.10):

- **Instrument-validity gates (a result is VOID unless BOTH hold):**
  - V1 (positive control works): the true-gradient ORACLE
    (bptt-style; measurement/tests only, NEVER in the shipped learning
    path) reaches heldout **>= 0.95** -- proves the harness +
    optimization are sound (the cheap probes failed exactly here).
  - V2 (instrument discriminates): the WRONG-SIGN local rule heldout
    **<= 0.30** -- proves MNIST defeats the readout confound (a
    rescued wrong-sign would void the instrument, as at cheap scale).
- **Load-bearing PASS (all, multi-seed >= 3, permuted control, never
  tuned):** correct-sign dendritic-LOCAL heldout **>= 0.90**; AND
  wrong-sign **<= 0.30** (V2); AND global-scalar baseline (the
  point-neuron / W->A-verdict anchor) **<= 0.30**; AND emergent
  feedback-alignment cosine (local update vs true gradient) INCREASES
  over training and ends **>= 0.30** (the Lillicrap signature); AND
  the permuted-label control **<= 0.30** (a result that does not beat
  its own permuted control is not real -- the 2026-05-03 catcher).
- **MANDATORY anti-cheat smell-test:** scrutinize a PASS HARDER than a
  FAIL (the Generator-S lesson); recompute from recorded JSON; read
  the training curves + the wrong-sign/permuted transcripts; no
  re-run; no bar-tuning.

PASS (scrutinized) => the #1 credit-assignment lever genuinely works
biologically-locally at the literature's discriminating scale on real
data. HONEST CEILING, never spun: this addresses the credit-assignment
ROOT only -- it is NOT #3 (developmental/embodiment) and NOT
"conversation solved"; a PASS makes integration into the conversational
stack a separate, large, later effort. FAIL at the literature's OWN
discriminating regime, locally => the strongest possible honest
terminus (the lever does not work at feasible local scale even where
it is supposed to). Either way decision-relevant, propagated without
spin, NOT config-cranked.

## Architecture (net-new harness; validated stack reused UNMODIFIED / DRY)

Recommended **Arch A** (B = CIFAR-conv, C = bio-W->A-integration are
explicit LATER increments, conditional on A; NOT in this design):

- **Dataset (self-contained):** MNIST 28x28, fetched ONCE from a public
  source and cached as a local `.npz` (mirror the
  `research/runners/corpus_fetch.py` cache discipline: download-once,
  verify, cache; thereafter zero external dependency -- self-contained
  at runtime; public dataset is owner-authorized). A clear NOT-RUNNABLE
  exit if the cache is absent and offline.
- **Net-new `sim/dendritic_mlp.py`** -- a deep rate MLP with segregated
  per-layer compartments + per-layer FIXED-RANDOM feedback matrices
  (set once from seed, never learned, never derived from forward
  weights -- no weight transport; spy-asserted). Numerically robust
  (input normalization, He/appropriate init, update/grad clipping,
  stable softmax) so V1 (oracle) genuinely trains -- the cheap probes'
  failures (vanishing/exploding) are a SOLVED engineering detail here,
  pre-registered as instrument-validity (V1), NOT a science result.
  Hidden learning delegates to the committed `sim.dendritic_plasticity`
  (byte-UNMODIFIED); output by local delta. NO torch.autograd/backward
  anywhere in the shipped learning path (biologically-local by
  construction; asserted).
- **Net-new `research/runners/dendritic_fair_core.py`** -- pure
  FIXED-bar verdict + multiseed aggregate with its OWN frozen
  `_DFAIR_*` constants + V1/V2 instrument-validity gates (fail-closed
  if either validity gate unmet, missing control, non-finite). Mirrors
  the hardened `generator_h_core`/`dendritic_core` discipline; imports/
  modifies NONE of them.
- **Net-new `research/runners/dendritic_fair_gate.py`** -- the
  kill-safe/pausable runner. Reuses `sim.train_checkpoint`
  (atomic `.tmp`+`os.replace` save/load/`resume_epoch`) + the
  validated `scaled_subword_lm_train` KeyboardInterrupt->clean-
  checkpointed-exit pattern (per-epoch checkpoint; resume from last
  completed epoch; multi-seed resume-stable). Trains, per seed, the 5
  conditions {oracle, correct-local, wrong-sign-local, global-scalar,
  permuted-label}, records per-epoch curves + emergent-alignment
  cosine + heldout, writes JSON + ASCII verdict, `<3 seeds -> exit 2`,
  ASCII-only, honest-ceiling banner. Honest propagation = CONTROLLER's
  post-run job.
- **Reused byte-UNMODIFIED (verify empty-diff each commit):**
  `sim.dendritic_plasticity` (the credit-assignment core),
  `sim.dendritic_neuron`, `research/runners/abstention_gate.py` +
  `tests/test_abstention_gate.py` (the distinctive no-confab moat --
  MUST stay byte-identical + green), every frozen `*_core`
  (gate/song_g1/subword_lm/generator_g/generator_h/dendritic),
  `sim/bptt_snn*` (true-gradient ORACLE -- tests/measurement ONLY),
  `sim/bridge.py`, `bio_three_factor`. NO new global bar.

## Kill-safe / pausable (HARD owner requirement)

Per-epoch atomic checkpoint via `sim.train_checkpoint` (full state:
all layer weights, all fixed-random feedback matrices, per-seed RNG,
epoch, condition, alignment history). `KeyboardInterrupt` flushes a
final checkpoint and exits 0 cleanly; re-invocation resumes from the
last completed (seed, condition, epoch). The week-scale run is
interruptible at any point with zero lost work beyond the current
epoch. Multi-seed loop is resume-stable (seed-deterministic data +
init).

## Honest ceiling / risks (no overclaiming)

- The decisive risk is V1/V2: if the deep MLP cannot be made to train
  stably (V1) or MNIST does not defeat the confound at the chosen
  net (V2), the instrument is void -- that is an honest "instrument
  not soundly constructible even at fair scale", propagated as such,
  NOT spun and NOT a science PASS/FAIL of the rule.
- A PASS is reported STRICTLY as "biologically-local credit assignment
  works at the literature's discriminating scale on MNIST" -- NOT as
  conversational capability, NOT #3, NOT an LLM. Integration is a
  separate later effort.
- Self-contained at RUNTIME (trained weights only; MNIST cache is a
  training-time public dataset; no external LLM/autograd at runtime).
- Local 3090; numpy/CPU acceptable for MNIST-MLP scale, GPU optional;
  kill-safe; ASCII-only. Comfortably within the authorized week with
  margin; B/C deferred to stay within budget.

## Out of scope (YAGNI; explicit later increments)

CIFAR-conv (Arch B) and bio-W->A integration (Arch C) only if A
PASSES. No predictive-coding microcircuit. No modification of any
validated/frozen module. No autograd in the shipped learning path. No
config-cranking; an A FAIL at the literature's own discriminating
regime is the honest terminus.

## Scientific basis

Lillicrap, Cownden, Tweed & Akerman 2016 (random feedback / feedback
alignment, MNIST, training-emergent alignment -- the canonical
discriminating benchmark); Guerguiev, Lillicrap & Richards 2017
(segregated dendrites, spiking, deep); Sacramento, Costa, Bengio &
Senn 2018; Larkum 2013; Urbanczik & Senn 2014. The hardened
pre-registered FIXED-bar multi-seed + V1/V2 instrument-validity +
permuted-control + mandatory-smell-test discipline is the adjudicator.
