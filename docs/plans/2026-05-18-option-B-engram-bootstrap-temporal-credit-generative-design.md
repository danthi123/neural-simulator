# Option B — engram-bootstrapped temporal-credit GENERATIVE compositional learning in the REAL spiking concept-pool substrate (design)

> Standing autonomy: documented design calls; brainstorm ->
> writing-plans -> subagent-driven-development -> pre-registered in-sim
> THREE-STATE gate -> honest propagation EVERY outcome. No
> config-cranking, no overclaim, the no-confab moat byte-identical.
> Owner-authorized verbatim: "Can you autonomously work on B? We want
> generation, not just knowledge storage."

## Goal (one sentence)

Test whether the validated Tonegawa engram-tag bind (catalog D.14 --
reward-FREE one-shot Hebbian co-activation, an independently-validated
bridge API) can BOOTSTRAP the rewarded episode the compose-bridge VOID
provably lacked (`n_rewarded = 0`), so the thrice-validated
temporal-credit/eligibility mechanism can then GENERATIVELY refine that
bootstrapped bind in a MINIMAL slice of the real spiking `sim.bridge`
concept-pool architecture.

## Why this, and why now (deliberation -- no re-litigation)

The 5x-triangulated meta-finding is established this session and is NOT
re-opened here: a principled, analytically-checkable local credit
signal is constructible/checkable (TD-critic V1; compose-abstract;
pop-transfer; PC V1 cos~0.995), but cheaply turning it into a sound
discriminating *learner* at feasible local scale is the recurring
infeasibility (dendritic/conv-FA BOUNDARY; compose-bridge
spiking-bootstrap VOID; PC-learning-loop VOID). Temporal credit is the
SOLE clean validated PASS, boundaried EXACTLY at the spiking-dynamics
integration step.

The compose-bridge VOID's own diagnosis named the precise blocker and
the precise unlock:

> "the minimal spiking population-vote readout/teacher/reward loop
> never bootstraps a single rewarded episode during training
> (`n_rewarded = 0`), so the TD error delta is identically 0 ... The
> only thing that would make the in-bridge V1 positive control soundly
> constructible is a deeper instrument-engineering pass on the spiking
> concept-pool readout/teacher/reward loop (so a rewarded episode
> bootstraps and the bridge's native eligibility->reward path
> engages)."

Option B is NOT "one more fix of the same kind." It introduces a
**mechanistically distinct, independently-validated** bootstrap:
Tonegawa engram-tagging (`start_engram_recording` /
`commit_engram_tag` / `stimulate_tag`), which is **reward-FREE and
one-shot** -- it binds a co-fired ensemble in a single encoding pass
with NO dependence on a rewarded episode ever occurring. It is the
exact mechanism that structurally dissolves `n_rewarded = 0`. The
prior compose-bridge run had no engram bootstrap; it relied on the
population-vote loop bootstrapping reward on its own, which it
provably cannot. Engram-tagging is itself multi-seed VALIDATED in this
project (87.5% stim-recall; 90% multitag retrieval) -- a *knowledge
storage* result. The open, decision-relevant question the user
explicitly authorized: does temporal-credit turn that stored bind into
a *generative* refinement (composition that GENERATES, not just
recalls)?

## Falsify-cheaply precursor -- HONESTLY INAPPLICABLE (transparently logged, NOT skipped)

A throwaway pure-numpy probe (`_probe_engram_bootstrap_td.py`, 5 seeds,
DELETED post-decision; recorded evidence preserved at
`research/findings/raw/engram_bootstrap_probe_recorded.txt`) tested
engram-one-shot-bind -> temporal-credit-refinement on an abstract
analog with controls {cold, engram_only, no_trace, permuted_boot,
wrongsign}. Result: the plain-`cold` control scored **1.000 all 5
seeds** -- the abstract task is TRIVIALLY TD-learnable WITHOUT any
bootstrap. Honest consequence (NOT spun): the cheap abstract probe
**cannot reproduce the spiking-substrate-INTRINSIC `n_rewarded = 0`
bootstrap blocker** -- plain TD already solves the abstract task, so
there is nothing for the engram bootstrap to rescue at abstract scale.
Manufacturing an abstract "failure" to make the bootstrap look
necessary would be strawmanning / config-cranking toward a desired
outcome -- forbidden by the discipline.

Therefore the cheap falsify-first gate is **honestly INAPPLICABLE**
for Option B (the blocker exists only in the spiking substrate), NOT
skipped and NOT waved away. This exactly mirrors the compose-bridge
increment's own structure: its cheap precursor (pop-transfer GREEN)
validated the *mechanism* but explicitly NOT the spiking-readout
bootstrapping; the in-sim gate decided the science honestly and
returned VOID. Here the cheap precursor cannot even be constructed
without strawmanning -- so the pre-registered in-bridge THREE-STATE
gate is the ONLY sound test, and the design pre-states this as the
honest terminus. **This is a known, transparently-recorded risk, NOT a
license to overclaim a likely outcome.**

## Architecture (maximally DRY; net-new vs reused-UNMODIFIED)

**Reused UNMODIFIED (DRY; byte-empty in every commit-scoped diff):**
- `research/runners/compose_bridge_core.py` -- the
  adversarially-hardened FIXED-bar THREE-STATE verdict with its OWN
  frozen `_CBR_V1_ACC_MIN=0.80 / _CBR_SCI_ACC_MIN=0.80 /
  _CBR_CTRL_ACC_MAX=0.35 / _CBR_MIN_SEEDS=3`. **Reused byte-UNMODIFIED**
  -- it is a pure, science-agnostic numeric THREE-STATE gate
  (instrument-validity-FIRST, fail-closed, VOID strictly distinct from
  FAIL, diverged-numeric = correctly-failed, non-numeric junk -> VOID).
  No new movable bar is introduced anywhere. The engram-bootstrap gate
  emits the same metric keys; the frozen bars are INHERITED, not
  re-pre-registered (the most anti-cheat-safe choice).
- The kill-safe CLI/verdict scaffold structure of
  `research/runners/compose_bridge_gate.py` (mirrored, not copied: the
  per-seed atomic checkpoint via REUSED `sim.train_checkpoint`,
  KeyboardInterrupt-clean-exit, recompute-from-recorded-JSON).
- The validated Tonegawa engram bridge API on `SimulationBridge`:
  `start_engram_recording` / `commit_engram_tag` / `stimulate_tag` /
  `clear_tag_drive` -- byte-UNMODIFIED (catalog D.14, already shipped +
  multi-seed validated; the reward-FREE one-shot bind).
- The validated temporal-credit/eligibility mechanism:
  `sim.compose_temporal_bind` / `sim.td_value_critic` logic +
  `sim.kernels.fused_eligibility_trace_decay` + the bridge native
  `cp_eligibility_trace` reward-modulation path + the REUSED
  `NeuromodulatorManager` C.30 TD-delta=phasic-DA -- byte-UNMODIFIED.
- `research/runners/text_minimal_isolation.py`
  `build_biological_brain_regions` (the real spiking v16 concept-pool
  setting) -- byte-UNMODIFIED.
- `sim.train_checkpoint`, `sim.neuromodulators`, `sim.bridge`,
  `sim.backend`, every frozen `*_core` (incl. `compose_bridge_core`,
  `compose_bind_core`, `td_critic_core`, `dendritic_fair_core`), the
  no-confab moat (`abstention_gate` + `tests/test_abstention_gate.py`,
  MUST stay 7/7) -- byte-UNMODIFIED. NO autograd/torch in the shipped
  path.

**Net-new (load-bearing) -- the engram-bootstrap WIRING ONLY:**
1. `research/runners/engram_bootstrap_gate.py` -- kill-safe runner.
   Per (verb, motor) compositional pair: (i) drive verb at t_A +
   motor-teacher at t_R inside an open
   `start_engram_recording`/`commit_engram_tag` window -> a reward-FREE
   one-shot engram tag spanning the verb_pool + motor co-firing
   ensemble (this is the bootstrap the prior run lacked); (ii) during
   training, `stimulate_tag` reactivates that ensemble each episode so
   a rewarded episode DOES occur (`n_rewarded > 0`), engaging the
   bridge native eligibility->reward path; (iii) the validated
   temporal-credit/eligibility trace then refines the bind across the
   TEMPORAL GAP (the GENERATIVE step under test). Conditions x seeds
   (see gate below). Imports the REUSED modules above; produces the
   `compose_bridge_core` metric keys; >= 8 distinct verb->motor
   bindings; greedy noise-free eval accuracy; NO autograd; `--tiny-synth`
   smoke shrinks pools/episodes (toy verdict NOT propagated).

**The mechanism-isolation design (load-bearing science-integrity
point -- the adversarial reviewer MUST probe this):** every condition
gets the IDENTICAL engram bootstrap (same encoding, same
`stimulate_tag`, same drive/gap/readout/reward/RNG consumption). The
conditions differ ONLY in the temporal-credit refinement on top:
- `td`: engram-bootstrap + the validated temporal-credit/eligibility
  trace refines across the gap (gamma=0.95, lambda=0.9, via the bridge
  native eligibility->reward path).
- `engram_only` (the FAITHFUL analog, NOT a strawman): byte-identical
  to `td` in EVERY respect -- same engram bootstrap, same
  `stimulate_tag`, same drive, gap, readout, reward, RNG consumption --
  EXCEPT the eligibility trace is suppressed across the gap
  (`bridge.cp_eligibility_trace[:] = 0.0` each gap step). This is the
  validated knowledge-storage baseline (engram recall ALONE). The
  science question is precisely: does temporal-credit add GENERATIVE
  refinement ON TOP of the validated engram bind? `engram_only` failing
  while `td` succeeds is the ONLY signature that isolates "generation"
  from "stored recall". If BOTH succeed equally, temporal-credit adds
  nothing generative here (honest FAIL/VOID, never spun as a win).
- `permuted`: pi (verb->motor) re-randomized per episode.
- `wrongsign`: TD delta sign-flipped.
- V1 (instrument soundness): `td` with gap=0 -- the no-gap
  engram-bootstrapped TD must learn the verb->motor bijection (proves
  the in-bridge instrument itself bootstraps + learns; directly
  addresses the prior VOID's unmet V1).

## Pre-registered in-bridge THREE-STATE gate (REUSED compose_bridge_core, frozen, NEVER tuned)

- V1: in-bridge engram-bootstrapped `td` on the NO-GAP verb->motor
  binding reaches `_CBR_V1_ACC_MIN = 0.80` (frozen, inherited
  byte-UNMODIFIED). **A sound true engram+TD no-gap learner that cannot
  meet V1 is an honest VOID, NOT a reason to soften a bar** (exactly as
  the prior compose-bridge run honestly VOIDed).
- Science: in-bridge engram-bootstrap + temporal-credit on the GAPPED
  verb->motor bind reaches `_CBR_SCI_ACC_MIN = 0.80`.
- Controls (must fail): `engram_only` (faithful storage-only analog),
  `permuted`, `wrongsign` all <= `_CBR_CTRL_ACC_MAX = 0.35`.
- THREE-STATE instrument-validity-FIRST fail-closed: VOID if V1 unmet
  or a control learns / is missing / non-numeric; **PASS iff sound +
  discriminating + science met AND `engram_only` genuinely fails**
  (the generative signature); else FAIL (sound + discriminating yet
  temporal-credit ALSO fails to add generative refinement in-bridge =
  the strongest honest triangulation that the remaining blocker is
  spiking-dynamics integration, not the temporal-credit principle).
  `_CBR_*` byte-UNCHANGED, recomputed from the recorded JSON, no
  re-run, no bar-tuning.

## Honest ceiling (stated up front, NEVER spun)

- **IS (only if PASS):** the validated engram bind, refined by the
  validated temporal-credit mechanism, produces small-scale GENERATIVE
  compositional learning in a MINIMAL slice of the real spiking
  concept-pool architecture -- the first in-architecture *generative*
  (not merely stored-recall) dent in the composition blocker, where
  the faithful storage-only `engram_only` analog cannot.
- **IS NOT (the project boundary, never spun):** open-ended fluent
  composition. NOT an LLM. NOT conversation-solved. NOT compositional
  *language*. NOT the full vocab, NOT chat-integrated, NOT scaled. This
  is a minimal-spiking-slice MECHANISM-TRANSFER test of whether
  temporal-credit converts a validated engram bind into a *generative*
  one. Open-ended fluent composition remains the honest project
  boundary at feasible local scale (Generator-G NEGATIVE; Generator-F
  coherent-simple ceiling; Phase-2.3a ~3-4 order scale gap) -- this
  increment does NOT move that boundary and will not be reported as if
  it does.
- A faithful FAIL/VOID is the strongest honest triangulation (6th
  independent direction) that the remaining blocker is spiking-dynamics
  integration, NOT the thrice-validated temporal-credit principle, and
  NOT a license to escalate. PASS/FAIL/VOID all decision-relevant +
  propagated honestly.

## Explicitly NOT in scope (YAGNI / honesty)

Full conversational composition; scaled concept-pool vocab; chat_repl
integration; predictive coding / laminar microcircuit; the
compose-bridge owner-fork (deeper readout/teacher/reward
re-engineering WITHOUT engram -- engram bootstrap is the
better-motivated, validated, reward-free alternative the user
authorized). An honest in-sim PASS/FAIL/VOID here is the terminus of
THIS increment.

## Build sequence (subagent-driven; anti-cheat) -- detailed by writing-plans

Task 0 grounding pin (commit now; green only after the gate runner
exists) -> Phase A: the net-new `engram_bootstrap_gate.py` (kill-safe;
REUSES `compose_bridge_core` verdict byte-UNMODIFIED + the engram
bridge API + the validated temporal-credit path +
`build_biological_brain_regions` + `train_checkpoint` + NM, ALL
byte-UNMODIFIED; pure-TDD where the substrate allows; the spiking run
itself validated by the gate, project pattern) -> **DEDICATED
ADVERSARIAL REVIEWER on the load-bearing runner BEFORE Phase B**
(explicitly probe: is the in-bridge discrimination genuinely isolated
to the temporal-credit *generative refinement* on top of an identical
engram bootstrap -- i.e. is `engram_only` a FAITHFUL storage-only
analog identical to `td` minus EXACTLY the gap-trace, or a strawman
crippled elsewhere? does the engram bootstrap genuinely make
`n_rewarded > 0` so the bridge native eligibility->reward path
engages? are the engram API + temporal-credit +
build_biological_brain_regions reused byte-UNMODIFIED, not
copy-paste-tweaked? can a non-discriminating/V1-broken in-bridge run
be scored PASS not VOID? are the inherited `_CBR_*` movable by
results? any autograd?) -> STRENGTHEN-only fixes, frozen bars
byte-unchanged -> Phase B LOAD-BEARING no-harm (PROTECTED set
byte-UNTOUCHED across `git diff <plan-base>..HEAD`; moat 7/7;
full suite green; assert no shipped path imports torch.autograd) ->
Task 5 CONTROLLER-ONLY: grounding-first tiny run (toy verdict NOT
propagated) + decisive kill-safe multi-seed in-sim run (seeds 42 43 44
45 46; FIXED pre-registered config) + MANDATORY anti-cheat smell-test
(scrutinize a nominal PASS HARDER than a FAIL: recompute from recorded
JSON; V1 genuine + non-degenerate; the in-bridge discrimination
genuinely isolated to the *generative* signature `td > engram_only`;
permuted/wrongsign fail; `n_rewarded > 0` confirmed; NO re-run, NO
bar-tuning, NO overclaim) + honest propagation EVERY outcome (findings
doc + capability_status pillar n=74 [status VALIDATED if PASS /
BOUNDARY / NEGATIVE] + schema test green + push BOTH remotes origin &
gitea). Task 5 is brought back to the controller, NOT a subagent task.

## Anti-cheat plan (non-negotiable)

Pre-registered FIXED-bar THREE-STATE (REUSED `compose_bridge_core`
byte-UNMODIFIED -- no new movable bar anywhere); instrument-validity-
FIRST fail-closed; dedicated adversarial reviewer on the load-bearing
runner BEFORE Phase B (mirror the TD-critic / compose-bind /
compose-bridge reviews that found real holes -> STRENGTHEN-only);
controller trust-but-verify EVERY subagent git diff with the PROTECTED
set byte-empty in each commit-scoped diff; mandatory controller
smell-test scrutinizing a nominal PASS harder than a FAIL; the honest
cheap-precursor-INAPPLICABLE finding recorded transparently (NOT spun
as "de-risked"); honest propagation every outcome (PASS/FAIL/VOID);
the no-confab moat byte-identical + 7/7 green throughout; NO
config-cranking past the pre-registered in-bridge terminus; the honest
ceiling baked in and never spun. A maxed FAIL or honest VOID (or a
caught false-PASS) is an honest finding to propagate, not a failure to
hide.

## CORRECTED OPERATING MODE -- VOID/BOUNDARY triggers PIVOT, never STOP (durable; survives compaction)

Process self-correction (owner critique 2026-05-18, accepted without
spin): prior VOID/BOUNDARY outcomes (dendritic, compose-bridge, PC)
were each followed by a "handed to the OWNER eyes-open" deferral that
functioned as a STOP. That conflated two distinct things:

- "Do NOT config-crank the SAME instrument's budget/params past a
  pre-registered terminus toward a desired PASS" -- CORRECT, kept.
- "Therefore defer the deeper architecture work to the owner" --
  WRONG. Standing autonomy explicitly authorizes autonomous
  architecture iteration; a VOID means *the cheap instrument can't
  de-risk it / this mechanism boundaried*, NOT *stop*. The anti-cheat
  discipline forbids faking a PASS; it never mandated stopping after a
  VOID.

**Corrected rule (non-negotiable, applies to THIS arc and all
following):** an honest VOID/BOUNDARY/FAIL is propagated (findings +
pillar + both remotes) and then IMMEDIATELY triggers a pivot to the
next genuinely-distinct architecture below -- autonomously,
non-stop, NO owner-deferral, NO pause. Owner forks are recorded as
*optional parallel escalations*, never as the reason to halt the
autonomous arc.

### Pivot queue -- genuinely-distinct architectures, same goal (local generative composition); each is NOT a config-crank of another

- **Q1 (active): engram-bootstrap + temporal-credit, in-bridge** (this
  design). Dissolves the diagnosed `n_rewarded=0` with a validated
  reward-free Tonegawa bind, then temporal-credit refines generatively.
- **Q2: two-module constrained decoding.** The validated coherent-simple
  generator (Generator-F) PROPOSES tokens; the validated no-confab
  grounded memory VETOES ungrounded tokens token-by-token (constrained
  decoding interface). Genuinely distinct from Generator-G NEGATIVE
  (which tried to make ONE small model both fluent and faithful). Goal:
  faithful generative composition as TWO coupled validated modules.
  Pre-registered THREE-STATE: V1 = generator alone fluent + grounded
  memory alone faithful (both already validated); science = the coupled
  decoder generates fluent text whose grounded claims are gated faithful
  above a pre-registered no-confab-preserving bar; controls =
  generator-without-gate (drifts), gate-with-shuffled-grounding (must
  fail). Honest ceiling: NOT open-ended fluency; a faithfulness-gated
  small generator.
- **Q3: laminar cortical microcircuit (Larkum apical-basal) with a
  NON-PC-training-loop learning signal.** Reuse the DURABLE sound PC
  inference (V1 cos~0.995 held) for hierarchical generative inference,
  but learn via engram-bootstrap / target-propagation -- explicitly NOT
  the PC training-loop accumulation that VOIDed. Genuinely distinct from
  the PC-learning VOID (different learning signal on a sound inference
  core).
- **Q4: concept-level pretraining objective for the surrogate-grad
  cortex, rewired into the validated v16 concept-pool substrate.**
  Phase-2.3a NEGATIVE was a char-level-next-char OBJECTIVE mismatch (an
  architecture fault, not merely scale). A concept/word-prediction
  objective + concept-pool readout is an unexplored architecture, not a
  scale-crank.

Each Qn has its own pre-registered THREE-STATE gate + honest ceiling,
written at its turn. The arc proceeds Q1 -> (on any non-PASS, propagate
+ pivot) -> Q2 -> Q3 -> Q4, non-stop, until a validated PASS or the
queue is genuinely exhausted (at which point new catalog-grounded
architectures are enumerated, still non-stop). This file is the durable
record so a compacted future context CONTINUES the queue rather than
re-deferring.
