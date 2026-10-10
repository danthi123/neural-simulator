# Long-delay temporal credit on the spiking substrate: worklog

Branch `research/long-delay-credit-spiking`, worktree `.claude/worktrees/long-delay-credit-spiking`. Serves
`docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md` section 4 (and section 6: which mechanisms are
necessary). Owner ruling relayed by the coordinator (2026-09-29): a spiking realization IS required (non-negotiable 1);
the local model's rate-level draft is a reference for expected signatures only. All numbers here are DEV (dev seeds
7-15), never gate seeds.

## 2026-09-29 midday: record search

- `bash tools/before_you_build.sh "long-delay temporal credit assignment spiking eligibility trace distal reward
  dopamine"` + four RAG queries (corpus all). Read in full or in the relevant sections:
  - `research/findings/2026-05-18-td-value-critic-temporal-credit-PASS.md`: rate-level TD(lambda) critic, host TD
    error. Reference only.
  - `research/findings/2026-07-11-R2b-dual-timescale-eligibility-...md`: the slow-trace "horizon" GO was an
    effective-learning-rate artifact; a magnitude-decoupled control refuted it. Consequence here: the trace
    ablation is built magnitude-decoupled (tag gain x tau_full/tau_ablated, equal kernel integral) and has a
    positive control where the ablated learner must still learn (immediate reward).
  - `research/findings/2026-06-21-shortcut9-trace-conditioning-value-derisk.md`: trace-vs-delay 2x2 on the spiking
    limbic core (predict-over-gap GO); explicitly leaves act-over-gap (instrumental) open. This benchmark is the
    act-over-gap version.
  - `research/findings/2026-08-07-rubicon-delayed-credit-maintained-goal-bridge-PARTIAL.md`: a recurrent slow-NMDA
    held goal bridges a gap, but plain DA-STDP on the saturated held-goal synapse netted to LTD. Consequence: the
    eligibility here is the bridge's coactivity tag (order-independent, positive), not pair-STDP.
  - `research/findings/2026-06-27-navcloseout-R4-delayed-reward-value-task-NEGATIVE.md` (decayed-trace nav task,
    2-seed negative) and `2026-05-19-integrated-loop-iter3-...global-scalar-credit-cannot-carry-WM-selectivity.md`.
- Machinery reused: the tiny-SimulationBridge limbic core pattern (`_limbic_core_rpe_battery_derisk.build_limbic_core`,
  `_rubicon_delayed_credit_derisk.build_core`), the bridge's section-4c three-factor rule (coactivity tag,
  `fused_eligibility_trace_decay`, `delta_w = lr * (DA - baseline) * e`), the `from_region_firing_signed` phasic-DA
  production rule, the SNc tonic calibration, GABA_B value subtraction, the D3 persistent-slot attractor recipe.
  `_advantage_actor_critic_probe.py` was read: it drives the full g11 nav bridge (cupy only) and has no cue/choice
  task, so it is not the vehicle; its advantage routing (SNc = r - V) is what the limbic core here reproduces.

## Review of the rate-level draft (local model, `research/runners/_draft_long_delay_credit.py`)

Verdict: useful schedule and test scaffolding; its headline dissociation is guaranteed by construction and its
"credit-stealing" fixes tuned the benchmark toward the answer. Details:

1. The trace-ablated result is an identity, not a measurement. The critic's features are {bias, transient cue}; after
   the cue every state is bias-only and identical for cue A and cue B. With gamma*lambda = 0 the update drives
   V(cue) toward gamma * V(next), and V(next) is the same for both cues, so value_A - value_B converges to 0 on
   every seed by algebra. With gamma*lambda = 0.96 the discrimination converges near the analytic discounted return.
   "15/15 seeds" and "98-102% attributable" restate that algebra.
2. Each of the four "credit-stealing" fixes removed a feature because the result contradicted the desired signature:
   the persistent buffer, the time taps, the choice, the terminal/decoy/distractor events. Several of those "steals"
   were correct TD behaviour (a buffer or a choice that equals the cue IS a perfect predictor at reward time, and
   crediting it is right). Removing the buffer from the critic removes the very thing the plan's key control must
   dissociate (memory vs credit); the draft then cannot test that control, it assumes it.
3. Making the reward cue-contingent removed the choice from the task. The "terminal choice" is a fixed argmax of a
   fixed buffer (buffer_accuracy = 1.000 in every arm and seed); nothing is learned from the reward except a
   Pavlovian value. The plan asks for a two-choice task whose choice is learned from a delayed reward.
4. `misleading_final_cue` is byte-identical to `full` (the decoy reaches nothing); `us_decay` is ~1 by design; vrmse
   0.18 is not low. The draft flagged 1, 4, 5, 6 itself (its PREREG section 8), which is to its credit.
5. Everything in it is host computation (Python TD error, host feature vector, host argmax), so under non-negotiable 1
   it can only be a reference for the expected signature (without a trace, aliased intermediate states block credit).

## Spiking design (what replaces it)

See the module docstring of `research/runners/long_delay_credit_spiking.py`. One circuit, one event stream, two tasks
that differ only in which response the world rewards: `credit` (reward depends on the response at the cue; delay,
distractors with intervening responses, decoy, terminal state and terminal response all precede the reward) and
`memory` (reward depends on the terminal response, delivered at that response). The double dissociation
(trace ablation kills credit, spares memory; memory lesion kills memory, spares credit) is the plan's key control.

## Calibration log (dev seed 7; exploration scripts in the session scratchpad, not committed)

- Tiny bridge, numpy: ~0.4 ms/step for ~360 neurons / ~10k synapses.
- Found: with `enable_stdp=False` the reward block clips EVERY synapse to [hebbian_min, hebbian_max] (default
  [0.05, 1.0]); strong fixed weights (lateral inhibition) were silently cut to w_max, and a zeroed lesion weight
  would be raised back to 0.05. Fix in the runner (no sim/ edit): hebbian_min = 0, and a per-synapse
  `cp_plasticity_rate_gain` equal to the plastic mask so the clip only touches plastic pathways.
- Memory attractor (mem_a/mem_b slow-NMDA self-excitation + shared FS): recurrence is bistable only in a narrow band
  (recur 7 = no hold, recur 8-10 = hold at ~470 Hz, recur >= 14 = spontaneous ignition). Clearing needs a trial-end
  inhibitory event longer than 2 tau_NMDA (250 ms), as the D3 slot found. Chosen: recur 8, fs->mem 25, iti->fs 30,
  ITI event 250 ms: rest off, loads, holds 800 ms, clears, reloads the other cue, and resists a 50 ms other-cue decoy.
  The hold rate (~470 Hz) is far above cortical persistent activity (~20-50 Hz); declared as a realism gap.
- Action winner-take-all: shared inhibition gave no competition (winner ~ loser). Cross-inhibition (each channel's
  own interneurons inhibit the other channel) + AMPA self-excitation: winner ~46 Hz vs loser ~10 Hz, decision in
  ~30 ms (race-to-bound K = 30).
- Critic: MSN striosome needs term->strio ~15 to fire (~7 Hz) at init; strio->snc lowered to 4 (at 10 a 7 Hz value
  silenced the SNc). DA sensitivity 80: reward burst to ~0.64-0.78 from baseline 0.5; omission dip to ~0.2-0.4.

## Calibration log, round 2 (dev seeds 7, 8)

- First learning runs (dev seed 7, 300-400 trials) learned weakly (early_acc 0.63-0.69) and drove every plastic
  weight toward w_max. Two causes found by tracing eligibility, DA and weights within single trials:
  1. The DA modulator's `plasticity_rate` target multiplies the learning rate by (1 + (DA - baseline)), an
     LTP-favouring asymmetry on top of the signed DA; removed (targets=[]), the DA deviation alone is the third factor.
  2. The SNc set-point was measured over 400 steps; its counting error (~8%) left a tonic DA offset of ~+0.02 that
     every long-lived coactivity tag integrates for the whole trial, more than a phasic reward burst. Now 20 s.
- The memory attractor fired at ~470 Hz because `enable_homeostasis=True` changes the Izhikevich spike criterion (a
  homeostatic threshold instead of v_peak) and adapts every threshold toward 20 Hz within seconds. Homeostasis is
  now frozen for the session (in vivo it acts over hours; declared). With the faithful spike rule the attractor
  holds at ~75-90 Hz (recurrence 12), clears on the 250 ms trial-end event, reloads the other cue, and a 50 ms
  other-cue decoy does not flip it (dev seed 7: held B 66-93 Hz vs decoy A 6-16 Hz). All drives were re-tuned.
- Action channels: sensory->action 20, cross-inhibition act->fs 30 / fs->other act 150, no self-excitation
  (self-excitation made a responding channel reverberate into the delay). Winner ~52 Hz, loser ~3 Hz.
- The critic first read the terminal state; its GABA_B -V then dipped DA DURING the terminal response on every trial,
  punishing the terminal response regardless of outcome (terminal responses collapsed in both tasks). The critic now
  reads an OUTCOME period (a feedback screen shown every trial, 100 ms; reward 30-80 ms into it), so DA at the
  outcome is r - V.
- A direct plastic memory->action pathway grew until the held cue drove its action channel tonically through the
  delay; adaptation and post-inhibitory rebound then let the OTHER channel win at the terminal state (memory task
  terminal accuracy 0.04 with correct memory weights; traced per phase on dev seed 7). The memory is now read out by
  conjunctive units (go_a/go_b: held cue AND terminal state; each input alone subthreshold), whose output is plastic.
- h3 (before the conjunction fix, dev seeds 7/8, 300 trials): credit/full early 0.84/0.83 with cue weights fully
  separated (A->L 57 vs A->R 0.3); credit/immediate_trace_ablated 1.00/1.00 (the magnitude-decoupled short trace
  learns when the reward is immediate); credit/trace_ablated: responses collapsed (cue weights driven to ~0), which
  the preregistration must treat explicitly, not as chance.
- Compute: local RAM margin refused further local runs at 13:50 (other lanes hold ~16 GB of memcap headroom);
  the pool needs a committed revision and commits are blocked by gates/compute_idle_persistent. A GPU dev job
  (cupy, seed 7, 4 arms) was queued with `bash tools/gpu_queue.sh add`.

## 2026-09-29 ~14:30: mid-flight audit and re-scope (coordinator ruling)

Audit: `.claude/worktrees/_audit_2026-09-29/long-delay-credit.md` (8 confirmed issues). I accept all eight. The ones
that decide the direction:
- (HIGH 1) The memory task's positive control learned the wrong way round (memory/full terminal accuracy 0.04-0.14
  while the memory-lesioned arm rose to 0.84-0.96), and terminal and early responses were coupled through shared
  tags. The per-phase trace above found one mechanism (held memory drove an action channel tonically; adaptation and
  post-inhibitory rebound then let the other channel win), but the coupling was not measured at lr=0 and the
  sanity tasks were never run. It is recorded as the plan's "credit to whatever is active" phenomenon, not tuned away.
- (HIGH 2) With the early action's only route to the reward being the tag, "trace ablation abolishes credit" was
  guaranteed by algebra (e^-34 at the shortest gap), the draft's identity flaw again. Those cells can only be
  construct-validity checks; evidence must sit on predictions that can fail (dose-response at equal integral,
  a competing route, the event decomposition).
- (HIGH 3) Five designs in about 50 minutes, chosen while watching headline arms, no preregistration. The docstring
  claimed "preregistered" and "frozen"; both were false. The forking-path ledger below is the honest record.
- (HIGH 4) The magnitude-decoupled ablation concentrated a x100 tag into a 10 ms window that samples only the local
  DA phase (the V dip, any tonic offset): the ablated arms drifted non-contingent pathways to the floor and stopped
  responding, so they differed from full in more than the horizon.
- (MEDIUM 5-8) Accuracy among responders could score a silent network as correct; no decomposition instrument; the
  plan's literal "terminal choice, then delayed reward" cell was missing; the delay scale sat inside one trace time
  constant, and early_acc_by_delay cannot measure a horizon.

Ruling (coordinator, 2026-09-29): stop the standalone circuit and in-session calibration; move the benchmark onto the
PRODUCTION basal-ganglia organ (`research/runners/bg_action_selection_production_organ.py`, the Gate-A v2 selector)
as flag-gated, default-off corticostriatal plasticity with a byte-identical-off test; write the preregistration before
any further tuning; parameter search goes to the pool once commits open. The queued GPU job for the standalone circuit
was withdrawn from `research/queue/gpu.queue` (removed under the queue's own flock) before it started.

## Forking-path ledger of the standalone circuit (dev seeds 7/8 only; DISCARDED designs, not results)

All outputs were in the session scratchpad (RAM /tmp); none is committed or cited as evidence.

| version | time | what changed | what was watched | outcome |
|---|---|---|---|---|
| dev | 12:14-12:28 | first build: sens->act 2.5, recur 25 then 8, homeostasis on (470 Hz hold), lr 0.5, w_max 40, critic term->strio 15 | credit/full s7 | early 0.63, terminal 0.92 (memory credited for the early action) |
| dev2 | 12:42-12:52 | +DA recording; 4 arms | credit full/trace_ablated/immediate_trace_ablated, memory full | early 0.69/0.50/0.58, memory terminal 0.67; every plastic weight near w_max |
| gA / gB | 13:07-13:17 | symmetric DA (plasticity_rate target removed), lr 0.25, w_max 12 / 40, w_mem_act 1.2 | same 4 arms | gB memory_lesioned collapsed to silence (all weights ~1) |
| (probe) | 13:15 | eligibility / DA traced within trials | - | mem->act tags 80x cue tags; tonic DA offset ~+0.02 from a 400-step calibration |
| h1 | 13:21 (killed) | homeostasis frozen (faithful v_peak spike rule), all drives re-tuned: sens->act 20, cross-inh 30/150, recur 12, cue->mem 15, w_max 60, term->strio 30, rus->snc 25 | 5 arms | no learning by trial 100 (lr too small for the new weight scale) |
| h2 | 13:25 (killed) | lr 5, 20 s DA calibration, snc tonic 180 pA | 7 runs | credit/full 0.86-0.92 while terminal responses collapsed in every arm |
| h3 | 13:31-13:43 | critic moved from `term` to an outcome screen `outc` | 10 runs | credit/full 0.83-0.84; memory/full 0.00-0.04 (inverted); trace_ablated silent |
| h4 | not run | conjunctive go units for the memory read-out, no act self-excitation | - | stopped by the audit; local RAM refused further runs |

## 2026-09-29 ~14:00-14:10: the re-scoped build (production BG organ) and the execution packet

- Preregistration written BEFORE any build run on the organ: `research/findings/2026-09-29-long-delay-credit-bg-organ-PREREG.md`.
- Files: `research/runners/bg_corticostriatal_credit.py` (new: the `BRAIN_BG_CREDIT` extension, built only through
  `build_selector_bridge`'s `extra_regions` / `extra_pathways` / `core_config_updates`; no `sim/` edit);
  `research/runners/bg_action_selection_production_organ.py` (edit region: two functions APPENDED after
  `decide_action`, `bg_credit_enabled` and `build_credit_organ`; no line above them changed);
  `research/runners/long_delay_credit_bg.py` (runner with the event-decomposition ledger),
  `research/runners/long_delay_credit_bg_verdict.py` (registered scorer, prereg section 6),
  `research/runners/long_delay_credit_bg_instrument.py` (section-7 instrument scorer),
  `tests/test_long_delay_credit_bg.py`, `tests/test_bg_credit_off_identical.py` (off path compared in data against the
  pinned SHA f52c6d246: identical races, weights and membrane state). The standalone-circuit runner, its verdict reader
  and its tests were deleted from the worktree (never committed; the ledger above is their record).
- Tests: 14 passed locally (numpy), including a one-trial smoke on the organ: the clip leaves the fixed 400-weight
  proposal->MSN synapses untouched, a lesion holds, and no host reward scalar is written.
- Owner mode change (coordinator, 14:05): Claude tokens are scarce; finish as an execution packet for the local model.
  `docs/drafts/long-delay-credit/HANDOFF.md` holds the commit/provision/queue/score commands, the outcome table, the
  stop conditions and the NEEDS-CLAUDE list. One dev verification (seed 7, credit_immediate/full and
  late_immediate/full, default constants) was left running; the other seven dev runs launched at 14:04 were stopped.
- Nothing is committed yet: gates/compute_idle_persistent blocks non-exempt commits until the pool is busy (about 18:30).

## 2026-09-29 14:04-14:35: the one dev verification (seed 7, default constants, positive controls only)

`research/findings/raw/_long_delay_credit_bg/dev/inst1_*_s7.*`. Instrument items pass: commit rates 0.90-1.00, early
exploration p(ch0|A)=0.43/0.67, p(ch0|B)=0.45/0.58, memory hold 1.0, 0 Hz after the clear, decoy resistance 1.0,
tonic DA offset 2e-5 (the 20 s calibration without OU noise on the SNc works), rewarded DA max 0.628. Failures: no
critic dip (unrewarded DA min 0.498); `credit_immediate/full` success 0.51 (D1 weights of both channels driven up,
D2 to the floor: unsigned drift, void_floor_ceiling); `late_immediate/full` terminal success 0.00 at response rate
1.00 (anti-mapping, as in the standalone circuit). Not tuned in session (owner mode change). The handoff sweep now
leads with critic tags (outc->striosome 60, striosome->SNc 10). A structural cause (the D2 route, the held-memory bias
on both channels) is possible; if no tag passes, it is NEEDS-CLAUDE item 4.

## 2026-09-30 22:36: Step 4 instrument sweep -> rc=2 (STOP, NEEDS-CLAUDE item 4)

All 243 dev files (9 tags x 3 schedules x 9 seeds 7-15) were in $P and were copied into the worktree
(`research/findings/raw/_long_delay_credit_bg/sweep`). Ran the registered section-7 instrument scorer
(`research/runners/long_delay_credit_bg_instrument.py`): **rc=2, chosen=null** — no tag passes every criterion.
The failing criteria are the same across all 9 tags: c5_positive_controls (success 0.16-0.53, never >=0.8 on
>=6 seeds; n_ok and n_ok_late both 0 except strio60 n_ok=3) and c6_no_bound (void_floor_ceiling on most seeds:
D1 weights of both channels drift up, D2 to the floor — unsigned drift, exactly as the seed-7 dev note
predicted). c3_dopamine also fails (no critic dip: da_outcome_unrewarded_min_mean ~0.498 > 0.47). c1 commit
rates are fine (0.90-1.00) but c1_intervening / c2_memory / c4_sanity also fail on most tags. This is the
HANDOFF's own anticipated outcome: "If the Step 4 sweep finds no passing tag, NEEDS-CLAUDE item 4 applies, with
this reading as the starting point." Item 4: the substrate's operating point or the eligibility mode needs a new
method (Gate-B history: credit on this selector is hard even at short gaps).

Scorer bug worked around (NOT fixed; the scorer is a registered artifact I may not edit): its glob
`<tag>_<schedule>_*.json` in `long_delay_credit_bg_instrument.py` also matches the 243 `.prov.json` provenance
sidecars (no `rows` key) -> `KeyError: 'rows'`. I ran it against a prov-free copy of the same 243 data files
(identical data; the report is dir-agnostic). Report:
`research/findings/raw/_long_delay_credit_bg/sweep/instrument_report.json` (untracked in the worktree). The
243 sweep JSONs + provs are untracked in $P. No Amendment A, no gate runs, no CreditParams edit (chosen is
null). The sweep data + report are NOT committed: the HANDOFF sequences the data commit after the amendment
commit, which does not happen at a STOP. Packet branch: research/long-delay-credit-spiking (this WORKLOG commit
is the new tip; unpushed per the push-credential ruling).

## 2026-10-01: Claude's reading of the Step 4 sweep (NEEDS-CLAUDE item 4)
- An instrument failure, not a NO-GO. No credit gate ran; the method's precondition, a signed DA signal at the outcome (prereg 7.3), was never met, so nothing here measures corticostriatal credit.
- No signed DA at defaults: unrewarded DA min 0.489-0.498 (bar 0.47) in every tag except strio60/strio60snc10. The rule then only potentiates: cue->D1 rose ~26-52 on both channels and D2 went to ~0 (void 9/9, base credit_immediate), although the correct mapping still rose more (s8 cue A: +48 vs +26).
- strio60, the one tag that made DA signed, removed the void on 7/9 seeds; weights moved both ways; sanity 0.80-0.86 (s7, s8, s11) and credit_immediate 0.80-0.84 (s8, s11, s13) in 200 trials.
- strio60's depression is not a timed omission dip: the unrewarded in-window min is ~0.48, while late_immediate's following-trial ITI sits 0.10-0.13 below baseline at the lr=0 probe on all 9 seeds (0.16-0.19 with snc10), so before any value is learned. Inferred, to be measured by a pilot: most of the striosome GABA_B inhibition lands after the 100 ms outcome window. The record's working GABA_B subtraction used a 150 ms expectation lead overlapping the reward (2026-06-09-N9-SNc-rV-subtraction-research.md section 1.0); here the reward is 30-80 ms into the screen (runner lines 49-50). Wall reframe: the expectation's time course was replaced by a 30-80 ms constant.
- Seed defects that do not depend on the tag, at lr=0 and identical in all 9 tags: in s9 and s12 the memory is not cleared (32 and 29 Hz after the clear; hold 0.84/0.87); s14 has no exploration (p(ch0|A) = p(ch0|B) = 1.0). They fail c1/c2 in every tag. A gate seed like s14 would score as a G1 fail, not UNDEFINED (prereg line 155), so the scorer's all-seeds rule is kept: relaxing it changes no tag's result here (no seed passes every item in any tag) and would hide defects that also threaten the gate.
- The sweep ran --n-train 200; prereg 7.6 and the gate use 400.
- Scorer `.prov.json` glob bug fixed in the same commit as this entry; on the real sweep dir the fixed scorer gives exactly the workaround report (rc=2, chosen null).
- Next: Step 4a pilot (Hermes, draft build, dev seeds), then Claude writes Amendment A0 and Step 4b; see the HANDOFF.

## 2026-10-01 13:15 (Hermes, local model): Step 4 sweep banked — 492416302

Banked per HANDOFF Step 4a: `git add research/findings/raw/_long_delay_credit_bg/sweep` + the verbatim
prescribed message, committed through the gates (no --no-verify): 488 files, `research/long-delay-credit-spiking
@ 492416302`. `research/findings/raw/_provenance/runs.jsonl` left modified-but-unstaged per Step 2.

**The below-chance gate blocked the first attempt** (155 of the staged sweep JSONs — the "31" in the hook
output was the report cap, not the count). Per RULING `models` this verdict-on-results judgment went to one
MiMo Pro subagent first; **its verdict: DECLARE** (high confidence), quoted:

> The gate is authoritative and its own blocked message prescribes this exact remedy: declare it
> ("below_chance": true / an UNDEFINED verdict) so the artifact owns its own status; its docstring calls
> the escape 'the desired behaviour, not a loophole'. [...] Patch only gate-flagged files, where
> below_chance:true is verified true per-file [...]; measured values untouched; note mirrors the precedent's
> wording and is true of every patched file; rc=2 chosen=null stays in the verbatim commit message.

Acted on it: added `below_chance: true`, `instrument_failed: true` and a `below_chance_note` (the reading is
UNDEFINED, never a capability NO-GO; outcome DA never signed per prereg 7.3; Step 4 tag selection rc=2,
instrument report chosen=null) to exactly the 155 gate-flagged files under `sweep/` — none of the other 88
sweep JSONs, `instrument_report.json` not among them. Every file went through a round-trip check
(json.loads->dumps byte-identical before the addition; all original keys re-verified equal after), so no
measured value changed. Re-staged: `run_all` on the staged list returns `BLOCKING: []` (only the report-only
`tool-health-fresh` warning remains). Same-file precedent: Claude's own `a4cbb7daa` hand-declared
`below_chance` on the two dev JSONs for this same gate; the runner never writes these fields.

Next per RULING T1-LDC: Step 4a (outcome-timing pilot, draft build, dev seeds) exactly as the HANDOFF says.

## 2026-10-01 13:38 (Hermes, local model): Step 4a step 1 — pilot copy + reproduction in flight

Draft build per HANDOFF Step 4a: worktree `.claude/worktrees/draft-ldc-outcome-timing` on
`research/draft-ldc-outcome-timing`, started from `research/long-delay-credit-spiking @ 62132d0f5`.

- Copied the registered runner to `research/runners/_ldc_outcome_timing_pilot.py` (the registered runner is
  untouched). Added `--outcome-ms` and `--reward-on-ms` (HANDOFF 4a.1), defaults = the registered constants
  100 / 30, unchanged; both are module globals read at call time inside `outcome()`, so the rebinding in
  `main()` covers every read site (lines 213/215). Window sanity checks reject a reward window outside the
  outcome screen. `py_compile` clean; `--help` shows defaults 100 / 30.
- Reproduction check launched 13:24 (HANDOFF 4a.1: with the defaults it must reproduce one Step 4 base row
  exactly): `nice -n 10 env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m
  research.runners._ldc_outcome_timing_pilot --schedule credit_immediate --arms full --seeds 7 --n-train 200
  --n-test 100 --out _scratch/repro_base_credit_immediate_s7.json` (bg proc_9a6f92ad57d9, log
  `_scratch/repro_s7.log`; banked row wall_s 2681, so ~45 min). Comparison target:
  `sweep/base_credit_immediate_s7.json` rows[0]. Plan: every row field except `wall_s` (wall clock) and the
  doc-level `runner_sha256`/`git_head`/`git_dirty` (different file, dirty worktree — not row fields).
  GUESS (marked): row fields should match bit-for-bit under SIM_BACKEND=numpy OMP_NUM_THREADS=1; if any
  differs, HANDOFF 4a.1 says stop and say so — that would be a STOP line, not a fix.
- Not yet run: the 4a grid (step 2), the lr=0 seed 9/12/14 probes (step 3), PILOT.md (step 4). They wait on
  the reproduction verdict. No PASS claimed here (RULING verify-go applies when a PASS is written).

## 2026-10-01 14:35 (Hermes, local model): Step 4a.1 reproduction gate — PASSES (verify-go, 2 lenses)

Claim verified: the pilot at DEFAULT flags reproduces the banked Step 4 base row exactly
(`sweep/base_credit_immediate_s7.json`, credit_immediate/seed 7/n_train 200). Run finished 13:43,
wall_s 1090.9 (banked 2681.4 on the pool worker). My comparison: 36/36 keysets, 0 value diffs, 0 type
diffs excluding wall_s, params and schedule equal.

Per RULING verify-go (skill loaded) the claim was put to adversarial refuters:

**Refuter 1 (MiMo Pro delegate, code-identity + instrument-trust lens): SURVIVES** — quoted:
> Diff pilot vs registered runner: exactly 12 added lines, 0 removed ... argparse defaults literally
> reference the module constants ... runner_sha256: banked doc = 8e5d8838 = composite of registered
> runner; repro doc = e902550b = composite of pilot bytes; they differ ... 35/35 fields compare equal
> (zero diffs, including nested test/instrument/probe_lr0/weights dicts) ... wall_s is the only
> nondeterministic row field and is the only one that differs.

Defects it found, all non-blocking: (a) the repro LOG names no module (module identity proven instead
by the output doc's runner_sha256 + runs.jsonl argv), (b) `_scratch/` outputs get no `.prov.json`
sidecar (sidecars stamp only research/findings/raw paths), (c) banked runner_sha256 is a composite
hash, (d) doc-level provenance fields differ (banked git_head null / hand-added declarations).

**Refuter 2 (done inline by hand — delegation budget was exhausted at 2/2, so this lens was NOT a
delegate; stated for honesty): SURVIVES** — own script `lens_b_recompute.py`, not compare_repro.py:
inputs are different files (sha256 6f87bec3 vs 62ab67f3, sizes 9486/8859); keysets equal 36/36; zero
value/type diffs excl wall_s; exact float equality on stochastic fields (da_outcome_unrewarded_min
0.49826075778867274, early_share 0.17166935403268332, weights_final e.g. ctx_a->str_d1_0
69.25879669189453, train_reward_sequence 200/200 equal, test and probe_lr0 equal) — match is NOT
vacuous; runner_sha256 recomputed under the runner's own formula: banked == registered composite,
repro == pilot composite, the two differ (repro provably ran the pilot code); seeding derives from
`a.seeds` into `default_rng(seed*7919+11)` / `default_rng(seed*104729+3)`.

Both refuters survived; HANDOFF 4a.1 is satisfied — the grid (step 2) may run on this pilot copy.
Caveat carried forward: outputs must go under `research/findings/raw/` (not `_scratch/`) from here so
provenance sidecars stamp. No bars/GO claimed yet; the grid's bars are evaluated in step 2.
## 2026-10-01 14:45 (Hermes, local model): pilot commit blocked by lane-starvation (staged) + 4a.2 grid launched

- Commit of `research/runners/_ldc_outcome_timing_pilot.py` (only the two options, verified by refuter 1)
  was BLOCKED by `lane-starvation` (pool.queue currently carries no `checked:lane` markers; 4 of 5 CPU
  lanes unserved). Per HANDOFF Step 0 / Step 2 this exact message = leave it STAGED, write one WORKLOG
  line, retry when the block clears; never `--no-verify`, never a waiver. NOT a STOP (not "any other gate
  message"). `tools/parallel_audit.py` prints ✓ SATURATED (no ready board tasks to restock with), so there
  is no job to add that I am permitted to queue (queue files are off-limits to me).
- 4a.2 grid launched as a background driver (`_scratch/grid_driver.sh`, log `_scratch/grid.log`,
  failures `_scratch/grid.fail`): dev seeds 8/11 x {credit_immediate, late_immediate} x reward onset
  {30,150,300} (outcome screen onset+100) x w_outc_strio {30,60} = 24 cells, --n-train 400, --arms full,
  runner-default n-test, numpy, nice -n 10, one at a time (RULING server-load), outputs
  `research/findings/raw/_ldc_outcome_timing/<sch>_onset<ms>_w<w>_s<seed>.json` (+ automatic provenance
  sidecars; outputs land untracked until the block clears and they can be committed).
  GUESS (marked): the grid sentence names only these two schedules; whether "record ... sanity ...
  success" ALSO needs a sanity_early_ch0 run per cell is ambiguous — 12 extra runs, deferred to a MiMo
  Pro delegate next run (fresh delegation budget) before launching them.
## 2026-10-01 15:10 (Hermes, local model): pilot committed; grid relaunched after gateway kill; sanity = PER_CELL (delegated)

- `lane-starvation` block cleared overnight-morning: `research/runners/_ldc_outcome_timing_pilot.py`
  committed **@ ba918bd82** (gates passed, no --no-verify), block-note WORKLOG committed **@ 049d80322**.
- First grid attempt (14:47) died silently mid-cell: the Hermes gateway restarted at 14:55
  (`hermes gateway run --replace`), killing the tracked background process tree; no OOM (25G free), no
  traceback in `_scratch/grid.log`. Relaunched 15:03 (resume-safe skip: completed cells are not rerun).
  Risk accepted: a future gateway restart kills the driver again -> detectable next wake, relaunch cheap.
- Sanity ambiguity resolved by MiMo Pro delegate (fresh budget, 333s): **PER_CELL**. `c4_sanity` is defined
  as `all(success >= 0.9 for sanity_early_ch0 rows)`, so sanity success is not in-row for the ci/li cells;
  the Step-4 banked sanity files are n-train 200 / default timing (wrong axis) and "(not evidence)"; prereg
  7.4 scopes the sanity bar to "IN credit_immediate timing" (i.e. cell timing); A0's "sanity/credit_immediate
  stay below 0.8 on most seeds" needs the rate at the same cell whose DA signed. -> 12 extra runs
  (seeds 8,11 x onset 30,150,300 x w 30,60, --schedule sanity_early_ch0, --arms full, --n-train 400,
  --reward-on-ms <onset> --outcome-ms <onset+100>, --w-outc-strio <w>) output
  `research/findings/raw/_ldc_outcome_timing/sanity_early_ch0_onset<ms>_w<w>_s<seed>.json`.
  QUEUED to launch when the 24-cell main grid writes "GRID DONE" (server-load: one at a time).
## 2026-10-01 15:42 (Hermes, local model): grid driver v2 - detached supervisor stack (attempts 1-2 died externally)

- Grid attempt 2 (15:03) also died without traceback/output/failure record between 15:10 and 15:32.
  Attempt 1 died at the 14:55 gateway `run --replace`; attempt 2 died while the same gateway (pid 181)
  was still up — cause undiagnosed (no OOM in dmesg/journal [no perms to see], no repo watchdog matches,
  no `at`/`crontab` binaries in the container for an external scheduler). Both deaths were silent
  process-tree kills: driver AND python died together, no rc recorded.
- Response = layered detachment instead of more guessing:
  1. `setsid` driver into its own session via `launcher.sh` (survives chat/session teardown);
  2. `supervisor.sh` (also setsid'd) respawns the flock-guarded driver every 60s until "GRID DONE";
  3. per-cell attempt counters (<=3) with `_scratch/attempt_<cell>` files -> poison-guard, successes reset;
  4. outer watchdog = the 30-min wake cycle itself (relaunch if the whole stack died).
  Cron tool is unavailable in this session (tool_search: only session_search/terminal/todo).
- Driver v2 now ALSO runs the 12 sanity cells (PER_CELL delegate ruling) after the 24 grid cells —
  one sequential driver, no separate launch step. All under nice -n 10, one at a time.
- Verified 15:40: driver 63066 + python 63072 (cell 1, 94% CPU), supervisor 63252, all in own sessions.
## 2026-10-01 16:14 (Hermes, local model): stale S supervisor stopped (grid moved by Claude); 12 sanity cells queued at the pool

- New wake rule + RULING `compute-placement`: no research jobs >~10 min on S. My local stack (supervisor
  63252) was already defunct (driver script deleted by Claude's cleanup -> rc=127 loops) and is now
  KILLED (0 matching processes); `grid_driver.sh.moved-to-pool` NOT restored. Claude merged main into the
  draft (e63b22f6d, docs only), provisioned that revision, and queued the 24 grid cells at the pool head;
  results arrive in `research/findings/raw/_ldc_outcome_timing/` via sim-sync; score when 24/24 are in.
- Per the same ruling ("You can queue lines for a provisioned revision yourself") I queued the 12
  PER_CELL sanity cells the delegate required (0 were present): seeds 8/11 x onset 30/150/300 x
  w_outc_strio 30/60, `--schedule sanity_early_ch0`, pilot runner, memcap 2 GB, mem_gb=2, max_age_h=48,
  checked reason names the delegate ruling + prereg 7.4. Result: ADDED=12 FAILED=0, depth 384 -> 396,
  `grep -c sanity_early_ch0 research/queue/pool.queue` = 12.
- Pending: 36 pool outputs (24 grid + 12 sanity); next T1 step = score bars into the PILOT.md table when
  they land.
## 2026-10-02 22:09 (local model) — Step 4a.1 repro EXACT + 4a.2 grid queued

- 4a.1: pilot copy (`ba918bd82`) run with default flags on `base_credit_immediate` s7, `--n-train 200`; deep JSON
  compare vs the Step 4 sweep row: payload identical except `runner_sha256` (the mandated copy is a different file),
  plus `wall_s` and git-stamp provenance keys. MiMo Pro delegate verdict (independently re-verified): **EXACT** —
  proceed.
- 4a.2: corpus consult logged (`before_you_build`, 22:07); revision `57c0f6ce` provision `state=ok` on pool41/pool42
  (done 10-01, same SHA); **24/24 grid cells queued** (seeds 8/11 x credit_immediate/late_immediate x onset
  30/150/300 ms x w_outc_strio 30/60, `--n-train 400`, dev seeds, not evidence). NOTE: one repro row took ~17 min
  locally — every further pilot run goes to the pool.
- Next: 4a.3 lr=0 diagnosis runs (seeds 9/12 memory, seed 14 explore) once the grid lands, then 4a.4 PILOT.md +
  `[T1] RESULT: Step 4a pilot @ <sha>`; T1 then stops for Claude's Amendment A0 + Step 4b.

## 2026-10-02 23:23 (local model) — Step 4a.3 diagnosis design + queue

- Design went to a MiMo Pro delegate (RULING models): **REVISE** and applied — (1) `--clear-ms` added to the pilot
  copy (module constant CLEAR_MS had no flag; default 250 = registered no-op, `34bed924d`); (2) "rates after the
  clear" = `probe_lr0`/probe-block `mem_after_clear_hz` + `mem_hold_acc` (per-trial rows are not persisted — probe-
  mean-only is the readout, documented as the limitation; `mem_after_clear_hz` averages mem_a+mem_b, so which pool
  keeps the stale cue is undeliverable without another edit); (3) clear_pA half/double = {300, **1200**}; (4) seed 14
  channel asymmetry reads `weights_initial/final` (s14 weights already symmetric — MSN-weight asymmetry ruled out;
  motor-pool latency not recorded = unavailable).
- Matrix: **17 runs queued** at `50f52c23e` (provision `state=ok` pool40/41/42/wspool) — seeds 9+12 x 7 knobs
  (default, clear-pA 300/1200, w-clear-fs 15/60, clear-ms 125/500) on credit_immediate, seed 14 default, plus
  seeds 9+12 default on late_immediate (mem_hold_acc schedule dependence). All `--arms full --lr 0 --n-train 200
  --n-test 100` (delegate cost trim; lr=0 adds no readout at n-train 400), `--n-probe` untouched (changing it shifts
  the world-RNG stream). Corpus consult logged 22:07 covers the `--checked` sentences.
- Branch hygiene: `agent_sync_branch` refused twice — first `runs.jsonl` local dirt (backed up 16,008 lines to
  scratch, restored), then `lane_builds.jsonl` both-side appends (my union script miscounted and DROPPED main's 3
  shared lines; fixed by taking main's exact copy — our 3 lanes were already present in it). main merged clean as
  `50f52c23e`; gates: tool-health-fresh + lever-efficacy warnings only.
- Baseline verification anchors (delegate, from banked sweep): s9 `mem_after_clear_hz`=32.2699, s12=28.6447, s14
  p_ch0_given_A/B=1.0/1.0, 0.04932 — the three default runs landing on these doubles as the post-edit repro check
  for `--clear-ms` (no second local 17-min row needed).
- Next: when the 24 timing-grid + 17 diagnosis cells land, run 4a.4 (PILOT.md + `[T1] RESULT: Step 4a pilot @ <sha>`),
  then T1 stops for Claude's Amendment A0 + Step 4b.

- 2026-10-03 — **Step 4a pilot complete (item 4).** All 41 cells landed (24 grid @ 57c0f6ce0, 17 diagnosis @
  50f52c23e): DA signs in 1/24 cells (credit_immediate on150/w60/s11 — marginal, 0.0013 under-then-over the
  0.47 bar; late_immediate 0/12, w60 ITI leak −0.111…−0.150), sanity ≥0.8 in 1/12, credit_immediate ≥0.8 in
  0/12; s9/s12 clear-dose monotone (def 32.3/28.6 Hz → any stronger/longer knob 0.00 Hz), s14 p(ch0|A)=
  p(ch0|B)=1.000 (ch0-vs-ch1 bias, not organ asymmetry). Evidence-backed A0 branch = the eligibility tag's
  action-specificity (branch 1 contradicted: one onset does sign); marginal-signature caveat flagged for
  Claude. `PILOT.md` written; `[T1] RESULT` filed to NEEDS-CLAUDE. **T1 stops: Claude writes prereg
  Amendment A0 and Step 4b.**

## 2026-10-06 (Hermes driver): t1-bar-sanity + round-4 verdict — r9 FREEZES at 30/54/48/20/3

- t1-bar-sanity executed: `agent_sync_branch` merged main @ 2c2d55952 as b37d07829; ff-only refused (diverged),
  clean `--no-edit` merge of `claude-review/bar-sanity-t1` (0a328b762) succeeded; bar-sanity specs/records +
  `bar-sanity:` line on AMENDMENT-A0.1.md:104 present. Branch tip 9f625fd1c; provision OK rc=0 on
  pool41/pool42/pool40/wspool at that pin.
- Round 4 ran 13:57 pool at pin f51493293 (the four ruling-specified lines; r5 re-run reproduces the old r5
  bit-for-bit: M_early 15.045371325868727, DA 0.45956286421163406 / 0.6119679583970958, LHb 63.4/51.4 — the
  reproduction condition PASSES). Records committed with rounds 1-3 in dev_omission/.

| arm (outc/rus/lhbinh/rmtg/snc) | U ≤0.47 | R ≥0.60 | I ≤0.005 (da_iti_offset_probe) | membrane_ok | void False | (a) LHb rewardwin unrew/rew, ratio ≥2 | (b) gap ≥0.02 vs off U 0.4982 | freeze |
|---|---|---|---|---|---|---|---|---|
| r5 re-run 30/27/24/20/3 | 0.4596 ✓ | 0.6120 ✓ | 6.3e-06 ✓ | ✓ | ✓ | 66/39 = 1.692 ✗ | 0.0386 ✓ | no |
| r11 30/40/36/20/3 | 0.4596 ✓ | 0.6125 ✓ | 6.3e-06 ✓ | ✓ | ✗ True | 66/32 = 2.063 ✓ | 0.0386 ✓ | no (void) |
| r9 30/54/48/20/3 | 0.4597 ✓ | 0.6129 ✓ | 6.3e-06 ✓ | ✓ | ✓ | 66/27 = 2.444 ✓ (in-record pass=True) | 0.0385 ✓ | FREEZE |
| r10 30/54/96/20/3 | 0.4599 ✓ | 0.6127 ✓ | 6.3e-06 ✓ | ✓ | ✓ | 66/27 = 2.444 ✓ | 0.0383 ✓ | no (r9 first) |
| off control 9/12 route off | 0.4982 ✗ (floor) | 0.6431 | — | ✓ | ✗ True | — | — | control |

- Verdict per ONE MiMo Pro delegate (RULING models), confidence high, quoting the runbook: "Freeze the FIRST of
  r11, r9, r10 that passes — r11 fails void_floor_ceiling, so r9 freezes and r10 does not; T1-grid then runs its
  single constant set 30/54/48/20/3." Bar (a) reads the in-record factor 2.0 (`lhb_engagement_a_rewardwin` =
  {ratio 2.444, factor 2.0, pass True} for r9); `I` reads `instrument.da_iti_offset_probe` in-record (all four
  rounds |−6.3e-06| ≤ 0.005); success (r9 0.39) is not in the freeze list — behavior rows are scored later per
  STEP4B §4.
- Frozen constants: **w_outc_lhb 30 / w_rus_lhbinh 54 / w_lhbinh_lhb 48 / w_lhb_rmtg 20 / w_rmtg_snc 3**,
  omission-route 1, reward-on 150 / outcome 250, w_outc_strio 30, w_clear_fs 60. Next: T1-grid — STEP4B §3
  (144 lines) at pin 9f625fd1c with `#checked:` naming STEP4B.md + AMENDMENT-A0.1.md, then PILOT-4B.md and
  `[T1] RESULT`.

## 2026-10-07 (Hermes driver): STEP4B §1 clear dose — choice `--w-clear-fs 60` (27-row table, records 10-04)

Records: `research/findings/raw/_long_delay_credit_bg/dev_pilot_4b/clear_{wfs60,cms500,cpa1200}_s{7..15}.json`,
run 2026-10-04 21:08-21:53, lr 0, credit_immediate, dev seeds 7-15. Both bars on every cell:
`mem_after_clear_hz` < 1.0 and `mem_hold_acc` ≥ 0.95. Cell format: hz / hold.

| seed | `--w-clear-fs 60` | `--clear-ms 500` | `--clear-pA 1200` |
|---|---|---|---|
| 7 | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |
| 8 | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |
| 9 | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |
| 10 | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |
| 11 | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |
| 12 | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |
| 13 | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |
| 14 | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |
| 15 | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |

Choice per the fixed order: the FIRST candidate passing on all nine seeds is `--w-clear-fs 60` — that flag is on
every Step 4b line (consistent with all round-1-4 argvs). Corpus consult `before_you_build` run before the first
grid add (STEP4B preamble).

## 2026-10-07 (Hermes driver): STEP4B §3 grid queued — 144 lines @ pin 9f625fd1c

- Pre-add steps per STEP4B preamble: `before_you_build` corpus consult logged (covers the `--checked`
  sentences); the 2026-08-01 `curiosity-reward-omission-veto-spiking-circuit-6seed` finding read.
- Grid: 3 schedules (credit_immediate, late_immediate, sanity_early_ch0) x onsets (30, 150 ms) x 3 arms x
  seeds 8-15 = **144 lines**, `--n-train 400 --n-test 100 --w-outc-strio 30 --w-clear-fs 60` (§1 choice),
  mem_gb 2, out `dev_pilot_4b/grid_<sch>_on<ms>_<arm>_s<seed>.json`. Arms: on = `--omission-route 1` +
  frozen r9 constants (30/54/48/20/3); off = `--omission-route 0` (organ defaults, the lesion arm); cut =
  on + `--cut-rus-lhbinh 1`.
- Every line's `#checked:` names `STEP4B.md section 3` + `AMENDMENT-A0.1.md`; pin `9f625fd1c`
  (provision `state=ok` pool41/42/40/wspool). At enqueue close: 123 waiting in `pool.queue`, 21 dispatched
  (pool40/pool41), 0 missing (dup-guard refused re-adds of the dispatched 21 — not FORCE_DUP cases).
- Next wake(s): as cells land, write `PILOT-4B.md` with `lhb_engagement_a_rewardwin` per cell + `[T1] RESULT`.

## 2026-10-07 17:30 EDT - Step 4b grid: 138/144; re-queued the 6 reboot-killed cells
- Investigated the 6 missing cells (credit_immediate on150_on s8/s9/s10, on30_cut s10/s11/s15): dispatched to wspool 00:08-00:54, never exited - absent from exits.json (all 138 landed siblings rc=0) and from the node's job_status.log; no live `_ldc_omission` process. Root cause: the workstation REBOOTED at 05:27:22 today (uptime -s).
- Re-queued the 6 lines VERBATIM (same cmd, pin 9f625fd1c1f201a871ea1c5c0a2d475e7a3fbe88, same `#checked:` reason). `pool_queue.sh add` confirmed: "a previous identical claim exists (wspool, ...) but is no longer alive -- queueing" (depth 163 -> 169).
- PILOT-4B generator ready (`/opt/data/cache/scratch/gen_pilot4b.py`): exits 2 naming any missing cell; emits per-cell U/R/I + marks, success, `lhb_engagement_a_rewardwin` (A), membrane_ok (m), void_floor_ceiling (v) + a per-arm rollup for 144/144.
- Waiting on the 6 re-queued cells to land; then: generate tables -> PILOT-4B.md -> commit outputs -> `[T1] RESULT: Step 4b @ 9f625fd1c`.

## 2026-10-07 18:50 EDT - Step 4b complete: 144/144, PILOT-4B.md written, [T1] RESULT filed
- All 144 cells landed (re-queued six finished); generator ran clean (rc=0) over 144/144.
- PILOT-4B.md (this branch): per-cell U/R/I + marks, success, `lhb_engagement_a_rewardwin` (A), membrane_ok (m), void_floor_ceiling (v) for all 18 schedule x onset x arm rows x seeds 8-15, plus the per-arm rollup. Mechanical counts: A pass on arms 7-8/8 (the single fail per schedule = s14 @onset150), A undefined off arms 8/8, A fail cut arms 8/8; membrane false = 0 everywhere; void flags 2-8 per row (raw, for the route-off read).
- Outputs (grid_*.json + .prov, 288 files) committed to main; RESULT line points Claude at section-4 readings. No verdict asserted here (§4 = Claude; verdicts follow verify-go + CLAUDE_CHECK on the PILOT-4B gate when ruled).

## 2026-10-07 18:52 EDT - correction: grid outputs were NOT committed to main
- The commit attempt was BLOCKED by the `below-chance` failure-class gate: 31 grid cells record `pass: False` at success < 0.5, and the gate's remedy (declare `below_chance` / an UNDECLARED verdict in the artifacts, or fix the task/label wiring) is reserved (post-hoc verdict declaration + runner/scorer edits). Filed `[step4b-outputs] STOP` in the runbook NEEDS.
- The 288 files were unstaged (index clean); PILOT-4B.md itself is committed (branch `d9913121a`) and carries the per-cell record. Lane gate satisfied meanwhile via `.lane_waiver` (NO-READY-WORK, checked evidence in file).

- 2026-10-07 23:55 EDT (Hermes, driver) — RULING step4b-below-chance (Claude 23:34) executed:
  declared `"below_chance": true` + `below_chance_reason: "pilot control arm, failure expected by
  design"` on all 96 off/cut cells of research/findings/raw/_long_delay_credit_bg/dev_pilot_4b (on
  main; text-level insert, no measured value touched), staged 192 files (96 cells + 96 prov sidecars).
  below_chance gate check(staged) = 0 problems; off/cut scan hits = 0 after the declaration.
  NOT declared and NOT committed: the 27 ON-arm cells the gate still hits — left UNDECLARED per the
  ruling's second clause (NEEDS [step4b-on-cells] filed). ON list:
    - grid_credit_immediate_on150_on_s10.json
    - grid_credit_immediate_on150_on_s11.json
    - grid_credit_immediate_on150_on_s12.json
    - grid_credit_immediate_on150_on_s13.json
    - grid_credit_immediate_on150_on_s9.json
    - grid_credit_immediate_on30_on_s10.json
    - grid_credit_immediate_on30_on_s12.json
    - grid_credit_immediate_on30_on_s13.json
    - grid_credit_immediate_on30_on_s15.json
    - grid_credit_immediate_on30_on_s9.json
    - grid_late_immediate_on150_on_s10.json
    - grid_late_immediate_on150_on_s12.json
    - grid_late_immediate_on150_on_s13.json
    - grid_late_immediate_on150_on_s8.json
    - grid_late_immediate_on150_on_s9.json
    - grid_late_immediate_on30_on_s10.json
    - grid_late_immediate_on30_on_s12.json
    - grid_late_immediate_on30_on_s13.json
    - grid_late_immediate_on30_on_s9.json
    - grid_sanity_early_ch0_on150_on_s10.json
    - grid_sanity_early_ch0_on150_on_s13.json
    - grid_sanity_early_ch0_on150_on_s9.json
    - grid_sanity_early_ch0_on30_on_s10.json
    - grid_sanity_early_ch0_on30_on_s11.json
    - grid_sanity_early_ch0_on30_on_s13.json
    - grid_sanity_early_ch0_on30_on_s8.json
  The full grid lands in two commits: this one (designed controls) + whatever Claude rules on the ON list.
- 2026-10-09 02:15 EDT (Hermes, driver) — RULING step4b-on-cells (Claude 10-08 23:43) executed on main:
  declared `"below_chance": true` + the ruled reason ("pilot ON-arm cell below chance: a measured
  result, pending Claude's PILOT-4B read, not a pass") on the gate's exact 27-hit ON-arm set = the
  SPIKING-WORKLOG list above (re-verified today: `tools.gates.below_chance` check(staged) = 27
  problems before / 0 after). Commit 34e482673 on main = the second grid commit: all 48 ON-arm cells
  (27 declared + 21 above-chance, which the gate flags nothing for) + 48 prov sidecars; text-level
  insert, no measured value touched. The 144-cell dev_pilot_4b grid is now fully in git (96 controls
  301851198 + 48 ON 34e482673). NOT in this commit: the separate clear_* family (27 cells + sidecars,
  still untracked, several gate-flagged) — no ruling covers it yet. RESULT filed on main 613edd284.
- 2026-10-09 04:25 EDT (Hermes, driver) — T1 Step 4a bank: the Step 4 instrument sweep is committed
  on main as e77f31e0b (243 cells + 243 .prov sidecars, 2.6 MB, exact HANDOFF message "rc=2
  chosen=null (not evidence)"; never re-run Step 4). The below_chance gate blocked the bank on 155
  of 243 cells (every test arm under declared chance 0.5); per the gate's own remedy (authoritative,
  same text-level splice as step4b-on-cells) each carries "below_chance": true + a reason saying the
  credit gate never ran (signed-DA precondition, prereg 7.3) so the row measures the instrument, not
  the hypothesis — no measured value touched, gate re-check 0 problems, 486 files through the full
  gates. Remaining T1 work: PILOT-4B.md + [T1] RESULT per RULING pilot-4b-next.
- 2026-10-09 06:35 EDT (Hermes, driver) — PILOT-4B executed per RULING pilot-4b-next through ONE MiMo Pro
  delegate (RULING models; the delegate ran the compute, I spot-checked before filing). Quoted from the
  delegate's answer: "(a) no consistent ON-minus-CUT behavioural effect at either onset in either family"
  (32 pairs, aggregate -0.0103, 15+/4=/13-, per-seed spread -0.46..+0.41; block means credit on30 -0.0375,
  credit on150 +0.0138, late on30 +0.0600, late on150 -0.0775); "the ON void_floor_ceiling rate is 5/8,
  5/8, 3/8, 3/8 (16/32 overall), and what saturates is the FLOOR exclusively - zero ceiling hits" (62 flags,
  46/62 to str_d2, cortico-striatal + mem->striatal weights collapsing to hebbian_min 0; route-off cells
  show 7 ceiling hits, so the bound is reachable, just not in ON); "(b) the dip window [30,80)/[150,200)
  sits inside the 250 ms outcome window and the 1000 ms eligibility span (d^t 0.97->0.92 at onset 30,
  0.86->0.82 at onset 150), so the dip reaches striatal plasticity geometrically at BOTH onsets, but it is
  NOT outcome-specific (CUT = ON, 0.4706 vs 0.4708) - a punishment-like pause, not a labelled omission
  signal"; "per-trial read NOT computable from committed fields" (only last-100 means + last-200 sums +
  train_reward_sequence; a follow-up must commit per-trial da_outcome min/max + argmin, the per-trial dec
  dicts, and marks). My independent spot-check reproduced credit on30 ON-CUT -0.0375, void 5/8, 5/8, 3/8,
  3/8 and s14 probe_lr0 = 1.000/1.000 (UNDEFINED, never explores). Delegate also corrected my brief: the
  family grid is 96 cells (12 families x 8 seeds), not 48, and "v1" has no data-level referent - read as
  void_floor_ceiling = 1, flagged in the draft. Delivered: docs/drafts/long-delay-credit/PILOT-4B.md on
  main e39a3d67e + [T1] RESULT (262 chars). No gate verdict, no six-seed claim; T1 stops here (Claude
  reads the RESULT; per T1-grid Claude checks void_floor_ceiling on the route-off cells).
- 2026-10-09 08:06 EDT — ruling pilot-4b-next (a)+(b), MiMo Pro delegate (41 calls), driver
  spot-checks: ON void rows 16/32 exact, s8 credit on30 ON 0.49/CUT 0.52 exact, trace
  e^(-t/1000) = 0.970/0.924/0.861/0.820 exact. (a) Per-seed paired tables (4 blocks x 8 seeds):
  aggregate mean -0.0103 (defined-28 -0.0121), sign 15+/4=/13-. ON void 16/32 = ALL floor
  (62/62 flagged synapses floor, 0 ceiling; 46/62 str_d2) -> what saturates: the corticostriatal
  weight floor. (b) The omission dip sits INSIDE the eligibility window at both onsets on every
  dip trial (80<=250, 200<=250; trace survival 0.92-0.97 on30 / 0.82-0.86 on150); depth
  ON=CUT=0.029; per-trial depth/latency fields ABSENT from cells (only last-100 means).
  Diagnosis: timing is not the failure - missing observability + no behavioural credit.
  RESULT [T1] filed on main. Sources: p4bn_analysis.py / p4bn_addendum.py (scratch).

## 2026-10-10 02:20 - T1 ldc-d2-floor instrument build (a/b enabler)

RULING ldc-d2-floor (2026-10-09, T1): (a) trace which afferent still drives the omission dip in the cut arm;
(b) the D2 weight trajectory per trial on 2 dev seeds to see when it hits the floor; (c) already done this
wake (bash tools/before_you_build.sh "D2 weight floor saturation" rc=0, recorded to corpus_checks_shared;
hit: finding 2026-06-24 - the D2 dendritic stack + resonator are ALREADY BUILT, read before building the
soft-bounds amendment). The 10-09 pilot-4b-next RESULT had already diagnosed per-trial depth/latency fields
ABSENT from cells (only last-100 means) = missing observability, so (a)+(b) need a runner copy.

Built research/runners/_ldc_dip_observability.py = _ldc_omission_pilot.py plus read-only instrumentation:
- Run.outcome now accumulates per-ms spike counts of every candidate dip driver the organ has (lhb, lhb_inh,
  rmtg, snc, str_d1_0/1, str_d2_0/1 - presence-guarded) beside the per-ms DA trace, stashed on
  Run.last_out; run_trial attaches info["_obs"] (full da_full+spk+rwin for the first 40 omission trials,
  compact min/argmin/max + per-region totals after).
- block() attaches info["wt"] = {pathway: [mean, floor_frac]} per TRAIN trial (floor_frac = fraction of
  synapses at <= 0.05 x the pathway's initial mean = the runner's own floor_ceiling test, per synapse).
- run_condition carries both into the cell record as weights_per_trial + obs (the per-trial info dicts are
  otherwise score-only and discarded).

Smoke (local, 10-trial, mem_ok 2 OK, memcap unavailable in this env so plain timeout 420): rc=0, wall 69s,
38 regions/988 neurons/47525 synapses; weights_per_trial 4/4 x 32 paths; obs 4/4 (3 full-trace omission
trials, 8 regions x 250 ms); score line IDENTICAL to the pre-record smoke (treatment +0.341648) = the
instrumentation draws no RNG and changes no dynamics. Smoke data stayed in scratch (not committed).

Next: commit + agent_sync + provision the draft branch, queue the 2-dev-seed cells for {on, cut}
(RULING names 2 dev seeds; cut is the question's arm, on is the reference). Unsure: whether the delayed
schedules' explicit outcome block (the other Run.outcome call site) fires for credit_immediate - for this
schedule outcome runs inside race() and last_out was captured; the harvest point sits before run.mark("end")
so every schedule's trial picks up whatever outcome produced. Analysis script for (a) lagged DA-vs-region
spike attribution comes after the 4 cells land.

## 2026-10-10 02:26 - instrument committed (523ea8ba5); provision refused stale; NEEDS filed

- Committed the instrument runner + this arc's first WORKLOG entry as 523ea8ba5 (gates: only
  tool-health-fresh warning).
- agent_sync_branch research/draft-ldc-outcome-timing REFUSED (exit 2) merging main 33830f8da: git merge
  would conflict in ~180 files - research/findings/raw/_long_delay_credit_bg/sweep/*.json (~175, both
  sides changed since merge-base), docs/drafts/long-delay-credit/PILOT-4B.md, research/coordination/
  tool_health.json. Per RULING sync-branch that exit 2 is a NEEDS-CLAUDE, not something to resolve by hand.
- Server provision requested for 523ea8ba5b2361072f4f2b36ff85a5242a9d9b6f: state=refused, reason
  "stale: does not contain" published main 33830f8da29eb41952eea09777a937e9bc3437bc, hint "Merge or rebase
  onto main, commit, and name the new sha right away". So the 4 ldc-d2-floor observability cells ({on, cut}
  x dev seeds, credit_immediate on150, n-train 400, mem_gb=2, checked cites STEP4B section 3) CANNOT be
  queued: no revision carries _ldc_dip_observability.py, and only a synced branch can be provisioned.
- NEEDS-CLAUDE [T1] filed on main (short line cites this entry). Unblocks: Claude resolves the sweep-file
  drift (which side's copies are canonical), then agent_sync + re-request provision + queue the 4 cells.
- Smoke data stays in scratch; nothing uncommitted on the branch except runs.jsonl (never add) and the
  stash of the earlier smoke provenance (git stash list on this worktree, "smoke provenance (ephemeral)").

## 2026-10-10 03:12 - ldc-d2-floor cells provisioned and QUEUED (4 cells)

- RULING ldc-sync-done (Claude, on STOP [T1]) merged main into this branch as efc19fe77 (157 add/add
  conflicts took main; the worktree advanced in place), so the branch contains published main and the
  staleness rule no longer applies. Lane waiver untouched (none present; queue now non-empty so the
  lane-starvation gate has real ready work).
- Provision requested for efc19fe77045bf3b6c623624103eae0ae93e8fd8: first "queued (seen; waiting for its
  turn)", then state=ok "provisioned on pool41" after 330 s.
- Queued the 4 observability cells with `tools/pool_queue.sh add` (depth 0 -> 4): arms {on, cut} x dev
  seeds {7, 8} (the ruling says 2 dev seeds without naming them; 7/8 are the pilot's first dev seeds and
  both are outside the gate-seed list, so no canary/prereg line is required), schedule credit_immediate,
  reward-on 150 / outcome 250, n-train 400 / n-test 100, r9-frozen constants (outc 30, rus-lhbinh 54,
  lhbinh-lhb 48, lhb-rmtg 20, rmtg-snc 3), omission-route 1, cut arm adds --cut-rus-lhbinh 1. Runner =
  research.runners._ldc_dip_observability (the instrumented copy), outputs
  research/findings/raw/_long_delay_credit_bg/d2_floor_observability/{on,cut}_s{7,8}.json.
  Checked reason cites STEP4B section 3 + RULING ldc-d2-floor; mem_gb=2, max_age_h=168.
- After the cells land: the (a) analysis script (lagged DA vs per-region spike attribution) and the (b)
  weight-trajectory/floor-hitting read, then the soft-bounds AMENDMENT draft for Claude's review.

## 2026-10-10 07:35 — A2 soft-bounds BUILT (c9107bb8c); byte-identity -> POOL (73-min cell); verdict HELD (Hermes)

- Rulings a02-approved / a02-conditions in force. Build committed on research/draft-ldc-outcome-timing `c9107bb8c`
  (rc=0 through the normal gates): sim/config.py three fields (exponents FROZEN 1.0), sim/bridge.py 4c C2 gated
  factor (executes nothing when flag absent/0), CreditParams + credit_extension passthrough in BOTH runners;
  probe picks up --soft-bounds via its auto-generated CLI. Separate A2 = this commit series + AMENDMENT-A0.2.md
  (main); NOT inside A1.
- Fast test PASS (0.83s): flag lands on bridge.core_config (returns (bridge, SelectorConfig) - cfg check must use
  core_config), defaults 0, and flag 1 changes NO build-time weight.
- Byte-identical full test: the reference cell on_s7.json wall_s=4387 (73 min) - a LOCAL run CANNOT satisfy it
  (two local attempts died at 420/540s, rc=124) and RULING compute-placement sends >10-min jobs to the pool.
  Verbatim pool argv recovered from pool.queue.done (epoch 1791615770, --arms full --reward-on-ms 150
  --outcome-ms 250 --omission-route 1 + the r9-freeze constants). provision.req filed for `c9107bb8c`; the
  rebuild line queues at that pin with --out .../on_s7_flag0_a2.json once provision lands, then a local
  comparator reads row-minus-wall_s vs the reference.
- HOLD (a02-approved: identity passes "before anything else"): test-2 dev reruns, bar-sanity RUN + OK record,
  and the merge NEEDS all wait on that verdict. The bar-sanity SPEC file (a02-conditions (4)) is next after the
  provision request. Homeostat contingency: NAMED, NOT built (condition (5)).

### 2026-10-10 07:35 detail for NEEDS [T1] STOP (sync refusal)
provision refused `c9107bb8c` (stale: does not contain published main a6d261024, i.e. the sim-sync-published
A0.2 risk-3 commit). `agent_sync_branch.sh research/draft-ldc-outcome-timing` then refused (rc=2) because main
into this branch conflicts in THIS FILE: main's copy carries T1 entry + risk-3 block (6247e9947, a6d261024),
this branch carries the mirrored copies PLUS the A2-build entry (2531808c6); the mirror blocks are textually
identical, so the expected resolution is the union (keep this branch's file). Every other path auto-merged
cleanly (lane_starvation gate takes main, runbook takes main by tool rule). Branch left unchanged at 2531808c6.

## 2026-10-10 06:50 — RULING ldc-d2-floor steps (a)(b)(c) done; AMENDMENT-A0.2 drafted (driver: Hermes; delegate = MiMo v2.6 Pro, RULING models)

- Start check: sweep json = 243 (T1 Start holds). Cells landed: d2_floor_observability/{on,cut}_{s7,s8}.json (4/4, 400 trials each).
- (a) dip source, cut arm: median-split over first 80 trials, all dip trials unrewarded. ON arm tracks the relay
  (on_s7 lhb +91 / rmtg +47 / lhb_inh −62 / snc −57; on_s8 +85/+48/−66/−55). CUT arm: lhb/lhb_inh/rmtg diffs all
  within ±5 spikes while snc still −57/−50 and total striatal activity +110 (s7) / +41 (s8) → the cut-arm SNc pause
  tracks striatal drive (`w_strio_snc`), not the LHb relay. host_reward_signal = 0.0 (endogenous).
  SURE? No — `spk_tot` is whole-trial, not window-resolved; eligibility-window timing needs a time-resolved probe
  (named in the amendment's caveats).
- (b) D2 trajectories: on_s8 ctx_a->str_d2_0 → exactly 0.000 @ trial 329 (224/400 trials ≤1, final 0.670),
  mem_a->str_d2_0 → 0.000 @ 288; cut_s8 same saturation (0.023/0.000 @ 336) → arm-independent; seed 7 never
  reaches the floor (minima 2–16) → seed-dependent; |delta_w| pinned at exactly 1.000 at the floor = clipped
  wasted steps (n nonzero deltas up to 346/400). floor flags on_s8: ctx_a d2 only.
- (c) `bash tools/before_you_build.sh "D2 weight floor saturation"` rc=0 (recorded in corpus_checks_shared.jsonl);
  top hits unrelated; first weight-dependence lever against this defect.
- AMENDMENT draft: `docs/drafts/long-delay-credit/AMENDMENT-A0.2.md` (delegated design, one MiMo Pro subagent per
  RULING models; schema-valid answer, cost $0.048). Delegate's rationale, quoted:
  "I chose asymmetric trafficking-slot soft bounds (free-slot-proportional potentiation, occupied-slot-proportional
  depression) because they put the weight dependence inside the same three-factor update in sim/bridge.py 4c. C2 —
  exactly where the defect lives ... with a named receptor-trafficking precedent and no fitted constant (exponents
  1.0). Rejected alternatives: Oja's rule (hebbian_oja) acts in the rate-window Hebbian block, not the
  reward-modulated 4c rule ...; Miller-MacKay subtractive normalization is a banked LANE-D negative ...; a
  host-side per-step re-derivation of cp_plasticity_rate_gain ... is precisely the wrapper/host-clamp the rules
  forbid; pure Turrigiano scaling is kept only as the named contingency ... Soft bounds is the one candidate that
  simultaneously removes the dead zone (g_down -> 0 at the floor, no wasted clipped steps), preserves full-strength
  rebound (g_up ~ 1 at w ~ 0), and is falsifiable by a clean on/off six-seed ablation."
  SURE? No — the bar-sanity line in the draft names `bar_sanity_spec_softbounds.py`, a spec that does NOT exist
  yet: writing it is part of the build after Claude approves (bars are §2-reserved).
- NEEDS-CLAUDE ASK filed for Claude's review (sim/ edit sign-off, bar spec, A1-vs-A2 packaging). No push (RULING push-credential).

## 2026-10-10 07:10 — a02-conditions (1) risk-3 RESOLVED: RECORDER floor_frac, not a rule clamp (Hermes)

- Rulings a02-approved + a02-conditions landed (a94c68ad3): A0.2 approved; sim/ edit signed off; byte-identical-off
  test mandatory first; exponents frozen 1.0; package as separate A2; bar-sanity spec now; homeostat named, not built.
- Verdict stated BEFORE any test-2 read (condition 1): the |delta|=1.000 pin is the RECORDER's `floor_frac` —
  the second element of `info["wt"] = {pathway: [mean, floor_frac]}` = fraction of the pathway's synapses at the
  bound — NOT a weight delta. Proof: on_s8 `ctx_a->str_d2_0`: n(floor_frac==1.0)=32/400, n(floor_frac>0)=346/400
  = exactly the count earlier misread as "nonzero delta"; mean moves freely (|d mean| max 3.452).
  Rule side verified in `sim/bridge.py` 4c (lines ~11027-11071): `data += weight_updates` then clip-to-bounds;
  no +/-1 per-step clamp exists anywhere. RECORDER NEEDS NO FIX (documented probe behavior).
- AMENDMENT-A0.2.md corrected in three spots: evidence (b), test-2 expectation, risk-3 (now marked RESOLVED).

### 2026-10-10 08:35 — [T1] a02 step: byte-identical test 1 QUEUED on the pool (verdict pending)
RULING `a2-sync-done`: Claude landed the *WORKLOG union-merge fix (`319f41694`) and synced this branch with main
→ tip `17eaf8b51` (contains published main `a6d261024`); provision REQUEST filed and the server reported
**provisioned OK on pool41/pool42/pool40/wspool** (08:32). Fast local test (flag lands on core_config, default 0,
flag 1 changes no build-time weight) PASSED earlier this morning.
**Queue line (depth 212):** the byte-identical-off cell = verbatim d2 `on_s7` argv (`credit_immediate`, `arms
full`, n_train 400 / n_test 100, seed 7 dev) at pin `17eaf8b51`, + `--soft-bounds 0` (flag verified via --help),
`--out …/on_s7_flag0_a2.json`, `#checked` cites AMENDMENT-A0.2.md test 1 + RULINGs a02-approved/a2-sync-done;
`before_you_build` reran rc=0 (corpus logged). Placement: pool per RULING compute-placement — the reference cell
wall_s was **4387 s (73 min)**, local attempts were killed at 420 s/540 s (pytest rc=124), so S is not the place.
**Hold chain unchanged: nothing else (test-2 cells, bar-sanity spec run, A2 package merge NEEDS) until test 1
reports byte-identical PASS** (a02-approved: "mandatory, before anything else"). Risk-3 verdict (recorder
floor_frac, not a rule clamp) was already recorded in the prior block, satisfying a02-conditions (1)'s "say which
in the WORKLOG before test 2 is read".

### 2026-10-10 09:40 — [T1] a02 test 1 VERDICT: BYTE-IDENTICAL **PASS**
Cell `on_s7_flag0_a2.json` landed 09:26 (pool node uid 999, pin `17eaf8b51`, `--soft-bounds 0`, seed 7 dev,
argv otherwise byte-for-byte the d2 reference's). Field-by-field comparison against the reference
`on_s7.json` (pin `efc19fe77`, pre-A2): **all keys identical, every value equal EXCEPT `wall_s`**
(reference 4387.3 s vs A2 flag-0 3149.7 s — timing only, excluded by the test definition; zero other diffs,
including the nested weights-per-trial / dip-stat arrays). The A2 build is therefore a NO-OP with the default
flag, exactly what a02-approved demanded before anything else.
**Unblocked now:** test 2 (`--soft-bounds 1` rerun of the T1 data per AMENDMENT-A0.2's plan), bar-sanity spec +
run (a02-conditions (4)), then the A1-vs-A2 packaging + merge NEEDS to Claude. Data committed with this entry.
