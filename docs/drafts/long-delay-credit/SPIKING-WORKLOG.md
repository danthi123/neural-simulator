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
