---
type: finding
status: live
date: 2026-09-29
lane: B (plan step 4)
mechanism: PRE-REGISTRATION of long-delay temporal credit on the production basal-ganglia organ -- flag-gated (BRAIN_BG_CREDIT, default OFF) corticostriatal three-factor plasticity on the Gate-A v2 selector (cue/context -> striatal D1/D2 coactivity tags, spiking SNc dopamine, a striosomal critic), measured on a delayed-cue two-choice benchmark with an event-decomposition instrument
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. No gate seed has been run. No result is claimed here.
runner: research/runners/long_delay_credit_bg.py (to be built after this commit; its SHA is frozen by Amendment A)
artifacts:
  - docs/drafts/long-delay-credit/SPIKING-WORKLOG.md
---

# Long-delay temporal credit on the production basal-ganglia organ: PRE-REGISTRATION (2026-09-29)

Serves `docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md` section 4 (long-delay temporal credit) and
section 6 (which mechanisms are necessary, by controlled ablation). Branch `research/long-delay-credit-spiking`, cut
from main `f52c6d246` (the pinned pre-change SHA for the byte-identical-off test).

## Why this document exists now

A standalone 350-neuron circuit built earlier today was stopped by a mid-flight audit
(`.claude/worktrees/_audit_2026-09-29/long-delay-credit.md`): five designs were tried in about 50 minutes while the
headline arms were being watched, its trace-necessity cell was fixed by the wiring, and its memory positive control
learned the wrong mapping. The ledger of those discarded designs is in `docs/drafts/long-delay-credit/SPIKING-WORKLOG.md`.
None of that is evidence here. This preregistration is written BEFORE any build or tuning on the new substrate, and it
fixes which quantities may be tuned (section 7) so the prediction arms are never used to choose constants.

A separate, earlier rate-level draft by the local model (kept as the reference for expected signatures, not as
evidence toward this prereg's own verdict) is committed at `research/findings/raw/_draft_long_delay_credit_dev.json`
and `research/findings/raw/_draft_long_delay_credit_dev_sweep.json`; its trace-ablation dissociation is an algebraic
identity of that draft's own feature design (docs/drafts/long-delay-credit/SPIKING-WORKLOG.md), which is exactly why
the capability claim here needs the spiking substrate instead.

## 1. Question

Does the brain's own action selector credit a delayed reward to the synapses that produced an EARLIER choice, rather
than to whatever is active when the reward arrives (a held cue, the terminal state, a late decoy, an intervening
action), and which mechanisms are necessary for that? "A network that holds A in mind through the delay is not
thereby assigning a later reward to the earlier causal event" (plan section 4) is tested with an instrument that
records which event's eligibility received the credited weight change, not only with behaviour.

## 2. Substrate (one brain; host = world + body)

- The production selector: `bg_action_selection_production_organ.py` builds the Gate-A v2 topology
  (`_vocal_action_selector_gate.build_selector_bridge`, 2 channels: proposal -> str_d1/str_d2 -> GPi/GPe/STN ->
  thalamus -> commit -> motor; shared arousal enables a race, OU noise on the proposal pools breaks symmetry, the first
  motor pool to cross the commit threshold with a clean loser is the choice). Unchanged.
- The extension, added only through the builder's own `extra_regions` / `extra_pathways` / `core_config_updates`
  and only when `BRAIN_BG_CREDIT` is on (new, separate functions; no edit of the existing organ path; no `sim/` edit):
  cortical cue/context pools (`ctx_a`, `ctx_b`), distractor pools (`ctx_x`, `ctx_y`), a neutral decoy pool (`ctx_n`),
  the terminal-state pool (`ctx_t`, identical on every trial), an outcome screen (`ctx_o`), a working-memory pair
  (`mem_a`, `mem_b`, slow-NMDA recurrent hold + shared FS inhibition + a trial-end clear, the D3/Rubicon recipe), and
  the limbic core (`reward_us -> snc <- striosome` via GABA_B; the dopamine modulator is produced from SNc firing by the
  `from_region_firing_signed` rule).
- Plastic (the credit targets): every `ctx_* -> str_d1_c / str_d2_c` and `mem_* -> str_d1_c / str_d2_c`
  (corticostriatal), and `ctx_o -> striosome` (the critic). Memory reaches the MSNs through `nmda_slow` synapses
  (Mg2+-gated): it biases MSNs the arousal barrage has depolarized and cannot drive a race alone (the precedent is
  the selection-seam lane's need->value route). Eligibility = the bridge's coactivity tag decayed by
  `sim.kernels.fused_eligibility_trace_decay` with `reward_eligibility_tau_ms`; weight change =
  `lr * (DA - baseline) * e` (bridge section 4c), with the D1/D2 sign asymmetry (D2 synapses move opposite to DA).
  `cfg.current_reward_signal` stays 0 (asserted): there is no host reward scalar and no host RPE.
- Declared host shortcuts: the WORLD schedules stimuli and drives `reward_us`; the BODY reads the committed motor pool
  (the organ's existing first-crossing read); the SNc tonic set-point is measured from the network's own pacemaking
  before training. None of these sees the cue -> action mapping or computes a credit signal.

## 3. Trial schedule (world)

ITI with a trial-end event (clears working memory) -> CUE (`ctx_a` or `ctx_b`, 50/50) with shared arousal: the EARLY
race -> cue removed -> DELAY = base x {0.5, 1, 2, 4, 8}, base 250 ms (125 ms to 2 s; the early-commit-to-reward gap is
then about 1-4 s, i.e. 1-4 trace time constants at tau = 1 s, not inside one) with 1-2 distractor presentations, each
with arousal (INTERVENING races) -> final DECOY (`ctx_n`; the OTHER cue in the misleading arms) with arousal -> TERMINAL
state (`ctx_t`, identical every trial) with arousal: the TERMINAL race -> [schedule-dependent post-terminal events] ->
OUTCOME screen (`ctx_o`, every trial) with the reward (`reward_us`) 30-80 ms into it if earned.

Correct mapping (to be learned): cue A -> channel 0, cue B -> channel 1.

Schedules (which commit the world rewards):
- `credit` (early commit, delayed reward): reward iff the EARLY commit is correct; outcome after the terminal race.
- `credit_immediate`: reward iff the early commit is correct, outcome starts at the early commit (positive control).
- `late_delayed` (the plan's literal "terminal choice, then delayed reward"): reward iff the TERMINAL commit is
  correct; after the terminal commit one more distractor race, then the outcome 300-800 ms after the terminal commit.
- `late_immediate`: reward iff the terminal commit is correct, outcome at the terminal commit (the memory positive
  control: memory needed, credit short).
- `sanity_*` (instrument only): reward iff the relevant commit is channel 0, whatever the cue; must be learned.

## 4. Arms

| schedule | arms |
|---|---|
| credit | full, trace_ablated_integral, trace_ablated_peak, shuffled_yoked, misleading_final_cue, memory_lesioned, critic_lesioned, reward_to_da_lesioned |
| credit_immediate | full, trace_ablated_integral |
| late_delayed | full, trace_ablated_integral, misleading_final_cue, memory_lesioned, shuffled_yoked |
| late_immediate | full, trace_ablated_integral, memory_lesioned |
| dose (credit, FIXED delay per run) | delay x {0.5, 2, 8} x tau {300, 1000, 3000} ms at equal kernel integral |

- `trace_ablated_integral`: tau 1000 -> 20 ms with the tag gain x50 (equal kernel integral, the R2b lesson).
  `trace_ablated_peak`: tau 20 ms with gain 1 (equal peak). Both are reported; neither alone carries a horizon claim.
- `shuffled_yoked`: the reward sequence of the same seed's `full` run replayed trial by trial, independent of the
  commits (its own rng stream).
- Lesions (TERMS.md: verified to hold at measurement): `memory_lesioned` zeroes the mem recurrence;
  `critic_lesioned` zeroes striosome -> SNc; `reward_to_da_lesioned` zeroes reward_us -> SNc.

## 5. Measurements

- Test block: 200 trials, plasticity frozen (lr = 0), no reward (the outcome screen still shown). Training: 400 trials.
- SUCCESS = fraction of test trials whose RELEVANT commit is correct; a trial with no clean commit counts as not
  correct. chance = 0.5 for a responsive agent; declared in every row (`chance: 0.5`).
- RESPONSE RATE = fraction of test trials with a clean commit in the relevant race. An arm-seed cell with response
  rate < 0.80 is VOID (never a pass of any kind, including "abolished").
- EVENT DECOMPOSITION (the credit instrument). The eligibility array of the plastic synapses is snapshotted at every
  window boundary. Because every tag decays with the same factor d per step, the eligibility at outcome onset t0
  splits exactly into contributions c_k from each window k (previous-trial carry-over, cue/early race, each
  intervening race, decoy, terminal race, quiet periods): c_k = (e(end_k) - e(start_k) d^len_k) d^(t0 - end_k). With
  A = lr * sum_{t >= t0} (DA_t - b) d^(t - t0) over the outcome window, the credited change attributable to window k
  is A * c_k per synapse. MAPPING CREDIT per pathway family and window: M = mean over the correct-mapping synapses
  (a->0, b->1) minus mean over the wrong-mapping synapses (a->1, b->0), D1 counted +, D2 counted - (their sign
  asymmetry). Reported per trial and summed over the last 200 training trials:
  M_early (ctx cue synapses, cue window), M_decoy (ctx cue synapses, decoy window), M_mem (mem synapses, all
  windows), M_term (mem synapses in the terminal window, the held-memory-at-reward route), carry-over and the rest.
  EARLY SHARE = M_early / (|M_early| + |M_decoy| + |M_mem| + |M_carry|). Non-contingent drift is reported per
  pathway (ctx_t, ctx_x, ctx_y, ctx_n -> striatum mean weights): a plastic pathway at its floor or ceiling voids
  that cell.
- Instrument read-outs: memory hold accuracy at the end of the delay; DA at the outcome (rewarded max, unrewarded
  min); the organ's commit latency; the lr=0 coupling P(terminal commit == early commit) at initialisation.

## 6. Gates (frozen bars; each per seed; a gate PASSES at >= 5 of 6 defined seeds; "this gate FAILS if ...")

Construct-validity checks (expected by construction; reported, NOT evidence):
- C1 `credit/trace_ablated_integral` success <= 0.65: the tag is the only route from the early commit to the reward.
- C2 `credit/memory_lesioned` success >= 0.75: the memory has no route to the early race.

Evidence (each can fail on a realistic outcome):
- G1 credit capability: `credit/full` success >= 0.75. FAILS if a 1-4 s gap is beyond what a 1 s coactivity tag and
  signed SNc dopamine can bridge on this selector (the Gate-B record shows credit on this selector is hard even at
  short gaps).
- G2 credit lands on the earlier event: in `credit/full`, M_early > 0 and EARLY SHARE >= 0.5. FAILS if the reward is
  credited mainly through the held memory or carry-over.
- G3 horizon at equal integral (dose): success(delay x8, tau 300) <= success(delay x0.5, tau 300) - 0.15, AND
  success(delay x8, tau 3000) >= success(delay x8, tau 300) + 0.15. FAILS if a slower tag does not buy a longer
  reach at matched magnitude (the R2b result would then repeat) or if the gap does not matter.
- G4 the ablated learner is competent: `credit_immediate/trace_ablated_integral` success >= 0.75. If G4 fails, C1
  and the short-tau dose cells cannot be read as horizon effects (VOID as horizon).
- G5 memory positive control: `late_immediate/full` success >= 0.75 and its memory hold accuracy >= 0.90.
- G6 memory is necessary for a terminal choice: `late_immediate/memory_lesioned` success <= 0.65 and >= 0.15 below
  `late_immediate/full` on the same seed.
- G7 the plan's literal cell: `late_delayed/full` success >= 0.75. FAILS if a held cue drives the terminal choice
  but the delayed reward is not credited to it across the post-terminal race.
- G8 no leak: `credit/shuffled_yoked` and `late_delayed/shuffled_yoked` success <= 0.65.
- G9 dopamine gates the learning: `credit/reward_to_da_lesioned` success <= 0.65.

Verdict: GO iff G1, G2, G3, G4, G8 and G9 pass (the credit claim) AND G5, G6, G7 pass (the memory/credit
dissociation). Any measured FAIL on these -> NO-GO (a verdict on this method, with the next method named, per THE
LAW). A gate whose defined seeds cannot reach 5 -> UNDEFINED, never a pass. A seed is undefined for a schedule if its
`full` arm has response rate < 0.80, or (late schedules) memory hold < 0.90.

Mechanism necessity (plan section 6), from the six-seed success means per schedule: NECESSARY (the ablated arm fails
<= 0.65 on >= 5/6 seeds while full passes), CONTRIBUTING (mean drop >= 0.10), NOT NECESSARY (drop < 0.10), UNDEFINED.
Mechanisms: slow eligibility tag (both ablations), memory recurrence, critic -V, reward -> SNc.

Preregistered signatures (reported beside the verdict, not part of it):
- S1 `credit/misleading_final_cue`: success >= 0.75 and M_decoy small against M_early (|M_decoy| < |M_early|): the
  decoy is the cue-class event nearest the reward, and the credit still lands on the early event.
- S2 `late_delayed/misleading_final_cue`: success >= 0.75 (the held cue resists a 50 ms other-cue decoy).
- S3 `late_delayed/memory_lesioned` with the misleading decoy is not run; `late_immediate/memory_lesioned` success
  among responders is expected <= 0.5 when the decoy is the other cue (not gated).
- S4 credit misattribution: in `credit/full`, terminal-commit accuracy (unrewarded) above 0.5 means reward credited to
  the held memory's terminal choice; reported with M_mem, M_term.

## 7. What may be tuned before Amendment A, and on what

Dev seeds only (7-15). Parameter search runs headless on the pool (`tools/sweep_pool.sh`) once a revision is
committed. Constants may be chosen ONLY on these instrument criteria and positive controls, never on C1, C2, G1-G3,
G6-G9, S1-S4 arms:
1. at lr = 0: clean-commit rate >= 0.9 in every race type; each cue's early commit goes to each channel on >= 15% of
   40 trials (exploration exists); losing channel's motor spikes <= the organ's clean_loser_ratio;
2. memory hold >= 0.9 at 2 s delay, cleared by the trial-end event, not flipped by a 50 ms other-cue decoy (hold
   measured, the decoy arm's success never looked at);
3. DA at the outcome: rewarded max >= baseline + 0.10, unrewarded min (critic trained on 50% reward) <= baseline -
   0.03; tonic offset |mean DA - baseline| over 20 no-reward trials <= 0.005; <!--derived-->
4. `sanity_early_ch0` and `sanity_terminal_ch0` learned (success >= 0.9) in `credit_immediate` timing;
5. `credit_immediate/full` and `late_immediate/full` success >= 0.8 on >= 6 of 9 dev seeds;
6. no plastic pathway at its floor or ceiling after 400 training trials in those positive controls.
Amendment A then records: the constants, the runner SHA-256, the dev evidence for 1-6, and nothing else. Bars in
section 6 are never amended.

## 8. What a GO would and would not show

A GO shows that the production selector's corticostriatal synapses, with spiking dopamine and a 1 s coactivity tag,
credit a 1-4 s delayed reward to the early choice, that the credit is measured to land on the early event rather than
on the held memory or a late decoy, and that a slower tag buys reach at matched magnitude. It would not show
production use: the plasticity is default-off, the cue pools are benchmark inputs, and the load-bearing fraction is
not changed by it. The next rung (the selection-seam lane's need -> value mapping learned instead of wired) uses the
same `BRAIN_BG_CREDIT` plasticity on `intero_* -> str_d1_*`.
