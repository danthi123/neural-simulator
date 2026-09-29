# long-delay-credit WORKLOG (task 3, Weekend work)

Branch: research/draft-long-delay-credit · worktree .claude/worktrees/draft-long-delay-credit
Serves docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md section 4.
ALL results here are DEV (dev seeds only), NOT a verdict. Claude reviews after Tuesday.

## 2026-09-29 01:15  Worktree created + prior-work search (required before the first lever)

- `git worktree add .claude/worktrees/draft-long-delay-credit -b research/draft-long-delay-credit`
  (based on main f52c6d246).
- `bash tools/before_you_build.sh "long delay temporal credit assignment eligibility trace"`
  (main checkout, read-only). Top hits (POINTERs, read in full):
  1. `research/findings/2026-05-18-td-value-critic-temporal-credit-PASS.md` — the VALIDATED
     biologically-canonical TD(lambda) value-function critic (phasic-DA TD error), reuses
     `sim.kernels.fused_eligibility_trace_decay` UNMODIFIED, NO autograd. Schultz cue-shift
     transfer 0.997, us_decay 0.0015, vrmse 0.003 (3 seeds). THIS is the machinery to reuse.
  2. `research/findings/2026-07-11-R2b-dual-timescale-eligibility-GO-...md` — CORRECTED/REFUTED:
     a "slow eligibility horizon" lift was an effective-learning-rate artifact (magnitude-
     decoupled EMA control collapses it). LESSON: do NOT claim a slow-trace horizon effect
     without a magnitude-decoupled control. Not directly needed here (fixed gamma*lambda).
  3. `research/findings/2026-05-18-compose-temporal-credit-PASS.md` — temporal credit bridges a
     composition gap (context, not needed for this benchmark).
- Reusable machinery confirmed present:
  - `sim/td_value_critic.py` (102 L): TD(lambda) critic, `analytic_vstar`, `csc_features`,
    `scale_free_transfer`, `run_pavlovian` (modes td/no_bootstrap/permuted/wrongsign).
  - `sim/kernels.py:1630` `fused_eligibility_trace_decay(trace, decay_factor) = trace*decay_factor`
    (REUSE UNMODIFIED).
  - `sim/neuromodulators.py` (phasic-DA framing; the TD error IS the phasic-DA signal per the
    TD-critic finding).
  - `research/runners/td_critic_core.py` (frozen-bar THREE-STATE verdict pattern to mirror).
  - `research/runners/_advantage_actor_critic_probe.py` — the existing SPOKING actor-critic is
    cupy/GPU-bound (imports g11_bg_runner, cupy). OUT of my CPU-only scope (runbook: no GPU jobs).

## 2026-09-29 01:20  Design decision (flagged UNSURE for Claude)

- The runbook task 3 says "on a small spiking network" but also "REUSE existing
  eligibility-trace / dopamine machinery". The only existing spiking actor-critic
  (`_advantage_actor_critic_probe.py`) is cupy/GPU-bound and out of my CPU-only scope. The
  VALIDATED temporal-credit machinery (`sim/td_value_critic.py`) is rate-level (pure numpy, CPU).
- DECISION (mine, flagged UNSURE): implement the benchmark RATE-LEVEL, reusing the validated
  TD/eligibility/dopamine machinery byte-for-byte (CPU, dev seeds). This directly answers the
  plan's science question (persistent cue memory vs temporal credit assignment) and reuses the
  validated kernel. Whether a SPIKING realization is required is an UNSURE item for Claude —
  it is a design choice that changes what the experiment means, so it is flagged, not decided.
- The benchmark separates the two mechanisms the plan names:
  - PERSISTENT CUE MEMORY = a fixed slow-decay working-memory buffer (persistent activity) that
    holds the cue through the variable delay; the actor reads it out to make the choice.
  - TEMPORAL CREDIT ASSIGNMENT = the TD eligibility trace carries the delayed reward's TD error
    back to the earlier cue/choice event; measured by RPE transfer to the cue (not the terminal
    state / final decoy / reward time).
  - TRACE ABLATION cuts the eligibility trace (gamma*lambda -> 0) for the credit path ONLY; the
    working-memory buffer (a separate mechanism) is untouched. Prediction: credit (RPE transfer)
    collapses while the buffer still holds the cue -> the plan's key dissociation.

## 2026-09-29 (overnight)  Iteration: making the credit REACH the cue (credit-stealing debugging)

The plan's key control requires the delayed reward's credit to anchor to the EARLIER cue, not to
whatever is active at reward time. Debugging showed a recurring failure mode — **credit-stealing by
a temporally-closer or perfectly-predictive feature** — three times over, each fixed by removing the
stealing feature from the critic's feature set (it stays in the WORLD/timeline):

1. **Reward made CUE-contingent** (r = 1 iff cue is the target, cue A), not choice-contingent. A
   diagnostic showed that with a choice-contingent reward the value anchored to the (late) choice and
   the taps (w_cue ~ 0). Making the reward cue-contingent makes the cue the DIRECT predictor.
2. **Removed the time-since-cue taps.** The taps sit between the cue and the reward and steal the
   credit (a diagnostic showed the taps carrying the weight, w_cue ~ 0). Also the plan's "no clock".
3. **Removed the terminal choice as a critic feature.** The choice IS the memory read-out (argmax of
   the buffer), so it equals the cue and is a PERFECT predictor of the cue-contingent reward
   (diagnostic: w_choice ~ 0.83, w_cue ~ 0). The choice is still COMPUTED for the buffer_accuracy
   probe; it is just not a critic feature.
4. **Removed the terminal/decoy/distractor as critic features.** These sit a few steps BEFORE the
   reward, so the trace leaked credit to them (diagnostic: w_terminal ~ 0.30, w_decoy ~ 0.29
   alongside w_cue ~ 0.50). The plan requires the credit to anchor to the CUE, not the intervening
   events. They remain in the WORLD (the timeline) but are not critic features.

Result: the critic sees ONLY {bias, transient cue}. The reward's TD error must be carried BACK across
the delay by the eligibility trace to reach the cue.

**Dev-seed result (seed 7, 2000 trials):**

| arm | buffer_acc | val_A | val_B | disc | rpe_term | rpe_decoy |
|---|---|---|---|---|---|---|
| full | 1.0 | 0.547 | 0.042 | 0.505 | 0.002 | 0.002 |
| trace_ablated | 1.0 | 0.247 | 0.252 | -0.005 | 0.005 | 0.005 |
| immediate | 1.0 | 0.623 | 0.034 | 0.589 | 0.002 | 0.002 |
| misleading_final_cue | 1.0 | 0.547 | 0.042 | 0.505 | 0.002 | 0.002 |
| shuffled_reward | 1.0 | 0.284 | 0.326 | -0.042 | 0.002 | 0.002 |

The plan's key dissociation is now measurable: `value_discrimination` is HIGH in `full` (0.505, credit
reached the cue) and ~0 in `trace_ablated` (-0.005, credit cut), while `buffer_accuracy` is 1.0 in
BOTH (memory intact). The terminal/decoy RPEs dropped to ~0.002 (no more stealing). `shuffled_reward`
shows no cue structure (disc ~0).

**NOTE (us_decay ~1.0 in all arms):** the cue is TRANSIENT (removed after CUE_STEPS=2), so the critic's
value is a spike at the cue onset and is NOT sustained to the reward step; the reward is never
"predicted" at the reward time (V(reward) = bias only). This is a consequence of the transient cue, not
a failure. The credit read-out is `value_discrimination`, not `us_decay`. Whether the benchmark should
instead SUSTAIN the cue feature through the delay (so the reward is predicted and us_decay -> 0) is an
UNSURE item for Claude (PREREG section 8.4) — it changes what is measured.

Tests updated to the new metric names (buffer_accuracy, value_discrimination); the stale
choice_accuracy/rpe_transfer_to_cue/IDX_CHOICE_A/IDX_TERMINAL references are gone. `test_no_autograd`
now checks import statements only (the docstring mentions "autograd" in prose). 6/6 pass.

## 2026-09-29 04:45  15-seed dev sweep + two NON-BLOCKING warnings (for Claude)

Ran the planned small dev sweep (seeds 1-15, all 5 arms, 2000 trials, ~31 s CPU) to
`research/findings/raw/_draft_long_delay_credit_dev_sweep.json` (75 rows). The plan's key
dissociation is robust across every seed. Attribution (full vs trace_ablated, value_discrimination)
is 98-102% attributable to the eligibility-trace manipulation on all 15 seeds.

| arm | buf_acc | disc (mean±sd) | val_A | val_B | vrmse | rpe_term | rpe_decoy |
|---|---|---|---|---|---|---|---|
| full | 1.000 | +0.509 ± 0.021 | 0.555 | 0.047 | 0.186 | 0.002 | 0.002 |
| trace_ablated | 1.000 | −0.001 ± 0.005 | 0.257 | 0.258 | 0.186 | 0.005 | 0.005 |
| immediate | 1.000 | +0.590 ± 0.025 | 0.630 | 0.039 | 0.176 | 0.002 | 0.002 |
| misleading_final_cue | 1.000 | +0.509 ± 0.021 | 0.555 | 0.047 | 0.186 | 0.002 | 0.002 |
| shuffled_reward | 1.000 | −0.011 ± 0.033 | 0.298 | 0.310 | 0.204 | 0.002 | 0.002 |

- `full` disc range +0.473..+0.562; `trace_ablated` disc range −0.008..+0.010. The dissociation
  (credit reaches the cue in `full`, collapses in `trace_ablated`, memory intact in both) holds on
  15/15 seeds. `immediate` (post-choice delay 0) is EASIER (higher disc, +0.590) than `full`
  (+0.509) — shorter credit horizon, as expected. `shuffled_reward` learns no cue structure
  (disc ≈ 0) — the instrument does not leak.

**NON-BLOCKING WARNING 1 — `misleading_final_cue` is currently a NO-OP (identical to `full`).**
The final decoy (choice_time−2) is filtered out of BOTH the critic feature set (only {bias, cue})
and the buffer (driven only by the original cue's active window). So a "misleading" final decoy has
no effect: the arm's numbers are byte-identical to `full` on all 15 seeds. Consequence: this arm does
NOT currently test "does the network hold the original cue against a misleading final decoy" — that
control is vacuous as built. There is a real tension: the feature-set restriction that PREVENTS
credit-stealing (filter the decoy) is the same change that makes the misleading-decoy control vacuous.
Claude's call: (a) drop the arm, (b) make the decoy a critic/buffer input (reintroduces the
credit-stealing risk the design removed), or (c) reframe the arm's claim. Flagged in PREREG §8.5.

**NON-BLOCKING WARNING 2 — `buffer_accuracy` is at CEILING (1.000) in every arm and seed.**
The fixed working-memory buffer (MEM_DECAY=0.98) holds the cue perfectly through the delay, so
`buffer_accuracy` does not discriminate between arms. It is a valid sanity check (memory is intact,
and — crucially — it stays 1.000 in `trace_ablated`, confirming the ablation cut CREDIT not MEMORY),
but the discriminating read-out is `value_discrimination`. Flagged in PREREG §8.6.
