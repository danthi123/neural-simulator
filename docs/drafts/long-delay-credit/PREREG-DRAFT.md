# DRAFT preregistration — long-delay temporal-credit benchmark (plan section 4)

**STATUS: DRAFT — not a verdict. Claude reviews/finalizes after Tuesday (2026-09-29).**
Dev seeds only (7, and 1-15 for the small dev sweep). No gate seeds. No `sim/` edit.
Serves `docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md` section 4
("Long-delay temporal credit").

## 0. Question (the plan's key control)

Separate **persistent cue memory** from **temporal credit assignment**: a network that holds
cue A in mind through the delay is not thereby assigning a later reward to the earlier causal
event. The benchmark asks whether the delayed reward is credited to the EARLIER cue (not to
whatever is active at reward time — the terminal state, the final decoy, the intervening
distractors), and whether cutting the eligibility trace breaks that credit while leaving the
cue held in mind.

## 1. Reused machinery (byte-for-byte, NO `sim/` edit)

- `sim.kernels.fused_eligibility_trace_decay(trace, decay_factor) = trace * decay_factor`
  — the eligibility-trace decay, reused UNMODIFIED for the critic trace.
- The TD-error formula `delta = r + gamma*V(s') - V(s)` (phasic-DA signal) — the same
  formula as the validated `sim/td_value_critic.py` (2026-05-18 PASS).
- `analytic_vstar` pattern — the exact discounted return for instrument validity (V1).

(The earlier DRAFT also listed an actor with action-eligibility and the
`scale_free_transfer` RPE metric; both are DROPPED — see the design note in section 3. The
read-out is now a FIXED buffer probe, and the credit read-out is the cue value
discrimination, not a scale-free RPE transfer.)

## 2. Task structure (discrete time, per trial, T = 44 steps)

- **Cue** A or B (50/50) at a JITTERED onset in {1,2,3,4} (unpredictable arrival, so the
  value jumps at the cue), active for CUE_STEPS=2 steps, then REMOVED.
- **Delay** from cue offset to the choice: `delay = BASE * m`, m drawn uniformly from
  {0.5, 1, 2, 4, 8}, BASE=4 -> delay in {2,4,8,16,32} steps. Variable so the network cannot
  learn a clock.
- **Distractors** at a random step during the delay (uninformative; the "intervening events").
- **Terminal state** at choice_time-1: a neutral prompt carrying NO cue information.
- **Final decoy** at choice_time-2: a final distractor. In `full` it is neutral; in
  `misleading_final_cue` it deliberately suggests the WRONG answer (shows the other cue).
- **Choice** at choice_time = onset + CUE_STEPS + delay: a FIXED read-out of the buffer
  (the memory probe). It is NOT a learned action and NOT a critic feature (see section 3).
- **Post-choice delay** CHOICE_TO_REWARD=3 steps (0 in the `immediate` arm), then **reward**:
  r = 1.0 iff the CUE is the target (cue A) — the reward is CUE-contingent, so the cue is the
  DIRECT predictor. In `shuffled_reward`, r is an independent coin flip (no cue structure).
- choice_time = onset + CUE_STEPS + delay, so in [5, 38] (onset in {1,2,3,4}, delay in
  {2,4,8,16,32}); reward_time = choice_time + CHOICE_TO_REWARD (full, so [8, 41]) or
  choice_time (immediate, so [5, 38]). All within T=44.

## 3. Mechanisms (two SEPARATE, so the ablation dissociates them)

- **Working-memory buffer (PERSISTENT CUE MEMORY):** `buf_c[t+1] = MEM_DECAY * buf_c[t] +
  cue_c_active[t]`, MEM_DECAY=0.98, FIXED (not learned, NOT the TD trace). Persistent
  activity holds the cue through the delay (0.98^32 ~= 0.53 at the longest delay). The
  "terminal choice" is a FIXED read-out of the buffer (`argmax(buf)` at choice_time) — a
  memory probe that is UNAFFECTED by the trace ablation.
- **Critic (TEMPORAL CREDIT):** V(s) = w . phi(s); **phi = {bias, cue_A_active,
  cue_B_active} ONLY** — the transient cue (active CUE_STEPS steps at onset) plus a bias.
  Update `w += ALPHA_C * delta * e_c`, `e_c = gamma*lambda*e_c + phi` (kernel reused).
  The reward is cue-contingent and the cue is the ONLY predictive feature, so the reward's
  TD error must be carried BACK across the delay by the eligibility trace to reach the cue.
  That is the temporal credit under test.
- **TRACE ABLATION** (`trace_ablated` arm): set gamma*lambda = 0 for the critic trace (the
  credit path). The working-memory buffer (MEM_DECAY) is UNCHANGED — it is a separate
  mechanism. This is the plan's key control.

**Why the feature set is ONLY {bias, cue}** (each excluded feature is a later anchor that
would steal the credit from the cue — the same failure mode, found by diagnostic, three
times over):
- **NO persistent buffer** as a critic feature: still active at reward time -> the reward
  would be credited to it directly, and the trace ablation would not dissociate memory from
  credit.
- **NO time-since-cue taps**: the plan's "no clock"; the taps sit between the cue and the
  reward and steal the credit from the cue (a diagnostic showed the taps carrying the
  weight, w_cue ~ 0).
- **NO terminal choice**: the choice IS the memory read-out (argmax of the buffer), so it
  equals the cue and is a PERFECT predictor of the cue-contingent reward (a diagnostic
  showed w_choice ~ 0.83, w_cue ~ 0).
- **NO terminal/decoy/distractor**: these sit a few steps BEFORE the reward, so the trace
  leaks credit to them (a diagnostic showed w_terminal ~ 0.30, w_decoy ~ 0.29 alongside
  w_cue ~ 0.50). The plan requires the credit to anchor to the CUE (the earlier causal
  event), not the intervening events. They remain in the WORLD (the timeline) but are not
  critic features, so the only anchor is the cue and the trace must span the full delay to
  reach it.

## 4. Arms

| arm | delay | terminal | final decoy | post-choice delay | reward | trace |
|---|---|---|---|---|---|---|
| `full` | variable 0.5-8x | uninformative | neutral | 3 | cue A -> 1 | ON |
| `trace_ablated` | variable | uninformative | neutral | 3 | cue A -> 1 | **OFF (gl=0)** |
| `immediate` | variable | uninformative | neutral | **0** | cue A -> 1 | ON |
| `misleading_final_cue` | variable | uninformative | **suggests wrong** | 3 | cue A -> 1 | ON |
| `shuffled_reward` | variable | uninformative | neutral | 3 | **coin flip** | ON |

## 5. Metrics (per arm, per seed)

- `buffer_accuracy`: fraction of trials buffer-read-out == original cue (PERSISTENT MEMORY
  read-out; UNAFFECTED by the trace ablation).
- `value_at_cue_A` / `value_at_cue_B`: the critic value at the cue onset, per cue (late
  window).
- `value_discrimination` = value_at_cue_A − value_at_cue_B (TEMPORAL CREDIT read-out; HIGH
  iff the delayed reward's credit reached the EARLIER cue — the rewarded cue is valued above
  the non-rewarded cue; ~0 in `trace_ablated`).
- `rpe_at_terminal`: mean |RPE| at the terminal state (should be ~0: uninformative).
- `rpe_at_decoy`: mean |RPE| at the final decoy (should be ~0 in `full`).
- `us_decay`: |RPE| at reward time, late / early. **NOTE: ~1.0 in ALL arms by design** — the
  cue is TRANSIENT (removed after CUE_STEPS=2), so the critic's value is a spike at the cue
  onset and is NOT sustained to the reward step; the reward is therefore never "predicted"
  at the reward time (V(reward) = bias only). This is a consequence of the transient cue,
  not a failure. The credit read-out is `value_discrimination`, not `us_decay`.
- `vrmse`: critic value RMSE vs the analytic discounted return (instrument validity, V1).

## 6. Pre-registered expected signatures (DRAFT bars — Claude finalizes)

Dev-seed reference (15-seed sweep, seeds 1-15, 2000 trials, `_draft_long_delay_credit_dev_sweep.json`):
full disc=+0.509±0.021, trace_ablated disc=−0.001±0.005, immediate disc=+0.590±0.025,
misleading disc=+0.509 (NO-OP, identical to full — see §8.5), shuffled disc=−0.011±0.033;
buffer_accuracy=1.000 in ALL (ceiling — see §8.6); rpe_at_terminal/rpe_at_decoy ~0.002 in all.
The full-vs-trace_ablated dissociation holds on 15/15 seeds (attribution 98-102% to the trace).

- `full`: buffer_accuracy >= 0.80; **value_discrimination >= 0.30** (credit reached the cue);
  rpe_at_terminal low (~0); rpe_at_decoy low (~0); vrmse finite and low (critic converged).
- `trace_ablated`: **Dissociation** — value_discrimination collapses (<= 0.10, ~0) while
  buffer_accuracy STAYS HIGH (>= 0.80). Memory intact, credit cut. This is the plan's key
  control.
- `immediate`: like `full` (shorter credit horizon -> at least as easy); value_discrimination
  high.
- `misleading_final_cue`: INTENDED — the network should HOLD the original cue: buffer_accuracy
  stays high and value_discrimination stays high (not fooled by the final decoy). A drop would show
  the final decoy hijacks the read-out (a real, reportable effect). **AS BUILT this arm is a NO-OP**
  (identical to `full` on all 15 dev seeds): the decoy is filtered out of both the critic and the
  buffer, so it has no effect. See §8.5 — Claude must decide whether to drop it, make the decoy an
  input (reintroducing the credit-stealing risk), or reframe the claim.
- `shuffled_reward`: no learnable structure -> value_discrimination ~0 (nothing anchors to
  the cue), vrmse higher. buffer_accuracy reflects the buffer (still holds the cue, since the
  buffer is cue-driven, not reward-driven).

## 7. Anti-cheat / validity

- V1 (vrmse) must be finite and low in `full` before any science reading (instrument sound).
- The trace ablation must change the credit metric (value_discrimination) — if it does not,
  the trace is not load-bearing and the run is VOID (not a science result).
- `shuffled_reward` must not learn cue structure (value_discrimination ~0) — if it does, the
  instrument leaks.
- No autograd anywhere; the eligibility kernel is reused unmodified (asserted in the test).

## 8. UNSURE items (flagged for Claude, NOT decided here)

1. **Spiking vs rate:** the runbook says "small spiking network"; the only existing spiking
   actor-critic is cupy/GPU-bound (out of CPU-only scope). This DRAFT is rate-level, reusing
   the validated TD/eligibility/dopamine machinery. Whether a spiking realization is required
   changes what the experiment means -> Claude's call.
2. **Actor read-out vs learned:** this DRAFT uses a FIXED buffer read-out (a pure memory
   probe) rather than a credit-learned actor. Rationale: the plan's control is about CREDIT
   reaching the cue, and a learned actor would add a second credit path (the action
   eligibility) that muddies the dissociation. The buffer is the memory mechanism; the critic
   is the credit mechanism. Whether a learned choice is also required is Claude's call.
3. **Bars:** all bars in section 6 are DRAFT; Claude sets the frozen bars before any gate run.
4. **us_decay ~1.0:** the transient cue means the reward is never predicted at the reward step
   (value not sustained). Whether the benchmark should instead SUSTAIN the cue feature through
   the delay (so the reward is predicted and us_decay -> 0) is a design choice that changes
   what is measured -> Claude's call. The current design measures credit-to-the-cue (the plan's
   key control); a sustained-cue design would measure reward prediction.
5. **`misleading_final_cue` is a NO-OP as built.** The final decoy is filtered out of both the
   critic feature set and the buffer, so it has no effect and the arm is byte-identical to `full`
   on all 15 dev seeds. The feature-set restriction that PREVENTS credit-stealing is the same
   change that makes this control vacuous. Claude's call: (a) drop the arm, (b) make the decoy a
   critic/buffer input (reintroduces the credit-stealing risk the design removed — likely needs a
   re-run of the credit-stealing debugging), or (c) reframe the arm's claim.
6. **`buffer_accuracy` at ceiling (1.000, all arms/seeds).** The fixed buffer (MEM_DECAY=0.98)
   holds the cue perfectly, so buffer_accuracy does not discriminate between arms. It still
   confirms memory is intact (and that the trace ablation cut CREDIT, not MEMORY — it stays 1.000
   in `trace_ablated`), but the discriminating read-out is `value_discrimination`. Whether the
   buffer should be made harder (lower MEM_DECAY / longer delays) so buffer_accuracy is not
   saturated is Claude's call.
