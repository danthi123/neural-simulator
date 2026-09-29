"""DRAFT long-delay temporal-credit benchmark (plan section 4). DEV, not a verdict.

A delayed-cue two-choice benchmark (plan section 4, verbatim structure):
  cue A or B at the start, removed; a VARIABLE delay (0.5x-8x base, no clock);
  distractors during the delay; an uninformative terminal state; a final decoy that
  can suggest the WRONG answer; the terminal choice (a memory read-out), then a
  DELAYED reward; a shuffled-reward control; eligibility traces that can be ablated.

The plan's KEY CONTROL separates PERSISTENT CUE MEMORY from TEMPORAL CREDIT
ASSIGNMENT: "a network that holds A in mind through the delay is not thereby
assigning a later reward to the earlier causal event." Two SEPARATE mechanisms:

  - PERSISTENT CUE MEMORY = a fixed slow-decay working-memory buffer (persistent
    activity) that holds the cue through the variable delay. Probed DIRECTLY
    (buffer_accuracy = argmax of the buffer at the choice step) -- a read-out that
    is UNAFFECTED by the trace ablation. The "terminal choice" IS this read-out.
  - TEMPORAL CREDIT ASSIGNMENT = the TD eligibility trace carrying the delayed
    reward's TD error back to the EARLIER cue, so the critic's value ANCHORS to the
    cue (value_jump_at_cue rises). The reward is CONTINGENT ON THE CUE (r = 1 iff
    the cue is the target, cue A), so the cue is the DIRECT predictor of the reward
    and the value must anchor to it -- not to the terminal state, the final decoy,
    or whatever is active at reward time.

The TRACE ABLATION (gamma*lambda -> 0) cuts the credit path (the critic trace) ONLY;
the working-memory buffer (a separate mechanism) stays. Predicted dissociation:
buffer_accuracy stays high (memory intact) while value_jump_at_cue collapses (credit
cut). That is the plan's key control.

WHY the reward is CUE-contingent (not choice-contingent): the plan's control credits
the reward to the EARLIER causal event (the cue). A diagnostic showed that when the
reward is contingent on the (late) CHOICE, the value anchors to the choice and the
taps (w_cue ~ 0, the value spikes at the choice step) -- the cue gets no credit.
Making the reward cue-contingent makes the cue the direct predictor, so the value
anchors to the cue and the trace ablation cleanly cuts it.

REUSE (NO sim/ edit, byte-for-byte):
  - sim.kernels.fused_eligibility_trace_decay (the eligibility-trace decay),
    UNMODIFIED, for the critic trace.
  - the TD-error formula delta = r + gamma*V(s') - V(s) (the phasic-DA signal),
    the same as the validated sim/td_value_critic.py (2026-05-18 PASS).
  - the analytic-vstar pattern (the exact discounted return, for instrument validity
    V1).
NO autograd anywhere. Deterministic seeded. CPU/numpy. Dev seeds only (7, 1-15).

Timeline (per trial, T steps; the cue onset is JITTERED so its arrival is
temporally unpredicted, exactly as the validated Pavlovian task requires for a
value that jumps at the cue):
  t=onset (in {1,2,3,4}): cue A or B appears, active CUE_STEPS=2 steps, removed.
  delay = BASE * m, m in {0.5,1,2,4,8} (BASE=4 -> {2,4,8,16,32}); variable, no clock.
  choice_time = onset + CUE_STEPS + delay: the FIXED buffer read-out (the memory
    probe / "terminal choice").
  decoy at choice_time-2 (neutral in `full`; suggests the WRONG cue in
  `misleading_final_cue`); terminal (uninformative) at choice_time-1.
  CHOICE_TO_REWARD=3 steps (0 in `immediate`), then reward r=1 iff the cue is the
  target (cue A) (`shuffled_reward`: r is an independent coin flip -> no cue
  structure, so nothing can anchor to the cue).

Arms: full | trace_ablated | immediate | misleading_final_cue | shuffled_reward.
Metrics: buffer_accuracy (memory probe), value_at_cue_A/B + value_discrimination
(temporal credit), vrmse (V1), us_decay, rpe_at_terminal, rpe_at_decoy.
See docs/drafts/long-delay-credit/PREREG-DRAFT.md.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

from sim.kernels import fused_eligibility_trace_decay  # REUSED UNMODIFIED
from tools.lab import attributable_to                  # the plan's point: whose is the difference?

# --- schedule constants (task structure, NOT science bars) ---
T = 44                 # steps per trial
CUE_STEPS = 2          # cue active steps
ONSETS = (1, 2, 3, 4)  # jittered cue onset (unpredictable arrival -> value jumps at cue)
BASE_DELAY = 4
DELAY_MULTS = (0.5, 1.0, 2.0, 4.0, 8.0)   # -> delay in {2,4,8,16,32}
CHOICE_TO_REWARD = 3
TARGET_CUE = 0         # cue A is the rewarded ("target") cue: r = 1 iff cue == TARGET_CUE
N_TRIALS = 2000
EARLY = 200            # early window for the us_decay baseline
LATE = 200             # late window for the asymptotic metrics

# --- learning constants ---
GAMMA = 0.98           # discount (higher than the validated 0.90-horizon task: the
                       # credit must span a 32-step delay, so the distant reward keeps
                       # value and the trace reaches back to the cue).
LAMBDA = 0.98          # eligibility-trace coefficient (gamma*lambda = 0.9604, credit
                       # horizon ~ log(0.01)/log(0.9604) ~= 108 steps >> the 40-step
                       # reward->cue gap). Same kernel (fused_eligibility_trace_decay),
                       # a slower decay so the temporal credit spans the long delay.
ALPHA_C = 0.02         # critic step (lower than the validated 0.05: the trial is
                       # longer (T=44) and the credit horizon is longer (decay 0.9604),
                       # so the eligibility trace accumulates over more steps and the
                       # effective update is larger -- a smaller step keeps it stable).
MEM_DECAY = 0.98       # working-memory buffer decay (FIXED, not the TD trace)

# --- feature layout (critic) ---
# The critic sees ONLY the TRANSIENT cue (active CUE_STEPS steps at onset) plus a
# bias. Every other feature is deliberately EXCLUDED because it would give the credit
# a LATER anchor to grab instead of spanning the delay back to the cue:
#   - NO persistent buffer: still active at reward time -> the reward would be
#     credited to it directly, and the trace ablation would not dissociate memory
#     from credit.
#   - NO time-since-cue taps: the plan's "no clock" (a variable delay means the
#     network must not hold a time-to-reward representation); the taps sit between
#     the cue and the reward and steal the credit from the cue.
#   - NO terminal choice: the choice IS the memory read-out (argmax of the buffer),
#     so it equals the cue and is a PERFECT predictor of the cue-contingent reward.
#     A diagnostic showed the credit anchors to it (w_choice ~ 0.83, w_cue ~ 0).
#   - NO terminal/decoy/distractor: these sit a few steps BEFORE the reward, so the
#     trace leaks credit to them (a diagnostic showed w_terminal ~ 0.30, w_decoy ~ 0.29
#     alongside w_cue ~ 0.50). The plan requires the credit to anchor to the CUE (the
#     earlier causal event), not the intervening events. They remain in the WORLD
#     (the timeline) but are not critic features, so the only anchor is the cue and
#     the trace must span the full delay to reach it.
# So the ONLY predictive feature is the transient cue, and the reward's TD error must
# be carried BACK across the delay by the eligibility trace to reach it -- that is
# the temporal credit under test. (The choice is still COMPUTED for the buffer
# accuracy probe; it is just not a critic feature.)
IDX_BIAS = 0
IDX_CUE_A = 1                  # transient cue (active CUE_STEPS steps at onset)
IDX_CUE_B = 2
N_FEAT = 3


def _analytic_vstar():
    """Exact expected discounted return along the cue-anchored timeline, averaged
    over the delay distribution. (Instrument validity, V1 -- mirrors
    sim.td_value_critic.analytic_vstar.) The critic sees the cue ONLY during the cue
    (steps 0..CUE_STEPS-1); after that the cue is gone and the critic can hold only the
    CUE-AVERAGE value (it no longer knows which cue was shown). So:
      vstar[cue][k] = (1 if cue==TARGET else 0) * mean_d gamma**(reward_tap - k)   k < CUE_STEPS
      vstar[cue][k] = 0.5 * mean_d gamma**(reward_tap - k)                          k >= CUE_STEPS
    (the 0.5 is the 50/50 cue average; cue B contributes 0, cue A contributes the
    discounted return). reward_tap = CUE_STEPS + delay + CHOICE_TO_REWARD (the reward
    step relative to the cue onset)."""
    v = np.zeros((2, T))
    for k in range(T):
        s_target = 0.0
        for m in DELAY_MULTS:
            d = int(round(BASE_DELAY * m))
            reward_tap = CUE_STEPS + d + CHOICE_TO_REWARD
            if reward_tap > k:
                s_target += GAMMA ** (reward_tap - k)
        s_target /= len(DELAY_MULTS)
        if k < CUE_STEPS:
            v[TARGET_CUE][k] = s_target          # cue-specific: only the target cue
            v[1 - TARGET_CUE][k] = 0.0           # the non-target cue has no reward
        else:
            v[0][k] = v[1][k] = 0.5 * s_target   # cue-average (the cue is gone)
    return v


def _build_trial(rng, arm):
    """One trial's timeline + ground truth. Host = the WORLD (it only schedules
    stimuli and the reward, which the plan assigns to the world), not a brain
    shortcut."""
    onset = int(rng.choice(ONSETS))
    delay = int(round(BASE_DELAY * float(rng.choice(DELAY_MULTS))))
    cue = int(rng.integers(0, 2))            # 0=A (target/rewarded), 1=B
    choice_time = onset + CUE_STEPS + delay
    terminal_time = choice_time - 1
    decoy_time = choice_time - 2
    c2r = 0 if arm == "immediate" else CHOICE_TO_REWARD
    reward_time = choice_time + c2r
    # a distractor at a random step strictly inside the delay (an "intervening event")
    lo, hi = onset + CUE_STEPS + 1, choice_time - 3
    distractor_time = int(rng.integers(lo, hi)) if hi > lo else -1
    return {
        "onset": onset, "delay": delay, "cue": cue,
        "choice_time": choice_time, "terminal_time": terminal_time,
        "decoy_time": decoy_time, "distractor_time": distractor_time,
        "reward_time": reward_time,
    }


def _phi(t, trial, choice=None):
    """Critic feature vector at step t: bias + the TRANSIENT cue (active only at onset).
    NO buffer, NO taps, NO choice, NO terminal/decoy/distractor -- see the feature-layout
    note: the credit must span the delay to reach the cue, which is gone by reward time.
    (`choice` is kept in the signature for caller compatibility; it is NOT a critic
    feature -- the choice is the memory read-out and would be a perfect predictor of the
    cue-contingent reward.)"""
    x = np.zeros(N_FEAT)
    x[IDX_BIAS] = 1.0
    cue = trial["cue"]
    if trial["onset"] <= t < trial["onset"] + CUE_STEPS:
        x[IDX_CUE_A if cue == 0 else IDX_CUE_B] = 1.0
    return x


def run_arm(arm: str, seed: int, n_trials: int = N_TRIALS) -> dict:
    """One arm, one seed. Returns the per-seed metric dict (DEV, not a verdict)."""
    rng = np.random.default_rng(seed)
    gl = 0.0 if arm == "trace_ablated" else GAMMA * LAMBDA   # the trace ablation
    w = np.zeros(N_FEAT)          # the critic (the temporal-credit mechanism)
    vstar = _analytic_vstar()

    buf_correct = 0
    late_dUS, early_dUS = [], []
    term_rpe, decoy_rpe = [], []
    val_at_cue = {0: [], 1: []}    # V(onset) per cue (late window) -- the credit read-out
    v2 = []                        # per-step (V - analytic)^2 over the late window

    for trial_i in range(n_trials):
        trial = _build_trial(rng, arm)
        onset, choice_time, reward_time = (trial["onset"], trial["choice_time"],
                                           trial["reward_time"])
        buf = np.zeros(2)
        e_c = np.zeros(N_FEAT)
        choice = None
        late = trial_i >= n_trials - LATE

        for t in range(T):
            # world: cue active? (the buffer is the PERSISTENT-MEMORY mechanism)
            cue_active = onset <= t < onset + CUE_STEPS
            buf = MEM_DECAY * buf + np.where(
                cue_active, [1.0 if trial["cue"] == 0 else 0.0,
                             1.0 if trial["cue"] == 1 else 0.0], 0.0)
            # the "terminal choice" is a FIXED read-out of the buffer (the memory
            # probe): it is UNAFFECTED by the trace ablation, so it measures memory,
            # not credit.
            if t == choice_time:
                choice = int(np.argmax(buf))
                buf_correct += 1 if choice == trial["cue"] else 0
            # reward (world): the cue is the DIRECT predictor (r = 1 iff cue is the
            # target); shuffled -> coin flip (no cue structure).
            r = 0.0
            if t == reward_time:
                if arm == "shuffled_reward":
                    r = 1.0 if rng.random() < 0.5 else 0.0   # no cue structure
                else:
                    r = 1.0 if trial["cue"] == TARGET_CUE else 0.0
            # critic value (current w, before update -- as in the validated task).
            # The critic does NOT read the buffer: the cue is represented only by the
            # transient feature (at onset), so the credit must span the delay to reach
            # it.
            x_t = _phi(t, trial, choice)
            v_t = float(x_t @ w)
            x_tp1 = _phi(t + 1, trial, choice) if t + 1 < T else np.zeros(N_FEAT)
            v_tp1 = float(x_tp1 @ w) if t + 1 < T else 0.0
            delta = r + GAMMA * v_tp1 - v_t        # the phasic-DA TD error
            if late:
                if t == reward_time:
                    late_dUS.append(abs(delta))
                if t == trial["terminal_time"]:
                    term_rpe.append(abs(delta))
                if t == trial["decoy_time"]:
                    decoy_rpe.append(abs(delta))
                if t == onset:
                    # the CUE-ANCHORED value: V at the cue onset, recorded PER CUE.
                    # The discrimination (value_A - value_B) is HIGH iff the temporal
                    # credit reached the cue (the rewarded cue is valued above the
                    # non-rewarded cue); ~0 in trace_ablated (no credit -> no value).
                    val_at_cue[trial["cue"]].append(v_t)
                # V1 (instrument validity): V vs the CUE-SPECIFIC analytic value
                tap = t - onset
                if 0 <= tap < T:
                    v2.append((v_t - vstar[trial["cue"]][tap]) ** 2)
            if trial_i < EARLY and t == reward_time:
                early_dUS.append(abs(delta))
            # critic eligibility (the reused kernel, UNMODIFIED) + weight update
            e_c = np.asarray(fused_eligibility_trace_decay(e_c, gl)) + x_t
            w = w + ALPHA_C * delta * e_c

    l_us = float(np.mean(late_dUS)) if late_dUS else 0.0
    e_us = float(np.mean(early_dUS)) if early_dUS else 0.0
    us_decay = l_us / (e_us + 1e-9)
    vrmse = float(np.sqrt(np.mean(v2))) if v2 else 0.0
    val_A = float(np.mean(val_at_cue[0])) if val_at_cue[0] else 0.0
    val_B = float(np.mean(val_at_cue[1])) if val_at_cue[1] else 0.0
    return {
        "arm": arm, "seed": seed,
        "buffer_accuracy": round(buf_correct / max(1, n_trials), 4),
        "value_at_cue_A": round(val_A, 4),
        "value_at_cue_B": round(val_B, 4),
        "value_discrimination": round(val_A - val_B, 4),
        "vrmse": round(vrmse, 4),
        "us_decay": round(us_decay, 4),
        "rpe_at_terminal": round(float(np.mean(term_rpe)) if term_rpe else 0.0, 5),
        "rpe_at_decoy": round(float(np.mean(decoy_rpe)) if decoy_rpe else 0.0, 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="7")
    ap.add_argument("--arms", type=str,
                    default="full,trace_ablated,immediate,misleading_final_cue,shuffled_reward")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_draft_long_delay_credit_dev.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rows = []
    for seed in seeds:
        for arm in arms:
            m = run_arm(arm, seed, args.n_trials)
            rows.append(m)
            sys.stderr.write("LDC_RESULT " + json.dumps(m) + "\n")
            sys.stderr.flush()
    # Attribution (the plan's whole point: WHOSE is the difference?). The trace ablation
    # (full vs trace_ablated) is the key control: what fraction of the value_discrimination
    # effect is attributable to the eligibility trace, i.e. NOT present in the control?
    # ~1.0 => the trace owns the credit; ~0 => the control has it too (the trace is not
    # load-bearing and the run is void). Measuring both arms is not the same as asking this.
    for seed in seeds:
        full = next((r for r in rows if r["seed"] == seed and r["arm"] == "full"), None)
        ab = next((r for r in rows if r["seed"] == seed and r["arm"] == "trace_ablated"), None)
        if full is not None and ab is not None:
            attributable_to(
                "eligibility trace (full vs trace_ablated), value_discrimination seed %d" % seed,
                full["value_discrimination"], ab["value_discrimination"])
    summary = {
        "task": "draft_long_delay_credit",
        "status": "DEV, not a verdict (dev seeds; Claude reviews after Tuesday)",
        "plan": "docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md section 4",
        "reuse": "sim.kernels.fused_eligibility_trace_decay (unmodified); TD-error + "
                 "analytic-vstar from sim.td_value_critic",
        "arms": arms, "seeds": seeds, "n_trials": args.n_trials,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    sys.stderr.write("LDC_WROTE " + args.out + "\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()
