"""nav close-out R4 — the DELAYED-REWARD task that proves the spiking nav VALUE-critic is load-bearing.

THE GAP THIS CLOSES (research gate 2026-06-27-nav-loop-closure-research-gate.md, R4):
The merged "one brain" nav value-critic is SPIKING-DEFAULT-ON (CYCLE 1B: enable_neural_critic +
spiking_snc + spiking_reward_us; δ=r−V is the SNc firing minus the striosome GABA_B subtraction). But
its LOAD-BEARINGNESS has only ever been shown on IMMEDIATE-reward nav, where the gridworld is
orient-solvable and a value BASELINE is not strictly necessary — so lesioning the value barely moves the
score (the #9 nav deploy was a QUALIFIED-NEGATIVE for exactly this reason, Δ7.2%, and the whole gain over
the point-neuron baseline was the NMDA on the critic slice, not the value). The canonical paradigm where a
value/critic is PROVABLY needed is a DELAYED reward (trace-conditioning logic): a temporal GAP between the
predictive event and the reward, where the ONLY way to learn the right policy is to carry a learned value
across the gap (catalog F.22/F.23; Hesslow-Yeo 2002; the eNeuro-2025 NAc-DA-encodes-the-trace-period
result; Sutton-Barto bootstrapping + eligibility traces).

THE TASK (a DELAYED-REWARD NAV variant + a 2×2 factorial):
On the SAME moving-goal nav episode (run_moving_goal_episode — ~90% reuse, NO sim/ edit), introduce a
temporal GAP between the per-step approach event and its reward delivery. The reward for the action taken
at step t is BUFFERED and delivered `delay` steps later, so the spiking SNc burst (and the corticostriatal
STDP it gates) arrives in a CS-free gap that the value critic + eligibility trace must bridge.

  2×2 FACTORIAL:  { value-critic ON, OFF }  ×  { immediate (gap=0), delayed (gap>0) }

  - value ON  = the SPIKING striosome critic subtracts V at the SNc via GABA_B (the merged default;
                enable_neural_critic=True).
  - value OFF = the SAME spiking critic, but its GABA_B→SNc subtraction is LESIONED (cp_gabab_synapse_mask
                zeroed in a prebuilt_post_init_hook — the established value lesion, _merged_navcritic_
                valuetrain.lesion_gabab) → the critic fires but cannot supply the value baseline.
  - delay      = a homeostatic_hook closure (the existing run_moving_goal_episode per-trial hook) that
                 buffers each step's reward and releases it `delay` steps later (zero edit to the episode
                 loop; the SNc/critic/STDP see the reward at the RELEASE step, in the gap).

THE DE-RISK (the load-bearing signature, validate-by-function — feedback_validate_signal_by_its_function):
  (G_HEADLINE) value ON  >  value OFF  on the DELAYED arm     (the value is load-bearing across the gap)
  (G_DISCRIM ) value ON  ≈  value OFF  on the IMMEDIATE arm    (the value is NOT needed without a gap — the
               direct answer to the orient-solvable confound that sank the #9 nav deploy)
  ⇒ the critic's HELP must be SPECIFIC to the delayed condition: a value×delay INTERACTION, not a general
    boost. Quantified: improvement_delayed (ON−OFF) ≫ improvement_immediate (ON−OFF).

ANTI-CHEATS:
  - (AC_PERMUTE) PERMUTED-DELAY control: the buffered reward is released after the gap but assigned to a
    RANDOM later step's reward magnitude (the CS→reward contingency is broken). With the contingency
    destroyed, the value cannot bridge anything → value ON ≈ value OFF on the permuted-delayed arm (the
    headline help must come from the genuine CS→reward structure, not the mere presence of a gap).
  - (AC_LESION-SPECIFIC) the value lesion (GABA_B zeroed) must collapse the DELAYED-arm advantage but NOT
    the IMMEDIATE-arm score (the G_DISCRIM gate IS this anti-cheat).
  - (AC_MOAT) the conversational no-confab moat is ARRAY-DISJOINT from the nav/limbic critic by
    construction (cp_rf_w_re/im separate from cp_connections); this nav-only harness does not touch the
    composer. (If run on the merged agent, re-assert check_moat — flagged below; this standalone harness
    builds the nav bridge only, so the moat is preserved by construction.)
  - (AC_REGIME) deterministic regime faithfulness; grid-32 (NEVER grid-8, the documented false-GO scale);
    6 seeds for the variable effect.

SCORE: the nav cost = mean Manhattan distance to the goal over the episode (LOWER is better) AND
n_steps_at_goal (HIGHER is better). The result dict's "mean_distance_overall" / "n_steps_at_goal".

DISCIPLINE (the B1/R1-a stall lesson): this module does the BUILD + a CPU smoke (the delay hook +
lesion hook + the 2×2 config matrix are pure/CPU-testable; run_moving_goal_episode is CuPy-ONLY so the
full episode is the GPU eval). The GPU eval command + de-risk criteria are in the findings doc, FLAGGED
"FOR CONTROLLER TO RUN". This module does NOT run the long GPU eval and does NOT background-and-wait.

Reproduce:
  # CPU smoke (no bridge; validates the delay hook delivers reward late + the 2×2 config matrix):
  SIM_BACKEND=numpy python -m research.runners._navcloseout_R4_delayed_reward_value --smoke

  # ONE 2×2 arm (GPU — FOR CONTROLLER): e.g. value ON, delayed
  SIM_BACKEND=cupy python -m research.runners._navcloseout_R4_delayed_reward_value \
      --arm value_on_delayed --seed 42 --grid-size 32 --n-steps 1800 --reward-delay 12

  # the full 2×2 + permuted control for one seed (GPU — FOR CONTROLLER):
  SIM_BACKEND=cupy python -m research.runners._navcloseout_R4_delayed_reward_value \
      --factorial --seed 42 --grid-size 32 --n-steps 1800 --reward-delay 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

# ── Task defaults (the validated nav-scale knobs; grid-32 per the gate, NEVER grid-8) ──
DEFAULT_GRID = 32
DEFAULT_N_STEPS = 1800
DEFAULT_REWARD_DELAY = 12     # the temporal GAP (steps) between an approach action and its reward delivery
DEFAULT_CRITIC_WARMUP = 8     # latent-learning trials so the value critic has acquired V before the test


def multi_goal_schedule(grid_size: int, n_steps: int):
    """The documented 4-phase moving-goal schedule, scaled to grid_size + n_steps.

    Mirrors g11_bg_runner's `--goal-schedule multi` (the flagship benchmark): four corner goals at ~75%
    and ~12% of grid extent, phase boundaries at the quarters of the episode. run_moving_goal_episode
    expects a LIST of (step, (gx,gy)) tuples (the string "multi" is a CLI-only convenience converted at
    the argparse level — passing it to the function directly would error).
    """
    gs = int(grid_size)
    far = (max(0, gs - 2), max(0, gs - 2))      # ~75% (was (6,6) on the 8×8 grid)
    far_west = (max(0, 1), max(0, gs - 2))      # was (1,6)
    sw = (max(0, 1), max(0, 1))                  # was (1,1)
    far_se = (max(0, gs - 2), max(0, 1))         # was (6,1)
    q = max(1, int(n_steps) // 4)
    return [(0, far), (q, far_west), (2 * q, sw), (3 * q, far_se)]


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# (1) THE DELAY HOOK — buffers each step's reward and releases it `delay` steps later.
#     Implemented via run_moving_goal_episode's existing homeostatic_hook (called every step AFTER the
#     natural reward is finalized, as: gated_reward, new_goal = hook(reward, x, y, gx, gy, step, dist)).
#     ZERO edit to run_moving_goal_episode — the SNc/critic/STDP see the reward at the RELEASE step.
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def make_delay_hook(delay_steps: int, *, permute: bool = False, permute_seed: int = 0):
    """Return a homeostatic_hook closure that delivers each step's reward `delay_steps` later.

    delay_steps == 0  -> pure pass-through (returns the reward unchanged; identical to no delay).
    delay_steps  > 0  -> FIFO buffer: at step t, gate the natural reward to 0 and enqueue it; release the
                         reward enqueued `delay_steps` ago. The reward therefore arrives in a CS-free GAP
                         that the value critic + eligibility trace must bridge.

    permute=True (AC_PERMUTE): the released reward magnitude is SHUFFLED across the buffer window (the
        CS->reward contingency is broken: the reward delivered now is NOT the one earned `delay` steps ago
        but a randomly-permuted one). With the contingency destroyed, the value cannot bridge anything.

    The hook is a stateful closure; one fresh hook per arm/seed (NEVER reuse across runs).
    The buffer is primed with `delay_steps` zeros, so the first `delay_steps` steps deliver 0 (the gap
    before the first earned reward matures) — the schedule is causal.
    """
    from collections import deque

    delay_steps = int(delay_steps)
    rng = np.random.default_rng(int(permute_seed))
    # FIFO of pending rewards; primed with `delay` zeros so step t releases what was earned at t-delay.
    # UNBOUNDED (no maxlen): we append THEN popleft each step, so the queue length stays == delay and the
    # lag is EXACTLY delay (a maxlen deque would evict on append before the popleft -> off-by-one).
    buf = deque([0.0] * max(delay_steps, 0))

    def hook(reward, x, y, gx, gy, step, dist_after):
        if delay_steps <= 0:
            return float(reward), None      # immediate arm: pass-through (no gap)
        buf.append(float(reward))           # enqueue what was just earned
        if not permute:
            return float(buf.popleft()), None   # release what was earned `delay` steps ago
        # PERMUTED (AC_PERMUTE): release a RANDOMLY-CHOSEN element of the current in-flight window instead
        # of the oldest (breaks the earned<->delivered TIME pairing while preserving the SAME gap length
        # and drawing only from genuinely-earned reward magnitudes — no fabricated reward). The chosen
        # slot is then refilled from the front and the front popped, so the queue length stays == delay
        # and the multiset of earned rewards drains over the episode.
        j = int(rng.integers(0, len(buf)))
        released = buf[j]
        buf[j] = buf[0]
        buf.popleft()
        return float(released), None

    return hook


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# (2) THE VALUE-CRITIC LESION HOOK — zeros the striosome_value->snc GABA_B route so the spiking critic
#     fires but cannot supply the value baseline (the established lesion; _merged_navcritic_valuetrain.
#     lesion_gabab). Passed as run_moving_goal_episode's prebuilt_post_init_hook (called AFTER build,
#     BEFORE the episode loop). The lesion PERSISTS for the whole episode (no restore — value is OFF).
#     ZERO edit to run_moving_goal_episode.
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def make_value_lesion_hook(extra_hook=None):
    """Return a prebuilt_post_init_hook that zeros cp_gabab_synapse_mask (the value-critic GABA_B lesion).

    extra_hook: an optional second prebuilt_post_init_hook to chain (e.g. a conv-freeze hook on the merged
                bridge). Called first, then the GABA_B lesion is applied.
    Returns (hook, box) where box["n_gabab_cut"] is filled at build time (the number of GABA_B synapses
    silenced — a non-zero count is the proof the lesion landed on a real value route).
    """
    box = {"n_gabab_cut": None}

    def hook(bridge):
        if extra_hook is not None:
            extra_hook(bridge)
        import cupy as cp  # noqa: F401  (the GPU episode path)
        m_mask = getattr(bridge, "cp_gabab_synapse_mask", None)
        if m_mask is None:
            box["n_gabab_cut"] = 0
            return
        n_cut = int((m_mask.get() if hasattr(m_mask, "get") else np.asarray(m_mask)).sum())
        bridge.cp_gabab_synapse_mask = cp.zeros_like(m_mask)
        if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
            bridge.cp_conductance_g_gabab[:] = cp.float32(0.0)
        box["n_gabab_cut"] = n_cut

    return hook, box


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# (3) THE 2×2 ARM CONFIG MATRIX — value {ON,OFF} × timing {immediate,delayed} (+ the permuted control).
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# Each arm = (value_on: bool, reward_delay: int, permute: bool). The episode kwargs are otherwise shared
# (the same merged spiking-default nav config) so the ONLY differences are the two factors.
ARM_SPECS = {
    "value_on_immediate":  dict(value_on=True,  reward_delay=0,  permute=False),
    "value_off_immediate": dict(value_on=False, reward_delay=0,  permute=False),
    "value_on_delayed":    dict(value_on=True,  reward_delay=DEFAULT_REWARD_DELAY, permute=False),
    "value_off_delayed":   dict(value_on=False, reward_delay=DEFAULT_REWARD_DELAY, permute=False),
    # AC_PERMUTE: the contingency-broken delayed arms (value ON/OFF) — the headline help must vanish here.
    "value_on_delayed_permuted":  dict(value_on=True,  reward_delay=DEFAULT_REWARD_DELAY, permute=True),
    "value_off_delayed_permuted": dict(value_on=False, reward_delay=DEFAULT_REWARD_DELAY, permute=True),
}


def build_episode_kwargs(arm: str, *, seed: int, grid_size: int, n_steps: int,
                         reward_delay: int, critic_warmup_trials: int):
    """Assemble the run_moving_goal_episode kwargs for one 2×2 arm.

    The shared base is the SPIKING-DEFAULT merged nav config (CYCLE 1B): perceived_approach_reward +
    spiking_snc + enable_neural_critic + spiking_reward_us + enable_critic_homeostasis are all the
    library defaults of run_moving_goal_episode, so we pass ONLY the factor knobs explicitly.

    Returns (kwargs, value_lesion_box) — value_lesion_box is None for value-ON arms.
    """
    spec = ARM_SPECS[arm].copy()
    if spec["reward_delay"] != 0:
        spec["reward_delay"] = int(reward_delay)   # allow CLI override of the gap length

    delay_hook = make_delay_hook(
        spec["reward_delay"], permute=spec["permute"], permute_seed=seed)

    kwargs = dict(
        seed=seed, n_steps=int(n_steps), grid_size=int(grid_size),
        # multi-goal moving-goal schedule (the documented flagship benchmark task) — the LIST form
        # (run_moving_goal_episode expects (step,(gx,gy)) tuples; "multi" is a CLI-only convenience).
        goal_schedule=multi_goal_schedule(grid_size, n_steps),
        # the SPIKING-DEFAULT merged nav limbic core (CYCLE 1B) — these ARE the library defaults; passed
        # explicitly here so the harness is self-documenting and robust to any future default change.
        perceived_approach_reward=True, spiking_snc=True,
        enable_neural_critic=True, enable_critic_homeostasis=True, spiking_reward_us=True,
        # latent-learning warm-up so the value critic has acquired V BEFORE the test (the value can only
        # be load-bearing if it has learned something to predict across the gap).
        critic_warmup_trials=int(critic_warmup_trials),
        # the soft-bound that stops the actor saturating during reward-STDP at nav scale (the documented
        # nav default; the actor is ceiling-bound, not soft-bound — see step2a finding).
        stdp_w_max_override=400.0,
        # the DELAY: buffer each step's reward, release it `delay` steps later (zero-edit homeostatic_hook).
        homeostatic_hook=delay_hook,
        verbose=False, progress_print_interval=0,
    )

    value_lesion_box = None
    if not spec["value_on"]:
        # value OFF: lesion the striosome_value->snc GABA_B subtraction (the spiking critic still fires,
        # but supplies NO value baseline). prebuilt_post_init_hook fires AFTER build, before the episode.
        lesion_hook, box = make_value_lesion_hook()
        kwargs["prebuilt_post_init_hook"] = lesion_hook
        value_lesion_box = box

    return kwargs, value_lesion_box


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# (4) RUN ONE ARM (GPU) — calls run_moving_goal_episode. NOT called by the CPU smoke.
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def run_arm(arm: str, *, seed: int, grid_size: int, n_steps: int, reward_delay: int,
            critic_warmup_trials: int, out_dir: str):
    """Run one 2×2 arm on the real bridge (GPU). Returns the arm's score summary."""
    from research.runners.g11_bg_runner import run_moving_goal_episode

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"R4_{arm}_seed{seed}.json")
    kwargs, value_lesion_box = build_episode_kwargs(
        arm, seed=seed, grid_size=grid_size, n_steps=n_steps,
        reward_delay=reward_delay, critic_warmup_trials=critic_warmup_trials)

    print(f"[R4 arm={arm} seed={seed}] grid={grid_size} n_steps={n_steps} "
          f"delay={ARM_SPECS[arm]['reward_delay'] if ARM_SPECS[arm]['reward_delay']==0 else reward_delay} "
          f"value_on={ARM_SPECS[arm]['value_on']} permute={ARM_SPECS[arm]['permute']}", flush=True)
    res = run_moving_goal_episode(out_path=out_path, **kwargs)
    # run_moving_goal_episode writes out_path; it also returns the result dict (read either).
    if res is None:
        with open(out_path) as f:
            res = json.load(f)

    summary = dict(
        arm=arm, seed=int(seed), grid_size=int(grid_size), n_steps=int(n_steps),
        reward_delay=int(ARM_SPECS[arm]["reward_delay"] if ARM_SPECS[arm]["reward_delay"] == 0 else reward_delay),
        value_on=bool(ARM_SPECS[arm]["value_on"]), permute=bool(ARM_SPECS[arm]["permute"]),
        mean_distance_overall=float(res.get("mean_distance_overall", float("nan"))),
        n_steps_at_goal=int(res.get("n_steps_at_goal", 0)),
        mean_distance_quarters=res.get("mean_distance_quarters"),
        n_gabab_cut=(value_lesion_box["n_gabab_cut"] if value_lesion_box is not None else None),
        out_path=out_path,
    )
    print(f"[R4 arm={arm} seed={seed}] mean_dist={summary['mean_distance_overall']:.3f} "
          f"at_goal={summary['n_steps_at_goal']} gabab_cut={summary['n_gabab_cut']}", flush=True)
    return summary


def factorial(seed: int, *, grid_size: int, n_steps: int, reward_delay: int,
              critic_warmup_trials: int, out_dir: str, include_permuted: bool = True):
    """Run the full 2×2 (+ permuted control) for one seed and compute the load-bearing verdict."""
    arms = ["value_on_immediate", "value_off_immediate", "value_on_delayed", "value_off_delayed"]
    if include_permuted:
        arms += ["value_on_delayed_permuted", "value_off_delayed_permuted"]

    summ = {a: run_arm(a, seed=seed, grid_size=grid_size, n_steps=n_steps, reward_delay=reward_delay,
                       critic_warmup_trials=critic_warmup_trials, out_dir=out_dir) for a in arms}
    return summarize_factorial(summ, seed=seed)


def summarize_factorial(summ: dict, *, seed: int):
    """Compute the value×delay interaction from a 2×2(+permuted) summary dict.

    Score = mean_distance_overall (LOWER is better), so an IMPROVEMENT = OFF − ON (positive when ON is
    lower/better). The load-bearing signature: improvement_delayed ≫ improvement_immediate, and the
    permuted-delayed improvement ≈ 0.
    """
    def md(a):
        return float(summ[a]["mean_distance_overall"]) if a in summ else float("nan")

    imp_immediate = md("value_off_immediate") - md("value_on_immediate")     # ON better => positive
    imp_delayed = md("value_off_delayed") - md("value_on_delayed")
    interaction = imp_delayed - imp_immediate                                # the value×delay interaction
    out = dict(
        seed=int(seed),
        mean_distance=dict((a, md(a)) for a in summ),
        improvement_immediate=imp_immediate,
        improvement_delayed=imp_delayed,
        value_x_delay_interaction=interaction,
        # per-seed PASS proxies (the multi-seed verdict aggregates these):
        helps_on_delayed=bool(imp_delayed > 0.0),
        neutral_on_immediate=bool(abs(imp_immediate) <= max(0.5, 0.5 * abs(imp_delayed))),
        interaction_positive=bool(interaction > 0.0),
    )
    if "value_on_delayed_permuted" in summ:
        imp_permuted = md("value_off_delayed_permuted") - md("value_on_delayed_permuted")
        out["improvement_delayed_permuted"] = imp_permuted
        # AC_PERMUTE: the genuine help must exceed the permuted (contingency-broken) help.
        out["permute_control_ok"] = bool(imp_delayed > imp_permuted)
    out["summary"] = summ
    return out


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# (5) THE CPU SMOKE — pure/CPU validation: the delay hook + lesion hook + the 2×2 config matrix are
#     well-formed, and reward arrives DELAYED. run_moving_goal_episode is CuPy-only, so the full episode
#     is the GPU eval (flagged for the controller). NO bridge built here.
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def smoke():
    print("=" * 78)
    print("[R4 SMOKE] CPU validation of the delayed-reward 2×2 harness (no bridge; run_moving_goal_episode is CuPy-only)")
    print("=" * 78)
    ok = True

    # ── (a) the delay hook delivers reward DELAYED by exactly `delay` steps ──
    delay = 5
    hook = make_delay_hook(delay)
    earned = [0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0]
    delivered = []
    for t, r in enumerate(earned):
        d, ng = hook(r, 0, 0, 0, 0, t, 0)
        delivered.append(d)
        assert ng is None
    # released[t] should == earned[t-delay] (0 for t<delay). Verify the lag is EXACTLY `delay`.
    expected = [0.0] * delay + earned[:-delay]
    lag_ok = (delivered == expected)
    total_in = sum(earned[:-delay])     # rewards that have had time to mature within the window
    total_out = sum(delivered)
    print(f"  (a) delay hook: delay={delay}")
    print(f"      earned   : {earned}")
    print(f"      delivered: {delivered}")
    print(f"      expected : {expected}")
    print(f"      lag == delay (delivered[t]==earned[t-delay], 0 for t<delay): {lag_ok}")
    print(f"      reward conserved over matured window (in {total_in} == out {total_out}): {total_in == total_out}")
    ok = ok and lag_ok and (total_in == total_out)

    # ── (b) immediate arm (delay=0) is a pure pass-through (byte-identical reward, no gap) ──
    hook0 = make_delay_hook(0)
    passthrough = [hook0(r, 0, 0, 0, 0, t, 0)[0] for t, r in enumerate(earned)]
    pass_ok = (passthrough == earned)
    print(f"  (b) immediate arm (delay=0) pass-through == earned: {pass_ok}")
    ok = ok and pass_ok

    # ── (c) the PERMUTED control breaks the earned<->delivered timing but preserves the gap + multiset ──
    permh = make_delay_hook(delay, permute=True, permute_seed=42)
    perm_delivered = [permh(r, 0, 0, 0, 0, t, 0)[0] for t, r in enumerate(earned)]
    # the permuted stream is NOT the simply-lagged stream (contingency broken) but draws from the same
    # reward values (no fabricated reward; it's a shuffle within the rolling window).
    contingency_broken = (perm_delivered != expected)
    drawn_from_earned = set(perm_delivered).issubset(set(earned) | {0.0})
    print(f"  (c) permuted control: delivered != simply-lagged (contingency broken): {contingency_broken}")
    print(f"      permuted stream draws only from earned reward values (no fabricated reward): {drawn_from_earned}")
    print(f"      permuted delivered: {perm_delivered}")
    ok = ok and contingency_broken and drawn_from_earned

    # ── (d) the value-lesion hook zeros a (mock) GABA_B mask and records the cut count ──
    class _MockBridge:
        def __init__(self):
            self.cp_gabab_synapse_mask = np.array([1, 0, 1, 1, 0, 1], dtype=np.int32)
            self.cp_conductance_g_gabab = np.ones(6, dtype=np.float32)

    # the real hook imports cupy; test the LOGIC against a numpy mock by inlining the same operation.
    mock = _MockBridge()
    n_cut = int(mock.cp_gabab_synapse_mask.sum())
    mock.cp_gabab_synapse_mask = np.zeros_like(mock.cp_gabab_synapse_mask)
    mock.cp_conductance_g_gabab[:] = 0.0
    lesion_ok = (n_cut == 4 and int(mock.cp_gabab_synapse_mask.sum()) == 0
                 and float(mock.cp_conductance_g_gabab.sum()) == 0.0)
    print(f"  (d) value-lesion logic: n_gabab_cut={n_cut} (expected 4), mask zeroed + g_gabab cleared: {lesion_ok}")
    ok = ok and lesion_ok

    # ── (e) the 2×2 config matrix is well-formed: the two factors are set correctly per arm ──
    matrix_ok = True
    for arm, spec in ARM_SPECS.items():
        kwargs, vbox = build_episode_kwargs(
            arm, seed=42, grid_size=DEFAULT_GRID, n_steps=DEFAULT_N_STEPS,
            reward_delay=DEFAULT_REWARD_DELAY, critic_warmup_trials=DEFAULT_CRITIC_WARMUP)
        # value ON  => no lesion hook (vbox None, no prebuilt_post_init_hook for the value lesion).
        # value OFF => a lesion hook is wired (vbox present, prebuilt_post_init_hook set).
        value_factor_ok = ((spec["value_on"] and vbox is None and "prebuilt_post_init_hook" not in kwargs)
                           or (not spec["value_on"] and vbox is not None and "prebuilt_post_init_hook" in kwargs))
        # timing factor => a homeostatic_hook is always wired; the gap matches the spec.
        timing_hook = kwargs.get("homeostatic_hook")
        timing_ok = callable(timing_hook)
        # the shared spiking-default nav config is present (goal_schedule is the multi LIST of 4 phases).
        gsched = kwargs.get("goal_schedule")
        spiking_ok = (kwargs.get("spiking_snc") and kwargs.get("enable_neural_critic")
                      and kwargs.get("spiking_reward_us") and kwargs.get("perceived_approach_reward")
                      and isinstance(gsched, list) and len(gsched) == 4 and gsched[0][0] == 0)
        arm_ok = value_factor_ok and timing_ok and spiking_ok
        matrix_ok = matrix_ok and arm_ok
        print(f"  (e) arm {arm:30s}: value_factor={value_factor_ok} timing_hook={timing_ok} "
              f"spiking_default={spiking_ok} -> {arm_ok}")
    ok = ok and matrix_ok

    # ── (f) the verdict aggregator computes the value×delay interaction from a synthetic 2×2 ──
    #   synth: value HELPS on delayed (OFF worse), neutral on immediate, permuted help ~0.
    synth = {
        "value_on_immediate":  {"mean_distance_overall": 6.0},
        "value_off_immediate": {"mean_distance_overall": 6.1},   # ~neutral
        "value_on_delayed":    {"mean_distance_overall": 6.5},
        "value_off_delayed":   {"mean_distance_overall": 9.0},   # value OFF much worse on delayed
        "value_on_delayed_permuted":  {"mean_distance_overall": 8.8},
        "value_off_delayed_permuted": {"mean_distance_overall": 9.0},  # permuted: ~no help
    }
    v = summarize_factorial(synth, seed=42)
    verdict_ok = (v["helps_on_delayed"] and v["neutral_on_immediate"]
                  and v["interaction_positive"] and v["permute_control_ok"])
    print(f"  (f) verdict aggregator on synthetic 2×2: imp_immediate={v['improvement_immediate']:.2f} "
          f"imp_delayed={v['improvement_delayed']:.2f} interaction={v['value_x_delay_interaction']:.2f} "
          f"permute_ok={v['permute_control_ok']} -> {verdict_ok}")
    ok = ok and verdict_ok

    print("=" * 78)
    print(f"[R4 SMOKE] {'PASS' if ok else 'FAIL'} — delay hook delivers reward late, immediate=pass-through, "
          f"permuted breaks contingency, value-lesion logic sound, 2×2 matrix well-formed, verdict aggregator sound.")
    print("[R4 SMOKE] (run_moving_goal_episode is CuPy-only -> the full 2×2 episode is the GPU eval; see the findings doc.)")
    print("=" * 78)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="CPU smoke (no bridge): validate the harness is well-formed")
    ap.add_argument("--arm", type=str, default=None, choices=sorted(ARM_SPECS.keys()),
                    help="run ONE 2×2 arm on the real bridge (GPU)")
    ap.add_argument("--factorial", action="store_true", help="run the full 2×2 (+permuted) for one seed (GPU)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid-size", type=int, default=DEFAULT_GRID)
    ap.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    ap.add_argument("--reward-delay", type=int, default=DEFAULT_REWARD_DELAY)
    ap.add_argument("--critic-warmup-trials", type=int, default=DEFAULT_CRITIC_WARMUP)
    ap.add_argument("--no-permuted", action="store_true", help="skip the permuted-delay control arms")
    ap.add_argument("--out-dir", type=str, default="research/findings/raw/navcloseout_R4")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        ok = smoke()
        raise SystemExit(0 if ok else 1)

    if args.arm is not None:
        s = run_arm(args.arm, seed=args.seed, grid_size=args.grid_size, n_steps=args.n_steps,
                    reward_delay=args.reward_delay, critic_warmup_trials=args.critic_warmup_trials,
                    out_dir=args.out_dir)
        path = args.out or os.path.join(args.out_dir, f"R4_{args.arm}_seed{args.seed}_summary.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(s, f, indent=2)
        print(f"  wrote {path}")
        return

    if args.factorial:
        v = factorial(args.seed, grid_size=args.grid_size, n_steps=args.n_steps,
                      reward_delay=args.reward_delay, critic_warmup_trials=args.critic_warmup_trials,
                      out_dir=args.out_dir, include_permuted=not args.no_permuted)
        path = args.out or os.path.join(args.out_dir, f"R4_factorial_seed{args.seed}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(v, f, indent=2)
        print(f"\n[R4 FACTORIAL seed={args.seed}] imp_immediate={v['improvement_immediate']:.3f} "
              f"imp_delayed={v['improvement_delayed']:.3f} interaction={v['value_x_delay_interaction']:.3f}")
        print(f"  helps_on_delayed={v['helps_on_delayed']} neutral_on_immediate={v['neutral_on_immediate']} "
              f"interaction_positive={v['interaction_positive']} "
              f"permute_control_ok={v.get('permute_control_ok')}")
        print(f"  wrote {path}")
        return

    ap.error("pass --smoke (CPU), --arm <name> (one GPU arm), or --factorial (full 2×2, GPU)")


if __name__ == "__main__":
    main()
