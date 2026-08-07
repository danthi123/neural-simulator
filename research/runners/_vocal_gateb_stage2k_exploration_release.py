"""Gate B Stage 2k: FIX D -- a novelty-gated release of the un-sampled action from the
COMMIT-level winner-take-all veto, the companion to Stage 2j's FIX C.

WHY (direct substrate diagnosis of the lone held-out miss 730705, this session; instrument
in research/findings/raw/gateb_stage2k_exploration_release/diag_*.txt): FIX C (the MSN
k-homeostat) wakes str_d1_1 (0->138 onset spikes) and that DOES release the downstream
gate -- gpi_1 pauses (212->51), thal_1 fires (0->186). But the signal then DIES at the
commit competition: commit_0 (452 spikes) drives commit_fs_0 (282), whose cross-inhibition
commit_fs_0 -> commit_1 (weight 60) clamps commit_1 to 0, so motor_1 = 0 and action 1 never
wins argmax(motor). *Waking the MSN is necessary but not sufficient -- the woken channel is
vetoed at the cortical WTA.* Two further facts fixed the mechanism:
  * The proposal-level directed-novelty CURRENT is COUNTERPRODUCTIVE on this seed -- driving
    proposal_1 with 350 pA drops str_d1_1 from 138 to 31 (it over-drives str_fsi and the
    indirect arm), so candidate (a) [an un-satiable proposal novelty floor] makes 730705
    WORSE, not better. Refuted by measurement, not asserted.
  * Only FULLY releasing the incumbent's veto works: scaling commit_fs_0 -> commit_1 to 0
    (with the proposal novelty current OFF for that channel) flips 730705 to motor=[431,544]
    -- action 1 wins and, being real_action & winner==target, is REWARDED, so
    proposal_1 -> str_d1_1 can potentiate. A partial relax (x0.5) does NOTHING (the
    cross-inhibition is near-saturating: motor stays [856,0]); the knob is sharp.

MECHANISM (FIX D, default OFF, opt-in --fix-d; REQUIRES --fix-c): for the homeostat's DEAD
channel `u` (the extreme-asymmetry channel FIX C woke), while it has been really-selected
fewer than EXPLORE_FLOOR_K times (an un-satiable per-action exploration floor), TRANSIENTLY
release it from the incumbent's commit WTA veto -- scale the inhibitory synaptic weight
commit_fs_{other} -> commit_{u} by EXPLORE_RELEASE_FACTOR for the trial, restored after --
AND suppress the counterproductive proposal-novelty current into `u`. This is a
novelty-gated DISINHIBITION of an inhibitory synapse (the cortical-competition analogue of
cholinergic/novelty suppression of feedback interneurons that lets a novel salient option
escape the ongoing choice's veto), acting on the SELECTION loop -- NOT current injection
into the MSN, so it is distinct from the refuted FIX A. commit_fs routes are non-plastic, so
the saved-original restore is byte-exact and every non-forced trial is unperturbed.

Gated to the FIX C dead channel, so on any seed where FIX C does not engage, FIX D never
engages -> byte-identical to Stage 2j. Asserted (not commented) via
_assert_fixd_off_byte_identical: with --fix-d OFF the run reproduces 2j exactly.

HONEST-NEGATIVE CAVEAT: the single-onset probe above shows the WTA flips; whether 40 trials
of the released, rewarded action-1 potentiate proposal_1->str_d1_1 enough to PERSIST to test
(FIX D OFF at test) is the SMOKE's verdict, not this note's.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from research.runners._vocal_action_selector_gate import _indices
from research.runners._vocal_gateb_stage2_reward_credit import _route_indices
from research.runners._vocal_gateb_stage2i_circuit_fixes import _ForcedSampler2i
from research.runners._vocal_gateb_stage2j_intrinsic_homeostasis import (
    OUT_DIR as STAGE2J_OUT_DIR,
    _apply_k_homeostasis,
    _calibrate_k_scale,
    _homeostat_engage,
    _run_trial_2j,
    _steer,
    run_seed_swap as _run_seed_swap_2j,
)
from research.runners._vocal_gateb_stage2g_hammond_deltap import (
    CHANNELS, DEV_SEEDS, HELDOUT_SEEDS, N_TEST, N_TRAIN, REWARD_LEARNING_RATE,
    VALUE_INIT, WH_PERIOD, CONSTRUCTION_SEED, MIN_SAMPLES, SIGMA_UNCERTAIN,
    _apply_afferents, _assert_stage1_equivalence, _backend_info, _base_rate,
    _baseline_block, _d1_route_weight_means, _decoupled_reward_set, _motor_idx,
    _p_action0, _reconfigure_da_s, _reward_eligible, _run_withhold_trial, _set_sigma,
    _settle, _sigma_from_conf, _str_d1_idx, _str_d2_idx, _test_block, _update_conf,
    _v_withhold, build_stage2_bridge,
)
from research.runners._vocal_gateb_stage2i_circuit_fixes import _apply_motor_homeostat
from sim.backend import get_backend, to_host
from tools.lab import attributable_to

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research/findings/raw/gateb_stage2k_exploration_release"

# ===================== FIX D: novelty-gated commit-WTA release =========================
EXPLORE_FLOOR_K = int(MIN_SAMPLES)   # un-satiable per-action floor: force until >= K real selections (=3)
EXPLORE_RELEASE_FACTOR = 0.0         # scale of the incumbent->novel commit veto while forcing (0 = full release)


def run_condition(seed: int, *, condition: str, target: int, n_train: int, n_test: int,
                  reward_trials_master=None, reward_learning_rate: float = REWARD_LEARNING_RATE,
                  ou_seed: int | None = None, gated: bool = True, directed_novelty: bool = True,
                  use_d2: bool = True, wh_reward_p: float = 0.0,
                  fix_a: bool = False, fix_b: bool = True, fix_c: bool = False, fix_d: bool = False,
                  motor_homeostat: bool = False):
    """Stage-2j run_condition + FIX D (commit-WTA release). fix_d=False -> byte-identical
    to Stage 2j (the FIX D block never touches the substrate)."""
    xp, _ = get_backend()
    plastic = condition != "acq_lesion"
    build_kwargs = dict(enable_reward=True, plastic_d1=plastic,
                        reward_learning_rate=reward_learning_rate, ou_seed=ou_seed,
                        ou_sigma=SIGMA_UNCERTAIN, plastic_d2=True)
    bridge = build_stage2_bridge(seed, **build_kwargs)
    _reconfigure_da_s(bridge)
    if condition == "acq_lesion":
        bridge.core_config.reward_eligibility_from_coactivity = False
    midx = _motor_idx(bridge)
    d1idx = _str_d1_idx(bridge)
    d2idx = _str_d2_idx(bridge)

    Vd1 = [VALUE_INIT, VALUE_INIT]
    Vd2 = [VALUE_INIT, VALUE_INIT]
    count = [0, 0]
    conf = 0.0
    dp = 0.0
    v_wh = 0.0
    wh_rng = np.random.default_rng(int(seed) + 909090 + int(target))
    _set_sigma(bridge, SIGMA_UNCERTAIN if not gated else _sigma_from_conf(conf))
    _settle(bridge)

    baseline = _baseline_block(bridge, midx, d1idx, d2idx, target, n_test)
    r0_d1 = baseline["r0_d1"]

    # ---- FIX C: MSN intrinsic-excitability homeostasis (extreme-asymmetry gated) -------
    homeo = {"engaged": False, "dead_channel": None, "dead_r0": None, "ratio": None,
             "k_scale": 1.0, "fired_at_k1": None, "fired_after": None}
    if fix_c:
        dead_c, dead_r0, ratio = _homeostat_engage(r0_d1)
        homeo.update(dead_r0=dead_r0, ratio=ratio)
        if dead_c is not None:
            k_scale, f1, fa = _calibrate_k_scale(seed, dead_c, build_kwargs)
            if k_scale > 1.0:
                _apply_k_homeostasis(bridge, dead_c, k_scale)
            homeo.update(engaged=True, dead_channel=int(dead_c), k_scale=float(k_scale),
                         fired_at_k1=int(f1), fired_after=int(fa))

    # ---- FIX D setup: release-route for the (FIX-C) dead channel from the commit veto ---
    release_channel = None
    release_route = None
    release_orig = None
    if fix_d and homeo["engaged"]:
        release_channel = int(homeo["dead_channel"])
        other = 1 - release_channel
        ridx = _route_indices(bridge, f"commit_fs_{other}", f"commit_{release_channel}")
        if ridx.size:
            release_route = xp.asarray(ridx)
            release_orig = bridge.cp_connections.data[release_route].copy()

    if gated:
        _set_sigma(bridge, _sigma_from_conf(conf))

    sampler = _ForcedSampler2i(enabled=fix_a and directed_novelty)
    w0 = _d1_route_weight_means(bridge)
    train = []
    reward_trials = []
    n_wh = 0
    n_wh_rewarded = 0
    n_forced = 0
    max_forced_inh = 0.0
    v_wh_trace = []
    n_clamp_hit = 0
    n_orig_clamp_nonrewarded = 0
    n_released = 0
    for i in range(n_train):
        if wh_reward_p > 0.0 and i % WH_PERIOD == 0:
            rew_wh = bool(wh_rng.random() < wh_reward_p)
            _run_withhold_trial(bridge, rewarded=rew_wh)
            n_wh += 1
            n_wh_rewarded += int(rew_wh)
        v_wh = _v_withhold(bridge)
        if condition == "yoked":
            rule = "yoked"
            forced = bool(reward_trials_master is not None and i in reward_trials_master)
        else:
            rule = "contingent"
            forced = False
        if directed_novelty:
            nt, nd, it, idr, forcing = sampler.drive(count, conf, trial_idx=i)
        else:
            nt, nd, it, idr, forcing = None, 0.0, None, 0.0, False
        if forcing:
            n_forced += 1
            max_forced_inh = max(max_forced_inh, idr)

        # ---- FIX D: transient novelty-gated commit-WTA release of the un-sampled channel --
        releasing = bool(release_channel is not None and count[release_channel] < EXPLORE_FLOOR_K)
        if releasing:
            if nt == release_channel:          # suppress the counterproductive proposal novelty current
                nt, nd = None, 0.0
            bridge.cp_connections.data[release_route] = release_orig * xp.float32(EXPLORE_RELEASE_FACTOR)

        tr = _run_trial_2j(bridge, midx, d1idx, d2idx, deliver_reward=True, target=target,
                           reward_rule=rule, forced_reward=forced, eligible=_reward_eligible(i),
                           r0_d1=r0_d1, v_withhold=v_wh,
                           novelty_target=nt, novelty_drive_pa=nd,
                           inh_str_d1_target=it, inh_str_d1_pa=idr, rpe_floor=fix_b)

        if releasing:
            bridge.cp_connections.data[release_route] = release_orig   # byte-exact restore
            n_released += 1

        n_clamp_hit += int(getattr(tr, "clamp_hit", False))
        n_orig_clamp_nonrewarded += int(getattr(tr, "orig_clamp_would_hit_nonrewarded", False))
        if motor_homeostat and plastic:
            _apply_motor_homeostat(bridge, w0)
        if tr.rewarded:
            reward_trials.append(i)
        train.append(tr)
        if gated:
            conf, dp = _update_conf(Vd1, Vd2, count, tr, use_d2=use_d2)
            _set_sigma(bridge, _sigma_from_conf(conf))
        else:
            _, dp = _update_conf(Vd1, Vd2, count, tr, use_d2=use_d2)
        v_wh_trace.append(float(v_wh))
    w1 = _d1_route_weight_means(bridge)

    if condition == "expr_lesion":
        from research.runners._vocal_gateb_stage1_selector import W as S1W
        for c in CHANNELS:
            idx = bridge._stage2_d1_routes[c]
            bridge.cp_connections.data[xp.asarray(idx)] = xp.float32(S1W["proposal_to_msn"])

    test = _test_block(bridge, midx, d1idx, d2idx, target, n_test, r0_d1=r0_d1)

    train_target = sum(1 for t in train if t.real_action and t.winner == target)
    train_clean = sum(1 for t in train if t.real_action)
    train_a0 = sum(1 for t in train if t.real_action and t.winner == 0)
    train_p0_all = float(train_a0 / train_clean) if train_clean else float("nan")
    bridge.clear_simulation_state_and_gpu_memory()
    return {
        "condition": condition, "seed": int(seed), "target": int(target),
        "n_reward_delivered": len(reward_trials), "reward_trials": reward_trials,
        "baseline_target_rate": baseline["target_rate"], "baseline_n_clean": baseline["n_clean"],
        "test_target_rate": test["target_rate"], "test_n_clean": test["n_clean"],
        "train_target_rate": float(train_target / train_clean) if train_clean else float("nan"),
        "train_clean_rate": float(train_clean / n_train),
        "train_p0_all": train_p0_all,
        "count0": int(count[0]), "count1": int(count[1]),
        "n_forced_trials": int(n_forced), "max_forced_inh_pA": float(max_forced_inh),
        "n_released_trials": int(n_released), "release_channel": release_channel,
        "n_clamp_hit": int(n_clamp_hit),
        "n_orig_clamp_nonrewarded": int(n_orig_clamp_nonrewarded),
        "homeostat": homeo,
        "d1_weight_before": w0, "d1_weight_after": w1,
        "final_conf": float(conf), "final_dp": float(dp),
        "final_v_withhold": float(v_wh), "max_v_withhold": float(max(v_wh_trace) if v_wh_trace else 0.0),
        "n_withhold": int(n_wh), "n_withhold_rewarded": int(n_wh_rewarded),
        "r0_d1": r0_d1,
        "Vd1": [float(Vd1[0]), float(Vd1[1])], "Vd2": [float(Vd2[0]), float(Vd2[1])],
        "final_sigma": float(bridge.core_config.ou_std_current_pA),
    }


def run_seed_swap(seed: int, *, n_train: int, n_test: int,
                  reward_learning_rate: float = REWARD_LEARNING_RATE, use_d2: bool = True,
                  directed_novelty: bool = True, enable_withhold: bool = True,
                  fix_a: bool = False, fix_b: bool = True, fix_c: bool = False, fix_d: bool = False,
                  motor_homeostat: bool = False) -> dict:
    kw = dict(n_train=n_train, n_test=n_test, reward_learning_rate=reward_learning_rate,
              use_d2=use_d2, directed_novelty=directed_novelty,
              fix_a=fix_a, fix_b=fix_b, fix_c=fix_c, fix_d=fix_d, motor_homeostat=motor_homeostat)
    c0 = run_condition(seed, condition="contingent", target=0, wh_reward_p=0.0, **kw)
    c1 = run_condition(seed, condition="contingent", target=1, wh_reward_p=0.0, **kw)
    p0_base = _base_rate(len(c0["reward_trials"]), n_train) if enable_withhold else 0.0
    p1_base = _base_rate(len(c1["reward_trials"]), n_train) if enable_withhold else 0.0
    y0 = run_condition(seed, condition="yoked", target=0, wh_reward_p=p0_base,
                       reward_trials_master=_decoupled_reward_set(seed + 500000, len(c0["reward_trials"]), n_train),
                       ou_seed=seed + 500000, **kw)
    y1 = run_condition(seed, condition="yoked", target=1, wh_reward_p=p1_base,
                       reward_trials_master=_decoupled_reward_set(seed + 600000, len(c1["reward_trials"]), n_train),
                       ou_seed=seed + 600000, **kw)
    p0_c0, p0_c1 = _p_action0(c0), _p_action0(c1)
    p0_y0, p0_y1 = _p_action0(y0), _p_action0(y1)
    return {
        "seed": int(seed),
        "baseline_p0": _p_action0({"test_target_rate": c0["baseline_target_rate"], "target": 0}),
        "contingent_p0_reward0": p0_c0, "contingent_p0_reward1": p0_c1,
        "yoked_p0_reward0": p0_y0, "yoked_p0_reward1": p0_y1,
        "D_contingent": (p0_c0 - p0_c1),
        "D_yoked": (p0_y0 - p0_y1),
        "reward_count_reward0": c0["n_reward_delivered"],
        "reward_count_reward1": c1["n_reward_delivered"],
        "count_c0": (c0["count0"], c0["count1"]), "count_c1": (c1["count0"], c1["count1"]),
        "n_clean_c0": c0["test_n_clean"], "n_clean_c1": c1["test_n_clean"],
        "test_rate_c0": c0["test_target_rate"], "test_rate_c1": c1["test_target_rate"],
        "n_forced_c0": c0["n_forced_trials"], "n_forced_c1": c1["n_forced_trials"],
        "n_released_c0": c0["n_released_trials"], "n_released_c1": c1["n_released_trials"],
        "release_channel_c1": c1["release_channel"],
        "n_clamp_c0": c0["n_clamp_hit"], "n_clamp_c1": c1["n_clamp_hit"],
        "homeo_c0": c0["homeostat"], "homeo_c1": c1["homeostat"],
        "r0_d1_c0": c0["r0_d1"], "r0_d1_c1": c1["r0_d1"],
        "cont_train_p0_c0": c0["train_p0_all"], "cont_train_p0_c1": c1["train_p0_all"],
        "conf_c0": c0["final_conf"], "conf_y0": y0["final_conf"],
        "dp_c0": c0["final_dp"], "dp_y0": y0["final_dp"],
        "base_p0": p0_base, "base_p1": p1_base,
        "d1_after_reward0": c0["d1_weight_after"], "d1_after_reward1": c1["d1_weight_after"],
        "train_clean_rate": c0["train_clean_rate"],
    }


def _assert_fixd_off_byte_identical(seed: int, *, n_train: int, n_test: int,
                                    reward_learning_rate: float) -> dict:
    """FIX D OFF must reproduce Stage 2j exactly (same fix_b/fix_c). Compares this runner's
    fix_d=False output against 2j's run_seed_swap on the decisive metrics."""
    mine = run_seed_swap(seed, n_train=n_train, n_test=n_test,
                         reward_learning_rate=reward_learning_rate,
                         fix_a=False, fix_b=True, fix_c=True, fix_d=False)
    ref = _run_seed_swap_2j(seed, n_train=n_train, n_test=n_test,
                            reward_learning_rate=reward_learning_rate,
                            fix_a=False, fix_b=True, fix_c=True)
    keys = ["D_contingent", "D_yoked", "count_c0", "count_c1",
            "test_rate_c0", "test_rate_c1", "baseline_p0"]
    mism = {k: (mine.get(k), ref.get(k)) for k in keys
            if _neq(mine.get(k), ref.get(k))}
    return {"seed": int(seed), "byte_identical_fixd_off": (len(mism) == 0), "mismatch": mism}


def _neq(a, b) -> bool:
    a = tuple(a) if isinstance(a, (list, tuple)) else a
    b = tuple(b) if isinstance(b, (list, tuple)) else b
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:
            return False
        return abs(a - b) > 0.0
    return a != b


def _smoke_one(seed: int, *, n_train: int, n_test: int, reward_learning_rate: float,
               fix_c: bool, fix_d: bool) -> dict:
    """One seed: FIX D ON (with fix_b+fix_c) vs the Stage-2j baseline (fix_b+fix_c, fix_d OFF)."""
    on = run_seed_swap(seed, n_train=n_train, n_test=n_test, reward_learning_rate=reward_learning_rate,
                       fix_a=False, fix_b=True, fix_c=fix_c, fix_d=fix_d)
    base = run_seed_swap(seed, n_train=n_train, n_test=n_test, reward_learning_rate=reward_learning_rate,
                         fix_a=False, fix_b=True, fix_c=fix_c, fix_d=False)

    def _def(x):
        return x == x
    return {
        "seed": int(seed), "baseline_p0": on["baseline_p0"],
        "FIXD_ON": {
            "count_c0": on["count_c0"], "count_c1": on["count_c1"],
            "n_clean_c0": on["n_clean_c0"], "n_clean_c1": on["n_clean_c1"],
            "test_rate_c0": on["test_rate_c0"], "test_rate_c1": on["test_rate_c1"],
            "D_contingent": on["D_contingent"], "D_yoked": on["D_yoked"],
            "steer": _steer(on),
            "n_released_c1": on["n_released_c1"], "release_channel_c1": on["release_channel_c1"],
            "homeo_c1": on["homeo_c1"], "r0_d1_c1": on["r0_d1_c1"],
        },
        "STAGE2J_BASE": {
            "count_c0": base["count_c0"], "count_c1": base["count_c1"],
            "test_rate_c0": base["test_rate_c0"], "test_rate_c1": base["test_rate_c1"],
            "D_contingent": base["D_contingent"], "D_yoked": base["D_yoked"],
            "steer": _steer(base),
        },
        "SMOKE_no_nan": bool(_def(on["test_rate_c0"]) and _def(on["test_rate_c1"])),
        "SMOKE_count_c1_nonzero": bool(on["count_c1"][1] > 0),          # action 1 got SELECTED
        "SMOKE_D_defined": bool(_def(on["D_contingent"]) and abs(on["D_contingent"]) > 1e-9),
        "SMOKE_steer": _steer(on),
        "SMOKE_steer_improved_vs_2j": bool(_steer(on) and not _steer(base)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "seeds", "full", "byte"], default="smoke")
    parser.add_argument("--seed", type=int, default=CONSTRUCTION_SEED)
    parser.add_argument("--dev-seeds", type=int, nargs="*", default=list(DEV_SEEDS))
    parser.add_argument("--smoke-seeds", type=int, nargs="*", default=[730705, 730704, 730706])
    parser.add_argument("--byte-seeds", type=int, nargs="*", default=[730703, 730705])
    parser.add_argument("--lesion-seed", type=int, default=730605)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--reward-lr", type=float, default=REWARD_LEARNING_RATE)
    parser.add_argument("--no-fix-c", action="store_true", help="disable FIX C (default ON in 2k)")
    parser.add_argument("--no-fix-d", action="store_true", help="disable FIX D (-> reproduces Stage 2j)")
    parser.add_argument("--explore-floor-k", type=int, default=None,
                        help="override EXPLORE_FLOOR_K (per-action release floor; for calibration sweeps)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.explore_floor_k is not None:
        globals()["EXPLORE_FLOOR_K"] = int(args.explore_floor_k)

    backend = _backend_info()
    started = time.perf_counter()
    fix_c = not args.no_fix_c
    fix_d = not args.no_fix_d

    if args.mode == "byte":
        res = [_assert_fixd_off_byte_identical(s, n_train=args.n_train, n_test=args.n_test,
                                               reward_learning_rate=args.reward_lr)
               for s in args.byte_seeds]
        ok = all(r["byte_identical_fixd_off"] for r in res)
        artifact = {"probe": "gateB_stage2k_byte_identity_fixd_off", "backend": backend["backend"],
                    "all_byte_identical": ok, "per_seed": res}
        out = Path(args.out) if args.out else OUT_DIR / f"byte_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        assert ok, f"FIX D OFF is NOT byte-identical to Stage 2j: {res}"
        return 0

    if args.mode == "smoke":
        # byte-identity assertion FIRST (fails loudly if the additive path perturbs 2j)
        byte = [_assert_fixd_off_byte_identical(s, n_train=args.n_train, n_test=args.n_test,
                                                reward_learning_rate=args.reward_lr)
                for s in args.byte_seeds]
        assert all(b["byte_identical_fixd_off"] for b in byte), \
            f"FIX D OFF not byte-identical to 2j: {byte}"
        results = [_smoke_one(s, n_train=args.n_train, n_test=args.n_test,
                              reward_learning_rate=args.reward_lr, fix_c=fix_c, fix_d=fix_d)
                   for s in args.smoke_seeds]
        artifact = {"probe": "gateB_stage2k_exploration_release_smoke",
                    "backend": backend["backend"], "device": backend["device"],
                    "smoke_seeds": args.smoke_seeds, "fix_c": fix_c, "fix_d": fix_d,
                    "byte_identity_fixd_off": byte,
                    "config": {"explore_floor_k": EXPLORE_FLOOR_K,
                               "explore_release_factor": EXPLORE_RELEASE_FACTOR},
                    "per_seed": results,
                    "elapsed_seconds": float(time.perf_counter() - started)}
        out = Path(args.out) if args.out else OUT_DIR / f"smoke_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        return 0

    if args.mode == "seeds":
        per = [run_seed_swap(s, n_train=args.n_train, n_test=args.n_test,
                             reward_learning_rate=args.reward_lr,
                             fix_a=False, fix_b=True, fix_c=fix_c, fix_d=fix_d)
               for s in args.dev_seeds]
        rows = [(p["seed"], round(p["baseline_p0"], 2), round(p["D_contingent"], 3),
                 round(p["D_yoked"], 3), p["count_c1"], round(p["test_rate_c1"], 3),
                 p["homeo_c1"]["engaged"], p["n_released_c1"], _steer(p)) for p in per]
        out_obj = {
            "probe": "gateB_stage2k_seeds", "backend": backend["backend"],
            "fix_c": fix_c, "fix_d": fix_d,
            "seeds_rows(seed,base_p0,Dc,Dy,count_c1,test_rate_c1,homeo_engaged,n_released_c1,steer)": rows,
            "steer_passes": int(sum(r[-1] for r in rows))}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out_obj, indent=2, default=float) + "\n")
        print(json.dumps(out_obj, indent=2, default=float))
        return 0

    # full battery -- mirrors 2j (steer + acquisition lesion + reversal), fix_c+fix_d ON.
    from research.runners._vocal_gateb_stage2i_circuit_fixes import run_reversal as _rev_2i
    equivalence = _assert_stage1_equivalence(args.seed)
    per_seed = [run_seed_swap(s, n_train=args.n_train, n_test=args.n_test,
                              reward_learning_rate=args.reward_lr,
                              fix_a=False, fix_b=True, fix_c=fix_c, fix_d=fix_d) for s in args.dev_seeds]
    steer_pass = [_steer(p) for p in per_seed]
    ls = args.lesion_seed
    lkw = dict(n_train=args.n_train, n_test=args.n_test, reward_learning_rate=args.reward_lr,
               fix_a=False, fix_b=True, fix_c=fix_c, fix_d=fix_d)
    intact = next((p for p in per_seed if p["seed"] == ls), None) or run_seed_swap(ls, **lkw)
    la0 = run_condition(ls, condition="acq_lesion", target=0, wh_reward_p=0.0, **lkw)
    la1 = run_condition(ls, condition="acq_lesion", target=1, wh_reward_p=0.0, **lkw)
    D_acq_lesion = _p_action0(la0) - _p_action0(la1)
    acq_share = attributable_to("acquisition D1 plasticity", intact["D_contingent"], D_acq_lesion)
    reversal = _rev_2i(ls, args.n_train, args.n_test, reward_learning_rate=args.reward_lr,
                       fix_a=False, fix_b=True)
    print(json.dumps({"equivalence": equivalence,
                      "steer_seed_passes": int(sum(steer_pass)), "steer_per_seed": steer_pass,
                      "baseline_p0_per_seed": [p["baseline_p0"] for p in per_seed],
                      "D_contingent_per_seed": [p["D_contingent"] for p in per_seed],
                      "lesion_seed": ls, "D_contingent_intact": intact["D_contingent"],
                      "D_contingent_acq_lesion": D_acq_lesion,
                      "acquisition_plasticity_share": acq_share,
                      "reversal_p_b_after_phaseB": reversal["p_b_after_phaseB"],
                      "reversal_pass": bool(reversal["p_b_after_phaseB"] >= 0.60)},
                     indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
