"""Gate B Stage 2m: FIX E -- an intrinsic-excitability homeostat at the BG-OUTPUT pools
(GPi + thalamic relay), the direct analogue of FIX C (Stage 2j MSN k-homeostat) moved
downstream to the stage where Stage 2l located the last held-out miss (730705). Desai 1999 /
Turrigiano 2011: activity-dependent intrinsic homeostasis via Na/K channel density, realised
as a scale on the Izhikevich gain `cp_izh_k` (NOT current injection -- current into these
pools was shown counterproductive in Stage 2l). Additive, DEFAULT-OFF, byte-identical when
off (ASSERTED). Authoritative backend = numpy.

WHAT FIX E DOES (target-blind, equalises baselines -- it must not manufacture the policy):
for each BG-output region in {gpi, thal}, measure BOTH channels' BASELINE onset firing on a
same-seed PROBE bridge (pre-training, target-agnostic; the training bridge's RNG is untouched)
and regulate each channel's intrinsic gain k toward the region's COMMON cross-channel set-point
(the two channels' mean) -- an over-active channel is scaled DOWN, an under-active channel UP.
The set-point is shared by both channels and the measurement is target-blind, so FIX E cannot
encode which action is rewarded. Engages only under an EXTREME BG-output asymmetry (gate), so
every non-engaging seed is byte-identical. Applied as a standing intrinsic property via a
build-time wrapper around Stage-2k's build_stage2_bridge (2k/2l stay intact).

VERDICT (this file, numpy, DIRECT measurement -- an HONEST NEGATIVE / RELOCATION, with a
concrete legitimacy-preserving closing stack). The standing FIX E smoke (fix_c on, fix_d off)
does NOT flip 730705 (test_rate_c1=0, count_c1=[40,0]). BUT FIX E refutes Stage 2l's headline:
it partially equalises the channel-0-open BG-output baseline (gpi [37,215]->[37,143],
thal [203,0]->[151,0]) and, on the FIX-D-trained bridge, INVERTS the thalamic AGGREGATE at
test (thal [273,215] -> [215,228], thal_1 > thal_0) -- so a target-blind intrinsic homeostat
at the BG output CAN invert the thalamic drive 2l called unfixable. Measured:
  * FIX E is NECESSARY-NOT-SUFFICIENT: with thal_1>thal_0 the motor winner still does NOT flip
    (commit=[388,0], action 0) -- the commit WTA integrates thal_0's TEMPORAL head-start
    (thal_0 fires first, commit_0 ignites and latches). Even the Stage-2l de-latch leaves
    commit=[388,359] (early head-start spikes bias the integrator).
  * The channel-0-open lock is set at BASELINE by the str_d1 firing asymmetry (str_d1_0 ~86,
    str_d1_1 ~0 -> gpi_0 paused, thal_0 primed to ~272). str_d1_0 baseline is NOT k-reducible
    (stays ~86 at k*0.1) -- consistent with 2j (k does not silence an already-firing MSN).
  * With FIX D OFF (the requested standing test) a second wall re-emerges: TRAINING-TIME
    EXPLORATION -- action 1 is never sampled (count_c1=[40,0]) so no policy forms to express;
    FIX E's standing symmetrisation does not overcome the lock during ordinary trials.
  * CLOSING STACK (probe-level, FIX-D-trained bridge, legitimacy-preserving): FIX E + Stage-2l
    commit de-latch + an onset entry-state equalisation (reset both gpi AND thal channels to a
    common membrane value at onset, a TRN-like selection-epoch reset) -> thal=[214,255],
    commit=[371,399], motor=[772,779] = 11/12 action 1; the same stack on an UNTRAINED bridge
    stays action 0 (motor [742,135]) so it does not manufacture the policy.
CONCLUSION: FIX E (BG-output intrinsic homeostat) is a genuine step -- it inverts the thalamic
drive -- but NECESSARY-NOT-SUFFICIENT standalone. Per the arc owner's standing instruction,
this honest relocation (with the closing-stack lead) is a legitimate checkpoint.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np

import research.runners._vocal_gateb_stage2k_exploration_release as k
from research.runners._vocal_gateb_stage2g_hammond_deltap import (
    CHANNELS, DEV_SEEDS, N_TEST, N_TRAIN, REWARD_LEARNING_RATE, SIGMA_UNCERTAIN,
    ONSET_STEPS, GAP_STEPS, CONSTRUCTION_SEED,
    _apply_afferents, _backend_info, _reconfigure_da_s, _set_sigma, _settle,
    _sigma_from_conf, build_stage2_bridge,
)
from research.runners._vocal_gateb_stage2j_intrinsic_homeostasis import _steer
from research.runners._vocal_gateb_stage2l_commit_normalization import (
    _build_fixc_trained, _test_cascade, _fmt,
)
from research.runners._vocal_gateb_stage2_reward_credit import _route_indices
from research.runners._vocal_action_selector_gate import _indices
from sim.backend import get_backend, to_host

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research/findings/raw/gateb_stage2m_bg_output_homeostat"

# ---- FIX E config (BG-output intrinsic-excitability homeostat) ----------------------------
FIXE_REGIONS = ("gpi", "thal")
FIXE_ASYM_RATIO = 5.0          # engage only under an extreme BG-output baseline asymmetry
FIXE_MIN_SETPOINT = 2.0        # ignore near-silent regions (avoid divide-by-noise)
FIXE_K_GRID_DOWN = (0.6, 0.4, 0.25, 0.15)   # scale an over-active channel's gain down
FIXE_K_GRID_UP = (1.5, 2.0, 2.5, 3.0)       # scale an under-active channel's gain up
FIXE_ON = False                # module-level default OFF (byte-identical to Stage 2k/2l)

_FIXE_CACHE: dict[int, dict] = {}


def _bg_baseline_fire(bridge, regions, steps: int = ONSET_STEPS) -> dict:
    """One target-blind onset run; accumulate per-region spike counts simultaneously."""
    idxs = {r: np.asarray(_indices(bridge, r)) for r in regions}
    tot = {r: 0 for r in regions}
    for _ in range(steps):
        _apply_afferents(bridge, arousal=True)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        fs = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        for r in regions:
            tot[r] += int(fs[idxs[r]].sum())
    return tot


def _measure_bg(seed: int, k_scales: dict | None = None) -> dict:
    """Baseline BG-output firing per channel on a canonical same-seed probe bridge (optionally
    with candidate k_scales applied), target-blind. Probe RNG never touches the training bridge."""
    xp, _ = get_backend()
    bk = dict(enable_reward=True, plastic_d1=True, ou_seed=None,
              ou_sigma=SIGMA_UNCERTAIN, plastic_d2=True)
    b = build_stage2_bridge(seed, **bk)
    _reconfigure_da_s(b)
    _set_sigma(b, _sigma_from_conf(0.0))
    _settle(b)
    if k_scales:
        for reg, sc in k_scales.items():
            if sc != 1.0:
                idx = xp.asarray(_indices(b, reg))
                b.cp_izh_k[idx] = b.cp_izh_k[idx] * xp.float32(sc)
    regions = [f"{r}_{c}" for r in FIXE_REGIONS for c in CHANNELS]
    fired = _bg_baseline_fire(b, regions)
    b.clear_simulation_state_and_gpu_memory()
    return fired


def _calibrate_fixe(seed: int) -> dict:
    """Per-seed target-blind homeostat calibration: for each BG-output region, if the two
    channels are extremely asymmetric, pick each channel's k-scale (from the DOWN grid for the
    over-active channel, UP grid for the under-active one) that brings its baseline firing
    closest to the region's cross-channel mean set-point. Returns {region_c: k_scale}."""
    if seed in _FIXE_CACHE:
        return _FIXE_CACHE[seed]
    base = _measure_bg(seed)
    scales: dict[str, float] = {}
    engaged = {}
    for r in FIXE_REGIONS:
        f0 = base[f"{r}_0"]
        f1 = base[f"{r}_1"]
        setpoint = 0.5 * (f0 + f1)
        hi = max(f0, f1)
        lo = max(min(f0, f1), 1e-6)
        if hi < FIXE_MIN_SETPOINT or (hi / lo) <= FIXE_ASYM_RATIO:
            engaged[r] = {"engaged": False, "f0": f0, "f1": f1, "setpoint": setpoint}
            continue
        engaged[r] = {"engaged": True, "f0": f0, "f1": f1, "setpoint": setpoint}
        for c in CHANNELS:
            fc = base[f"{r}_{c}"]
            grid = FIXE_K_GRID_DOWN if fc > setpoint else FIXE_K_GRID_UP
            best_ks, best_err = 1.0, abs(fc - setpoint)
            for ks in grid:
                probe = _measure_bg(seed, {f"{r}_{c}": ks})
                err = abs(probe[f"{r}_{c}"] - setpoint)
                if err < best_err:
                    best_ks, best_err = ks, err
            if best_ks != 1.0:
                scales[f"{r}_{c}"] = float(best_ks)
    out = {"scales": scales, "diag": engaged, "baseline": base}
    _FIXE_CACHE[seed] = out
    return out


def _apply_fixe(bridge, seed: int) -> dict:
    """Apply the per-seed FIX E k-scales to a freshly-built bridge (standing intrinsic
    property). Intrinsic only: touches no reward/DA signal and selects no action."""
    xp, _ = get_backend()
    cal = _calibrate_fixe(seed)
    for reg, sc in cal["scales"].items():
        idx = xp.asarray(_indices(bridge, reg))
        bridge.cp_izh_k[idx] = bridge.cp_izh_k[idx] * xp.float32(sc)
    return cal


@contextlib.contextmanager
def _patched_fixe(enabled: bool):
    """Wrap Stage-2k's build_stage2_bridge so every trial/test/lesion bridge carries the
    standing FIX E homeostat. enabled=False is a no-op -> byte-identical to Stage 2k/2l."""
    if not enabled:
        yield
        return
    orig = k.build_stage2_bridge

    def wrapped(seed, **kwargs):
        b = orig(seed, **kwargs)
        _apply_fixe(b, int(seed))
        return b

    k.build_stage2_bridge = wrapped
    try:
        yield
    finally:
        k.build_stage2_bridge = orig


def run_seed_swap_2m(seed: int, *, n_train: int, n_test: int, fix_e: bool,
                     reward_learning_rate: float = REWARD_LEARNING_RATE,
                     fix_c: bool = True, fix_d: bool = False) -> dict:
    with _patched_fixe(fix_e):
        return k.run_seed_swap(seed, n_train=n_train, n_test=n_test,
                               reward_learning_rate=reward_learning_rate,
                               fix_a=False, fix_b=True, fix_c=fix_c, fix_d=fix_d)


def run_condition_2m(seed: int, *, condition: str, target: int, fix_e: bool, **kw):
    with _patched_fixe(fix_e):
        return k.run_condition(seed, condition=condition, target=target,
                               fix_a=False, fix_b=True, fix_c=True, fix_d=False, **kw)


# ---------------------------------------------------------------- byte-identity (off) ------
def _neq(a, b):
    a = tuple(a) if isinstance(a, (list, tuple)) else a
    b = tuple(b) if isinstance(b, (list, tuple)) else b
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:
            return False
        return abs(a - b) > 0.0
    return a != b


def _assert_fixe_off_byte_identical(seed: int, *, n_train: int, n_test: int,
                                    reward_learning_rate: float) -> dict:
    """FIX E off must reproduce the Stage-2k base (fix_c on, fix_d off) exactly."""
    mine = run_seed_swap_2m(seed, n_train=n_train, n_test=n_test, fix_e=False,
                            reward_learning_rate=reward_learning_rate)
    ref = k.run_seed_swap(seed, n_train=n_train, n_test=n_test,
                          reward_learning_rate=reward_learning_rate,
                          fix_a=False, fix_b=True, fix_c=True, fix_d=False)
    keys = ["D_contingent", "D_yoked", "count_c0", "count_c1",
            "test_rate_c0", "test_rate_c1", "baseline_p0"]
    mism = {kk: (mine.get(kk), ref.get(kk)) for kk in keys if _neq(mine.get(kk), ref.get(kk))}
    return {"seed": int(seed), "byte_identical_fixe_off": (len(mism) == 0), "mismatch": mism}


# ---------------------------------------------------------------- smoke (honest verdict) ---
def _smoke_one(seed: int, *, n_train: int, n_test: int, reward_learning_rate: float) -> dict:
    on = run_seed_swap_2m(seed, n_train=n_train, n_test=n_test, fix_e=True,
                          reward_learning_rate=reward_learning_rate)
    base = k.run_seed_swap(seed, n_train=n_train, n_test=n_test,
                           reward_learning_rate=reward_learning_rate,
                           fix_a=False, fix_b=True, fix_c=True, fix_d=False)
    cal = _calibrate_fixe(seed)

    def _def(x):
        return x == x
    return {
        "seed": int(seed),
        "fixe_scales": cal["scales"], "fixe_diag": cal["diag"],
        "FIXE_ON": {
            "count_c1": on["count_c1"], "test_rate_c1": on["test_rate_c1"],
            "test_rate_c0": on["test_rate_c0"],
            "D_contingent": on["D_contingent"], "D_yoked": on["D_yoked"],
            "steer": _steer(on),
        },
        "STAGE2K_BASE": {
            "count_c1": base["count_c1"], "test_rate_c1": base["test_rate_c1"],
            "D_contingent": base["D_contingent"], "steer": _steer(base),
        },
        "SMOKE_no_nan": bool(_def(on["test_rate_c0"]) and _def(on["test_rate_c1"])),
        "SMOKE_730705_test_rate_c1_flips": bool(on["test_rate_c1"] > 0.0),   # the target
        "SMOKE_steer": _steer(on),
        "SMOKE_steer_improved_vs_2k": bool(_steer(on) and not _steer(base)),
    }


# ---------------------------------------------------------------- legitimacy (acq lesion) --
def _legitimacy_acq_lesion(seed: int, *, n_train: int, n_test: int,
                           reward_learning_rate: float) -> dict:
    """FIX E must NOT manufacture action 1 without the D1 learning. Build an UNTRAINED
    (acq_lesion) bridge WITH FIX E on and ask whether action 1 wins at test. If it does,
    FIX E is a shortcut. D_contingent must stay owned by acquisition D1 plasticity."""
    from research.runners._vocal_gateb_stage2g_hammond_deltap import _p_action0
    lkw = dict(n_train=n_train, n_test=n_test, reward_learning_rate=reward_learning_rate)
    la0 = run_condition_2m(seed, condition="acq_lesion", target=0, fix_e=True, **lkw)
    la1 = run_condition_2m(seed, condition="acq_lesion", target=1, fix_e=True, **lkw)
    D_acq_lesion = _p_action0(la0) - _p_action0(la1)
    # action 1 at test under acq_lesion+FIX E: must NOT win (p_action0 should stay high)
    p0_la1 = _p_action0(la1)
    return {"seed": int(seed), "D_contingent_acq_lesion_fixe_on": float(D_acq_lesion),
            "p_action0_target1_acq_lesion": float(p0_la1),
            "acq_lesion_action1_does_not_win": bool(p0_la1 >= 0.5),
            "n_clean_target1": int(la1["test_n_clean"]),
            "test_rate_target1": float(la1["test_target_rate"])}


# ---------------------------------------------------------------- diagnostic (evidence) ----
def run_diag(seed: int) -> str:
    xp, _ = get_backend()
    lines = [f"Gate B Stage 2m -- FIX E (BG-output intrinsic homeostat) on {seed} (numpy).",
             "Q: does an intrinsic-excitability homeostat at gpi/thal flip the seed? A: NO.", ""]

    # 1) target-blind BG-output baseline (the channel-0-open lock) + FIX E calibration
    cal = _calibrate_fixe(seed)
    b = cal["baseline"]
    lines.append(f"[baseline, target-blind] "
                 f"gpi=[{b['gpi_0']},{b['gpi_1']}] thal=[{b['thal_0']},{b['thal_1']}]")
    lines.append(f"[FIX E calibration] scales={cal['scales']}")
    for r in FIXE_REGIONS:
        d = cal["diag"][r]
        lines.append(f"  {r}: engaged={d['engaged']} f0={d['f0']} f1={d['f1']} "
                     f"setpoint={d['setpoint']:.1f}")
    # baseline AFTER applying FIX E scales (did it equalise?)
    post = _measure_bg(seed, cal["scales"]) if cal["scales"] else b
    lines.append(f"[baseline AFTER FIX E] "
                 f"gpi=[{post['gpi_0']},{post['gpi_1']}] thal=[{post['thal_0']},{post['thal_1']}]")
    lines.append("")

    # 2) trained-bridge test cascade with FIX E scales applied (does thal invert / motor flip?)
    def scale_regions(bridge, scales):
        for reg, sc in scales.items():
            idx = xp.asarray(_indices(bridge, reg))
            bridge.cp_izh_k[idx] = bridge.cp_izh_k[idx] * xp.float32(sc)

    bb, cnt, w, r0 = _build_fixc_trained(seed, 1)
    lines.append(f"[trained target=1] count={cnt} r0_d1={[round(x,1) for x in r0]}")
    lines.append(_fmt("  no FIX E            ", *_test_cascade(bb)))
    bb.clear_simulation_state_and_gpu_memory()
    bb, _, _, _ = _build_fixc_trained(seed, 1)
    scale_regions(bb, cal["scales"])
    lines.append(_fmt("  FIX E (k-homeostat) ", *_test_cascade(bb)))
    bb.clear_simulation_state_and_gpu_memory()

    lines.append("")
    lines.append("VERDICT: FIX E partially equalises the BG-output baseline and INVERTS the")
    lines.append("thalamic AGGREGATE at test (thal_0>thal_1 -> thal_1>thal_0, e.g. [273,215]->")
    lines.append("[215,228]) -- refuting Stage 2l's 'thal favors action 0, unfixable'. But the")
    lines.append("motor winner does NOT flip: the commit WTA integrates thal_0's TEMPORAL")
    lines.append("head-start (thal_0 fires first, commit_0 latches) so commit=[..,0], action 0.")
    lines.append("FIX E is NECESSARY-NOT-SUFFICIENT. The baseline head-start is set by the str_d1")
    lines.append("firing asymmetry (str_d1_0 ~86 vs str_d1_1 ~0), not intrinsic-k-reducible.")
    lines.append("HONEST NEGATIVE / RELOCATION (closing stack: FIX E + 2l de-latch + onset")
    lines.append("entry-state equalisation -> 11/12 action 1, legitimacy-preserving; see finding).")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "seeds", "full", "byte", "diag", "legit"],
                        default="smoke")
    parser.add_argument("--seed", type=int, default=CONSTRUCTION_SEED)
    parser.add_argument("--dev-seeds", type=int, nargs="*", default=list(DEV_SEEDS))
    parser.add_argument("--smoke-seeds", type=int, nargs="*", default=[730705])
    parser.add_argument("--byte-seeds", type=int, nargs="*", default=[730703, 730705])
    parser.add_argument("--diag-seed", type=int, default=730705)
    parser.add_argument("--legit-seed", type=int, default=730705)
    parser.add_argument("--lesion-seed", type=int, default=730605)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--reward-lr", type=float, default=REWARD_LEARNING_RATE)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    backend = _backend_info()
    started = time.perf_counter()

    if args.mode == "diag":
        txt = run_diag(args.diag_seed)
        out = Path(args.out) if args.out else OUT_DIR / f"diag_{args.diag_seed}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt)
        print(txt)
        return 0

    if args.mode == "byte":
        res = [_assert_fixe_off_byte_identical(s, n_train=args.n_train, n_test=args.n_test,
                                               reward_learning_rate=args.reward_lr)
               for s in args.byte_seeds]
        ok = all(r["byte_identical_fixe_off"] for r in res)
        artifact = {"probe": "gateB_stage2m_byte_identity_fixe_off",
                    "backend": backend["backend"], "all_byte_identical": ok, "per_seed": res}
        out = Path(args.out) if args.out else OUT_DIR / f"byte_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        assert ok, f"FIX E OFF is NOT byte-identical to Stage 2k base: {res}"
        return 0

    if args.mode == "legit":
        res = _legitimacy_acq_lesion(args.legit_seed, n_train=args.n_train, n_test=args.n_test,
                                     reward_learning_rate=args.reward_lr)
        out = Path(args.out) if args.out else OUT_DIR / f"legit_{args.legit_seed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2, default=float) + "\n")
        print(json.dumps(res, indent=2, default=float))
        return 0

    if args.mode == "smoke":
        # byte-identity assertion FIRST (fails loudly if the additive path perturbs base when off)
        byte = [_assert_fixe_off_byte_identical(s, n_train=args.n_train, n_test=args.n_test,
                                                reward_learning_rate=args.reward_lr)
                for s in args.byte_seeds]
        assert all(b["byte_identical_fixe_off"] for b in byte), \
            f"FIX E OFF not byte-identical to 2k base: {byte}"
        results = [_smoke_one(s, n_train=args.n_train, n_test=args.n_test,
                              reward_learning_rate=args.reward_lr)
                   for s in args.smoke_seeds]
        legit = _legitimacy_acq_lesion(args.legit_seed, n_train=args.n_train, n_test=args.n_test,
                                       reward_learning_rate=args.reward_lr)
        artifact = {"probe": "gateB_stage2m_bg_output_homeostat_smoke",
                    "backend": backend["backend"], "device": backend["device"],
                    "smoke_seeds": args.smoke_seeds,
                    "byte_identity_fixe_off": byte,
                    "legitimacy_acq_lesion": legit,
                    "config": {"fixe_asym_ratio": FIXE_ASYM_RATIO,
                               "fixe_k_grid_down": list(FIXE_K_GRID_DOWN),
                               "fixe_k_grid_up": list(FIXE_K_GRID_UP)},
                    "verdict": "HONEST NEGATIVE / RELOCATION: FIX E inverts the thalamic "
                               "aggregate at test (refuting 2l) but is necessary-not-sufficient "
                               "standalone -- the commit integrates thal_0's temporal head-start, "
                               "and with fix_d off the exploration wall re-emerges (count_c1=[40,0]). "
                               "Closing stack (probe): FIX E + 2l de-latch + onset entry-eq -> 11/12.",
                    "per_seed": results,
                    "elapsed_seconds": float(time.perf_counter() - started)}
        out = Path(args.out) if args.out else OUT_DIR / f"smoke_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        return 0

    if args.mode == "seeds":
        per = [run_seed_swap_2m(s, n_train=args.n_train, n_test=args.n_test, fix_e=True,
                                reward_learning_rate=args.reward_lr)
               for s in args.dev_seeds]
        rows = [(p["seed"], round(p["D_contingent"], 3), round(p["D_yoked"], 3),
                 p["count_c1"], round(p["test_rate_c1"], 3), _steer(p)) for p in per]
        out_obj = {"probe": "gateB_stage2m_seeds", "backend": backend["backend"],
                   "rows(seed,Dc,Dy,count_c1,test_rate_c1,steer)": rows,
                   "steer_passes": int(sum(r[-1] for r in rows))}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out_obj, indent=2, default=float) + "\n")
        print(json.dumps(out_obj, indent=2, default=float))
        return 0

    # full battery under FIX E -- dev steer + acquisition lesion + reversal, mirrors 2k.
    with _patched_fixe(True):
        rc = k.main(["--mode", "full", "--seed", str(args.seed),
                     "--dev-seeds", *[str(s) for s in args.dev_seeds],
                     "--lesion-seed", str(args.lesion_seed),
                     "--n-train", str(args.n_train), "--n-test", str(args.n_test),
                     "--reward-lr", str(args.reward_lr)])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
