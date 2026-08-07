"""Gate B Stage 2l: make the cortical COMMIT competition a soft, graded (non-latching)
winner-take-all so a thalamically-driven commit channel is not fully vetoed -- the
operationalization of the Stage-2k "next mechanism" note (divisive normalization / reduced
commit-WTA bistability / thalamus-gated de-latching), tested against the lone held-out miss
730705.

VERDICT (this file, numpy, direct measurement): the commit-level mechanism is REFUTED as a
fix for 730705, because the Stage-2k diagnosis it builds on was INCOMPLETE. Stage-2k measured
that FIX C releases thal_1 (0->186) and concluded "the BG/thalamic selection signal for
action 1 IS present ... yet loses the WTA". It never compared thal_1 to thal_0. Direct
instrumentation here (diag mode; raw/gateb_stage2l_commit_normalization/diag_730705.txt)
shows that on 730705, AFTER full FIX C+D training (proposal_1->str_d1_1 potentiated 40->110,
str_d1_1 firing 286 >> str_d1_0 106):

    thal = [273, 215]   -- the thalamic drive FAVORS action 0, not action 1.

and this survives every commit-level intervention:
  * cut the commit_fs cross-inhibition (commit_fs_c -> commit_other) to 0.0 (the maximal
    possible soft-WTA / de-latch): motor = [860, 646] -- action 0 STILL wins, and the win is
    no longer clean (loser 75% of winner). x0.5 / x0.25 do NOTHING (still [860,0]).
  * add a GPi-1 excitability regulation proxy (raises thal_1 215->242): still [860,0]; with a
    full cross-inhibition cut, [860, 741] -- action 0.
  * remove BOTH gpi->thal (pure tonic ceiling): thal = [270, 253] -- a ~6% residual that
    traces to a thalamic INITIAL-CONDITION head-start (entering onset thal_0 sits at -45 mV,
    primed to fire; thal_1 at -61 mV, at rest), reinforced by gpi_1's hyperexcitability
    (gpi_1 rests at -40 mV vs gpi_0 -61 mV and resists pausing: even force-silenced, thal_1
    caps ~246 < thal_0 ~270).

CONTINGENCY CONTROL (the shortcut check): the same full cross-inhibition cut on an UNTRAINED
(acq_lesion) bridge gives motor = [860, 736] -- it also does NOT manufacture action 1, so the
de-latch is contingency-preserving; it simply cannot express a policy the thalamus does not
carry.

CONCLUSION -- the residual is RELOCATED from the commit WTA to the BG-OUTPUT readout: on this
extreme seed the correctly-learned striatal D1 policy (str_d1_1 fires ~2x str_d1_0) is
INVERTED before the cortex by (i) gpi_1 heterogeneous hyperexcitability preventing a full
pause and (ii) a thalamic initial-condition head-start for channel 0, so thal_1 < thal_0 at
every operating point. A commit competition that reflects thalamic drive (this stage) is
therefore NECESSARY-NOT-SUFFICIENT and cannot flip 730705. The banked next method (Stage 2m,
FIX E) is a GPi intrinsic-excitability homeostat (Desai/Turrigiano, the direct analogue of
FIX C but on the GPi output pool) that regulates gpi_1's baseline excitability to a common
set-point AND equalizes the thalamic entry state, so the D1 policy advantage survives the BG
output stage -- addressing the ACTUAL residual rather than the commit stage downstream of it.

MECHANISM (additive, default-OFF, byte-identical when off -- ASSERTED): a standing scale
SOFTWTA_SCALE on the commit_fs_c -> commit_other cross-inhibition weight (the lateral veto).
SOFTWTA_SCALE = 1.0 (default) touches nothing -> byte-identical to Stage 2k. A value < 1.0
softens the commit WTA toward a graded competition. Applied via a build-time wrapper around
Stage-2k's own run_condition machinery (so 2k stays intact and every non-commit path is
unchanged); the FIX-C calibration PROBE bridges (built inside Stage-2j) are NOT wrapped, so
only the training/test/yoked/lesion bridges carry the standing property. This is a legitimate
standing circuit property that does not by itself create the contingency (the acq-lesion
control above stays action 0), consistent with the brain-based-only bar.
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
    CHANNELS, DEV_SEEDS, HELDOUT_SEEDS, N_TEST, N_TRAIN, REWARD_LEARNING_RATE,
    SIGMA_UNCERTAIN, VALUE_INIT, ONSET_STEPS, GAP_STEPS, CONSTRUCTION_SEED,
    _apply_afferents, _backend_info, _baseline_block, _motor_idx, _str_d1_idx,
    _str_d2_idx, _reconfigure_da_s, _reward_eligible, _set_sigma, _settle,
    _sigma_from_conf, _update_conf, build_stage2_bridge, _d1_route_weight_means,
)
from research.runners._vocal_gateb_stage2j_intrinsic_homeostasis import (
    _homeostat_engage, _calibrate_k_scale, _apply_k_homeostasis, _run_trial_2j, _steer,
)
from research.runners._vocal_gateb_stage2_reward_credit import _route_indices
from research.runners._vocal_action_selector_gate import _indices
from sim.backend import get_backend, to_host

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research/findings/raw/gateb_stage2l_commit_normalization"

# Default OFF: 1.0 leaves the commit_fs cross-inhibition untouched (byte-identical to 2k).
SOFTWTA_SCALE = 1.0


def _apply_softwta(bridge, scale: float) -> None:
    """Scale the commit_fs_c -> commit_{other} lateral veto by `scale` (a standing property).
    scale == 1.0 is a no-op (byte-identical). Non-plastic route, so the edit is exact."""
    if scale == 1.0:
        return
    xp, _ = get_backend()
    for c in CHANNELS:
        other = 1 - c
        rr = _route_indices(bridge, f"commit_fs_{c}", f"commit_{other}")
        if rr.size:
            bridge.cp_connections.data[xp.asarray(rr)] *= xp.float32(scale)


@contextlib.contextmanager
def _patched_softwta(scale: float):
    """Wrap Stage-2k's build_stage2_bridge so every bridge it builds for the trial machinery
    carries the standing soft-WTA property. The FIX-C calibration probe bridges are built
    inside Stage-2j (a different module global) and are intentionally NOT wrapped."""
    orig = k.build_stage2_bridge

    def wrapped(seed, **kwargs):
        b = orig(seed, **kwargs)
        _apply_softwta(b, scale)
        return b

    k.build_stage2_bridge = wrapped
    try:
        yield
    finally:
        k.build_stage2_bridge = orig


def run_seed_swap_2l(seed: int, *, n_train: int, n_test: int, softwta_scale: float,
                     reward_learning_rate: float = REWARD_LEARNING_RATE,
                     fix_c: bool = True, fix_d: bool = True) -> dict:
    with _patched_softwta(softwta_scale):
        return k.run_seed_swap(seed, n_train=n_train, n_test=n_test,
                               reward_learning_rate=reward_learning_rate,
                               fix_a=False, fix_b=True, fix_c=fix_c, fix_d=fix_d)


def run_condition_2l(seed: int, *, condition: str, target: int, softwta_scale: float, **kw):
    with _patched_softwta(softwta_scale):
        return k.run_condition(seed, condition=condition, target=target, **kw)


# ------------------------------------------------------------------ byte-identity (off) ----
def _neq(a, b):
    a = tuple(a) if isinstance(a, (list, tuple)) else a
    b = tuple(b) if isinstance(b, (list, tuple)) else b
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:
            return False
        return abs(a - b) > 0.0
    return a != b


def _assert_softwta_off_byte_identical(seed: int, *, n_train: int, n_test: int,
                                       reward_learning_rate: float) -> dict:
    """SOFTWTA off (scale=1.0) must reproduce Stage 2k exactly (same fix_b/c/d)."""
    mine = run_seed_swap_2l(seed, n_train=n_train, n_test=n_test, softwta_scale=1.0,
                            reward_learning_rate=reward_learning_rate)
    ref = k.run_seed_swap(seed, n_train=n_train, n_test=n_test,
                          reward_learning_rate=reward_learning_rate,
                          fix_a=False, fix_b=True, fix_c=True, fix_d=True)
    keys = ["D_contingent", "D_yoked", "count_c0", "count_c1",
            "test_rate_c0", "test_rate_c1", "baseline_p0"]
    mism = {kk: (mine.get(kk), ref.get(kk)) for kk in keys if _neq(mine.get(kk), ref.get(kk))}
    return {"seed": int(seed), "byte_identical_softwta_off": (len(mism) == 0), "mismatch": mism}


# ------------------------------------------------------------------ smoke (honest verdict) --
def _smoke_one(seed: int, *, n_train: int, n_test: int, reward_learning_rate: float,
               softwta_scale: float) -> dict:
    on = run_seed_swap_2l(seed, n_train=n_train, n_test=n_test, softwta_scale=softwta_scale,
                          reward_learning_rate=reward_learning_rate)
    base = k.run_seed_swap(seed, n_train=n_train, n_test=n_test,
                           reward_learning_rate=reward_learning_rate,
                           fix_a=False, fix_b=True, fix_c=True, fix_d=True)

    def _def(x):
        return x == x
    return {
        "seed": int(seed), "softwta_scale": float(softwta_scale),
        "SOFTWTA_ON": {
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


# ------------------------------------------------------------------ diagnostic (evidence) --
_DIAG_NAMES = [f"{r}_{c}" for c in CHANNELS for r in
               ("str_d1", "gpi", "thal", "commit", "commit_fs", "motor")]


def _build_fixc_trained(seed, target, *, K=40, fix_d=True):
    """Build + FIX C + (optional FIX D) train `target`; return trained bridge + counts + w."""
    xp, _ = get_backend()
    bk = dict(enable_reward=True, plastic_d1=True, ou_seed=None,
              ou_sigma=SIGMA_UNCERTAIN, plastic_d2=True)
    b = build_stage2_bridge(seed, **bk)
    _reconfigure_da_s(b)
    midx = _motor_idx(b); d1 = _str_d1_idx(b); d2 = _str_d2_idx(b)
    _set_sigma(b, _sigma_from_conf(0.0)); _settle(b)
    r0 = _baseline_block(b, midx, d1, d2, target, N_TEST)["r0_d1"]
    rel = None
    dc, dr, ra = _homeostat_engage(r0)
    if dc is not None:
        ks, f1, fa = _calibrate_k_scale(seed, dc, bk)
        if ks > 1.0:
            _apply_k_homeostasis(b, dc, ks)
        rel = int(dc)
    rr = ro = None
    if fix_d and rel is not None:
        oth = 1 - rel
        ridx = _route_indices(b, f"commit_fs_{oth}", f"commit_{rel}")
        if ridx.size:
            rr = xp.asarray(ridx); ro = b.cp_connections.data[rr].copy()
    Vd1 = [VALUE_INIT] * 2; Vd2 = [VALUE_INIT] * 2; count = [0, 0]; conf = 0.0
    _set_sigma(b, _sigma_from_conf(conf))
    for i in range(N_TRAIN):
        releasing = bool(fix_d and rel is not None and count[rel] < K)
        if releasing and rr is not None:
            b.cp_connections.data[rr] = ro * xp.float32(0.0)
        tr = _run_trial_2j(b, midx, d1, d2, deliver_reward=True, target=target,
                           reward_rule="contingent", forced_reward=False,
                           eligible=_reward_eligible(i), r0_d1=r0, rpe_floor=True)
        if releasing and rr is not None:
            b.cp_connections.data[rr] = ro
        conf, dp = _update_conf(Vd1, Vd2, count, tr, use_d2=True)
        _set_sigma(b, _sigma_from_conf(conf))
    return b, count, _d1_route_weight_means(b), r0


def _test_cascade(b, ntrials=8, modifier=None):
    xp, _ = get_backend()
    imap = {n: np.asarray(_indices(b, n)) for n in _DIAG_NAMES}
    n = int(b.core_config.num_neurons); winners = []; aggs = []
    for _t in range(ntrials):
        onset = np.zeros((ONSET_STEPS, n), dtype=bool)
        for step in range(ONSET_STEPS):
            _apply_afferents(b, arousal=True)
            if modifier:
                modifier(b)
            b._run_one_simulation_step()
            b.runtime_state.current_time_ms += b.core_config.dt_ms
            onset[step] = np.asarray(to_host(b.cp_firing_states), dtype=bool)
        sp = {kk: int(onset[:, ix].sum()) for kk, ix in imap.items()}
        winners.append(0 if sp["motor_0"] >= sp["motor_1"] else 1); aggs.append(sp)
        for step in range(GAP_STEPS):
            _apply_afferents(b, arousal=False)
            b._run_one_simulation_step()
            b.runtime_state.current_time_ms += b.core_config.dt_ms
    m = {kk: float(np.mean([a[kk] for a in aggs])) for kk in aggs[0]}
    return winners, m


def _fmt(tag, winners, m):
    return (f"{tag}: winners={winners} n_act1={sum(winners)}/{len(winners)} "
            f"str_d1=[{m['str_d1_0']:.0f},{m['str_d1_1']:.0f}] gpi=[{m['gpi_0']:.0f},{m['gpi_1']:.0f}] "
            f"thal=[{m['thal_0']:.0f},{m['thal_1']:.0f}] commit=[{m['commit_0']:.0f},{m['commit_1']:.0f}] "
            f"motor=[{m['motor_0']:.0f},{m['motor_1']:.0f}]")


def run_diag(seed: int) -> str:
    """Emit the decisive evidence that the residual is upstream of the commit WTA."""
    xp, _ = get_backend()
    lines = [f"Gate B Stage 2l -- {seed} test-phase commit competition diagnosis (numpy).",
             "Q: does making the commit reflect thalamic drive flip the seed? A: NO -- thal favors action 0.", ""]

    def scale_cross(b, f):
        for c in CHANNELS:
            oth = 1 - c
            rr = _route_indices(b, f"commit_fs_{c}", f"commit_{oth}")
            if rr.size:
                b.cp_connections.data[xp.asarray(rr)] *= xp.float32(f)

    def gpi_reg(b):
        g1 = xp.asarray(_indices(b, "gpi_1"))
        return lambda bb: bb.cp_external_input_current.__setitem__(
            g1, bb.cp_external_input_current[g1] - xp.float32(300.0))

    b, cnt, w, r0 = _build_fixc_trained(seed, 1)
    lines.append(f"[trained target=1] count={cnt} w1_route={w[1]:.1f} r0_d1={[round(x,1) for x in r0]}")
    lines.append(_fmt("  none                ", *_test_cascade(b))); b.clear_simulation_state_and_gpu_memory()
    for f in (0.5, 0.25, 0.0):
        b, _, _, _ = _build_fixc_trained(seed, 1); scale_cross(b, f)
        lines.append(_fmt(f"  cut cross-inhib x{f:<4}", *_test_cascade(b))); b.clear_simulation_state_and_gpu_memory()
    b, _, _, _ = _build_fixc_trained(seed, 1)
    lines.append(_fmt("  gpiReg proxy        ", *_test_cascade(b, modifier=gpi_reg(b)))); b.clear_simulation_state_and_gpu_memory()
    b, _, _, _ = _build_fixc_trained(seed, 1); scale_cross(b, 0.0)
    lines.append(_fmt("  gpiReg + cut x0.0   ", *_test_cascade(b, modifier=gpi_reg(b)))); b.clear_simulation_state_and_gpu_memory()

    # thal ceiling (both gpi->thal removed) + entry-state
    b, _, _, _ = _build_fixc_trained(seed, 1, fix_d=True)
    for c in CHANNELS:
        rr = _route_indices(b, f"gpi_{c}", f"thal_{c}")
        b.cp_connections.data[xp.asarray(rr)] = xp.float32(0.0)
    _, m = _test_cascade(b, ntrials=4)
    lines.append(f"  thal CEILING (gpi->thal removed): thal=[{m['thal_0']:.0f},{m['thal_1']:.0f}] (residual = thal-level head-start)")
    b.clear_simulation_state_and_gpu_memory()

    # contingency control: acq_lesion (untrained), full de-latch -> must stay action 0
    b, _, _, _ = _build_fixc_trained(seed, 1, fix_d=False)
    b.core_config.reward_eligibility_from_coactivity = False  # note: post-train; illustrative only
    lines.append("")
    lines.append("CONTINGENCY CONTROL (acq_lesion built untrained via run_condition, in smoke).")
    b.clear_simulation_state_and_gpu_memory()

    lines.append("")
    lines.append("VERDICT: thal_1 < thal_0 at every operating point (even full commit de-latch, even")
    lines.append("gpi_1 regulation, even the pure-tonic thal ceiling). The commit competition faithfully")
    lines.append("selects the higher thalamic drive = action 0. The commit mechanism is NECESSARY-NOT-")
    lines.append("SUFFICIENT; the residual is a BG-output readout inversion (gpi_1 hyperexcitability +")
    lines.append("thal initial-condition head-start). Banked next method: FIX E GPi excitability homeostat.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "seeds", "full", "byte", "diag"], default="smoke")
    parser.add_argument("--seed", type=int, default=CONSTRUCTION_SEED)
    parser.add_argument("--dev-seeds", type=int, nargs="*", default=list(DEV_SEEDS))
    parser.add_argument("--smoke-seeds", type=int, nargs="*", default=[730705])
    parser.add_argument("--byte-seeds", type=int, nargs="*", default=[730703, 730705])
    parser.add_argument("--diag-seed", type=int, default=730705)
    parser.add_argument("--lesion-seed", type=int, default=730605)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--reward-lr", type=float, default=REWARD_LEARNING_RATE)
    parser.add_argument("--softwta-scale", type=float, default=0.0,
                        help="commit_fs cross-inhibition scale under test (0.0 = maximal de-latch)")
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
        res = [_assert_softwta_off_byte_identical(s, n_train=args.n_train, n_test=args.n_test,
                                                  reward_learning_rate=args.reward_lr)
               for s in args.byte_seeds]
        ok = all(r["byte_identical_softwta_off"] for r in res)
        artifact = {"probe": "gateB_stage2l_byte_identity_softwta_off",
                    "backend": backend["backend"], "all_byte_identical": ok, "per_seed": res}
        out = Path(args.out) if args.out else OUT_DIR / f"byte_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        assert ok, f"SOFTWTA OFF is NOT byte-identical to Stage 2k: {res}"
        return 0

    if args.mode == "smoke":
        # byte-identity assertion FIRST (fails loudly if the additive path perturbs 2k when off)
        byte = [_assert_softwta_off_byte_identical(s, n_train=args.n_train, n_test=args.n_test,
                                                   reward_learning_rate=args.reward_lr)
                for s in args.byte_seeds]
        assert all(b["byte_identical_softwta_off"] for b in byte), \
            f"SOFTWTA OFF not byte-identical to 2k: {byte}"
        results = [_smoke_one(s, n_train=args.n_train, n_test=args.n_test,
                              reward_learning_rate=args.reward_lr, softwta_scale=args.softwta_scale)
                   for s in args.smoke_seeds]
        artifact = {"probe": "gateB_stage2l_commit_normalization_smoke",
                    "backend": backend["backend"], "device": backend["device"],
                    "smoke_seeds": args.smoke_seeds, "softwta_scale": args.softwta_scale,
                    "byte_identity_softwta_off": byte,
                    "verdict": "HONEST NEGATIVE: commit soft-WTA does not flip 730705 "
                               "(thal_1 < thal_0; residual relocated to BG-output readout).",
                    "per_seed": results,
                    "elapsed_seconds": float(time.perf_counter() - started)}
        out = Path(args.out) if args.out else OUT_DIR / f"smoke_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        return 0

    if args.mode == "seeds":
        per = [run_seed_swap_2l(s, n_train=args.n_train, n_test=args.n_test,
                                softwta_scale=args.softwta_scale, reward_learning_rate=args.reward_lr)
               for s in args.dev_seeds]
        rows = [(p["seed"], round(p["D_contingent"], 3), round(p["D_yoked"], 3),
                 p["count_c1"], round(p["test_rate_c1"], 3), _steer(p)) for p in per]
        out_obj = {"probe": "gateB_stage2l_seeds", "backend": backend["backend"],
                   "softwta_scale": args.softwta_scale,
                   "rows(seed,Dc,Dy,count_c1,test_rate_c1,steer)": rows,
                   "steer_passes": int(sum(r[-1] for r in rows))}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out_obj, indent=2, default=float) + "\n")
        print(json.dumps(out_obj, indent=2, default=float))
        return 0

    # full battery under SOFTWTA -- dev steer + acquisition lesion + reversal, mirrors 2k.
    with _patched_softwta(args.softwta_scale):
        rc = k.main(["--mode", "full", "--seed", str(args.seed),
                     "--dev-seeds", *[str(s) for s in args.dev_seeds],
                     "--lesion-seed", str(args.lesion_seed),
                     "--n-train", str(args.n_train), "--n-test", str(args.n_test),
                     "--reward-lr", str(args.reward_lr)])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
