"""Gate B Stage 2n: FIX F -- port the RECORD-GROUNDED accumulate-then-commit readout
(2026-06-06-N6-accumulator-commit-readout-BOUNDARY / -deep-research) onto the vocal Gate B
COMMIT stage, to close the lone held-out miss 730705.

WHY (verified 2026-08-07, Stage 2l/2m): 730705's D1 policy is CORRECTLY LEARNED after FIX C+D
training (str_d1_1 ~286 >> str_d1_0 ~106 -- a STRONG correct signal, unlike the WEAK nav signal
that made N6 a boundary). But at test the cortical commit WTA LATCHES on thal_0's TRANSIENT
temporal head-start (thal_0 enters onset primed to ~272 and fires FIRST, so commit_0 ignites and
latches) instead of INTEGRATING the SUSTAINED correct drive. Stage 2l cut the commit_fs
cross-inhibition (de-latch) and it did NOT flip; Stage 2m's BG-output homeostat (FIX E) INVERTS
the thalamic aggregate (thal [273,215] -> [215,228]) but is necessary-not-sufficient because the
commit still integrates the temporal head-start. The un-tried, appropriately-scoped mechanism is
the N6 recipe's missing ingredient: NMDA-SLOW recurrent self-excitation on the commit pools so
the commit ACCUMULATES the sustained drive (Wang 2002 tau~100ms) rather than latching on the
transient onset -- combined with FIX E (which makes the sustained thal_1 aggregate exceed thal_0
so the accumulator integrates the CORRECT channel to the bound).

THE MECHANISM (FIX F, additive, DEFAULT-OFF, byte-identical when off -- ASSERTED; NO sim/ edit):
The Gate B commit pools are built with recurrent self-excitation already
(internal_density=COMMIT_INTERNAL_DENSITY, internal_weight=0.5) but enable_nmda=False -- i.e. the
recurrence is AMPA-FAST (the degenerate, un-integrating case the N6 deep-research doc names).
FIX F turns that recurrence NMDA-SLOW, restricted to the commit pools only, exactly as the nav N6
readout did (global cfg.enable_nmda=True + a per-neuron cp_nmda_neuron_mask over the commit
neurons; NMDA tau_decay=100ms, Wang 2002). It also softens the commit competition per the recipe:
GENTLE (lowered) commit_fs cross-inhibition + (optionally) lowered thal->commit feedforward
weight (evidence, not saturation). Applied as a standing SYNAPTIC/receptor property via a
build-time wrapper around Stage-2k's build_stage2_bridge (2k/2l/2m stay intact). FIX F OFF is a
no-op wrapper -> byte-identical. Authoritative backend = numpy (numpy always runs the
NMDA-capable Python step and reads cfg.enable_nmda live, so post-build enabling is exact).

ANTI-CHEATS (decisive):
  (1) INTEGRATION not a persistent latch/rewire: with FIX F OFF the seed still fails (asserted
      byte-identical to 2k); the acquisition-LESION (untrained, no D1 learning) with FIX F ON must
      still NOT pick action 1 (contingency stays owned by D1 plasticity).
  (2) byte-identical when off (ASSERTED before the smoke).
  (3) recall/test uses no privileged signal -- FIX F is target-blind (a receptor kind + two weight
      scales), applied identically at train and test; FIX E is target-blind (Stage 2m).

VERDICT (numpy, full train->test smoke -- outcome (ii): NO-GO, a precisely-located SHORTCUT).
730705's action 1 DOES express at test (test_rate_c1 0->1.0) and the NMDA integration genuinely
overcomes thal_0's temporal head-start (commit [184,504] while the thal aggregate still favors
action 0). BUT the gentle cross-inhibition required to let commit_1 ENTER (xinhib x0.1; x0.25+
never fires) also unmasks FIX C's target-blind intrinsic gain: on an UNTRAINED (acq-lesion) bridge
action 1 wins anyway (test_rate_target1=1.0, acq_lesion_action1_does_not_win=false) and the target-0
contingency breaks (test_rate_c0 1.0->0.1, steer=false). The Gate B commit veto was doing
legitimate work -- it enforced that only a LEARNED str_d1 overcomes the head-start; the de-latch
sacrifices that. Byte-identical when off (ASSERTED). 730705 remains a characterized heterogeneity
boundary; Gate B holds at >=5/6. See
research/findings/2026-08-07-gateB-stage2n-accumulate-then-commit-NMDA-integration-closes-730705.md.
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
from research.runners._vocal_gateb_stage2m_bg_output_homeostat import (
    _apply_fixe, _calibrate_fixe,
)
from research.runners._vocal_gateb_stage2_reward_credit import _route_indices
from research.runners._vocal_action_selector_gate import _indices
from sim.backend import get_backend, to_host

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research/findings/raw/gateb_stage2n_accumulate_commit"

# ---- FIX F config (accumulate-then-commit on the vocal commit stage) ----------------------
# Values grounded in the N6 nav readout recipe (2026-06-06): NMDA-slow recurrence
# (nmda_ratio 0.5), lowered feedforward (evidence not saturation), gentle cross-inhibition.
# Held FIXED (not swept per-seed) -- FIX E already carries the per-seed target-blind adaptation.
FIXF_NMDA_RATIO = 0.5          # NMDA:AMPA ratio on the commit pools (Wang 2002 slow reverberation)
FIXF_XINHIB_SCALE = 0.1        # GENTLE commit_fs cross-inhibition (soft-WTA -- lets the loser ENTER
                               # so the NMDA accumulator can integrate it; per N6 recipe's gentle veto)
FIXF_FF_SCALE = 1.0            # thal->commit feedforward scale (1.0 = untouched; <1 = evidence-not-saturation)
FIXF_ON = False                # module-level default OFF (byte-identical to Stage 2k/2l/2m base)


def _fixe_engaged(seed: int) -> bool:
    """FIX F recruitment gate = the SAME target-blind extreme-BG-output-asymmetry detector FIX E
    uses (Stage 2m). On seeds where the gate does not fire, FIX F is a no-op -> byte-identical, so
    the existing >=5/6 well-behaved seeds are untouched (the FIX D<-FIX C recruitment discipline)."""
    cal = _calibrate_fixe(int(seed))
    return any(d.get("engaged") for d in cal["diag"].values())


def _commit_indices(bridge):
    xp, _ = get_backend()
    idx = np.concatenate([_indices(bridge, f"commit_{c}") for c in CHANNELS])
    return xp.asarray(idx)


def _apply_accumulate_commit(bridge, *, nmda_ratio: float, xinhib_scale: float,
                             ff_scale: float) -> dict:
    """Turn the commit pools' (already-present) recurrent self-excitation NMDA-SLOW, restricted
    to the commit neurons via cp_nmda_neuron_mask (the SAME mechanism the nav N6 readout used),
    and soften the commit competition. Standing, target-blind, selects no action. Returns diag."""
    xp, _ = get_backend()
    cfg = bridge.core_config
    # ACCUMULATE: NMDA-slow integration on the commit pools only.
    cfg.enable_nmda = True
    cfg.nmda_ratio = float(nmda_ratio)
    n = int(cfg.num_neurons)
    mask = xp.zeros(n, dtype=xp.float32)
    cidx = _commit_indices(bridge)
    mask[cidx] = xp.float32(1.0)
    bridge.cp_nmda_neuron_mask = mask
    # COMMIT: gentle cross-inhibition (soft-WTA, per recipe) ...
    if xinhib_scale != 1.0:
        for c in CHANNELS:
            other = 1 - c
            rr = _route_indices(bridge, f"commit_fs_{c}", f"commit_{other}")
            if rr.size:
                bridge.cp_connections.data[xp.asarray(rr)] *= xp.float32(xinhib_scale)
    # ... and (optionally) lowered thal->commit feedforward (evidence, not saturation).
    if ff_scale != 1.0:
        for c in CHANNELS:
            rr = _route_indices(bridge, f"thal_{c}", f"commit_{c}")
            if rr.size:
                bridge.cp_connections.data[xp.asarray(rr)] *= xp.float32(ff_scale)
    return {"nmda_ratio": float(nmda_ratio), "xinhib_scale": float(xinhib_scale),
            "ff_scale": float(ff_scale), "n_commit_masked": int(cidx.size)}


@contextlib.contextmanager
def _patched_accum(accum_on: bool, fix_e: bool,
                   nmda_ratio: float, xinhib_scale: float, ff_scale: float,
                   force_engage: bool = False):
    """Wrap Stage-2k's build_stage2_bridge so every trial/test/lesion bridge carries the standing
    FIX F accumulate-commit property (and, if fix_e, the Stage-2m BG-output homeostat's k-scales
    too). FIX F engages ONLY where the (target-blind) FIX-E extreme-asymmetry gate fires -- so on
    the well-behaved seeds it is a no-op -> byte-identical. accum_on=False -> no-op wrapper.
    force_engage bypasses the gate (isolation tests only)."""
    if not accum_on:
        yield
        return
    orig = k.build_stage2_bridge

    def wrapped(seed, **kwargs):
        b = orig(seed, **kwargs)
        if not (force_engage or _fixe_engaged(int(seed))):
            return b  # recruitment gate did not fire -> byte-identical
        if fix_e:
            _apply_fixe(b, int(seed))
        _apply_accumulate_commit(b, nmda_ratio=nmda_ratio,
                                 xinhib_scale=xinhib_scale, ff_scale=ff_scale)
        return b

    k.build_stage2_bridge = wrapped
    try:
        yield
    finally:
        k.build_stage2_bridge = orig


def run_seed_swap_2n(seed: int, *, n_train: int, n_test: int, accum_on: bool, fix_e: bool,
                     reward_learning_rate: float = REWARD_LEARNING_RATE,
                     nmda_ratio: float = FIXF_NMDA_RATIO, xinhib_scale: float = FIXF_XINHIB_SCALE,
                     ff_scale: float = FIXF_FF_SCALE, fix_c: bool = True, fix_d: bool = True) -> dict:
    with _patched_accum(accum_on, fix_e, nmda_ratio, xinhib_scale, ff_scale):
        return k.run_seed_swap(seed, n_train=n_train, n_test=n_test,
                               reward_learning_rate=reward_learning_rate,
                               fix_a=False, fix_b=True, fix_c=fix_c, fix_d=fix_d)


def run_condition_2n(seed: int, *, condition: str, target: int, accum_on: bool, fix_e: bool,
                     nmda_ratio: float = FIXF_NMDA_RATIO, xinhib_scale: float = FIXF_XINHIB_SCALE,
                     ff_scale: float = FIXF_FF_SCALE, **kw):
    with _patched_accum(accum_on, fix_e, nmda_ratio, xinhib_scale, ff_scale):
        return k.run_condition(seed, condition=condition, target=target,
                               fix_a=False, fix_b=True, fix_c=True, fix_d=True, **kw)


# ---------------------------------------------------------------- byte-identity (off) ------
def _neq(a, b):
    a = tuple(a) if isinstance(a, (list, tuple)) else a
    b = tuple(b) if isinstance(b, (list, tuple)) else b
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:
            return False
        return abs(a - b) > 0.0
    return a != b


def _assert_fixf_off_byte_identical(seed: int, *, n_train: int, n_test: int,
                                    reward_learning_rate: float) -> dict:
    """FIX F off (accum_on=False, fix_e=False) must reproduce the Stage-2k base (fix_c+fix_d)."""
    mine = run_seed_swap_2n(seed, n_train=n_train, n_test=n_test, accum_on=False, fix_e=False,
                            reward_learning_rate=reward_learning_rate)
    ref = k.run_seed_swap(seed, n_train=n_train, n_test=n_test,
                          reward_learning_rate=reward_learning_rate,
                          fix_a=False, fix_b=True, fix_c=True, fix_d=True)
    keys = ["D_contingent", "D_yoked", "count_c0", "count_c1",
            "test_rate_c0", "test_rate_c1", "baseline_p0"]
    mism = {kk: (mine.get(kk), ref.get(kk)) for kk in keys if _neq(mine.get(kk), ref.get(kk))}
    return {"seed": int(seed), "byte_identical_fixf_off": (len(mism) == 0), "mismatch": mism}


# ---------------------------------------------------------------- legitimacy (acq lesion) --
def _legitimacy_acq_lesion(seed: int, *, n_train: int, n_test: int, reward_learning_rate: float,
                           fix_e: bool) -> dict:
    """FIX F (+FIX E) must NOT manufacture action 1 without D1 learning. Untrained (acq_lesion)
    bridge WITH the accumulator ON: action 1 must NOT win at test (p_action0 stays high)."""
    from research.runners._vocal_gateb_stage2g_hammond_deltap import _p_action0
    lkw = dict(n_train=n_train, n_test=n_test, reward_learning_rate=reward_learning_rate)
    la0 = run_condition_2n(seed, condition="acq_lesion", target=0, accum_on=True, fix_e=fix_e, **lkw)
    la1 = run_condition_2n(seed, condition="acq_lesion", target=1, accum_on=True, fix_e=fix_e, **lkw)
    D_acq_lesion = _p_action0(la0) - _p_action0(la1)
    p0_la1 = _p_action0(la1)
    return {"seed": int(seed), "D_contingent_acq_lesion_fixf_on": float(D_acq_lesion),
            "p_action0_target1_acq_lesion": float(p0_la1),
            "acq_lesion_action1_does_not_win": bool(p0_la1 >= 0.5),
            "n_clean_target1": int(la1["test_n_clean"]),
            "test_rate_target1": float(la1["test_target_rate"])}


# ---------------------------------------------------------------- smoke (honest verdict) ---
def _smoke_one(seed: int, *, n_train: int, n_test: int, reward_learning_rate: float,
               fix_e: bool, nmda_ratio: float, xinhib_scale: float, ff_scale: float) -> dict:
    on = run_seed_swap_2n(seed, n_train=n_train, n_test=n_test, accum_on=True, fix_e=fix_e,
                          reward_learning_rate=reward_learning_rate, nmda_ratio=nmda_ratio,
                          xinhib_scale=xinhib_scale, ff_scale=ff_scale)
    base = k.run_seed_swap(seed, n_train=n_train, n_test=n_test,
                           reward_learning_rate=reward_learning_rate,
                           fix_a=False, fix_b=True, fix_c=True, fix_d=True)

    def _def(x):
        return x == x
    return {
        "seed": int(seed), "fix_e": bool(fix_e),
        "FIXF_ON": {
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


# ---------------------------------------------------------------- diagnostic (evidence) ----
def _cascade_accum(b, ntrials=8, nmda_ratio=FIXF_NMDA_RATIO,
                   xinhib_scale=FIXF_XINHIB_SCALE, ff_scale=FIXF_FF_SCALE, fix_e=False,
                   seed=None):
    """Apply FIX E (optional) + FIX F to an already-trained bridge, then run the test cascade
    with per-step commit accumulation tracing (proof of INTEGRATION vs latch)."""
    xp, _ = get_backend()
    if fix_e and seed is not None:
        _apply_fixe(b, int(seed))
    _apply_accumulate_commit(b, nmda_ratio=nmda_ratio, xinhib_scale=xinhib_scale, ff_scale=ff_scale)
    # commit indices for the per-step ramp trace
    c0i = np.asarray(_indices(b, "commit_0")); c1i = np.asarray(_indices(b, "commit_1"))
    ramp0 = np.zeros(ONSET_STEPS); ramp1 = np.zeros(ONSET_STEPS)
    winners, aggs = _test_cascade(b, ntrials=ntrials)
    # one extra instrumented trial for the ramp (fresh onset)
    n = int(b.core_config.num_neurons)
    for step in range(ONSET_STEPS):
        _apply_afferents(b, arousal=True)
        b._run_one_simulation_step()
        b.runtime_state.current_time_ms += b.core_config.dt_ms
        fs = np.asarray(to_host(b.cp_firing_states), dtype=bool)
        ramp0[step] = fs[c0i].sum(); ramp1[step] = fs[c1i].sum()
    cum0 = np.cumsum(ramp0); cum1 = np.cumsum(ramp1)
    return winners, aggs, cum0, cum1


def run_diag(seed: int, *, fix_e: bool, nmda_ratio: float, xinhib_scale: float,
             ff_scale: float) -> str:
    lines = [f"Gate B Stage 2n -- FIX F accumulate-then-commit on {seed} (numpy).",
             f"config: nmda_ratio={nmda_ratio} xinhib_scale={xinhib_scale} ff_scale={ff_scale} "
             f"fix_e={fix_e}",
             "Q: does NMDA-slow commit integration overtake thal_0's temporal head-start? ", ""]

    # baseline (no FIX F): the latch on the transient head-start
    bb, cnt, w, r0 = _build_fixc_trained(seed, 1)
    lines.append(f"[trained target=1] count={cnt} r0_d1={[round(x,1) for x in r0]}")
    lines.append(_fmt("  base (no FIX F)     ", *_test_cascade(bb)))
    bb.clear_simulation_state_and_gpu_memory()

    # FIX F only (accumulate-commit), no FIX E
    bb, _, _, _ = _build_fixc_trained(seed, 1)
    winners, aggs, cum0, cum1 = _cascade_accum(bb, nmda_ratio=nmda_ratio,
                                               xinhib_scale=xinhib_scale, ff_scale=ff_scale,
                                               fix_e=False, seed=seed)
    lines.append(_fmt("  FIX F only          ", winners, aggs))
    bb.clear_simulation_state_and_gpu_memory()

    # FIX E + FIX F (the intended combination)
    bb, _, _, _ = _build_fixc_trained(seed, 1)
    winners, aggs, cum0, cum1 = _cascade_accum(bb, nmda_ratio=nmda_ratio,
                                               xinhib_scale=xinhib_scale, ff_scale=ff_scale,
                                               fix_e=True, seed=seed)
    lines.append(_fmt("  FIX E + FIX F       ", winners, aggs))
    # ramp trace (proof of integration): commit cumulative spikes across the onset window
    marks = [4, 9, 14, 19, ONSET_STEPS - 1]
    marks = [m for m in marks if m < ONSET_STEPS]
    lines.append("  [commit cumulative-spike ramp, FIX E+FIX F] "
                 + " ".join(f"t{m}:c0={cum0[m]:.0f}/c1={cum1[m]:.0f}" for m in marks))
    bb.clear_simulation_state_and_gpu_memory()

    lines.append("")
    lines.append("READ: if commit_1 cumulative overtakes commit_0 across the window (c1>c0 at the")
    lines.append("late marks) while base latches commit_0, the win is INTEGRATION of the sustained")
    lines.append("correct drive over the transient head-start (the N6 accumulate-then-commit).")
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
    parser.add_argument("--fix-e", action="store_true", default=False,
                        help="ALSO apply the Stage-2m BG-output homeostat k-scales (aggregate "
                             "inversion). Default OFF: the NMDA accumulator integrates the "
                             "sustained-correct drive over the transient head-start WITHOUT needing "
                             "the aggregate inverted (the cleanest integration-is-load-bearing test).")
    parser.add_argument("--nmda-ratio", type=float, default=FIXF_NMDA_RATIO)
    parser.add_argument("--xinhib-scale", type=float, default=FIXF_XINHIB_SCALE)
    parser.add_argument("--ff-scale", type=float, default=FIXF_FF_SCALE)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    backend = _backend_info()
    started = time.perf_counter()

    if args.mode == "diag":
        txt = run_diag(args.diag_seed, fix_e=args.fix_e, nmda_ratio=args.nmda_ratio,
                       xinhib_scale=args.xinhib_scale, ff_scale=args.ff_scale)
        out = Path(args.out) if args.out else OUT_DIR / f"diag_{args.diag_seed}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt)
        print(txt)
        return 0

    if args.mode == "byte":
        res = [_assert_fixf_off_byte_identical(s, n_train=args.n_train, n_test=args.n_test,
                                               reward_learning_rate=args.reward_lr)
               for s in args.byte_seeds]
        ok = all(r["byte_identical_fixf_off"] for r in res)
        artifact = {"probe": "gateB_stage2n_byte_identity_fixf_off",
                    "backend": backend["backend"], "all_byte_identical": ok, "per_seed": res}
        out = Path(args.out) if args.out else OUT_DIR / f"byte_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        assert ok, f"FIX F OFF is NOT byte-identical to Stage 2k base: {res}"
        return 0

    if args.mode == "legit":
        res = _legitimacy_acq_lesion(args.legit_seed, n_train=args.n_train, n_test=args.n_test,
                                     reward_learning_rate=args.reward_lr, fix_e=args.fix_e)
        out = Path(args.out) if args.out else OUT_DIR / f"legit_{args.legit_seed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2, default=float) + "\n")
        print(json.dumps(res, indent=2, default=float))
        return 0

    if args.mode == "smoke":
        # byte-identity assertion FIRST (fails loudly if the additive path perturbs base when off)
        byte = [_assert_fixf_off_byte_identical(s, n_train=args.n_train, n_test=args.n_test,
                                                reward_learning_rate=args.reward_lr)
                for s in args.byte_seeds]
        assert all(b["byte_identical_fixf_off"] for b in byte), \
            f"FIX F OFF not byte-identical to 2k base: {byte}"
        results = [_smoke_one(s, n_train=args.n_train, n_test=args.n_test,
                              reward_learning_rate=args.reward_lr, fix_e=args.fix_e,
                              nmda_ratio=args.nmda_ratio, xinhib_scale=args.xinhib_scale,
                              ff_scale=args.ff_scale)
                   for s in args.smoke_seeds]
        legit = _legitimacy_acq_lesion(args.legit_seed, n_train=args.n_train, n_test=args.n_test,
                                       reward_learning_rate=args.reward_lr, fix_e=args.fix_e)
        artifact = {"probe": "gateB_stage2n_accumulate_commit_smoke",
                    "backend": backend["backend"], "device": backend["device"],
                    "smoke_seeds": args.smoke_seeds, "fix_e": bool(args.fix_e),
                    "byte_identity_fixf_off": byte,
                    "legitimacy_acq_lesion": legit,
                    "config": {"nmda_ratio": args.nmda_ratio, "xinhib_scale": args.xinhib_scale,
                               "ff_scale": args.ff_scale, "fix_e": bool(args.fix_e)},
                    "per_seed": results,
                    "elapsed_seconds": float(time.perf_counter() - started)}
        out = Path(args.out) if args.out else OUT_DIR / f"smoke_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        return 0

    if args.mode == "seeds":
        per = [run_seed_swap_2n(s, n_train=args.n_train, n_test=args.n_test, accum_on=True,
                                fix_e=args.fix_e, reward_learning_rate=args.reward_lr,
                                nmda_ratio=args.nmda_ratio, xinhib_scale=args.xinhib_scale,
                                ff_scale=args.ff_scale)
               for s in args.dev_seeds]
        rows = [(p["seed"], round(p["D_contingent"], 3), round(p["D_yoked"], 3),
                 p["count_c1"], round(p["test_rate_c1"], 3), _steer(p)) for p in per]
        out_obj = {"probe": "gateB_stage2n_seeds", "backend": backend["backend"],
                   "config": {"nmda_ratio": args.nmda_ratio, "xinhib_scale": args.xinhib_scale,
                              "ff_scale": args.ff_scale, "fix_e": bool(args.fix_e)},
                   "rows(seed,Dc,Dy,count_c1,test_rate_c1,steer)": rows,
                   "steer_passes": int(sum(r[-1] for r in rows))}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out_obj, indent=2, default=float) + "\n")
        print(json.dumps(out_obj, indent=2, default=float))
        return 0

    # full battery under FIX E + FIX F -- dev steer + acquisition lesion + reversal, mirrors 2k.
    with _patched_accum(True, args.fix_e, args.nmda_ratio, args.xinhib_scale, args.ff_scale):
        rc = k.main(["--mode", "full", "--seed", str(args.seed),
                     "--dev-seeds", *[str(s) for s in args.dev_seeds],
                     "--lesion-seed", str(args.lesion_seed),
                     "--n-train", str(args.n_train), "--n-test", str(args.n_test),
                     "--reward-lr", str(args.reward_lr)])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
