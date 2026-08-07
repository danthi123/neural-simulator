"""Gate B Stage 2o: a LEARNING-GATED FIX-F commit-integration threshold -- the legitimate
close of the lone held-out miss 730705, fixing the Stage-2n shortcut.

WHY (Stage-2n root cause, 2026-08-07): the FULL stack FIX C (wake str_d1_1 so it CAN fire) +
FIX D (train-time release so str_d1_1 LEARNS to ~286) + FIX F (test-time NMDA-slow accumulate-
then-commit) DID make 730705 express action 1 at test -- but the BINARY cross-inhibition
de-latch it needed (xinhib x0.1) also unmasked FIX C's target-blind intrinsic gain, so an
UNTRAINED (acq-lesion) bridge with str_d1_1 woken to ~124 ALSO picked action 1 (the acq-lesion
control failed = a SHORTCUT). The Gate B commit veto was doing legitimate work: it enforced that
only a LEARNED (~286-spike) str_d1_1 overcomes thal_0's temporal head-start. The de-latch is a
binary switch -- it cannot tell 286-from-learning apart from 124-from-gain-alone.

THE LEGITIMATE HYPOTHESIS (this stage): the drive from a LEARNED str_d1_1 (~286) and from a
woken-UNLEARNED str_d1_1 (~124) differ ~2.3x. Instead of a binary de-latch, add a LEARNING-GATED
FIX-F THRESHOLD: raise the commit-pool NMDA-reverberation IGNITION threshold (a graded intrinsic
property, target-blind, both commit channels equally) so the SUSTAINED drive from the LEARNED
str_d1_1 crosses the commit-integration bound and ignites commit_1, but the weaker drive from the
woken-UNLEARNED str_d1_1 does NOT. The test is whether a commit-integration threshold sits BETWEEN
the learned-286 and unlearned-124 drives.

THE MECHANISM (FIX-O commit threshold, additive on top of Stage-2n's FIX F; DEFAULT-OFF,
byte-identical when off -- ASSERTED; NO sim/ edit):
  * FIX F (Stage 2n, reused verbatim): NMDA-slow recurrence on the commit pools (Wang 2002,
    tau~100ms) + gentle commit_fs cross-inhibition so the loser can ENTER the accumulator.
  * FIX-O (new): scale BOTH commit pools' intrinsic gain cp_izh_k by COMMIT_K_SCALE (<1.0 = less
    excitable = MORE accumulated NMDA depolarisation required to ignite the reverberation). This
    is the SAME cp_izh_k intrinsic-excitability knob FIX C (MSN) and FIX E (BG output) use
    (Desai 1999 / Turrigiano 2011 intrinsic set-point), applied to the commit pools. It is a
    graded ignition BOUND, not a binary veto -- the accumulate-to-bound the record-grounded N6
    readout describes. Target-BLIND: both channels are scaled equally, so it cannot encode which
    action is rewarded (the target-0 contingency must survive it, and the acq-lesion must not).

THE KNOB SWEPT: COMMIT_K_SCALE (the commit ignition threshold) x NMDA_RATIO (the reverberation
gain) -- NOT the binary xinhib de-latch Stage-2n swept. A separating window exists iff some
(commit_k, nmda) makes the LEARNED cascade win action 1 while the UNLEARNED (acq-lesion) cascade
stays action 0.

DECISIVE ANTI-CHEAT (the whole point): the acquisition-lesion (untrained, plastic_d1=False) must
NOT express action 1 (acq_lesion_action1_does_not_win=true, D_contingent_acq_lesion~0). A flip the
acq-lesion also produces is a shortcut = NO-GO (exactly Stage 2n). ALSO: target-0 contingency must
hold (test_rate_c0 high, steer holds), and byte-identity when off.

HONEST OUTCOMES (a smoke is NOT a verdict):
  (i)  730705 flips legitimately (acq-lesion picks action-0, D1 owns the contingency, target-0
       intact, no dev regression) -> PROMISING; needs the full dev+held-out validation for a 6/6.
  (ii) flips but acq-lesion still fails, OR target-0 breaks, OR NO separating window exists ->
       730705 is a CONCLUSIVELY-characterized heterogeneity boundary; Gate B stands at >=5/6.
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
    SIGMA_UNCERTAIN, CONSTRUCTION_SEED,
    _backend_info, _reconfigure_da_s, _set_sigma, _settle, _sigma_from_conf,
    build_stage2_bridge, _motor_idx, _str_d1_idx, _str_d2_idx, _baseline_block,
    _p_action0,
)
from research.runners._vocal_gateb_stage2j_intrinsic_homeostasis import (
    _steer, _homeostat_engage, _calibrate_k_scale, _apply_k_homeostasis,
)
from research.runners._vocal_gateb_stage2l_commit_normalization import (
    _build_fixc_trained, _test_cascade, _fmt,
)
from research.runners._vocal_gateb_stage2m_bg_output_homeostat import _apply_fixe
from research.runners._vocal_gateb_stage2n_accumulate_commit import (
    _apply_accumulate_commit, _fixe_engaged, _commit_indices,
    FIXF_NMDA_RATIO, FIXF_XINHIB_SCALE, FIXF_FF_SCALE,
)
from research.runners._vocal_action_selector_gate import _indices
from research.runners._vocal_gateb_stage2g_hammond_deltap import ONSET_STEPS
from sim.backend import get_backend

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research/findings/raw/gateb_stage2o_learning_gated_commit"

# ---- FIX-O config (learning-gated commit-integration threshold on top of Stage-2n FIX F) ----
# COMMIT_K_SCALE < 1.0 = raise the commit NMDA-reverberation ignition threshold (Desai/Turrigiano
# intrinsic set-point on cp_izh_k, the same knob FIX C/E use). 1.0 = Stage-2n behaviour.
FIXO_COMMIT_K_SCALE = 1.0                       # default: no threshold raise (= Stage 2n)
FIXO_COMMIT_K_GRID = (1.0, 0.7, 0.5, 0.4, 0.3, 0.2)   # ignition-threshold sweep (descending excitability)
FIXO_ON = False                                 # module-level default OFF (byte-identical to 2k)

# separation criteria for "a threshold window exists" (fraction of test-cascade trials picking act 1)
SEP_LEARNED_MIN = 0.5     # LEARNED bridge must pick action 1 on >= this fraction of trials
SEP_UNLEARNED_MAX = 0.0   # UNLEARNED (acq-lesion) bridge must pick action 1 on <= this fraction


# ---------------------------------------------------------------- FIX-O mechanism ----------
def _apply_commit_threshold(bridge, commit_k_scale: float) -> None:
    """Raise the commit-pool NMDA-reverberation IGNITION threshold: scale BOTH commit pools'
    intrinsic gain cp_izh_k by commit_k_scale (<1 = less excitable = more accumulated NMDA drive
    needed to ignite). Target-BLIND (both channels equally). Same cp_izh_k knob FIX C/E use."""
    if commit_k_scale == 1.0:
        return
    xp, _ = get_backend()
    idx = _commit_indices(bridge)          # commit_0 AND commit_1 (target-blind)
    bridge.cp_izh_k[idx] = bridge.cp_izh_k[idx] * xp.float32(commit_k_scale)


def _apply_accumulate_commit_2o(bridge, *, nmda_ratio: float, xinhib_scale: float,
                                ff_scale: float, commit_k_scale: float) -> dict:
    """Stage-2n FIX F (NMDA-slow commit + gentle de-latch) + FIX-O ignition threshold."""
    diag = _apply_accumulate_commit(bridge, nmda_ratio=nmda_ratio, xinhib_scale=xinhib_scale,
                                    ff_scale=ff_scale)
    _apply_commit_threshold(bridge, commit_k_scale)
    diag["commit_k_scale"] = float(commit_k_scale)
    return diag


@contextlib.contextmanager
def _patched_accum_2o(accum_on: bool, fix_e: bool, nmda_ratio: float, xinhib_scale: float,
                      ff_scale: float, commit_k_scale: float, force_engage: bool = False):
    """Wrap Stage-2k's build_stage2_bridge so every trial/test/lesion bridge carries the standing
    FIX F + FIX-O commit property. Engages only where the (target-blind) FIX-E extreme-asymmetry
    gate fires -> byte-identical on the well-behaved seeds. accum_on=False -> pure no-op wrapper."""
    if not accum_on:
        yield
        return
    orig = k.build_stage2_bridge

    def wrapped(seed, **kwargs):
        b = orig(seed, **kwargs)
        if not (force_engage or _fixe_engaged(int(seed))):
            return b                      # recruitment gate did not fire -> byte-identical
        if fix_e:
            _apply_fixe(b, int(seed))
        _apply_accumulate_commit_2o(b, nmda_ratio=nmda_ratio, xinhib_scale=xinhib_scale,
                                    ff_scale=ff_scale, commit_k_scale=commit_k_scale)
        return b

    k.build_stage2_bridge = wrapped
    try:
        yield
    finally:
        k.build_stage2_bridge = orig


def run_seed_swap_2o(seed: int, *, n_train: int, n_test: int, accum_on: bool, fix_e: bool,
                     reward_learning_rate: float = REWARD_LEARNING_RATE,
                     nmda_ratio: float = FIXF_NMDA_RATIO, xinhib_scale: float = FIXF_XINHIB_SCALE,
                     ff_scale: float = FIXF_FF_SCALE, commit_k_scale: float = FIXO_COMMIT_K_SCALE,
                     fix_c: bool = True, fix_d: bool = True) -> dict:
    with _patched_accum_2o(accum_on, fix_e, nmda_ratio, xinhib_scale, ff_scale, commit_k_scale):
        return k.run_seed_swap(seed, n_train=n_train, n_test=n_test,
                               reward_learning_rate=reward_learning_rate,
                               fix_a=False, fix_b=True, fix_c=fix_c, fix_d=fix_d)


def run_condition_2o(seed: int, *, condition: str, target: int, accum_on: bool, fix_e: bool,
                     nmda_ratio: float = FIXF_NMDA_RATIO, xinhib_scale: float = FIXF_XINHIB_SCALE,
                     ff_scale: float = FIXF_FF_SCALE, commit_k_scale: float = FIXO_COMMIT_K_SCALE,
                     **kw):
    with _patched_accum_2o(accum_on, fix_e, nmda_ratio, xinhib_scale, ff_scale, commit_k_scale):
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


def _assert_fixo_off_byte_identical(seed: int, *, n_train: int, n_test: int,
                                    reward_learning_rate: float) -> dict:
    """FIX O off (accum_on=False) must reproduce the Stage-2k base (fix_c+fix_d)."""
    mine = run_seed_swap_2o(seed, n_train=n_train, n_test=n_test, accum_on=False, fix_e=False,
                            reward_learning_rate=reward_learning_rate)
    ref = k.run_seed_swap(seed, n_train=n_train, n_test=n_test,
                          reward_learning_rate=reward_learning_rate,
                          fix_a=False, fix_b=True, fix_c=True, fix_d=True)
    keys = ["D_contingent", "D_yoked", "count_c0", "count_c1",
            "test_rate_c0", "test_rate_c1", "baseline_p0"]
    mism = {kk: (mine.get(kk), ref.get(kk)) for kk in keys if _neq(mine.get(kk), ref.get(kk))}
    return {"seed": int(seed), "byte_identical_fixo_off": (len(mism) == 0), "mismatch": mism}


# ---------------------------------------------------------------- onset entry-state reset --
# The Stage-2m closing-stack lead, now CODED: a TRN-like selection-epoch reset that equalises
# each BG-output region's two channels' membrane state at onset, removing thal_0's initial-
# condition head-start so the commit competition can reflect the (learning-gated) thal ORDERING
# instead of latching on whichever channel entered onset primed. Target-BLIND (both channels set
# to their shared per-region mean; encodes no action).
_RESET_REGIONS = ("thal", "gpi", "commit", "commit_fs")


def _onset_equalise(bridge, regions=_RESET_REGIONS) -> None:
    xp, _ = get_backend()
    v = bridge.cp_membrane_potential_v
    u = bridge.cp_recovery_variable_u
    for r in regions:
        i0 = xp.asarray(_indices(bridge, f"{r}_0"))
        i1 = xp.asarray(_indices(bridge, f"{r}_1"))
        vm = xp.float32(0.5 * (float(v[i0].mean()) + float(v[i1].mean())))
        um = xp.float32(0.5 * (float(u[i0].mean()) + float(u[i1].mean())))
        v[i0] = vm; v[i1] = vm
        u[i0] = um; u[i1] = um


def _make_onset_reset_modifier():
    """Stateful modifier for _test_cascade: reset ONLY at the first onset step of each trial
    (the modifier is called once per onset step; ONSET_STEPS calls per trial, none during gap)."""
    st = {"n": 0}

    def mod(b):
        if st["n"] % ONSET_STEPS == 0:
            _onset_equalise(b)
        st["n"] += 1

    return mod


# ---------------------------------------------------------------- the UNLEARNED substrate --
def _build_fixc_untrained(seed: int, target: int):
    """The acq-lesion substrate: FIX-C-woken but UNTRAINED (plastic_d1=False, no D1 learning,
    reward-eligibility off) -- str_d1_1 woken to its FIX-C baseline (~124) but never potentiated.
    Mirrors run_condition(condition='acq_lesion') EXACTLY (same plastic_d1=False build_kwargs for
    the FIX-C k calibration) so the woken-unlearned drive matches the real acq-lesion case."""
    xp, _ = get_backend()
    bk = dict(enable_reward=True, plastic_d1=False, ou_seed=None,
              ou_sigma=SIGMA_UNCERTAIN, plastic_d2=True)
    b = build_stage2_bridge(seed, **bk)
    _reconfigure_da_s(b)
    b.core_config.reward_eligibility_from_coactivity = False
    midx = _motor_idx(b); d1 = _str_d1_idx(b); d2 = _str_d2_idx(b)
    _set_sigma(b, _sigma_from_conf(0.0)); _settle(b)
    r0 = _baseline_block(b, midx, d1, d2, target, N_TEST)["r0_d1"]
    dc, _dr, _ra = _homeostat_engage(r0)
    if dc is not None:
        ks, _f1, _fa = _calibrate_k_scale(seed, dc, bk)
        if ks > 1.0:
            _apply_k_homeostasis(b, dc, ks)
    return b, r0


# ---------------------------------------------------------------- separating-window sweep --
def _cascade_at_threshold(build_trained: bool, seed: int, *, nmda_ratio: float,
                          xinhib_scale: float, ff_scale: float, commit_k_scale: float,
                          fix_e: bool, onset_reset: bool = False, ntrials: int = 8):
    """Build the LEARNED (build_trained=True) or UNLEARNED (acq-lesion) bridge on `seed`, apply
    FIX F + FIX-O at commit_k_scale (optionally + the onset entry-state reset), run the test
    cascade. Returns (n_act1, ntrials, aggregates)."""
    if build_trained:
        b, _cnt, _w, _r0 = _build_fixc_trained(seed, 1)
    else:
        b, _r0 = _build_fixc_untrained(seed, 1)
    if fix_e:
        _apply_fixe(b, int(seed))
    _apply_accumulate_commit_2o(b, nmda_ratio=nmda_ratio, xinhib_scale=xinhib_scale,
                                ff_scale=ff_scale, commit_k_scale=commit_k_scale)
    mod = _make_onset_reset_modifier() if onset_reset else None
    winners, m = _test_cascade(b, ntrials=ntrials, modifier=mod)
    b.clear_simulation_state_and_gpu_memory()
    return int(sum(winners)), int(ntrials), m


def _separating_window_sweep(seed: int, *, commit_k_grid, nmda_ratio: float, xinhib_scale: float,
                             ff_scale: float, fix_e: bool, onset_reset: bool = False,
                             ntrials: int = 8) -> dict:
    """The DECISIVE cheap instrument: does a commit-integration threshold sit BETWEEN the learned
    (~286) and unlearned (~124) str_d1_1 drives? For each commit_k_scale, run the test cascade on
    both the LEARNED and the UNLEARNED (acq-lesion) bridge and record whether action 1 wins."""
    rows = []
    for ck in commit_k_grid:
        l_act1, l_n, l_m = _cascade_at_threshold(True, seed, nmda_ratio=nmda_ratio,
                                                 xinhib_scale=xinhib_scale, ff_scale=ff_scale,
                                                 commit_k_scale=ck, fix_e=fix_e,
                                                 onset_reset=onset_reset, ntrials=ntrials)
        u_act1, u_n, u_m = _cascade_at_threshold(False, seed, nmda_ratio=nmda_ratio,
                                                 xinhib_scale=xinhib_scale, ff_scale=ff_scale,
                                                 commit_k_scale=ck, fix_e=fix_e,
                                                 onset_reset=onset_reset, ntrials=ntrials)
        l_frac = l_act1 / l_n
        u_frac = u_act1 / u_n
        separates = bool(l_frac >= SEP_LEARNED_MIN and u_frac <= SEP_UNLEARNED_MAX)
        rows.append({
            "commit_k_scale": float(ck),
            "learned_n_act1": l_act1, "learned_frac": round(l_frac, 3),
            "learned_str_d1": [round(l_m["str_d1_0"], 0), round(l_m["str_d1_1"], 0)],
            "learned_gpi": [round(l_m["gpi_0"], 0), round(l_m["gpi_1"], 0)],
            "learned_thal": [round(l_m["thal_0"], 0), round(l_m["thal_1"], 0)],
            "learned_commit": [round(l_m["commit_0"], 0), round(l_m["commit_1"], 0)],
            "learned_motor": [round(l_m["motor_0"], 0), round(l_m["motor_1"], 0)],
            "unlearned_n_act1": u_act1, "unlearned_frac": round(u_frac, 3),
            "unlearned_str_d1": [round(u_m["str_d1_0"], 0), round(u_m["str_d1_1"], 0)],
            "unlearned_gpi": [round(u_m["gpi_0"], 0), round(u_m["gpi_1"], 0)],
            "unlearned_thal": [round(u_m["thal_0"], 0), round(u_m["thal_1"], 0)],
            "unlearned_commit": [round(u_m["commit_0"], 0), round(u_m["commit_1"], 0)],
            "unlearned_motor": [round(u_m["motor_0"], 0), round(u_m["motor_1"], 0)],
            "separates": separates,
            "margin": round(l_frac - u_frac, 3),
        })
    sep_rows = [r for r in rows if r["separates"]]
    window_exists = len(sep_rows) > 0
    if window_exists:
        # among separating rows, the one with the strongest learned expression (then largest margin)
        best = max(sep_rows, key=lambda r: (r["learned_frac"], r["margin"]))
    else:
        # no window: pick the largest-margin row (for documenting the boundary)
        best = max(rows, key=lambda r: r["margin"])
    return {"seed": int(seed), "window_exists": window_exists,
            "best_commit_k_scale": best["commit_k_scale"],
            "best_row": best, "sweep": rows,
            "sep_criteria": {"learned_min_frac": SEP_LEARNED_MIN,
                             "unlearned_max_frac": SEP_UNLEARNED_MAX}}


# ---------------------------------------------------------------- legitimacy (acq lesion) --
def _legitimacy_acq_lesion(seed: int, *, n_train: int, n_test: int, reward_learning_rate: float,
                           fix_e: bool, nmda_ratio: float, xinhib_scale: float, ff_scale: float,
                           commit_k_scale: float) -> dict:
    """FIX O (+FIX F/E) must NOT manufacture action 1 without D1 learning. Untrained (acq_lesion)
    bridge WITH the accumulate-commit ON at commit_k_scale: action 1 must NOT win at test."""
    lkw = dict(n_train=n_train, n_test=n_test, reward_learning_rate=reward_learning_rate)
    la0 = run_condition_2o(seed, condition="acq_lesion", target=0, accum_on=True, fix_e=fix_e,
                           nmda_ratio=nmda_ratio, xinhib_scale=xinhib_scale, ff_scale=ff_scale,
                           commit_k_scale=commit_k_scale, **lkw)
    la1 = run_condition_2o(seed, condition="acq_lesion", target=1, accum_on=True, fix_e=fix_e,
                           nmda_ratio=nmda_ratio, xinhib_scale=xinhib_scale, ff_scale=ff_scale,
                           commit_k_scale=commit_k_scale, **lkw)
    D_acq_lesion = _p_action0(la0) - _p_action0(la1)
    p0_la1 = _p_action0(la1)
    return {"seed": int(seed), "commit_k_scale": float(commit_k_scale),
            "D_contingent_acq_lesion_fixo_on": float(D_acq_lesion),
            "p_action0_target1_acq_lesion": float(p0_la1),
            "acq_lesion_action1_does_not_win": bool(p0_la1 >= 0.5),
            "acquisition_plasticity_share_ok": bool(abs(D_acq_lesion) < 0.30),
            "n_clean_target1": int(la1["test_n_clean"]),
            "test_rate_target1": float(la1["test_target_rate"])}


# ---------------------------------------------------------------- smoke (honest verdict) ---
def _full_pipeline_at(seed: int, *, n_train: int, n_test: int, reward_learning_rate: float,
                      fix_e: bool, nmda_ratio: float, xinhib_scale: float, ff_scale: float,
                      commit_k_scale: float) -> dict:
    """Full train->test seed_swap on the real pipeline at a chosen commit_k_scale."""
    on = run_seed_swap_2o(seed, n_train=n_train, n_test=n_test, accum_on=True, fix_e=fix_e,
                          reward_learning_rate=reward_learning_rate, nmda_ratio=nmda_ratio,
                          xinhib_scale=xinhib_scale, ff_scale=ff_scale, commit_k_scale=commit_k_scale)
    base = k.run_seed_swap(seed, n_train=n_train, n_test=n_test,
                           reward_learning_rate=reward_learning_rate,
                           fix_a=False, fix_b=True, fix_c=True, fix_d=True)

    def _def(x):
        return x == x
    return {
        "seed": int(seed), "commit_k_scale": float(commit_k_scale), "fix_e": bool(fix_e),
        "FIXO_ON": {
            "count_c0": on["count_c0"], "count_c1": on["count_c1"],
            "test_rate_c1": on["test_rate_c1"], "test_rate_c0": on["test_rate_c0"],
            "D_contingent": on["D_contingent"], "D_yoked": on["D_yoked"],
            "steer": _steer(on),
        },
        "STAGE2K_BASE": {
            "count_c1": base["count_c1"], "test_rate_c1": base["test_rate_c1"],
            "test_rate_c0": base["test_rate_c0"],
            "D_contingent": base["D_contingent"], "steer": _steer(base),
        },
        "SMOKE_no_nan": bool(_def(on["test_rate_c0"]) and _def(on["test_rate_c1"])),
        "SMOKE_730705_test_rate_c1_flips": bool(on["test_rate_c1"] > 0.0),
        "SMOKE_target0_intact": bool(_def(on["test_rate_c0"]) and on["test_rate_c0"] >= 0.5),
        "SMOKE_steer": _steer(on),
        "SMOKE_steer_improved_vs_2k": bool(_steer(on) and not _steer(base)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "sweep", "seeds", "full", "byte"],
                        default="smoke")
    parser.add_argument("--seed", type=int, default=CONSTRUCTION_SEED)
    parser.add_argument("--dev-seeds", type=int, nargs="*", default=list(DEV_SEEDS))
    parser.add_argument("--smoke-seeds", type=int, nargs="*", default=[730705])
    parser.add_argument("--byte-seeds", type=int, nargs="*", default=[730703, 730705])
    parser.add_argument("--legit-seed", type=int, default=730705)
    parser.add_argument("--lesion-seed", type=int, default=730605)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--reward-lr", type=float, default=REWARD_LEARNING_RATE)
    parser.add_argument("--fix-e", action="store_true", default=False,
                        help="ALSO apply the Stage-2m BG-output homeostat (aggregate inversion). "
                             "Default OFF: the learning-gated threshold is tested WITHOUT needing "
                             "the thalamic aggregate inverted (the cleanest test).")
    parser.add_argument("--nmda-ratio", type=float, default=FIXF_NMDA_RATIO)
    parser.add_argument("--xinhib-scale", type=float, default=FIXF_XINHIB_SCALE)
    parser.add_argument("--ff-scale", type=float, default=FIXF_FF_SCALE)
    parser.add_argument("--commit-k-scale", type=float, default=None,
                        help="fix the commit ignition threshold (skip the sweep in smoke)")
    parser.add_argument("--commit-k-grid", type=float, nargs="*", default=list(FIXO_COMMIT_K_GRID))
    parser.add_argument("--onset-reset", action="store_true", default=False,
                        help="ALSO apply the TRN-like onset entry-state reset (equalise thal/gpi "
                             "membrane at onset, removing thal_0's head-start) in the sweep cascade.")
    parser.add_argument("--cascade-trials", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    backend = _backend_info()
    started = time.perf_counter()

    if args.mode == "byte":
        res = [_assert_fixo_off_byte_identical(s, n_train=args.n_train, n_test=args.n_test,
                                               reward_learning_rate=args.reward_lr)
               for s in args.byte_seeds]
        ok = all(r["byte_identical_fixo_off"] for r in res)
        artifact = {"probe": "gateB_stage2o_byte_identity_fixo_off",
                    "backend": backend["backend"], "all_byte_identical": ok, "per_seed": res}
        out = Path(args.out) if args.out else OUT_DIR / f"byte_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        assert ok, f"FIX O OFF is NOT byte-identical to Stage 2k base: {res}"
        return 0

    if args.mode == "sweep":
        sweep = _separating_window_sweep(args.legit_seed, commit_k_grid=args.commit_k_grid,
                                         nmda_ratio=args.nmda_ratio, xinhib_scale=args.xinhib_scale,
                                         ff_scale=args.ff_scale, fix_e=args.fix_e,
                                         onset_reset=args.onset_reset, ntrials=args.cascade_trials)
        artifact = {"probe": "gateB_stage2o_separating_window_sweep",
                    "backend": backend["backend"], "fix_e": bool(args.fix_e),
                    "onset_reset": bool(args.onset_reset),
                    "config": {"nmda_ratio": args.nmda_ratio, "xinhib_scale": args.xinhib_scale,
                               "ff_scale": args.ff_scale},
                    "sweep": sweep, "elapsed_seconds": float(time.perf_counter() - started)}
        out = Path(args.out) if args.out else OUT_DIR / f"sweep_{args.legit_seed}_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        return 0

    if args.mode == "smoke":
        # (1) byte-identity assertion FIRST (fails loudly if the additive path perturbs base when off)
        byte = [_assert_fixo_off_byte_identical(s, n_train=args.n_train, n_test=args.n_test,
                                                reward_learning_rate=args.reward_lr)
                for s in args.byte_seeds]
        assert all(b["byte_identical_fixo_off"] for b in byte), \
            f"FIX O OFF not byte-identical to 2k base: {byte}"

        seed = args.smoke_seeds[0]
        # (2) the DECISIVE cheap instrument: does a separating threshold window exist?
        if args.commit_k_scale is not None:
            sweep = None
            best_ck = float(args.commit_k_scale)
        else:
            sweep = _separating_window_sweep(seed, commit_k_grid=args.commit_k_grid,
                                             nmda_ratio=args.nmda_ratio,
                                             xinhib_scale=args.xinhib_scale,
                                             ff_scale=args.ff_scale, fix_e=args.fix_e,
                                             ntrials=args.cascade_trials)
            best_ck = sweep["best_commit_k_scale"]

        # (3) the REAL-pipeline verdict at the best operating point: trained flip + target-0 intact
        pipeline = _full_pipeline_at(seed, n_train=args.n_train, n_test=args.n_test,
                                     reward_learning_rate=args.reward_lr, fix_e=args.fix_e,
                                     nmda_ratio=args.nmda_ratio, xinhib_scale=args.xinhib_scale,
                                     ff_scale=args.ff_scale, commit_k_scale=best_ck)
        # (4) the DECISIVE anti-cheat at the same operating point
        legit = _legitimacy_acq_lesion(seed, n_train=args.n_train, n_test=args.n_test,
                                       reward_learning_rate=args.reward_lr, fix_e=args.fix_e,
                                       nmda_ratio=args.nmda_ratio, xinhib_scale=args.xinhib_scale,
                                       ff_scale=args.ff_scale, commit_k_scale=best_ck)

        legitimate_flip = bool(pipeline["SMOKE_730705_test_rate_c1_flips"]
                               and legit["acq_lesion_action1_does_not_win"]
                               and legit["acquisition_plasticity_share_ok"]
                               and pipeline["SMOKE_target0_intact"])
        outcome = "(i) PROMISING: legitimate flip" if legitimate_flip else \
                  "(ii) boundary: no legitimate flip at any swept operating point"
        artifact = {"probe": "gateB_stage2o_learning_gated_commit_smoke",
                    "backend": backend["backend"], "device": backend["device"],
                    "smoke_seed": seed, "fix_e": bool(args.fix_e),
                    "best_commit_k_scale": best_ck,
                    "window_exists": (sweep["window_exists"] if sweep else None),
                    "byte_identity_fixo_off": byte,
                    "separating_window_sweep": sweep,
                    "full_pipeline_at_best": pipeline,
                    "legitimacy_acq_lesion": legit,
                    "config": {"nmda_ratio": args.nmda_ratio, "xinhib_scale": args.xinhib_scale,
                               "ff_scale": args.ff_scale, "fix_e": bool(args.fix_e),
                               "commit_k_grid": args.commit_k_grid},
                    "LEGITIMATE_FLIP": legitimate_flip,
                    "OUTCOME": outcome,
                    "elapsed_seconds": float(time.perf_counter() - started)}
        out = Path(args.out) if args.out else OUT_DIR / f"smoke_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        return 0

    if args.mode == "seeds":
        ck = args.commit_k_scale if args.commit_k_scale is not None else FIXO_COMMIT_K_SCALE
        per = [run_seed_swap_2o(s, n_train=args.n_train, n_test=args.n_test, accum_on=True,
                                fix_e=args.fix_e, reward_learning_rate=args.reward_lr,
                                nmda_ratio=args.nmda_ratio, xinhib_scale=args.xinhib_scale,
                                ff_scale=args.ff_scale, commit_k_scale=ck)
               for s in args.dev_seeds]
        rows = [(p["seed"], round(p["D_contingent"], 3), round(p["D_yoked"], 3),
                 p["count_c1"], round(p["test_rate_c1"], 3), _steer(p)) for p in per]
        out_obj = {"probe": "gateB_stage2o_seeds", "backend": backend["backend"],
                   "config": {"nmda_ratio": args.nmda_ratio, "xinhib_scale": args.xinhib_scale,
                              "ff_scale": args.ff_scale, "commit_k_scale": ck,
                              "fix_e": bool(args.fix_e)},
                   "rows(seed,Dc,Dy,count_c1,test_rate_c1,steer)": rows,
                   "steer_passes": int(sum(r[-1] for r in rows))}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out_obj, indent=2, default=float) + "\n")
        print(json.dumps(out_obj, indent=2, default=float))
        return 0

    # full battery under FIX F + FIX-O -- dev steer + acquisition lesion + reversal, mirrors 2k.
    ck = args.commit_k_scale if args.commit_k_scale is not None else FIXO_COMMIT_K_SCALE
    with _patched_accum_2o(True, args.fix_e, args.nmda_ratio, args.xinhib_scale, args.ff_scale, ck):
        rc = k.main(["--mode", "full", "--seed", str(args.seed),
                     "--dev-seeds", *[str(s) for s in args.dev_seeds],
                     "--lesion-seed", str(args.lesion_seed),
                     "--n-train", str(args.n_train), "--n-test", str(args.n_test),
                     "--reward-lr", str(args.reward_lr)])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
