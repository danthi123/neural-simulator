"""Gate B Stage 2p: FIX G -- a STRIATAL feedforward-inhibition / MSN down-state homeostat that
silences the channel-0-open lock AT ITS SOURCE (the str_d1 baseline firing asymmetry), instead of
paving over its downstream symptom (the thalamic temporal head-start) at the commit stage.

WHY THIS LOCUS (the corrected diagnosis after Stages 2i-2o). On the lone held-out miss 730705 the
str_d1 D1 policy is CORRECTLY LEARNED (str_d1_1 ~286 >> str_d1_0 ~106) and Stage 2m's FIX E
(BG-output homeostat) even INVERTS the thalamic aggregate (thal_1 > thal_0). Yet the motor winner
does not flip, because the commit WTA integrates thal_0's TEMPORAL head-start. Stages 2l/2m traced
that head-start to a single upstream cause: str_d1_0 fires ~86 spikes at BASELINE while str_d1_1
fires ~0, so gpi_0 is paused and thal_0 enters the onset PRIMED (~-45 mV) with a head-start, while
gpi_1's hyperexcitability keeps thal_1 at rest. Stage 2m recorded that this baseline lock is NOT
k-reducible (intrinsic-gain scaling does not silence an already-firing MSN) and that the only way
found to overcome the head-start downstream -- a commit de-latch (Stage 2n/2o) -- also unmasks the
unlearned FIX-C channel and FAILS the acquisition-lesion anti-cheat. The head-start was therefore
"characterized" but never attacked AT ITS SOURCE.

WHAT FIX G DOES (target-blind; it must not manufacture the policy). FIX G is the direct STRIATAL
analogue of FIX E, but it uses feedforward INHIBITION (a tonic hyperpolarizing current, the FSI /
down-state companion process) instead of an intrinsic-gain scale -- because inhibition CAN silence
an already-firing MSN where `cp_izh_k` cannot. On a same-seed PROBE bridge (pre-training,
target-agnostic; the training bridge's RNG is untouched) it measures each str_d1 channel's baseline
onset firing, and, only under an EXTREME baseline asymmetry (gate), it applies a standing
hyperpolarizing bias to the OVER-ACTIVE channel (identified purely by baseline activity, never by
which action is rewarded), calibrated to bring that channel's baseline down to the quiet channel's
level (the cross-channel set-point). The bias is re-applied every step by wrapping the bridge's
own step (after the afferents are set), so it is a standing intrinsic property, additive and
DEFAULT-OFF -- byte-identical to the Stage-2k base when off. Authoritative backend = numpy.

SCAFFOLD HONESTY (brain-based-only standard). A host-set tonic hyperpolarizing current is a
SCAFFOLD for a real spiking striatal FSI feedforward-inhibition population (parvalbumin+ FSIs hold
MSNs in the hyperpolarized down-state until strong coordinated cortical drive arrives). It is in
the same methodological class as FIX C/E (host-set standing intrinsic properties) and is declared
as a switched-off biological process in the verdict; the follow-on is to realise it as an FSI pool.

VERDICT is EARNED via tools.verdict.Verdict + tools.lab (no host formula decides the credit; the
YOKED control is sacred; the acquisition-lesion must stay action 0). This file emits its own
verdict; read the printed status, do not assume a GO.

>>> OUTCOME (numpy, this file) -- HONEST NEGATIVE / mechanism REFUTED + a corrected diagnosis. FIX G
    does NOT engage a working bias on 730705 (`--mode refute` records the evidence, seconds):
      1. THERE IS NO str_d1 PRE-CUE BASELINE LOCK. At arousal=FALSE (true baseline) str_d1 = [0, 0].
         Stage 2m's "baseline str_d1_0 ~86" was the arousal=TRUE ONSET (cue) response, not a
         quiescent baseline -- so "silence the str_d1 baseline lock" attacks a firing pattern that
         only exists WHILE the cue drives the striatum.
      2. FEEDFORWARD INHIBITION BACKFIRES ON THESE MSNs. The IZH2007 striatal MSNs sit in a
         negative-b regime (cp_izh_b ~ -2), so a hyperpolarizing current does NOT silence the
         over-active channel -- it triggers POST-INHIBITORY REBOUND and str_d1_0 fires MORE
         (ext -400 pA -> 86; -2000 -> 129; -10000 -> 1783). No intrinsic knob (k, b, a, d, vpeak,
         intrinsic current) and no OU-noise change moves the 86-spike onset count either. So the
         calibrated bias is 0.0 and FIX G is INERT (byte-identical to the Stage-2k base).
      3. THE REAL RESIDUAL IS DOWNSTREAM AND UNCHANGED. With the trained policy correct
         (str_d1 = [104, 286]) and FIX E inverting the thalamic aggregate (thal = [215, 228],
         thal_1 > thal_0), the commit still latches to channel 0: commit = [388, 0] (commit_1 never
         ignites), motor = [795, 0] -> action 0. cp_izh_vr is already channel-symmetric, so the
         gpi/thal entry-state asymmetry Stage 2l measured is a DYNAMIC state (gpi-pause timing), not
         a resting-potential parameter -- equalising vr changes nothing. This reproduces the Stage
         2o boundary: the commit WTA cannot read out the (real) learned thalamic advantage.
    CORRECTED next mechanism (named, not deferred): the ONE thing shown to flip 730705 legitimately
    is Stage 2m's onset entry-state equalisation (11/12, no de-latch). Realise it AS BIOLOGY, not a
    host membrane reset: a spiking TRN-like feedforward-inhibition pool that synchronises the
    thalamic onset each selection epoch (the str_fsi population is an in-model template), removing
    the gpi-pause TIMING head-start at the thalamus so the commit's ignition race reflects the
    higher (learned) thalamic drive rather than whichever channel de-inhibits first.
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
    ONSET_STEPS, CONSTRUCTION_SEED,
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
from research.runners._vocal_action_selector_gate import _indices
from sim.backend import get_backend, to_host
from tools.verdict import Verdict
from tools.lab import assert_backend, attributable_to

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research/findings/raw/gateb_stage2p_striatal_ffi_downstate"

# ---- FIX G config (striatal feedforward-inhibition / MSN down-state homeostat) -------------
FIXG_ASYM_RATIO = 3.0          # engage only under an extreme str_d1 baseline asymmetry
FIXG_MIN_SETPOINT = 5.0        # ignore near-silent channels (avoid divide-by-noise)
# hyperpolarizing bias grid (pA, applied to the OVER-ACTIVE channel); more negative = stronger FFI
FIXG_BIAS_GRID = (-10.0, -25.0, -50.0, -100.0, -200.0, -400.0)

_FIXG_CACHE: dict[int, dict] = {}


# ---------------------------------------------------------------- FIX E helper (for stacking)
def _scale_regions(bridge, scales: dict) -> None:
    xp, _ = get_backend()
    for reg, sc in scales.items():
        if sc != 1.0:
            idx = xp.asarray(_indices(bridge, reg))
            bridge.cp_izh_k[idx] = bridge.cp_izh_k[idx] * xp.float32(sc)


# ---------------------------------------------------------------- FIX G calibration ----------
def _str_d1_baseline_fire(bridge, *, bias_channel=None, bias_pa: float = 0.0,
                          arousal: bool = True, steps: int = ONSET_STEPS) -> dict:
    """One target-blind run; per-channel str_d1 spike counts. arousal=True is the cue ONSET;
    arousal=False is the true (pre-cue) baseline. If bias_channel is given, a hyperpolarizing bias
    is (re)applied to that channel each step, AFTER the afferents are set."""
    xp, _ = get_backend()
    idx = {c: np.asarray(_indices(bridge, f"str_d1_{c}")) for c in CHANNELS}
    bidx = xp.asarray(_indices(bridge, f"str_d1_{bias_channel}")) if bias_channel is not None else None
    tot = {c: 0 for c in CHANNELS}
    for _ in range(steps):
        _apply_afferents(bridge, arousal=arousal)
        if bidx is not None and bias_pa != 0.0:
            bridge.cp_external_input_current[bidx] = xp.float32(bias_pa)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        fs = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        for c in CHANNELS:
            tot[c] += int(fs[idx[c]].sum())
    return tot


def _measure_str_d1(seed: int, *, bias_channel=None, bias_pa: float = 0.0,
                    arousal: bool = True) -> dict:
    """str_d1 firing per channel on a canonical same-seed probe bridge (optionally with a candidate
    FFI bias applied), target-blind. Probe RNG never touches the training bridge."""
    bk = dict(enable_reward=True, plastic_d1=True, ou_seed=None,
              ou_sigma=SIGMA_UNCERTAIN, plastic_d2=True)
    b = build_stage2_bridge(seed, **bk)
    _reconfigure_da_s(b)
    _set_sigma(b, _sigma_from_conf(0.0))
    _settle(b)
    fired = _str_d1_baseline_fire(b, bias_channel=bias_channel, bias_pa=bias_pa, arousal=arousal)
    b.clear_simulation_state_and_gpu_memory()
    return fired


def _calibrate_fixg(seed: int) -> dict:
    """Target-blind: measure str_d1 baseline per channel; if extremely asymmetric, choose the
    hyperpolarizing bias (from FIXG_BIAS_GRID) applied to the OVER-ACTIVE channel that brings its
    baseline closest to the quiet channel's set-point. Returns the calibration record."""
    if seed in _FIXG_CACHE:
        return _FIXG_CACHE[seed]
    base = _measure_str_d1(seed)
    f0, f1 = base[0], base[1]
    hi = max(f0, f1)
    lo = max(min(f0, f1), 1e-6)
    over = 0 if f0 >= f1 else 1
    setpoint = float(min(f0, f1))          # bring the over-active channel down to the quiet one
    engaged = bool(hi >= FIXG_MIN_SETPOINT and (hi / lo) > FIXG_ASYM_RATIO)
    bias = 0.0
    post_over = hi
    curve = []
    if engaged:
        best_bias, best_err = 0.0, abs(hi - setpoint)
        for pa in FIXG_BIAS_GRID:
            probe = _measure_str_d1(seed, bias_channel=over, bias_pa=pa)
            err = abs(probe[over] - setpoint)
            curve.append({"bias_pa": float(pa), "str_d1": [probe[0], probe[1]]})
            if err < best_err:
                best_err, best_bias, post_over = err, float(pa), probe[over]
        bias = best_bias
    out = {"seed": int(seed), "baseline": [int(f0), int(f1)], "over_channel": int(over),
           "setpoint": setpoint, "engaged": engaged, "bias_pa": float(bias),
           "baseline_over": int(hi), "post_over": int(post_over), "probe_curve": curve}
    _FIXG_CACHE[seed] = out
    return out


def _apply_fixg(bridge, seed: int) -> dict:
    """Apply the standing FIX G FFI bias by wrapping the bridge's own step so the hyperpolarizing
    current is (re)set on the over-active str_d1 channel every step, after the afferents are set.
    Intrinsic/synaptic only: touches no reward/DA signal and selects no action."""
    xp, _ = get_backend()
    cal = _calibrate_fixg(seed)
    if not cal["engaged"] or cal["bias_pa"] == 0.0:
        return cal
    bidx = xp.asarray(_indices(bridge, f"str_d1_{cal['over_channel']}"))
    pa = xp.float32(cal["bias_pa"])
    orig_step = bridge._run_one_simulation_step

    def wrapped_step(*a, **kw):
        bridge.cp_external_input_current[bidx] = pa   # str_d1 external current is never set elsewhere
        return orig_step(*a, **kw)

    bridge._run_one_simulation_step = wrapped_step
    return cal


@contextlib.contextmanager
def _patched_2p(fix_e: bool, fix_g: bool):
    """Wrap Stage-2k's build_stage2_bridge so every trial/test/lesion bridge carries the standing
    FIX E and/or FIX G properties. Both off -> a no-op wrapper -> byte-identical to Stage 2k."""
    orig = k.build_stage2_bridge

    def wrapped(seed, **kwargs):
        b = orig(seed, **kwargs)
        if fix_e:
            _apply_fixe(b, int(seed))
        if fix_g:
            _apply_fixg(b, int(seed))
        return b

    k.build_stage2_bridge = wrapped
    try:
        yield
    finally:
        k.build_stage2_bridge = orig


def run_seed_swap_2p(seed: int, *, n_train: int, n_test: int, fix_e: bool, fix_g: bool,
                     reward_learning_rate: float = REWARD_LEARNING_RATE,
                     fix_c: bool = True, fix_d: bool = True) -> dict:
    with _patched_2p(fix_e, fix_g):
        return k.run_seed_swap(seed, n_train=n_train, n_test=n_test,
                               reward_learning_rate=reward_learning_rate,
                               fix_a=False, fix_b=True, fix_c=fix_c, fix_d=fix_d)


def run_condition_2p(seed: int, *, condition: str, target: int, fix_e: bool, fix_g: bool,
                     fix_d: bool = True, **kw):
    with _patched_2p(fix_e, fix_g):
        return k.run_condition(seed, condition=condition, target=target,
                               fix_a=False, fix_b=True, fix_c=True, fix_d=fix_d, **kw)


# ---------------------------------------------------------------- byte-identity (off) --------
def _neq(a, b):
    a = tuple(a) if isinstance(a, (list, tuple)) else a
    b = tuple(b) if isinstance(b, (list, tuple)) else b
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:
            return False
        return abs(a - b) > 0.0
    return a != b


def _assert_off_byte_identical(seed: int, *, n_train: int, n_test: int,
                               reward_learning_rate: float) -> dict:
    """FIX E off + FIX G off must reproduce the Stage-2k base (fix_c on, fix_d on) exactly."""
    mine = run_seed_swap_2p(seed, n_train=n_train, n_test=n_test, fix_e=False, fix_g=False,
                            reward_learning_rate=reward_learning_rate)
    ref = k.run_seed_swap(seed, n_train=n_train, n_test=n_test,
                          reward_learning_rate=reward_learning_rate,
                          fix_a=False, fix_b=True, fix_c=True, fix_d=True)
    keys = ["D_contingent", "D_yoked", "count_c0", "count_c1",
            "test_rate_c0", "test_rate_c1", "baseline_p0"]
    mism = {kk: (mine.get(kk), ref.get(kk)) for kk in keys if _neq(mine.get(kk), ref.get(kk))}
    return {"seed": int(seed), "byte_identical_off": (len(mism) == 0), "mismatch": mism}


# ---------------------------------------------------------------- legitimacy (acq lesion) ----
def _legitimacy_acq_lesion(seed: int, *, n_train: int, n_test: int,
                           reward_learning_rate: float, fix_e: bool, fix_g: bool) -> dict:
    """FIX G (+FIX E) must NOT manufacture action 1 without the D1 learning. Build an UNTRAINED
    (acq_lesion) bridge WITH the fixes on and ask whether action 1 wins at test."""
    from research.runners._vocal_gateb_stage2g_hammond_deltap import _p_action0
    lkw = dict(n_train=n_train, n_test=n_test, reward_learning_rate=reward_learning_rate)
    la0 = run_condition_2p(seed, condition="acq_lesion", target=0, fix_e=fix_e, fix_g=fix_g, **lkw)
    la1 = run_condition_2p(seed, condition="acq_lesion", target=1, fix_e=fix_e, fix_g=fix_g, **lkw)
    D_acq_lesion = _p_action0(la0) - _p_action0(la1)
    p0_la1 = _p_action0(la1)
    return {"seed": int(seed), "D_contingent_acq_lesion": float(D_acq_lesion),
            "p_action0_target1_acq_lesion": float(p0_la1),
            "acq_lesion_action1_does_not_win": bool(p0_la1 >= 0.5),
            "n_clean_target1": int(la1["test_n_clean"]),
            "test_rate_target1": float(la1["test_target_rate"])}


# ---------------------------------------------------------------- refutation evidence --------
def _refute_evidence(seed: int) -> dict:
    """The decisive cheap evidence that FIX G's premise is refuted (seconds, no training):
      (1) there is no str_d1 PRE-CUE baseline (arousal=False -> [0,0]);
      (2) the FFI bias grid triggers rebound (hyperpolarising the over-active channel INCREASES it);
      (3) with FIX E the thal aggregate inverts (thal_1>thal_0) yet the commit still latches to 0."""
    base_off = _measure_str_d1(seed, arousal=False)
    base_on = _measure_str_d1(seed, arousal=True)
    cal = _calibrate_fixg(seed)               # probe_curve holds the FFI rebound grid
    fe = _calibrate_fixe(seed)
    b, cnt, w, r0 = _build_fixc_trained(seed, 1)
    _scale_regions(b, fe["scales"])
    winners, m = _test_cascade(b)
    b.clear_simulation_state_and_gpu_memory()
    return {
        "seed": int(seed),
        "str_d1_true_baseline_arousal_off": [base_off[0], base_off[1]],
        "str_d1_cue_onset_arousal_on": [base_on[0], base_on[1]],
        "no_precue_baseline_lock": bool(base_off[0] == 0 and base_off[1] == 0),
        "ffi_rebound_grid": cal["probe_curve"],
        "ffi_bias_reduces_firing": bool(cal["bias_pa"] != 0.0),   # expected False (refuted)
        "fixg_calibrated_bias_pa": cal["bias_pa"],
        "fixe_scales": fe["scales"],
        "fixe_trained_cascade": {
            "str_d1": [round(m["str_d1_0"], 0), round(m["str_d1_1"], 0)],
            "thal": [round(m["thal_0"], 0), round(m["thal_1"], 0)],
            "commit": [round(m["commit_0"], 0), round(m["commit_1"], 0)],
            "motor": [round(m["motor_0"], 0), round(m["motor_1"], 0)],
            "n_act1": int(sum(winners)), "n_trials": len(winners),
        },
        "thal_inverts_but_commit_latches": bool(m["thal_1"] > m["thal_0"] and m["commit_1"] == 0),
    }


# ---------------------------------------------------------------- cheap decisive instrument --
def run_diag(seed: int, *, fix_e: bool = True) -> str:
    """The decisive cheap instrument: does silencing the str_d1 baseline head-start at its SOURCE
    let the trained-bridge commit read out the learned D1 policy (flip to action 1), WITHOUT the
    de-latch, while an UNTRAINED bridge stays action 0? Pre-trained cascade, seconds not minutes."""
    lines = [f"Gate B Stage 2p -- FIX G (str_d1 feedforward-inhibition / down-state) on {seed}.",
             "Q: does silencing the str_d1 baseline lock at its SOURCE flip 730705 legitimately?", ""]
    cal = _calibrate_fixg(seed)
    lines.append(f"[str_d1 baseline target-blind] {cal['baseline']} over_channel={cal['over_channel']} "
                 f"setpoint={cal['setpoint']:.0f} engaged={cal['engaged']} bias_pa={cal['bias_pa']}")
    for pc in cal["probe_curve"]:
        lines.append(f"    bias {pc['bias_pa']:>8.0f} pA -> str_d1={pc['str_d1']}")
    lines.append("")

    # trained target=1: baseline vs FIX G vs (optionally) FIX E + FIX G
    b, cnt, w, r0 = _build_fixc_trained(seed, 1)
    lines.append(f"[trained target=1] count={cnt} r0_d1={[round(x, 1) for x in r0]}")
    lines.append(_fmt("  no fix              ", *_test_cascade(b)))
    b.clear_simulation_state_and_gpu_memory()

    b, _, _, _ = _build_fixc_trained(seed, 1)
    _apply_fixg(b, seed)
    lines.append(_fmt("  FIX G (str_d1 FFI)  ", *_test_cascade(b)))
    b.clear_simulation_state_and_gpu_memory()

    if fix_e:
        fe = _calibrate_fixe(seed)
        b, _, _, _ = _build_fixc_trained(seed, 1)
        _scale_regions(b, fe["scales"])
        _apply_fixg(b, seed)
        lines.append(_fmt("  FIX E + FIX G       ", *_test_cascade(b)))
        b.clear_simulation_state_and_gpu_memory()

    # contingency control: UNTRAINED (fix_d=False -> the release never fires -> policy never forms)
    b, cnt2, _, _ = _build_fixc_trained(seed, 1, fix_d=False)
    _apply_fixg(b, seed)
    lines.append("")
    lines.append(f"[UNTRAINED fix_d=False count={cnt2}] contingency control -- FIX G must NOT flip it")
    lines.append(_fmt("  FIX G untrained     ", *_test_cascade(b)))
    b.clear_simulation_state_and_gpu_memory()
    lines.append("")
    lines.append("READ: if a FIX row reaches n_act1=8/8 on the trained bridge while 'FIX G untrained'")
    lines.append("stays n_act1=0/8, silencing the str_d1 source-lock is a legitimate close of 730705.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- full smoke (earned verdict) -
def _smoke(seed: int, *, n_train: int, n_test: int, reward_learning_rate: float,
           fix_e: bool, fix_g: bool) -> dict:
    backend = _backend_info()
    assert_backend(backend["backend"], note="Gate B numpy-authoritative")

    # 1) byte-identity when off (both fixes off -> Stage-2k base)
    byte = _assert_off_byte_identical(seed, n_train=n_train, n_test=n_test,
                                      reward_learning_rate=reward_learning_rate)

    # 2) FIX-on full contingency battery vs the Stage-2k base
    cal = _calibrate_fixg(seed)
    on = run_seed_swap_2p(seed, n_train=n_train, n_test=n_test, fix_e=fix_e, fix_g=fix_g,
                          reward_learning_rate=reward_learning_rate)
    base = k.run_seed_swap(seed, n_train=n_train, n_test=n_test,
                           reward_learning_rate=reward_learning_rate,
                           fix_a=False, fix_b=True, fix_c=True, fix_d=True)

    # 3) acquisition-lesion legitimacy (untrained must stay action 0)
    legit = _legitimacy_acq_lesion(seed, n_train=n_train, n_test=n_test,
                                   reward_learning_rate=reward_learning_rate,
                                   fix_e=fix_e, fix_g=fix_g)

    steer_on = _steer(on)
    flips = bool(on["test_rate_c1"] > 0.0)
    contingency_gap = float(on["D_contingent"] - on["D_yoked"])

    # ATTRIBUTION (tools.lab): whose is the contingency? Subtract the acquisition-lesion arm from the
    # intact arm -- measuring both is not the same as asking whose the difference was (gap#5 banked
    # both numbers one key apart while the clamp, not the lever, owned 97% of the change).
    acq_attr = attributable_to("acquisition D1 plasticity",
                                   on["D_contingent"], legit["D_contingent_acq_lesion"])

    # ---- EARN the verdict: no host formula decides credit; the yoked control is sacred --------
    v = Verdict("Gate B 730705 delayed-credit action-specific + reversible via striatal FFI (FIX G)")
    v.disabled("spiking striatal FSI feedforward-inhibition population",
               why="FIX G is a host-set tonic hyperpolarizing bias -- an honest SCAFFOLD for a real "
                   "PV+ FSI / MSN-down-state pool; measured under this isolation")
    v.require("FIX G engages (str_d1 baseline asymmetric)", cal["engaged"], expect=True)
    v.knob("fixg_bias_pa", requested=cal["bias_pa"], applied=cal["bias_pa"])
    v.reaches("str_d1 over-channel baseline silenced toward set-point",
              before=cal["baseline_over"], after=cal["post_over"])
    v.require("byte-identical when off (Stage-2k GO protected)", byte["byte_identical_off"], expect=True)
    v.control("YOKED is not contingent (sacred anti-cheat)",
              treatment=on["D_contingent"], control=on["D_yoked"], min_separation=0.20)
    v.require("acquisition-lesion does NOT manufacture action 1 (contingency owned by D1)",
              legit["acq_lesion_action1_does_not_win"], expect=True)
    v.floor("contingent credit above no-credit floor", measured=on["D_contingent"], floor=0.0)
    v.require("held-out 730705 expresses action 1 at test (test_rate_c1>0)", flips, expect=True)
    decided = v.decide(go=bool(steer_on and flips and legit["acq_lesion_action1_does_not_win"]
                               and contingency_gap >= 0.20))

    return {
        "seed": int(seed), "fix_e": fix_e, "fix_g": fix_g,
        "fixg_cal": cal,
        "refute_evidence": _refute_evidence(seed),
        "byte_identity_off": byte,
        "FIX_ON": {"count_c1": on["count_c1"], "test_rate_c1": on["test_rate_c1"],
                   "test_rate_c0": on["test_rate_c0"],
                   "D_contingent": on["D_contingent"], "D_yoked": on["D_yoked"],
                   "contingency_gap": contingency_gap, "steer": steer_on},
        "STAGE2K_BASE": {"count_c1": base["count_c1"], "test_rate_c1": base["test_rate_c1"],
                         "D_contingent": base["D_contingent"], "D_yoked": base["D_yoked"],
                         "steer": _steer(base)},
        "legitimacy_acq_lesion": legit,
        "acquisition_attribution": acq_attr,
        "flips": flips,
        "verdict": decided,
    }


def _reversal(seed: int, *, n_train: int, n_test: int, reward_learning_rate: float,
              fix_e: bool, fix_g: bool) -> dict:
    """Reversibility anti-cheat: train target 0, then flip the contingency to target 1. The choice
    must REVERSE with the contingency (not stay latched). Uses the Stage-2k reversal machinery."""
    from research.runners._vocal_gateb_stage2g_hammond_deltap import _p_action0
    with _patched_2p(fix_e, fix_g):
        r0 = k.run_condition(seed, condition="intact", target=0, n_train=n_train, n_test=n_test,
                             reward_learning_rate=reward_learning_rate,
                             fix_a=False, fix_b=True, fix_c=True, fix_d=True)
        r1 = k.run_condition(seed, condition="intact", target=1, n_train=n_train, n_test=n_test,
                             reward_learning_rate=reward_learning_rate,
                             fix_a=False, fix_b=True, fix_c=True, fix_d=True)
    return {"seed": int(seed), "p_action0_target0": float(_p_action0(r0)),
            "p_action0_target1": float(_p_action0(r1)),
            "reverses": bool(_p_action0(r0) > _p_action0(r1))}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["diag", "smoke", "byte", "legit", "seeds", "reversal", "refute"],
                        default="diag")
    parser.add_argument("--seed", type=int, default=730705)
    parser.add_argument("--diag-seed", type=int, default=730705)
    parser.add_argument("--smoke-seeds", type=int, nargs="*", default=[730705])
    parser.add_argument("--byte-seeds", type=int, nargs="*", default=[730703, 730705])
    parser.add_argument("--dev-seeds", type=int, nargs="*", default=list(DEV_SEEDS))
    parser.add_argument("--fix-e", action="store_true", help="stack FIX E (BG-output homeostat)")
    parser.add_argument("--no-fix-g", action="store_true", help="disable FIX G (control)")
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--reward-lr", type=float, default=REWARD_LEARNING_RATE)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    backend = _backend_info()
    fix_g = not args.no_fix_g
    started = time.perf_counter()

    if args.mode == "diag":
        txt = run_diag(args.diag_seed, fix_e=args.fix_e)
        out = Path(args.out) if args.out else OUT_DIR / f"diag_{args.diag_seed}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt)
        print(txt)
        return 0

    if args.mode == "byte":
        res = [_assert_off_byte_identical(s, n_train=args.n_train, n_test=args.n_test,
                                          reward_learning_rate=args.reward_lr)
               for s in args.byte_seeds]
        ok = all(r["byte_identical_off"] for r in res)
        artifact = {"probe": "gateB_stage2p_byte_identity_off",
                    "backend": backend["backend"], "device": backend["device"],
                    "all_byte_identical": ok, "per_seed": res}
        out = Path(args.out) if args.out else OUT_DIR / f"byte_{backend['backend']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        assert ok, f"FIX E/G OFF is NOT byte-identical to Stage 2k base: {res}"
        return 0

    if args.mode == "refute":
        res = _refute_evidence(args.seed)
        artifact = {"probe": "gateB_stage2p_refute_evidence",
                    "backend": backend["backend"], "device": backend["device"], "evidence": res}
        out = Path(args.out) if args.out else OUT_DIR / f"refute_{args.seed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
        print(json.dumps(artifact, indent=2, default=float))
        return 0

    if args.mode == "legit":
        res = _legitimacy_acq_lesion(args.seed, n_train=args.n_train, n_test=args.n_test,
                                     reward_learning_rate=args.reward_lr,
                                     fix_e=args.fix_e, fix_g=fix_g)
        print(json.dumps(res, indent=2, default=float))
        return 0

    if args.mode == "reversal":
        res = _reversal(args.seed, n_train=args.n_train, n_test=args.n_test,
                        reward_learning_rate=args.reward_lr, fix_e=args.fix_e, fix_g=fix_g)
        print(json.dumps(res, indent=2, default=float))
        return 0

    if args.mode == "seeds":
        per = [run_seed_swap_2p(s, n_train=args.n_train, n_test=args.n_test,
                                fix_e=args.fix_e, fix_g=fix_g, reward_learning_rate=args.reward_lr)
               for s in args.dev_seeds]
        rows = [(p["seed"], round(p["D_contingent"], 3), round(p["D_yoked"], 3),
                 p["count_c1"], round(p["test_rate_c1"], 3), _steer(p)) for p in per]
        out_obj = {"probe": "gateB_stage2p_seeds", "backend": backend["backend"],
                   "rows(seed,Dc,Dy,count_c1,test_rate_c1,steer)": rows,
                   "steer_passes": int(sum(r[-1] for r in rows))}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out_obj, indent=2, default=float) + "\n")
        print(json.dumps(out_obj, indent=2, default=float))
        return 0

    # mode == smoke : the full earned-verdict battery on the held-out miss
    results = [_smoke(s, n_train=args.n_train, n_test=args.n_test,
                      reward_learning_rate=args.reward_lr, fix_e=args.fix_e, fix_g=fix_g)
               for s in args.smoke_seeds]
    artifact = {"probe": "gateB_stage2p_striatal_ffi_downstate_smoke",
                "backend": backend["backend"], "device": backend["device"],
                "fix_e": args.fix_e, "fix_g": fix_g,
                "smoke_seeds": args.smoke_seeds,
                "config": {"fixg_asym_ratio": FIXG_ASYM_RATIO,
                           "fixg_bias_grid": list(FIXG_BIAS_GRID),
                           "fixg_min_setpoint": FIXG_MIN_SETPOINT},
                "per_seed": results,
                "status": [r["verdict"]["status"] for r in results],
                "elapsed_seconds": float(time.perf_counter() - started)}
    out = Path(args.out) if args.out else OUT_DIR / f"smoke_{backend['backend']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
    print(json.dumps(artifact, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
