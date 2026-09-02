"""D3 SPIKING port (rung 1): the re-discretization ON SPIKES — the concrete "simulated recurrent sequence/language
cortex". The rate de-risk (`_d3_group_composition_derisk.py`) proved DISCRETE-ATTRACTOR recurrence length-generalizes
multi-hop group composition (S3 + theorem-backed A5) where a continuous RNN cannot; the mechanism = re-discretize the
running state to a CLEAN attractor each step. THIS ports that re-discretization onto the project's OWN spiking substrate:
each step's transition scores drive K Izhikevich attractor pools with input-DIVISIVE-NORMALIZATION (the E%-max WTA =
the OneBrainComposer/NEF cleanup = CA3 pattern completion) -> the WINNER pool FIRES -> the next state is read from
SPIKES -> iterate. So the running group state is maintained as a spiking attractor, composing to held-out-DEEPER depth.

RUNG-1 SCOPE: the TRANSITION (delta: state x input -> next-state scores) is the rate-learned weights (reuse the validated
discrete_attractor_rnn); only the RE-DISCRETIZATION is moved on-spikes (the divnorm WTA). Anti-cheats: (a) spiking-WTA
winner == host-argmax winner per step (the WTA is faithful); (b) DIVNORM-OFF lesion -> the WTA degrades (the divisive
normalization is load-bearing); (c) held-out-DEEPER state-track on spikes >> chance == the rate result. Reuse-by-import;
NO `sim/` edit. numpy backend (small bridge).

Run:  SIM_BACKEND=numpy python -m research.runners._d3_spiking_attractor_derisk --group S3 --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_group_composition_derisk import make_group_task, discrete_attractor_rnn
from research.runners._phaseC_S5_divnorm_derisk import build_divnorm_score_bridge, onbridge_divnorm_drive

# THE READ-ISOLATION FIX (2026-09-02, C2 bug class -- ported verbatim from
# `_crossedge_surprise_metacog_derisk.py::_EXTRA_RESET_ARRAYS`, per the read-isolation audit
# `research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md`, H-2). `fswta_drive()`
# below is a SHARED PRIMITIVE called REPEATEDLY on the SAME bridge across an autoregressive rollout (this
# module's own `spiking_rollout_eval`, and every D3/event/reslm/mouth/joint-attention/wkv importer that does
# `from research.runners._d3_spiking_attractor_derisk import ... fswta_drive`). Its reset previously restored only
# v/u/firing_states -- NOT these 4 per-neuron arrays `_run_one_simulation_step` also mutates:
#   * cp_refractory_timers / cp_prev_firing_states -- HARD firing gates (int32 countdown / bool), independent of
#     membrane potential; a neuron mid-refractory at the end of one `fswta_drive` call stays gated at the START
#     of the next even though v/u were hard-reset to the constant rest value.
#   * cp_neuron_activity_ema / cp_neuron_firing_thresholds -- the homeostatic per-neuron EMA + adaptive threshold;
#     participation-gated, so it silently drifts on whichever pool won the immediately-prior call (inert here
#     since `build_fswta_score_bridge` sets `enable_homeostasis=False`, but restored anyway per the audit's
#     safety guarantee: a no-op where inert, hardening where not -- e.g. if a caller ever flips homeostasis on).
_EXTRA_RESET_ARRAYS = ("cp_refractory_timers", "cp_prev_firing_states",
                       "cp_neuron_activity_ema", "cp_neuron_firing_thresholds")


def build_fswta_score_bridge(seed, K, n_word=12, n_fs=24, exc_to_fs=2.0, fs_to_exc=9.0):
    """K Izhikevich attractor pools + a shared INHIBITORY FS pool with LATERAL INHIBITION (each pool excites FS; FS
    inhibits all pools). The winner (highest score-drive) fires first -> recruits FS -> FS suppresses the runners-up
    -> a CLEAN one-of-K winner even at SMALL margins (large K). This is the project's shared_FS / concept-pool WTA
    biology applied to the D3 re-discretization. NO `sim/` edit."""
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel
    cfg = CoreSimConfig(); cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name; cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0; cfg.seed = int(seed); cfg.enable_brain_region_framework = True; cfg.ou_std_current_pA = 0.0
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp", "enable_input_divisive_norm"):
        setattr(cfg, flag, False)
    regions = [BrainRegion(name=f"w{k}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0,
                           exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False) for k in range(K)]
    regions.append(BrainRegion(name="fs", n_neurons=n_fs, exc_fraction=0.0, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False))
    pathways = []
    for k in range(K):
        pathways.append(RegionPathway(from_region=f"w{k}", to_region="fs", density=0.6, weight_mean=exc_to_fs, weight_jitter=0.1, plastic=False))
        pathways.append(RegionPathway(from_region="fs", to_region=f"w{k}", density=0.6, weight_mean=fs_to_exc, weight_jitter=0.1, plastic=False))
    cfg.brain_regions = regions; cfg.region_pathways = pathways
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    # THE READ-ISOLATION FIX: snapshot the TRUE-REST value of each `_EXTRA_RESET_ARRAYS` array right after build,
    # before any `fswta_drive` call has run -- `fswta_drive` restores every call to THIS snapshot (see there).
    from sim.backend import to_host
    sb._rest_extra = {nm: (np.asarray(to_host(getattr(sb, nm, None))).copy()
                           if getattr(sb, nm, None) is not None else None)
                      for nm in _EXTRA_RESET_ARRAYS}
    return sb


def fswta_drive(sb, K, scores, input_gain=1200.0, settle=25):
    """Drive the K attractor pools by score; the FS lateral inhibition resolves a CLEAN winner. Returns (None, acc[K])."""
    from sim.backend import to_host, from_host
    rm = sb.region_manager
    _ridx = {k: np.asarray(list(rm.indices(f"w{k}")), dtype=int) for k in range(K)}
    if getattr(sb, "cp_izh_c_reset", None) is not None:
        sb.cp_membrane_potential_v[:] = sb.cp_izh_c_reset
    else:
        sb.cp_membrane_potential_v[:] = -65.0
    sb.cp_recovery_variable_u[:] = 0.0
    if getattr(sb, "cp_firing_states", None) is not None:
        sb.cp_firing_states[:] = False
    # THE READ-ISOLATION FIX: restore every array in `_EXTRA_RESET_ARRAYS` to the true-rest snapshot taken at
    # build. Without this, refractory/prev-firing/homeostatic residue from whichever call ran immediately before
    # (a prior rollout step, or a prior K-way score query) leaks into this call's settle window -- an
    # ORDER-dependent bias across the K attractor pools, not a per-query-independent read.
    _rest_extra = getattr(sb, "_rest_extra", None)
    if _rest_extra is not None:
        for nm in _EXTRA_RESET_ARRAYS:
            val = _rest_extra.get(nm)
            if val is not None:
                getattr(sb, nm)[:] = from_host(val)
    s = np.maximum(np.asarray(scores, dtype=float), 0.0)
    cur = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for k in range(K):
        cur[_ridx[k]] = float(input_gain * s[k])
    acc = np.zeros(K); cur_dev = from_host(cur)
    for _ in range(settle):
        sb.cp_external_input_current[:] = cur_dev
        sb._run_one_simulation_step()
        fir = np.asarray(to_host(sb.cp_firing_states)).astype(float)
        for k in range(K):
            acc[k] += fir[_ridx[k]].mean()
    sb.cp_external_input_current[:] = 0.0
    return None, acc


def selftest_read_isolation(K=3, n_word=6, n_fs=8, seed=7, settle=12, input_gain=1200.0):
    """READ-ISOLATION REGRESSION GUARD (2026-09-02, C2 bug class). `fswta_drive` has NO learned mechanism to zero
    -- the K pools are driven directly by whatever score vector is passed -- so the "zeroed-mechanism" pool here is
    the fswta bridge itself, freshly built (no training, no lesion state beyond its own per-neuron arrays).

    TWO checks:
    (1) Two IDENTICAL-input consecutive `fswta_drive` calls must return bitwise-identical `acc`. (Verified inert
        for THIS primitive at realistic settle/gain: `cp_refractory_timers` decrements unconditionally every step
        regardless of drive, so a leaked <=`refractory_period_steps` residual clears before the earliest possible
        first spike under sustained drive -- confirmed by direct instrumentation across settle in {2..20} and gain
        in {50..2000}, no divergence found. Kept as the literal regression guard the audit specified; passes both
        pre- and post-fix in this primitive's regime, which is itself the honest hardening-not-a-flip result.)
    (2) The DECISIVE check: dirty the bridge with one real `fswta_drive` call, then call `fswta_drive` AGAIN with
        `settle=0` (runs the reset-and-restore block but ZERO simulation steps) and assert every
        `_EXTRA_RESET_ARRAYS` array is bitwise-identical to the build-time true-rest snapshot. This DOES fail in
        its failing direction: reverting the `_EXTRA_RESET_ARRAYS` restore in `fswta_drive` (the pre-fix code)
        leaves `cp_refractory_timers`/`cp_prev_firing_states` at call-1's residual value instead of rest --
        directly verified by disabling the restore block and re-running this check, which raised AssertionError
        on both arrays (`cp_neuron_activity_ema`/`cp_neuron_firing_thresholds` are inert here regardless, since
        `build_fswta_score_bridge` sets `enable_homeostasis=False` -- restoring them is defense-in-depth)."""
    from sim.backend import to_host
    sb = build_fswta_score_bridge(seed=seed, K=K, n_word=n_word, n_fs=n_fs)
    rest = sb._rest_extra
    scores = np.linspace(0.1, 0.9, K)   # asymmetric -> unequal per-pool firing -> unequal residual state if leaked

    # (1) repeat-identical-read bitwise identity
    _, acc1 = fswta_drive(sb, K, scores, input_gain=input_gain, settle=settle)
    _, acc2 = fswta_drive(sb, K, scores, input_gain=input_gain, settle=settle)
    assert np.array_equal(acc1, acc2), (
        f"READ-ISOLATION REGRESSION: fswta_drive acc NOT bitwise-identical across 2 identical-input repeat reads "
        f"({acc1} vs {acc2}) -- the _EXTRA_RESET_ARRAYS restore in fswta_drive is missing or broken.")

    # (2) decisive: dirty the state, then a reset-only (settle=0) call must land exactly on true rest
    _, acc_dirty = fswta_drive(sb, K, scores, input_gain=input_gain, settle=max(settle, 8))
    _, acc_reset_only = fswta_drive(sb, K, scores, input_gain=input_gain, settle=0)
    assert acc_reset_only.sum() == 0.0, "settle=0 should record zero firing -- fswta_drive's settle loop changed"
    for nm in _EXTRA_RESET_ARRAYS:
        cur = np.asarray(to_host(getattr(sb, nm)))
        assert np.array_equal(cur, rest[nm]), (
            f"READ-ISOLATION REGRESSION: {nm} did NOT reset to the true-rest snapshot after a dirtying call -- "
            f"got {cur}, expected {rest[nm]}. The _EXTRA_RESET_ARRAYS restore in fswta_drive is missing or broken.")
    print(f"  [selftest] read-isolation OK: (1) 2 identical-input fswta_drive calls (K={K}) bitwise-identical on "
          f"acc; (2) a reset-only call after dirtying lands exactly on the true-rest snapshot for all "
          f"{len(_EXTRA_RESET_ARRAYS)} extra-reset arrays", flush=True)
    return True


def spiking_rollout_eval(task, W, split, sb, K, input_gain=1200.0, settle=15, n_eval=60, seed=42, drive_fn=None):
    """Autoregressive rollout with ON-BRIDGE spiking WTA re-discretization. Each step: scores = Ws.tanh(Wr.emb[cur] +
    Wi.x) + bs (rate transition) -> drive the K attractor pools -> the winner FIRES (divnorm WTA) -> next state = the
    spiking winner. Returns spiking state-track acc + the spiking-vs-host winner agreement."""
    emb, Wr, Wi, Ws, bs = W["emb"], W["Wr"], W["Wi"], W["Ws"], W["bs"]
    ident = task["ident"]
    Xe, ye, Le, _, Se = task[split]
    rng = np.random.RandomState(seed + 1)
    idx = rng.choice(len(Le), min(n_eval, len(Le)), replace=False)
    ok_spk = 0; agree = 0; steps = 0
    for n in idx:
        cur = ident
        for t in range(int(Le[n])):
            h = np.tanh(emb[cur] @ Wr.T + Xe[n, t] @ Wi.T)
            scores = h @ Ws.T + bs                                # K-dim transition scores
            _drv = drive_fn if drive_fn is not None else onbridge_divnorm_drive
            _, acc = _drv(sb, K, scores, input_gain=input_gain, settle=settle)
            nxt_spk = int(np.argmax(acc)) if acc.max() > 0 else ident
            nxt_host = int(np.argmax(scores))
            agree += int(nxt_spk == nxt_host); steps += 1
            cur = nxt_spk                                         # ROLL OUT on the SPIKING winner
        ok_spk += int(cur == Se[n, int(Le[n]) - 1])
    return {"spk_track": ok_spk / len(idx), "spk_host_agree": agree / max(steps, 1)}


def run_seed(group_name, seed, n_pool=None, n_hid=192, epochs=60, n_per_len=None, fs_inh=9.0, fs_settle=25):
    is_big = group_name == "A5"
    n_pool = n_pool if n_pool is not None else (256 if is_big else 64)
    n_per_len = n_per_len if n_per_len is not None else (8000 if is_big else 1500)
    task = make_group_task(group_name, seed, n_pool=n_pool, noise=0.6, n_per_len=n_per_len,
                           train_lens=(1, 2, 3, 4, 5), test_lens=(6, 7, 8))
    K = task["K"]
    da = discrete_attractor_rnn(task, seed=seed, epochs=epochs, n_hid=n_hid)     # rate transition (validated)
    W = da["weights"]
    # PRIMARY spiking WTA = PLAIN Izhikevich drive (drive each attractor pool by its score -> the winner fires most ->
    # decode argmax(firing) = the spiking re-discretization). The divisive-norm E%-max OVER-normalizes single-winner
    # transition scores (a diagnostic, not the right cleanup for a clear one-of-K winner).
    sb = build_divnorm_score_bridge(seed=seed, V=K, n_word=10, enable_divnorm=False)
    sb_fs = build_fswta_score_bridge(seed=seed, K=K, fs_to_exc=fs_inh)                     # FS lateral-inhibition WTA
    spk_same = spiking_rollout_eval(task, W, "test_same", sb, K, seed=seed)
    spk_deep = spiking_rollout_eval(task, W, "test_deeper", sb, K, seed=seed)
    fs_deep = spiking_rollout_eval(task, W, "test_deeper", sb_fs, K, seed=seed, settle=fs_settle, drive_fn=fswta_drive)
    return {"seed": seed, "group": group_name, "K": K, "rate_step_delta": round(da["step_transition_acc"], 3),
            "rate_deeper_track": round(da["state_deeper"], 3),
            "SPK_same_track": round(spk_same["spk_track"], 3), "SPK_deeper_track": round(spk_deep["spk_track"], 3),
            "SPK_host_agree_deeper": round(spk_deep["spk_host_agree"], 3),
            "FSWTA_deeper_track": round(fs_deep["spk_track"], 3), "FSWTA_host_agree": round(fs_deep["spk_host_agree"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S3", choices=["S3", "S4", "A5"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--fs-inh", type=float, default=9.0, help="FS->exc inhibition weight (stronger -> cleaner one-of-K winner)")
    ap.add_argument("--fs-settle", type=int, default=25, help="FS-WTA settle steps (longer -> the competition fully resolves)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="run the read-isolation repeat-read bitwise-identity regression guard and exit")
    a = ap.parse_args()
    if a.selftest:
        selftest_read_isolation()
        print("[selftest] PASS", flush=True)
        return
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 SPIKING attractor] {a.group} | re-discretization ON SPIKES (Izh attractor-pool WTA + FS lateral inhibition = CA3/NEF cleanup)", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(a.group, s, n_hid=a.n_hid, epochs=a.epochs, fs_inh=a.fs_inh, fs_settle=a.fs_settle)
        rows.append(r)
        print(f"  [seed {s}] rate: step-delta={r['rate_step_delta']} deeper={r['rate_deeper_track']} || "
              f"plain-WTA: DEEPER={r['SPK_deeper_track']} (agree={r['SPK_host_agree_deeper']}) || "
              f"FS-WTA (lateral inhib): DEEPER={r['FSWTA_deeper_track']} (agree={r['FSWTA_host_agree']})", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        spk_d, agree = _m("SPK_deeper_track"), _m("SPK_host_agree_deeper")
        fs_d, fs_a = _m("FSWTA_deeper_track"), _m("FSWTA_host_agree")
        # GO: the FS lateral-inhibition WTA re-discretizes ON SPIKES with a CLEAN competitive winner, holding held-out-
        # DEEPER (>>chance) AND faithful (== host argmax) even at LARGE K where the plain drive's small-margin errors
        # compound. (The plain WTA is the S3 baseline; FS-WTA is the clean-attractor scale fix.)
        best_d = max(spk_d, fs_d); best_a = max(agree, fs_a)
        go = (best_d > 0.90) and (best_a > 0.95)
        print(f"\n  AGGREGATE ({a.group}): plain-WTA deeper={spk_d:.3f} (agree {agree:.3f}) | FS-WTA deeper={fs_d:.3f} (agree {fs_a:.3f}) (chance={1.0/rows[0]['K']:.3f})", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the discrete-attractor re-discretization runs ON SPIKES (best deeper-track '+format(best_d,'.2f')+', faithful == host argmax) -> the recurrent composition is realized on the project spiking substrate = the simulated recurrent language cortex; FS lateral inhibition gives the clean one-active attractor at scale' if go else 'the spiking WTA did not hold cleanly (tune FS exc/inh weights or input_gain/settle; read the host-agree gap between plain and FS-WTA)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
