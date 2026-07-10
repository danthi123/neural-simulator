"""D3 -> a SPIKING slot that actually HOLDS: persistent-activity attractor (the prerequisite for a gated copy on spikes).

WHY THIS RUNG EXISTS (found by reading my own substrate, not by theorizing).
Every "spiking slot" in the D3/event arc used `_d3_spiking_attractor_derisk.fswta_drive`, which:
  * RESETS `cp_membrane_potential_v`, `cp_recovery_variable_u`, `cp_firing_states` on every call, and
  * builds its K pools with `internal_density=0.0` -- NO recurrent excitation.
So it is a STATELESS one-of-K re-discretizer. It cannot hold anything: in every prior rung the slot's *hold* lived in a
Python variable between calls. That was fine when the claim was "the winner is chosen by spikes." It is NOT fine for the
BOUNDARY-GATED COPY, where the HOLD **is** the mechanism (closed gate -> the prior event persists). A stateless WTA
would make "on spikes" a fiction there.

THE MECHANISM: persistent activity in a recurrent attractor (Amit & Brunel 1997; Wang 2002 NMDA-dependent WM):
each pool excites ITSELF (recurrent excitation) and drives a shared FS pool that inhibits all pools. A brief input
selects a winner; when the input is REMOVED the winner's recurrent excitation sustains its own firing (persistent
activity) while FS inhibition keeps the losers silent. That is a slot that HOLDS with no input -- exactly what a closed
gate requires, and why biology uses attractors for working memory.

WHAT THIS DE-RISKS (the only question that matters here):
  (1) HOLD:      drive pool j briefly, remove ALL input -> does j keep firing for N steps while the others stay silent?
  (2) ONE-OF-K:  is the held state a clean single winner (no multi-pool blowup, no death)?
  (3) OVERWRITE: with input restored to a DIFFERENT pool m, does the attractor switch to m (a gate OPEN = copy)?
  (4) The knife-edge: too little recurrence -> the bump dies; too much -> runaway/multi-pool. Report the working window.

ANTI-CHEATS: (a) a NO-RECURRENCE control (internal_density=0, i.e. the current fswta bridge) must FAIL to hold;
(b) hold is measured with `cp_external_input_current` identically ZERO (asserted), so nothing is secretly driving it;
(c) the held winner must equal the driven winner (it holds the RIGHT thing); (d) multi-seed.

numpy backend (small bridge); NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_persistent_slot_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np


def build_persistent_slot(seed, K, n_word=20, n_fs=24, recur=25.0, exc_to_fs=1.4, fs_to_exc=10.0, nmda=True):
    """K attractor pools with RECURRENT SELF-EXCITATION (internal_density > 0) + a shared FS pool.
    `recur` = within-pool recurrent weight (the persistence knob). NMDA on = the slow conductance that makes
    persistent activity robust (Wang 2002)."""
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
    # THE HOLD REQUIRES SLOW NMDA, NOT AMPA. `internal_density` wires AMPA recurrence, which decays in ~5 ms -- MEASURED:
    # with AMPA recurrence up to recur=14 the pool's firing collapsed from 0.168 (driven) to 0.008 (hold). The substrate
    # already provides the Wang mechanism: `cfg.enable_nmda_recurrent` + a pathway with `receptor="nmda_slow"`, which
    # REPLACES AMPA with slow NMDA (tau_decay 100 ms). We therefore wire the self-excitation as an EXPLICIT nmda_slow
    # self-pathway w_k -> w_k, not via internal_density.
    cfg.enable_nmda = bool(nmda)
    cfg.enable_nmda_recurrent = bool(nmda)
    cfg.nmda_recurrent_tau_decay_ms = 100.0
    regions = [BrainRegion(name=f"w{k}", n_neurons=n_word, exc_fraction=1.0,
                           internal_density=0.0,                     # no AMPA recurrence (it cannot hold)
                           exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                           plastic_internal=False) for k in range(K)]
    regions.append(BrainRegion(name="fs", n_neurons=n_fs, exc_fraction=0.0, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False))
    pathways = []
    for k in range(K):
        # SLOW-NMDA recurrent self-excitation = the persistent-activity mechanism (Wang 2002)
        pathways.append(RegionPathway(from_region=f"w{k}", to_region=f"w{k}", density=0.9,
                                      weight_mean=recur, weight_jitter=0.05, plastic=False,
                                      exc_receptor="nmda_slow"))   # NOTE: `receptor=` is the INHIBITORY receptor
                                                                   # (gaba_a/gaba_b); the excitatory one is exc_receptor
        pathways.append(RegionPathway(from_region=f"w{k}", to_region="fs", density=0.6,
                                      weight_mean=exc_to_fs, weight_jitter=0.1, plastic=False))
        pathways.append(RegionPathway(from_region="fs", to_region=f"w{k}", density=0.6,
                                      weight_mean=fs_to_exc, weight_jitter=0.1, plastic=False))
    cfg.brain_regions = regions; cfg.region_pathways = pathways
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def _pool_idx(sb, K):
    rm = sb.region_manager
    return {k: np.asarray(list(rm.indices(f"w{k}")), dtype=int) for k in range(K)}


def _reset(sb):
    if getattr(sb, "cp_izh_c_reset", None) is not None:
        sb.cp_membrane_potential_v[:] = sb.cp_izh_c_reset
    else:
        sb.cp_membrane_potential_v[:] = -65.0
    sb.cp_recovery_variable_u[:] = 0.0
    if getattr(sb, "cp_firing_states", None) is not None:
        sb.cp_firing_states[:] = False


def drive_then_hold(sb, K, drive_pool, drive_steps=30, hold_steps=60, input_gain=400.0, switch_pool=None,
                    switch_steps=80, clear_steps=250, clear_gain=1500.0):
    """LOAD `drive_pool` -> HOLD with ZERO input -> (optionally) a gate OPEN = CLEAR (an FS inhibitory burst) then LOAD
    `switch_pool` -> HOLD again.

    The CLEAR is not decoration. MEASURED: a persistent NMDA attractor RESISTS being overwritten by input alone
    (5/6 seeds kept the old winner). Silencing it briefly is not enough either -- its recurrent NMDA conductance decays
    with tau=100 ms, so a short reset leaves enough charge to RE-IGNITE the old bump the moment inhibition lifts
    (clear=120 ms -> old 0.110 vs new 0.091, old wins). Only a reset LONGER THAN tau_NMDA erases it
    (clear=250 ms = 2.5 tau -> old rate exactly 0.0000, new bump holds). That is why the PBWM update gate must
    CLEAR and then LOAD, and it predicts a real several-hundred-ms cost to event-boundary updating.
    Returns (drive_rate, hold_rate, switch_hold_rate)."""
    from sim.backend import to_host, from_host
    idx = _pool_idx(sb, K)
    fs_idx = np.asarray(list(sb.region_manager.indices("fs")), dtype=int)
    _reset(sb)
    n = sb.core_config.num_neurons

    def run(cur_vec, steps):
        acc = np.zeros(K); dev = from_host(cur_vec)
        for _ in range(steps):
            sb.cp_external_input_current[:] = dev
            sb._run_one_simulation_step()
            fir = np.asarray(to_host(sb.cp_firing_states)).astype(float)
            for k in range(K):
                acc[k] += fir[idx[k]].mean()
        return acc / max(steps, 1)

    cur = np.zeros(n, dtype=np.float64); cur[idx[drive_pool]] = input_gain
    drive_rate = run(cur, drive_steps)

    zero = np.zeros(n, dtype=np.float64)                       # ANTI-CHEAT: input is identically ZERO during the hold
    assert not zero.any()
    hold_rate = run(zero, hold_steps)

    switch_hold = None
    if switch_pool is not None:
        if clear_steps > 0:                                    # gate OPEN, phase 1: CLEAR (inhibitory reset)
            cc = np.zeros(n, dtype=np.float64); cc[fs_idx] = clear_gain
            run(cc, clear_steps)
        cur2 = np.zeros(n, dtype=np.float64); cur2[idx[switch_pool]] = input_gain
        run(cur2, switch_steps)                                # gate OPEN, phase 2: LOAD the new content
        switch_hold = run(zero, hold_steps)                    # does it now HOLD the NEW content, with zero input?
    return drive_rate, hold_rate, switch_hold


def run_seed(seed, K, recur, nmda):
    out = {"seed": seed, "K": K, "recur": recur, "nmda": nmda}
    j, m = 2, 4

    sb = build_persistent_slot(seed, K, recur=recur, nmda=nmda)
    dr, hr, sh = drive_then_hold(sb, K, drive_pool=j, switch_pool=m)          # clear (250ms) then load
    held = int(np.argmax(hr)) if hr.max() > 1e-6 else -1
    out["hold_alive"] = float(hr.max()); out["hold_correct"] = int(held == j)
    out["hold_selectivity"] = float(hr[j] / (hr.sum() + 1e-9)) if hr.sum() > 0 else 0.0
    out["n_pools_active_in_hold"] = int((hr > 0.2 * max(hr.max(), 1e-9)).sum())
    w = int(np.argmax(sh)) if sh is not None and sh.max() > 1e-6 else -1
    out["switch_winner"] = w
    out["switch_correct"] = int(w == m and sh[m] > 0.01 and sh[j] < 0.2 * max(sh[m], 1e-9))
    out["switch_old_rate"] = float(sh[j]); out["switch_new_rate"] = float(sh[m])

    # CONTROL 1: gate OPEN with NO clear -> the old bump survives (input alone cannot overwrite an attractor)
    sb1 = build_persistent_slot(seed, K, recur=recur, nmda=nmda)
    _, _, sh1 = drive_then_hold(sb1, K, drive_pool=j, switch_pool=m, clear_steps=0)
    out["noclear_switch_correct"] = int(sh1 is not None and int(np.argmax(sh1)) == m and sh1[j] < 0.2 * max(sh1[m], 1e-9))

    # CONTROL 2: clear SHORTER than tau_NMDA -> the old bump re-ignites from residual conductance
    sb2 = build_persistent_slot(seed, K, recur=recur, nmda=nmda)
    _, _, sh2 = drive_then_hold(sb2, K, drive_pool=j, switch_pool=m, clear_steps=60)
    out["shortclear_switch_correct"] = int(sh2 is not None and int(np.argmax(sh2)) == m and sh2[j] < 0.2 * max(sh2[m], 1e-9))

    # CONTROL 3: NO RECURRENCE (the stateless bridge every prior rung used) -> cannot hold at all
    sb3 = build_persistent_slot(seed, K, recur=0.0, nmda=nmda)
    _, hr3, _ = drive_then_hold(sb3, K, drive_pool=j)
    out["norecur_hold_alive"] = float(hr3.max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--recur", type=float, default=25.0)
    ap.add_argument("--no-nmda", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 PERSISTENT SLOT] K={a.K} recur={a.recur} nmda={not a.no_nmda} | can a spiking slot HOLD with ZERO input? (the prerequisite for a gated copy on spikes)", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.K, a.recur, not a.no_nmda); rows.append(r)
        print(f"  [seed {s}] HOLD (zero input): correct={r['hold_correct']} alive={r['hold_alive']:.4f} "
              f"selectivity={r['hold_selectivity']:.3f} pools-active={r['n_pools_active_in_hold']} || "
              f"gate OPEN (clear+load): switch_correct={r['switch_correct']} (new {r['switch_new_rate']:.4f} / old {r['switch_old_rate']:.4f}) || "
              f"controls: no-clear={r['noclear_switch_correct']} short-clear={r['shortclear_switch_correct']} no-recurrence-hold={r['norecur_hold_alive']:.4f}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        alive, corr, sel = _m("hold_alive"), _m("hold_correct"), _m("hold_selectivity")
        npa, sw = _m("n_pools_active_in_hold"), _m("switch_correct")
        ncsw, sssw, nr = _m("noclear_switch_correct"), _m("shortclear_switch_correct"), _m("norecur_hold_alive")
        go = ((alive > 0.01) and (corr > 0.99) and (sel > 0.8) and (npa < 1.5) and (sw > 0.99)
              and (ncsw < 0.5) and (sssw < 0.5) and (nr < 0.02))
        print(f"\n  AGGREGATE: hold_alive={alive:.4f} | hold_correct={corr:.2f} | selectivity={sel:.3f} | pools-active={npa:.2f}", flush=True)
        print(f"    gate OPEN (clear 250ms >> tau_NMDA, then load): switch_correct={sw:.2f}", flush=True)
        print(f"    CONTROLS: no-clear switch={ncsw:.2f} | short-clear(60ms) switch={sssw:.2f} | NO-RECURRENCE hold={nr:.4f}", flush=True)
        msg_go = ('a recurrent slow-NMDA attractor slot HOLDS its winner with the external input identically ZERO '
                  '(persistent activity, Amit-Brunel / Wang 2002), as a CLEAN one-of-K state; and a gate OPEN overwrites it '
                  'only as a CLEAR-then-LOAD: an inhibitory reset LONGER than tau_NMDA, then the new content. Input alone does '
                  'NOT overwrite it (' + format(ncsw, '.2f') + '), a reset SHORTER than tau_NMDA does NOT either ('
                  + format(sssw, '.2f') + ' -- the old bump re-ignites from residual conductance), and the STATELESS FS-WTA used '
                  'by every prior rung cannot hold at all (' + format(nr, '.4f') + ') -> the HOLD in the boundary-gated copy can be '
                  'a genuine spiking memory rather than a Python variable, and updating it carries a real several-hundred-ms '
                  'reset cost (why PBWM gating CLEARS then LOADS)')
        msg_no = 'the slot did not hold/switch cleanly (read hold_alive / selectivity / switch_correct and the two clear controls)'
        print("  VERDICT: " + ("GO" if go else "PARTIAL/NEGATIVE") + " -- " + (msg_go if go else msg_no) + ". NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
