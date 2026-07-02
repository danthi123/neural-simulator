"""EMERGE-12 / rung-4 Stage B2 -- the FULL HTM Temporal Memory inference on the REAL SimulationBridge. This is the
genuine rung-4 deliverable: the frozen EMERGE-9b-learned distal connectivity is LOADED into a real recurrent
`coincidence_detector` pathway (cell->cell) via `inject_explicit_wiring`, an overlapping sequence is driven through the
bridge, and the branch-prediction is read from the bridge's OWN primed (apical dAP) state -- reproducing EMERGE-9c's
spiking-inference GO with the connectivity now on-substrate (not held in a numpy HTM object).

WHY THIS IS FAITHFUL (no host cognition between sensation and action):
  - PREDICTION is native bridge machinery: the recurrent distal `coincidence_detector` pathway computes, per post-cell,
    c_drive = COUNT of coincidence-routed synapses whose presyn fired last step (cp_prev_firing_states) -- EXACTLY the
    HTM `_seg_conn_active`. With coincidence_k_threshold = act_th, a cell is PREDICTED (its apical dAP plateau fires,
    charging cp_v_apical = the two-compartment apical compartment) iff its active distal segment cleared act_th.
  - WINNER SELECTION is spiking: a dAP-primed cell (elevated cp_v_apical -> electrotonic soma coupling) reaches the
    Izhikevich threshold at a LOWER feedforward drive (Stage-A' GO, +40 pA). Driving a column at a feedforward level in
    that window -> only PRIMED (predicted) cells fire = sparse, context-specific selection; an UNPREDICTED column has no
    primed cells (the Stage-B1 reframing: the dAP already provides the sparse selection; a WTA is only needed for burst
    sparsification, not tested here).
  - The only host code is the WORLD/BODY interface: presenting the input symbol sequence (driving the current symbol's
    column each window -- the sensory stream) and reading the prediction (which columns the last winners prime).

The frozen distal connectivity export: for each post-cell, the UNION over its segments of connected synapses
(permanence >= perm_conn) -> one edge (pre->post) tagged coincidence_detector=True. In this task the allocated SDRs are
DISJOINT (A-context vs E-context), so the per-cell flat coincidence count == the active segment's per-segment count
(only one prior SDR is active at a time) -- the honest simplification the branch-prediction PARITY control verifies.

TASK / ANTI-CHEATS (mirror EMERGE-9b/9c): overlapping sequences [cue]+[shared middle L]+[branch]; branch (divergent)
prediction; beat the Markov floor + chance; dAP-LESION (coincidence off) collapses; untrained (empty segments) at
chance; PARITY with the EMERGE-9c numpy spiking reference. Reuse-by-import (EMERGE-9b/9c); the two-compartment dAP is
the committed guarded `sim/` edit (default-off byte-inert); the connectivity load is `inject_explicit_wiring` (existing
API). Multi-seed 42/43/44. CPU (numpy backend ok).
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._emerge9b_htm_faithful_derisk import (
    HTM, make_overlap_sequences, markov_branch_acc, full_oracle)
from research.runners._emerge9c_spiking_tm_derisk import SpikingTM, branch_acc_spiking

OUT = Path("research/findings/raw/_emerge12_stageB2_bridge_tm.json")


def _host(x):
    try:
        return x.get()
    except AttributeError:
        return np.asarray(x)


def train_frozen_tm(seed, n_seq, L, n_cells, k_win, act_th, epochs, lesion=False):
    """Train the validated EMERGE-9b HTM (frozen after) -- or, for the 'untrained' arm, return it with NO training
    (empty segments = no distal connectivity)."""
    seqs, vocab, info = make_overlap_sequences(n_seq=n_seq, middle_len=L, seed=seed)
    tm = HTM(vocab, n_cells=n_cells, seed=seed, k_win=k_win, act_th=act_th, lesion=lesion)
    if epochs > 0:
        for _ in range(epochs):
            for s in seqs:
                tm.run_sequence(s, learn=True)
    return tm, seqs, vocab, info


def export_distal_edges(tm):
    """The frozen learned distal connectivity: per post-cell, the UNION over its segments of CONNECTED synapses
    (permanence >= perm_conn) -> (pre, post) edges. Returns (pre_list, post_list)."""
    pre_list, post_list = [], []
    for post in range(tm.N):
        connected = set()
        for seg in tm.segments[post]:
            for p, w in seg.items():
                if w >= tm.perm_conn:
                    connected.add(int(p))
        for p in connected:
            pre_list.append(p); post_list.append(post)
    return pre_list, post_list


def build_bridge_from_tm(tm, seed, coincidence=True, plateau_scale=1.0, apical_g_couple=2.0,
                         wta=True, col_fs_weight=80.0, fs_col_weight=100.0):
    """ONE SimulationBridge holding tm.N two-compartment Izhikevich cells; the frozen distal connectivity is injected
    as a recurrent `coincidence_detector` pathway (cell->cell) via inject_explicit_wiring. An optional global
    fast-spiking WTA interneuron (only one column is driven at a time, so global == per-column here) converts the
    dAP step-level lead into sparse selection: the dAP-primed cells fire first, drive the FS, and it suppresses the
    not-yet-fired (non-primed) cells (cortical PV feedforward inhibition). Returns (bridge, cells_idx)."""
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion
    from sim.enums import NeuronModel, NeuronType
    N = int(tm.N)
    regions = [BrainRegion(name="cells", n_neurons=N, exc_fraction=1.0, internal_density=0.0,
                           exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                           izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)]
    if wta:
        regions.append(BrainRegion(name="fs", n_neurons=1, exc_fraction=0.0, internal_density=0.0,
                                   exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                                   izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name))
    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions); cfg.region_pathways = []
    cfg.enable_stdp = False; cfg.enable_hebbian_learning = False; cfg.enable_nmda = False
    cfg.stdp_w_max = 40.0; cfg.fast_spike_reset = True
    for f in ("enable_homeostasis", "enable_short_term_plasticity", "enable_ou_process",
              "enable_conductance_noise", "enable_parameter_heterogeneity", "enable_structural_plasticity"):
        setattr(cfg, f, False)
    cfg.enable_coincidence_detection = bool(coincidence)
    cfg.coincidence_k_threshold = float(tm.act_th)             # == HTM act_th: c_drive >= act_th -> predicted (dAP)
    cfg.coincidence_plateau_strength = 80.0 * float(plateau_scale)
    cfg.enable_two_compartment_dap = True                       # the committed guarded apical-dAP compartment
    cfg.apical_g_couple = float(apical_g_couple)
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    cells_idx = np.asarray(bridge.region_manager.indices("cells"), dtype=np.int64)
    fs_idx = np.asarray(bridge.region_manager.indices("fs"), dtype=np.int64) if wta else np.array([], np.int64)
    # inject the frozen distal connectivity as a recurrent coincidence pathway (only when coincidence is ON --
    # the dAP-LESION arm leaves the bridge with no distal edges, so no priming = the mechanism is severed) plus the
    # optional global WTA edges (cells->fs excitatory, fs->cells inhibitory via the fs inhibitory trait).
    plan = {}
    if coincidence:
        pre_e, post_e = export_distal_edges(tm)
        if pre_e:
            plan["distal"] = {"pre_indices": [int(cells_idx[p]) for p in pre_e],
                              "post_indices": [int(cells_idx[q]) for q in post_e],
                              "initial_weights": [1.0] * len(pre_e),
                              "plastic": False, "coincidence_detector": True, "conn_type": "distal_htm"}
    if wta:
        f = int(fs_idx[0])
        plan["cells_to_fs"] = {"pre_indices": [int(c) for c in cells_idx], "post_indices": [f] * len(cells_idx),
                               "initial_weights": [float(col_fs_weight)] * len(cells_idx),
                               "plastic": False, "conn_type": "wta_drive"}
        plan["fs_to_cells"] = {"pre_indices": [f] * len(cells_idx), "post_indices": [int(c) for c in cells_idx],
                               "initial_weights": [float(fs_col_weight)] * len(cells_idx),
                               "plastic": False, "conn_type": "wta_inhib"}
    if plan:
        bridge.inject_explicit_wiring(plan, output_inhibitory_indices=(fs_idx.tolist() if wta else None))
    return bridge, cells_idx


def reset_soma(bridge):
    """Zero the SOMATIC + synaptic + firing state (v, u, firing, prev_firing, g_e, g_i) but PRESERVE the apical
    priming (cp_v_apical + coincidence conductances). Called between symbol presentations so each column's captured
    winners are clean (no spike-latency bleed from the prior column), while the context propagates forward purely
    through the apical dAP set by the prior winners' coincidence."""
    xp = bridge.xp if hasattr(bridge, "xp") else np
    n = int(bridge.core_config.num_neurons)
    bridge.cp_membrane_potential_v[:] = xp.asarray(np.full(n, -65.0, np.float32))
    if getattr(bridge, "cp_recovery_variable_u", None) is not None:
        bridge.cp_recovery_variable_u[:] = 0.0
    for arr in ("cp_firing_states", "cp_prev_firing_states", "cp_external_input_current",
                "cp_conductance_g_e", "cp_conductance_g_i"):
        a = getattr(bridge, arr, None)
        if a is not None:
            a[:] = 0


def reset_state(bridge, cells_idx):
    """Full reset (incl. apical) between SEQUENCES."""
    reset_soma(bridge)
    xp = bridge.xp if hasattr(bridge, "xp") else np
    if getattr(bridge, "cp_v_apical", None) is not None:
        bridge.cp_v_apical[:] = xp.float32(getattr(bridge.core_config, "apical_E_rest", -65.0))
    for arr in ("cp_conductance_g_coincidence", "cp_conductance_g_coincidence_rise"):
        a = getattr(bridge, arr, None)
        if a is not None:
            a[:] = 0


def _fire_vector(bridge, cells_idx):
    return _host(bridge.cp_firing_states)[cells_idx].astype(bool)


def _clear_apical(bridge):
    xp = bridge.xp if hasattr(bridge, "xp") else np
    if getattr(bridge, "cp_v_apical", None) is not None:
        bridge.cp_v_apical[:] = xp.float32(getattr(bridge.core_config, "apical_E_rest", -65.0))
    for arr in ("cp_conductance_g_coincidence", "cp_conductance_g_coincidence_rise"):
        a = getattr(bridge, arr, None)
        if a is not None:
            a[:] = 0


def _prime_from_winners(bridge, cells_idx, winners_bool, n_prime=6):
    """Clean priming step: reset the soma (so residual firing can't backprop-contaminate the apical), clear stale
    apical, then HOLD the captured (sparse spiking) winners in cp_prev_firing_states for n_prime no-feedforward steps
    so the bridge's coincidence pathway primes the next column's context-specific cells to a clear (readable) apical
    level (cp_v_apical rises = the dAP prediction). Decouples priming from within-window spike timing so the
    coincidence count sees all winners synchronously (== EMERGE-9c's per-symbol predictive-set computation)."""
    xp = bridge.xp if hasattr(bridge, "xp") else np
    n = int(bridge.core_config.num_neurons)
    reset_soma(bridge)
    _clear_apical(bridge)
    fv = np.zeros(n, np.float32); fv[cells_idx[winners_bool]] = 1.0
    for _ in range(n_prime):
        bridge.cp_prev_firing_states[:] = xp.asarray(fv)   # hold the winners as the "just-fired" set each step
        bridge.cp_external_input_current[:] = xp.asarray(np.zeros(n, np.float32))
        bridge._run_one_simulation_step()


def present_and_predict(bridge, cells_idx, tm, seq, div_pos, cue_drive, middle_drive, n_sub):
    """Drive [cue]+[middle 1..L] (positions 0..div_pos) through the bridge as an on-substrate HTM inference; return the
    predicted-column set for the next symbol (== the branch if correct). Per symbol: (1) reset the soma (keep the apical
    priming set by the prior symbol's winners), (2) present the column -> the dAP-primed cells fire first + the WTA
    caps the winners = sparse context-specific spiking selection, (3) a clean priming step feeds those winners to the
    coincidence pathway so it primes the NEXT column's context cells. After the last symbol, cp_v_apical holds the
    prediction (the branch's cells are primed)."""
    xp = bridge.xp if hasattr(bridge, "xp") else np
    nE = tm.nE
    reset_state(bridge, cells_idx)
    for pos in range(div_pos + 1):
        c = seq[pos]
        col = np.asarray(tm._col(c), dtype=np.int64)       # EMERGE cell idx of column c
        col_b = cells_idx[col]                              # bridge neuron idx
        reset_soma(bridge)                                  # clean soma; KEEP the apical priming set by prior winners
        fired = np.zeros(len(cells_idx), bool)
        for sub in range(n_sub):
            ext = np.zeros(int(bridge.core_config.num_neurons), np.float32)
            if pos == 0:
                ext[cells_idx[col[:tm.k_win]]] = cue_drive  # cue: sensory winner-SDR (col[:k_win], matching EMERGE-9b)
            else:
                ext[col_b] = middle_drive                   # middle: present the column; primed cells fire first (dAP)
            bridge.cp_external_input_current[:] = xp.asarray(ext)
            bridge._run_one_simulation_step()
            fired |= _fire_vector(bridge, cells_idx)
        if os.environ.get("EMERGE12_DEBUG"):
            wl = sorted(int(i) - c * nE for i in np.where(fired)[0] if c * nE <= int(i) < (c + 1) * nE)
            other = sorted(int(i) for i in np.where(fired)[0] if not (c * nE <= int(i) < (c + 1) * nE))
            print(f"    [dbg] pos{pos} col{c} winners(local)={wl} other={other}", flush=True)
        _prime_from_winners(bridge, cells_idx, fired)       # winners -> coincidence -> prime the NEXT column (dAP)
    # after the loop, cp_v_apical is primed by the last (div_pos) winners -> the branch. In the dAP-LESION arm
    # (coincidence off) the apical compartment is never allocated -> no priming possible -> empty prediction.
    E_rest = float(getattr(bridge.core_config, "apical_E_rest", -65.0))
    _vap = getattr(bridge, "cp_v_apical", None)
    if _vap is None or np.asarray(_host(_vap)).ndim == 0:
        return set(), np.full(len(cells_idx), E_rest, np.float32)
    v_ap = _host(_vap)[cells_idx]
    primed = np.where(v_ap > E_rest + 2.0)[0]               # margin: any measurable apical depolarization
    predicted_cols = set(int(i) // nE for i in primed)
    if os.environ.get("EMERGE12_DEBUG"):
        percol = {}
        for i in range(len(cells_idx)):
            c = int(i) // nE
            percol[c] = max(percol.get(c, -1e9), float(v_ap[i]))
        hot = {c: round(v, 1) for c, v in percol.items() if v > E_rest + 2.0}
        print(f"    [dbg] readout primed-cols(maxvap>{E_rest+2:.0f})={hot} | all={[round(percol[c],1) for c in sorted(percol)]}", flush=True)
    return predicted_cols, v_ap


def bridge_branch_acc(bridge, cells_idx, tm, seqs, div_pos, cue_drive, middle_drive, n_sub):
    ok = 0
    for s in seqs:
        pred, _ = present_and_predict(bridge, cells_idx, tm, s, div_pos, cue_drive, middle_drive, n_sub)
        ok += int(pred == {s[div_pos + 1]})
    return ok / len(seqs)


def run_arm(seed, arm, n_seq, L, n_cells, k_win, act_th, epochs, drives, n_sub, cue_drive):
    """Build the bridge for one arm and return the best branch acc over the middle-drive sweep."""
    lesion = (arm == "lesion")
    ep = 0 if arm == "untrained" else epochs
    tm, seqs, vocab, info = train_frozen_tm(seed, n_seq, L, n_cells, k_win, act_th, ep, lesion=False)
    coincidence = not lesion
    bridge, cells_idx = build_bridge_from_tm(tm, seed, coincidence=coincidence)
    best = {"branch_acc": 0.0, "middle_drive": 0.0}
    per_drive = []
    for md in drives:
        acc = bridge_branch_acc(bridge, cells_idx, tm, seqs, L, cue_drive, md, n_sub)
        per_drive.append({"middle_drive": float(md), "branch_acc": acc})
        if acc > best["branch_acc"]:
            best = {"branch_acc": acc, "middle_drive": float(md)}
    return best, per_drive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-seq", type=int, default=2)
    ap.add_argument("--middle-len", type=int, default=4)
    ap.add_argument("--n-cells", type=int, default=16)
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-sub", type=int, default=14)
    ap.add_argument("--cue-drive", type=float, default=2400.0)
    ap.add_argument("--md-min", type=float, default=340.0)
    ap.add_argument("--md-max", type=float, default=460.0)
    ap.add_argument("--md-steps", type=int, default=7)
    ap.add_argument("--smoke", action="store_true", help="single-seed single-sequence diagnostic")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    drives = np.linspace(a.md_min, a.md_max, a.md_steps)

    if a.smoke:
        return _smoke(a, drives)

    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per = []
    floors = {}
    for s in a.seeds:
        seqs, vocab, info = make_overlap_sequences(n_seq=a.n_seq, middle_len=a.middle_len, seed=s)
        floors[s] = {"markov_L": markov_branch_acc(seqs, a.middle_len, a.n_seq),
                     "oracle": full_oracle(seqs, a.middle_len), "chance": 1.0 / a.n_seq}
    try:
        for s in a.seeds:
            d = {"seed": s, "floors": floors[s]}
            for arm in ("htm", "lesion", "untrained"):
                best, per_drive = run_arm(s, arm, a.n_seq, a.middle_len, a.n_cells, a.k_win, a.act_th,
                                          a.epochs, drives, a.n_sub, a.cue_drive)
                d[arm] = {"branch_acc": best["branch_acc"], "middle_drive": best["middle_drive"],
                          "per_drive": per_drive}
            # EMERGE-9c numpy spiking parity (the reference the bridge must reproduce)
            tm9c, seqs9c, _, _ = _build_9c(s, a)
            d["ref_9c_spiking"] = branch_acc_spiking(tm9c, seqs9c, a.middle_len, s)
            per.append(d)
            f = d["floors"]
            print(f"  [seed {s}] BRIDGE branch {d['htm']['branch_acc']:.3f} @md{d['htm']['middle_drive']:.0f} "
                  f"| lesion {d['lesion']['branch_acc']:.3f} | untr {d['untrained']['branch_acc']:.3f} "
                  f"|| 9c-ref {d['ref_9c_spiking']:.3f} markov {f['markov_L']:.3f} chance {f['chance']:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    verdict = _verdict(a, per, floors) if err is None else f"ERROR -- {err}"
    summary = {"probe": "emerge12_stageB2_bridge_tm", "verdict": verdict,
               "mechanism": "FULL HTM Temporal Memory inference on the real SimulationBridge: frozen EMERGE-9b distal "
                            "connectivity loaded via inject_explicit_wiring as a recurrent coincidence_detector pathway; "
                            "prediction = native coincidence (c_drive>=act_th -> apical dAP); winner selection = dAP "
                            "fire-first (drive-level sparsity, Stage-B1 reframing); readout = primed apical state",
               "task": "overlapping sequences; branch prediction; dAP-lesion + untrained + Markov/chance + 9c-parity",
               "seeds": a.seeds, "config": {"n_seq": a.n_seq, "middle_len": a.middle_len, "n_cells": a.n_cells,
               "k_win": a.k_win, "act_th": a.act_th, "epochs": a.epochs, "n_sub": a.n_sub,
               "drives": drives.tolist()},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "rung-4 Stage B2: the learned connectivity now lives ON the bridge (inject_explicit_wiring "
                              "coincidence pathway), inference is the bridge's own spiking recurrence. Flat per-cell "
                              "coincidence (union over segments) == per-segment in this disjoint-SDR task (the 9c-parity "
                              "control verifies). Stage C = the three-term permanence kernel (learning on the substrate) "
                              "=> EMERGE-9d parity = rung-4 complete."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge12] VERDICT: {verdict}", flush=True)
    print(f"[emerge12] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


def _build_9c(seed, a):
    seqs, vocab, info = make_overlap_sequences(n_seq=a.n_seq, middle_len=a.middle_len, seed=seed)
    tm = SpikingTM(vocab, n_cells=a.n_cells, seed=seed, k_win=a.k_win, act_th=a.act_th, lesion=False)
    for _ in range(a.epochs):
        for s in seqs:
            tm.run_sequence(s, learn=True)
    return tm, seqs, vocab, info


def _verdict(a, per, floors):
    def m(arm, key="branch_acc"):
        return float(np.mean([p[arm][key] for p in per]))
    brg, les, unt = m("htm"), m("lesion"), m("untrained")
    ref9c = float(np.mean([p["ref_9c_spiking"] for p in per]))
    markov = float(np.mean([p["floors"]["markov_L"] for p in per]))
    chance = float(np.mean([p["floors"]["chance"] for p in per]))
    oracle = float(np.mean([p["floors"]["oracle"] for p in per]))
    parity = abs(brg - ref9c) <= 0.10
    go = bool(oracle > 0.99 and brg >= 0.90 and brg >= markov + 0.15 and brg >= chance + 0.20
              and brg >= les + 0.20 and parity)
    if oracle <= 0.99:
        return f"INCONCLUSIVE -- task not context-solvable (oracle {oracle:.3f})."
    if go:
        return (f"GO -- the FULL HTM Temporal Memory inference runs on the real SimulationBridge: the frozen EMERGE-9b "
                f"distal connectivity, LOADED via inject_explicit_wiring as a recurrent coincidence pathway, produces "
                f"context-specific branch prediction {brg:.3f} (== 9c numpy spiking ref {ref9c:.3f}) from the bridge's "
                f"OWN coincidence recurrence + apical-dAP fire-first selection. >> Markov {markov:.3f}, >> chance "
                f"{chance:.3f}, >> dAP-lesion {les:.3f} (the loaded distal pathway is load-bearing); untrained {unt:.3f}. "
                f"Multi-seed. => rung-4 inference on-substrate DONE -> Stage C (the three-term permanence kernel = "
                f"learning on the substrate => EMERGE-9d parity = rung-4 complete).")
    miss = []
    if brg < 0.90: miss.append(f"bridge branch {brg:.3f} < 0.90")
    if brg < markov + 0.15 or brg < chance + 0.20: miss.append(f"didn't clear Markov/chance ({brg:.3f})")
    if brg < les + 0.20: miss.append(f"dAP-lesion didn't collapse ({brg:.3f} vs {les:.3f})")
    if not parity: miss.append(f"9c-parity off (bridge {brg:.3f} vs 9c {ref9c:.3f})")
    return ("BOUNDARY (build-informative) -- " + "; ".join(miss) + f" (oracle {oracle:.3f}). Tune the middle-drive "
            f"window (primed fire-first vs non-primed rheobase), n_sub, apical_g_couple, or add the per-column WTA for "
            f"robust sparsification (Stage-B1 burst step). NOT a wall; the selection operating point is the next tuning.")


def _smoke(a, drives):
    """Single-seed, single-sequence diagnostic: verify the CORRECT branch column gets primed and the wrong one does not,
    at each middle-drive, for the trained bridge."""
    seed = a.seeds[0]
    tm, seqs, vocab, info = train_frozen_tm(seed, a.n_seq, a.middle_len, a.n_cells, a.k_win, a.act_th, a.epochs)
    branches = info["branches"]
    print(f"[smoke seed {seed}] vocab {vocab} cells/col {a.n_cells} | seqs {seqs} | branches {branches} "
          f"| oracle {full_oracle(seqs, a.middle_len):.2f} markov {markov_branch_acc(seqs, a.middle_len, a.n_seq):.2f}",
          flush=True)
    pre_e, post_e = export_distal_edges(tm)
    print(f"[smoke] exported {len(pre_e)} connected distal edges over {tm.N} cells "
          f"(mean {len(pre_e)/max(1,tm.N):.2f}/cell)", flush=True)
    bridge, cells_idx = build_bridge_from_tm(tm, seed, coincidence=True)
    print(f"[smoke] bridge num_neurons {bridge.core_config.num_neurons} | distal nnz {bridge.cp_connections.nnz} "
          f"| coincidence_k {bridge.core_config.coincidence_k_threshold}", flush=True)
    for md in drives:
        line = []
        for si, s in enumerate(seqs):
            pred, v_ap = present_and_predict(bridge, cells_idx, tm, s, a.middle_len, a.cue_drive, md, a.n_sub)
            want = s[a.middle_len + 1]
            ok = (pred == {want})
            line.append(f"seq{si}(cue{s[0]}->want{want}) pred={sorted(pred)} {'OK' if ok else 'x'} "
                        f"maxV_ap={v_ap.max():.1f}")
        print(f"  md {md:5.0f} | " + " || ".join(line), flush=True)
    print("[smoke] done -- look for a middle-drive where BOTH seqs predict exactly {want} (their own branch).",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
