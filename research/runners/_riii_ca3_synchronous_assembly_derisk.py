"""R-iii gap#5 — SYNCHRONY-ISOLATION de-risk: does DIRECT SYNCHRONOUS assembly encoding grow the within-ensemble
CA3 recurrent weights to the completion scale (the diagnostic-pinned residual)?

2026-07-18. The 2026-07-14 arc pinned the functional-completion blocker: the learned within-ensemble weights stay
~7.5 (co-activity-limited), ~200× below the ~1600 the hand-installed attractor needs, because the members fire
ASYNCHRONOUSLY. This session's cap-vs-synchrony test confirmed it (hebb_max 30→2000 byte-identical → NOT the cap).
The 2026-07-14 finding named but NEVER BUILT the fix: "my tests drove the UPSTREAM input, not the assembly cells
directly" (Kopsick-Ascoli 2024: drive the assembly PCs together in a gamma window → dense co-firing → strong LTP).

This de-risk ISOLATES that hypothesis: pre-assign a sparse CA3 assembly per pattern, drive THOSE cells DIRECTLY with
strong SYNCHRONOUS gamma-pulsed current during encoding (all fire together each ON window), rate-window LTP + the
committed EMERGE-40 competition ON, then recall a 50% partial cue directly on CA3 → does the held-out 50% FIRE
(functional pattern completion)? GO = h_comp≥0.30 & ≥2× non-stored, competition load-bearing (lam=0 vs 0.5),
async-control collapses (sync OFF → back to the ~7.5 weak weights → no completion). If GO → synchrony IS the fix →
the follow-on wires the EMERGENT mossy/DG assembly selection (experience-derived). If NO even with perfect synchrony
→ a deeper issue. GPU.
"""
import argparse
import os
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._riii_ca3_coincidence_completion_derisk import _build, _set_gates  # noqa: E402
from research.runners._riii_ca3_competitive_completion_payoff_derisk import _extract_ca3ca3_coincidence  # noqa: E402
from research.runners.validate_trisynaptic_loop import measure_region_response  # noqa: E402


def _extract_ca3ca3_all(bridge, ca3_idx, to_host):
    """ca3->ca3 synapses (ALL, NO coincidence mask) -> (flat_pos, pre_local, post_local). For the Wang nmda_slow mode,
    where cp_coincidence_synapse_mask is None (coincidence detection off)."""
    conn = bridge.cp_connections
    nnz = int(conn.nnz)
    indptr = np.asarray(to_host(conn.indptr)); indices = np.asarray(to_host(conn.indices))
    pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
    post_of = indices[:nnz]
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}
    ca3_set = set(ca3_pos.keys())
    flat, pre_l, post_l = [], [], []
    for k in range(nnz):
        pre, post = int(pre_of[k]), int(post_of[k])
        if pre in ca3_set and post in ca3_set:
            flat.append(k); pre_l.append(ca3_pos[pre]); post_l.append(ca3_pos[post])
    return (np.asarray(flat, dtype=np.int64), np.asarray(pre_l, dtype=np.int64), np.asarray(post_l, dtype=np.int64))


def run(seed, n_ca3=1000, n_mem=2, assembly_frac=0.012, train_events=120, sync_on=2, sync_off=4,
        encode_drive=700.0, recall_drive=250.0, lam_dep_wi=0.5, hebb_max=2000.0, ca3_fb_inhib=20.0,
        reset_steps=15, drive_steps=48, recall_steps=60, ens_thresh=2, no_sync=False,
        coact_thresh=0.02, hebb_lr=None, k_thresh=18.0, plateau_strength=120.0, apical_R=50.0, apical_gc=None,
        permute_recall=False, bistable=False, nmda_recurrent=False, nmda_tau=100.0):
    # DIAGNOSED LEVERS (2026-07-18 workflow): the rate-window LTP is an EMA-trace rule -- a cell's co-activity trace
    # tops out ~0.03-0.2 (point Izh fires ~0.2 duty @700pA), so coact_thresh MUST be BELOW it (~0.02) or nothing
    # potentiates; the gamma OFF-gap DECAYS the EMA (0.9^off) so CONTINUOUS drive (sync_off<=1) is required, NOT
    # synchrony; higher hebb_lr + strong drive (~3000pA -> ~0.5 duty) climb the weight toward the completion scale.
    from sim.backend import get_backend, to_host
    from sim.kernels import fused_htm_winner_inactive_depression
    cp, _ = get_backend()
    # WANG-2002 mode (nmda_recurrent): the ca3->ca3 recurrent is SOMATIC slow-NMDA (the bistable attractor itself);
    # the dendritic-coincidence dAP readout (coincidence/two_comp) is OFF. Else: the dAP-coincidence readout (default).
    bridge = _build(seed, n_ca3=n_ca3, ca3w=6.0, ca3_density=0.5,
                    coincidence=(not nmda_recurrent), two_comp=(not nmda_recurrent),
                    nmda_recurrent=nmda_recurrent, nmda_tau=nmda_tau, apical_R=apical_R,
                    apical_gc=apical_gc, k_thresh=k_thresh, plateau_strength=plateau_strength,
                    train=True, hebb_max=hebb_max, hebb_rate=True, ca3_fb_inhib=ca3_fb_inhib,
                    coact_thresh=coact_thresh, hebb_lr=hebb_lr)
    rm = bridge.region_manager
    ca3_idx = list(rm.indices("ca3")); ca3_arr = cp.asarray(ca3_idx, dtype=cp.int64)
    n = bridge.core_config.num_neurons
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}
    rng = np.random.default_rng(seed * 17 + 3)

    # PRE-ASSIGN sparse assemblies (~1% of CA3), disjoint-ish (random draw).
    n_assy = max(6, int(assembly_frac * n_ca3))
    assemblies = [np.asarray(sorted(rng.choice(ca3_idx, n_assy, replace=False)), dtype=np.int64) for _ in range(n_mem)]

    _extract = _extract_ca3ca3_all if nmda_recurrent else _extract_ca3ca3_coincidence
    flat_h, pre_l_h, post_l_h = _extract(bridge, ca3_idx, to_host)
    conn = bridge.cp_connections
    do_comp = lam_dep_wi > 0.0 and len(flat_h) > 0
    if do_comp:
        flat_pos = cp.asarray(flat_h, dtype=cp.int64)
        pre_local = cp.asarray(pre_l_h, dtype=cp.int64)
        post_local = cp.asarray(post_l_h, dtype=cp.int64)

    def _apply_competition(member_mask_local):
        fpre = member_mask_local[pre_local]; fpost = member_mask_local[post_local]
        w = conn.data[flat_pos]
        w = fused_htm_winner_inactive_depression(w, fpre, fpost, float(lam_dep_wi), 0.0, float(hebb_max))
        w = fused_htm_winner_inactive_depression(w, fpost, fpre, float(lam_dep_wi), 0.0, float(hebb_max))
        conn.data[flat_pos] = w

    _set_gates(bridge, 1.0)
    period = int(sync_on) + int(sync_off)
    for m, assy in enumerate(assemblies):
        assy_arr = cp.asarray(assy, dtype=cp.int64)
        # the KNOWN assembly is the competition member set (pre-assigned; local ca3 positions)
        member_mask = cp.zeros(len(ca3_idx), dtype=cp.float32)
        member_mask[cp.asarray([ca3_pos[int(g)] for g in assy], dtype=cp.int64)] = 1.0
        for ev in range(train_events):
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
            for _st in range(drive_steps):
                bridge.cp_external_input_current[:] = 0.0
                on = True if no_sync else ((_st % period) < int(sync_on))   # no_sync = drive every step (async control)
                if on:
                    bridge.cp_external_input_current[assy_arr] = float(encode_drive)   # DIRECT synchronous assembly drive
                bridge._run_one_simulation_step()
            if do_comp:
                _apply_competition(member_mask)
        bridge.cp_external_input_current[:] = 0.0
    _set_gates(bridge, 0.0)

    # within-ensemble vs member->silent weight read (did the weights GROW to the completion scale?)
    def _wstats():
        d = np.asarray(to_host(conn.data))
        wi, ws = [], []
        assy_set = set(int(g) for a in assemblies for g in a)
        for k, (pl, ql) in enumerate(zip(pre_l_h, post_l_h)):
            pre_g = ca3_idx[pl]; post_g = ca3_idx[ql]
            if pre_g in assy_set and post_g in assy_set:
                wi.append(d[flat_h[k]])
            elif pre_g in assy_set and post_g not in assy_set:
                ws.append(d[flat_h[k]])
        return (float(np.mean(wi)) if wi else 0.0), (float(np.mean(ws)) if ws else 0.0)
    w_within, w_silent = _wstats()
    non_stored0 = np.array([g for g in ca3_idx if g not in set(int(x) for a in assemblies for x in a)], dtype=np.int64)

    if bistable:
        # GENUINE CUE-GATED COMPLETION TEST: hard-SILENCE the network (clear v/u/firing/conductances to rest), then
        # drive a condition, and read the HELD (non-cued stored) members' firing. Real pattern completion requires:
        #   NO-CUE       -> held SILENT (the attractor is not self-sustaining/always-on)
        #   CORRECT cue  -> held FIRES (partial cue A ignites the full pattern A)
        #   PERMUTED cue -> held SILENT (specific: a random cue does NOT ignite A)
        # (The prior "completion" failed because measure_region_response never silenced the self-sustaining attractor.)
        from sim.backend import from_host
        n_all = bridge.core_config.num_neurons

        def _hard_silence(settle=30):
            if getattr(bridge, "cp_izh_c_reset", None) is not None:
                bridge.cp_membrane_potential_v[:] = bridge.cp_izh_c_reset
            else:
                bridge.cp_membrane_potential_v[:] = -65.0
            bridge.cp_recovery_variable_u[:] = 0.0
            if getattr(bridge, "cp_firing_states", None) is not None:
                bridge.cp_firing_states[:] = False
            for _a in ("cp_conductance_g_nmda_recurrent", "cp_conductance_g_e", "cp_conductance_g_i",
                       "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise"):
                _arr = getattr(bridge, _a, None)
                if _arr is not None:
                    _arr[:] = 0.0
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(settle):     # confirm it stays silent (a bistable attractor will; a self-sustaining one re-ignites)
                bridge._run_one_simulation_step()

        def _measure(cue_idx):
            _hard_silence()
            cur = np.zeros(n_all)
            if cue_idx is not None and len(cue_idx):
                cur[np.asarray(cue_idx, dtype=int)] = float(recall_drive)
            dev = from_host(cur.astype(np.float64)); spk = np.zeros(len(ca3_idx))
            for _ in range(recall_steps):
                bridge.cp_external_input_current[:] = dev; bridge._run_one_simulation_step()
                spk += np.asarray(to_host(bridge.cp_firing_states))[ca3_arr_host].astype(float)
            bridge.cp_external_input_current[:] = 0.0
            return spk / recall_steps
        ca3_arr_host = np.asarray(ca3_idx, dtype=int)

        nocue_l, cue_l, perm_l, silence_l = [], [], [], []
        for m, assy in enumerate(assemblies):
            a = assy.copy(); np.random.default_rng(seed + m).shuffle(a)
            half = max(2, len(a) // 2); cue, held = a[:half], a[half:]
            hp = [ca3_pos[int(g)] for g in held]
            # NO-CUE: also read the whole assembly's rest firing (self-sustain check)
            r0 = _measure(None); nocue_l.append(float(np.mean(r0[hp])))
            silence_l.append(float(np.mean(r0[[ca3_pos[int(g)] for g in a]])))
            r1 = _measure(cue); cue_l.append(float(np.mean(r1[hp])))
            perm = np.random.default_rng(seed * 7 + m + 999).choice(non_stored0, len(cue), replace=False)
            r2 = _measure(perm); perm_l.append(float(np.mean(r2[hp])))
        held_cue = float(np.mean(cue_l)); held_nocue = float(np.mean(nocue_l)); held_perm = float(np.mean(perm_l))
        rest = float(np.mean(silence_l))
        # GENUINE bistable completion (relative to the Wang low-rate background, NOT a dead net): the correct cue must
        # IGNITE the high state (>=0.20) AND be >=3x BOTH the no-cue low state AND the permuted -- i.e. only the correct
        # partial cue reaches the high attractor state; no-cue/permuted stay in the low state. The low background is
        # capped (<=0.10) so it is a genuine LOW state, not a near-self-sustaining one. Above-baseline completion signal
        # (cue-rest) vs permuted residual (perm-rest) reported for transparency.
        sig = held_cue - rest; perm_sig = held_perm - rest
        go = (held_cue >= 0.20 and held_cue >= 3.0 * (held_nocue + 1e-6) and held_cue >= 3.0 * (held_perm + 1e-6)
              and held_nocue <= 0.10)
        return {"seed": seed, "w_within": w_within, "held_cue": held_cue, "held_nocue": held_nocue,
                "held_perm": held_perm, "rest_firing": rest, "sig": float(sig), "perm_sig": float(perm_sig),
                "go": bool(go)}

    # RECALL: partial cue (50% of each assembly) DIRECT on CA3 -> does the held-out 50% fire?
    non_stored = np.array([g for g in ca3_idx if g not in set(int(x) for a in assemblies for x in a)], dtype=np.int64)
    held_list, ns_list = [], []
    held_abs_l, cue_abs_l, ns_abs_l = [], [], []
    for m, assy in enumerate(assemblies):
        a = assy.copy(); np.random.default_rng(seed + m).shuffle(a)
        half = max(2, len(a) // 2); cue, held = a[:half], a[half:]
        if permute_recall:
            # ANTI-CHEAT: cue a RANDOM NON-assembly set (same size) -> the stored assembly's held members must NOT
            # complete (rules out "any cue completes anything" / a drive artifact independent of the learned attractor).
            cue = np.asarray(np.random.default_rng(seed * 7 + m + 999).choice(non_stored, len(cue), replace=False), dtype=np.int64)
        resp = measure_region_response(bridge, "ca3", cue.tolist(), drive_pA=recall_drive,
                                       drive_region="ca3", n_steps=recall_steps)
        held_abs = float(np.mean(resp[[ca3_pos[int(g)] for g in held]]))
        cue_abs = float(np.mean(resp[[ca3_pos[int(g)] for g in cue]]))
        ns_abs = float(np.mean(resp[[ca3_pos[int(g)] for g in non_stored[:40]]]))
        held_abs_l.append(held_abs); cue_abs_l.append(cue_abs); ns_abs_l.append(ns_abs)
        cue_act = cue_abs or 1.0
        held_list.append(held_abs / (cue_act + 1e-9))
        ns_list.append(ns_abs / (cue_act + 1e-9))
    h_comp, n_comp = float(np.mean(held_list)), float(np.mean(ns_list))
    go = h_comp >= 0.30 and h_comp >= 2.0 * (n_comp + 1e-9)
    return {"seed": seed, "w_within": w_within, "w_silent": w_silent,
            "w_ratio": (w_within / (w_silent + 1e-9)), "h_comp": h_comp, "n_comp": n_comp, "go": bool(go),
            "held_abs": float(np.mean(held_abs_l)), "cue_abs": float(np.mean(cue_abs_l)), "ns_abs": float(np.mean(ns_abs_l))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-ca3", type=int, default=1000)
    ap.add_argument("--lam-dep-wi", type=float, default=0.5)
    ap.add_argument("--hebb-max", type=float, default=2000.0)
    ap.add_argument("--encode-drive", type=float, default=700.0)
    ap.add_argument("--no-sync", action="store_true", help="ASYNC control: drive every step (no gamma pulse)")
    a = ap.parse_args()
    t0 = time.time()
    print(f"[R-iii synchrony-isolation] n_ca3={a.n_ca3} lam={a.lam_dep_wi} hebb_max={a.hebb_max} "
          f"encode_drive={a.encode_drive} no_sync={a.no_sync}", flush=True)
    for s in [int(x) for x in a.seeds.split(",")]:
        r = run(s, n_ca3=a.n_ca3, lam_dep_wi=a.lam_dep_wi, hebb_max=a.hebb_max,
                encode_drive=a.encode_drive, no_sync=a.no_sync)
        print(f"  [seed {s}] w_within={r['w_within']:.1f} w_silent={r['w_silent']:.1f} ratio={r['w_ratio']:.2f} | "
              f"FUNCTIONAL h_comp={r['h_comp']:.3f} non-stored={r['n_comp']:.3f} -> {'GO' if r['go'] else 'NO'} "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
