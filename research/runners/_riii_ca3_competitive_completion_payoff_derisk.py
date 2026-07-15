"""R-iii GO BAR 2 (the PAYOFF): does the LEARNED competitive-Hebbian CA3 attractor (formed by the heterosynaptic
winner-inactive depression, formation GO 2026-07-14: within/silent 3.7-10.9x vs pure-LTP 1.01x) actually COMPLETE a
partial cue through the committed dendritic-dAP plateau read-out — reversing the documented failure where the
UNIFORMLY-potentiated attractor gave held-out c_drive 75.9 < non-stored 84.0 (the plateau fired indiscriminately)?

This composes the two GO pieces WITHOUT re-deriving either: (1) the CYCLE-1068 dendritic-coincidence completion
read-out (`_riii_ca3_coincidence_completion_derisk._build` routes ca3->ca3 through the supralinear plateau; the
`coincidence_weighted_drive` c_drive is what the plateau reads); (2) the competitive-Hebbian formation (the committed
EMERGE-40 `fused_htm_winner_inactive_depression` applied to the ca3->ca3 recurrents each encoding window, keyed to
CA3's own firing, both directions -- validated in `_riii_ca3_competitive_formation_derisk`). NO `sim/` edit.

GO BAR 2: with competition ON, held-out c_drive > non-stored c_drive (the LEARNED selective attractor drives the
plateau specifically for stored members); with competition OFF (lam_dep_wi=0), held-out c_drive <= non-stored (the
documented pure-LTP failure). ANTI-CHEATS carried from formation: (A) lam=0 control reverts to non-selective;
(B) the completion measures HELD-OUT (non-cued) stored members, not the cue; (C) non-stored specificity.
"""
from __future__ import annotations
import argparse, time
import numpy as np
from research.runners._riii_ca3_coincidence_completion_derisk import _build, _set_gates
from research.runners.validate_trisynaptic_loop import measure_region_response, build_drive_pattern
from research.runners._riii_ca3_competitive_formation_derisk import _extract_ca3ca3_coincidence


def run_payoff(seed, n_mem=2, train_events=150, drive_pA=200.0, coincidence=True, n_lang=384, n_ca3=150, n_dg=300,
               ca3_density=0.5, ca3_weight=6.0, k_thresh=18.0, plateau_strength=120.0, hebb_max=30.0,
               hebb_rate=True, lam_dep_wi=0.0, comp_both_dir=True, ens_thresh=2,
               ca3_fb_inhib=None, ca3_fb_n=None, mossy_weight=None, mossy_density=None, dg_ffi_weight=None,
               recall_drive_pA=None, two_comp=True, apical_R=50.0, apical_gc=None, recall_inhib_scale=1.0,
               sync_on=None, sync_off=None, reset_steps=15, drive_steps=55, recall_steps=60):
    # DEFAULT to the CYCLE-1068 VALIDATED dendritic-dAP read-out (two_comp + thin-high-R apical + calibrated k_thresh),
    # NOT the point-neuron read-out (two_comp=False) that CYCLE-1067 proved fails even on a good hand-installed attractor.
    from sim.backend import get_backend, to_host, get_sparse_module
    from sim.kernels import fused_htm_winner_inactive_depression
    cp, _ = get_backend()
    bridge = _build(seed, n_lang=n_lang, n_ca3=n_ca3, n_dg=n_dg, ca3_density=ca3_density, ca3w=ca3_weight,
                    coincidence=coincidence, k_thresh=k_thresh, plateau_strength=plateau_strength,
                    weighted=True, two_comp=two_comp, apical_R=apical_R, apical_gc=apical_gc, train=True,
                    hebb_max=hebb_max, hebb_rate=hebb_rate,
                    ca3_fb_inhib=ca3_fb_inhib, ca3_fb_n=ca3_fb_n, mossy_weight=mossy_weight,
                    mossy_density=mossy_density, dg_ffi_weight=dg_ffi_weight)
    rm = bridge.region_manager
    lang = list(rm.indices("language_input")); lang_arr = np.asarray(lang, dtype=np.int64)
    ca3_idx = list(rm.indices("ca3")); ca3_arr = cp.asarray(ca3_idx, dtype=cp.int64)
    n_lang = len(lang)
    patterns = [build_drive_pattern(n_neurons=n_lang, sparsity=0.1, seed=seed * 100 + m) for m in range(n_mem)]

    conn = bridge.cp_connections
    flat_h, pre_l_h, post_l_h = _extract_ca3ca3_coincidence(bridge, ca3_idx, to_host)
    do_comp = lam_dep_wi > 0.0 and len(flat_h) > 0
    if do_comp:
        flat_pos = cp.asarray(flat_h, dtype=cp.int64)
        pre_local = cp.asarray(pre_l_h, dtype=cp.int64)
        post_local = cp.asarray(post_l_h, dtype=cp.int64)

    def _apply_competition(fired):
        # `fired` is the CUMULATIVE ensemble mask (0/1 per ca3 cell): a cell that fired >= ens_thresh times across
        # THIS pattern's events so far is a stable assembly member -> protected from depression even on events where
        # it happens to be silent (robust to the distributed/async firing that eroded the per-event within-ensemble).
        fpre = fired[pre_local]; fpost = fired[post_local]
        w = conn.data[flat_pos]
        w = fused_htm_winner_inactive_depression(w, fpre, fpost, float(lam_dep_wi), 0.0, float(hebb_max))
        if comp_both_dir:
            w = fused_htm_winner_inactive_depression(w, fpost, fpre, float(lam_dep_wi), 0.0, float(hebb_max))
        conn.data[flat_pos] = w

    stored = {}
    _set_gates(bridge, 1.0)
    rec_last = min(10, max(1, train_events // 3))
    for m, pat in enumerate(patterns):
        drv = cp.asarray(lang_arr[pat], dtype=cp.int64)
        spikes = cp.zeros(len(ca3_idx), dtype=cp.float32)
        ens_acc = cp.zeros(len(ca3_idx), dtype=cp.float32)      # cumulative per-cell firing = the stable assembly
        for ev in range(train_events):
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
            recording = ev >= train_events - rec_last
            win_fire = cp.zeros(len(ca3_idx), dtype=cp.float32)
            if sync_on is not None:
                # KOPSICK gamma-paced SYNCHRONOUS encoding: drive in tight ON pulses so the assembly members fire
                # TOGETHER in each gamma window -> dense within-window co-firing -> STRONG within-ensemble LTP ->
                # a high-absolute attractor (the diagnostic-pinned residual), while OFF windows keep it sparse.
                _period = int(sync_on) + int(sync_off)
                for _st in range(drive_steps):
                    bridge.cp_external_input_current[:] = 0.0
                    if (_st % _period) < int(sync_on):
                        bridge.cp_external_input_current[drv] = float(drive_pA)
                    bridge._run_one_simulation_step()
                    f = bridge.cp_firing_states[ca3_arr].astype(cp.float32)
                    win_fire += f
                    if recording:
                        spikes += f
            else:
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_external_input_current[drv] = float(drive_pA)
                for _ in range(drive_steps):
                    bridge._run_one_simulation_step()
                    f = bridge.cp_firing_states[ca3_arr].astype(cp.float32)
                    win_fire += f
                    if recording:
                        spikes += f
            ens_acc += win_fire
            if do_comp:
                _apply_competition((ens_acc >= float(ens_thresh)).astype(cp.float32))
        bridge.cp_external_input_current[:] = 0.0
        sp = to_host(spikes)
        n_stored = max(4, int(0.10 * len(ca3_idx)))
        top = np.argsort(-sp)[:n_stored]; top = top[sp[top] > 0]
        stored[m] = np.array([ca3_idx[i] for i in top], dtype=np.int64)
    _set_gates(bridge, 0.0)

    # ---- RECALL-TIME inhibition-knob DIAGNOSTIC (research gate 2026-07-14, highest info-per-minute): scale the
    # ca3_pv_basket->ca3 feedback-inhibition weight by g at RECALL only (encoding unchanged). If completion
    # robustifies in some g<1 window -> "members are crushed by global inhibition" is the true bottleneck ->
    # assembly-selective iSTDP/E->I-STDP is worth building; if no g robustifies -> the recurrent-weight structure
    # is the bottleneck (iSTDP won't help). NO sim/ edit -- scales the masked cp_connections.data slice. ----
    if recall_inhib_scale != 1.0 and ca3_fb_inhib is not None:
        try:
            basket = set(int(x) for x in rm.indices("ca3_pv_basket"))
            ca3set = set(int(x) for x in ca3_idx)
            nnz0 = int(conn.nnz)
            indptr0 = to_host(conn.indptr); indices0 = to_host(conn.indices)
            pre0 = np.searchsorted(indptr0, np.arange(nnz0), side="right") - 1
            flat_ie = [k for k in range(nnz0) if int(pre0[k]) in basket and int(indices0[k]) in ca3set]
            if flat_ie:
                fpos = cp.asarray(np.asarray(flat_ie, dtype=np.int64), dtype=cp.int64)
                conn.data[fpos] = conn.data[fpos] * float(recall_inhib_scale)
        except Exception as _e:
            print(f"  [recall_inhib_scale] skipped: {_e}", flush=True)

    # ---- completion read-out (identical to run_seed) ----
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}
    stored_all = set(int(x) for m in range(n_mem) for x in stored[m])
    non_stored = np.array([g for g in ca3_idx if int(g) not in stored_all], dtype=np.int64)

    def _cdrive_for_cue(cue_global):
        csp = get_sparse_module()
        nnz = int(conn.nnz)
        mask = bridge.cp_coincidence_synapse_mask[:nnz].astype(cp.float32)
        data = conn.data[:nnz] * mask
        mat = csp.csr_matrix((data, conn.indices, conn.indptr), shape=conn.shape)
        x = cp.zeros(conn.shape[0], dtype=cp.float32)
        x[cp.asarray(cue_global, dtype=cp.int64)] = 1.0
        return to_host((mat.T @ x))

    held_list, nonstored_list, held_cd, non_cd = [], [], [], []
    for m in range(n_mem):
        se = stored[m]
        if len(se) < 4:
            return None
        np.random.default_rng(seed + m).shuffle(se)
        n_part = max(2, int(0.5 * len(se)))
        cue, held = se[:n_part], se[n_part:]
        _rdrv = recall_drive_pA if recall_drive_pA is not None else drive_pA   # recall cue drive: decoupled from the (low) training drive
        part_resp = measure_region_response(bridge, "ca3", cue, drive_pA=_rdrv, drive_region="ca3", n_steps=recall_steps)
        held_pos = [ca3_pos[int(g)] for g in held if int(g) in ca3_pos]
        cue_pos = [ca3_pos[int(g)] for g in cue if int(g) in ca3_pos]
        ns_pos = [ca3_pos[int(g)] for g in non_stored[:40] if int(g) in ca3_pos]
        cue_act = float(np.mean(part_resp[cue_pos])) if cue_pos else 1.0
        held_list.append((float(np.mean(part_resp[held_pos])) if held_pos else 0.0) / (cue_act + 1e-9))
        nonstored_list.append((float(np.mean(part_resp[ns_pos])) if ns_pos else 0.0) / (cue_act + 1e-9))
        cd = _cdrive_for_cue(cue)
        held_cd.append(float(np.mean([cd[int(g)] for g in held])) if len(held) else 0.0)
        non_cd.append(float(np.mean([cd[int(g)] for g in non_stored[:40]])) if len(non_stored) else 0.0)

    h_cd, n_cd = float(np.mean(held_cd)), float(np.mean(non_cd))
    h_comp, n_comp = float(np.mean(held_list)), float(np.mean(nonstored_list))
    # FUNCTIONAL completion (the definition of pattern completion): does the HELD-OUT (non-cued) stored member
    # REACTIVATE from the partial cue (h_comp high), SPECIFICALLY (>> non-stored n_comp)? This is the read-out
    # AFTER the plateau + FS-WTA dynamics -- the true completion, vs the raw weighted c_drive diagnostic.
    go = h_comp >= 0.30 and h_comp >= 2.0 * (n_comp + 1e-9)
    print(f"[R-iii competitive completion PAYOFF] seed {seed} lam_dep_wi={lam_dep_wi} train_events={train_events}", flush=True)
    print(f"  c_drive (raw weighted diagnostic): held-out={h_cd:.2f}  non-stored={n_cd:.2f}", flush=True)
    print(f"  FUNCTIONAL completion activity (ratio to cue): held-out={h_comp:.3f}  non-stored={n_comp:.3f}", flush=True)
    verdict = (f"GO: held-out completes {h_comp:.3f} >> non-stored {n_comp:.3f} -> the LEARNED attractor COMPLETES "
               f"the held-out members from a partial cue, SPECIFICALLY") if go else \
              (f"NO: held-out completion {h_comp:.3f} not >> non-stored {n_comp:.3f} (bar: held>=0.30 AND held>=2x non)")
    print(f"  VERDICT -> {verdict}", flush=True)
    return {"seed": seed, "lam_dep_wi": lam_dep_wi, "held_cdrive": h_cd, "nonstored_cdrive": n_cd,
            "held_completion": h_comp, "nonstored_completion": n_comp, "go": go}


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--train-events", type=int, default=150)
    ap.add_argument("--lam-dep-wi", type=float, default=0.0, help="heterosynaptic winner-inactive depression (0=OFF control)")
    ap.add_argument("--k-thresh", type=float, default=18.0)
    ap.add_argument("--one-dir", action="store_true")
    ap.add_argument("--ens-thresh", type=int, default=2, help="cumulative-firing threshold for a ca3 cell to count as a stable assembly member (protected from depression)")
    ap.add_argument("--drive-pA", type=float, default=200.0, help="encoding drive: LOWER -> fewer CA3 fire -> sparser ensemble")
    ap.add_argument("--ca3-fb-inhib", type=float, default=None, help="ca3_pv_basket->ca3 FEEDBACK inhibition weight (FS-WTA sparsifier; None=off)")
    ap.add_argument("--ca3-fb-n", type=int, default=None)
    ap.add_argument("--mossy-weight", type=float, default=None)
    ap.add_argument("--mossy-density", type=float, default=None, help="Kopsick sparse mossy: dg->ca3 density (0.10->~0.05 = fewer, decorrelated detonators)")
    ap.add_argument("--dg-ffi-weight", type=float, default=None, help="Kopsick sparse DG code: dg_pv_basket->dg FF-inhibition weight (stronger -> sparser DG assembly)")
    ap.add_argument("--recall-drive-pA", type=float, default=None, help="recall cue drive (decoupled from the low training drive; default = training drive)")
    ap.add_argument("--n-ca3", type=int, default=150, help="CA3 size (scale lever: larger -> larger, more redundant ensembles -> robust completion)")
    ap.add_argument("--n-mem", type=int, default=2)
    ap.add_argument("--recall-inhib-scale", type=float, default=1.0, help="scale the ca3_pv_basket->ca3 feedback inhibition at RECALL only (diagnostic: does relaxing member-crushing inhibition robustify completion?)")
    ap.add_argument("--no-two-comp", action="store_true", help="use the point-neuron read-out (two_comp=False) instead of the dendritic-dAP default")
    ap.add_argument("--sync-on", type=int, default=None, help="Kopsick gamma-pulse ON steps (synchronous encoding -> strong within-ensemble LTP)")
    ap.add_argument("--sync-off", type=int, default=None, help="gamma-pulse OFF steps between volleys")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    t0 = time.time()
    results = []
    for s in seeds:
        r = run_payoff(seed=s, n_mem=a.n_mem, n_ca3=a.n_ca3, train_events=a.train_events, lam_dep_wi=a.lam_dep_wi,
                       k_thresh=a.k_thresh, comp_both_dir=not a.one_dir, ens_thresh=a.ens_thresh, drive_pA=a.drive_pA,
                       ca3_fb_inhib=a.ca3_fb_inhib, ca3_fb_n=a.ca3_fb_n, mossy_weight=a.mossy_weight,
                       mossy_density=a.mossy_density, dg_ffi_weight=a.dg_ffi_weight,
                       recall_drive_pA=a.recall_drive_pA, recall_inhib_scale=a.recall_inhib_scale,
                       two_comp=not a.no_two_comp, sync_on=a.sync_on, sync_off=a.sync_off)
        if r is not None:
            results.append(r)
    n_go = sum(1 for r in results if r["go"])
    print(f"\n=== PAYOFF {n_go}/{len(results)} GO (lam={a.lam_dep_wi}) ({time.time()-t0:.0f}s) ===", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
