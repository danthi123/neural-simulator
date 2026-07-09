"""R-iii on-substrate READ-OUT test (decouples the read-out mechanism from the learning problem). CYCLE 1066 found
the ca3->ca3 rate-Hebbian does NOT form a specific within-ensemble attractor (weights stay ~uniform init), so the
plateau has nothing to amplify. This test INSTALLS a clean, known attractor BY HAND (a clearly-labeled teaching
scaffold, to be replaced by emergent potentiation) and asks the decisive read-out questions ON THE REAL SPIKING
BRIDGE:
  (Q1) Does the plateau (coincidence ON, calibrated k_thresh) COMPLETE a partial cue of the installed attractor,
       SPECIFICALLY (held-out members fire, non-members don't)?  -> the on-substrate version of the CYCLE-1065 GO.
  (Q2) Does the LINEAR point-neuron (coincidence OFF) complete the SAME installed attractor? If YES, the CYCLE-1064
       boundary was the MISSING ATTRACTOR, not the point-neuron summation limit. If NO (linear fails, plateau
       succeeds), the dendritic non-linearity is genuinely required even with an attractor present.

The attractor: N_ENS disjoint random CA3 ensembles; within-ensemble ca3->ca3 recurrents set to W_HIGH, all other
ca3->ca3 recurrents to W_LOW (a Marr autoassociator by construction). Recall drives a partial cue (half the
ensemble) directly on CA3; completion = held-out (non-cued) member firing / cue firing; specificity = non-ensemble
firing. Anti-cheats: (A) LINEAR vs PLATEAU at the SAME installed attractor (isolates the read-out); (B) SPECIFICITY
(non-ensemble neurons stay low); (C) FLAT control -- install NO attractor (all recurrents W_LOW) -> neither
completes (the completion rides the installed structure). NO `sim/` edit (writes cp_connections.data via the public
array, exactly as the harness drives cp_external_input_current).
"""
from __future__ import annotations
import argparse, time
import numpy as np
from research.runners._riii_ca3_coincidence_completion_derisk import _build
from research.runners.validate_trisynaptic_loop import measure_region_response


def _ca3_recurrent_flat(bridge, ca3_set):
    """Flat synapse indices of the ca3->ca3 recurrents + their (pre,post), from the CSR (row=pre, col=post)."""
    from sim.backend import to_host
    conn = bridge.cp_connections; nnz = int(conn.nnz)
    indptr = to_host(conn.indptr); indices = to_host(conn.indices)
    pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
    post_of = indices[:nnz]
    idx = [k for k in range(nnz) if int(pre_of[k]) in ca3_set and int(post_of[k]) in ca3_set]
    return np.array(idx, dtype=np.int64), pre_of, post_of


def run_seed(seed, coincidence, k_thresh=20.0, w_high=15.0, w_low=1.5, plateau_strength=140.0,
             n_ens=3, ens_size=20, n_ca3=150, ca3_density=0.5, drive_pA=200.0, recall_steps=60, flat=False, mg=None,
             two_comp=False, apical_R=None, apical_gc=None, scramble=False):
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    bridge = _build(seed, n_ca3=n_ca3, ca3_density=ca3_density, ca3w=6.0, coincidence=coincidence,
                    k_thresh=k_thresh, plateau_strength=plateau_strength, weighted=True, train=False, mg=mg,
                    two_comp=two_comp, apical_R=apical_R, apical_gc=apical_gc)
    rm = bridge.region_manager
    ca3_idx = list(rm.indices("ca3")); ca3_set = set(int(x) for x in ca3_idx)
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ca3_idx)
    ensembles = [np.array(perm[e * ens_size:(e + 1) * ens_size], dtype=np.int64) for e in range(n_ens)]
    ens_of = {int(g): e for e, ens in enumerate(ensembles) for g in ens}

    # INSTALL the attractor: all ca3->ca3 = W_LOW; within-ensemble = W_HIGH (unless flat control -> all W_LOW).
    # SCRAMBLE control: the SAME NUMBER of W_HIGH synapses, but on RANDOM ca3->ca3 synapses (wrong structure, same
    # weight budget) -> if completion needs the RIGHT ensemble structure (not just some strong synapses), it collapses.
    flat_idx, pre_of, post_of = _ca3_recurrent_flat(bridge, ca3_set)
    data = to_host(bridge.cp_connections.data)
    within_idx = [int(k) for k in flat_idx
                  if (ens_of.get(int(pre_of[k])) is not None and ens_of.get(int(pre_of[k])) == ens_of.get(int(post_of[k])))]
    for k in flat_idx:
        data[int(k)] = w_low
    if not flat:
        if scramble:
            hi = rng.choice(np.asarray(flat_idx, dtype=np.int64), size=len(within_idx), replace=False)
        else:
            hi = np.asarray(within_idx, dtype=np.int64)
        for k in hi:
            data[int(k)] = w_high
    bridge.cp_connections.data[:] = cp.asarray(data, dtype=bridge.cp_connections.data.dtype)

    # Rung-0 CALIBRATION probe: the ACTUAL per-step weighted coincident drive (same masked-weighted transposed matvec
    # the plateau reads vs prev_firing) on the installed attractor, so k_thresh can be set between held and non.
    def _cdrive(cue_global):
        if getattr(bridge, "cp_coincidence_synapse_mask", None) is None:
            return None
        from sim.backend import get_sparse_module
        csp = get_sparse_module()
        nnz = int(bridge.cp_connections.nnz)
        mk = bridge.cp_coincidence_synapse_mask[:nnz].astype(cp.float32)
        d = bridge.cp_connections.data[:nnz] * mk
        mat = csp.csr_matrix((d, bridge.cp_connections.indices, bridge.cp_connections.indptr), shape=bridge.cp_connections.shape)
        x = cp.zeros(bridge.cp_connections.shape[0], dtype=cp.float32)
        x[cp.asarray(cue_global, dtype=cp.int64)] = 1.0
        return to_host(mat.T @ x)

    held_c, non_c = [], []
    diag_cd = {"h": [], "n": []}
    non_ens = np.array([g for g in ca3_idx if int(g) not in ens_of], dtype=np.int64)
    for e, ens in enumerate(ensembles):
        se = ens.copy(); rng.shuffle(se)
        n_part = max(2, int(0.5 * len(se)))
        cue, held = se[:n_part], se[n_part:]
        resp = measure_region_response(bridge, "ca3", cue, drive_pA=drive_pA, drive_region="ca3", n_steps=recall_steps)
        hp = [ca3_pos[int(g)] for g in held]; cp_ = [ca3_pos[int(g)] for g in cue]
        npos = [ca3_pos[int(g)] for g in non_ens[:40]]
        ca = float(np.mean(resp[cp_])) + 1e-9
        held_c.append(float(np.mean(resp[hp])) / ca)
        non_c.append(float(np.mean(resp[npos])) / ca)
        if coincidence and not flat:
            cd = _cdrive(cue)
            if cd is not None:
                diag_cd["h"].append(float(np.mean([cd[int(g)] for g in held])))
                diag_cd["n"].append(float(np.mean([cd[int(g)] for g in non_ens[:40]])))
    return {"heldout": float(np.mean(held_c)), "nonens": float(np.mean(non_c)),
            "cdrive_held": float(np.mean(diag_cd["h"])) if diag_cd["h"] else None,
            "cdrive_non": float(np.mean(diag_cd["n"])) if diag_cd["n"] else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--k-thresh", type=float, default=20.0)
    ap.add_argument("--w-high", type=float, default=15.0)
    ap.add_argument("--w-low", type=float, default=1.5)
    ap.add_argument("--plateau-strength", type=float, default=140.0)
    ap.add_argument("--drive-pA", type=float, default=200.0, help="recall cue drive (CYCLE-1064 transmission needed ~800)")
    ap.add_argument("--mg", type=float, default=None, help="nmda_mg_concentration (lower opens the Mg2+ block -> plateau at rest)")
    ap.add_argument("--two-comp", action="store_true", help="regenerate the plateau on the apical dAP compartment")
    ap.add_argument("--apical-R", type=float, default=None, help="apical input resistance (higher = larger local dV per plateau current)")
    ap.add_argument("--apical-gc", type=float, default=None, help="apical->soma coupling conductance")
    ap.add_argument("--scramble", action="store_true", help="add the SCRAMBLE control (same w_high budget, random structure)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"[R-iii on-substrate read-out] installed attractor w_high={a.w_high} w_low={a.w_low} k_thresh={a.k_thresh} "
          f"| PLATEAU vs LINEAR vs FLAT-control, held-out completion + specificity", flush=True)
    import json
    rows = []
    kw = dict(k_thresh=a.k_thresh, w_high=a.w_high, w_low=a.w_low, plateau_strength=a.plateau_strength,
              drive_pA=a.drive_pA, mg=a.mg, two_comp=a.two_comp, apical_R=a.apical_R, apical_gc=a.apical_gc)
    for s in seeds:
        t0 = time.time()
        plat = run_seed(s, coincidence=True, **kw)
        lin = run_seed(s, coincidence=False, **kw)
        flatc = run_seed(s, coincidence=True, flat=True, **kw)
        scr = run_seed(s, coincidence=True, scramble=True, **kw) if a.scramble else None
        row = {"seed": s, "plateau_held": plat["heldout"], "plateau_non": plat["nonens"],
               "linear_held": lin["heldout"], "linear_non": lin["nonens"],
               "flat_held": flatc["heldout"], "plateau_vs_linear": plat["heldout"] - lin["heldout"]}
        if scr is not None:
            row["scramble_held"] = scr["heldout"]
        rows.append(row)
        _cd = f"c_drive[held={plat.get('cdrive_held'):.1f} non={plat.get('cdrive_non'):.1f}]" if plat.get("cdrive_held") is not None else ""
        _sc = f"SCRAMBLE held={scr['heldout']:.3f} |" if scr is not None else ""
        print(f"  [seed {s}] PLATEAU held={plat['heldout']:.3f}(non {plat['nonens']:.3f}) | "
              f"LINEAR held={lin['heldout']:.3f}(non {lin['nonens']:.3f}) | FLAT-ctrl held={flatc['heldout']:.3f} | {_sc} {_cd} "
              f"(plateau-vs-linear={row['plateau_vs_linear']:+.3f}) ({time.time()-t0:.0f}s)", flush=True)
    if a.json and rows:
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        ph = [r["plateau_held"] for r in rows]; lh = [r["linear_held"] for r in rows]
        pn = [r["plateau_non"] for r in rows]; fh = [r["flat_held"] for r in rows]
        print(f"\n  AGGREGATE: PLATEAU held-out={np.mean(ph):.3f} (non {np.mean(pn):.3f}) | LINEAR held-out={np.mean(lh):.3f} | FLAT-ctrl={np.mean(fh):.3f}", flush=True)
        # Interpret: does either read-out complete the installed attractor specifically, and is the plateau needed?
        plat_go = all(h > 0.30 for h in ph) and all(n < 0.20 for n in pn) and all(f < 0.20 for f in fh)
        lin_completes = np.mean(lh) > 0.30
        print(f"  PLATEAU completes installed attractor specifically (flat collapses): {'YES' if plat_go else 'no'} | "
              f"LINEAR also completes it: {'YES -> CYCLE-1064 boundary was the MISSING ATTRACTOR, not the point-neuron limit' if lin_completes else 'NO -> the dendritic non-linearity is genuinely required'}", flush=True)


if __name__ == "__main__":
    main()
