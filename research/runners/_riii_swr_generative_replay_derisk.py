"""R-iii SWR generative-replay loop, Rung 1 (the payoff of the emergent-completion capstone, CYCLE 1076): does the
validated PARTIAL-cue CA3 completion drive a CORRECT + assembly-SPECIFIC downstream ca1 (Schaffer) pattern? That is
the prerequisite for replay-driven systems consolidation -- during NREM sharp-wave ripples (SWR), a partial/spontaneous
trigger reactivates the FULL CA3 assembly (completion), which drives ca1 -> cortex, and STDP consolidates the pattern
offline. Phase 1.3 validated consolidation with FULL-tag-drive replay; the NEW capability the capstone adds is
GENERATIVE replay -- reactivating the full pattern (and its correct ca1 projection) from a DEGRADED/partial cue.

Rung 1 (this runner): after the direct-synchronous FORMATION (Kopsick, CYCLE 1075) of K sparse CA3 assemblies, drive
(a) the FULL assembly and (b) a PARTIAL cue (half), reading the ca1 (Schaffer target) response each time. GO if the
PARTIAL-cue ca1 pattern MATCHES the full-assembly ca1 pattern (completion -> correct downstream projection) AND is
assembly-SPECIFIC (partial-A's ca1 != full-B's ca1). Anti-cheats: (LINEAR) coincidence OFF -> the partial cue does not
complete -> its ca1 pattern does NOT match the full pattern (the dendritic completion is what carries the pattern to
ca1); (CROSS) partial-A vs full-B ca1 cosine stays low. Reuse-by-import of the capstone's _build / _train_assemblies /
_recall (the ca3->ca1 Schaffer pathway, weight 4.0 fixed feedforward, is already wired via
enable_hippocampus_consolidation). NO `sim/` edit. 6-seed.

Rung 2 (next, if GO): open the `ca3_to_ca1` STDP gate during the replay phase and show the ca1 projection STRENGTHENS
+ then ca1 reactivates the pattern WITHOUT the ca3 recurrent completion (consolidated), vs a no-replay control.
"""
from __future__ import annotations
import argparse, time
import numpy as np
from research.runners._riii_ca3_coincidence_completion_derisk import _build
from research.runners._riii_ca3_emergent_completion_derisk import _train_assemblies, _recall


def _recall_burst(bridge, cp, cue_cells, read_cells, drive_pA, n_volley=6, gamma_on=8, gamma_off=12, clamp_cells=None):
    """SWR-ripple reactivation: drive the (partial) cue in GAMMA-SYNCHRONOUS volleys (like the formation protocol +
    a real sharp-wave ripple = a strong synchronous population burst), so the completed CA3 assembly fires
    SYNCHRONOUSLY -> a strong coincident Schaffer drive to ca1 (vs a gentle sustained cue, which the CYCLE-1076 probe
    showed does NOT drive ca1 even at 15x Schaffer weight). Accumulate the read_cells firing across the volleys."""
    from sim.backend import to_host
    _clamp = cp.asarray(clamp_cells, dtype=cp.int64) if clamp_cells is not None and len(clamp_cells) else None
    cue = cp.asarray(cue_cells, dtype=cp.int64)
    read = cp.asarray(read_cells, dtype=cp.int64)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):                                          # brief settle
        if _clamp is not None:
            bridge.cp_external_input_current[_clamp] = -5000.0
        bridge._run_one_simulation_step()
    acc = cp.zeros(len(read_cells), dtype=cp.float32)
    for _v in range(n_volley):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[cue] = float(drive_pA)   # ON pulse (synchronous volley)
        if _clamp is not None:
            bridge.cp_external_input_current[_clamp] = -5000.0
        for _ in range(gamma_on):
            bridge._run_one_simulation_step()
            acc += bridge.cp_firing_states[read].astype(cp.float32)
        bridge.cp_external_input_current[:] = 0.0
        if _clamp is not None:
            bridge.cp_external_input_current[_clamp] = -5000.0
        for _ in range(gamma_off):
            bridge._run_one_simulation_step()
            acc += bridge.cp_firing_states[read].astype(cp.float32)
    bridge.cp_external_input_current[:] = 0.0
    return to_host(acc)


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(a @ b / (na * nb))


def _scale_pathway(bridge, cp, pre_idx, post_idx, factor):
    """Scale the weights of every CSR edge from pre_idx -> post_idx by `factor` (a post-build pathway potentiation;
    biologically the SWR-ripple engagement of the ca3->ca1 Schaffer). Vectorized; host round-trip once."""
    from sim.backend import to_host, from_host
    conn = bridge.cp_connections
    nnz = int(conn.nnz)
    indptr = to_host(conn.indptr); indices = to_host(conn.indices)
    pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
    post_of = indices[:nnz]
    mask = np.isin(pre_of, np.asarray(pre_idx)) & np.isin(post_of, np.asarray(post_idx))
    if not mask.any():
        return 0
    data = to_host(conn.data[:nnz])
    data[mask] = data[mask] * float(factor)
    conn.data[:nnz] = from_host(data.astype(conn.data.dtype))
    return int(mask.sum())


def run_seed(seed, n_ca3=500, n_assembly=12, n_mem=3, presentations=60, drive_pA=1000.0, cue_drive=1000.0,
             hebb_lr=10.0, gamma_on=8, gamma_off=12, ca3_fb_inhib=120.0, k_thresh=20.0, plateau_strength=300.0,
             apical_R=50.0, hebb_max=120.0, schaffer_boost=6.0, burst=True, coincidence=True):
    from sim.backend import get_backend
    cp, _ = get_backend()
    bridge = _build(seed, n_ca3=n_ca3, ca3_density=0.5, ca3w=6.0, coincidence=coincidence, two_comp=True,
                    apical_R=apical_R, k_thresh=k_thresh, plateau_strength=plateau_strength, weighted=True, train=True,
                    hebb_rate=True, hebb_lr=hebb_lr, hebb_decay=0.0, coact_thresh=0.001, ca3_fb_inhib=ca3_fb_inhib,
                    hebb_max=hebb_max)
    rm = bridge.region_manager
    ca3_idx = np.asarray(list(rm.indices("ca3")), dtype=np.int64)
    ca1_idx = np.asarray(list(rm.indices("ca1")), dtype=np.int64)
    if schaffer_boost != 1.0:                                    # potentiate the Schaffer ca3->ca1 so the completed assembly drives ca1
        _n_sch = _scale_pathway(bridge, cp, ca3_idx, ca1_idx, schaffer_boost)
        if coincidence:                                          # print once (the main run, not the linear control)
            print(f"    [schaffer] scaled {_n_sch} ca3->ca1 edges x{schaffer_boost}", flush=True)
    try:
        basket = np.asarray(list(rm.indices("ca3_pv_basket")), dtype=np.int64)
    except Exception:
        basket = None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ca3_idx)
    assemblies = [np.array(perm[m * n_assembly:(m + 1) * n_assembly], dtype=np.int64) for m in range(n_mem)]
    _train_assemblies(bridge, cp, assemblies, presentations, drive_pA, gamma_on, gamma_off)

    # ca1 (Schaffer target) response to FULL assembly drive vs PARTIAL cue (completion), per assembly
    full_ca1, part_ca1 = [], []
    for asm in assemblies:
        a = asm.copy(); rng.shuffle(a)
        cue = a[:len(a) // 2]                                     # partial cue = half the assembly
        _rc = _recall_burst if burst else _recall
        full_ca1.append(_rc(bridge, cp, asm, ca1_idx, cue_drive, clamp_cells=basket))
        part_ca1.append(_rc(bridge, cp, cue, ca1_idx, cue_drive, clamp_cells=basket))
    # match = cos(partial-cue ca1, same-assembly full ca1); cross = cos(partial-A ca1, full-B ca1) B!=A
    match = float(np.mean([_cos(part_ca1[m], full_ca1[m]) for m in range(n_mem)]))
    cross_vals = [_cos(part_ca1[m], full_ca1[o]) for m in range(n_mem) for o in range(n_mem) if o != m]
    cross = float(np.mean(cross_vals)) if cross_vals else 0.0
    ca1_fire = float(np.mean([np.sum(v) for v in full_ca1]))       # raw ca1 spikes on a FULL-assembly drive (is ca1 driven at all?)
    return {"match": match, "cross": cross, "ca1_fire": ca1_fire}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-ca3", type=int, default=500)
    ap.add_argument("--n-assembly", type=int, default=12)
    ap.add_argument("--presentations", type=int, default=60)
    ap.add_argument("--schaffer-boost", type=float, default=6.0, help="post-build ca3->ca1 Schaffer potentiation (SWR-ripple engagement) so the 12-cell completed assembly drives ca1 above threshold")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"[R-iii SWR GEN-REPLAY Rung1] n_ca3={a.n_ca3} n_assembly={a.n_assembly} pres={a.presentations} "
          f"| partial-cue completion -> ca1 (Schaffer) pattern: correct + specific?", flush=True)
    import json
    kw = dict(n_ca3=a.n_ca3, n_assembly=a.n_assembly, presentations=a.presentations, schaffer_boost=a.schaffer_boost)
    rows = []
    for s in seeds:
        t0 = time.time()
        on = run_seed(s, coincidence=True, **kw)
        lin = run_seed(s, coincidence=False, **kw)
        row = {"seed": s, "match": on["match"], "cross": on["cross"], "linear_match": lin["match"]}
        rows.append(row)
        print(f"  [seed {s}] partial->ca1 MATCH={on['match']:.3f} (cross {on['cross']:.3f}) | "
              f"LINEAR match={lin['match']:.3f} | ca1_raw_fire={on['ca1_fire']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    if a.json and rows:
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        m = [r["match"] for r in rows]; c = [r["cross"] for r in rows]; lm = [r["linear_match"] for r in rows]
        go = (all(x > 0.50 for x in m) and all(mm - cc > 0.20 for mm, cc in zip(m, c))
              and all(mm - ll > 0.20 for mm, ll in zip(m, lm)))
        print(f"\n  AGGREGATE: MATCH={np.mean(m):.3f} cross={np.mean(c):.3f} LINEAR-match={np.mean(lm):.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the partial-cue CA3 completion drives the CORRECT + assembly-SPECIFIC ca1 (Schaffer) pattern (match > cross, and > the linear no-completion control) = generative replay can carry a full pattern to the consolidation target from a DEGRADED cue -> Rung 2 (open ca3_to_ca1 STDP + show consolidation)' if go else 'the completion does not yet drive a clean specific ca1 pattern; check the Schaffer drive / read window'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
