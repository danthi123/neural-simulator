"""R-iii CA3 formation, the KOPSICK-2024-CORRECT protocol (read the working model's methods myself, 2026-07-09):
the working CA3 autoassociator forms an assembly by DRIVING A SPARSE SUBSET OF PYRAMIDAL CELLS DIRECTLY, SYNCHRONOUSLY
(each assembly cell fires ~4 spikes in a 20ms gamma window), and binds them with a SYMMETRIC co-activity rule -- the
sparsity comes from the sparse INPUT PROTOCOL, NOT from feedback inhibition, and the trisynaptic loop (my prior
distributed 35-47% code) is NOT part of the formation test (that is the separate pattern-separation problem). This
runner isolates the recurrent autoassociator FORMATION exactly as Kopsick does: define K disjoint SPARSE CA3
assemblies, drive each assembly's cells DIRECTLY in gamma-synchronous volleys (with the ca3->ca3 recurrent plasticity
gate open + the rate-window / symmetric co-activity Hebbian), then measure the learned within-assembly recurrent
weight vs the assembly->non-assembly weight. GO = within-assembly >> cross (a strong SPECIFIC attractor, ratio >=3x)
where the trisynaptic-routed distributed code plateaued at ~1.4x.

Anti-cheats: (A) NO-TRAIN (fresh, gate closed) -> no potentiation; (B) PERMUTED assembly (shuffle which cells are
"the assembly" AFTER training) -> the measured within/cross separation collapses (the potentiation rode the TRAINED
membership); (C) the assembly is SPARSE + the drive is DIRECT (the Kopsick protocol, not the distributed loop). numpy
-smoke / GPU. Reuse-by-import of _build (the ca3_pv_basket / rate-window / coincidence machinery). NO `sim/` edit.

Ref: Kopsick et al. 2024, J Comput Neurosci, "Formation and Retrieval of Cell Assemblies in a Biologically Realistic
Spiking Neural Network Model of Area CA3" (PMC10996657): 20ms gamma window, 200ms theta, symmetric STDP tau=20ms,
assembly ~0.37% sparse, ~40 presentations; sparsity from the input protocol not inhibition.
"""
from __future__ import annotations
import argparse, time
import numpy as np
from research.runners._riii_ca3_coincidence_completion_derisk import _build, _set_gates


def run_seed(seed, n_assembly=12, n_mem=3, presentations=60, drive_pA=600.0, hebb_lr=5.0,
            gamma_on=8, gamma_off=12, n_ca3=150, ca3_density=0.5, do_train=True, permute=False, ca3_fb_inhib=None,
            hebb_max=None):
    """Kopsick protocol: K disjoint sparse CA3 assemblies, each driven DIRECTLY in gamma volleys with the recurrent
    rate-window Hebbian on. Optional feedback inhibition (ca3_fb_inhib) suppresses the recurrent SPILLOVER to non-
    assembly cells (the drive selects the members; the inhibition sparsifies the rest -> cross stays low over
    presentations). Returns the learned within-assembly vs cross recurrent-weight separation + ratio."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    bridge = _build(seed, n_ca3=n_ca3, ca3_density=ca3_density, ca3w=6.0, coincidence=False, train=True,
                    hebb_rate=True, hebb_lr=hebb_lr, hebb_decay=0.0, coact_thresh=0.001, ca3_fb_inhib=ca3_fb_inhib,
                    hebb_max=hebb_max)
    rm = bridge.region_manager
    ca3_idx = list(rm.indices("ca3")); ca3_set = set(int(x) for x in ca3_idx)
    ca3_arr = np.asarray(ca3_idx, dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ca3_arr)
    assemblies = [np.array(perm[m * n_assembly:(m + 1) * n_assembly], dtype=np.int64) for m in range(n_mem)]

    if do_train:
        _set_gates(bridge, 1.0)   # open ca3_swr_burst (recurrent plasticity)
        for _pres in range(presentations):
            for asm in assemblies:
                drv = cp.asarray(asm, dtype=cp.int64)
                # settle
                bridge.cp_external_input_current[:] = 0.0
                for _ in range(6):
                    bridge._run_one_simulation_step()
                # THETA window: gamma-synchronous volleys -- drive the assembly cells DIRECTLY, ON for gamma_on steps
                # (the ~20ms gamma window; all assembly cells fire together = synchrony), OFF for gamma_off (theta gap).
                for _v in range(3):                          # a few gamma volleys per theta window
                    bridge.cp_external_input_current[:] = 0.0
                    bridge.cp_external_input_current[drv] = float(drive_pA)
                    for _ in range(gamma_on):
                        bridge._run_one_simulation_step()
                    bridge.cp_external_input_current[:] = 0.0
                    for _ in range(gamma_off):
                        bridge._run_one_simulation_step()
        _set_gates(bridge, 0.0)

    # measure: the ca3->ca3 recurrent weights, within-assembly vs cross (assembly-member -> non-assembly cell)
    conn = bridge.cp_connections; nnz = int(conn.nnz)
    indptr = to_host(conn.indptr); indices = to_host(conn.indices); data = to_host(conn.data[:nnz])
    pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
    post_of = indices[:nnz]
    asm_of = {}
    _asm_use = assemblies
    if permute:  # anti-cheat: shuffle membership AFTER training -> separation must collapse
        pool = np.concatenate(assemblies); rng.shuffle(pool)
        _asm_use = [pool[m * n_assembly:(m + 1) * n_assembly] for m in range(n_mem)]
    for m, asm in enumerate(_asm_use):
        for g in asm:
            asm_of[int(g)] = m
    within, cross = [], []
    for k in range(nnz):
        pre, post = int(pre_of[k]), int(post_of[k])
        if pre not in ca3_set or post not in ca3_set:
            continue
        pm, qm = asm_of.get(pre), asm_of.get(post)
        if pm is not None and qm is not None and pm == qm:
            within.append(float(data[k]))
        elif pm is not None:                                # assembly-member -> other CA3 cell
            cross.append(float(data[k]))
    mean = lambda a: float(np.mean(a)) if a else 0.0
    w, c = mean(within), mean(cross)
    return {"within": w, "cross": c, "ratio": (w / c if c > 0 else 0.0), "n_within": len(within)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-assembly", type=int, default=12)
    ap.add_argument("--n-ca3", type=int, default=150, help="CA3 size (bigger -> sparser assembly fraction, Kopsick ~0.37%)")
    ap.add_argument("--presentations", type=int, default=60)
    ap.add_argument("--drive-pA", type=float, default=600.0)
    ap.add_argument("--hebb-lr", type=float, default=5.0)
    ap.add_argument("--gamma-on", type=int, default=8)
    ap.add_argument("--gamma-off", type=int, default=12)
    ap.add_argument("--ca3-fb-inhib", type=float, default=None, help="feedback inhibition to suppress recurrent spillover to non-members")
    ap.add_argument("--hebb-max", type=float, default=None, help="hebbian_max_weight (raise -> within-assembly ceiling higher -> ratio past 3x toward 10x)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"[R-iii CA3 DIRECT-ASSEMBLY (Kopsick 2024)] n_assembly={a.n_assembly} presentations={a.presentations} "
          f"gamma_on={a.gamma_on} drive={a.drive_pA} | within-assembly vs cross recurrent weight", flush=True)
    import json
    rows = []
    kw = dict(n_assembly=a.n_assembly, n_ca3=a.n_ca3, presentations=a.presentations, drive_pA=a.drive_pA, hebb_lr=a.hebb_lr,
              gamma_on=a.gamma_on, gamma_off=a.gamma_off, ca3_fb_inhib=a.ca3_fb_inhib, hebb_max=a.hebb_max)
    for s in seeds:
        t0 = time.time()
        tr = run_seed(s, do_train=True, **kw)
        no = run_seed(s, do_train=False, **kw)
        pm = run_seed(s, do_train=True, permute=True, **kw)
        row = {"seed": s, "within": tr["within"], "cross": tr["cross"], "ratio": tr["ratio"],
               "notrain_ratio": no["ratio"], "permuted_ratio": pm["ratio"]}
        rows.append(row)
        print(f"  [seed {s}] within={tr['within']:.2f} cross={tr['cross']:.2f} RATIO={tr['ratio']:.2f}x | "
              f"no-train={no['ratio']:.2f}x permuted={pm['ratio']:.2f}x ({time.time()-t0:.0f}s)", flush=True)
    if a.json and rows:
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        r = [x["ratio"] for x in rows]; nt = [x["notrain_ratio"] for x in rows]; pmr = [x["permuted_ratio"] for x in rows]
        go = all(x > 3.0 for x in r) and all(x < 1.5 for x in nt) and all(x < 1.5 for x in pmr)
        print(f"\n  AGGREGATE: within/cross RATIO={np.mean(r):.2f}x | no-train={np.mean(nt):.2f}x permuted={np.mean(pmr):.2f}x", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the DIRECT sparse SYNCHRONOUS assembly protocol (Kopsick) forms a STRONG specific CA3 attractor (within-assembly >> cross, >=3x) that the trisynaptic-routed distributed code could not (plateaued ~1.4x); no-train + permuted collapse -> the potentiation rode the trained synchronous assembly -> the recurrent autoassociator FORMS; next: the CYCLE-1068 dendritic completion on THIS learned attractor = emergent CA3 completion' if go else 'the ratio is not yet >=3x with clean controls; sweep gamma_on/drive/presentations'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
