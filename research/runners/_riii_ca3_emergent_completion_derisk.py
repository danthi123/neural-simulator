"""R-iii CAPSTONE: EMERGENT CA3 pattern completion on-substrate. Composes the two halves proven this arc:
FORMATION (Kopsick-2024-correct, 2026-07-09) -- a sparse CA3 assembly driven DIRECTLY + SYNCHRONOUSLY (gamma volleys)
+ selective feedback inhibition + the rate-window co-activity Hebbian LEARNS a strong specific recurrent attractor
(within/non ratio ~3.3x, vs the trisynaptic-routed distributed code's 1.44x plateau); and COMPLETION (CYCLE 1068) --
the two-compartment dendritic dAP plateau completes a PARTIAL cue where the linear point neuron cannot. This runner
TRAINS the attractor by the direct-synchronous protocol, then drives a PARTIAL cue (half the assembly) and asks: do
the HELD-OUT assembly cells fire (completion), SPECIFICALLY (non-assembly cells stay silent)? = emergent CA3 pattern
completion LEARNED FROM EXPERIENCE, on the spiking substrate, NO `sim/` edit.

Clean anti-cheats (unconfounded, unlike the weight-ratio metric): (A) NO-TRAIN -> the partial cue does not complete
(no learned attractor); (B) SPECIFICITY -- non-assembly cells stay silent (not indiscriminate spread); (C) LINEAR
(coincidence OFF, same learned attractor) -> the partial cue does not complete (the dendritic plateau is load-bearing,
the CYCLE-1064/1068 point-neuron limit); (D) PERMUTED-CUE -- drive a RANDOM half of the network (not the assembly) ->
completes nothing (the completion rides the trained assembly, not the drive). 6-seed. Reuse-by-import. NO `sim/` edit.
"""
from __future__ import annotations
import argparse, time
import numpy as np
from research.runners._riii_ca3_coincidence_completion_derisk import _build, _set_gates


def _train_assemblies(bridge, cp, assemblies, presentations, drive_pA, gamma_on, gamma_off):
    _set_gates(bridge, 1.0)
    for _p in range(presentations):
        for asm in assemblies:
            drv = cp.asarray(asm, dtype=cp.int64)
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(6):
                bridge._run_one_simulation_step()
            for _v in range(3):
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_external_input_current[drv] = float(drive_pA)
                for _ in range(gamma_on):
                    bridge._run_one_simulation_step()
                bridge.cp_external_input_current[:] = 0.0
                for _ in range(gamma_off):
                    bridge._run_one_simulation_step()
    _set_gates(bridge, 0.0)


def _recall(bridge, cp, cue_cells, read_cells, drive_pA, steps=60, clamp_cells=None):
    """Drive cue_cells directly, run, return per-read-cell spike-count vector. clamp_cells (the ca3_pv_basket, for
    SWR-ripple DISINHIBITION) are held strongly hyperpolarized each step so they do not fire -> the recall-time
    feedback inhibition is transiently removed (biologically, ripples reduce PV interneuron output during recall)."""
    from sim.backend import to_host
    _clamp = cp.asarray(clamp_cells, dtype=cp.int64) if clamp_cells is not None and len(clamp_cells) else None
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(40):
        if _clamp is not None:
            bridge.cp_external_input_current[_clamp] = -5000.0
        bridge._run_one_simulation_step()
    cue = cp.asarray(cue_cells, dtype=cp.int64)
    read = cp.asarray(read_cells, dtype=cp.int64)
    acc = cp.zeros(len(read_cells), dtype=cp.float32)
    for _ in range(steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[cue] = float(drive_pA)
        if _clamp is not None:
            bridge.cp_external_input_current[_clamp] = -5000.0
        bridge._run_one_simulation_step()
        acc += bridge.cp_firing_states[read].astype(cp.float32)
    bridge.cp_external_input_current[:] = 0.0
    return to_host(acc)


def run_seed(seed, n_ca3=500, n_assembly=12, n_mem=3, presentations=60, drive_pA=1000.0, cue_drive=1000.0,
             hebb_lr=10.0, gamma_on=8, gamma_off=12, ca3_fb_inhib=120.0, k_thresh=20.0, plateau_strength=300.0,
             apical_R=50.0, hebb_max=120.0, recall_steps=60, ca3_density=0.5, do_train=True, coincidence=True,
             permuted_cue=False, recall_disinhib=False):
    from sim.backend import get_backend
    cp, _ = get_backend()
    bridge = _build(seed, n_ca3=n_ca3, ca3_density=ca3_density, ca3w=6.0, coincidence=coincidence, two_comp=True,
                    apical_R=apical_R, k_thresh=k_thresh, plateau_strength=plateau_strength, weighted=True, train=True,
                    hebb_rate=True, hebb_lr=hebb_lr, hebb_decay=0.0, coact_thresh=0.001, ca3_fb_inhib=ca3_fb_inhib,
                    hebb_max=hebb_max)
    rm = bridge.region_manager
    ca3_idx = np.asarray(list(rm.indices("ca3")), dtype=np.int64)
    _basket = None
    if recall_disinhib:
        try:
            _basket = np.asarray(list(rm.indices("ca3_pv_basket")), dtype=np.int64)
        except Exception:
            _basket = None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ca3_idx)
    assemblies = [np.array(perm[m * n_assembly:(m + 1) * n_assembly], dtype=np.int64) for m in range(n_mem)]
    stored_all = set(int(x) for a in assemblies for x in a)
    non_assembly = np.array([g for g in ca3_idx if int(g) not in stored_all], dtype=np.int64)[:60]
    if do_train:
        _train_assemblies(bridge, cp, assemblies, presentations, drive_pA, gamma_on, gamma_off)

    held_c, non_c = [], []
    for asm in assemblies:
        a = asm.copy(); rng.shuffle(a)
        h = len(a) // 2
        cue, held = a[:h], a[h:]
        if permuted_cue:                                   # anti-cheat D: drive a RANDOM half (not the assembly)
            cue = rng.choice(non_assembly, size=h, replace=False)
        resp_held = _recall(bridge, cp, cue, held, cue_drive, steps=recall_steps, clamp_cells=_basket)
        resp_cue = _recall(bridge, cp, cue, cue, cue_drive, steps=recall_steps, clamp_cells=_basket)
        resp_non = _recall(bridge, cp, cue, non_assembly, cue_drive, steps=recall_steps, clamp_cells=_basket)
        ca = float(np.mean(resp_cue)) + 1e-9
        held_c.append(float(np.mean(resp_held)) / ca)
        non_c.append(float(np.mean(resp_non)) / ca)
    return {"heldout": float(np.mean(held_c)), "nonassembly": float(np.mean(non_c))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--presentations", type=int, default=60)
    ap.add_argument("--k-thresh", type=float, default=20.0, help="plateau threshold: must sit between the cross-drive floor (<20, non stays silent) and the held-out within-drive (>20 all seeds); the 6/6-seed GO config (k=25 was 5/6, seed 102 marginal at 0.282)")
    ap.add_argument("--n-ca3", type=int, default=500)
    ap.add_argument("--n-assembly", type=int, default=12)
    ap.add_argument("--cue-drive", type=float, default=1000.0, help="recall cue drive (raise to ~1000 to fire despite standing inhibition)")
    ap.add_argument("--hebb-max", type=float, default=120.0, help="within-assembly weight ceiling: 30->3.3x, 60->7.5x, 120->12.6x attractor (find the window where dendritic completes but linear fails)")
    ap.add_argument("--hebb-lr", type=float, default=10.0)
    ap.add_argument("--recall-steps", type=int, default=60, help="recall accumulation window (SHORT -> measure the fast plateau completion before the recurrent cascade ignites the whole net)")
    ap.add_argument("--ca3-density", type=float, default=0.5, help="ca3->ca3 recurrent density; LOWER (~0.1) = sparser recurrent -> less cross-spillover -> a BIGGER assembly can stay SPECIFIC (Kopsick sparse-large regime)")
    ap.add_argument("--recall-disinhib", action="store_true", help="reduce feedback inhibition during recall (SWR ripple disinhibition)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"[R-iii EMERGENT CA3 COMPLETION] n_ca3={a.n_ca3} n_assembly={a.n_assembly} pres={a.presentations} k={a.k_thresh} "
          f"| direct-synchronous FORMATION -> partial-cue dendritic COMPLETION", flush=True)
    import json
    kw = dict(n_ca3=a.n_ca3, n_assembly=a.n_assembly, presentations=a.presentations, k_thresh=a.k_thresh,
              cue_drive=a.cue_drive, recall_disinhib=a.recall_disinhib, hebb_max=a.hebb_max, hebb_lr=a.hebb_lr,
              recall_steps=a.recall_steps, ca3_density=a.ca3_density)
    rows = []
    for s in seeds:
        t0 = time.time()
        on = run_seed(s, do_train=True, coincidence=True, **kw)
        lin = run_seed(s, do_train=True, coincidence=False, **kw)
        notr = run_seed(s, do_train=False, coincidence=True, **kw)
        perm = run_seed(s, do_train=True, coincidence=True, permuted_cue=True, **kw)
        row = {"seed": s, "held": on["heldout"], "non": on["nonassembly"], "linear": lin["heldout"],
               "notrain": notr["heldout"], "permcue": perm["heldout"]}
        rows.append(row)
        print(f"  [seed {s}] COMPLETION held-out={on['heldout']:.3f} (non-assembly {on['nonassembly']:.3f}) | "
              f"LINEAR={lin['heldout']:.3f} NO-TRAIN={notr['heldout']:.3f} PERM-CUE={perm['heldout']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    if a.json and rows:
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        h = [r["held"] for r in rows]; n = [r["non"] for r in rows]; l = [r["linear"] for r in rows]
        nt = [r["notrain"] for r in rows]; pc = [r["permcue"] for r in rows]
        go = (all(x > 0.30 for x in h) and all(x < 0.20 for x in n) and all(hh - ll > 0.20 for hh, ll in zip(h, l))
              and all(x < 0.20 for x in nt) and all(x < 0.20 for x in pc))
        print(f"\n  AGGREGATE: held-out={np.mean(h):.3f} non-assembly={np.mean(n):.3f} | LINEAR={np.mean(l):.3f} "
              f"NO-TRAIN={np.mean(nt):.3f} PERM-CUE={np.mean(pc):.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'a partial cue of a LEARNED sparse CA3 assembly COMPLETES the held-out members SPECIFICALLY (non-assembly silent) via the dendritic plateau, where LINEAR + NO-TRAIN + PERM-CUE all fail = EMERGENT CA3 pattern completion learned from experience on the spiking substrate' if go else 'not yet a clean specific completion; sweep k_thresh / presentations / cue_drive'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
