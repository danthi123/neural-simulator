"""R-iii attractor-formation diagnostic (CYCLE 1066 follow-on): WHY does the ca3->ca3 rate-Hebbian potentiate the
recurrents UNIFORMLY (c_drive held ~= nonstored ~17.5 after the hebbian_max_weight fix) instead of writing a
SPECIFIC within-ensemble attractor? Two hypotheses this run distinguishes decisively:
  (H1) DIFFUSE ensemble: too many CA3 neurons fire during training (not a sparse 15-of-150 ensemble), so member->
       "non-stored" pairs actually co-fire and potentiate -> no specificity. Fix = sparsify CA3 (D.12).
  (H2) NON-COINCIDENT rule: the Hebbian potentiates broadly (post-only / not pre*post) so member->silent grows too.
       Fix = a genuinely coincident rule.
Distinguisher: measure (a) CA3 training SPARSITY (how many neurons fire), and (b) the WITHIN-ensemble recurrent
weight vs the member->TRULY-SILENT (bottom-firing) recurrent weight. If within-ensemble >> member->silent, the
attractor DID form and the earlier "nonstored" sample just caught diffuse-firing neurons (H1). If member->silent is
also high, the rule is non-specific (H2). Reuse-by-import of the CYCLE-1066 harness. NO `sim/` edit.
"""
from __future__ import annotations
import argparse, time
import numpy as np
from research.runners._riii_ca3_coincidence_completion_derisk import _build, _set_gates
from research.runners.validate_trisynaptic_loop import build_drive_pattern


def run(seed=42, n_mem=2, train_events=100, drive_pA=200.0, n_lang=384, n_ca3=150, n_dg=300,
        ca3_density=0.5, ca3_weight=6.0, hebb_max=30.0, hebb_lr=None, hebb_decay=None, hebb_sym=False, reset_steps=15, drive_steps=55):
    from sim.backend import get_backend, to_host, get_sparse_module
    cp, _ = get_backend()
    csp = get_sparse_module()
    bridge = _build(seed, n_lang=n_lang, n_ca3=n_ca3, n_dg=n_dg, ca3_density=ca3_density, ca3w=ca3_weight,
                    coincidence=True, weighted=True, train=True, hebb_max=hebb_max, hebb_lr=hebb_lr,
                    hebb_decay=hebb_decay, hebb_sym=hebb_sym)
    rm = bridge.region_manager
    lang = np.asarray(list(rm.indices("language_input")), dtype=np.int64)
    ca3_idx = list(rm.indices("ca3")); ca3_arr = cp.asarray(ca3_idx, dtype=cp.int64)
    n_lang = len(lang)
    patterns = [build_drive_pattern(n_neurons=n_lang, sparsity=0.1, seed=seed * 100 + m) for m in range(n_mem)]
    stored = {}
    global_act = np.zeros(len(ca3_idx))                       # cumulative CA3 activity across ALL training (sparsity read)
    _set_gates(bridge, 1.0)
    rec_last = min(10, max(1, train_events // 3))
    for m, pat in enumerate(patterns):
        drv = cp.asarray(lang[pat], dtype=cp.int64)
        spikes = cp.zeros(len(ca3_idx), dtype=cp.float32)
        for ev in range(train_events):
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[drv] = float(drive_pA)
            rec = ev >= train_events - rec_last
            for _ in range(drive_steps):
                bridge._run_one_simulation_step()
                if rec:
                    spikes += bridge.cp_firing_states[ca3_arr].astype(cp.float32)
        bridge.cp_external_input_current[:] = 0.0
        sp = to_host(spikes); global_act += sp
        n_stored = max(4, int(0.10 * len(ca3_idx)))
        top = np.argsort(-sp)[:n_stored]; top = top[sp[top] > 0]
        stored[m] = np.array([ca3_idx[i] for i in top], dtype=np.int64)
    _set_gates(bridge, 0.0)

    # SPARSITY: fraction of CA3 firing above 10% of the peak during the (last) drive window
    peak = global_act.max() + 1e-9
    frac_active = float(np.mean(global_act > 0.1 * peak))
    n_active = int(np.sum(global_act > 0.1 * peak))

    # WEIGHT STRUCTURE: extract the ca3->ca3 coincidence-masked synapses (pre=row, post=col in cp_connections CSR),
    # classify by ensemble membership of BOTH endpoints, and read the current (trained) weight.
    conn = bridge.cp_connections
    nnz = int(conn.nnz)
    mask = to_host(bridge.cp_coincidence_synapse_mask[:nnz]).astype(bool)
    indptr = to_host(conn.indptr); indices = to_host(conn.indices); data = to_host(conn.data[:nnz])
    pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1     # row (pre) for each flat synapse
    post_of = indices[:nnz]
    ens_of = {}                                                            # global CA3 idx -> memory id
    for m in range(n_mem):
        for g in stored[m]:
            ens_of[int(g)] = m
    ca3_set = set(int(x) for x in ca3_idx)
    silent_thresh = np.quantile(global_act, 0.25)                          # bottom-quartile CA3 = "truly silent"
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}
    silent_set = set(int(ca3_idx[i]) for i in range(len(ca3_idx)) if global_act[i] <= silent_thresh)

    within, mem_to_silent, mem_to_other = [], [], []
    for k in range(nnz):
        if not mask[k]:
            continue
        pre, post = int(pre_of[k]), int(post_of[k])
        if pre not in ca3_set or post not in ca3_set:
            continue
        w = float(data[k])
        pe, po = ens_of.get(pre), ens_of.get(post)
        if pe is not None and po is not None and pe == po:
            within.append(w)                                               # member -> same-ensemble member
        elif pe is not None and post in silent_set:
            mem_to_silent.append(w)                                        # member -> truly-silent neuron
        elif pe is not None:
            mem_to_other.append(w)                                         # member -> other (fired-somewhat) neuron
    mean = lambda a: float(np.mean(a)) if a else 0.0
    print(f"[R-iii attractor diag] seed {seed} train_events={train_events} hebb_max={hebb_max} hebb_lr={hebb_lr}", flush=True)
    print(f"  SPARSITY: {n_active}/{len(ca3_idx)} CA3 active (>10% peak) = {frac_active:.2f}  (sparse attractor wants << 0.30)", flush=True)
    print(f"  ca3->ca3 recurrent weight (init {ca3_weight}):", flush=True)
    print(f"    within-ensemble (member->member) = {mean(within):.2f}  (n={len(within)})", flush=True)
    print(f"    member->TRULY-SILENT             = {mean(mem_to_silent):.2f}  (n={len(mem_to_silent)})", flush=True)
    print(f"    member->other(fired-somewhat)    = {mean(mem_to_other):.2f}  (n={len(mem_to_other)})", flush=True)
    sep = mean(within) - mean(mem_to_silent)
    if frac_active > 0.30 and mean(within) > 1.5 * (mean(mem_to_silent) + 1e-9):
        verdict = "H1 DIFFUSE: within-ensemble IS potentiated above member->silent, but the ensemble is NOT sparse -> non-members co-fire; FIX = sparsify CA3 (D.12)"
    elif mean(within) <= 1.5 * (mean(mem_to_silent) + 1e-9):
        verdict = "H2 NON-SPECIFIC RULE: member->silent grows ~as much as within-ensemble -> the Hebbian is not selectively coincident; FIX = a genuinely pre*post-coincident potentiation"
    else:
        verdict = f"ATTRACTOR FORMED: within-ensemble ({mean(within):.1f}) >> member->silent ({mean(mem_to_silent):.1f}) and ensemble is sparse ({frac_active:.2f})"
    print(f"  WITHIN-vs-SILENT separation = {sep:+.2f} -> {verdict}", flush=True)
    return {"frac_active": frac_active, "within": mean(within), "mem_to_silent": mean(mem_to_silent),
            "mem_to_other": mean(mem_to_other), "sep": sep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-events", type=int, default=100)
    ap.add_argument("--hebb-max", type=float, default=30.0)
    ap.add_argument("--hebb-lr", type=float, default=None)
    ap.add_argument("--ca3-density", type=float, default=0.5)
    ap.add_argument("--drive-pA", type=float, default=200.0, help="encoding drive: LOWER -> fewer CA3 fire -> sparser ensemble -> more within-ensemble-specific coincidence (D.12 sparsity lever)")
    ap.add_argument("--hebb-decay", type=float, default=None, help="hebbian_weight_decay; set 0 to test offset-vs-decay")
    ap.add_argument("--hebb-sym", action="store_true", help="SYMMETRIC (offset-free) co-activity Hebbian -- the CA3 attractor-formation fix")
    a = ap.parse_args()
    t0 = time.time()
    run(seed=a.seed, train_events=a.train_events, hebb_max=a.hebb_max, hebb_lr=a.hebb_lr,
        ca3_density=a.ca3_density, drive_pA=a.drive_pA, hebb_decay=a.hebb_decay, hebb_sym=a.hebb_sym)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
