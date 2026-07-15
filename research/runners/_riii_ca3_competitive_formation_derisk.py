"""R-iii CA3 attractor FORMATION via COMPETITIVE-HEBBIAN learning (2026-07-14, boundary-surpass of the 2026-07-09
formation-saturation blocker). The 2026-07-09 result: ALL FOUR pure-LTP rules (causal-offset / symmetric / rate-window
BCM), even with feedback-inhibition + gamma-sync + mossy detonator, form only a WEAK ~1.44x within-ensemble separation
(member->silent grows in LOCKSTEP with member->member -> the H2 "NON-SPECIFIC RULE" verdict, reproduced first-hand
2026-07-14 at separation -0.01). Root cause (Zenke-Agnes-Gerstner 2015 Nat Commun 6:6922; Litwin-Kumar & Doiron 2014):
pure homosynaptic LTP has NO term coupling within-assembly potentiation to DEPRESSION of the same cell's other
synapses, so in the distributed 35-47%-active code the "silent" non-members co-fire enough to potentiate in lockstep.

THE MECHANISM (the boundary-surpassing deep-research gate's recommendation, adversarially verified 2026-07-14):
COMPETITIVE-HEBBIAN = the committed EMERGE-40 heterosynaptic winner-inactive DEPRESSION kernel
(`sim/kernels.fused_htm_winner_inactive_depression`, verified sim/kernels.py:432) applied to the ca3->ca3 RECURRENT
weights for the FIRST TIME (it has only ever run FEEDFORWARD in EMERGE-38/39). Alongside the bridge's existing
rate-window LTP, each encoding window we DEPRESS the recurrent synapses of a fired cell to/from cells that did NOT
co-fire (both directions -> the kernel called twice with swapped args). Net = LTP lifts member->member UP while the
heterosynaptic term forces member->silent DOWN -> a BIMODAL (winner-take-all-in-weight-space) SELECTIVE attractor.
The competition DOWN-term is the load-bearing NEW ingredient (supplies the per-postsynaptic competition the pure-LTP
rules lacked); it is genuinely new (a-1 confirmed: the entire tried set is pure LTP). NO `sim/` edit -- the committed
kernel is imported + applied runner-side to `cp_connections.data` on the ca3->ca3 coincidence synapses.

GO BAR 1 (formation): within/silent ratio clears ~3x (vs 1.44x pure-Hebbian) AND the break comes from member->silent
DROPPING below init (not just within rising) -- the mechanism-discrimination anti-cheat separating competition from
the known hebbian_max weight-ceiling lever. ANTI-CHEAT: lam_dep_wi=0 must revert to ~1.44x (the competition term is
load-bearing). Reuse-by-import of the CYCLE-1066 harness (`_riii_ca3_attractor_diag`). Extends that diagnostic.
"""
from __future__ import annotations
import argparse, time
import numpy as np
from research.runners._riii_ca3_coincidence_completion_derisk import _build, _set_gates
from research.runners.validate_trisynaptic_loop import build_drive_pattern


def _extract_ca3ca3_coincidence(bridge, ca3_idx, to_host):
    """One-time (structure is fixed during training): flat CSR positions + local ca3 indices of the ca3->ca3
    coincidence-masked recurrent synapses. Returns (flat_pos, pre_local, post_local) as host int arrays."""
    conn = bridge.cp_connections
    nnz = int(conn.nnz)
    mask = to_host(bridge.cp_coincidence_synapse_mask[:nnz]).astype(bool)
    indptr = to_host(conn.indptr); indices = to_host(conn.indices)
    pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1     # row (pre) for each flat synapse
    post_of = indices[:nnz]
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}
    ca3_set = set(ca3_pos.keys())
    flat, pre_l, post_l = [], [], []
    for k in range(nnz):
        if not mask[k]:
            continue
        pre, post = int(pre_of[k]), int(post_of[k])
        if pre in ca3_set and post in ca3_set:
            flat.append(k); pre_l.append(ca3_pos[pre]); post_l.append(ca3_pos[post])
    return (np.asarray(flat, dtype=np.int64), np.asarray(pre_l, dtype=np.int64), np.asarray(post_l, dtype=np.int64))


def run(seed=42, n_mem=2, train_events=150, drive_pA=200.0, n_lang=384, n_ca3=150, n_dg=300,
        ca3_density=0.5, ca3_weight=6.0, hebb_max=30.0, hebb_lr=None, hebb_rate=True,
        coact_decay=None, coact_thresh=None, ca3_fb_inhib=None, ca3_fb_n=None, mossy_weight=None,
        sync_on=None, sync_off=None, reset_steps=15, drive_steps=55,
        lam_dep_wi=0.0, comp_both_dir=True, fire_thresh=1):
    from sim.backend import get_backend, to_host, get_sparse_module
    from sim.kernels import fused_htm_winner_inactive_depression
    cp, _ = get_backend()
    bridge = _build(seed, n_lang=n_lang, n_ca3=n_ca3, n_dg=n_dg, ca3_density=ca3_density, ca3w=ca3_weight,
                    coincidence=True, weighted=True, train=True, hebb_max=hebb_max, hebb_lr=hebb_lr,
                    hebb_rate=hebb_rate, coact_decay=coact_decay, coact_thresh=coact_thresh,
                    ca3_fb_inhib=ca3_fb_inhib, ca3_fb_n=ca3_fb_n, mossy_weight=mossy_weight)
    rm = bridge.region_manager
    lang = np.asarray(list(rm.indices("language_input")), dtype=np.int64)
    ca3_idx = list(rm.indices("ca3")); ca3_arr = cp.asarray(ca3_idx, dtype=cp.int64)
    n_lang = len(lang)
    patterns = [build_drive_pattern(n_neurons=n_lang, sparsity=0.1, seed=seed * 100 + m) for m in range(n_mem)]

    # ---- COMPETITIVE-HEBBIAN setup: cache the ca3->ca3 coincidence synapse flat positions + local indices ----
    conn = bridge.cp_connections
    flat_h, pre_l_h, post_l_h = _extract_ca3ca3_coincidence(bridge, ca3_idx, to_host)
    do_comp = lam_dep_wi > 0.0 and len(flat_h) > 0
    if do_comp:
        flat_pos = cp.asarray(flat_h, dtype=cp.int64)
        pre_local = cp.asarray(pre_l_h, dtype=cp.int64)
        post_local = cp.asarray(post_l_h, dtype=cp.int64)

    def _apply_competition(win_fire):
        """Heterosynaptic winner-inactive depression on the ca3->ca3 recurrents for THIS window.
        win_fire: per-ca3-cell spike count this window. fired = (>= fire_thresh)."""
        fired = (win_fire >= fire_thresh).astype(cp.float32)           # per ca3 cell: co-fired this window
        fpre = fired[pre_local]; fpost = fired[post_local]              # per synapse
        w = conn.data[flat_pos]
        # Direction 1 (kernel as-is): depress SILENT-pre -> WINNER-post  (dep = (1-fpre)*fpost*lam)
        w = fused_htm_winner_inactive_depression(w, fpre, fpost, float(lam_dep_wi), 0.0, float(hebb_max))
        if comp_both_dir:
            # Direction 2 (swapped): depress WINNER-pre -> SILENT-post  (dep = (1-fpost)*fpre*lam) -> lowers member->silent
            w = fused_htm_winner_inactive_depression(w, fpost, fpre, float(lam_dep_wi), 0.0, float(hebb_max))
        conn.data[flat_pos] = w

    stored = {}
    global_act = np.zeros(len(ca3_idx))
    _set_gates(bridge, 1.0)
    rec_last = min(10, max(1, train_events // 3))
    for m, pat in enumerate(patterns):
        drv = cp.asarray(lang[pat], dtype=cp.int64)
        spikes = cp.zeros(len(ca3_idx), dtype=cp.float32)
        for ev in range(train_events):
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
            rec = ev >= train_events - rec_last
            win_fire = cp.zeros(len(ca3_idx), dtype=cp.float32)         # this-event CA3 co-firing (for competition)
            if sync_on is not None:
                _period = int(sync_on) + int(sync_off)
                for _st in range(drive_steps):
                    bridge.cp_external_input_current[:] = 0.0
                    if (_st % _period) < int(sync_on):
                        bridge.cp_external_input_current[drv] = float(drive_pA)
                    bridge._run_one_simulation_step()
                    f = bridge.cp_firing_states[ca3_arr].astype(cp.float32)
                    win_fire += f
                    if rec:
                        spikes += f
            else:
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_external_input_current[drv] = float(drive_pA)
                for _ in range(drive_steps):
                    bridge._run_one_simulation_step()
                    f = bridge.cp_firing_states[ca3_arr].astype(cp.float32)
                    win_fire += f
                    if rec:
                        spikes += f
            if do_comp:
                _apply_competition(win_fire)                            # heterosynaptic competition, once per event
        bridge.cp_external_input_current[:] = 0.0
        sp = to_host(spikes); global_act += sp
        n_stored = max(4, int(0.10 * len(ca3_idx)))
        top = np.argsort(-sp)[:n_stored]; top = top[sp[top] > 0]
        stored[m] = np.array([ca3_idx[i] for i in top], dtype=np.int64)
    _set_gates(bridge, 0.0)

    # ---- SPARSITY + weight-structure diagnostic (identical to _riii_ca3_attractor_diag) ----
    peak = global_act.max() + 1e-9
    frac_active = float(np.mean(global_act > 0.1 * peak))
    n_active = int(np.sum(global_act > 0.1 * peak))
    nnz = int(conn.nnz)
    mask = to_host(bridge.cp_coincidence_synapse_mask[:nnz]).astype(bool)
    indptr = to_host(conn.indptr); indices = to_host(conn.indices); data = to_host(conn.data[:nnz])
    pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
    post_of = indices[:nnz]
    ens_of = {}
    for m in range(n_mem):
        for g in stored[m]:
            ens_of[int(g)] = m
    ca3_set = set(int(x) for x in ca3_idx)
    silent_thresh = np.quantile(global_act, 0.25)
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
            within.append(w)
        elif pe is not None and post in silent_set:
            mem_to_silent.append(w)
        elif pe is not None:
            mem_to_other.append(w)
    mean = lambda a: float(np.mean(a)) if a else 0.0
    w_in, w_sil = mean(within), mean(mem_to_silent)
    ratio = w_in / (w_sil + 1e-9)
    sep = w_in - w_sil
    print(f"[R-iii competitive formation] seed {seed} lam_dep_wi={lam_dep_wi} both_dir={comp_both_dir} "
          f"train_events={train_events} n_comp_syn={len(flat_h)}", flush=True)
    print(f"  SPARSITY: {n_active}/{len(ca3_idx)} CA3 active = {frac_active:.2f}", flush=True)
    print(f"  ca3->ca3 recurrent weight (init {ca3_weight}):", flush=True)
    print(f"    within-ensemble (member->member) = {w_in:.2f}  (n={len(within)})", flush=True)
    print(f"    member->TRULY-SILENT             = {w_sil:.2f}  (n={len(mem_to_silent)})", flush=True)
    print(f"    member->other(fired-somewhat)    = {mean(mem_to_other):.2f}  (n={len(mem_to_other)})", flush=True)
    print(f"    ratio within/silent = {ratio:.2f}   separation = {sep:+.2f}", flush=True)
    # GO BAR 1: ratio >= 3x AND member->silent DROPPED below init (competition, not just weight-ceiling)
    silent_dropped = w_sil < ca3_weight
    go = ratio >= 3.0 and silent_dropped
    if go:
        verdict = (f"GO: ratio {ratio:.2f} >= 3x AND member->silent {w_sil:.2f} DROPPED below init {ca3_weight} "
                   f"-> SELECTIVE attractor formed by competition")
    elif ratio >= 3.0:
        verdict = (f"PARTIAL: ratio {ratio:.2f} >= 3x but member->silent {w_sil:.2f} did NOT drop below init "
                   f"{ca3_weight} -> gain is weight-ceiling, not competition (mechanism-discrimination anti-cheat)")
    else:
        verdict = f"BOUNDARY: ratio {ratio:.2f} < 3x -> competition did not form a selective attractor at this lam"
    print(f"  VERDICT -> {verdict}", flush=True)
    return {"seed": seed, "lam_dep_wi": lam_dep_wi, "both_dir": comp_both_dir, "frac_active": frac_active,
            "within": w_in, "mem_to_silent": w_sil, "mem_to_other": mean(mem_to_other), "ratio": ratio,
            "sep": sep, "silent_dropped": silent_dropped, "go": go, "n_comp_syn": len(flat_h)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-events", type=int, default=150)
    ap.add_argument("--hebb-max", type=float, default=30.0)
    ap.add_argument("--hebb-lr", type=float, default=None)
    ap.add_argument("--ca3-density", type=float, default=0.5)
    ap.add_argument("--drive-pA", type=float, default=200.0)
    ap.add_argument("--no-hebb-rate", action="store_true", help="disable the rate-window LTP (competition-only test)")
    ap.add_argument("--coact-decay", type=float, default=None)
    ap.add_argument("--coact-thresh", type=float, default=None)
    ap.add_argument("--ca3-fb-inhib", type=float, default=None)
    ap.add_argument("--ca3-fb-n", type=int, default=None)
    ap.add_argument("--mossy-weight", type=float, default=None)
    ap.add_argument("--sync-on", type=int, default=None)
    ap.add_argument("--sync-off", type=int, default=None)
    ap.add_argument("--lam-dep-wi", type=float, default=0.0, help="heterosynaptic winner-inactive depression rate (0=OFF=anti-cheat control)")
    ap.add_argument("--one-dir", action="store_true", help="apply competition in one direction only (default both)")
    ap.add_argument("--fire-thresh", type=int, default=1, help="per-window spike count for a ca3 cell to count as 'co-fired'")
    a = ap.parse_args()
    t0 = time.time()
    run(seed=a.seed, train_events=a.train_events, hebb_max=a.hebb_max, hebb_lr=a.hebb_lr,
        ca3_density=a.ca3_density, drive_pA=a.drive_pA, hebb_rate=not a.no_hebb_rate,
        coact_decay=a.coact_decay, coact_thresh=a.coact_thresh, ca3_fb_inhib=a.ca3_fb_inhib,
        ca3_fb_n=a.ca3_fb_n, mossy_weight=a.mossy_weight, sync_on=a.sync_on, sync_off=a.sync_off,
        lam_dep_wi=a.lam_dep_wi, comp_both_dir=not a.one_dir, fire_thresh=a.fire_thresh)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
