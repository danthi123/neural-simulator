#!/usr/bin/env python
"""Does a reader population LEARN place tuning from a moving sweep under COMPETITIVE normalisation?

MECHANISM UNDER TEST: Oja (multiplicative) vs Miller-MacKay (subtractive) normalisation.
Plain Hebbian grows every co-active synapse, so tuning blurs. Competitive normalisation makes
potentiating one synapse COST the others.

WHAT THIS PROBE REFUSES TO DO (each rule earned by a retraction on 2026-07-28):
  R1. lr=0 IS AN ARM. Every selectivity number is printed beside its lr=0 twin at the SAME seed and
      the SAME wiring density. The claim is the DIFFERENCE, never the absolute.
  R2. dW != 0 proves nothing. Only delta-vs-lr0 counts.
  R3. The read population MUST FIRE. Read-region spikes are counted and printed; 0 spikes => VOID
      (Hebbian needs a POSTsynaptic spike), not "negative".
  R4. Selectivity tracks WIRING. Density is FIXED across every arm; peak/mean is reported under the
      same convention (over all n_place inputs, zeros included) for lr0 and trained alike.
  R5. Bounds are checked and printed against the design weight.
  R6. The initial selectivity of the trained arm is asserted == the lr0 arm (same seed => same wiring),
      so a "gain" cannot be a different random draw.

HONEST SCAFFOLD DISCLOSURE: the learning rule is computed HOST-SIDE (read b.cp_connections.data,
compute dw, write back). The pre/post factors are exponential traces of the bridge's ACTUAL spikes
(b.cp_firing_states) -- not a host-invented rate. The subtractive form mirrors the engine's own
cfg.btsp_mean_subtract formula (sim/bridge.py:8153, per-POSTsynaptic-cell mean of the increment),
which could not be reused directly here because that code path is gated behind the BTSP
eligibility/plateau signals and a btsp_w_max default of 5.0 (our design weight is 250).
This is a test of whether the RULE works, not a claim that the rule is on-substrate.
"""
import os, sys, json, argparse, time

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")

import numpy as np
import logging
logging.disable(logging.INFO)

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.regions import BrainRegion, RegionPathway
from sim import SimulationBridge

N_PLACE = 60
N_READ = 12
SWEEP_LEN = 60
BUMP_SIGMA = 5.0
BUMP_PA = 3000.0
W_MEAN = 250.0
W_JITTER = 0.3
W_MIN = 0.0
W_MAX = 750.0          # R5: 3x the design weight of 250 -> a "potentiation" is never negative
TRACE_TAU = 20.0       # ms; exponential filter of ACTUAL spikes -> the pre/post factors


def build(seed, density):
    R = [BrainRegion(name="place", n_neurons=N_PLACE, exc_fraction=1.0, internal_density=0.0),
         BrainRegion(name="read", n_neurons=N_READ, exc_fraction=1.0, internal_density=0.0)]
    P = [RegionPathway(from_region="place", to_region="read", density=density,
                       weight_mean=W_MEAN, weight_jitter=W_JITTER, plastic=True)]
    cfg = CoreSimConfig(seed=seed, dt_ms=1.0, enable_brain_region_framework=True,
                        brain_regions=R, region_pathways=P,
                        enable_hebbian_learning=False,      # the ONLY rule is the host rule below
                        enable_stdp=False, enable_homeostasis=False,
                        enable_structural_plasticity=False, enable_ou_process=False,
                        enable_reward_modulation=False)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    place = np.asarray(b.region_manager.indices("place"))
    read = np.asarray(b.region_manager.indices("read"))
    return b, cfg, place, read


def synapse_map(b, place, read):
    """cp_connections is CSR (n x n), ROW = PRE, COL = POST. Walk the PRE (place) rows,
    keep entries whose POST lands in read. Returns flat data indices + local pre/post."""
    indptr = np.asarray(b.cp_connections.indptr)
    indices = np.asarray(b.cp_connections.indices)
    read_lut = -np.ones(int(indptr.size - 1), dtype=np.int64)
    read_lut[read] = np.arange(read.size)
    d_idx, pre_l, post_l = [], [], []
    for pl, p in enumerate(place):
        for k in range(int(indptr[p]), int(indptr[p + 1])):
            r = read_lut[indices[k]]
            if r >= 0:
                d_idx.append(k); pre_l.append(pl); post_l.append(int(r))
    return (np.asarray(d_idx, dtype=np.int64), np.asarray(pre_l, dtype=np.int64),
            np.asarray(post_l, dtype=np.int64))


def wmat(b, d_idx, pre_l, post_l):
    """(n_read x n_place) dense view of the learned weights; absent synapses stay 0."""
    W = np.zeros((N_READ, N_PLACE), dtype=np.float64)
    W[post_l, pre_l] = np.asarray(b.cp_connections.data)[d_idx]
    return W


def selectivity(W, wired_only=False):
    """R4 convention: peak / mean over ALL N_PLACE inputs (zeros included) -- the same convention the
    retracted claims used, so lr0 and trained are like-for-like. wired_only=True restricts to each
    cell's actual afferents (removes the pure-wiring component)."""
    out = []
    for j in range(W.shape[0]):
        row = W[j]
        v = row[row > 0] if wired_only else row
        if v.size == 0 or v.mean() <= 0:
            continue
        out.append(float(v.max() / v.mean()))
    return float(np.mean(out)) if out else float("nan")


def drive(b, place, t, mode="sweep", rng=None):
    """mode='sweep': the moving activity bump (the real stimulus).
    mode='shuffled': ANTI-CHEAT. The identical per-step current profile (same number of driven cells,
    same total pA, same distribution) but re-permuted across place cells EVERY step, so no two place
    cells reliably co-fire. Any selectivity gain that survives this is generic 'rich-get-richer'
    afferent competition, NOT learned place structure."""
    prof = BUMP_PA * np.exp(-0.5 * ((np.arange(N_PLACE) - (t % SWEEP_LEN)) / BUMP_SIGMA) ** 2)
    if mode == "shuffled":
        prof = prof[rng.permutation(N_PLACE)]
    b.cp_external_input_current[:] = 0.0
    b.cp_external_input_current[place] = prof


def run_arm(seed, rule, lr, n_sweeps, n_eval_sweeps, density, inp="sweep"):
    """rule in {none, hebb, oja, subtr}. rule='none' (or lr=0) is the lr=0 ARM (R1)."""
    b, cfg, place, read = build(seed, density)
    rng = np.random.default_rng(1000 + seed)
    d_idx, pre_l, post_l = synapse_map(b, place, read)
    W_init = wmat(b, d_idx, pre_l, post_l)
    n_aff = (W_init > 0).sum(axis=1)

    # per-post-cell L2 norm at init -> Oja normalises the afferent vector to THIS norm, so the total
    # drive (and therefore R3 firing) is conserved rather than collapsing.
    w0 = np.asarray(b.cp_connections.data)[d_idx].astype(np.float64)
    l2_0 = np.sqrt(np.bincount(post_l, weights=w0 ** 2, minlength=N_READ))
    l2_0 = np.maximum(l2_0, 1e-9)

    dec = float(np.exp(-1.0 / TRACE_TAU))
    tr_pre = np.zeros(N_PLACE); tr_post = np.zeros(N_READ)
    read_spk_train = 0
    for s in range(n_sweeps):
        for t in range(SWEEP_LEN):
            drive(b, place, t, inp, rng)
            b._run_one_simulation_step()
            fs = np.asarray(b.cp_firing_states)
            sp_pre = fs[place].astype(np.float64)
            sp_post = fs[read].astype(np.float64)
            read_spk_train += int(sp_post.sum())
            tr_pre = tr_pre * dec + sp_pre
            tr_post = tr_post * dec + sp_post
            if rule == "none" or lr == 0.0 or tr_post.max() <= 0.0:
                continue
            x = tr_pre[pre_l]; y = tr_post[post_l]
            w = np.asarray(b.cp_connections.data)[d_idx].astype(np.float64)
            if rule == "hebb":
                dw = lr * y * x
            elif rule == "oja":
                # dw = lr * y * (x - y*u) on u = w / l2_0  (self-normalising, |u|->1)
                u = w / l2_0[post_l]
                dw = lr * y * (x - y * u) * l2_0[post_l]
            elif rule == "subtr":
                # Miller-MacKay: subtract the per-POSTsynaptic-cell mean increment (mirrors the
                # engine's cfg.btsp_mean_subtract at sim/bridge.py:8153) -> sum_i dw_ij == 0.
                h = lr * y * x
                s_h = np.bincount(post_l, weights=h, minlength=N_READ)
                c_h = np.bincount(post_l, minlength=N_READ).astype(np.float64)
                dw = h - (s_h / np.maximum(c_h, 1.0))[post_l]
            else:
                raise ValueError(rule)
            b.cp_connections.data[d_idx] = np.clip(w + dw, W_MIN, W_MAX).astype(
                b.cp_connections.data.dtype)

    W_fin = wmat(b, d_idx, pre_l, post_l)

    # ---- EVAL: plasticity OFF for every arm; spike raster binned by sweep phase (secondary metric)
    counts = np.zeros((N_READ, SWEEP_LEN))
    read_spk_eval = 0
    for s in range(n_eval_sweeps):
        for t in range(SWEEP_LEN):
            drive(b, place, t, inp, rng)
            b._run_one_simulation_step()
            fs = np.asarray(b.cp_firing_states)[read]
            counts[:, t] += fs
            read_spk_eval += int(fs.sum())

    # Secondary metric. peak/mean on a 60-bin histogram is an artifact trap (a cell with 1 spike
    # scores 60.0), so: (a) coarse 12 bins, (b) a >=MIN_SPK floor per cell, (c) the primary form is
    # the spike mass in the best CONTIGUOUS sixth of the sweep (chance = 1/6 = 0.1667 exactly).
    PHASE_BINS, MIN_SPK, WIN = 12, 12, 10
    loc, mass, active, skipped = [], [], 0, 0
    cell_spk = [float(counts[j].sum()) for j in range(N_READ)]
    cell_loc = [float("nan")] * N_READ
    cell_mass = [float("nan")] * N_READ
    for j in range(N_READ):
        tot = counts[j].sum()
        if tot <= 0:
            continue
        active += 1
        if tot < MIN_SPK:
            skipped += 1
            continue
        h = counts[j].reshape(PHASE_BINS, SWEEP_LEN // PHASE_BINS).sum(axis=1)
        loc.append(float(h.max() / h.mean()))
        cell_loc[j] = loc[-1]
        wrap = np.concatenate([counts[j], counts[j]])
        best = max(wrap[s:s + WIN].sum() for s in range(SWEEP_LEN))
        mass.append(float(best / tot))
        cell_mass[j] = mass[-1]
    n_at_max = int((np.asarray(b.cp_connections.data)[d_idx] >= W_MAX - 1e-6).sum())
    n_at_min = int((np.asarray(b.cp_connections.data)[d_idx] <= W_MIN + 1e-6).sum())
    return dict(
        seed=seed, rule=rule, lr=lr, density=density, inp=inp,
        sel_init=selectivity(W_init), sel_final=selectivity(W_fin),
        sel_init_wired=selectivity(W_init, True), sel_final_wired=selectivity(W_fin, True),
        read_spikes_train=read_spk_train, read_spikes_eval=read_spk_eval,
        spike_localisation=float(np.mean(loc)) if loc else float("nan"),
        spike_mass_best_sixth=float(np.mean(mass)) if mass else float("nan"),
        n_cells_scored=len(loc), n_cells_below_min_spk=skipped,
        n_read_cells_firing=active, n_syn=int(d_idx.size),
        mean_afferents=float(n_aff.mean()),
        mean_abs_dw=float(np.abs(W_fin - W_init).sum() / max(1, d_idx.size)),
        mean_w_final=float(W_fin[W_init > 0].mean()),
        aff_sum_drift=float(np.mean(W_fin.sum(1) - W_init.sum(1))),
        n_at_wmax=n_at_max, n_at_wmin=n_at_min,
        cell_spk=cell_spk, cell_loc=cell_loc, cell_mass=cell_mass,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--rules", nargs="+", default=["none", "hebb", "oja", "subtr"])
    ap.add_argument("--lr", type=float, nargs="+", default=[1.0],
                    help="one value, or one per --rules (per-rule lr, set by the |dw| calibration)")
    ap.add_argument("--input", default="sweep", choices=["sweep", "shuffled"])
    ap.add_argument("--sweeps", type=int, default=60)
    ap.add_argument("--eval-sweeps", type=int, default=150)
    ap.add_argument("--density", type=float, default=0.35)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lrs = a.lr if len(a.lr) == len(a.rules) else [a.lr[0]] * len(a.rules)
    lr_of = dict(zip(a.rules, lrs))
    print(f"[cfg] input={a.input} density={a.density} w_mean={W_MEAN} w_max={W_MAX} "
          f"(R5: w_max = {W_MAX/W_MEAN:.1f}x design weight) lr={lr_of} "
          f"sweeps={a.sweeps} eval={a.eval_sweeps}")
    rows = []
    for seed in a.seeds:
        for rule in a.rules:
            t0 = time.time()
            r = run_arm(seed, rule, 0.0 if rule == "none" else lr_of[rule],
                        a.sweeps, a.eval_sweeps, a.density, a.input)
            r["secs"] = round(time.time() - t0, 1)
            rows.append(r)
            print(f"  seed={seed:3d} rule={rule:6s} lr={r['lr']:<5g} "
                  f"sel_init={r['sel_init']:6.3f} sel_final={r['sel_final']:6.3f} "
                  f"(wired {r['sel_init_wired']:5.3f}->{r['sel_final_wired']:5.3f}) "
                  f"spk_train={r['read_spikes_train']:6d} spk_eval={r['read_spikes_eval']:5d} "
                  f"cells_firing={r['n_read_cells_firing']:2d}/{N_READ} "
                  f"loc={r['spike_localisation']:6.3f} mass6={r['spike_mass_best_sixth']:5.3f} "
                  f"scored={r['n_cells_scored']:2d} |dw|={r['mean_abs_dw']:7.2f} "
                  f"w_fin={r['mean_w_final']:6.1f} aff_drift={r['aff_sum_drift']:+8.2f} "
                  f"@wmax={r['n_at_wmax']:3d} @wmin={r['n_at_wmin']:3d} [{r['secs']}s]")

    print("\n=== VERDICT TABLE (R1: the claim is the DIFFERENCE from the lr=0 arm) ===")
    base = {r["seed"]: r for r in rows if r["rule"] == "none"}
    for rule in a.rules:
        if rule == "none":
            continue
        d_sel, d_loc, d_mass, l0s, trs = [], [], [], [], []
        for r in rows:
            if r["rule"] != rule:
                continue
            b0 = base.get(r["seed"])
            if b0 is None:
                continue
            l0s.append(round(b0["sel_final"], 3)); trs.append(round(r["sel_final"], 3))
            d_sel.append(r["sel_final"] - b0["sel_final"])
            if np.isfinite(r["spike_localisation"]) and np.isfinite(b0["spike_localisation"]):
                d_loc.append(r["spike_localisation"] - b0["spike_localisation"])
            if np.isfinite(r["spike_mass_best_sixth"]) and np.isfinite(b0["spike_mass_best_sixth"]):
                d_mass.append(r["spike_mass_best_sixth"] - b0["spike_mass_best_sixth"])
            assert abs(r["sel_init"] - b0["sel_init"]) < 1e-6, "R6 VIOLATED: different wiring"
        print(f"  {rule:6s}: lr0={l0s} trained={trs}  "
              f"delta_selectivity={np.mean(d_sel):+.4f} "
              f"delta_spike_loc={(np.mean(d_loc) if d_loc else float('nan')):+.4f} "
              f"delta_mass6={(np.mean(d_mass) if d_mass else float('nan')):+.4f}")
    tot_spk = sum(r["read_spikes_train"] for r in rows)
    print(f"\n[R3] total read-region spikes across all arms = {tot_spk} "
          f"({'OK' if tot_spk > 0 else 'VOID -- population silent'})")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
