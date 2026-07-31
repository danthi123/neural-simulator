"""Does TEMPORALLY-ASYMMETRIC STDP on a travelling activity sweep produce LOCALISED
place tuning in a reader population that lr=0 does not?

THE MEASUREMENT DISCIPLINE (each rule earned by a retraction 2026-07-28/29):
  1. lr=0 IS AN ARM. Identical seed => identical wiring => identical initial weights
     (asserted by hash). The claim is trained - lr0, never the trained number alone.
  2. dW != 0 proves nothing. Only a DIFFERENCE from the lr=0 arm counts.
  3. ASSERT THE POPULATION FIRES. Read-region spikes counted. 0 spikes => VOID.
  4. Selectivity tracks WIRING. Density held FIXED across arms (0.35).
  5. stdp_w_max (600) >> weight_mean (250): the STDP rule is soft-bound, so a bound
     below the design weight makes every "LTP" event strongly negative.
  6. STDP EVENTS MUST OCCUR WITH NONZERO delta_t. The banked project prior is that
     STDP is inert for SYMMETRIC co-occurrence (656k events, 0 weight change, at
     delta_t ~ 0). We independently re-derive the delta_t distribution the bridge
     itself uses, per step, and VOID the result if events are absent / all dt==0.

Run:  .venv/bin/python research/runners/_stdp_sweep_place_tuning_probe.py
"""
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")

import hashlib
import json
import logging

import numpy as np

logging.disable(logging.INFO)

from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402
from sim import SimulationBridge  # noqa: E402

N_PLACE = 60
N_READ = 12
DENSITY = 0.35          # HELD FIXED across every arm (rule 4)
WEIGHT_MEAN = 250.0     # >=200 required for the pathway to make its target fire (rule 3)
STDP_W_MAX = 600.0      # >> WEIGHT_MEAN (rule 5)
SWEEP_LEN = 60
SIGMA = 5.0
AMP = 3000.0
N_TRAIN_SWEEPS = 40
N_TEST_SWEEPS = 30   # frozen; needs enough spikes that localisation isn't a small-n artifact
STDP_WINDOW_MS = 100.0  # max(tau_plus, tau_minus) * 5, as the bridge computes it


def build_bridge(seed, a_plus, a_minus):
    R = [
        BrainRegion(name="place", n_neurons=N_PLACE, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="read", n_neurons=N_READ, exc_fraction=1.0, internal_density=0.0),
    ]
    P = [
        RegionPathway(from_region="place", to_region="read", density=DENSITY,
                      weight_mean=WEIGHT_MEAN, weight_jitter=0.3, plastic=True),
    ]
    cfg = CoreSimConfig(
        seed=seed, dt_ms=1.0,
        enable_brain_region_framework=True, brain_regions=R, region_pathways=P,
        # STDP is the mechanism under test; every other rule OFF so nothing else can move a weight.
        enable_stdp=True,
        stdp_a_plus=a_plus, stdp_a_minus=a_minus,
        stdp_tau_plus_ms=20.0, stdp_tau_minus_ms=20.0,
        stdp_w_min=0.0, stdp_w_max=STDP_W_MAX,
        enable_hebbian_learning=False,
        enable_homeostasis=False,
        enable_structural_plasticity=False,
        enable_synaptic_scaling=False,
        enable_reward_modulation=False,
        enable_ou_process=False,
    )
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b, cfg


def pathway_synapse_index(b, place_idx, read_idx):
    """CSR: ROW = PRE, COL = POST. Return (syn_ids, pre_ids, post_ids) for place->read."""
    csr = b.cp_connections
    indptr = np.asarray(csr.indptr)
    indices = np.asarray(csr.indices)
    read_set = set(int(i) for i in read_idx)
    syn, pre, post = [], [], []
    for p in place_idx:
        p = int(p)
        for k in range(int(indptr[p]), int(indptr[p + 1])):
            c = int(indices[k])
            if c in read_set:
                syn.append(k)
                pre.append(p)
                post.append(c)
    return np.array(syn, dtype=np.int64), np.array(pre, dtype=np.int64), np.array(post, dtype=np.int64)


def weight_matrix(b, syn, pre, post, place_idx, read_idx):
    """(n_read x n_place) dense matrix of the current learned weights; 0 = unwired."""
    p_of = {int(v): i for i, v in enumerate(place_idx)}
    r_of = {int(v): i for i, v in enumerate(read_idx)}
    W = np.zeros((N_READ, N_PLACE), dtype=np.float64)
    data = np.asarray(b.cp_connections.data)
    for k, a, c in zip(syn, pre, post):
        W[r_of[int(c)], p_of[int(a)]] = float(data[int(k)])
    return W


def selectivity(W):
    """PRIMARY METRIC: mean over reader cells of (peak weight / mean weight) across
    its WIRED place inputs. Computed over wired inputs only, so it cannot be inflated
    by counting structural zeros."""
    vals = []
    for r in range(W.shape[0]):
        w = W[r]
        w = w[w != 0.0]
        if w.size == 0 or w.mean() <= 0:
            continue
        vals.append(float(w.max() / w.mean()))
    return (float(np.mean(vals)) if vals else float("nan")), vals


def drive_step(b, place_idx, t_phase, read_idx=None, read_tonic=0.0):
    b.cp_external_input_current[:] = 0.0
    bump = AMP * np.exp(-0.5 * ((np.arange(N_PLACE) - t_phase) / SIGMA) ** 2)
    b.cp_external_input_current[place_idx] = bump
    if read_tonic and read_idx is not None:
        # Raises the POSTsynaptic firing rate WITHOUT touching the wiring, so the lr=0
        # control at the same tonic level stays like-for-like (rule 4: fixed density).
        b.cp_external_input_current[read_idx] = read_tonic
    b._run_one_simulation_step()
    # ⚠️ THE CLOCK IS THE CALLER'S JOB. `_run_one_simulation_step` does NOT advance
    # runtime_state.current_time_ms -- only `step_simulation` does (bridge.py:4141).
    # STDP stamps cp_last_spike_time with current_time_ms, so a probe that omits this
    # stamps EVERY spike with t=0 => delta_t == 0 for every synapse => the STDP kernel
    # returns zero change for every event (fused_stdp_weight_update: >0 LTP, <0 LTD,
    # ==0 nothing). That is an INSTRUMENT artifact, not a substrate property.
    b.runtime_state.current_time_ms += b.core_config.dt_ms
    b.runtime_state.current_time_step += 1


def stdp_event_audit(b, pre, post):
    """Independently re-derive the delta_t set the bridge's STDP block just used
    (bridge.py:7784-7811): candidate = pre_fired | post_fired, valid = both spiked,
    within = |dt| < window. Returns (n_events, n_nonzero_dt, sum|dt|, n_ltp, n_ltd)."""
    fired = np.asarray(b.cp_firing_states).astype(bool)
    lst = np.asarray(b.cp_last_spike_time)
    cand = fired[pre] | fired[post]
    dt = lst[post] - lst[pre]
    valid = (lst[pre] > -500.0) & (lst[post] > -500.0)
    act = cand & valid & (np.abs(dt) < STDP_WINDOW_MS)
    d = dt[act]
    return int(act.sum()), int((d != 0.0).sum()), float(np.abs(d).sum()), int((d > 0).sum()), int((d < 0).sum())


def run_arm(seed, a_plus, a_minus, label, read_tonic=0.0):
    b, cfg = build_bridge(seed, a_plus, a_minus)
    place_idx = np.asarray(b.region_manager.indices("place"))
    read_idx = np.asarray(b.region_manager.indices("read"))
    syn, pre, post = pathway_synapse_index(b, place_idx, read_idx)
    nnz0 = int(b.cp_connections.nnz)
    assert syn.size == nnz0, f"non-pathway synapses present: {syn.size} of {nnz0}"

    W0 = weight_matrix(b, syn, pre, post, place_idx, read_idx)
    w0_hash = hashlib.sha1(np.ascontiguousarray(W0).tobytes()).hexdigest()[:12]

    ev = nz = ltp = ltd = 0
    absdt = 0.0
    read_spikes_train = 0
    spikes_per_sweep = []
    for s in range(N_TRAIN_SWEEPS):
        sweep_spk = 0
        for t in range(SWEEP_LEN):
            drive_step(b, place_idx, t, read_idx, read_tonic)
            sweep_spk += int(np.asarray(b.cp_firing_states)[read_idx].sum())
            e, n, a, p, m = stdp_event_audit(b, pre, post)
            ev += e; nz += n; absdt += a; ltp += p; ltd += m
        read_spikes_train += sweep_spk
        spikes_per_sweep.append(sweep_spk)

    W1 = weight_matrix(b, syn, pre, post, place_idx, read_idx)
    assert int(b.cp_connections.nnz) == nnz0, "synapse count changed (structural plasticity leaked in)"

    # ---- FROZEN read-out (the recall read must be READ-ONLY; tonight's retraction) ----
    cfg.stdp_a_plus = 0.0
    cfg.stdp_a_minus = 0.0
    W_before_read = W1.copy()
    for t in range(SWEEP_LEN):                       # frozen warm-up sweep
        drive_step(b, place_idx, t, read_idx, read_tonic)
    raster = np.zeros((N_READ, SWEEP_LEN), dtype=np.int64)
    for s in range(N_TEST_SWEEPS):
        for t in range(SWEEP_LEN):
            drive_step(b, place_idx, t, read_idx, read_tonic)
            f = np.asarray(b.cp_firing_states)[read_idx].astype(bool)
            raster[f, t] += 1
    W_after_read = weight_matrix(b, syn, pre, post, place_idx, read_idx)
    read_drift = float(np.abs(W_after_read - W_before_read).max())

    sel0, _ = selectivity(W0)
    sel1, per_cell = selectivity(W1)

    # SECONDARY: is each reader's SPIKING localised to one sweep phase?
    # ⚠️ CONFOUND: a cell with 1 spike scores 1.000 mechanically, so a LOWER-firing arm
    # wins this metric for free. Controlled by a SPIKE-COUNT-MATCHED shuffle: redraw each
    # cell's own n spikes uniformly over the 60 phases and recompute. The claim is
    # loc - loc_shuffled, which is exactly 0 for any purely-count-driven effect.
    win = 9  # 15% of the 60-step sweep
    rng = np.random.default_rng(12345)

    def best_win(counts):
        return max(counts[i:i + win].sum() for i in range(SWEEP_LEN - win + 1))

    loc, loc_null, prefs, spk_per_cell, n_active = [], [], [], [], 0
    for r in range(N_READ):
        tot = int(raster[r].sum())
        if tot == 0:
            continue
        n_active += 1
        spk_per_cell.append(tot)
        loc.append(float(best_win(raster[r]) / tot))
        nulls = []
        for _ in range(200):
            sh = np.bincount(rng.integers(0, SWEEP_LEN, size=tot), minlength=SWEEP_LEN)
            nulls.append(best_win(sh) / tot)
        loc_null.append(float(np.mean(nulls)))
        prefs.append(float((raster[r] * np.arange(SWEEP_LEN)).sum() / tot))

    # Does STDP exploit the sweep's TEMPORAL ORDER at all? For each reader, take its
    # initial preferred place input (argmax W0) and split its wired inputs into those
    # EARLIER in the sweep and those LATER. Temporally-asymmetric STDP predicts
    # LTP for earlier inputs (dt = t_post - t_pre > 0) and LTD for later ones, i.e. a
    # strongly POSITIVE early-minus-late asymmetry (the Mehta backward-shift signature).
    dW = W1 - W0
    early, late = [], []
    for r in range(N_READ):
        wired = np.nonzero(W0[r])[0]
        if wired.size == 0:
            continue
        pref = int(np.argmax(W0[r]))
        d = wired - pref
        e = wired[(d <= -3) & (d >= -25)]
        l = wired[(d >= 3) & (d <= 25)]
        if e.size:
            early.append(float(dW[r, e].mean()))
        if l.size:
            late.append(float(dW[r, l].mean()))
    dW_early = float(np.mean(early)) if early else float("nan")
    dW_late = float(np.mean(late)) if late else float("nan")

    return dict(
        label=label, seed=seed, a_plus=a_plus, a_minus=a_minus, read_tonic=read_tonic,
        nnz=nnz0, w0_hash=w0_hash,
        sel_initial=sel0, sel_final=sel1, sel_per_cell=[round(v, 3) for v in per_cell],
        dW_absmean=float(np.abs(W1 - W0)[W0 != 0].mean()),
        dW_absmax=float(np.abs(W1 - W0).max()),
        w_final_mean=float(W1[W0 != 0].mean()), w_final_min=float(W1[W0 != 0].min()),
        w_final_max=float(W1[W0 != 0].max()),
        frac_at_wmax=float((W1[W0 != 0] > 0.999 * STDP_W_MAX).mean()),
        frac_at_wmin=float((W1[W0 != 0] < 1e-6).mean()),
        read_spikes_train=read_spikes_train,
        spikes_first5_sweeps=spikes_per_sweep[:5], spikes_last5_sweeps=spikes_per_sweep[-5:],
        read_spikes_test=int(raster.sum()), n_active_read_cells=n_active,
        spike_localisation=(float(np.mean(loc)) if loc else float("nan")),
        spike_localisation_shuffled=(float(np.mean(loc_null)) if loc_null else float("nan")),
        spike_localisation_vs_shuffle=(float(np.mean(loc) - np.mean(loc_null)) if loc else float("nan")),
        spikes_per_active_cell=spk_per_cell,
        dW_early_inputs=dW_early, dW_late_inputs=dW_late,
        dW_early_minus_late=dW_early - dW_late,
        pref_phases=[round(v, 1) for v in prefs],
        stdp_events=ev, stdp_events_nonzero_dt=nz,
        stdp_mean_abs_dt=(absdt / ev if ev else 0.0),
        stdp_ltp=ltp, stdp_ltd=ltd,
        read_only_drift=read_drift,
    )


def main():
    seeds = [42, 43, 44]
    arms = [
        # (label, a_plus, a_minus, read_tonic_pA)
        # tonic=0   : the sparse-post regime. LTD events swamp LTP ~5:1.
        # tonic=300 : LTP/LTD near BALANCED (all 12 readers active, ~60 Hz) -- the regime
        #             where an asymmetric rule has its best shot. Each tonic level carries
        #             its OWN lr=0 control, so the comparison is always like-for-like.
        ("lr0", 0.0, 0.0, 0.0),
        ("stdp", 0.012, 0.010, 0.0),          # Song et al. 2000 canonical = the config defaults
        ("stdp_lo", 0.0012, 0.0010, 0.0),     # 10x lower, separates learning from saturation
        ("lr0_t300", 0.0, 0.0, 300.0),
        ("stdp_t300", 0.012, 0.010, 300.0),
        ("lr0_t450", 0.0, 0.0, 450.0),
        ("stdp_t450", 0.012, 0.010, 450.0),   # LTP now slightly EXCEEDS LTD
    ]
    pairs = [("stdp", "lr0"), ("stdp_lo", "lr0"),
             ("stdp_t300", "lr0_t300"), ("stdp_t450", "lr0_t450")]
    out = []
    for seed in seeds:
        for label, ap, am, tonic in arms:
            r = run_arm(seed, ap, am, label, read_tonic=tonic)
            out.append(r)
            print(f"[{label:10s} seed={seed}] sel_init={r['sel_initial']:.3f} "
                  f"sel_final={r['sel_final']:.3f} | read_spk train={r['read_spikes_train']} "
                  f"test={r['read_spikes_test']} act={r['n_active_read_cells']}/{N_READ} | "
                  f"loc={r['spike_localisation']:.3f} shuf={r['spike_localisation_shuffled']:.3f} "
                  f"(d={r['spike_localisation_vs_shuffle']:+.3f}) | dW|mean|={r['dW_absmean']:.3f} "
                  f"w=[{r['w_final_min']:.1f},{r['w_final_max']:.1f}] "
                  f"@max={r['frac_at_wmax']:.2f} @min={r['frac_at_wmin']:.2f} | "
                  f"ev={r['stdp_events']} nz={r['stdp_events_nonzero_dt']} "
                  f"|dt|={r['stdp_mean_abs_dt']:.2f} ltp/ltd={r['stdp_ltp']}/{r['stdp_ltd']} | "
                  f"drift={r['read_only_drift']:.2e} hash={r['w0_hash']}", flush=True)

    print("\n=== ANTI-CHEAT: identical wiring per seed across arms ===")
    for seed in seeds:
        hs = sorted(set(r["w0_hash"] for r in out if r["seed"] == seed))
        print(f"  seed {seed}: {hs}  identical_across_all_7_arms={len(hs) == 1}")

    def col(label, key):
        return [r[key] for r in out if r["label"] == label]

    print("\n=== PRIMARY: selectivity (peak/mean weight per reader cell) ===")
    for label, _, _, tn in arms:
        v = col(label, "sel_final")
        print(f"  {label:10s} (tonic {tn:5.0f}) per-seed {[round(x, 3) for x in v]}  mean={np.mean(v):.4f}")
    print("  --- THE CLAIM IS THE DIFFERENCE (trained - its own lr=0 arm) ---")
    for tr, ct in pairs:
        t, c = np.mean(col(tr, "sel_final")), np.mean(col(ct, "sel_final"))
        per = [round(a - b, 3) for a, b in zip(col(tr, "sel_final"), col(ct, "sel_final"))]
        print(f"  DELTA {tr:10s} - {ct:10s} = {t - c:+.4f}  (ratio {t / c:.3f}x)  per-seed {per}")

    print("\n=== SECONDARY: spiking localisation, spike-count-matched (loc - shuffled) ===")
    for label, _, _, tn in arms:
        print(f"  {label:10s} raw={np.nanmean(col(label,'spike_localisation')):.4f} "
              f"shuffled={np.nanmean(col(label,'spike_localisation_shuffled')):.4f} "
              f"EXCESS={np.nanmean(col(label,'spike_localisation_vs_shuffle')):+.4f}  "
              f"per-seed excess {[round(x, 3) for x in col(label,'spike_localisation_vs_shuffle')]}")
    for tr, ct in pairs:
        t = np.nanmean(col(tr, "spike_localisation_vs_shuffle"))
        c = np.nanmean(col(ct, "spike_localisation_vs_shuffle"))
        print(f"  DELTA {tr:10s} - {ct:10s} = {t - c:+.4f}")

    print("\n=== MECHANISM: does STDP exploit the sweep's temporal order? "
          "(asymmetric STDP predicts early >> late) ===")
    for label, _, _, tn in arms:
        print(f"  {label:10s} dW(earlier inputs)={np.nanmean(col(label,'dW_early_inputs')):+8.3f}  "
              f"dW(later)={np.nanmean(col(label,'dW_late_inputs')):+8.3f}  "
              f"early-late={np.nanmean(col(label,'dW_early_minus_late')):+.3f}  "
              f"per-seed {[round(x, 3) for x in col(label,'dW_early_minus_late')]}")

    print("\n=== ENGAGEMENT LEDGER (rule 3 + rule 6): the mechanism must actually run ===")
    for label, _, _, tn in arms:
        print(f"  {label:10s} read_spk(train)={int(np.mean(col(label,'read_spikes_train'))):5d} "
              f"active_cells={np.mean(col(label,'n_active_read_cells')):.1f}/{N_READ}  "
              f"stdp_events={int(np.mean(col(label,'stdp_events'))):6d} "
              f"nonzero_dt={int(np.mean(col(label,'stdp_events_nonzero_dt'))):6d} "
              f"mean|dt|={np.mean(col(label,'stdp_mean_abs_dt')):5.2f}ms "
              f"LTP:LTD={np.mean(col(label,'stdp_ltp')):.0f}:{np.mean(col(label,'stdp_ltd')):.0f} "
              f"read_only_drift={max(col(label,'read_only_drift')):.1e}")

    p = "/home/dant123/Projects/sim/research/findings/raw/_stdp_sweep_place_tuning_probe.json"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nraw -> {p}")


if __name__ == "__main__":
    main()
