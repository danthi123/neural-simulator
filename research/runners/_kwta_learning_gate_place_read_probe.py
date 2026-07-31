"""k-WTA LEARNING GATE probe: can a reader population LEARN place tuning from a moving sweep?

MECHANISM UNDER TEST (option (b) of the brief): a per-POSTsynaptic-cell k-WTA *learning* gate.
On every timestep only the top-k most-driven reader cells are allowed to update their weights;
every other reader cell's incoming synapses are frozen by zeroing their entry in
`bridge.cp_plasticity_rate_gain` (a per-SYNAPSE array). Biologically this stands in for fast
feedforward/lateral inhibition that lets only the best-driven cell's synapses be eligible for
potentiation.

WHY THIS SHAPE OF EXPERIMENT (each rule earned by a retraction):
  R1  lr=0 IS AN ARM. Every trained arm has a byte-matched lr=0 twin (identical wiring, identical
      step count, identical gating procedure). The claim is the DIFFERENCE, never the level.
  R2  dW != 0 proves nothing (the Hebbian decay term alone moves weights) -> only lr0-vs-trained
      deltas are reported as effects.
  R3  ASSERT THE POPULATION FIRES. Read-region spike counts are counted and a zero count is
      reported as VOID, not as a negative.
  R4  SELECTIVITY TRACKS WIRING. All arms share ONE wiring (asserted by hashing the initial
      weight vector across arms), so density can never explain a difference.
  R5  hebbian_max_weight (600) is set above the design weight (250) so potentiation is positive.
  R6  Report what was measured, including a clean negative.

ARMS (per seed):
  init        weights at t=0 (no steps)                      -- the pure-structure reference
  lr0_plain   lr=0.0,  no gate  (gain == 1 everywhere)
  hebb_plain  lr=0.02, no gate                               -- plain Hebbian baseline
  lr0_kwta    lr=0.0,  k-WTA learning gate active            -- THE control for the mechanism
  kwta        lr=0.02, k-WTA learning gate active            -- THE mechanism

PRIMARY METRIC   mean over reader cells of (peak weight / mean weight) across the 60 place inputs,
                 mean taken over ALL 60 place slots incl. structural zeros (this is the definition
                 that reproduces the quoted 1.7x @ d=1.0 / 4.5x @ d=0.35 / 11x @ d=0.15 structural
                 baseline). `sel_nz` (mean over existing synapses only) is reported alongside.
SECONDARY METRIC after training, plasticity is frozen and clean sweeps are replayed; per reader
                 cell we measure whether its SPIKING is localised to one sweep phase
                 (peak-phase count / mean-phase count) and whether different cells prefer
                 DIFFERENT phases (differentiation, not just reduced firing).

Usage:
    OMP_NUM_THREADS=1 .venv/bin/python research/runners/_kwta_learning_gate_place_read_probe.py
"""

import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")

import argparse
import hashlib
import json
import logging
import time

import numpy as np

logging.disable(logging.INFO)

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402
from sim import SimulationBridge  # noqa: E402

N_PLACE = 60
N_READ = 12
DENSITY = 0.35
WEIGHT_MEAN = 250.0
WEIGHT_JITTER = 0.3
DRIVE_PA = 3000.0
BUMP_SIGMA = 5.0
HEBB_MAX_W = 600.0
HEBB_DECAY = 1e-5
GATE_NAME = "pr"


# ---------------------------------------------------------------- bridge build


def build_bridge(seed, lr):
    """Build the place->read bridge. `plasticity_gate` is tagged in EVERY arm so the gated
    decay/clip code path is identical across arms (only `lr` and the per-step gain differ)."""
    R = [
        BrainRegion(name="place", n_neurons=N_PLACE, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="read", n_neurons=N_READ, exc_fraction=1.0, internal_density=0.0),
    ]
    P = [
        RegionPathway(
            from_region="place", to_region="read", density=DENSITY,
            weight_mean=WEIGHT_MEAN, weight_jitter=WEIGHT_JITTER, plastic=True,
            plasticity_gate=GATE_NAME,
        )
    ]
    cfg = CoreSimConfig(
        seed=seed, dt_ms=1.0, enable_brain_region_framework=True,
        brain_regions=R, region_pathways=P,
        enable_hebbian_learning=True, hebbian_learning_rate=lr,
        hebbian_max_weight=HEBB_MAX_W, hebbian_weight_decay=HEBB_DECAY,
        enable_stdp=False, enable_homeostasis=False,
        enable_structural_plasticity=False, enable_ou_process=False,
    )
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def csr_place_read_map(b, place_idx, read_idx):
    """Walk the CSR (ROW = PRE, COL = POST) and return the synapse-index map for place->read.

    Returns (syn_ids, pre_pos, post_pos): parallel int arrays; pre_pos in [0,N_PLACE),
    post_pos in [0,N_READ).
    """
    csr = b.cp_connections
    indptr = np.asarray(csr.indptr)
    indices = np.asarray(csr.indices)
    place_pos = {int(g): i for i, g in enumerate(place_idx)}
    read_pos = {int(g): j for j, g in enumerate(read_idx)}
    syn_ids, pre_pos, post_pos = [], [], []
    for g_pre in place_idx:
        g_pre = int(g_pre)
        for s in range(indptr[g_pre], indptr[g_pre + 1]):
            g_post = int(indices[s])
            if g_post in read_pos:
                syn_ids.append(s)
                pre_pos.append(place_pos[g_pre])
                post_pos.append(read_pos[g_post])
    return (np.asarray(syn_ids, dtype=np.int64),
            np.asarray(pre_pos, dtype=np.int64),
            np.asarray(post_pos, dtype=np.int64))


def read_W(b, syn_ids, pre_pos, post_pos):
    """Dense (N_READ, N_PLACE) view of the learned pathway weights (structural zeros stay 0)."""
    W = np.zeros((N_READ, N_PLACE), dtype=np.float64)
    W[post_pos, pre_pos] = np.asarray(b.cp_connections.data)[syn_ids]
    return W


# ---------------------------------------------------------------- metrics


def selectivity_percell(W, mask):
    """Per-cell peak/mean over ALL 60 place slots; NaN for cells with no synapses."""
    out = np.full(W.shape[0], np.nan)
    for j in range(W.shape[0]):
        if mask[j].sum() == 0 or W[j].max() <= 0 or W[j].mean() <= 0:
            continue
        out[j] = W[j].max() / W[j].mean()
    return out


def selectivity(W, mask):
    """Per-cell peak/mean. `sel_all` divides by the mean over ALL 60 place slots (the definition
    whose structural baseline is ~4.5x at density 0.35); `sel_nz` divides by the mean over the
    cell's EXISTING synapses only. Cells with no synapses / all-zero weights are skipped."""
    sel_all, sel_nz = [], []
    for j in range(W.shape[0]):
        row = W[j]
        nz = mask[j]
        if nz.sum() == 0 or row.max() <= 0:
            continue
        m_all = row.mean()
        m_nz = row[nz].mean()
        if m_all > 0:
            sel_all.append(row.max() / m_all)
        if m_nz > 0:
            sel_nz.append(row.max() / m_nz)
    return (float(np.mean(sel_all)) if sel_all else float("nan"),
            float(np.mean(sel_nz)) if sel_nz else float("nan"))


def bump(phase):
    return DRIVE_PA * np.exp(-0.5 * ((np.arange(N_PLACE) - phase) / BUMP_SIGMA) ** 2)


# ---------------------------------------------------------------- run one arm


def run_arm(seed, lr, gate_on, k, n_sweeps, probe_sweeps=20, shuffle_phase=False):
    b = build_bridge(seed, lr)
    place_idx = np.asarray(b.region_manager.indices("place"))
    read_idx = np.asarray(b.region_manager.indices("read"))
    syn_ids, pre_pos, post_pos = csr_place_read_map(b, place_idx, read_idx)
    mask = np.zeros((N_READ, N_PLACE), dtype=bool)
    mask[post_pos, pre_pos] = True

    W0 = read_W(b, syn_ids, pre_pos, post_pos)
    wiring_hash = hashlib.sha1(
        np.ascontiguousarray(W0).round(6).tobytes() + mask.tobytes()).hexdigest()[:12]

    if b.cp_plasticity_rate_gain is None:
        raise RuntimeError("cp_plasticity_rate_gain not allocated -- plasticity_gate tag missing")

    n_steps = n_sweeps * N_PLACE
    prev_place = np.zeros(N_PLACE, dtype=bool)
    read_spikes_total = 0
    place_spikes_total = 0
    win_hist = np.zeros((N_READ, N_PLACE), dtype=np.int64)   # who won at which phase
    n_gated_steps = 0

    # ANTI-CHEAT control: visit the SAME 60 bumps in a random order each sweep instead of as a
    # moving sweep. Identical stimulus set, identical total drive, NO sweep. If the selectivity gain
    # survives this, the gain is not "learned place tuning from a moving sweep".
    shuf_rng = np.random.default_rng(1000 + seed)
    phase_order = None
    if shuffle_phase:
        phase_order = np.concatenate([shuf_rng.permutation(N_PLACE) for _ in range(n_sweeps)])

    for t in range(n_steps):
        phase = int(phase_order[t]) if shuffle_phase else (t % N_PLACE)
        if gate_on:
            W = read_W(b, syn_ids, pre_pos, post_pos)
            drive = W @ prev_place.astype(np.float64)
            if drive.max() > 0:
                winners = np.argsort(-drive, kind="stable")[:k]
                win_mask = np.zeros(N_READ, dtype=np.float32)
                win_mask[winners] = 1.0
                b.cp_plasticity_rate_gain[syn_ids] = win_mask[post_pos]
                win_hist[winners, phase] += 1
                n_gated_steps += 1
            else:
                b.cp_plasticity_rate_gain[syn_ids] = 1.0
        b.cp_external_input_current[:] = 0.0
        b.cp_external_input_current[place_idx] = bump(phase)
        b._run_one_simulation_step()
        fired = np.asarray(b.cp_firing_states)
        prev_place = fired[place_idx].copy()
        place_spikes_total += int(prev_place.sum())
        read_spikes_total += int(fired[read_idx].sum())

    W_end = read_W(b, syn_ids, pre_pos, post_pos)
    pot_events = int(b._mock_total_plasticity_events)

    # ---- secondary: spiking localisation, plasticity FROZEN, clean sweeps
    b.set_global_plasticity_gain(0.0)

    def replay():
        spk = np.zeros((N_READ, N_PLACE), dtype=np.int64)
        for _ in range(probe_sweeps):
            for phase in range(N_PLACE):
                b.cp_external_input_current[:] = 0.0
                b.cp_external_input_current[place_idx] = bump(phase)
                b._run_one_simulation_step()
                spk[:, phase] += np.asarray(b.cp_firing_states)[read_idx].astype(np.int64)
        return spk

    def loc_metrics(spk):
        act = [j for j in range(N_READ) if spk[j].sum() > 0]
        lo, f5, pr = [], [], {}
        for j in act:
            r = spk[j].astype(float)
            lo.append(r.max() / r.mean())
            p = int(np.argmax(r))
            pr[j] = p
            near = np.abs(np.arange(N_PLACE) - p) <= 5
            f5.append(r[near].sum() / r.sum())
        return act, lo, f5, pr

    spk = replay()
    W_after_probe = read_W(b, syn_ids, pre_pos, post_pos)
    frozen_ok = bool(np.allclose(W_end, W_after_probe))
    active_cells, loc, frac5, pref = loc_metrics(spk)

    # GAIN-MATCHED replay: arms end at different overall weight SCALES (learning raises the mean),
    # and a higher mean drive alone changes firing rate -> changes measured localisation. Rescale the
    # whole pathway so every arm has the SAME mean weight, then replay: differences are then tuning
    # SHAPE, not output gain. (Plasticity already frozen, so this is measurement-only.)
    scale = WEIGHT_MEAN / float(W_end[mask].mean())
    b.cp_connections.data[syn_ids] = np.asarray(b.cp_connections.data)[syn_ids] * scale
    spk_gm = replay()
    act_gm, loc_gm, frac5_gm, pref_gm = loc_metrics(spk_gm)

    sel_all_0, sel_nz_0 = selectivity(W0, mask)
    sel_all_1, sel_nz_1 = selectivity(W_end, mask)

    # --- differentiation of the LEARNED weight profiles
    live = [j for j in range(N_READ) if mask[j].sum() > 0]
    def mean_pair_cos(W):
        rows = [W[j] / (np.linalg.norm(W[j]) + 1e-12) for j in live]
        cs = [float(rows[a] @ rows[bb]) for a in range(len(rows)) for bb in range(a + 1, len(rows))]
        return float(np.mean(cs)) if cs else float("nan")
    argmax_init = {j: int(np.argmax(W0[j])) for j in live}
    argmax_end = {j: int(np.argmax(W_end[j])) for j in live}
    n_argmax_moved = sum(1 for j in live if argmax_init[j] != argmax_end[j])
    # does the weight peak sit where the cell SPIKES?
    align = [abs(argmax_end[j] - pref[j]) for j in active_cells if j in argmax_end]

    # --- IS THE POTENTIATION A CONTIGUOUS PLACE FIELD?  (the actual claim under test)
    # Take the POSITIVE part of dW per cell and compute the weighted SD of place index. A learned
    # place field is a contiguous band -> SD ~ bump width. Potentiation scattered over the cell's
    # whole input set -> SD ~ the uniform-null 60/sqrt(12) = 17.3. `centres` are the per-cell field
    # centres: if learning is differentiating the cells, these should SPREAD over the sweep.
    dW_pos = np.clip(W_end - W0, 0.0, None)
    UNIFORM_NULL_SD = N_PLACE / np.sqrt(12.0)
    idx = np.arange(N_PLACE, dtype=float)
    field_sds, centres, cover = [], [], []
    for j in range(N_READ):
        wj = dW_pos[j]
        s = wj.sum()
        if s <= 0:
            continue
        p = wj / s
        c = float((p * idx).sum())
        sd = float(np.sqrt((p * (idx - c) ** 2).sum()))
        field_sds.append(sd)
        centres.append(c)
        cover.append(float((wj > 0).sum()))   # how many place inputs got ANY potentiation
    win_centre = {}
    for j in range(N_READ):
        h = win_hist[j].astype(float)
        if h.sum() > 0:
            win_centre[j] = float((h / h.sum() * idx).sum())

    return {
        "seed": seed, "lr": lr, "gate_on": gate_on, "k": k, "n_sweeps": n_sweeps,
        "shuffle_phase": shuffle_phase,
        "wiring_hash": wiring_hash,
        "n_syn": int(syn_ids.size),
        "density_actual": float(mask.sum() / (N_READ * N_PLACE)),
        "sel_all_init": sel_all_0, "sel_nz_init": sel_nz_0,
        "sel_all_end": sel_all_1, "sel_nz_end": sel_nz_1,
        "read_spikes_total": read_spikes_total,
        "place_spikes_total": place_spikes_total,
        "potentiation_events": pot_events,
        "pot_events_per_syn": pot_events / max(1, int(syn_ids.size)),
        "w_mean_init": float(W0[mask].mean()), "w_mean_end": float(W_end[mask].mean()),
        "w_max_end": float(W_end[mask].max()), "w_min_end": float(W_end[mask].min()),
        "dW_absmean": float(np.abs(W_end - W0)[mask].mean()),
        # DIAGNOSTIC: is potentiation EQUALIZING? The soft bound makes delta ∝ (max_w - w), so weak
        # synapses grow fastest. r << 0 means the rule flattens the weight vector no matter WHO learns.
        "corr_dW_vs_winit": (float(np.corrcoef(W0[mask], (W_end - W0)[mask])[0, 1])
                             if np.std((W_end - W0)[mask]) > 0 else float("nan")),
        "frac_at_bound": float((W_end[mask] > HEBB_MAX_W * 0.98).mean()),
        "n_gated_steps": n_gated_steps,
        "n_distinct_winners": int((win_hist.sum(axis=1) > 0).sum()),
        "win_share_top1": (float(win_hist.sum(axis=1).max() / max(1, win_hist.sum()))
                           if win_hist.sum() else 0.0),
        "probe_read_spikes": int(spk.sum()),
        "n_active_cells": len(active_cells),
        "spike_localisation": float(np.mean(loc)) if loc else float("nan"),
        "spike_frac_within_pm5": float(np.mean(frac5)) if frac5 else float("nan"),
        "n_distinct_pref_phase": len(set(pref.values())),
        "pref_phases": sorted(pref.values()),
        # gain-matched (output-scale-controlled) versions of the spiking metrics
        "gm_scale": float(scale),
        "gm_probe_read_spikes": int(spk_gm.sum()),
        "gm_n_active_cells": len(act_gm),
        "gm_spike_localisation": float(np.mean(loc_gm)) if loc_gm else float("nan"),
        "gm_spike_frac_within_pm5": float(np.mean(frac5_gm)) if frac5_gm else float("nan"),
        "gm_n_distinct_pref_phase": len(set(pref_gm.values())),
        "pair_cos_init": mean_pair_cos(W0), "pair_cos_end": mean_pair_cos(W_end),
        "n_argmax_moved": n_argmax_moved, "n_live_cells": len(live),
        "w_peak_vs_spike_pref_absdiff": float(np.mean(align)) if align else float("nan"),
        # per-cell selectivity + which cells actually RECEIVED potentiation ("learners"), so the
        # population mean can be decomposed into (did learners sharpen) vs (coverage dilution).
        "sel_percell_init": [None if np.isnan(x) else float(x)
                             for x in selectivity_percell(W0, mask)],
        "sel_percell_end": [None if np.isnan(x) else float(x)
                            for x in selectivity_percell(W_end, mask)],
        "learner_mask": [bool(dW_pos[j].sum() > 0) for j in range(N_READ)],
        "dW_field_sd": float(np.mean(field_sds)) if field_sds else float("nan"),
        "dW_field_sd_uniform_null": float(UNIFORM_NULL_SD),
        "dW_field_n_cells": len(field_sds),
        "dW_field_centre_spread": float(np.std(centres)) if len(centres) > 1 else float("nan"),
        "dW_field_centres": [round(c, 1) for c in centres],
        "dW_field_cover_inputs": float(np.mean(cover)) if cover else float("nan"),
        "win_centres": {int(j): round(v, 1) for j, v in win_centre.items()},
        "frozen_during_probe": frozen_ok,
    }


LR = 0.02

ARMS = [
    ("lr0_plain", 0.0, False),
    ("hebb_plain", None, False),
    ("lr0_kwta", 0.0, True),
    ("kwta", None, True),
]


def main():
    global WEIGHT_MEAN, HEBB_MAX_W, HEBB_DECAY, BUMP_SIGMA, DENSITY
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--sweeps", type=int, default=40)
    ap.add_argument("--weight-mean", type=float, default=WEIGHT_MEAN)
    ap.add_argument("--max-w", type=float, default=None, help="hebbian_max_weight (default 2.4x weight_mean)")
    ap.add_argument("--decay", type=float, default=HEBB_DECAY)
    ap.add_argument("--sigma", type=float, default=BUMP_SIGMA)
    ap.add_argument("--density", type=float, default=DENSITY)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--probe-sweeps", type=int, default=60)
    ap.add_argument("--shuffle-phase", action="store_true",
                    help="ANTI-CHEAT: same 60 bumps, random order (no moving sweep)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    WEIGHT_MEAN = a.weight_mean
    HEBB_MAX_W = a.max_w if a.max_w is not None else 2.4 * a.weight_mean
    HEBB_DECAY = a.decay
    BUMP_SIGMA = a.sigma
    DENSITY = a.density
    print(f"CONFIG: n_place={N_PLACE} n_read={N_READ} density={DENSITY} w_mean={WEIGHT_MEAN} "
          f"jitter={WEIGHT_JITTER} max_w={HEBB_MAX_W} decay={HEBB_DECAY} sigma={BUMP_SIGMA} "
          f"k={a.k} sweeps={a.sweeps} lr={a.lr} probe_sweeps={a.probe_sweeps}")

    res = {}
    t0 = time.time()
    for seed in a.seeds:
        for name, lr, gate in ARMS:
            lr = a.lr if lr is None else lr
            r = run_arm(seed, lr, gate, a.k, a.sweeps, probe_sweeps=a.probe_sweeps,
                        shuffle_phase=a.shuffle_phase)
            res.setdefault(name, []).append(r)
            print(f"[{time.time()-t0:7.1f}s] seed {seed} {name:11s} "
                  f"sel_all {r['sel_all_init']:.3f}->{r['sel_all_end']:.3f}  "
                  f"sel_nz {r['sel_nz_init']:.3f}->{r['sel_nz_end']:.3f}  "
                  f"readspk {r['read_spikes_total']:6d}  "
                  f"w {r['w_mean_init']:.1f}->{r['w_mean_end']:.1f} (max {r['w_max_end']:.1f}) "
                  f"dW {r['dW_absmean']:.2f}  bound {r['frac_at_bound']:.2f}  "
                  f"win:{r['n_distinct_winners']}/{r['win_share_top1']:.2f}  "
                  f"loc {r['spike_localisation']:.2f} pref{r['n_distinct_pref_phase']} "
                  f"probespk {r['probe_read_spikes']}  hash {r['wiring_hash']}",
                  flush=True)

    print("\n=== WIRING IDENTITY (rule 4: same density AND same wiring across arms) ===")
    for seed_i, seed in enumerate(a.seeds):
        hashes = {n: res[n][seed_i]["wiring_hash"] for n, _, _ in ARMS}
        print(f"  seed {seed}: {hashes}  identical={len(set(hashes.values()))==1}")

    print("\n=== PRIMARY: selectivity (peak/mean over all 60 place slots) ===")
    print(f"{'arm':12s} " + " ".join(f"s{s}" .rjust(8) for s in a.seeds) + "    mean")
    for name, _, _ in ARMS:
        v = [r["sel_all_end"] for r in res[name]]
        print(f"{name:12s} " + " ".join(f"{x:8.3f}" for x in v) + f"  {np.mean(v):8.3f}")
    v = [r["sel_all_init"] for r in res["kwta"]]
    print(f"{'init(t=0)':12s} " + " ".join(f"{x:8.3f}" for x in v) + f"  {np.mean(v):8.3f}")

    def delta(trained, control):
        tv = np.array([r["sel_all_end"] for r in res[trained]])
        cv = np.array([r["sel_all_end"] for r in res[control]])
        return tv, cv, tv - cv

    print("\n=== THE CLAIM = THE DIFFERENCE ===")
    for trained, control in (("kwta", "lr0_kwta"), ("hebb_plain", "lr0_plain")):
        tv, cv, d = delta(trained, control)
        print(f"  {trained:11s} - {control:11s}: per-seed {np.round(d,3).tolist()}  "
              f"mean delta {d.mean():+.3f}   ({np.round(tv,3).tolist()} vs {np.round(cv,3).tolist()})")

    print("\n=== SECONDARY: spiking localisation (frozen-weight replay, 20 sweeps) ===")
    for name, _, _ in ARMS:
        loc = [r["spike_localisation"] for r in res[name]]
        f5 = [r["spike_frac_within_pm5"] for r in res[name]]
        npf = [r["n_distinct_pref_phase"] for r in res[name]]
        nac = [r["n_active_cells"] for r in res[name]]
        spk = [r["probe_read_spikes"] for r in res[name]]
        print(f"  {name:12s} peak/mean {np.round(loc,2).tolist()} (mean {np.nanmean(loc):.2f})  "
              f"frac±5 {np.round(f5,3).tolist()} (mean {np.nanmean(f5):.3f})  "
              f"distinct_pref {npf}  active {nac}  spikes {spk}")
    print("  -- GAIN-MATCHED replay (all arms rescaled to the same mean weight = tuning SHAPE only) --")
    for name, _, _ in ARMS:
        r0 = res[name]
        print(f"  {name:12s} peak/mean {np.round([r['gm_spike_localisation'] for r in r0],2).tolist()} "
              f"(mean {np.nanmean([r['gm_spike_localisation'] for r in r0]):.2f})  "
              f"frac±5 {np.round([r['gm_spike_frac_within_pm5'] for r in r0],3).tolist()} "
              f"(mean {np.nanmean([r['gm_spike_frac_within_pm5'] for r in r0]):.3f})  "
              f"active {[r['gm_n_active_cells'] for r in r0]}  "
              f"spikes {[r['gm_probe_read_spikes'] for r in r0]}  "
              f"scale {np.round([r['gm_scale'] for r in r0],3).tolist()}")
    for trained, control in (("kwta", "lr0_kwta"), ("hebb_plain", "lr0_plain")):
        for key, lab in (("spike_localisation", "raw peak/mean"),
                         ("spike_frac_within_pm5", "raw frac±5"),
                         ("gm_spike_localisation", "GM  peak/mean"),
                         ("gm_spike_frac_within_pm5", "GM  frac±5")):
            tv = np.array([r[key] for r in res[trained]])
            cv = np.array([r[key] for r in res[control]])
            print(f"    {lab:14s} {trained:10s} - {control:10s}: {np.round(tv-cv,3).tolist()}  "
                  f"mean {np.nanmean(tv-cv):+.3f}")
    # NOISE FLOOR of the spiking metrics: two arms that BOTH did zero learning
    for key, lab in (("spike_localisation", "raw peak/mean"), ("spike_frac_within_pm5", "raw frac±5"),
                     ("gm_spike_localisation", "GM  peak/mean"),
                     ("gm_spike_frac_within_pm5", "GM  frac±5")):
        tv = np.array([r[key] for r in res["lr0_kwta"]])
        cv = np.array([r[key] for r in res["lr0_plain"]])
        print(f"    NOISE FLOOR  {lab:14s} lr0_kwta - lr0_plain (BOTH lr=0!): "
              f"{np.round(tv-cv,3).tolist()}  mean {np.nanmean(tv-cv):+.3f}")

    print("\n=== DIFFERENTIATION (is the k-WTA actually splitting the cells up?) ===")
    for name, _, _ in ARMS:
        r0 = res[name]
        print(f"  {name:12s} distinct_winners {[r['n_distinct_winners'] for r in r0]}  "
              f"top1_win_share {np.round([r['win_share_top1'] for r in r0],2).tolist()}  "
              f"pair_cos {np.round([r['pair_cos_init'] for r in r0],3).tolist()}"
              f"->{np.round([r['pair_cos_end'] for r in r0],3).tolist()}  "
              f"argmax_moved {[str(r['n_argmax_moved'])+'/'+str(r['n_live_cells']) for r in r0]}  "
              f"|w_peak-spk_pref| {np.round([r['w_peak_vs_spike_pref_absdiff'] for r in r0],1).tolist()}")

    print("\n=== PAIRED, LEARNERS-ONLY (removes coverage dilution: compare the SAME cells) ===")
    for trained, control in (("kwta", "lr0_kwta"), ("hebb_plain", "lr0_plain")):
        dl, nl, dall = [], [], []
        for i in range(len(a.seeds)):
            rt, rc = res[trained][i], res[control][i]
            lm = np.array(rt["learner_mask"])
            te = np.array([np.nan if x is None else x for x in rt["sel_percell_end"]], dtype=float)
            ce = np.array([np.nan if x is None else x for x in rc["sel_percell_end"]], dtype=float)
            sel = lm & ~np.isnan(te) & ~np.isnan(ce)
            dl.append(float(np.mean(te[sel] - ce[sel])) if sel.any() else float("nan"))
            nl.append(int(sel.sum()))
            ok = ~np.isnan(te) & ~np.isnan(ce)
            dall.append(float(np.mean(te[ok] - ce[ok])))
        print(f"  {trained:11s} learners-only delta {np.round(dl,3).tolist()} "
              f"(mean {np.nanmean(dl):+.3f})  n_learners {nl}/12   "
              f"all-cells delta {np.round(dall,3).tolist()} (mean {np.nanmean(dall):+.3f})")

    print("\n=== IS THE POTENTIATION A CONTIGUOUS PLACE FIELD? (uniform null SD = "
          f"{res['kwta'][0]['dW_field_sd_uniform_null']:.1f}) ===")
    for name, _, _ in ARMS:
        r0 = res[name]
        print(f"  {name:12s} dW_field_SD {np.round([r['dW_field_sd'] for r in r0],1).tolist()} "
              f"(mean {np.nanmean([r['dW_field_sd'] for r in r0]):.1f})  "
              f"cells_with_potentiation {[r['dW_field_n_cells'] for r in r0]}  "
              f"inputs_potentiated/cell {np.round([r['dW_field_cover_inputs'] for r in r0],1).tolist()}  "
              f"centre_spread {np.round([r['dW_field_centre_spread'] for r in r0],1).tolist()}")
    print(f"  seed42 kwta field centres: {res['kwta'][0]['dW_field_centres']}")
    print(f"  seed42 kwta win  centres: {res['kwta'][0]['win_centres']}")
    print(f"  seed42 plain  field centres: {res['hebb_plain'][0]['dW_field_centres']}")

    print("\n=== ENGAGEMENT ASSERTIONS (rule 3 + rule 2) ===")
    for name, _, _ in ARMS:
        rs = [r["read_spikes_total"] for r in res[name]]
        ps = [r["place_spikes_total"] for r in res[name]]
        pe = [r["potentiation_events"] for r in res[name]]
        pps = [round(r["pot_events_per_syn"], 1) for r in res[name]]
        fz = [r["frozen_during_probe"] for r in res[name]]
        dw = [round(r["dW_absmean"], 2) for r in res[name]]
        bd = [round(r["frac_at_bound"], 2) for r in res[name]]
        wm = [round(r["w_mean_end"], 1) for r in res[name]]
        cr = [round(r["corr_dW_vs_winit"], 3) for r in res[name]]
        print(f"  {name:12s} read_spk {rs} (>0 REQUIRED)  place_spk {ps}  "
              f"pot_events {pe} ({pps}/syn)  dW {dw}  w_mean_end {wm}  frac_at_bound {bd}  "
              f"corr(dW,w_init) {cr}  probe_frozen {fz}")
    void = any(r["read_spikes_total"] == 0 for name, _, _ in ARMS for r in res[name])
    print(f"\n  VERDICT-GATE: any arm with ZERO read spikes -> {void} "
          f"({'VOID' if void else 'population fired, result is interpretable'})")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
