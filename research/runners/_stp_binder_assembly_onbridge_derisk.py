"""ON-SUBSTRATE binder, the MONGILLO-ASSEMBLY reframe (RUNG 6f) — the ground-up redesign the FS-WTA compound (RUNG 6e)
needed. Instead of a feedforward FS-WTA (which can't fire the winner enough for Hebbian specificity while suppressing
losers), each SLOT is a self-exciting recurrent ATTRACTOR (Wang-2002): a brief barcode drive tips one slot's recurrent
excitation into a LATCHED high-rate state, FS-WTA keeps the others silent. This dissolves the two coupled tensions:
  (#4) the latched winner fires at a HIGH sustained rate -> its co-activity with the barcode clears the Hebbian
       coactivity threshold -> the barcode->winner synapses POTENTIATE (a real bind);
  (#3) the winner is clean + high-rate by the attractor latch, not a fragile feedforward margin.
Re-present a bound barcode -> its potentiated barcode->slot drives ITS attractor into the latch first -> retrieve; a
NOVEL barcode -> occupancy routing (bound attractors suppressed) -> a fresh attractor latches.

CHEAP-FIRST LADDER (gate each): (L) LATCH smoke — a brief barcode pulse latches ONE slot to a high sustained rate after
the pulse ends (the attractor works); (W) WTA — exactly one slot latches; (B) bind/retrieve — potentiation forms + re-
present retrieves. numpy-CPU; reuse-by-import (brain-region framework + rate-window Hebbian + FS-WTA); NO `sim/` edit.

Run: SIM_BACKEND=numpy python -m research.runners._stp_binder_assembly_onbridge_derisk --latch --seed 42
     SIM_BACKEND=numpy python -m research.runners._stp_binder_assembly_onbridge_derisk --derisk --seed 42
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import time

import numpy as np

_K = 4
_N_BARCODE = 48
_KACT = 6
_N_SLOT = 40
_N_FS = 24


def build_assembly_bridge(seed, plastic=True, hebb_lr=0.1, rec_w=18.0, fs_w=6.0, rec_density=0.4):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    regions = [BrainRegion(name="barcode", n_neurons=_N_BARCODE, exc_fraction=1.0, internal_density=0.0)]
    for s in range(_K):
        # SLOT = a self-exciting recurrent ATTRACTOR (internal_density + strong exc -> latch)
        regions.append(BrainRegion(name=f"slot{s}", n_neurons=_N_SLOT, exc_fraction=1.0,
                                   internal_density=rec_density, exc_weight_mean=rec_w, weight_jitter=0.2,
                                   plastic_internal=False))
    regions.append(BrainRegion(name="slot_fs", n_neurons=_N_FS, exc_fraction=0.0, internal_density=0.0))
    cfg.brain_regions = regions
    paths = []
    for s in range(_K):
        paths.append(RegionPathway(from_region="barcode", to_region=f"slot{s}", density=1.0,
                                   weight_mean=3000.0, weight_jitter=0.15, plastic=plastic))
        paths.append(RegionPathway(from_region=f"slot{s}", to_region="slot_fs", density=0.6, weight_mean=2.0, plastic=False))
        paths.append(RegionPathway(from_region="slot_fs", to_region=f"slot{s}", density=0.6, weight_mean=fs_w, plastic=False))
    cfg.region_pathways = paths
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = True; cfg.ou_noise_amplitude_pA = 40.0    # symmetry-break for the first-mention winner
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = plastic
    cfg.hebbian_rate_window = True
    cfg.hebbian_coactivity_decay = 0.9
    cfg.hebbian_learning_rate = hebb_lr
    cfg.hebbian_max_weight = 6000.0
    cfg.hebbian_coactivity_thresh = 0.05
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    idx = {"barcode": np.asarray(b.region_manager.indices("barcode"))}
    for s in range(_K):
        idx[f"slot{s}"] = np.asarray(b.region_manager.indices(f"slot{s}"))
    return b, idx


def _slot_rates(b, idx, fs):
    return np.array([fs[idx[f"slot{s}"]].sum() for s in range(_K)]) / _N_SLOT


def present(b, idx, code, pulse=25, hold=35, drive_pA=1400.0, learn=True, suppress_slots=()):
    """PULSE the barcode for `pulse` steps (tip the attractor), then HOLD `hold` steps with NO barcode drive (let the
    winner LATCH via its own recurrence). Return per-slot mean rate over the HOLD window (the latched state)."""
    from sim.backend import to_host
    xp = b.xp
    b.core_config.enable_hebbian_learning = bool(learn) and (b.core_config.hebbian_learning_rate > 0)
    bc = idx["barcode"]; active = bc[code > 0]
    hold_counts = np.zeros(_K, np.float64)
    for t in range(pulse + hold):
        b.cp_external_input_current[:] = 0.0
        if t < pulse:
            b.cp_external_input_current[active] = drive_pA
        for _ss in suppress_slots:
            b.cp_external_input_current[idx[f"slot{_ss}"]] = -1500.0     # occupied attractors suppressed
        b._run_one_simulation_step()
        if t >= pulse:
            hold_counts += _slot_rates(b, idx, np.asarray(to_host(b.cp_firing_states)).astype(np.float64)) * _N_SLOT
    b.cp_external_input_current[:] = 0.0
    return hold_counts / (hold * _N_SLOT)


def _mint(rng, M):
    codes = []
    while len(codes) < M:
        c = np.zeros(_N_BARCODE, np.float32); c[rng.choice(_N_BARCODE, _KACT, replace=False)] = 1.0
        if all(float((c > 0) @ (d > 0)) < 3 for d in codes):
            codes.append(c)
    return np.asarray(codes)


def run_latch(seed=42, **kw):
    """Does a brief barcode pulse LATCH one slot to a sustained high rate AFTER the pulse (the attractor works)?"""
    b, idx = build_assembly_bridge(seed, plastic=False, **kw)
    code = _mint(np.random.default_rng(seed), 1)[0]
    r = present(b, idx, code, learn=False)
    win = int(np.argmax(r)); margin = r[win] - np.sort(r)[-2]
    print(f"[assembly-latch seed={seed}] HOLD-window rates={np.round(r,3)} winner={win} latched_rate={r[win]:.3f} "
          f"margin={margin:.3f} -> {'LATCH OK' if r[win] > 0.15 and margin > 0.08 else 'no clean latch (tune rec_w/fs_w)'}")
    return r


def run_derisk(seed=42):
    b, idx = build_assembly_bridge(seed, plastic=True)
    rng = np.random.default_rng(seed); codes = _mint(rng, 4)
    bind = {}; occupied = []
    for e in (0, 1):
        w = None
        for _rep in range(3):
            r = present(b, idx, codes[e], learn=True, suppress_slots=tuple(occupied)); w = int(np.argmax(r))
        bind[e] = w; occupied.append(w)
    ok = 0
    for e in (0, 1):
        r = present(b, idx, codes[e], learn=False)
        if int(np.argmax(r)) == bind[e]:
            ok += 1
    distinct = bind[0] != bind[1]
    print(f"[assembly-derisk seed={seed}] bind={bind} distinct={distinct} retrieve={ok}/2 "
          f"-> {'RETRIEVE OK' if ok == 2 and distinct else 'iterate'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latch", action="store_true"); ap.add_argument("--derisk", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rec-w", type=float, default=18.0); ap.add_argument("--fs-w", type=float, default=6.0)
    a = ap.parse_args()
    t0 = time.time()
    if a.latch:
        run_latch(a.seed, rec_w=a.rec_w, fs_w=a.fs_w)
    else:
        run_derisk(a.seed)
    print(f"  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
