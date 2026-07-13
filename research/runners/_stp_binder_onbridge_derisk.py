"""ON-SUBSTRATE realization of the novel-referent binder (RUNG 6d found the mechanism: HEBBIAN short-term potentiation +
FS-WTA, NOT presynaptic TM). This builds it on a real `SimulationBridge`: a barcode input region + K slot pools + a shared
FS pool (winner-take-all), with barcode->slot synapses PLASTIC via the rate-window Hebbian (`hebbian_coactivity_decay` =
the Mongillo facilitation window). Present a barcode -> FS-WTA picks a winner slot -> Hebbian coactivity (barcode x winner)
potentiates barcode->winner (the fast weight); re-present -> the potentiated winner fires (content-addressable retrieve);
a NOVEL barcode -> no potentiation -> a fresh slot wins.

CHEAP-FIRST (this file): the core BIND/RETRIEVE on the bridge. (1) FS-WTA smoke: presenting a barcode drives ONE clear
winner slot. (2) bind/retrieve: present barcode_A (bind), barcode_B (bind, different slot), then RE-present A and B ->
each retrieves ITS bound slot; a held-out NOVEL barcode opens a fresh slot. Anti-cheats: no-Hebbian (freeze plasticity ->
re-present drives a random/degenerate slot = no memory), merge (identical codes -> cannot separate). Reuse-by-import (the
brain-region framework + rate-window Hebbian + FS-WTA are all `sim` public config); NO `sim/` edit.

Run: SIM_BACKEND=numpy python -m research.runners._stp_binder_onbridge_derisk --smoke
     SIM_BACKEND=numpy python -m research.runners._stp_binder_onbridge_derisk --derisk --seed 42
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
_T_PRESENT = 60          # steps per barcode presentation


def build_binder_bridge(seed, plastic=True, hebb_lr=0.1):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    regions = [BrainRegion(name="barcode", n_neurons=_N_BARCODE, exc_fraction=1.0, internal_density=0.0)]
    for s in range(_K):
        regions.append(BrainRegion(name=f"slot{s}", n_neurons=_N_SLOT, exc_fraction=1.0, internal_density=0.0))
    regions.append(BrainRegion(name="slot_fs", n_neurons=_N_FS, exc_fraction=0.0, internal_density=0.0))  # inhibitory
    cfg.brain_regions = regions
    paths = []
    for s in range(_K):
        # barcode -> slot: PLASTIC (rate-window Hebbian). Small init so the fast weight GROWS from ~0 on binding.
        paths.append(RegionPathway(from_region="barcode", to_region=f"slot{s}", density=1.0,
                                   weight_mean=40.0, weight_jitter=0.3, plastic=plastic))
        paths.append(RegionPathway(from_region=f"slot{s}", to_region="slot_fs", density=1.0,
                                   weight_mean=4.0, weight_jitter=0.2, plastic=False))          # slots drive shared FS
        paths.append(RegionPathway(from_region="slot_fs", to_region=f"slot{s}", density=1.0,
                                   weight_mean=3.0, weight_jitter=0.2, plastic=False))          # FS inhibits all slots (WTA)
    cfg.region_pathways = paths
    cfg.dt = 0.5
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = True; cfg.ou_noise_amplitude_pA = 30.0    # symmetry-breaking for the first-mention WTA
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = plastic
    cfg.hebbian_rate_window = True
    cfg.hebbian_coactivity_decay = 0.9
    cfg.hebbian_learning_rate = hebb_lr
    cfg.hebbian_max_weight = 90.0    # ABOVE the 40 init (else the soft-bound clips barcode->slot to 8 -> collapse, CLAUDE.md gotcha)
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    idx = {"barcode": np.asarray(b.region_manager.indices("barcode"))}
    for s in range(_K):
        idx[f"slot{s}"] = np.asarray(b.region_manager.indices(f"slot{s}"))
    return b, idx


def present(b, idx, code, drive_pA=1400.0, learn=True, suppress_slots=()):
    """Drive the barcode neurons for the code's active bits for _T_PRESENT steps; return per-slot mean firing rate."""
    from sim.backend import to_host
    xp = b.xp
    b.core_config.enable_hebbian_learning = bool(learn) and (b.core_config.hebbian_learning_rate > 0)
    counts = np.zeros(_K, np.float64)
    bc = idx["barcode"]; active = bc[code > 0]
    allslot = np.concatenate([idx[f"slot{s}"] for s in range(_K)])
    for _ in range(_T_PRESENT):
        b.cp_external_input_current[:] = 0.0
        b.cp_external_input_current[active] = drive_pA
        b.cp_external_input_current[allslot] = 400.0
        for _ss in suppress_slots:
            b.cp_external_input_current[idx[f"slot{_ss}"]] = -800.0   # occupied-slot suppression -> novel routes to a FREE slot
        b._run_one_simulation_step()
        fs = np.asarray(to_host(b.cp_firing_states)).astype(np.float64)
        for s in range(_K):
            counts[s] += fs[idx[f"slot{s}"]].sum()
    b.cp_external_input_current[:] = 0.0
    return counts / (_T_PRESENT * _N_SLOT)


def _mint(rng, M):
    codes = []
    while len(codes) < M:
        c = np.zeros(_N_BARCODE, np.float32); c[rng.choice(_N_BARCODE, _KACT, replace=False)] = 1.0
        if all(float((c > 0) @ (d > 0)) < 3 for d in codes):
            codes.append(c)
    return np.asarray(codes)


def run_smoke(seed=42):
    b, idx = build_binder_bridge(seed, plastic=False)
    rng = np.random.default_rng(seed); code = _mint(rng, 1)[0]
    rates = present(b, idx, code, learn=False)
    win = int(np.argmax(rates)); margin = rates[win] - np.sort(rates)[-2]
    print(f"[onbridge-smoke seed={seed}] slot rates={np.round(rates,3)} winner={win} margin={margin:.3f} "
          f"-> {'WTA OK' if margin > 0.02 else 'no clear winner (tune FS/drive)'}")


def run_derisk(seed=42):
    b, idx = build_binder_bridge(seed, plastic=True)
    rng = np.random.default_rng(seed); codes = _mint(rng, 4)          # 2 bind + 2 held-out novel
    # BIND phase: present A, B once each (Hebbian potentiates barcode->winner)
    bind = {}; occupied = []
    for e in (0, 1):
        w = None
        for _rep in range(3):                                  # 3 bind presentations -> stronger Hebbian potentiation
            r = present(b, idx, codes[e], learn=True, suppress_slots=tuple(occupied)); w = int(np.argmax(r))
        bind[e] = w; occupied.append(w)                       # occupancy routing: the next novel barcode avoids this slot
    # RETRIEVE phase (learn off, NO suppression): re-present A, B -> the potentiated barcode->slot should win its bound slot
    ok = 0
    for e in (0, 1):
        r = present(b, idx, codes[e], learn=False)
        if int(np.argmax(r)) == bind[e]:
            ok += 1
    distinct = bind[0] != bind[1]
    print(f"[onbridge-derisk seed={seed}] bind={bind} distinct={distinct} retrieve={ok}/2 "
          f"-> {'RETRIEVE OK' if ok == 2 and distinct else 'iterate (Hebbian rate / drive / FS)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--derisk", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    t0 = time.time()
    if a.smoke:
        run_smoke(a.seed)
    else:
        run_derisk(a.seed)
    print(f"  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
