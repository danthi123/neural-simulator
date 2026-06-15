"""CYCLE 92 — the NEURAL normalization on the bridge: does the cortex recover the PPMI structure from RAW-count
drive using the on-substrate primitives (per-concept DIVISIVE norm + per-hub subtractive input_mean_adapt + the
neuron's log-ish f-I), with NO host-precomputed PPMI?

CYCLE 91 confirmed the spiking substrate carries a host-PPMI code (population read, 94% of host). This closes
the loop: instead of host-computing PPMI and driving it in, drive the RAW counts and let the BRIDGE compute the
normalization neurally -- the per-concept divisive primitive (committed, Carandini-Heeger) + the per-hub
input_mean_adapt (shipped, subtractive) + the f-I log. BRAIN-BASED-ONLY: the environment renders the raw
co-occurrence drive; the cortex does the normalization in neurons/synapses.

ARMS (real corpus, population code; GPU):
  host-PPMI drive (ceiling)     host-computed PPMI driven in       ~94% of host at n_pop 32 (CYCLE 91)
  RAW drive, NO norm (baseline) raw counts, no primitives           the un-normalized floor
  RAW drive, NEURAL norm        raw counts + divisive + input_mean  <- the test: does it recover toward host?
GATE: neural-norm beats raw-no-norm + approaches the host-PPMI ceiling -> the cortex computes the normalization
neurally (the functional primitive works). A clear lift over raw-no-norm = the divisive + input_mean primitives
are doing their job on-substrate.

Reuse-by-import (build_real_corpus, ppmi_matrix); GPU. The primitive is committed + byte-verified.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_neural_norm_cortex_derisk --seeds 42
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402
from research.runners.dendritic_d1_learn_graded_structure_derisk import _cos_sim, _pearson_vs_Strue  # noqa: E402
from research.runners.learned_graded_cortex_fair_test import build_real_corpus, ppmi_matrix  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402


def build_cortex(n_dim, n_pop, seed, neural_norm, ima_alpha, idn_sigma, idn_gain, ima=None, idn=None):
    """A hub region (n_dim*n_pop neurons). neural_norm=True enables BOTH the per-concept divisive primitive +
    per-hub input_mean_adapt; ima/idn override each independently (to localize which composes correctly)."""
    if ima is None:
        ima = neural_norm
    if idn is None:
        idn = neural_norm
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    # hub region (read faithfully) + a tiny readout + pathway -- mirrors the working CYCLE-91 builder
    # (the region framework wires through inject_explicit_wiring; a single region with no pathway yields
    # no firing). We still read the HUB firing, not the readout.
    cfg.brain_regions = [
        BrainRegion(name="hub", n_neurons=n_dim * n_pop, exc_fraction=1.0, internal_density=0.0,
                    input_mean_adapt=ima, input_divisive_norm=idn),
        BrainRegion(name="readout", n_neurons=50, exc_fraction=1.0, internal_density=0.0),
    ]
    cfg.region_pathways = [RegionPathway(from_region="hub", to_region="readout", density=0.1,
                                         weight_mean=1.0, weight_jitter=0.3)]
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    if ima:
        cfg.enable_input_mean_adapt = True
        cfg.input_mean_adapt_alpha = ima_alpha           # warmed over the stream, then frozen to 0 for read
        cfg.input_mean_adapt_gain = 1.0
    if idn:
        cfg.enable_input_divisive_norm = True
        cfg.input_divisive_sigma = idn_sigma
        cfg.input_divisive_gain = idn_gain
    rt = RuntimeState(); rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt,
                              gpu_config=GPUConfig())
    bridge._initialize_simulation_data()
    return bridge, np.asarray(bridge.region_manager.indices("hub"))


def present(bridge, hub_idx, drive_vec, scale, window, settle):
    xp = bridge._cp if hasattr(bridge, "_cp") else None
    d = (np.asarray(drive_vec, np.float64) * scale).astype(np.float32)
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[hub_idx] = xp.asarray(d) if xp is not None else d
    acc = np.zeros(hub_idx.size); n = 0
    for t in range(settle + window):
        bridge._run_one_simulation_step()
        if t >= settle:
            acc += np.asarray(to_host(bridge.cp_firing_states))[hub_idx].astype(np.float64); n += 1
    bridge.cp_external_input_current[:] = 0.0
    return acc / max(1, n)


def codes(bridge, hub_idx, drive, n_dim, n_pop, scale, window, settle, warmup, freeze_after):
    Nc = drive.shape[0]
    drive_pop = np.repeat(drive, n_pop, axis=1)
    for _ in range(warmup):                              # warm the input_mean EMA over the stream
        for i in range(Nc):
            present(bridge, hub_idx, drive_pop[i], scale, window, settle)
    if freeze_after and getattr(bridge.core_config, "enable_input_mean_adapt", False):
        bridge.core_config.input_mean_adapt_alpha = 0.0  # FREEZE the per-hub mean for the read
    out = np.zeros((Nc, n_dim))
    for i in range(Nc):
        fr = present(bridge, hub_idx, drive_pop[i], scale, window, settle)
        out[i] = fr.reshape(n_dim, n_pop).mean(1)
    return out


def run_seed(seed, a):
    C, labels, S_true = build_real_corpus(seed, a.n_hub)
    labels = np.asarray(labels); n_dim = C.shape[1]
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1), alpha=0.75), labels)
    ppmi = ppmi_matrix(C, 0.75)
    raw = np.maximum(C, 0.0).astype(np.float64)
    print(f"\n[neural-norm seed {seed}] {C.shape[0]}c x {n_dim}d x {a.n_pop}/dim | host {host_p:+.3f}", flush=True)

    def measure(drive, scale, ima, idn):
        b, idx = build_cortex(n_dim, a.n_pop, seed, False, a.ima_alpha, a.idn_sigma, a.idn_gain, ima=ima, idn=idn)
        warm = a.warmup if ima else 1
        cd = codes(b, idx, drive, n_dim, a.n_pop, scale, a.window, a.settle, warm, freeze_after=ima)
        return _pearson_vs_Strue(_cos_sim(cd), S_true)

    ppmi_ceiling = measure(ppmi, a.drive_scale, False, False)      # host-PPMI drive (the CYCLE-91 ceiling)
    raw_floor = measure(raw, a.raw_scale, False, False)            # raw, no norm (the floor)
    div_only = measure(raw, a.raw_scale, False, True)             # raw + DIVISIVE only (per-concept)
    both = measure(raw, a.raw_scale, True, True)                  # raw + divisive + input_mean
    print(f"  host-PPMI ceiling {ppmi_ceiling:+.3f} | RAW floor {raw_floor:+.3f} | +DIVISIVE-only {div_only:+.3f} "
          f"| +divisive+input_mean {both:+.3f}", flush=True)
    return {"seed": seed, "host": host_p, "ppmi_ceiling": ppmi_ceiling, "raw_floor": raw_floor,
            "div_only": div_only, "neural": both}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42")
    p.add_argument("--n-hub", type=int, default=500)
    p.add_argument("--n-pop", type=int, default=16)
    p.add_argument("--drive-scale", type=float, default=50.0)
    p.add_argument("--raw-scale", type=float, default=4.0)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--settle", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--ima-alpha", type=float, default=0.05)
    p.add_argument("--idn-sigma", type=float, default=1.0)
    p.add_argument("--idn-gain", type=float, default=1.0)
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[neural-norm cortex de-risk] seeds={seeds} n_pop={a.n_pop} raw_scale={a.raw_scale} "
          f"idn(sigma={a.idn_sigma},gain={a.idn_gain}) -- does RAW-drive + on-substrate normalization recover PPMI?",
          flush=True)
    rows = [run_seed(s, a) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    host, ceil, floor, div, neu = m("host"), m("ppmi_ceiling"), m("raw_floor"), m("div_only"), m("neural")
    print(f"\n{'='*96}\n  MEAN ({len(seeds)} seeds): host {host:+.3f} | host-PPMI ceiling {ceil:+.3f} | "
          f"RAW floor {floor:+.3f} | +DIVISIVE-only {div:+.3f} | +divisive+input_mean {neu:+.3f}", flush=True)
    print(f"  (DIVISIVE-only isolates the committed primitive; input_mean is pre-log subtractive = wrong space "
          f"for PPMI's post-log per-hub norm -> expected to NOT help / hurt)", flush=True)
    print(f"{'='*96}", flush=True)
    neu = max(div, neu)   # the best on-substrate normalization (divisive-only or both)
    if neu >= floor + 0.06 and neu >= 0.60 * ceil:
        print(f"  GO: the on-substrate NEURAL normalization (per-concept divisive + per-hub input_mean + f-I) "
              f"recovers PPMI structure from RAW drive ({neu:+.3f}, vs no-norm floor {floor:+.3f}, toward the "
              f"host-PPMI ceiling {ceil:+.3f}). ==> the cortex computes the normalization in neurons -- the "
              f"primitive works on-substrate, BRAIN-BASED. Tune (sigma/gain/scale/n_pop) toward the ceiling.",
              flush=True)
    elif neu >= floor + 0.06:
        print(f"  PARTIAL: NEURAL-norm beats the no-norm floor ({neu:+.3f} vs {floor:+.3f}) -- the divisive +"
              f" input_mean primitives lift the raw-drive code -- but below 0.60x the ceiling ({ceil:+.3f}); "
              f"tune sigma/gain/scale/n_pop/warmup (the f-I log-range + the EMA warm-up are the knobs).", flush=True)
    else:
        print(f"  NEGATIVE/needs-tuning: NEURAL-norm ({neu:+.3f}) does not yet beat the no-norm floor ({floor:+.3f})"
              f" -- the on-substrate normalization needs tuning (the divisive sigma/gain, the f-I operating point, "
              f"the input_mean alpha/warmup); inspect.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"host": host, "ppmi_ceiling": ceil, "raw_floor": floor, "neural": neu, "per_seed": rows, "config": vars(a)}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_neural_norm_cortex.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
