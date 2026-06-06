"""A deep-grounding, interneuron-diversity cheap-first de-risk (the one remaining on-bridge lever for the attribute
residual). The single global FS pool can't do Földiák's PAIRWISE decorrelation: it inhibits ALL IT uniformly, so it
either under-decorrelates or over-suppresses the whole pool (the documented instability). The fix is LOCAL/SPECIFIC
inhibition. Simplest faithful realization: K topographic FS sub-pools, each doing local WTA on its OWN IT window
(disjoint here as the first test) -- K independent local decorrelators on the shared input instead of one global one.
GATE: does the worst-pair MAX coherence drop below the single-global-pool baseline (~0.91 toy)? If yes -> the diversity
direction works, justify the full (overlapping-window / multi-type) build. If no -> real multi-TYPE dynamics diversity
needed. NO sim/ edits -- pure brain-region-framework composition.
"""
import numpy as np

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.regions import BrainRegion, RegionPathway
from sim.bridge import SimulationBridge
from research.findings.raw._A_spiking_decorrelation import make_correlated_codebook, coherence, it_code


def _exc(name, n):
    return BrainRegion(name=name, n_neurons=n, exc_fraction=1.0, internal_density=0.0,
                       exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)


def _inh(name, n):
    return BrainRegion(name=name, n_neurons=n, exc_fraction=0.0, internal_density=0.0,
                       exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)


def build_topographic(seed, n_feat, K=8, it_per=25, fs_per=15, ff_weight=8.0):
    """K disjoint local WTA sub-pools: inp -> it_k (plastic Hebbian) + it_k <-> fs_k (LOCAL winner-take-all)."""
    regions = [_exc("inp", n_feat)]
    pathways = []
    for k in range(K):
        regions += [_exc(f"it{k}", it_per), _inh(f"fs{k}", fs_per)]
        pathways += [
            RegionPathway(from_region="inp", to_region=f"it{k}", density=0.5, weight_mean=ff_weight,
                          weight_jitter=2.0, plastic=True, plasticity_gate="ff"),
            RegionPathway(from_region=f"it{k}", to_region=f"fs{k}", density=0.5, weight_mean=1.0,
                          weight_jitter=0.2, plastic=True, plasticity_gate="lat"),   # local anti-Hebbian
            RegionPathway(from_region=f"fs{k}", to_region=f"it{k}", density=0.5, weight_mean=1.0,
                          weight_jitter=0.2, plastic=False),                          # local inhibition only
        ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = int(seed)
    cfg.enable_nmda = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_max_weight = 15.0
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_learning_rate = 0.01
    cfg.enable_homeostasis = True
    cfg.enable_reward_modulation = False
    cfg.enable_structural_plasticity = False
    cfg.enable_short_term_plasticity = False
    cfg.ou_std_current_pA = 0.0
    cfg.fast_spike_reset = True
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def run(seed, n_concepts=16, n_feat=256, n_blocks=4, epochs=40, K=8):
    X = make_correlated_codebook(n_concepts, n_feat, n_blocks, seed)
    b = build_topographic(seed, n_feat, K=K)
    rm = b.region_manager
    inp = np.asarray(rm.indices("inp"))
    it = np.concatenate([np.asarray(rm.indices(f"it{k}")) for k in range(K)])
    for g in ("ff", "lat"):
        try:
            b.set_plasticity_gate(g, 1.0)
        except KeyError:
            pass
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for i in rng.permutation(n_concepts):
            it_code(b, inp, it, X[i])
    for g in ("ff", "lat"):
        try:
            b.set_plasticity_gate(g, 0.0)
        except KeyError:
            pass
    Y = np.stack([it_code(b, inp, it, X[i]) for i in range(n_concepts)])
    active = (Y > 0).sum(1)
    return coherence(X), coherence(Y), float(active.mean()), int((active == 0).sum())


if __name__ == "__main__":
    print("single-global-pool baseline (committed _A_spiking_decorrelation +anti-Hebbian): max coh ~0.91 all seeds",
          flush=True)
    for seed in (42, 43, 44):
        (rm_, rx_), (ym_, yx_), act_, nsil_ = run(seed)
        verdict = "DIVERSITY HELPS (max<0.7)" if yx_ < 0.7 else "no improvement (max>=0.7)"
        print(f"seed={seed}: TOPOGRAPHIC-FS K=8 local | RAW coh {rm_:.3f}/{rx_:.3f} -> IT coh mean={ym_:.3f}/"
              f"max={yx_:.3f} | active={act_:.1f} silent={nsil_}/16  => {verdict}", flush=True)
