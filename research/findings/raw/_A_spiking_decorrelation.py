"""A deep-grounding arc, spiking on-bridge decorrelation de-risk: realize the ventral->IT decorrelation as a SPIKING
layer on the bridge (replacing the numpy ZCA / the numpy-Földiák reference). Competitive-learning form (the simpler
on-bridge cousin of Földiák the project already has the pieces for): an IT pool with PLASTIC Hebbian feed-forward
(input features -> IT) + FS LATERAL INHIBITION (the WTA that sparsifies + decorrelates) + HOMEOSTASIS (adaptive
thresholds keeping each IT neuron active ~p -> prevents dead/dominant units). Drive correlated input codes -> IT
learns distinct sparse codes -> measure the IT codes' cross-concept coherence (decorrelated like the numpy Földiák?).
GATE: post-training IT-code coherence << the raw input coherence (the substrate decorrelates). NO sim/ edits.
"""
import numpy as np

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.regions import BrainRegion, RegionPathway
from sim.bridge import SimulationBridge
from sim.backend import get_backend, to_host


def make_correlated_codebook(n_concepts, n_feat, n_blocks, seed):
    rng = np.random.default_rng(seed)
    block = rng.standard_normal((n_blocks, n_feat))
    X = np.zeros((n_concepts, n_feat))
    for i in range(n_concepts):
        X[i] = 0.75 * block[i % n_blocks] + 0.25 * rng.standard_normal(n_feat)
    return np.maximum(X, 0)


def coherence(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    G = np.abs(Xn @ Xn.T)
    off = G[~np.eye(len(X), dtype=bool)]
    return float(off.mean()), float(off.max())


def build(seed, n_feat, n_it=200, n_fs=120, ff_density=0.5, ff_weight=8.0, wta_weight=1.0):
    regions = [
        BrainRegion(name="inp", n_neurons=n_feat, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="it", n_neurons=n_it, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="fs", n_neurons=n_fs, exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    pathways = [
        RegionPathway(from_region="inp", to_region="it", density=ff_density, weight_mean=ff_weight, weight_jitter=2.0,
                      plastic=True, plasticity_gate="ff"),                 # plastic Hebbian feed-forward (strong: fire IT)
        RegionPathway(from_region="it", to_region="fs", density=0.5, weight_mean=1.0, weight_jitter=0.2,
                      plastic=True, plasticity_gate="lat"),                # anti-Hebbian-via-FS: co-active IT pairs
        RegionPathway(from_region="fs", to_region="it", density=0.5, weight_mean=wta_weight, weight_jitter=0.2,
                      plastic=False),                                       # strengthen shared FS drive -> decorrelate
    ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = int(seed)
    cfg.enable_nmda = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True          # competitive Hebbian feed-forward
    cfg.hebbian_max_weight = 15.0
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_learning_rate = 0.01
    cfg.enable_homeostasis = True               # adaptive thresholds -> each IT neuron active ~p (no dead/dominant)
    cfg.enable_reward_modulation = False
    cfg.enable_structural_plasticity = False
    cfg.enable_short_term_plasticity = False
    cfg.ou_std_current_pA = 0.0
    cfg.fast_spike_reset = True
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def _drive_features(b, inp_idx, feats, scale):
    cp, _ = get_backend()
    ext = cp.zeros(b.cp_external_input_current.shape[0], dtype=cp.float32)
    ext[cp.asarray(inp_idx, dtype=cp.int64)] = cp.asarray(feats * scale, dtype=cp.float32)
    b.cp_external_input_current[:] = ext


def it_code(b, inp_idx, it_idx, feats, scale=600.0, window=40, also=None):
    fn = feats / (float(feats.max()) + 1e-9)         # normalize to peak 1.0 -> strongest feature = `scale` pA (fires)
    _drive_features(b, inp_idx, fn, scale)
    acc = np.zeros(len(it_idx))
    acc2 = np.zeros(len(also)) if also is not None else None
    for _ in range(window):
        b._run_one_simulation_step()
        fs = np.asarray(to_host(b.cp_firing_states)).astype(float)
        acc += fs[it_idx]
        if also is not None:
            acc2 += fs[also]
    b.cp_external_input_current[:] = 0.0
    for _ in range(15):
        b._run_one_simulation_step()
    return (acc, acc2) if also is not None else acc


def run(seed, n_concepts=16, n_feat=256, n_blocks=4, epochs=40):
    X = make_correlated_codebook(n_concepts, n_feat, n_blocks, seed)
    b = build(seed, n_feat)
    rm = b.region_manager
    inp = np.asarray(rm.indices("inp")); it = np.asarray(rm.indices("it"))
    # pre-train probe: isolate the cold-start — does the input fire, and does the feed-forward drive IT initially?
    itc0, ic0 = it_code(b, inp, it, X[0], also=inp)
    print(f"  [probe seed={seed}] pre-train concept0: input_active={int((ic0 > 0).sum())}/{len(inp)} "
          f"IT_active={int((itc0 > 0).sum())}/{len(it)}", flush=True)
    for g in ("ff", "lat"):
        try:
            b.set_plasticity_gate(g, 1.0)
        except KeyError:
            pass
    for _ in range(epochs):
        for i in np.random.default_rng(seed).permutation(n_concepts):
            it_code(b, inp, it, X[i])
    for g in ("ff", "lat"):
        try:
            b.set_plasticity_gate(g, 0.0)
        except KeyError:
            pass
    Y = np.stack([it_code(b, inp, it, X[i]) for i in range(n_concepts)])
    active = (Y > 0).sum(1)                 # active IT units per concept (sparse code size)
    n_silent = int((active == 0).sum())     # concepts with NO IT firing (degenerate)
    return coherence(X), coherence(Y), float(active.mean()), n_silent, int(Y.sum())


if __name__ == "__main__":
    for seed in (42, 43, 44):
        (rm_, rx_), (ym_, yx_), act_, nsil_, tot_ = run(seed)
        verdict = ("DECORR (real firing, orthogonal)" if (nsil_ == 0 and ym_ < 0.2)
                   else "DEGENERATE (silent IT pool)" if tot_ == 0 or nsil_ > 0
                   else "CLUSTERED" if ym_ >= 0.2 else "?")
        print(f"seed={seed}: RAW coh mean={rm_:.3f}/max={rx_:.3f} -> IT-code coh mean={ym_:.3f}/max={yx_:.3f} "
              f"| mean_active_IT={act_:.1f}  n_silent_concepts={nsil_}/16  total_IT_spikes={tot_}  => {verdict}",
              flush=True)
