"""D cue-recall RESOLUTION attempt: a SPARSE recurrent heteroassociative memory. The dense v16 failed (capacity wall);
sparse codes give the Treves-Rolls capacity for clean cue->associate completion. Architecture: a shared pool with
SPARSE concept patterns (K-of-N) + a PLASTIC excitatory recurrent (the heteroassociative weights) + FS inhibition
(sparsity/WTA). Encode pairs (co-activate a's+b's sparse patterns -> recurrent a<->b grows via STDP); cue-recall
(drive a's pattern alone -> does b's pattern COMPLETE via the recurrent, selectively?); SWR consolidation (more
co-replay -> stronger). GATE: post-encode/SWR, driving a completes b's pattern far above the other concepts
(clean heteroassociative completion), SPECIFICALLY (driving a must NOT complete a non-associate). McClelland CLS +
Marr/Treves-Rolls sparse autoassociation. Design: docs/plans/2026-06-05-D-cue-recall-SWR-consolidation-design.md.
"""
import argparse

import numpy as np

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.regions import BrainRegion, RegionPathway
from sim.bridge import SimulationBridge
from sim.backend import get_backend, to_host
from research.runners.concept_pool_sparse_distributed import generate_sparse_patterns


def build(seed, n_pool=2000, n_fs=300, rec_density=0.6, fs_inh=1.2):
    regions = [
        BrainRegion(name="pool", n_neurons=n_pool, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="fs", n_neurons=n_fs, exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    pathways = [
        # the PLASTIC excitatory recurrent = the heteroassociative weights (zero-init, grown by co-fire STDP)
        RegionPathway(from_region="pool", to_region="pool", density=rec_density, weight_mean=0.0,
                      weight_jitter=0.0, plastic=True, plasticity_gate="recurrent"),
        # FS inhibition for sparsity / clean completion (the WTA that prevents a dense superposition)
        RegionPathway(from_region="pool", to_region="fs", density=0.3, weight_mean=1.0, weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="fs", to_region="pool", density=0.3, weight_mean=fs_inh, weight_jitter=0.2, plastic=False),
    ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = int(seed)
    cfg.enable_nmda = False
    cfg.enable_stdp = False
    cfg.stdp_w_max = 8.0
    # Direct Hebbian co-fire growth for the recurrent (the bridge's STDP without reward forms eligibility but never
    # applies it -> weight stays 0; Hebbian co-firing grows the region-pathway weight directly, confirmed prior).
    cfg.enable_hebbian_learning = True
    cfg.enable_reward_modulation = False
    # the default Hebbian caps growth at max_weight=1.0 (< the ~3.0 the lang->pool runs at) and floors every edge at
    # 0.05 (broad background). Lift the cap so co-fired a->b reaches functional strength; drop the floor so
    # non-co-fired edges stay 0 (clean selectivity); raise the rate so it reaches strength in a few cycles.
    cfg.hebbian_max_weight = 45.0
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_learning_rate = 0.004
    cfg.enable_structural_plasticity = False
    cfg.enable_short_term_plasticity = False
    cfg.ou_std_current_pA = 0.0
    cfg.fast_spike_reset = True
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def _pool_global(bridge, patterns):
    base = np.asarray(bridge.region_manager.indices("pool"))
    return [base[np.asarray(p)] for p in patterns]


def _drive(bridge, idx_list, pA):
    cp, _ = get_backend()
    n = bridge.cp_external_input_current.shape[0]
    ext = cp.zeros(n, dtype=cp.float32)
    for idx in idx_list:
        ext[cp.asarray(idx, dtype=cp.int64)] = pA
    bridge.cp_external_input_current[:] = ext


def co_replay(bridge, pg, pairs, cycles, pA=1100.0, on_steps=10, off_steps=5):
    try:
        bridge.set_plasticity_gate("recurrent", 1.0)
    except KeyError:
        pass
    for _ in range(cycles):
        for ai, bi in pairs:
            _drive(bridge, [pg[ai], pg[bi]], pA)
            for _ in range(on_steps):
                bridge._run_one_simulation_step()
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(off_steps):
                bridge._run_one_simulation_step()
    try:
        bridge.set_plasticity_gate("recurrent", 0.0)
    except KeyError:
        pass


def completion_profile(bridge, patterns, pg, cue_i, n_pool, window=40, pA=1100.0):
    """Drive cue's pattern alone -> accumulate pool firing -> EXCLUDE the cue's own (directly-driven) neurons ->
    cosine of the RECURRENT output to each concept's sparse pattern. Excluding the cue removes the pattern-overlap
    confound (overlapping concepts share the directly-driven neurons); what remains is the heteroassociative
    completion driven by the learned recurrent."""
    pool_base = np.asarray(bridge.region_manager.indices("pool"))
    _drive(bridge, [pg[cue_i]], pA)
    firing = np.zeros(n_pool)
    for _ in range(window):
        bridge._run_one_simulation_step()
        fs = np.asarray(to_host(bridge.cp_firing_states)).astype(float)
        firing += fs[pool_base]
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(15):
        bridge._run_one_simulation_step()
    firing[np.asarray(patterns[cue_i])] = 0.0           # exclude the directly-driven cue neurons -> recurrent output
    prof = np.zeros(len(patterns))
    nf = float(np.linalg.norm(firing))
    for j, pat in enumerate(patterns):
        v = np.zeros(n_pool); v[np.asarray(pat)] = 1.0
        prof[j] = float(firing @ v / (nf * np.linalg.norm(v))) if nf > 0 else 0.0
    return prof


def run(seed, n_concepts=4, pattern_size=120, n_pool=2000, enc_cycles=40, swr_cycles=40):
    b = build(seed, n_pool=n_pool)
    patterns = generate_sparse_patterns(n_concepts, n_pool, pattern_size, seed)
    pg = _pool_global(b, patterns)
    pairs = [(2 * k, 2 * k + 1) for k in range(n_concepts // 2)]   # (0,1),(2,3),... a->b associations

    co_replay(b, pg, pairs, enc_cycles)                            # ENCODE
    pool = np.asarray(b.region_manager.indices("pool"))
    M = to_host(b.cp_connections); sub = M[pool][:, pool]
    d = np.asarray(sub.data if hasattr(sub, "data") else sub)
    print(f"  [recurrent pool->pool weight after encode: mean={float(np.abs(d).mean()):.3f} "
          f"max={float(np.abs(d).max()):.3f} nnz={len(d)}]", flush=True)
    base = {(a, bb): _rank(completion_profile(b, patterns, pg, a, n_pool), bb, a) for a, bb in pairs}
    co_replay(b, pg, pairs, swr_cycles)                            # SWR consolidation
    post = {(a, bb): _rank(completion_profile(b, patterns, pg, a, n_pool), bb, a) for a, bb in pairs}
    return pairs, base, post


def _rank(profile, target, cue):
    """Is `target` the top completion (excluding the cue itself)? Return (is_top1, target_rank, margin)."""
    order = [j for j in np.argsort(-profile) if j != cue]
    rank = order.index(target)
    margin = profile[target] - (profile[order[0]] if order[0] != target else (profile[order[1]] if len(order) > 1 else 0))
    return (rank == 0, rank, float(margin))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--swr-cycles", type=int, default=40)
    args = p.parse_args()
    pairs, base, post = run(args.seed, swr_cycles=args.swr_cycles)
    nb = sum(1 for v in base.values() if v[0]); npo = sum(1 for v in post.values() if v[0])
    print(f"=== D sparse heteroassoc (seed={args.seed}) ===")
    print(f"cue->associate TOP-1 completion: post-ENCODE {nb}/{len(pairs)}  ->  post-SWR {npo}/{len(pairs)}")
    for (a, bb) in pairs:
        print(f"  c{a}->c{bb}: encode top1={base[(a,bb)][0]} rank={base[(a,bb)][1]} "
              f"-> swr top1={post[(a,bb)][0]} rank={post[(a,bb)][1]}", flush=True)
