"""CYCLE 95/96 — the CAPSTONE: the BIOLOGY-FAITHFUL online stream cortex ON THE SPIKING BRIDGE.

CYCLE 94 (numpy) proved a cortex that HEARS TinyStories word-by-word (online Hebbian co-occurrence in a
working-memory window + running-frequency normalization, NO batch matrix, NO whole-corpus PPMI) reaches the
target (+0.513). CYCLE 95 validated, on the real SimulationBridge, the three pieces that realize it on the
spiking substrate: the representation (population code, ~94% of host, CYCLE 91), the normalization (log-domain
double-centring circuit, +0.285, CYCLE 93b), and the LEARNING (rate-Hebbian co-occurrence, corr(M,C) +0.705 --
STDP is the WRONG rule, measured 656k events / 0 change at delta_t~0). This runner COMPOSES them: it streams the
ACTUAL corpus windows into co-activations on a population bridge and lets the bridge's own Hebbian synapses
accumulate the co-occurrence M -- the brain's cortex learning from the conversation stream, on the substrate.

FAITHFULNESS: the drive carries NO precomputed co-occurrence -- each window just co-activates the populations of
the words that co-occur in that window (exactly what arrives in the stream). The co-occurrence is LEARNED in the
synapses by Hebbian coincidence (pre at t-1 AND post at t), never tabulated host-side. The read-out is the
population block-mean of the learned weights + the log-domain double-centring (the validated normalization).

GATE (multi-seed): the on-bridge stream-learned code Pearson(cos, S_true) beats chance + approaches the host
reference (the log-double-centre of the batch counts) AND generalises (held-out). Anti-cheat: permuted ~0 (the
structure is LEARNED, not wired); the drive sees only the stream windows (no global statistics); the streamed M
matches the true co-occurrence C (corr(M,C)).

Reuse-by-import: the population bridge (build_assoc_bridge), the stream/tokenization (CYCLE 94), the taxonomy.
GPU (CuPy) for the real run; numpy only for a tiny smoke.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onbridge_stream_cortex_derisk --seeds 42 --n-per 16
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402
from research.runners.dendritic_d1_learn_graded_structure_derisk import _cos_sim, _pearson_vs_Strue, heldout_generalization  # noqa: E402
from research.runners.option_c_real_cooccurrence_derisk import TAXONOMY_8x8, taxonomy_to_vocab_categories  # noqa: E402
from research.runners.option_c_stageB_fair_test import STOPLIST  # noqa: E402

WINDOW = 2


def double_center(X):
    return X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean()


def load_token_stream():
    path = os.path.join(_REPO, "data", "corpus", "tinystories.txt")
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    return [re.findall(r"[a-z]+", s) for s in text.split("<|endoftext|>")]


def build_stream_bridge(n_target, n_hub, n_per, seed):
    """Population bridge: hub (context-word) region + target (concept) region, n_per neurons/concept, a
    fully-connected hub->target plastic pathway learning the co-occurrence by rate-Hebbian coincidence."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="hub", n_neurons=n_hub * n_per, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="target", n_neurons=n_target * n_per, exc_fraction=1.0, internal_density=0.0),
    ]
    cfg.region_pathways = [RegionPathway(from_region="hub", to_region="target", density=1.0,
                                         weight_mean=0.05, weight_jitter=0.0, plastic=True)]
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    cfg.enable_stdp = False                 # STDP lands at delta_t~0 for symmetric co-occurrence (CYCLE 95)
    cfg.enable_hebbian_learning = True      # rate-Hebbian coincidence == the (soft-bounded) co-occurrence count
    cfg.hebbian_learning_rate = 0.03
    cfg.hebbian_max_weight = 5.0
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_weight_decay = 0.00001
    rt = RuntimeState(); rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt,
                              gpu_config=GPUConfig())
    bridge._initialize_simulation_data()
    hub_region = np.asarray(bridge.region_manager.indices("hub"))
    tgt_region = np.asarray(bridge.region_manager.indices("target"))
    return bridge, hub_region, tgt_region


def run_seed(seed, stories, vocab, cat_ids, a):
    rng = np.random.RandomState(seed)
    targets = list(vocab)
    target_set = set(targets)
    Nt = len(targets)
    n_hub, n_per = a.n_hub, a.n_per
    S_true = (np.asarray(cat_ids)[:, None] == np.asarray(cat_ids)[None, :]).astype(np.float64)
    # Hubs = top-N frequent context words (a stream frequency statistic; a brain knows its common words).
    gfreq = Counter()
    for toks in stories:
        gfreq.update(toks)
    hubs = [w for w, _ in gfreq.most_common() if w not in STOPLIST and w not in target_set][:n_hub]
    hub_idx = {w: i for i, w in enumerate(hubs)}
    tgt_row = {w: i for i, w in enumerate(targets)}
    keep = target_set | set(hubs)

    # Host reference: the log-double-centre of the BATCH co-occurrence counts (the read-out ceiling).
    from research.runners.learned_graded_cortex_fair_test import build_real_corpus
    C, _, _ = build_real_corpus(seed, n_hub)
    host_ref = _pearson_vs_Strue(_cos_sim(double_center(np.log1p(C * 100.0))), S_true)

    bridge, hub_region, tgt_region = build_stream_bridge(Nt, n_hub, n_per, seed)
    xp = bridge._cp if hasattr(bridge, "_cp") else None
    n_hub_neurons, n_tgt_neurons = n_hub * n_per, Nt * n_per

    def present_window(tgt_ids, hub_ids):
        """Co-activate the populations of the target-words and context-hub-words present in this window, for
        a few steps -> the bridge's Hebbian synapses strengthen hub->target for every (hub, target) co-occurring
        pair (pre at t-1 AND post at t). NO precomputed co-occurrence in the drive -- only who co-occurs here."""
        hub_full = np.zeros(n_hub_neurons, np.float32)
        tgt_full = np.zeros(n_tgt_neurons, np.float32)
        for h in hub_ids:
            hub_full[h * n_per:(h + 1) * n_per] = a.hub_scale
        for t in tgt_ids:
            tgt_full[t * n_per:(t + 1) * n_per] = a.tgt_scale
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[hub_region] = xp.asarray(hub_full) if xp is not None else hub_full
        bridge.cp_external_input_current[tgt_region] = xp.asarray(tgt_full) if xp is not None else tgt_full
        for _ in range(a.window_steps):
            bridge._run_one_simulation_step()

    # STREAM: hear the stories in a seeded order; slide a working-memory window; co-activate each window's
    # target + context-hub populations. Stop after the window budget (enough co-activation to learn M).
    story_order = rng.permutation(len(stories))
    n_windows = 0
    t_stream = time.time()
    for si in story_order:
        if n_windows >= a.max_windows:
            break
        kept = [t for t in stories[si] if t in keep]
        for c in range(len(kept)):
            lo, hi = max(0, c - WINDOW), min(len(kept), c + WINDOW + 1)
            win = kept[lo:hi]
            tgt_ids = [tgt_row[w] for w in win if w in target_set]
            hub_ids = [hub_idx[w] for w in win if w in hub_idx]
            if tgt_ids and hub_ids:
                present_window(tgt_ids, hub_ids)
                n_windows += 1
                if n_windows >= a.max_windows:
                    break
    bridge.cp_external_input_current[:] = 0.0

    # READ: population block-mean of the learned hub->target weights -> M[target, hub], then log-double-centre.
    W = np.asarray(to_host(bridge.cp_connections.todense())).astype(np.float64)
    blk = W[np.ix_(hub_region, tgt_region)].reshape(n_hub, n_per, Nt, n_per).mean(axis=(1, 3))
    M = blk.T                                                  # (Nt, n_hub) stream-learned co-occurrence
    code = double_center(np.log1p(M * 100.0))
    p = _pearson_vs_Strue(_cos_sim(code), S_true)
    gen, ch = heldout_generalization(code, np.asarray(cat_ids))
    rng2 = np.random.RandomState(seed * 99 + 1); perm = rng2.permutation(np.asarray(cat_ids))
    Sp = (perm[:, None] == perm[None, :]).astype(np.float64)
    perm_p = _pearson_vs_Strue(_cos_sim(code), Sp)
    mc = float(np.corrcoef(M.flatten(), C.flatten())[0, 1]) if M.std() > 0 else 0.0
    print(f"\n[on-bridge stream seed {seed}] {Nt}t x {n_hub}h | n_per={n_per} | {n_windows} stream windows "
          f"({time.time()-t_stream:.0f}s) | host-ref {host_ref:+.3f}", flush=True)
    print(f"  on-bridge stream cortex: corr(M,C) {mc:+.3f} | normalized code {p:+.3f} "
          f"({p/max(host_ref,1e-9):.0%} of host-ref) (gen {gen:.2f}/ch {ch:.2f}) | permuted {perm_p:+.3f}",
          flush=True)
    return {"seed": seed, "host_ref": host_ref, "code": p, "corr_MC": mc, "gen": gen, "permuted": perm_p,
            "n_windows": n_windows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42")
    p.add_argument("--n-hub", type=int, default=300)
    p.add_argument("--n-per", type=int, default=16, help="neurons per concept (population code)")
    p.add_argument("--window-steps", type=int, default=2, help="bridge steps per stream window")
    p.add_argument("--max-windows", type=int, default=20000, help="stream-window budget (caps wall-clock)")
    p.add_argument("--hub-scale", type=float, default=250.0)
    p.add_argument("--tgt-scale", type=float, default=1200.0)
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[on-bridge stream cortex CAPSTONE] seeds={seeds} n_per={a.n_per} max_windows={a.max_windows} -- "
          f"does the SPIKING BRIDGE, hearing the corpus stream window-by-window (population Hebbian "
          f"co-occurrence, NO precomputed counts in the drive), learn the cortex that reaches the target?",
          flush=True)
    vocab, cat_ids, _ = taxonomy_to_vocab_categories(TAXONOMY_8x8)
    stories = load_token_stream()
    print(f"  loaded {len(stories)} stories; vocab {len(vocab)} targets", flush=True)
    rows = [run_seed(s, stories, vocab, cat_ids, a) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    ref, code_p, mc, gen, perm = m("host_ref"), m("code"), m("corr_MC"), m("gen"), m("permuted")
    print(f"\n{'='*98}\n  MEAN ({len(seeds)} seeds): host-ref {ref:+.3f} | on-bridge stream code {code_p:+.3f} "
          f"({code_p/max(ref,1e-9):.0%} of host-ref) | corr(M,C) {mc:+.3f} | gen {gen:.2f} | permuted {perm:+.3f}",
          flush=True)
    print(f"{'='*98}", flush=True)
    if code_p >= 0.60 * ref and gen > 0.40 and abs(perm) <= 0.10:
        print(f"  GO (capstone): the SPIKING BRIDGE learns the biology-faithful stream cortex -- hearing the "
              f"corpus window-by-window, its population Hebbian synapses accumulate the co-occurrence (corr(M,C) "
              f"{mc:+.3f}) and the read-out reaches {code_p:+.3f} ({code_p/ref:.0%} of host-ref), generalizes "
              f"{gen:.2f}, permuted-clean. ==> the online stream cortex (CYCLE 94 numpy) is REALIZED on the real "
              f"substrate: representation (population) + learning (Hebbian co-occurrence) + normalization "
              f"(log-double-centre), NO preprocessing, learns from the conversation stream.", flush=True)
    elif code_p >= 0.30 * ref:
        print(f"  PARTIAL: the on-bridge stream cortex reaches {code_p:+.3f} ({code_p/ref:.0%} of host-ref, "
              f"corr(M,C) {mc:+.3f}) -- raise n_per (population fidelity) or max_windows (more stream).", flush=True)
    else:
        print(f"  NEGATIVE/needs-tuning: on-bridge stream code {code_p:+.3f} (corr(M,C) {mc:+.3f}) -- inspect "
              f"the window co-activation / population read / stream budget.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"host_ref": ref, "code": code_p, "corr_MC": mc, "gen": gen, "permuted": perm, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_onbridge_stream_cortex.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
