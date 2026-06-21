"""Latency diagnostic for the fully-brain-based one-brain conversation (`consolidated_320_conversation_demo
--composer onebrain` with the production flips on). The V=320 run is impractically slow (~2 h, killed); this
harness PRECISELY isolates WHICH component dominates, with hard per-component, per-scale numbers, so the
optimization that follows is targeted, not guessed. Probe/timing only -- NO sim/ edit, reuse-by-import.

THE SUSPECTS (separated here):
  (a) the OneBrainComposer build -- the V-scaling co-resident bridge itself (n_total ~ k_max * (n_main*V)).
  (b) the integrated_loop SEQUENCER fabric build -- the K-way gated-disinhibition match cascade
      (build_sequencerK_bridge: ~2V + 4*K*V regions x n_word; the 244,580-neuron bridge from the killed log).
      Built lazily on the FIRST who/what query, so (first-query - steady-query) isolates it.
  (c) the integrated_loop PER-QUERY cost -- the sequencer run per who/what query (steady state) vs the host _scan.
  (d) the enable_learned_assoc eager build (LearnedAssocGraph, fixed n_pool=1500 -- V-independent) + its
      per-elaborate cost (graph() reads O(V^2) dense weight slices).
  (e) the enable_spiking_cleanup per-query cost -- the spiking Izhikevich WTA vs the host argmax over the membrane.

METHOD: build the components at V in {64,128,320} with a factorial of flips, timing build-time per component and
per-op time (store / who-what query [first + steady] / elaborate) with time.perf_counter + sim.backend.synchronize()
around each phase. Reuse the demo's exact code paths (the production OneBrainComposer + BrainConversationalAgent).

Run (GPU is the real path; the small scales are fast):
  SIM_BACKEND=cupy python -m research.runners._onebrain_latency_diagnostic --scales 64 128 --out <json>
  SIM_BACKEND=cupy python -m research.runners._onebrain_latency_diagnostic --scales 320 --out <json>   # last
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import synchronize, get_backend, is_gpu_backend
from research.runners.stream_taxonomy_320 import TAXONOMY_40x8
from research.runners.option_c_real_cooccurrence_derisk import taxonomy_to_vocab_categories

D = 128  # the production RFPhasorComposer phasor dimension (the demo's value)

# The demo's facts (9 stored: 8 affirm + 1 negate) + cues. Cap to a fixed K so per-op cost is comparable across V.
FACTS = [("dog", "eat", "apple"), ("cat", "play", "ball"), ("bird", "sleep", "tree"),
         ("girl", "run", "park"), ("boy", "look", "book"), ("lion", "eat", "cake"),
         ("rabbit", "jump", "garden"), ("mouse", "walk", "house")]
NEG_FACT = ("fish", "eat", "cake")
ABSENT_WHAT = [("dog", "sing"), ("cat", "run"), ("bird", "eat")]


def _projection(d_out, n_in, seed):
    rng = np.random.RandomState(seed * 7919 + 13)
    return (rng.standard_normal((d_out, n_in)) + 1j * rng.standard_normal((d_out, n_in))).astype(np.complex128)


def _grounded_phases(code_vec, proj):
    z = proj @ code_vec.astype(np.complex128)
    return (np.angle(z) % (2.0 * np.pi)) / (2.0 * np.pi)


class Timer:
    """perf_counter around a code block with a device sync at entry+exit so GPU async work is included."""
    def __init__(self):
        self.ms = {}

    def __call__(self, label):
        self._label = label
        return self

    def __enter__(self):
        synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        synchronize()
        self.ms[self._label] = (time.perf_counter() - self._t0) * 1000.0
        return False


def _load_codes(seed, V_words):
    """Grounded phases for the requested word subset, from the cached 320 stream-learned (neural) codes. The
    grounding is a fixed projection (cheap); it only affects code geometry, not the timing structure we measure."""
    cpath = os.path.join(_REPO, "research", "findings", "raw", f"_phaseB_stream_codes_320_neural_seed{seed}.npy")
    full_vocab, _cat, _ = taxonomy_to_vocab_categories(TAXONOMY_40x8)
    codes = np.load(cpath)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    proj = _projection(D, codes.shape[1], seed)
    cmap = {full_vocab[i]: codes[i] for i in range(len(full_vocab))}
    concepts = {w: cmap[w] for w in V_words}
    grounded = {w: _grounded_phases(cmap[w], proj) for w in V_words}
    return concepts, grounded


def _vocab_subset(V):
    """A V-word vocabulary that INCLUDES every word the demo facts/cues use, padded from the 320 taxonomy. So the
    onebrain store/query exercise the real facts at every scale while the cleanup-codebook size (the V-scaling
    driver of n_total + the sequencer fabric) is exactly V."""
    full_vocab, _cat, _ = taxonomy_to_vocab_categories(TAXONOMY_40x8)
    used = sorted({w for f in FACTS for w in f} | {NEG_FACT[0], NEG_FACT[1], NEG_FACT[2]}
                  | {w for c in ABSENT_WHAT for w in c})
    assert all(w in full_vocab for w in used), [w for w in used if w not in full_vocab]
    rest = [w for w in full_vocab if w not in used]
    chosen = used + rest[:max(0, V - len(used))]
    return sorted(chosen[:V])


def time_config(seed, V, integrated_loop, spiking_cleanup, learned_assoc, n_query=3):
    """Build + exercise the onebrain stack for one (V, flag) config; return a dict of build-ms + per-op-ms.
    Separates the OneBrainComposer build (a/b-static) from the LearnedAssocGraph build (d), and the first who/what
    query (triggers the sequencer fabric build = b) from a steady query (c). spiking_cleanup toggles (e)."""
    from research.runners.one_brain_composer import OneBrainComposer
    from research.runners.brain_conversational_agent import BrainConversationalAgent

    vocab = _vocab_subset(V)
    concepts, grounded = _load_codes(seed, vocab)
    T = Timer()

    # (a/b-static) the OneBrainComposer co-resident bridge build (n_total ~ k_max*(n_main*V)); the sequencer fabric
    # is NOT built here (it is lazy on the first query). Build the composer explicitly so we can time it in isolation,
    # then hand it to the agent (composer= overrides composer_kind).
    with T("build_composer"):
        composer = OneBrainComposer(seed=seed, D=D, vocab=vocab, grounded_codes=grounded,
                                    enable_attributed=False, enable_multiframe=False,
                                    enable_spiking_cleanup=bool(spiking_cleanup),
                                    integrated_loop=bool(integrated_loop))
        synchronize()

    # (d) the rest of the agent: the LearnedAssocGraph eager build (fixed n_pool=1500, V-independent) is the
    # dominant cost here; the parser is carried by the composer. Time the whole agent-wrap so (d) is captured.
    with T("build_agent_rest"):
        agent = BrainConversationalAgent(seed=seed, concepts=concepts, composer=composer,
                                         grounded_codes=grounded, enable_neural_render=True,
                                         enable_learned_assoc=bool(learned_assoc),
                                         enable_attributed=False, enable_multiframe=False)
        synchronize()

    # store (hear) -- 8 affirm + 1 negate, the demo's TURN 1. Each hear() parses + binds + writes a block; the
    # integrated_loop marks the sequencer drives dirty per store (rebuilt on the next query).
    with T("store_9_facts"):
        for a, v, o in FACTS:
            agent.hear(f"{a} {v} {o}", polarity="AFFIRM")
        agent.hear(f"{NEG_FACT[0]} {NEG_FACT[1]} {NEG_FACT[2]}", polarity="NEGATE")
        synchronize()

    # who/what FIRST query -- triggers the lazy sequencer fabric build (b) + drives build, when integrated_loop.
    with T("query_first"):
        _ = agent.what_does(FACTS[0][0], FACTS[0][1])
        synchronize()

    # who/what STEADY query (c) -- averaged over n_query; the sequencer/store CSR caches are warm now.
    qt = []
    for qi in range(n_query):
        a, v, o = FACTS[qi % len(FACTS)]
        with T(f"query_steady_{qi}"):
            _ = agent.what_does(a, v)
            synchronize()
        qt.append(T.ms[f"query_steady_{qi}"])
    query_steady_ms = float(np.mean(qt))

    # abstain query (the moat path) -- one absent cue, same routing.
    with T("query_abstain"):
        _ = agent.what_does(*ABSENT_WHAT[0])
        synchronize()

    # elaborate (d, per-op) -- dialogue planning over the assoc graph (learned graph reads O(V^2) when learned_assoc).
    with T("elaborate"):
        _ = agent.elaborate(FACTS[0][0])
        synchronize()

    # sizes for the scaling story.
    n_total = int(composer.n_total)
    seq_n = None
    if integrated_loop and getattr(composer, "_seq", None) is not None:
        seq_n = int(composer._seq[0].core_config.num_neurons)
    assoc_n = None
    if learned_assoc and getattr(agent, "_learned_assoc", None) is not None:
        assoc_n = int(agent._learned_assoc.bridge.core_config.num_neurons)

    return {
        "seed": seed, "V": V, "integrated_loop": bool(integrated_loop),
        "spiking_cleanup": bool(spiking_cleanup), "learned_assoc": bool(learned_assoc),
        "n_total_composer": n_total, "n_sequencer": seq_n, "n_assoc": assoc_n,
        "build_composer_ms": round(T.ms["build_composer"], 1),
        "build_agent_rest_ms": round(T.ms["build_agent_rest"], 1),
        "store_9_facts_ms": round(T.ms["store_9_facts"], 1),
        "query_first_ms": round(T.ms["query_first"], 1),
        "query_steady_ms": round(query_steady_ms, 1),
        "query_abstain_ms": round(T.ms["query_abstain"], 1),
        "elaborate_ms": round(T.ms["elaborate"], 1),
        # the derived sequencer-fabric build cost = first-query minus a steady query (the lazy build only fires once).
        "seq_fabric_build_ms": round(T.ms["query_first"] - query_steady_ms, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[64, 128, 320])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-query", type=int, default=3)
    ap.add_argument("--configs", nargs="+",
                    default=["off", "iloop", "iloop_clean", "assoc", "all"],
                    help="off=all flags off (host baseline); iloop=+integrated_loop; iloop_clean=+spiking_cleanup; "
                         "assoc=+learned_assoc only; all=production (the killed run's config).")
    ap.add_argument("--out", default="research/findings/raw/_onebrain_latency_diagnostic.json")
    a = ap.parse_args()

    xp, name = get_backend()
    print(f"[onebrain-latency] backend={name} gpu={is_gpu_backend()} scales={a.scales} configs={a.configs}\n",
          flush=True)

    CONFIGS = {
        "off":         dict(integrated_loop=False, spiking_cleanup=False, learned_assoc=False),
        "iloop":       dict(integrated_loop=True,  spiking_cleanup=False, learned_assoc=False),
        "iloop_clean": dict(integrated_loop=True,  spiking_cleanup=True,  learned_assoc=False),
        "assoc":       dict(integrated_loop=False, spiking_cleanup=False, learned_assoc=True),
        "all":         dict(integrated_loop=True,  spiking_cleanup=True,  learned_assoc=True),
    }

    results = []
    for V in a.scales:
        for cfg_name in a.configs:
            flags = CONFIGS[cfg_name]
            t0 = time.perf_counter()
            r = time_config(a.seed, V, n_query=a.n_query, **flags)
            r["config"] = cfg_name
            r["wall_s"] = round(time.perf_counter() - t0, 1)
            results.append(r)
            print(f"  V={V:>3} {cfg_name:>11} | build comp {r['build_composer_ms']:>8.1f} + rest "
                  f"{r['build_agent_rest_ms']:>8.1f} ms | store {r['store_9_facts_ms']:>7.1f} | q1 "
                  f"{r['query_first_ms']:>8.1f} | q-steady {r['query_steady_ms']:>7.1f} | seq-build "
                  f"{r['seq_fabric_build_ms']:>8.1f} | elab {r['elaborate_ms']:>7.1f} | "
                  f"n_total {r['n_total_composer']} seq {r['n_sequencer']} | {r['wall_s']}s", flush=True)
            # incremental dump so a kill mid-V still leaves the completed scales on disk.
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            with open(a.out, "w") as fh:
                json.dump({"backend": name, "results": results}, fh, indent=2, default=str)

    print(f"\n[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
