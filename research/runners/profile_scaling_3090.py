"""Profiling pass (owner-requested 2026-06-17): VRAM + wall-clock (real-time conversation latency, training-step
throughput) vs scale on the local RTX 3090 (24 GB), to decide local-only vs cloud for a small-LLM-scale
biology-grounded conversational agent.

Measures, with real production classes (GPU):
  (A) CONVERSATION LATENCY -- RFPhasorComposer store + who/what query, vs composer dimension D and knowledge-base
      size. This is the real-time per-turn cost. The query iterates the KB, so latency ~ KB_size x per-op, which is
      the dominant scaling concern for a large fact store.
  (B) PARSER turn cost -- BridgeParser.parse (a small fixed bridge op) for the per-turn comprehension component.
  (C) BRAIN-BASED CORTEX VRAM -- a SimulationBridge sized to a representative concept-cortex (neurons + synapses),
      vs neuron count, to get the hot-vocab VRAM ceiling for the FULLY-brain-based path (concept assemblies
      resident as real neurons; the current composer's numpy codes are VRAM-light, so VRAM only caps the
      fully-neural variant).
  (D) BRIDGE STEP THROUGHPUT -- steps/sec at a representative size, the unit of training wall-clock.

Run:  SIM_BACKEND=cupy python -u -m research.runners.profile_scaling_3090
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "cupy")


def gpu_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL)
        return float(out.decode().strip().split("\n")[0])
    except Exception:
        return float("nan")


def _free_gpu():
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


# ----------------------------------------------------------------------------- (A) conversation latency
def profile_single_ops(D=512, n=30):
    """Per-op breakdown: isolate the cost of ONE bind and ONE unbind (each = one 208-step resonate + a CPU sync),
    to localize whether the conversation latency is orchestration/launch-bound (fixable) vs compute-bound."""
    from research.runners.rf_phasor_composer import RFPhasorComposer
    comp = RFPhasorComposer(seed=42, D=D, vocab=[f"c{i}" for i in range(64)])
    comp._bind(comp.roles["agent"], comp.concepts["c0"])     # warm (compile kernels)
    comp._unbind_phases(comp._bind(comp.roles["agent"], comp.concepts["c0"]), "agent")
    t = time.time()
    for _ in range(n):
        comp._bind(comp.roles["agent"], comp.concepts["c0"])
    ms_bind = (time.time() - t) / n * 1000.0
    bound = comp._bind(comp.roles["agent"], comp.concepts["c0"])
    t = time.time()
    for _ in range(n):
        comp._unbind_phases(bound, "agent")
    ms_unbind = (time.time() - t) / n * 1000.0
    return {"D": D, "ms_per_bind_op": ms_bind, "ms_per_unbind_op": ms_unbind, "resonate_steps": comp.period + 8}


def profile_conversation(D, n_facts, n_query=8):
    from research.runners.rf_phasor_composer import RFPhasorComposer
    vocab = [f"c{i}" for i in range(max(64, n_facts * 3 + 16))]
    comp = RFPhasorComposer(seed=42, D=D, vocab=vocab)
    facts = [(vocab[3 * i], vocab[3 * i + 1], vocab[3 * i + 2]) for i in range(n_facts)]
    t = time.time()
    for a, ac, p in facts:
        comp.store(a, ac, p)
    t_store = (time.time() - t) / max(n_facts, 1) * 1000.0
    # warm one query (bridge cache), then time
    comp.query_patient(facts[0][0], facts[0][1])
    t = time.time()
    for i in range(n_query):
        a, ac, p = facts[i % len(facts)]
        comp.query_patient(a, ac)
    t_query = (time.time() - t) / n_query * 1000.0
    return {"D": D, "n_facts": n_facts, "ms_per_store": t_store, "ms_per_query": t_query}


# ----------------------------------------------------------------------------- (B) parser turn cost
def profile_parser():
    from research.runners.brain_conversational_agent import BridgeParser
    p = BridgeParser(seed=42)
    p.parse(["dog", "go", "north"])            # warm
    t = time.time()
    for _ in range(50):
        p.parse(["dog", "go", "north"])
    return (time.time() - t) / 50 * 1000.0


# ----------------------------------------------------------------------------- (C) brain-based cortex VRAM + (D) step throughput
def build_sized_bridge(n_neurons, density=0.02, seed=42):
    """A representative cortical bridge: n_neurons Izhikevich, sparse recurrent (density) -- the unit a concept
    cortex is built from. VRAM here is dominated by the synapse CSR (~ n_neurons^2 x density)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    cfg = CoreSimConfig()
    cfg.num_neurons = int(n_neurons)
    cfg.neuron_model_type = "IZHIKEVICH"
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.connections_per_neuron = max(1, int(n_neurons * density))
    cfg.num_traits = 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation"):
        if hasattr(cfg, f):
            setattr(cfg, f, False)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data()
    return b


def profile_cortex_vram_and_throughput(sizes):
    rows = []
    for n in sizes:
        _free_gpu()
        base = gpu_used_mb()
        t0 = time.time()
        b = build_sized_bridge(n)
        build_s = time.time() - t0
        used = gpu_used_mb() - base
        n_syn = int(getattr(b, "cp_connections", np.zeros(0)).shape[0]) if hasattr(b, "cp_connections") else -1
        # step throughput
        for _ in range(3):
            b._run_one_simulation_step()
        t = time.time()
        n_steps = 40
        for _ in range(n_steps):
            b._run_one_simulation_step()
        steps_per_s = n_steps / (time.time() - t)
        rows.append({"n_neurons": n, "vram_mb": round(used, 1), "build_s": round(build_s, 2),
                     "steps_per_s": round(steps_per_s, 1), "n_synapses": n_syn})
        print(f"  [cortex n={n:>6}] VRAM {used:7.1f} MB | build {build_s:5.2f}s | {steps_per_s:7.1f} steps/s",
              flush=True)
        del b
        _free_gpu()
    return rows


def main():
    t_all = time.time()
    print(f"[3090 scaling profile] GPU baseline {gpu_used_mb():.0f} MB used (of 24576).\n", flush=True)

    # (C)+(D) cortex VRAM + step throughput FIRST (the explicit VRAM question; trimmed sizes so it always lands)
    print("  BRAIN-BASED CORTEX VRAM + STEP THROUGHPUT (Izhikevich, 2% recurrent):", flush=True)
    cortex = profile_cortex_vram_and_throughput([2000, 8000, 20000])

    # op breakdown (localize the latency bottleneck)
    ops = profile_single_ops()
    print(f"\n  SINGLE-OP COST (D={ops['D']}, {ops['resonate_steps']} resonate steps each): "
          f"bind {ops['ms_per_bind_op']:.2f} ms | unbind {ops['ms_per_unbind_op']:.2f} ms "
          f"(each = sequential GPU step launches + a CPU sync)\n", flush=True)
    _free_gpu()

    # (B) parser
    ms_parse = profile_parser()
    print(f"  PARSER: {ms_parse:.2f} ms/parse (fixed per-turn comprehension cost)\n", flush=True)

    # (A) conversation latency: D sweep at KB=50, then KB sweep at D=512 (trimmed to land in budget)
    print("  CONVERSATION LATENCY (RFPhasorComposer store + who/what query):", flush=True)
    conv = []
    for D in (128, 512, 1024, 2048):
        r = profile_conversation(D, n_facts=50)
        conv.append(r)
        print(f"    D={D:>4} KB=50 : store {r['ms_per_store']:6.2f} ms | query {r['ms_per_query']:7.2f} ms/turn",
              flush=True)
        _free_gpu()
    conv_kb = []
    for kb in (10, 50, 100, 200):
        r = profile_conversation(512, n_facts=kb)
        conv_kb.append(r)
        print(f"    D=512  KB={kb:>4} : store {r['ms_per_store']:6.2f} ms | query {r['ms_per_query']:7.2f} ms/turn "
              f"(query iterates the KB)", flush=True)
        _free_gpu()

    out = {"gpu": "RTX 3090 (24576 MB)", "ms_per_parse": ms_parse, "single_ops": ops,
           "conversation_D_sweep": conv, "conversation_KB_sweep": conv_kb, "cortex": cortex,
           "elapsed_s": round(time.time() - t_all, 1)}
    path = os.path.join(_REPO, "research", "findings", "raw", "_profile_scaling_3090.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n  [saved] {path}  (total {out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
