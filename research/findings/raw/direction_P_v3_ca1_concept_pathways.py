"""Direction P-v3: CLS architectural fix via extra ca1 -> concept_pool pathways.

Wraps build_biological_brain_regions to APPEND 12 plastic pathways from
CA1 to each of the 16 concept pools (noun + verb + adjective). Then trains
fresh substrate, encodes hippocampal-only multitag tags via
region_filter=[ca3, ca1, dg], runs SWR consolidation through the new
pathways, tests cortex-only retrieval after hippocampus silencing.

P-v2 found HIPPO_ENCODING_INSUFFICIENT (pre-A 0.167; threshold 0.50):
hippocampal engram tag alone is insufficient for retrieval because the
only existing CA1 -> cortex pathway is ca1_to_motor (designed for word
-> motor binding in Phase 1.3). P-v3 ADDS the missing CA1 -> concept_pool
pathways that would carry concept-concept consolidation per CLS theory.

DISCIPLINE: build_biological_brain_regions is protected/byte-unchanged
in this file. Only its pathways list is APPENDED locally before bridge
construction. No protected/frozen/moat modification.

PRE-REGISTERED PASS: pre-A >= 0.50 AND pre-B < 0.30 AND (post - pre-B) >= 0.30.
Bar UNCHANGED from P-v2.
~3-5 hr wall total (substrate build + encode + SWR + tests per seed).
"""
from __future__ import annotations
import json
import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.findings.raw.direction_P_multitag_sleep_consolidation import (
    test_retrieval, silence_hippocampus, parse_pairs,
    N_LANG_INPUT, N_WORDS_FOR_ORTHOGONAL, ENCODING_STEPS,
    BALANCED_TEACHER_PA, TOP_K_ENGRAM, DRIVE_PA, SPARSITY,
    CONCEPT_PAIRS_STR,
    N_SWR_EVENTS, BURST_DURATION_MS, INTER_BURST_MS, SWR_DRIVE_PA,
)
from research.runners.concept_compose_train import _WORD_TO_IDX
from research.runners.compose_concept_engram import (
    encode_concept_pair, _ALL_CONCEPTS,
)
from research.runners.consolidation_trainer import run_swr_replay_phase
from research.runners.concept_pool_demo import (
    NOUN_NAMES, VERB_NAMES, ADJECTIVE_NAMES,
)
from research.runners.text_minimal_isolation import (
    build_biological_brain_regions,
)
from sim.regions import RegionPathway
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(_HERE, "direction_P_v3_ca1_concept_pathways.json")
SEEDS = [42, 43, 44]

N_PER_POOL = 200
N_FS_PER_POOL = 24
ENABLE_ADJECTIVE = True

CA1_TO_CONCEPT_DENSITY = 0.10
CA1_TO_CONCEPT_WEIGHT_MEAN = 2.0
CA1_TO_CONCEPT_WEIGHT_JITTER = 0.3
CA1_TO_CONCEPT_GATE = "ca1_to_concept_pool"


def build_bridge_with_ca1_concept_pathways(seed, verbose=False):
    """Same HIPPO concept-pool bridge as _build_bridge_with_hippo BUT
    APPENDS extra plastic pathways CA1 -> each concept_pool before bridge
    construction. build_biological_brain_regions itself is not modified.
    """
    from sim.config import (CoreSimConfig, VisualizationConfig,
                              RuntimeState, GPUConfig)
    from sim.bridge import SimulationBridge

    concept_internal_density = 0.05
    concept_exc_weight = 0.3
    concept_inh_weight = 0.8
    motor_internal_density = 0.10
    motor_exc_weight = 2.0
    motor_inh_weight = 4.0

    regions, pathways = build_biological_brain_regions(
        n_lang_input=N_LANG_INPUT,
        n_motor_per_action=N_PER_POOL,
        motor_internal_density=motor_internal_density,
        motor_exc_weight_mean=motor_exc_weight,
        motor_inh_weight_mean=motor_inh_weight,
        text_input_to_motor_density=0.30,
        text_input_to_motor_weight=3.0,
        text_input_to_motor_jitter=0.5,
        enable_motor_fs=True,
        n_motor_fs_per_action=N_FS_PER_POOL,
        enable_language_output=True,
        n_lang_output=N_LANG_INPUT,
        motor_to_language_output_weight=2.0,
        enable_noun_pools=True,
        noun_pool_names=NOUN_NAMES,
        n_noun_per_pool=N_PER_POOL,
        n_noun_fs_per_pool=N_FS_PER_POOL,
        enable_verb_pools=True,
        verb_pool_names=VERB_NAMES,
        n_verb_per_pool=N_PER_POOL,
        n_verb_fs_per_pool=N_FS_PER_POOL,
        enable_adjective_pools=ENABLE_ADJECTIVE,
        adjective_pool_names=ADJECTIVE_NAMES if ENABLE_ADJECTIVE else None,
        n_adjective_per_pool=N_PER_POOL,
        n_adjective_fs_per_pool=N_FS_PER_POOL,
        concept_pool_internal_density=concept_internal_density,
        concept_pool_exc_weight_mean=concept_exc_weight,
        concept_pool_inh_weight_mean=concept_inh_weight,
        enable_hippocampus_consolidation=True,
    )

    extra_pathways = []
    n_extra = 0
    for pool_kind, names in [
        ("noun_pool", NOUN_NAMES),
        ("verb_pool", VERB_NAMES),
        ("adjective_pool", ADJECTIVE_NAMES if ENABLE_ADJECTIVE else []),
    ]:
        for n in names:
            target_region = pool_kind + "_" + n
            extra_pathways.append(RegionPathway(
                from_region="ca1",
                to_region=target_region,
                density=CA1_TO_CONCEPT_DENSITY,
                weight_mean=CA1_TO_CONCEPT_WEIGHT_MEAN,
                weight_jitter=CA1_TO_CONCEPT_WEIGHT_JITTER,
                plastic=True,
                plasticity_gate=CA1_TO_CONCEPT_GATE,
            ))
            n_extra += 1

    pathways = list(pathways) + extra_pathways
    if verbose:
        print("  ADDED " + str(n_extra)
              + " extra ca1 -> concept_pool pathways (density "
              + str(CA1_TO_CONCEPT_DENSITY)
              + ", weight " + str(CA1_TO_CONCEPT_WEIGHT_MEAN)
              + ", gate " + CA1_TO_CONCEPT_GATE + ")", flush=True)

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.nmda_tau_decay = 100.0
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = 8.0
    cfg.fast_spike_reset = True

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    if verbose:
        n_concept_pools = (4 + 4 + (4 if ENABLE_ADJECTIVE else 0))
        total_neurons = sum(r.n_neurons for r in regions)
        print("[BUILD] P-v3 hippo-enabled concept-pool bridge: "
              + str(total_neurons) + " neurons total, "
              + str(n_concept_pools) + " concept pools "
              + "(" + str(n_concept_pools * N_PER_POOL) + " pool neurons); "
              + "hippocampus + " + str(n_extra)
              + " new ca1->concept pathways", flush=True)
    return bridge


def run_seed(seed):
    print("\n--- seed " + str(seed) + " (P-v3 ca1->concept) ---", flush=True)
    t_build = time.time()
    bridge = build_bridge_with_ca1_concept_pathways(seed=seed, verbose=True)
    build_min = (time.time() - t_build) / 60
    print("  fresh substrate built " + str(round(build_min, 2)) + " min", flush=True)

    for g in ("language_input_to_motor", "language_input_to_noun_pool",
              "language_input_to_verb_pool", "language_input_to_adjective_pool",
              "motor_to_language_output", "noun_pool_to_language_output",
              "verb_pool_to_language_output", "adjective_pool_to_language_output"):
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass

    rm = bridge.region_manager
    hippo_rf = []
    for r in ["ca3", "ca1", "dg"]:
        try:
            rm.indices(r)
            hippo_rf.append(r)
        except Exception:
            pass
    print("  hippo region_filter: " + str(hippo_rf), flush=True)
    if not hippo_rf:
        return None

    pairs = parse_pairs(CONCEPT_PAIRS_STR)
    valid = [w for w in _ALL_CONCEPTS if _WORD_TO_IDX[w] < N_WORDS_FOR_ORTHOGONAL]
    print("  encoding " + str(len(pairs)) + " pairs", flush=True)
    t_e = time.time()
    for a, b in pairs:
        encode_concept_pair(
            bridge, a, b, a + "_" + b,
            encoding_steps=ENCODING_STEPS,
            drive_pA=DRIVE_PA, sparsity=SPARSITY,
            n_lang_input=N_LANG_INPUT,
            n_words_for_orthogonal=N_WORDS_FOR_ORTHOGONAL,
            region_filter=hippo_rf, top_k=TOP_K_ENGRAM,
            balanced_teacher_pA=BALANCED_TEACHER_PA, verbose=False)
    enc_min = (time.time() - t_e) / 60
    print("  encoded " + str(round(enc_min, 2)) + " min", flush=True)

    print("  PRE-A (hippo active)", flush=True)
    pre_a = test_retrieval(bridge, pairs, hippo_rf, valid)
    print("    " + str(pre_a["n_full"]) + "/" + str(pre_a["n_total"])
          + " = " + str(round(pre_a["full_pass_rate"], 3)), flush=True)

    print("  PRE-B (hippo silenced)", flush=True)
    silence_hippocampus(bridge)
    pre_b = test_retrieval(bridge, pairs, hippo_rf, valid)
    print("    " + str(pre_b["n_full"]) + "/" + str(pre_b["n_total"])
          + " = " + str(round(pre_b["full_pass_rate"], 3)), flush=True)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    print("  SWR cycle (opening ca1->concept gate)", flush=True)
    try:
        bridge.set_plasticity_gate(CA1_TO_CONCEPT_GATE, 1.0)
    except Exception as e:
        print("  [WARN] gate open failed: " + str(e), flush=True)
    t_s = time.time()
    run_swr_replay_phase(bridge, n_swr_events=N_SWR_EVENTS,
        burst_duration_ms=BURST_DURATION_MS,
        inter_burst_ms=INTER_BURST_MS,
        swr_drive_pA=SWR_DRIVE_PA,
        rng=np.random.default_rng(seed * 7))
    sleep_min = (time.time() - t_s) / 60
    print("  sleep " + str(round(sleep_min, 2)) + " min", flush=True)

    print("  POST (hippo silenced, gate frozen)", flush=True)
    silence_hippocampus(bridge)
    try:
        bridge.set_plasticity_gate(CA1_TO_CONCEPT_GATE, 0.0)
    except Exception:
        pass
    post = test_retrieval(bridge, pairs, hippo_rf, valid)
    print("    " + str(post["n_full"]) + "/" + str(post["n_total"])
          + " = " + str(round(post["full_pass_rate"], 3)), flush=True)

    return {
        "seed": seed,
        "pre_a": pre_a,
        "pre_b": pre_b,
        "post": post,
        "build_minutes": build_min,
        "encode_minutes": enc_min,
        "sleep_minutes": sleep_min,
    }


def main():
    xp, name = get_backend()
    gpu = is_gpu_backend()
    print("=== Direction P-v3 ca1->concept pathways (backend "
          + name + ", GPU=" + str(gpu) + ") ===", flush=True)
    t0 = time.time()
    seed_results = []
    for s in SEEDS:
        r = run_seed(s)
        if r is not None:
            seed_results.append(r)
    total = (time.time() - t0) / 60
    if not seed_results:
        print("[FATAL no seed succeeded]", flush=True)
        return 1
    pa = [r["pre_a"]["full_pass_rate"] for r in seed_results]
    pb = [r["pre_b"]["full_pass_rate"] for r in seed_results]
    po = [r["post"]["full_pass_rate"] for r in seed_results]
    pa_m = float(np.mean(pa))
    pb_m = float(np.mean(pb))
    po_m = float(np.mean(po))
    print("\n=== MULTI-SEED P-v3 ===", flush=True)
    print("  pre-A: " + str(round(pa_m, 3)) + " per-seed "
          + str([round(x, 3) for x in pa]), flush=True)
    print("  pre-B: " + str(round(pb_m, 3)) + " per-seed "
          + str([round(x, 3) for x in pb]), flush=True)
    print("  post:  " + str(round(po_m, 3)) + " per-seed "
          + str([round(x, 3) for x in po]), flush=True)
    gain = po_m - pb_m
    print("  gain: " + ("+" if gain >= 0 else "") + str(round(gain, 3)), flush=True)
    print("  Wall: " + str(round(total, 1)) + " min", flush=True)

    if pa_m >= 0.50 and pb_m < 0.30 and gain >= 0.30:
        v = "CLS_CONFIRMED_PILLAR_N105_CANDIDATE"
    elif pa_m < 0.50:
        v = "HIPPO_ENCODING_INSUFFICIENT_EVEN_WITH_NEW_PATHWAYS"
    elif pb_m >= 0.30:
        v = "HIPPO_SILENCE_INEFFECTIVE"
    elif gain < 0.30:
        v = "SWR_DOES_NOT_TRANSFER_CONCEPT_ASSOC_VIA_CA1_PATHWAYS"
    else:
        v = "PARTIAL"
    print("  verdict: " + v, flush=True)

    out = {
        "backend": name, "gpu": gpu, "seeds": SEEDS,
        "pre_a_mean": pa_m, "pre_b_mean": pb_m, "post_mean": po_m,
        "gain": gain,
        "pre_a_per_seed": pa, "pre_b_per_seed": pb, "post_per_seed": po,
        "per_seed": seed_results, "verdict": v, "wall_minutes": total,
        "n_extra_pathways": 12 if ENABLE_ADJECTIVE else 8,
        "gate_name": CA1_TO_CONCEPT_GATE,
        "ca1_to_concept_density": CA1_TO_CONCEPT_DENSITY,
        "ca1_to_concept_weight_mean": CA1_TO_CONCEPT_WEIGHT_MEAN,
        "ca1_to_concept_weight_jitter": CA1_TO_CONCEPT_WEIGHT_JITTER,
        "pre_registered_bar": {
            "pre_a_min": 0.50, "pre_b_max": 0.30, "gain_min": 0.30,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote " + OUT_JSON, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
