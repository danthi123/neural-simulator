"""Direction P-v2: hippocampal-only engram + SWR cycle test.

v1 used cortical region_filter so silencing CA3 was vacuous.
v2 uses region_filter=ca3/ca1 so the tag is hippocampal; properly
tests if SWR transfers hippocampal associations to cortex.

Recipe:
1. Load cached HIPPO substrate
2. Encode K=8 pairs with hippocampal region_filter
3. Pre-sleep A (hippo active): baseline retrieval
4. Pre-sleep B (hippo silenced): should fail (validates design)
5. SWR cycle (200 events)
6. Post-sleep (hippo silenced): tests CLS transfer

Pre-registered PASS: A>=0.50 AND B<0.30 AND (post-B)>=0.30.
~30 min wall.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.runners.concept_compose_train import _WORD_TO_IDX
from research.runners.compose_concept_engram import (
    encode_concept_pair, _ALL_CONCEPTS,
)
from research.findings.raw.mode_unification_with_hippo_probe import (
    _build_bridge_with_hippo,
)
from research.findings.raw.direction_P_multitag_sleep_consolidation import (
    test_retrieval, silence_hippocampus, parse_pairs,
    N_LANG_INPUT, N_WORDS_FOR_ORTHOGONAL, ENCODING_STEPS,
    BALANCED_TEACHER_PA, TOP_K_ENGRAM, DRIVE_PA, SPARSITY,
    CONCEPT_PAIRS_STR,
    N_SWR_EVENTS, BURST_DURATION_MS, INTER_BURST_MS, SWR_DRIVE_PA,
)
from research.runners.consolidation_trainer import run_swr_replay_phase
from sim.backend import get_backend, is_gpu_backend


HIPPO_CACHE_DIR = os.path.join(_HERE, "direction_G_hippo_theta_gamma_cache")
OUT_JSON = os.path.join(_HERE, "direction_P_v2_hippo_only_engram.json")
SEEDS = [42, 43, 44]


def run_seed(seed):
    print(f"\n--- seed {seed} (v2 hippo-only tag) ---", flush=True)
    cache_p = os.path.join(HIPPO_CACHE_DIR, f"bridge_full_seed{seed}.simstate.h5")
    if not os.path.exists(cache_p):
        print(f"  cache missing", flush=True); return None
    cp, _ = get_backend()
    bridge = _build_bridge_with_hippo(
        seed=seed, enable_adjective=True, n_lang_input=N_LANG_INPUT,
        n_per_pool=200, n_fs_per_pool=24, verbose=False)
    bridge.load_checkpoint(cache_p)
    for g in ("language_input_to_motor", "language_input_to_noun_pool",
              "language_input_to_verb_pool", "language_input_to_adjective_pool",
              "motor_to_language_output", "noun_pool_to_language_output",
              "verb_pool_to_language_output", "adjective_pool_to_language_output"):
        try: bridge.set_plasticity_gate(g, 0.0)
        except Exception: pass
    rm = bridge.region_manager
    hippo_rf = []
    for r in ["ca3", "ca1", "dg"]:
        try: rm.indices(r); hippo_rf.append(r)
        except Exception: pass
    print(f"  hippo region_filter: {hippo_rf}", flush=True)
    if not hippo_rf: return None
    pairs = parse_pairs(CONCEPT_PAIRS_STR)
    valid = [w for w in _ALL_CONCEPTS if _WORD_TO_IDX[w] < N_WORDS_FOR_ORTHOGONAL]
    print(f"  encoding {len(pairs)} pairs (hippo-only tag)", flush=True)
    t_e = time.time()
    for a, b in pairs:
        encode_concept_pair(
            bridge, a, b, f"{a}_{b}", encoding_steps=ENCODING_STEPS,
            drive_pA=DRIVE_PA, sparsity=SPARSITY, n_lang_input=N_LANG_INPUT,
            n_words_for_orthogonal=N_WORDS_FOR_ORTHOGONAL,
            region_filter=hippo_rf, top_k=TOP_K_ENGRAM,
            balanced_teacher_pA=BALANCED_TEACHER_PA, verbose=False)
    print(f"  encoded {(time.time()-t_e)/60:.1f} min", flush=True)
    print(f"  PRE-A (hippo active)", flush=True)
    pre_a = test_retrieval(bridge, pairs, hippo_rf, valid)
    print(f"    {pre_a['n_full']}/{pre_a['n_total']} = {pre_a['full_pass_rate']:.3f}", flush=True)
    print(f"  PRE-B (hippo silenced)", flush=True)
    silence_hippocampus(bridge)
    pre_b = test_retrieval(bridge, pairs, hippo_rf, valid)
    print(f"    {pre_b['n_full']}/{pre_b['n_total']} = {pre_b['full_pass_rate']:.3f}", flush=True)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    print(f"  SWR cycle", flush=True)
    t_s = time.time()
    run_swr_replay_phase(bridge, n_swr_events=N_SWR_EVENTS,
        burst_duration_ms=BURST_DURATION_MS, inter_burst_ms=INTER_BURST_MS,
        swr_drive_pA=SWR_DRIVE_PA, rng=np.random.default_rng(seed * 7))
    print(f"  sleep {(time.time()-t_s)/60:.1f} min", flush=True)
    print(f"  POST (hippo silenced)", flush=True)
    silence_hippocampus(bridge)
    post = test_retrieval(bridge, pairs, hippo_rf, valid)
    print(f"    {post['n_full']}/{post['n_total']} = {post['full_pass_rate']:.3f}", flush=True)
    return {"seed": seed, "pre_a": pre_a, "pre_b": pre_b, "post": post}


def main():
    xp, name = get_backend(); gpu = is_gpu_backend()
    print(f"=== Direction P-v2 hippo-only tag (cupy GPU={gpu}) ===", flush=True)
    t0 = time.time(); seed_results = []
    for s in SEEDS:
        r = run_seed(s)
        if r is not None: seed_results.append(r)
    total = (time.time() - t0) / 60
    if not seed_results: print("[FATAL]", flush=True); return 1
    pa = [r["pre_a"]["full_pass_rate"] for r in seed_results]
    pb = [r["pre_b"]["full_pass_rate"] for r in seed_results]
    po = [r["post"]["full_pass_rate"] for r in seed_results]
    pa_m, pb_m, po_m = float(np.mean(pa)), float(np.mean(pb)), float(np.mean(po))
    print(f"\n=== MULTI-SEED ===", flush=True)
    print(f"  pre-A: {pa_m:.3f} per-seed {pa}", flush=True)
    print(f"  pre-B: {pb_m:.3f} per-seed {pb}", flush=True)
    print(f"  post:  {po_m:.3f} per-seed {po}", flush=True)
    print(f"  gain (post-preB): {po_m-pb_m:+.3f}", flush=True)
    print(f"  Wall: {total:.1f} min", flush=True)
    if pa_m >= 0.50 and pb_m < 0.30 and (po_m - pb_m) >= 0.30:
        v = "CLS_VALIDATED_PILLAR_N105_CANDIDATE"
    elif pa_m < 0.50:
        v = "HIPPO_ENCODING_INSUFFICIENT"
    elif pb_m >= 0.30:
        v = "HIPPO_SILENCE_INEFFECTIVE"
    elif (po_m - pb_m) < 0.30:
        v = "SWR_DOES_NOT_TRANSFER_CONCEPT_ASSOC"
    else:
        v = "PARTIAL"
    print(f"  verdict: {v}", flush=True)
    out = {"backend": name, "gpu": gpu, "seeds": SEEDS,
        "pre_a_mean": pa_m, "pre_b_mean": pb_m, "post_mean": po_m,
        "pre_a_per_seed": pa, "pre_b_per_seed": pb, "post_per_seed": po,
        "per_seed": seed_results, "verdict": v, "wall_minutes": total}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
