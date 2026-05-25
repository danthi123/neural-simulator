"""Direction P: multitag chat + Phase 1.3 hippocampal SWR
consolidation -> "associations across simulated sleep".

The two validated capabilities being combined:
- Multitag chat at 91.7% multi-seed (pillar n=100/n=101; works on
  both OPTION 3 and HIPPO-OPTION3 substrates)
- Phase 1.3 hippocampal SWR consolidation at 3/3 strict anti-cheat
  multi-seed (CLS / McClelland 1995 / Buzsaki 2015)

Test:
1. Load cached HIPPO-OPTION3 trained substrate (Direction G cache)
2. Encode K=8 multitag concept-pairs (validated pillar n=101 recipe)
3. Test multitag retrieval BEFORE sleep
4. Run SWR consolidation cycle (~200 events; CA3 -> Schaffer -> CA1
   -> cortex)
5. SILENCE hippocampus (CA3 firing clamped to zero)
6. Test multitag retrieval AFTER sleep with hippo silenced
7. Multi-seed

Pre-registered success: post-sleep-with-hippo-silenced retrieval >=
70% of pre-sleep retrieval. Demonstrates CONSOLIDATION worked --
associations transferred to cortex + survive without hippocampus.

~30-45 min wall multi-seed (reuses Direction G cached bridges).
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

from research.runners.concept_pool_demo import (
    DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB, ADJECTIVE_VOCAB,
)
from research.runners.concept_compose_train import _WORD_TO_IDX
from research.runners.compose_concept_engram import (
    encode_concept_pair, lang_output_pattern_during_stim,
    cosine_to_word, _ALL_CONCEPTS,
)
from research.findings.raw.mode_unification_with_hippo_probe import (
    _build_bridge_with_hippo,
)
from research.runners.consolidation_trainer import (
    run_swr_replay_phase,
)
from sim.backend import get_backend, is_gpu_backend


HIPPO_CACHE_DIR = os.path.join(
    _HERE, "direction_G_hippo_theta_gamma_cache")
OUT_JSON = os.path.join(
    _HERE, "direction_P_multitag_sleep_consolidation.json")
SEEDS = [42, 43, 44]

N_LANG_INPUT = 2048
N_WORDS_FOR_ORTHOGONAL = 16
ENCODING_STEPS = 500
BALANCED_TEACHER_PA = 500.0
TOP_K_ENGRAM = 100
DRIVE_PA = 200.0
SPARSITY = 0.05

CONCEPT_PAIRS_STR = ("apple:big,dog:small,cat:hot,river:cold,"
                     "big:hot,small:cold,apple:cat,dog:river")

N_SWR_EVENTS = 200
BURST_DURATION_MS = 100
INTER_BURST_MS = 50
SWR_DRIVE_PA = 100.0


def parse_pairs(pairs_str):
    pairs = []
    for ps in pairs_str.split(","):
        a, b = ps.strip().split(":")
        if a in _WORD_TO_IDX and b in _WORD_TO_IDX:
            pairs.append((a, b))
    return pairs


def test_retrieval(bridge, pairs, region_filter, valid_concepts):
    """Stim each tag; check both concepts in lang_output top-5."""
    cp, _ = get_backend()
    encoded_tags = [f"{a}_{b}" for a, b in pairs]

    n_total = len(pairs)
    n_full = 0
    n_partial = 0
    per_pair = []
    for (a, b), tag in zip(pairs, encoded_tags):
        try:
            pattern, n_lang_out = lang_output_pattern_during_stim(
                bridge, tag, drive_pA=1500.0, stim_steps=100)
            scores = {}
            for w in valid_concepts:
                cos = cosine_to_word(
                    pattern, w, n_lang_out,
                    n_words_for_orthogonal=N_WORDS_FOR_ORTHOGONAL,
                    sparsity=SPARSITY)
                scores[w] = cos
            top5 = sorted(scores.items(), key=lambda x: x[1],
                            reverse=True)[:5]
            top5_words = [w for w, s in top5]
            a_in = a in top5_words
            b_in = b in top5_words
            both_in = a_in and b_in
            any_in = a_in or b_in
            if both_in: n_full += 1
            if any_in: n_partial += 1
            per_pair.append({
                "pair": f"{a}_{b}", "top5": top5_words,
                "a_in_top5": a_in, "b_in_top5": b_in,
                "full_pass": both_in,
            })
        except Exception as e:
            per_pair.append({"pair": f"{a}_{b}", "error": str(e)})
    return {
        "n_full": n_full, "n_partial": n_partial, "n_total": n_total,
        "full_pass_rate": n_full / n_total if n_total > 0 else 0,
        "partial_pass_rate": n_partial / n_total if n_total > 0 else 0,
        "per_pair": per_pair,
    }


def silence_hippocampus(bridge):
    """Clamp CA3 firing to zero with strong inhibitory current."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    try:
        ca3_idx = list(rm.indices("ca3"))
        ca3_arr = cp.asarray(ca3_idx, dtype=cp.int64)
        bridge.cp_external_input_current[ca3_arr] = -2000.0
        return len(ca3_idx)
    except Exception:
        return 0


def run_one_seed(seed):
    print(f"\n--- seed {seed} ---", flush=True)
    cache_p = os.path.join(
        HIPPO_CACHE_DIR, f"bridge_full_seed{seed}.simstate.h5")
    if not os.path.exists(cache_p):
        print(f"  [seed {seed}] HIPPO bridge cache missing", flush=True)
        return None

    cp, _ = get_backend()
    print(f"  [seed {seed}] loading HIPPO substrate", flush=True)
    bridge = _build_bridge_with_hippo(
        seed=seed, enable_adjective=True,
        n_lang_input=N_LANG_INPUT, n_per_pool=200,
        n_fs_per_pool=24, verbose=False,
    )
    bridge.load_checkpoint(cache_p)

    for g in ("language_input_to_motor",
              "language_input_to_noun_pool",
              "language_input_to_verb_pool",
              "language_input_to_adjective_pool",
              "motor_to_language_output",
              "noun_pool_to_language_output",
              "verb_pool_to_language_output",
              "adjective_pool_to_language_output"):
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass

    rm = bridge.region_manager
    region_filter = []
    for kind, names in [
        ("noun_pool", ["APPLE", "RIVER", "DOG", "CAT"]),
        ("verb_pool", ["GO", "COME", "STOP", "LOOK"]),
        ("adjective_pool", ["BIG", "SMALL", "HOT", "COLD"]),
    ]:
        for n in names:
            try:
                rm.indices(f"{kind}_{n}")
                region_filter.append(f"{kind}_{n}")
            except Exception:
                pass

    pairs = parse_pairs(CONCEPT_PAIRS_STR)
    valid_concepts = [w for w in _ALL_CONCEPTS
                       if _WORD_TO_IDX[w] < N_WORDS_FOR_ORTHOGONAL]
    print(f"  [seed {seed}] encoding {len(pairs)} multitag pairs",
          flush=True)
    t_enc = time.time()
    for a, b in pairs:
        tag = f"{a}_{b}"
        encode_concept_pair(
            bridge, a, b, tag,
            encoding_steps=ENCODING_STEPS, drive_pA=DRIVE_PA,
            sparsity=SPARSITY, n_lang_input=N_LANG_INPUT,
            n_words_for_orthogonal=N_WORDS_FOR_ORTHOGONAL,
            region_filter=region_filter, top_k=TOP_K_ENGRAM,
            balanced_teacher_pA=BALANCED_TEACHER_PA,
            verbose=False,
        )
    print(f"  [seed {seed}] encoded in {(time.time()-t_enc)/60:.1f}"
          f" min", flush=True)

    print(f"  [seed {seed}] testing PRE-SLEEP retrieval", flush=True)
    t_pre = time.time()
    pre_result = test_retrieval(
        bridge, pairs, region_filter, valid_concepts)
    print(f"  [seed {seed}] PRE-SLEEP: {pre_result['n_full']}/"
          f"{pre_result['n_total']} FULL = "
          f"{pre_result['full_pass_rate']:.3f}; partial "
          f"{pre_result['partial_pass_rate']:.3f} "
          f"({(time.time()-t_pre)/60:.1f} min)", flush=True)

    print(f"  [seed {seed}] running SWR consolidation ({N_SWR_EVENTS}"
          f" events)", flush=True)
    t_sleep = time.time()
    rng = np.random.default_rng(seed * 7)
    run_swr_replay_phase(
        bridge,
        n_swr_events=N_SWR_EVENTS,
        burst_duration_ms=BURST_DURATION_MS,
        inter_burst_ms=INTER_BURST_MS,
        swr_drive_pA=SWR_DRIVE_PA,
        rng=rng,
    )
    print(f"  [seed {seed}] sleep cycle done"
          f" ({(time.time()-t_sleep)/60:.1f} min)", flush=True)

    print(f"  [seed {seed}] silencing hippocampus + testing "
          f"POST-SLEEP retrieval", flush=True)
    n_silenced = silence_hippocampus(bridge)
    print(f"  [seed {seed}] silenced {n_silenced} CA3 neurons",
          flush=True)
    t_post = time.time()
    post_result = test_retrieval(
        bridge, pairs, region_filter, valid_concepts)
    print(f"  [seed {seed}] POST-SLEEP (hippo silenced): "
          f"{post_result['n_full']}/{post_result['n_total']} FULL = "
          f"{post_result['full_pass_rate']:.3f}; partial "
          f"{post_result['partial_pass_rate']:.3f} "
          f"({(time.time()-t_post)/60:.1f} min)", flush=True)

    retention_ratio = (
        post_result["full_pass_rate"] /
        (pre_result["full_pass_rate"] + 1e-9))
    print(f"  [seed {seed}] retention: post/pre full ratio = "
          f"{retention_ratio:.2f}", flush=True)

    return {
        "seed": seed,
        "pre_sleep": pre_result,
        "post_sleep": post_result,
        "retention_full_ratio": retention_ratio,
    }


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction P: multitag + SWR consolidation ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  Tests if multitag-encoded associations survive a sleep",
          flush=True)
    print(f"  cycle + cortex-only retrieval (CLS / McClelland 1995)",
          flush=True)
    print(f"  Pre-registered: post-sleep >= 70% of pre-sleep retention",
          flush=True)

    t0 = time.time()
    seed_results = []
    for seed in SEEDS:
        r = run_one_seed(seed)
        if r is not None: seed_results.append(r)
    total_min = (time.time() - t0) / 60

    if not seed_results:
        print("[FATAL] no cached HIPPO bridges", flush=True)
        return 1

    pre_rates = [r["pre_sleep"]["full_pass_rate"]
                 for r in seed_results]
    post_rates = [r["post_sleep"]["full_pass_rate"]
                  for r in seed_results]
    retentions = [r["retention_full_ratio"] for r in seed_results]
    pre_mean = float(np.mean(pre_rates))
    post_mean = float(np.mean(post_rates))
    retention_mean = float(np.mean(retentions))

    print(f"\n=== MULTI-SEED RESULTS ===", flush=True)
    print(f"  pre-sleep multitag FULL mean:  {pre_mean:.3f} per-seed"
          f"={pre_rates}", flush=True)
    print(f"  post-sleep (hippo silenced) FULL mean: {post_mean:.3f}"
          f" per-seed={post_rates}", flush=True)
    print(f"  retention ratio (post/pre):     {retention_mean:.3f}"
          f" per-seed={[round(r, 2) for r in retentions]}",
          flush=True)
    print(f"  Wall: {total_min:.1f} min", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if retention_mean >= 0.70 and post_mean >= 0.50:
        verdict = "MULTITAG_SURVIVES_SLEEP_CONSOLIDATION_VALIDATED"
        print(f"  PASS: multitag associations SURVIVE sleep "
              f"consolidation at retention {retention_mean:.2f}x. "
              f"CLS transfers associations to cortex; hippo-"
              f"silenced retrieval works. Pillar n=105 CANDIDATE.",
              flush=True)
    elif retention_mean >= 0.40:
        verdict = "MULTITAG_PARTIAL_CONSOLIDATION"
        print(f"  Partial consolidation: retention {retention_mean:.2f}"
              f"x. SWR transfers some associations but not robustly.",
              flush=True)
    else:
        verdict = "MULTITAG_CONSOLIDATION_FAILS"
        print(f"  Consolidation fails: retention {retention_mean:.2f}"
              f"x. SWR doesn't transfer multitag bindings to cortex.",
              flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seeds": SEEDS,
        "concept_pairs": CONCEPT_PAIRS_STR,
        "n_swr_events": N_SWR_EVENTS,
        "pre_sleep_full_mean": pre_mean,
        "pre_sleep_per_seed": pre_rates,
        "post_sleep_full_mean": post_mean,
        "post_sleep_per_seed": post_rates,
        "retention_mean": retention_mean,
        "retention_per_seed": retentions,
        "per_seed": seed_results,
        "verdict": verdict, "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
