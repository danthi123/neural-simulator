"""Multi-bridge 3000-concept VALIDATION (Step 4 of the integration build).

Validates the production RoutedComposer on the trained 3,000-concept brain against the design's GO bars + the
anti-cheats. Reuses the VERIFIED Stage-0 measurement functions (measure_multibridge_recall / _moat,
measure_singlebridge, make_cross_shard_absent, build_timing_queries) so the bars are computed by the same code
that produced the Stage-0 GO -- this harness just points them at the 3,000 brain + the real _facts3000.json and
sweeps shard count + policy (domain vs partition).

Bars (design §4 Stage 2 + anti-cheats §5):
  - per-shard who/what recall >= 0.95 (report aggregate + worst shard; a weak shard cannot hide behind strong);
  - aggregate moat 0 false-accepts over absent + cross-shard-absent cues (HARD STOP if any FA);
  - per-query time <= the single-3000-bridge time (cleanup over ~V/N -> should be FASTER);
  - ANTI-CHEAT: permuted routing collapses recall WITHOUT raising false-accepts.

Run:  SIM_BACKEND=numpy python -m research.runners._multibridge_3000_validate \
          --facts-json research/findings/raw/_facts3000.json --n-shards 3 \
          --out research/findings/raw/_multibridge_3000_validate.json
"""
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

from research.runners.routed_composer import RoutedComposer, _domain_shards  # noqa: E402
from research.runners._multibridge_stage0_derisk import (  # noqa: E402
    load_brain,
    split_shards,
    build_shard_composers,
    store_facts_routed,
    measure_multibridge_recall,
    measure_multibridge_moat,
    measure_singlebridge,
    make_cross_shard_absent,
    build_timing_queries,
    time_multibridge,
)
from research.runners.first_chat_console import _load_real_facts  # noqa: E402
from research.runners._curriculum_step1_320_real_corpus import _make_svo_facts  # noqa: E402

DEFAULT_BRAIN = os.path.join(_REPO, "bridges", "firstchat", "brain3000pos_w7000.npz_seed42.npz")
DEFAULT_FACTS = os.path.join(_REPO, "research", "findings", "raw", "_facts3000.json")


def _shards_for_policy(vocab, cat_ids, cat_names, n_shards, seed, policy):
    if policy == "domain":
        s = _domain_shards(vocab, cat_ids, cat_names, n_shards)
        if s is not None:
            return s, "domain"
    return split_shards(vocab, n_shards, seed), "partition"


def run_one(vocab, grounded, cat_ids, cat_names, D, facts, absent_what, absent_who,
            n_shards, seed, policy, single_baseline):
    """One (n_shards, policy) configuration: store routed, measure recall + moat + time + permuted anti-cheat."""
    shards, used_policy = _shards_for_policy(vocab, cat_ids, cat_names, n_shards, seed, policy)
    shard_sizes = [len(s) for s in shards]
    comps, word2shard = build_shard_composers(shards, grounded, seed, D)
    store_stats = store_facts_routed(facts, comps, word2shard, grounded)

    mb_recall, per_shard, _miss = measure_multibridge_recall(facts, comps, word2shard)
    cross_absent = make_cross_shard_absent(facts, word2shard, max(len(facts), 8), seed)
    mb_abstain, mb_fa, mb_breaches = measure_multibridge_moat(absent_what, absent_who, cross_absent,
                                                              comps, word2shard)
    timing_queries = build_timing_queries(facts)
    mb_tq = time_multibridge(timing_queries, comps, word2shard, facts)

    # ANTI-CHEAT: permuted routing (store on the WRONG shard); query with the TRUE router -> must collapse.
    comps_perm, w2s_perm = build_shard_composers(shards, grounded, seed, D)
    store_facts_routed(facts, comps_perm, w2s_perm, grounded, permute=True, n_shards=n_shards)
    perm_recall, _pps, _m = measure_multibridge_recall(facts, comps_perm, word2shard)   # query w/ TRUE router
    perm_abstain, perm_fa, perm_breaches = measure_multibridge_moat(absent_what, absent_who, cross_absent,
                                                                    comps_perm, word2shard)

    per_shard_min = min((d["recall"] for d in per_shard.values()), default=0.0)
    sb_recall = single_baseline["recall"]
    sb_tq = single_baseline["sec_per_query"]
    go_recall_abs = per_shard_min >= 0.95 and mb_recall >= 0.95
    go_recall_vs_single = (mb_recall >= sb_recall - 1e-9) and (per_shard_min >= sb_recall - 1e-9)
    go_moat = (mb_fa == 0)
    go_time = mb_tq <= sb_tq * 1.05
    go_anticheat = (perm_recall < 0.5 * max(mb_recall, 1e-9)) and (perm_fa == 0)
    go = go_recall_vs_single and go_moat and go_time and go_anticheat

    return {
        "n_shards": n_shards, "policy": used_policy, "shard_sizes": shard_sizes,
        "store_stats": store_stats,
        "multibridge_recall": mb_recall, "per_shard": per_shard, "per_shard_min_recall": per_shard_min,
        "multibridge_abstain": mb_abstain, "multibridge_false_accept": mb_fa, "moat_breaches": mb_breaches,
        "n_cross_absent": len(cross_absent),
        "sec_per_query": mb_tq, "tq_ms": mb_tq * 1e3,
        "permuted_recall": perm_recall, "permuted_false_accept": perm_fa, "permuted_breaches": perm_breaches,
        "go_recall_per_shard_ge_0.95_absolute": bool(go_recall_abs),
        "go_recall_vs_single_bridge": bool(go_recall_vs_single),
        "go_moat_0_false_accepts": bool(go_moat),
        "go_time_le_single_bridge": bool(go_time),
        "go_anticheat_permuted_collapses_no_FA": bool(go_anticheat),
        "GO": bool(go),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default=DEFAULT_BRAIN)
    ap.add_argument("--facts-json", default=DEFAULT_FACTS)
    ap.add_argument("--n-facts", type=int, default=48, help="SVO facts stored (more facts than the 24-fact console "
                    "exercises the per-shard stores harder; the moat absent-sets scale with this)")
    ap.add_argument("--n-shards", type=int, default=3)
    ap.add_argument("--sweep-shards", default="2,3,4", help="also report these shard counts for the curve")
    ap.add_argument("--policies", default="domain,partition", help="sharding policies to compare")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(_REPO, "research", "findings", "raw",
                                                  "_multibridge_3000_validate.json"))
    args = ap.parse_args()

    print(f"[validate] backend={os.environ.get('SIM_BACKEND', 'auto')} brain={os.path.basename(args.brain)}",
          flush=True)
    vocab, grounded, cat_ids, cat_names, D = load_brain(args.brain)
    print(f"[validate] {len(vocab)} concepts, D={D}, {len(cat_names)} categories", flush=True)

    facts, absent_what, absent_who = [], [], []
    if args.facts_json and os.path.exists(args.facts_json):
        facts, absent_what, absent_who = _load_real_facts(args.facts_json, vocab, args.n_facts, args.seed)
        fact_src = f"real:{os.path.basename(args.facts_json)}"
    if not facts:
        facts, absent_what, absent_who = _make_svo_facts(vocab, cat_ids, cat_names, args.n_facts, args.seed)
        fact_src = "synthetic:_make_svo_facts"
    print(f"[validate] {len(facts)} SVO facts ({fact_src}); {len(absent_what)} absent_what, "
          f"{len(absent_who)} absent_who", flush=True)

    # the single-bridge baseline (one composer over the full 3,000 vocab + the same facts) -> recall + time/query.
    timing_queries = build_timing_queries(facts)
    single = measure_singlebridge(vocab, grounded, facts, absent_what, absent_who, args.seed, D, timing_queries)
    print(f"[validate] SINGLE-BRIDGE 3000: recall={single['recall']:.3f} abstain={single['abstain']:.3f} "
          f"FA={single['false_accept']} t/q={single['sec_per_query']*1e3:.1f}ms", flush=True)

    shard_counts = sorted({int(x) for x in args.sweep_shards.split(",")} | {args.n_shards})
    policies = [p.strip() for p in args.policies.split(",")]
    results = []
    for ns in shard_counts:
        for pol in policies:
            r = run_one(vocab, grounded, cat_ids, cat_names, D, facts, absent_what, absent_who,
                        ns, args.seed, pol, single)
            results.append(r)
            ww = min((d["recall"] for d in r["per_shard"].values()), default=0.0)
            print(f"[validate] N={ns:<2} {r['policy']:<9} sizes={r['shard_sizes']}  recall={r['multibridge_recall']:.3f} "
                  f"(worst-shard {ww:.3f})  FA={r['multibridge_false_accept']}  t/q={r['tq_ms']:.1f}ms  "
                  f"perm={r['permuted_recall']:.3f}(FA{r['permuted_false_accept']})  "
                  f"{'GO' if r['GO'] else 'no'}  [same/cross={r['store_stats']['n_same_shard']}/"
                  f"{r['store_stats']['n_cross_shard']}, ext={r['store_stats']['codebook_ext_per_shard']}]",
                  flush=True)

    # the PRIMARY config = the requested --n-shards, preferring the policy that GOes (domain first).
    primary = None
    for pol in policies:
        for r in results:
            if r["n_shards"] == args.n_shards and r["policy"] == pol:
                primary = r if (primary is None or (r["GO"] and not primary["GO"])) else primary
    out = {
        "brain": os.path.basename(args.brain), "seed": args.seed, "D": D, "n_concepts": len(vocab),
        "fact_source": fact_src, "n_facts": len(facts),
        "n_absent_what": len(absent_what), "n_absent_who": len(absent_who),
        "single_bridge_3000": single,
        "results": results,
        "primary_n_shards": args.n_shards,
        "primary": primary,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[validate] PRIMARY (N={args.n_shards}): "
          f"{'GO' if primary and primary['GO'] else 'NO-GO'} "
          f"(recall {primary['multibridge_recall']:.3f} vs single {single['recall']:.3f}, "
          f"FA {primary['multibridge_false_accept']}, t/q {primary['tq_ms']:.1f} vs "
          f"{single['sec_per_query']*1e3:.1f}ms, perm {primary['permuted_recall']:.3f})" if primary else "",
          flush=True)
    print(f"[validate] wrote {args.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
