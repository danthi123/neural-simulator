"""L3 wire-in de-risk: does SlotBinderComposer's PER-QUERY latency (after L2's fanout=32 sparsification AND
this session's O(KF)-readout vectorization fix -- see slotbinder_composer.py's `fill_idx_mat` change) sit within
an interactive budget at the REAL production scale, and how does it compare to the incumbent FHRR composer's
measured 0.9s/query (research/findings/2026-09-04-slotbinder-live-scale-derisk-NOGO-dense-pathway-blowup.md §6)?

Board/lane: rung L3 of research/findings/2026-09-04-vsa-composer-learned-retirement-ROADMAP.md ("wire
composer_kind='slotbinder' as the production default ... a 320-scale GPU re-verify"). This runner is the
DE-RISK half of L3 (measure readiness), not the wire-in-as-DEFAULT half (explicitly out of scope for this task
-- the flag stays default-off; see brain_conversational_agent.py / developed_brain_io.py / webapp/server.py
BRAIN_COMPOSER_KIND=slotbinder wiring landed alongside this runner).

Reuses the L2 runner's own live-bundle loader + fact sampler (`_slotbinder_l2_sparse_derisk.py`) so the SAME
404-fact/788-vocab K=2020/KF=1195 real topology and the SAME seed-dependent real-fact sampling this finding's
own parent (L2) used are exercised here -- comparable, not a fresh ad-hoc protocol. CPU/numpy only (matches
L1/L2's own backend choice + this task's cost-routing instruction: webapp deps -> LOCAL, not the mini-PC pool).

TWO things measured per (seed, fanout):
  (a) CORRECTNESS re-confirmation on the NOW-VECTORIZED code (recall / moat / mismatch) -- a regression check
      that the O(KF) readout fix changed ONLY latency, not any answer (the equivalence was ALSO separately
      verified bit-exact against the original loop on a real built bridge, in a throwaway scratch script; this
      is the SlotBinderComposer contract-level echo of that same guarantee).
  (b) EXPLICIT per-call wall-clock latency for individual `query_patient` calls (both the real-fact hits AND the
      moat/mismatch probes), not just the aggregate `build_and_store_seconds` L2 reported -- this is the number
      the roadmap's L3 gate and this task's "latency-vs-FHRR" criterion actually need.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._slotbinder_l2_sparse_derisk import (  # noqa: E402
    _load_live_bundle, _sample_facts, _prewire_dicts, slot_filler_nnz_formula,
)
from research.runners.slotbinder_composer import SlotBinderComposer, _ROLES  # noqa: E402

FHRR_REFERENCE = {
    "peak_rss_mb": 334, "mean_query_s": 0.9, "n_correct_of_3": 3,
    "store_all_404_facts_s": 20.6,
    "source": "research/findings/2026-09-04-slotbinder-live-scale-derisk-NOGO-dense-pathway-blowup.md #6, "
              "research/findings/raw/_fhrr_rf_composer_live_scale_404facts.json",
}


def run_seed(seed, fanout, n_facts, vocab, facts):
    sample, idx = _sample_facts(facts, seed, n_facts)
    prewire = _prewire_dicts(sample)
    t0 = time.time()
    c = SlotBinderComposer(seed=seed, vocab=vocab, max_facts=len(facts), fanout=fanout, prewire_facts=prewire)
    for f in sample:
        ok = c.store(f["agent"], f["action"], f["patient"], polarity=f.get("polarity"))
        if not ok:
            raise RuntimeError(f"store() rejected a REAL live-bundle fact: {f}")
    build_and_store_s = time.time() - t0
    K, KF = c._b._K_slots, len(c._vocab)

    # --- (a) per-fact query_patient, EACH individually timed -----------------------------------------------
    per_fact_latency = []
    for i, f in enumerate(sample):
        agent, action, patient = f["agent"], f["action"], f["patient"]
        t0 = time.time()
        got = c.query_patient(agent, action)
        dt = time.time() - t0
        per_fact_latency.append({"fact_idx_in_corpus": idx[i], "agent": agent, "action": action,
                                 "expected_patient": patient, "got_patient": got, "hit": got == patient,
                                 "query_latency_s": dt})

    # --- (b) moat probe (never-stored cue), timed -----------------------------------------------------------
    stored_pairs = {(f["agent"], f["action"]) for f in sample}
    all_words = c.words
    rng = np.random.default_rng(seed * 97 + 1)
    moat_latency = None
    for _try in range(200):
        a, v = all_words[rng.integers(len(all_words))], all_words[rng.integers(len(all_words))]
        if (a, v) in stored_pairs:
            continue
        t0 = time.time()
        got = c.query_patient(a, v)
        dt = time.time() - t0
        moat_latency = {"agent": a, "action": v, "abstained": got is None, "query_latency_s": dt}
        break

    # --- (c) mismatch probe (cross fact i's agent with fact j's action), timed --------------------------------
    mismatch_latency = None
    if len(sample) >= 2:
        a, v = sample[0]["agent"], sample[1]["action"]
        if (a, v) not in stored_pairs:
            t0 = time.time()
            got = c.query_patient(a, v)
            dt = time.time() - t0
            mismatch_latency = {"agent": a, "action_from_other_fact": v,
                                "did_not_leak_fact0_patient": got != sample[0]["patient"],
                                "query_latency_s": dt}

    all_query_latencies = [r["query_latency_s"] for r in per_fact_latency]
    if moat_latency:
        all_query_latencies.append(moat_latency["query_latency_s"])
    if mismatch_latency:
        all_query_latencies.append(mismatch_latency["query_latency_s"])

    return {
        "seed": seed, "fanout": fanout, "n_facts_sampled": len(sample), "sampled_corpus_indices": idx,
        # gates/device_and_cost: a result artifact must record its own backend (SIM_BACKEND defaults to numpy via
        # this file's own os.environ.setdefault, silently, so recording it explicitly is what makes the choice
        # auditable rather than assumed).
        "sim_backend": os.environ.get("SIM_BACKEND", "numpy"),
        "K": K, "KF": KF, "measured_nnz": int(c._b.cp_connections.nnz),
        "formula_nnz": slot_filler_nnz_formula(K, KF, fanout=fanout),
        "build_and_store_seconds": build_and_store_s,
        "per_fact_query_latency": per_fact_latency,
        "moat_probe": moat_latency,
        "mismatch_probe": mismatch_latency,
        "recall_accuracy_query_patient": sum(r["hit"] for r in per_fact_latency) / len(per_fact_latency),
        "moat_pass": bool(moat_latency and moat_latency["abstained"]),
        "mismatch_pass": bool(mismatch_latency is None or mismatch_latency["did_not_leak_fact0_patient"]),
        "mean_query_latency_s": float(np.mean(all_query_latencies)),
        "max_query_latency_s": float(np.max(all_query_latencies)),
        "min_query_latency_s": float(np.min(all_query_latencies)),
        "vs_fhrr_mean_ratio": float(np.mean(all_query_latencies)) / FHRR_REFERENCE["mean_query_s"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--fanout", type=int, default=32)
    ap.add_argument("--n-facts", type=int, default=2)
    ap.add_argument("--out-dir", type=str, default=os.path.join(_REPO, "research/findings/raw/_slotbinder_l3_latency_derisk"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    vocab, facts, brain = _load_live_bundle()
    print(f"live bundle: n_facts={len(facts)} vocab={len(vocab)} composer_kind={brain.get('composer_kind')}")
    print(f"FHRR reference (cited): {json.dumps(FHRR_REFERENCE)}")

    rows = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} fanout {args.fanout} ===", flush=True)
        t0 = time.time()
        row = run_seed(seed, args.fanout, args.n_facts, vocab, facts)
        wall = time.time() - t0
        row["wall_clock_this_seed_s"] = wall
        rows.append(row)
        out_path = os.path.join(args.out_dir, f"latency_f{args.fanout}_s{seed}.json")
        json.dump(row, open(out_path, "w"), indent=2)
        print(f"  seed {seed}: build_and_store={row['build_and_store_seconds']:.1f}s "
              f"mean_query={row['mean_query_latency_s']:.3f}s (vs FHRR {FHRR_REFERENCE['mean_query_s']}s, "
              f"ratio {row['vs_fhrr_mean_ratio']:.1f}x) recall={row['recall_accuracy_query_patient']} "
              f"moat_pass={row['moat_pass']} mismatch_pass={row['mismatch_pass']} wall={wall:.1f}s -> {out_path}",
              flush=True)

    summary = {
        "fanout": args.fanout, "n_facts_per_seed": args.n_facts, "seeds": args.seeds,
        "sim_backend": os.environ.get("SIM_BACKEND", "numpy"),
        "fhrr_reference": FHRR_REFERENCE,
        "all_recall_1.0": all(r["recall_accuracy_query_patient"] == 1.0 for r in rows),
        "all_moat_pass": all(r["moat_pass"] for r in rows),
        "all_mismatch_pass": all(r["mismatch_pass"] for r in rows),
        "mean_query_latency_s_across_seeds": float(np.mean([r["mean_query_latency_s"] for r in rows])),
        "max_query_latency_s_across_seeds": float(np.max([r["max_query_latency_s"] for r in rows])),
        "mean_build_and_store_s_across_seeds": float(np.mean([r["build_and_store_seconds"] for r in rows])),
        "rows": rows,
    }
    out_path = os.path.join(args.out_dir, f"summary_f{args.fanout}.json")
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"\n=== SUMMARY -> {out_path} ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
