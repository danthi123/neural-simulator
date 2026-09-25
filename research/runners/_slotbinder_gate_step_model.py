"""_slotbinder_gate_step_model.py -- count the simulation steps the SlotBinder production gate's slotbinder arm runs,
per phase, for the real seed-S N-fact sample, and check the count against the measured cupy runs (2026-09-25).

WHY. The N=404 dev-seed gate run sat silent for 6h08m (prereg AMENDMENT 1) and was read as stuck in the teach. The
arm's cost is (number of simulation steps) x (seconds per step), and the step count is fixed by the protocol, so it
can be computed exactly instead of guessed:
  teach:    5 `store_pair` calls per fact x `teach_steps` (40)
  query:    `SlotBinderComposer._match(cue_a, cue_v)` reads fact i's agent slot, then (only if it matches) its verb
            slot, and stops at the first fact matching both; a hit then reads the patient slot. Each read is
            `retr_steps` (40) steps. So a query for fact j costs (#facts before the first match) + 2 x (#of those
            sharing the cue's agent) + 3 reads under ideal reads.
  ablation: after `cp_connections.data[:] = 0` no filler fires, every read returns vocab[0], nothing matches, and
            every ablated query scans all N facts (N reads, +1 per fact whose agent is vocab[0]).
  moat / mismatch probes: one query each (the runner's own RNG draw and pairing).
Reads are modelled as IDEAL (each returns the stored word); real recall is 0.74-1.0 on these samples, so the count
is an estimate for the intact queries and exact for the teach and the ablation.

CHECK. The measured cupy sizing artifacts (N=8/32/128) give seconds per phase; dividing by the modelled step count
gives ms/step per phase. If the model were wrong the per-phase ms/step would disagree wildly between phases of one
run; they agree within the teach/read difference in per-step work (see the output). The L3 cupy latency artifact
(K=2020, the N=404 topology, 2 facts stored) gives the per-step cost at N=404 directly: its moat probe scanned 2
facts = 2 reads = 80 steps.

Writes one JSON (default research/findings/raw/_slotbinder_sparse_step/step_model_seed7.json). CPU only, seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from research.runners._slotbinder_l2_sparse_derisk import _load_live_bundle, _sample_facts  # noqa: E402

TEACH_STEPS, RETR_STEPS, ROLES = 40, 40, 5
SIZING = "research/findings/raw/_slotbinder_production_gate/sizing/seed7_n{n}.json"
L3 = "research/findings/raw/_slotbinder_l3_latency_derisk_cupy/latency_f32_s{s}.json"


def reads_for_query(facts, a, v, read_agent=None, read_verb=None):
    """Replicates SlotBinderComposer._match(cue_a=a, cue_v=v) + the patient read on a hit (ideal reads)."""
    n = 0
    for f in facts:
        n += 1
        if (f["agent"] if read_agent is None else read_agent) != a:
            continue
        n += 1
        if (f["action"] if read_verb is None else read_verb) != v:
            continue
        return n + 1
    return n


def model(sample, seed):
    N = len(sample)
    words = sorted({w for f in sample for w in (f["agent"], f["action"], f["patient"])})
    intact = [reads_for_query(sample, f["agent"], f["action"]) for f in sample]
    stored = {(f["agent"], f["action"]) for f in sample}
    rng = np.random.default_rng(seed * 97 + 3)             # the gate runner's own moat draw
    moat = 0
    for _ in range(300):
        a, v = words[rng.integers(len(words))], words[rng.integers(len(words))]
        if (a, v) in stored:
            continue
        moat = reads_for_query(sample, a, v)
        break
    mm = 0
    if N >= 2 and (sample[0]["agent"], sample[1]["action"]) not in stored:
        mm = reads_for_query(sample, sample[0]["agent"], sample[1]["action"])
    v0 = words[0]                                         # the composer's vocab is sorted(concepts); see docstring
    abl = [reads_for_query(sample, f["agent"], f["action"], read_agent=v0, read_verb=v0) for f in sample]
    teach = N * ROLES * TEACH_STEPS
    query = RETR_STEPS * (sum(intact) + moat + mm)
    ablation = RETR_STEPS * sum(abl)
    total = teach + query + ablation
    return {"n_facts": N, "teach_steps": teach, "intact_reads": sum(intact), "moat_reads": moat,
            "mismatch_reads": mm, "ablation_reads": sum(abl), "query_steps": query, "ablation_steps": ablation,
            "total_steps": total, "teach_share": teach / total, "query_share": query / total,
            "ablation_share": ablation / total}


def measured(m, n):
    p = os.path.join(_REPO, SIZING.format(n=n))
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    sb = d["arms"]["slotbinder"]
    q = sum(r["query_latency_s"] for r in sb["per_fact"])
    q += (sb.get("moat_probe") or {}).get("query_latency_s", 0.0)
    q += (sb.get("mismatch_probe") or {}).get("query_latency_s", 0.0)
    abl = sb["wall_clock_s"] - sb["build_seconds"] - q
    return {"artifact": SIZING.format(n=n), "teach_s": sb["build_seconds"], "query_s": q, "ablation_s": abl,
            "arm_wall_s": sb["wall_clock_s"],
            "ms_per_step_teach": 1000.0 * sb["build_seconds"] / m["teach_steps"],
            "ms_per_step_query": 1000.0 * q / m["query_steps"],
            "ms_per_step_ablation": 1000.0 * abl / m["ablation_steps"] if m["ablation_steps"] else None}


def l3_per_step():
    rows = []
    for s in (42, 43, 44, 100, 101, 102):
        p = os.path.join(_REPO, L3.format(s=s))
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        # the moat probe (a never-stored pair) scans both stored facts' agent slots: 2 reads = 80 steps
        rows.append({"seed": s, "artifact": L3.format(s=s), "K": d["K"], "nnz": d["measured_nnz"],
                     "moat_probe_s": d["moat_probe"]["query_latency_s"],
                     "ms_per_step": 1000.0 * d["moat_probe"]["query_latency_s"] / (2 * RETR_STEPS)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-facts", type=int, nargs="+", default=[8, 32, 128, 404])
    ap.add_argument("--out", default="research/findings/raw/_slotbinder_sparse_step/step_model_seed7.json")
    args = ap.parse_args()
    _, facts_full, _ = _load_live_bundle()
    out = {"seed": args.seed, "teach_steps_per_pair": TEACH_STEPS, "retr_steps_per_read": RETR_STEPS,
           "per_n": {}, "l3_cupy_per_step_at_k2020": l3_per_step()}
    for N in args.n_facts:
        sample, _ = _sample_facts(facts_full, args.seed, N)
        m = model(sample, args.seed)
        m["measured_cupy"] = measured(m, N)
        out["per_n"][str(N)] = m
        print(json.dumps({k: v for k, v in m.items() if k != "measured_cupy"}), flush=True)
        if m["measured_cupy"]:
            print("   measured cupy:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                                  for k, v in m["measured_cupy"].items()}), flush=True)
    l3 = out["l3_cupy_per_step_at_k2020"]
    if l3:
        out["l3_ms_per_step_mean"] = float(np.mean([r["ms_per_step"] for r in l3]))
        n404 = out["per_n"].get("404")
        if n404:
            out["n404_dense_cupy_projection_h"] = n404["total_steps"] * out["l3_ms_per_step_mean"] / 1000.0 / 3600.0
            out["n404_dense_cupy_teach_projection_min"] = (n404["teach_steps"] * out["l3_ms_per_step_mean"]
                                                           / 1000.0 / 60.0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("L3 per-step (ms):", [round(r["ms_per_step"], 2) for r in l3],
          "-> N=404 dense projection h:", round(out.get("n404_dense_cupy_projection_h", float("nan")), 2),
          "teach min:", round(out.get("n404_dense_cupy_teach_projection_min", float("nan")), 1))
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
