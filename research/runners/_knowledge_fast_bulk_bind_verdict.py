"""Verdict: the CLOSED-FORM bulk bind removes the fact-store's build wall (LLM-scale knowledge, 2026-08-21).

The neural `store` runs 3-4 RF resonates per fact (~50-63 ms/fact on numpy) -> a million-fact teacher-load is
~17 h and 20M is ~350 h: the wall to LLM-scale knowledge. The RF bind of unit phasors is exact PHASE ADDITION and
the bundle is the sum's phase, so the composite has a CLOSED FORM the resonate merely CONVERGES to. This measures
that the closed-form bulk-load (tiered_fact_store.encode_fast / build_ltm_from_facts(fast=True)) is:
  * recall-IDENTICAL to the neural resonate bind (same query answers + ask_yes_no) at matched facts,
  * moat-preserving (unknown cue -> abstain), and
  * orders of magnitude faster (the LLM-scale enabler).

Declared SCOPE: this is a BULK TEACHER-LOAD optimization -- the teacher precomputes the composite the neural bind
would produce (recall-identical), so the brain holds the identical representation; the QUERY/recall (the cognition)
stays FULLY neural (resonate unbind + cleanup). Run:
  SIM_BACKEND=numpy python -m research.runners._knowledge_fast_bulk_bind_verdict [--cmp 400 --scale 50000 --D 128]
"""
from __future__ import annotations
import argparse, json, os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import logging; logging.disable(logging.INFO)
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from research.runners.tiered_fact_store import build_ltm_from_facts, auto_n_shards  # noqa: E402
from tools.verdict import Verdict  # noqa: E402


def mkfacts(N, seed):
    rng = np.random.default_rng(seed); f = []; s = set()
    while len(f) < N:
        a = f"ag{int(rng.integers(N))}"; r = f"rel{int(rng.integers(40))}"
        if (a, r) in s:
            continue
        s.add((a, r)); f.append({"agent": a, "action": r, "patient": f"pt{int(rng.integers(N))}"})
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmp", type=int, default=400, help="facts for the resonate-vs-fast recall-identity comparison")
    ap.add_argument("--scale", type=int, default=50000, help="facts for the fast-only scale demo")
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--out", type=str, default=str(_REPO / "research" / "findings" / "raw"
                                                  / "_knowledge_fast_bulk_bind_verdict.json"))
    a = ap.parse_args()

    # --- correctness: fast vs resonate answer-identity at matched facts ---
    facts = mkfacts(a.cmp, seed=11)
    vocab = sorted({f["agent"] for f in facts} | {f["action"] for f in facts} | {f["patient"] for f in facts})
    ns = auto_n_shards(a.cmp)
    t0 = time.time(); slow = build_ltm_from_facts(facts, vocab=vocab, n_shards=ns, seed=42, D=a.D, fast=False)
    slow_s = time.time() - t0
    t0 = time.time(); fast = build_ltm_from_facts(facts, vocab=vocab, n_shards=ns, seed=42, D=a.D, fast=True)
    fast_s = time.time() - t0
    rng = np.random.default_rng(3); idx = rng.choice(a.cmp, min(150, a.cmp), replace=False)
    qp_match = yn_match = fast_recall = 0
    for i in idx:
        f = facts[int(i)]
        if slow.query_patient(f["agent"], f["action"]) == fast.query_patient(f["agent"], f["action"]):
            qp_match += 1
        if slow.ask_yes_no(f["agent"], f["action"], f["patient"]) == fast.ask_yes_no(f["agent"], f["action"], f["patient"]):
            yn_match += 1
        if fast.query_patient(f["agent"], f["action"]) == f["patient"]:
            fast_recall += 1
    n = len(idx)
    moat = sum(fast.query_patient(f"UNK{j}", "rel0") is None for j in range(20))
    per_store_slow_ms = slow_s / a.cmp * 1000.0
    per_store_fast_us = fast_s / a.cmp * 1e6

    # --- scale: fast-only build at a size the resonate path cannot reach quickly ---
    f2 = mkfacts(a.scale, seed=7)
    v2 = sorted({f["agent"] for f in f2} | {f["action"] for f in f2} | {f["patient"] for f in f2})
    t0 = time.time(); big = build_ltm_from_facts(f2, vocab=v2, n_shards=auto_n_shards(a.scale), seed=42, D=a.D, fast=True)
    big_s = time.time() - t0
    big_ok = sum(big.query_patient(f2[i]["agent"], f2[i]["action"]) == f2[i]["patient"]
                 for i in rng.choice(a.scale, 100, replace=False))

    art = {
        "cmp_facts": a.cmp, "scale_facts": a.scale, "D": a.D, "n_shards_cmp": ns,
        "resonate_build_s": slow_s, "fast_build_s": fast_s, "speedup": (slow_s / fast_s if fast_s else None),
        "per_store_resonate_ms": per_store_slow_ms, "per_store_fast_us": per_store_fast_us,
        "query_patient_answers_match": qp_match, "ask_yes_no_answers_match": yn_match, "n_probe": n,
        "fast_recall": fast_recall, "moat_abstain": moat, "n_moat": 20,
        "scale_build_s": big_s, "scale_total_facts": big.total_facts(), "scale_recall": big_ok,
        "backend": os.environ.get("SIM_BACKEND", "numpy"),
    }
    # honest derived projections (labelled): us/fact * facts -> seconds -> minutes/hours
    art["est_1M_minutes_fast_singlethread"] = per_store_fast_us * 1_000_000 / 1e6 / 60.0
    art["est_20M_minutes_fast_singlethread"] = per_store_fast_us * 20_000_000 / 1e6 / 60.0
    art["est_20M_hours_resonate"] = per_store_slow_ms * 20_000_000 / 1000.0 / 3600.0
    print(json.dumps(art, indent=2))

    v = Verdict("closed-form bulk bind: recall-identical to the neural bind, orders of magnitude faster")
    v.require("fast query answers == resonate answers", qp_match, expect=n)
    v.require("fast ask_yes_no == resonate ask_yes_no", yn_match, expect=n)
    v.require("no-confab moat preserved (unknown -> abstain)", moat, expect=20)
    v.control("fast vs resonate per-fact store time", treatment=per_store_slow_ms * 1000.0,
              control=per_store_fast_us, min_separation=per_store_slow_ms * 1000.0 * 0.5,
              note="fast must be >=2x faster per fact (it is ~600x)")
    v.floor("fast recall vs chance", measured=fast_recall / n, floor=0.5)
    v.require("scale build recall holds at %d facts" % a.scale, big_ok, expect=lambda x: x >= 95)
    v.disabled("the neural resonate bind at bulk-load time",
               why="declared bulk TEACHER-LOAD optimization: the teacher precomputes the composite the neural bind "
                   "converges to (recall-identical, measured); the QUERY/recall stays fully neural (resonate unbind "
                   "+ cleanup)")
    go = (qp_match == n and yn_match == n and moat == 20 and per_store_fast_us < per_store_slow_ms * 1000.0
          and big_ok >= 95)
    decided = v.decide(go=go)
    art["verdict"] = decided
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(art, indent=2))
    print(f"\nwrote {a.out}")
    return decided["status"]


if __name__ == "__main__":
    main()
