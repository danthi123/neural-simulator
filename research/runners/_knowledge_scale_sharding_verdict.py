"""Capacity verdict for the SHARDED FHRR fact-store (knowledge-scale de-risk, 2026-08-20).

Freshly MEASURES the sharded store against an unsharded RFPhasorComposer reference at the SAME K, then earns a
tools.verdict.Verdict from the measured relationships (not typed-in numbers):
  * the routed query must be much faster than the unsharded scan at the same K (the capacity win),
  * the sharded answers must be BYTE-IDENTICAL to the unsharded ones (routing must not change recall),
  * recall must beat chance, and
  * the no-confab moat (unknown cue -> abstain) must be preserved.

Declared TEST SCAFFOLD: synthetic facts + a HOST agent-hash router (the faithful version is a learned/spiking
cue->sub-population router). No sim/ edit, no production default changed. Run:
  SIM_BACKEND=numpy python -m research.runners._knowledge_scale_sharding_verdict [--K 2000 --S 16 --D 128]
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
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402
from research.runners.sharded_phasor_store import ShardedPhasorStore  # noqa: E402
from tools.verdict import Verdict  # noqa: E402


def make_facts(K, n_ag, n_rel, n_pt, seed):
    rng = np.random.default_rng(seed); facts, seen = [], set()
    while len(facts) < K:
        a = f"ag{rng.integers(n_ag)}"; r = f"rel{rng.integers(n_rel)}"
        if (a, r) in seen:
            continue
        seen.add((a, r)); facts.append((a, r, f"pt{rng.integers(n_pt)}"))
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--n-recall", type=int, default=12)
    ap.add_argument("--n-moat", type=int, default=8)
    ap.add_argument("--out", type=str, default=str(_REPO / "research" / "findings" / "raw"
                                                  / "_knowledge_scale_sharding_verdict.json"))
    a = ap.parse_args()
    NAG = max(50, a.K); NREL = 40; NPT = max(50, a.K)
    vocab = ([f"ag{i}" for i in range(NAG)] + [f"rel{i}" for i in range(NREL)]
             + [f"pt{i}" for i in range(NPT)])
    facts = make_facts(a.K, NAG, NREL, NPT, seed=11)

    ref = RFPhasorComposer(seed=42, D=a.D, vocab=vocab)
    t0 = time.time()
    for ag, r, p in facts:
        ref.store(ag, r, p)
    ref_build = time.time() - t0
    store = ShardedPhasorStore(n_shards=a.S, seed=42, D=a.D, vocab=vocab)
    t0 = time.time()
    for ag, r, p in facts:
        store.store(ag, r, p)
    sh_build = time.time() - t0

    idx = np.random.default_rng(3).choice(a.K, size=a.n_recall, replace=False)
    # unsharded: latency + recall
    t0 = time.time(); ref_correct = 0; ref_ans = []
    for i in idx:
        ag, r, p = facts[i]; ans = ref.query_patient(ag, r); ref_ans.append(ans)
        if ans == p:
            ref_correct += 1
    unrouted_ms = (time.time() - t0) / len(idx) * 1000
    # sharded: latency + recall + answer-identity
    t0 = time.time(); sh_correct = 0; mismatches = 0
    for k, i in enumerate(idx):
        ag, r, p = facts[i]; ans = store.query_patient(ag, r)
        if ans == p:
            sh_correct += 1
        if ans != ref_ans[k]:
            mismatches += 1
    routed_ms = (time.time() - t0) / len(idx) * 1000
    # moat
    ref_ab = sum(ref.query_patient(f"UNKNOWN_{j}", "rel0") is None for j in range(a.n_moat))
    sh_ab = sum(store.query_patient(f"UNKNOWN_{j}", "rel0") is None for j in range(a.n_moat))
    lb = store.load_balance()

    art = {
        "K": a.K, "S": a.S, "D": a.D, "vocab": len(vocab),
        "ref_build_s": ref_build, "sharded_build_s": sh_build,
        "unrouted_ms_per_query": unrouted_ms, "routed_ms_per_query": routed_ms,
        "speedup": (unrouted_ms / routed_ms) if routed_ms else None,
        "ref_recall": ref_correct, "sharded_recall": sh_correct, "n_recall": a.n_recall,
        "answer_mismatches_vs_unsharded": mismatches,
        "ref_moat_abstain": ref_ab, "sharded_moat_abstain": sh_ab, "n_moat": a.n_moat,
        "load_balance_min_max_mean_ratio": [lb[0], lb[1], lb[2], lb[3]],
        "chance": 1.0 / NPT,
    }
    print(json.dumps(art, indent=2))

    v = Verdict("sharded FHRR fact-store: capacity at tractable routed latency", chance=1.0 / NPT)
    v.control("routed vs unrouted latency at same K", treatment=unrouted_ms, control=routed_ms,
              min_separation=unrouted_ms * 0.5, note="routed must be >=2x faster than the full-K scan")
    v.require("routing preserves the answer (byte-identical to unsharded)", mismatches, expect=0)
    v.require("recall preserved vs unsharded", sh_correct, expect=lambda x: x >= ref_correct)
    v.floor("sharded recall vs chance", measured=sh_correct / a.n_recall, floor=1.0 / NPT)
    v.require("no-confab moat: every unknown cue abstains", sh_ab, expect=a.n_moat)
    v.disabled("learned/spiking cue->shard router",
               why="host agent-hash router is a declared capacity-de-risk scaffold; recall+moat inside each "
                   "shard remain the genuine RF/spiking reads")
    decided = v.decide(go=(routed_ms < unrouted_ms and mismatches == 0
                           and sh_correct >= ref_correct and sh_ab == a.n_moat))
    art["verdict"] = decided
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(art, indent=2))
    print(f"\nwrote {a.out}")
    return decided["status"]


if __name__ == "__main__":
    main()
