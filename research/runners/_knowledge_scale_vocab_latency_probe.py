"""KNOWLEDGE-SCALE VOCAB-LATENCY PROBE (board #66, knowledge-scale-bulk-bundle arc, 2026-08-28).

WHY. The 2026-08-20/2026-08-27 sharding findings established "sub-second at ANY K" for the tiered LTM's ROUTED
query, because shard size (not total fact count K) drives per-query cost. Those findings were measured against
the shipped `wikidata_100k` bundle (78,857 facts, vocab 23,914). This runner asks the question those findings
never tested: does the sub-second bar hold when a genuinely LARGER real-Wikidata bundle also brings a much
larger VOCABULARY (as real bulk-KB growth naturally does -- more facts pull in more distinct entities)?

METHOD. Through the EXACT production path (`developed_brain_io.load_developed_brain(..., ltm_bundle=<bundle>)`,
the same call `webapp/server.py` makes for `BRAIN_LTM_BUNDLE`), run a SMALL number of individual, INCREMENTALLY
PRINTED `what_does` / `is_it_true` queries against a real bundle and record per-query wall-clock latency. Small N
by design (this is a probe for the SCALING TREND across bundles of different vocab size, not a statistical
recall estimate -- recall/moat correctness at 500k scale is separately verified with a larger battery in the
companion finding). Run once per bundle to build a vocab-size vs. latency curve.

Run:
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._knowledge_scale_vocab_latency_probe \
      --bundle /home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k --n-probes 5 --seed 42 \
      --json research/findings/raw/knowledge_500k_verify/vocab_latency_100k.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--n-probes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    t0 = time.time()
    with open(os.path.join(a.bundle, "manifest.json")) as f:
        mani = json.load(f)
    with open(os.path.join(a.bundle, "facts.json")) as f:
        raw = json.load(f)
    facts = [r["fact"] if isinstance(r, dict) and "fact" in r else r for r in raw]
    D = int(mani["D"]); vocab = mani["vocab"]
    print(f"[0] manifest: n_facts={mani['n_facts']} vocab={len(vocab)} n_shards={mani['n_shards']} "
          f"({time.time()-t0:.1f}s)", flush=True)

    fm = {}
    for f in facts:
        agent, act, p = f.get("agent"), f.get("action"), f.get("patient")
        pol = f.get("polarity") or "AFFIRM"
        if isinstance(agent, str) and isinstance(act, str) and isinstance(p, str) and (agent, act) not in fm \
                and pol == "AFFIRM":
            fm[(agent, act)] = p
    keys = list(fm.keys())
    print(f"[0] ground-truth dict: {len(fm)} pairs ({time.time()-t0:.1f}s)", flush=True)

    from research.runners.developed_brain_io import save_developed_brain, load_developed_brain, _inner_agent
    from research.runners.brain_conversational_agent import BrainConversationalAgent

    conv_facts = [{"agent": "otter", "action": "caught", "patient": "clam", "polarity": "AFFIRM"}]
    cvocab = sorted({w for f in conv_facts for w in (f["agent"], f["action"], f["patient"])})
    ba = BrainConversationalAgent(seed=a.seed, concepts={w: None for w in cvocab}, composer_kind="rf",
                                  defer_parser=True)
    for f in conv_facts:
        ba.composer.store(f["agent"], f["action"], f["patient"], polarity=f["polarity"])
    tmp = tempfile.mkdtemp(prefix="ks_vocab_latency_")
    brain_dir = os.path.join(tmp, "brain")
    save_developed_brain(ba, brain_dir, seed=a.seed, D=D, composer_kind="rf")
    print(f"[1] tiny developed-brain saved ({time.time()-t0:.1f}s)", flush=True)

    tl0 = time.time()
    agent, load_manifest = load_developed_brain(brain_dir, ltm_bundle=a.bundle, use_multiturn=False, seed=a.seed)
    inner = _inner_agent(agent)
    ltm_load_s = time.time() - tl0
    print(f"[2] load_developed_brain(ltm_bundle=...) DONE in {ltm_load_s:.2f}s | "
          f"tiered={type(inner.composer).__name__} ltm={type(inner.composer.ltm).__name__} "
          f"total_facts={inner.composer.total_facts()} ({time.time()-t0:.1f}s total)", flush=True)

    rng = np.random.default_rng(a.seed)
    n = min(a.n_probes, len(keys))
    probe_idx = rng.choice(len(keys), size=n, replace=False)
    probes = [(keys[i][0], keys[i][1], fm[keys[i]]) for i in probe_idx]

    results = []
    for i, (agent_w, act, gt) in enumerate(probes):
        t1 = time.time()
        live = inner.what_does(agent_w, act)
        dt = time.time() - t1
        ok = (live == gt)
        results.append({"kind": "recall", "i": i, "cue": [agent_w, act], "gt": gt, "live": live, "ok": ok,
                         "latency_s": round(dt, 3)})
        print(f"[recall {i}] cue=({agent_w},{act}) gt={gt!r} live={live!r} ok={ok} latency={dt:.2f}s "
              f"({time.time()-t0:.1f}s total)", flush=True)

    t1 = time.time()
    live_unknown = inner.what_does("zzz_unknown_entity_xq", probes[0][1] if probes else "isa")
    dt = time.time() - t1
    confab = live_unknown is not None
    results.append({"kind": "moat", "cue": ["zzz_unknown_entity_xq", probes[0][1] if probes else "isa"],
                     "live": live_unknown, "confab": confab, "latency_s": round(dt, 3)})
    print(f"[moat] live={live_unknown!r} confab={confab} latency={dt:.2f}s ({time.time()-t0:.1f}s total)",
          flush=True)

    if probes:
        agent_w, act, gt = probes[0]
        t1 = time.time()
        yn = inner.is_it_true(agent_w, act, gt)
        dt = time.time() - t1
        results.append({"kind": "yesno", "cue": [agent_w, act, gt], "answer": yn, "latency_s": round(dt, 3)})
        print(f"[yesno] ({agent_w},{act},{gt}) -> {yn!r} latency={dt:.2f}s ({time.time()-t0:.1f}s total)",
              flush=True)

    shutil.rmtree(tmp, ignore_errors=True)

    recall_lats = [r["latency_s"] for r in results if r["kind"] == "recall"]
    out = {
        "bundle": a.bundle, "seed": a.seed,
        "n_facts": mani["n_facts"], "vocab_size": len(vocab), "n_shards": mani["n_shards"],
        "ltm_load_s": round(ltm_load_s, 2),
        "recall_ok": sum(1 for r in results if r["kind"] == "recall" and r["ok"]),
        "recall_checked": len(recall_lats),
        "moat_confab": sum(1 for r in results if r["kind"] == "moat" and r["confab"]),
        "recall_latency_median_s": round(float(np.median(recall_lats)), 2) if recall_lats else None,
        "recall_latency_min_s": round(float(np.min(recall_lats)), 2) if recall_lats else None,
        "recall_latency_max_s": round(float(np.max(recall_lats)), 2) if recall_lats else None,
        "results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"\nwrote {a.json}", flush=True)
    print(f"\n===== vocab={len(vocab)} n_facts={mani['n_facts']} "
          f"recall={out['recall_ok']}/{out['recall_checked']} moat_confab={out['moat_confab']} "
          f"median_latency={out['recall_latency_median_s']}s elapsed={out['elapsed_s']}s =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
