"""KNOWLEDGE CORE BUNDLE SOAK — the no-regression gate for making the CURATED wikidata core the BRAIN_LTM_BUNDLE
default (Task 2, board #133). Companion to _knowledge_core_curate.py (which builds + persists the bundle).

The generic tiered-store soak (_knowledge_scale_flip_soak.py) already proved the mechanism on SYNTHETIC facts, 6/6.
This soak proves the SAME no-regression properties on the ACTUAL SHIPPED ARTIFACT — the persisted, genuine-bind
(fast=False) ShardedPhasorStore built from REAL wikidata facts — through the EXACT production load path:

  (1) PRODUCTION LOAD PATH: load_developed_brain(ltm_bundle=<bundle dir>) (== what BRAIN_LTM_BUNDLE=<dir> calls)
      installs a TieredFactStore, recalls a known curated fact correctly, and ABSTAINS on an unknown cue (moat).
      This exercises the real persisted bundle end-to-end through a real agent.
  (2) BYTE-IDENTITY over 6 mission seeds (STRUCTURAL, real facts): rebuild the routed LTM + a single PLAIN-FLAT
      unsharded oracle from the SAME curated facts and confirm the tiered store returns the IDENTICAL answer for
      every cued read (what_does / is_it_true). Routing changes NO answer.
  (3) NO-CONFAB MOAT: unknown agents + a known agent with an action it never has -> abstain, identically to the
      oracle. 0 confabulations.
  (4) RECALL: live answer == the ordered first-match ground truth (curation dedups (subject,relation) so it is
      exact). >= 0.99.

The 6-seed structural rebuild uses the closed-form bulk bind (fast=True) for speed — recall-identical to the neural
resonate bind (finding 2026-08-21-closed-form-bulk-bind) — while the SHIPPED bundle validated in (1) is the genuine
fast=False build. Learn-through-use consolidation (BRAIN_D5_CONSOLIDATE, default-on) stays in the live path: the
bundle is the bulk cortical LTM the recent-conversation buffer falls through to.

GO => the technical no-regression gate for the default-on flip is closed on the real bundle; the flip itself
(pointing BRAIN_LTM_BUNDLE at this bundle by default) is the owner/Tuesday product decision.

Run (CPU/numpy, LOCAL):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._knowledge_core_bundle_soak \
    --bundle /home/dant123/Projects/sim-data/knowledge_bundles/wikidata_core_15k
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from research.runners.rf_phasor_composer import RFPhasorComposer
from research.runners.tiered_fact_store import TieredFactStore, build_ltm_from_facts, encode_fast

MISSION_SEEDS = [42, 43, 44, 100, 101, 102]
D = 128


def _load_bundle_facts(bundle):
    """ShardedPhasorStore.save writes facts.json as a list of {"shard": i, "fact": {...}} records; also tolerate a
    bare list of facts or a {"facts": [...]} dict."""
    with open(os.path.join(bundle, "facts.json"), "r", encoding="utf-8") as fh:
        d = json.load(fh)
    if isinstance(d, dict):
        d = d.get("facts", [])
    out = []
    for e in d:
        f = e.get("fact") if isinstance(e, dict) and "fact" in e else e
        if isinstance(f, dict) and isinstance(f.get("agent"), str) and isinstance(f.get("action"), str) \
                and isinstance(f.get("patient"), str):
            out.append(f)
    return out


def _vocab_of(facts):
    vs = set()
    for f in facts:
        for r in ("agent", "action", "patient"):
            w = f.get(r)
            if isinstance(w, str):
                vs.add(w)
    return sorted(vs)


def _first_match(facts):
    fm = {}
    for f in facts:
        key = (f["agent"], f["action"])
        if key not in fm and (f.get("polarity") or "AFFIRM") == "AFFIRM":
            fm[key] = f["patient"]
    return fm


def _build_plain_flat(facts, vocab, seed):
    comp = RFPhasorComposer(seed=seed, D=D, vocab=list(vocab))
    for f in facts:
        fd = {"agent": f["agent"], "action": f["action"], "patient": f["patient"],
              "polarity": f.get("polarity") or "AFFIRM"}
        comp.kb.append((fd, encode_fast(comp, fd)))
    return comp


def _install_ltm(agent, ltm):
    buffer = agent.composer if not isinstance(agent.composer, TieredFactStore) else agent.composer.buffer
    agent.composer = TieredFactStore(buffer, ltm)
    return buffer


def structural_seed_check(facts, vocab, fm, seed, probe_cap, moat_cap, agent):
    ltm = build_ltm_from_facts(facts, vocab=vocab, seed=seed, D=D, fast=True)
    _install_ltm(agent, ltm)
    oracle = _build_plain_flat(facts, vocab, seed)
    rng = np.random.default_rng(seed + 7)
    keys = [k for k in fm.keys()]
    idx = rng.choice(len(keys), size=min(probe_cap, len(keys)), replace=False)
    probes = [keys[int(i)] for i in idx]

    cell = {"seed": seed, "n_facts": len(facts), "n_shards": ltm.n_shards,
            "byte_identity_checked": 0, "byte_identity_mismatches": [],
            "moat_checked": 0, "moat_confab": 0, "moat_ok": 0,
            "recall_checked": 0, "recall_ok": 0}

    for (a, v) in probes:                                # warm
        agent.what_does(a, v)
    lat = []
    for (a, v) in probes:
        t0 = time.perf_counter()
        live = agent.what_does(a, v)
        lat.append(time.perf_counter() - t0)
        gt = fm.get((a, v))
        if gt is not None:
            cell["recall_checked"] += 1
            cell["recall_ok"] += int(live == gt)
        ref = oracle.query_patient(a, v)
        cell["byte_identity_checked"] += 1
        if live != ref and len(cell["byte_identity_mismatches"]) < 20:
            cell["byte_identity_mismatches"].append({"cue": [a, v], "oracle": repr(ref), "tiered": repr(live)})

    # moat: fabricated unknown agents + a known agent with an action it never has
    unknown_agents = [f"__nobody_{seed}_{j}__" for j in range(moat_cap)]
    actions = sorted({f["action"] for f in facts})
    mcues = [(ua, actions[int(rng.integers(0, len(actions)))]) for ua in unknown_agents]
    if facts:
        known_agent = facts[0]["agent"]
        mcues += [(known_agent, f"__no_such_action_{seed}__")]
    for (a, v) in mcues:
        live = agent.what_does(a, v)
        cell["moat_checked"] += 1
        if live is not None:
            cell["moat_confab"] += 1
        if live == oracle.query_patient(a, v):
            cell["moat_ok"] += 1

    cell["recall_rate"] = round(cell["recall_ok"] / cell["recall_checked"], 4) if cell["recall_checked"] else None
    cell["latency_ms_median"] = round(float(np.median(lat)) * 1000, 2) if lat else None
    cell["GO"] = bool(len(cell["byte_identity_mismatches"]) == 0
                      and cell["moat_confab"] == 0
                      and cell["moat_ok"] == cell["moat_checked"]
                      and (cell["recall_rate"] is None or cell["recall_rate"] >= 0.99))
    return cell


def production_load_path_check(bundle, facts, fm):
    """The REAL BRAIN_LTM_BUNDLE path on the ACTUAL persisted bundle: load_developed_brain(ltm_bundle=<bundle>)."""
    import tempfile
    out = {"available": False}
    tmp = None
    try:
        from research.runners.developed_brain_io import save_developed_brain, load_developed_brain
        from research.runners.brain_conversational_agent import BrainConversationalAgent
        tmp = tempfile.mkdtemp(prefix="core_bundle_soak_")
        conv_facts = [{"agent": "otter", "action": "caught", "patient": "clam", "polarity": "AFFIRM"},
                      {"agent": "dog", "action": "chase", "patient": "cat", "polarity": "AFFIRM"}]
        bvocab = sorted({w for f in conv_facts for w in (f["agent"], f["action"], f["patient"])})
        ba = BrainConversationalAgent(seed=42, concepts={w: None for w in bvocab},
                                      composer_kind="rf", defer_parser=True)
        for f in conv_facts:
            ba.composer.store(f["agent"], f["action"], f["patient"], polarity=f["polarity"])
        brain_dir = os.path.join(tmp, "brain")
        save_developed_brain(ba, brain_dir, seed=42, D=D, composer_kind="rf")
        agent, _mani = load_developed_brain(brain_dir, ltm_bundle=bundle, use_multiturn=False)
        inner = getattr(agent, "agent", agent)
        # a known AFFIRM curated fact (dedup guarantees an exact first-match ground truth)
        kf = None
        for f in facts:
            if (f.get("polarity") or "AFFIRM") == "AFFIRM" and fm.get((f["agent"], f["action"])) == f["patient"]:
                kf = f
                break
        kf = kf or facts[0]
        ltm_hit = inner.what_does(kf["agent"], kf["action"])
        buf_hit = inner.what_does("dog", "chase")
        moat = inner.what_does("__nobody_prod__", kf["action"])
        out.update({
            "available": True,
            "tiered_installed": type(inner.composer).__name__ == "TieredFactStore",
            "ltm_probe": [kf["agent"], kf["action"], kf["patient"]],
            "ltm_recall": repr(ltm_hit), "ltm_recall_ok": (ltm_hit == kf["patient"]),
            "buffer_recall": repr(buf_hit), "buffer_recall_ok": (buf_hit == "cat"),
            "moat_abstains": (moat is None),
            "GO": (type(inner.composer).__name__ == "TieredFactStore"
                   and ltm_hit == kf["patient"] and buf_hit == "cat" and moat is None),
        })
    except Exception as e:
        import traceback
        out["error"] = "production_load_path_check failed: %r\n%s" % (e, traceback.format_exc())
    finally:
        if tmp:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_core_15k")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--probe-cap", type=int, default=60)
    ap.add_argument("--moat-cap", type=int, default=40)
    ap.add_argument("--out", default="research/findings/raw/_knowledge_core/core_bundle_soak_verdict.json")
    a = ap.parse_args()

    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else MISSION_SEEDS
    if not os.path.exists(os.path.join(a.bundle, "facts.json")):
        print(f"[soak] bundle not found or has no facts.json: {a.bundle}", flush=True)
        return 1

    facts = _load_bundle_facts(a.bundle)
    vocab = _vocab_of(facts)
    fm = _first_match(facts)
    print(f"[soak] bundle={a.bundle} n_facts={len(facts)} vocab={len(vocab)}", flush=True)

    from research.runners.brain_conversational_agent import BrainConversationalAgent
    t0 = time.time()
    cells = []
    for s in seeds:
        agent = BrainConversationalAgent(seed=s, concepts={f"w{i}": None for i in range(8)},
                                         composer_kind="rf", defer_parser=True)
        c = structural_seed_check(facts, vocab, fm, s, a.probe_cap, a.moat_cap, agent)
        print(f"[soak] seed={s} GO={c['GO']} bi_mism={len(c['byte_identity_mismatches'])} "
              f"moat={c['moat_ok']}/{c['moat_checked']} recall={c['recall_rate']} "
              f"lat_med={c['latency_ms_median']}ms shards={c['n_shards']}", flush=True)
        cells.append(c)

    print("[soak] production load path check (load_developed_brain ltm_bundle=<bundle>) ...", flush=True)
    prod = production_load_path_check(a.bundle, facts, fm)
    print(f"[soak]   prod GO={prod.get('GO')} tiered={prod.get('tiered_installed')} "
          f"ltm_ok={prod.get('ltm_recall_ok')} buf_ok={prod.get('buffer_recall_ok')} "
          f"moat={prod.get('moat_abstains')}", flush=True)

    seeds_go = sum(1 for c in cells if c["GO"])
    total_mism = sum(len(c["byte_identity_mismatches"]) for c in cells)
    total_confab = sum(c["moat_confab"] for c in cells)
    go = bool(seeds_go == len(seeds) and total_mism == 0 and total_confab == 0 and prod.get("GO"))

    from tools.verdict import Verdict
    v = Verdict("knowledge core bundle soak: curated wikidata core is a no-regression BRAIN_LTM_BUNDLE (board #133)")
    v.require("6/6 seeds GO (byte-identity + moat + recall over real curated facts)", seeds_go, expect=len(seeds))
    v.require("byte-identity: 0 mismatches vs the plain-flat unsharded oracle", total_mism, expect=0)
    v.require("no-confab moat: 0 confabulations", total_confab, expect=0)
    v.require("production load path (load_developed_brain ltm_bundle) recall+buffer+moat GO",
              prod.get("GO"), expect=True)
    v.disabled("learned/spiking cue->shard router",
               "router hash(agent) mod S is a declared host scaffold; the in-shard FHRR recall + moat are the reads")
    decided = v.decide(go=go)

    verdict = dict(decided)
    verdict.update({
        "arc": "knowledge core bundle soak (board #133; curated wikidata core -> BRAIN_LTM_BUNDLE default)",
        "backend": os.environ.get("SIM_BACKEND", "?"), "bundle": a.bundle,
        "n_facts": len(facts), "vocab_size": len(vocab), "seeds": seeds,
        "seeds_go": seeds_go, "total_byte_identity_mismatches": total_mism, "total_moat_confab": total_confab,
        "cells": cells, "production_load_path": prod, "elapsed_s": round(time.time() - t0, 2),
    })
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print("\nwrote", a.out, "GO=", verdict.get("go"))
    return 0 if verdict.get("go") else 1


if __name__ == "__main__":
    sys.exit(main())
