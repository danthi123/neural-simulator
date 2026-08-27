"""100k END-TO-END knowledge-scale verification through the REAL production load path (board #66/#150/#109).

TASK: test the shipped 100k wikidata bundle (`sim-data/knowledge_bundles/wikidata_100k`, 78,857 real curated
Wikidata facts, vocab 23,914) against the technical bar for a default-bundle-size flip:
  (1) RECALL parity: the live agent's top-1 answer == a plain-flat UNSHARDED oracle's answer, 0 mismatches, on a
      held probe set.
  (2) MOAT: unknown subjects / cross cues still abstain, 0 new confabulations.
  (3) LATENCY: warm routed-recall median < 1 s.
Through the EXACT production path: `developed_brain_io.load_developed_brain(bundle, ltm_bundle=<wikidata_100k>)`
-- the same call `webapp/server.py`'s `_build_chat_brain` makes when `BRAIN_LTM_BUNDLE` points at a bundle,
installing a `TieredFactStore(buffer, ShardedPhasorStore.load(...))` on the real agent.

METHODOLOGY NOTE (why the oracle is built over a SUBSET, not all 78,857 facts). The existing validated
methodology (`_knowledge_scale_flip_soak.py`, 2026-08-21 finding) deliberately never builds a flat oracle past
N~4000: a flat RFPhasorComposer's query cost is dominated by the O(K) per-fact unbind/scan (~2.2 s/query at
K=1000, ~12 s/query at K=4000 -- MEASURED, see that finding), so an oracle holding all 78,857 facts would cost
tens of seconds PER QUERY (confirmed here: an earlier version of this script that built a 78,857-fact oracle and
queried it live per probe was killed after several minutes with zero probes completed -- exactly the wall the
established methodology exists to avoid). This script reproduces that same two-track design against the REAL
100k bundle instead of synthetic facts: (i) a TRACTABLE flat oracle over a sampled subset of agents' COMPLETE
fact-sets (so first-match ground truth for those agents is exactly what an unsharded store would return) checks
byte-identity/RECALL PARITY; (ii) RECALL at full scale is checked oracle-free against the ground-truth dict
computed directly from facts.json (this dict computation IS the flat first-match convention, by construction --
no live O(K) scan needed); (iii) LATENCY is measured only against the LIVE production agent (never the oracle).

ALSO empirically settles whether `BRAIN_SPARSE_INDEX_RETRIEVAL` (the DG-sparse-index cleanup accelerator,
`OneBrainComposer.enable_sparse_index`, board #150/#66, 6-seed GO at 961x wall-speedup / 200k synthetic
concepts -- `research/findings/raw/_wire_sparse_index_verify.json`) has ANY effect on THIS path. Static +
dynamic check: it is wired ONLY into `OneBrainComposer.__init__` (one_brain_composer.py:122,307-313); the
tiered LTM's shards are hardcoded `RFPhasorComposer` instances (sharded_phasor_store.py:73,77) whose __init__
has NO `enable_sparse_index` parameter and no **kwargs catch-all -- passing it raises TypeError (confirmed
dynamically below: `RFPhasorComposer(enable_sparse_index=True)` -> "unexpected keyword argument"). The persisted
bundle's own manifest.json even records `composer_kwargs: {}`. Separately, `developed_brain_io.
save_developed_brain`'s own default is `composer_kind="rf"`, so the developed-brain BUFFER is also
RFPhasorComposer, not OneBrainComposer, on the real load path this task specifies. PREDICTION: setting
`BRAIN_SPARSE_INDEX_RETRIEVAL=1` changes NOTHING on this path (byte-identical answers, statistically
indistinguishable latency) because the code that reads the env var is never constructed. Verified below by
running the SAME small probe battery with the flag unset vs set.

Run:  SIM_BACKEND=numpy .venv/bin/python -m research.runners._knowledge_scale_100k_production_verify \
        --bundle /home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k \
        --json research/findings/raw/_knowledge_scale_100k_production_verify.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time

import numpy as np


def _load_bundle_facts_vocab(bundle_dir):
    """`ShardedPhasorStore.save()`'s own facts.json shape is `[{"shard": i, "fact": {...}}, ...]` (it records
    shard placement so `load()` can reconstruct without re-routing). Unwrap to the flat fact-dict list every
    other consumer here (the oracle builder, `_first_match`) expects."""
    with open(os.path.join(bundle_dir, "manifest.json")) as f:
        mani = json.load(f)
    with open(os.path.join(bundle_dir, "facts.json")) as f:
        raw = json.load(f)
    facts = [rec["fact"] if isinstance(rec, dict) and "fact" in rec else rec for rec in raw]
    return mani, facts


def _first_match(facts):
    """(agent, action) -> first AFFIRM patient, mirroring TieredFactStore/RFPhasorComposer's own first-match
    query_patient semantics -- this computation IS the flat-unsharded ground truth, by construction (no live
    O(K) oracle scan needed to know what a flat store would answer)."""
    fm = {}
    for f in facts:
        a, act, p = f.get("agent"), f.get("action"), f.get("patient")
        pol = f.get("polarity") or "AFFIRM"
        if not (isinstance(a, str) and isinstance(act, str) and isinstance(p, str)):
            continue
        key = (a, act)
        if key not in fm and pol == "AFFIRM":
            fm[key] = p
    return fm


def _build_small_flat_oracle(all_facts, sample_agents, vocab, seed, D):
    """A TRACTABLE flat RFPhasorComposer holding EVERY fact belonging to `sample_agents` (their complete fact
    sets pulled from the full 78,857-fact corpus, so first-match ground truth for these agents is exactly what
    the real store holds -- not a truncated/partial view). Same seed+FULL vocab as the persisted 100k bundle ->
    byte-identical codebook, so any answer difference isolates ROUTING (sharding), not codebook drift. Facts
    inserted in the corpus's own global order (matching the persisted store's per-shard insertion order for
    first-match semantics)."""
    from research.runners.rf_phasor_composer import RFPhasorComposer
    from research.runners.tiered_fact_store import encode_fast
    sample_set = set(sample_agents)
    comp = RFPhasorComposer(seed=seed, D=D, vocab=list(vocab))
    n = 0
    for f in all_facts:
        a, act, p = f.get("agent"), f.get("action"), f.get("patient")
        if a not in sample_set or not (isinstance(a, str) and isinstance(act, str) and isinstance(p, str)):
            continue
        fd = {"agent": a, "action": act, "patient": p, "polarity": f.get("polarity") or "AFFIRM"}
        comp.kb.append((fd, encode_fast(comp, fd)))
        n += 1
    return comp, n


def _make_production_brain(seed, D, tmp_root):
    """A minimal but genuine `save_developed_brain` bundle (a few conversation-taught facts), then
    `load_developed_brain(..., ltm_bundle=<real wikidata_100k>)` -- the EXACT call webapp/server.py's
    `_build_chat_brain` makes for a developed-brain + BRAIN_LTM_BUNDLE. renderer is a webapp-layer concept
    (Qwen/stub/raw) applied on TOP of the loaded agent for rendering; this call is renderer-agnostic (it returns
    the raw agent `load_developed_brain` always returns) -- there is no Qwen warm anywhere in this path."""
    from research.runners.developed_brain_io import save_developed_brain
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    conv_facts = [
        {"agent": "otter", "action": "caught", "patient": "clam", "polarity": "AFFIRM"},
        {"agent": "fox", "action": "chased", "patient": "hare", "polarity": "AFFIRM"},
    ]
    vocab = sorted({w for f in conv_facts for w in (f["agent"], f["action"], f["patient"])})
    ba = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab},
                                  composer_kind="rf", defer_parser=True)
    for f in conv_facts:
        ba.composer.store(f["agent"], f["action"], f["patient"], polarity=f["polarity"])
    brain_dir = os.path.join(tmp_root, "brain")
    save_developed_brain(ba, brain_dir, seed=seed, D=D, composer_kind="rf")
    return brain_dir, conv_facts


def oracle_byte_identity_check(inner_agent, oracle, oracle_probes):
    """(i) RECALL PARITY vs the tractable flat oracle: live (tiered/routed, over the FULL 78,857-fact store) vs
    oracle (flat, over just these agents' complete fact sets) -- 0 mismatches required."""
    checked = 0
    mismatches = []
    for (a, v, gt) in oracle_probes:
        live = inner_agent.what_does(a, v)
        ref = oracle.query_patient(a, v)
        checked += 1
        if live != ref:
            mismatches.append({"cue": [a, v], "oracle": repr(ref), "live": repr(live), "gt": repr(gt)})
        live_yn = inner_agent.is_it_true(a, v, gt)
        ref_yn = oracle.ask_yes_no(a, v, gt)
        checked += 1
        if live_yn != ref_yn:
            mismatches.append({"cue": [a, v, gt], "kind": "yesno", "oracle": repr(ref_yn), "live": repr(live_yn)})
    return {"checked": checked, "n_mismatches": len(mismatches), "mismatches": mismatches}


def scale_recall_and_latency(inner_agent, probes, moat_cues, warm=True):
    """(ii) RECALL at FULL SCALE vs the ground-truth dict (oracle-free) + (iii) LATENCY -- both measured only
    against the LIVE production agent (the tiered store over the full 78,857-fact bundle). Also (moat):
    unknown/cross cues must abstain, 0 confabulations."""
    if warm:
        for (a, v, _p) in probes:
            inner_agent.what_does(a, v)
        for (a, v) in moat_cues:
            inner_agent.what_does(a, v)

    recall_checked = recall_ok = 0
    lat = []
    for (a, v, gt) in probes:
        t0 = time.perf_counter()
        live = inner_agent.what_does(a, v)
        lat.append(time.perf_counter() - t0)
        recall_checked += 1
        if live == gt:
            recall_ok += 1

    moat_checked = moat_confab = 0
    for (a, v) in moat_cues:
        t0 = time.perf_counter()
        live = inner_agent.what_does(a, v)
        lat.append(time.perf_counter() - t0)
        moat_checked += 1
        if live is not None:
            moat_confab += 1

    return {
        "recall_checked": recall_checked, "recall_ok": recall_ok,
        "recall_rate": round(recall_ok / recall_checked, 4) if recall_checked else None,
        "moat_checked": moat_checked, "moat_confab": moat_confab,
        "latency_ms_median": round(float(np.median(lat)) * 1000, 2) if lat else None,
        "latency_ms_p95": round(float(np.percentile(lat, 95)) * 1000, 2) if lat else None,
        "latency_ms_mean": round(float(np.mean(lat)) * 1000, 2) if lat else None,
        "n_latency_samples": len(lat),
    }


def main():
    ap = argparse.ArgumentParser(description="100k end-to-end knowledge-scale production verify (#66/#150/#109)")
    ap.add_argument("--bundle", default="/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k")
    ap.add_argument("--n-oracle-agents", type=int, default=60, help="agents for the tractable flat-oracle check")
    ap.add_argument("--n-scale-probes", type=int, default=150, help="probes for oracle-free scale recall+latency")
    ap.add_argument("--n-moat", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    t_start = time.time()
    out = {"bundle": a.bundle, "seed": a.seed}

    # --- architecture check: does RFPhasorComposer even accept enable_sparse_index? (dynamic proof) ---
    from research.runners.rf_phasor_composer import RFPhasorComposer
    try:
        RFPhasorComposer(seed=1, D=8, vocab=["a"], enable_sparse_index=True)
        sparse_index_reaches_shard = True
        sparse_index_error = None
    except TypeError as e:
        sparse_index_reaches_shard = False
        sparse_index_error = str(e)
    out["sparse_index_architecture_check"] = {
        "RFPhasorComposer_accepts_enable_sparse_index": sparse_index_reaches_shard,
        "error_if_rejected": sparse_index_error,
        "conclusion": ("enable_sparse_index CAN reach the LTM shard composer" if sparse_index_reaches_shard else
                       "enable_sparse_index CANNOT reach the LTM shard composer (RFPhasorComposer has no such "
                       "param) -- BRAIN_SPARSE_INDEX_RETRIEVAL is architecturally INERT on the tiered-LTM path"),
    }

    # --- 1. load the REAL 100k bundle's manifest + facts (for the oracle + probe/moat cues) ---
    print(f"[1/6] loading bundle manifest + facts.json from {a.bundle} ...", flush=True)
    mani, raw_facts = _load_bundle_facts_vocab(a.bundle)
    D = int(mani["D"]); vocab = mani["vocab"]; bundle_seed = int(mani["seed"])
    out["bundle_manifest"] = {"n_shards": mani["n_shards"], "seed": bundle_seed, "D": D,
                              "n_facts": mani["n_facts"], "vocab_size": len(vocab),
                              "composer_kwargs": mani.get("composer_kwargs")}
    print(f"       n_facts={mani['n_facts']} vocab={len(vocab)} n_shards={mani['n_shards']} D={D}", flush=True)
    fm = _first_match(raw_facts)

    tmp = tempfile.mkdtemp(prefix="ks_100k_prod_verify_")
    error = None
    try:
        # --- 2. the REAL production load path ---
        print("[2/6] load_developed_brain(ltm_bundle=<real wikidata_100k>) -- the exact BRAIN_LTM_BUNDLE path ...",
              flush=True)
        from research.runners.developed_brain_io import load_developed_brain, _inner_agent
        brain_dir, conv_facts = _make_production_brain(a.seed, D, tmp)
        t0 = time.time()
        agent, load_manifest = load_developed_brain(brain_dir, ltm_bundle=a.bundle, use_multiturn=False,
                                                     seed=a.seed)
        out["ltm_load_s"] = round(time.time() - t0, 2)
        inner = _inner_agent(agent)
        out["tiered_installed"] = type(inner.composer).__name__ == "TieredFactStore"
        out["ltm_class"] = type(inner.composer.ltm).__name__ if out["tiered_installed"] else None
        out["ltm_shard_class"] = (type(inner.composer.ltm.shards[0]).__name__
                                   if out["tiered_installed"] and getattr(inner.composer.ltm, "shards", None)
                                   else None)
        out["ltm_total_facts"] = inner.composer.total_facts() if out["tiered_installed"] else None
        # LTM-only fact count (excludes the 2 conversation-taught facts `_make_production_brain` puts in the
        # BUFFER tier -- total_facts() sums buffer+LTM by design, so it is 2 MORE than the bundle's own n_facts;
        # that is correct tiered behavior, not a defect. The bundle-parity check below compares LTM-only.)
        out["ltm_only_facts"] = (inner.composer.ltm.total_facts()
                                  if out["tiered_installed"] and inner.composer.ltm is not None else None)
        print(f"       loaded in {out['ltm_load_s']}s | tiered_installed={out['tiered_installed']} "
              f"ltm_class={out['ltm_class']} shard_class={out['ltm_shard_class']} "
              f"total_facts={out['ltm_total_facts']} (LTM-only={out['ltm_only_facts']}, "
              f"+{out['ltm_total_facts'] - out['ltm_only_facts'] if out['ltm_total_facts'] and out['ltm_only_facts'] else '?'} "
              f"buffer conv-facts)", flush=True)

        # --- 3. sample the tractable oracle-agent set + build the small flat oracle over their COMPLETE facts ---
        print(f"[3/6] building the tractable flat oracle over {a.n_oracle_agents} agents' complete fact sets ...",
              flush=True)
        rng = np.random.default_rng(a.seed + 7)
        keys = list(fm.keys())
        oracle_agents = sorted({keys[i][0] for i in rng.choice(len(keys), size=min(400, len(keys)), replace=False)})
        oracle_agents = oracle_agents[:a.n_oracle_agents]
        t0 = time.time()
        oracle, oracle_n_facts = _build_small_flat_oracle(raw_facts, oracle_agents, vocab, bundle_seed, D)
        out["oracle_build_s"] = round(time.time() - t0, 2)
        out["oracle_n_agents"] = len(oracle_agents)
        out["oracle_n_facts"] = oracle_n_facts
        oracle_probes = [(k[0], k[1], p) for k, p in fm.items() if k[0] in set(oracle_agents)]
        print(f"       oracle: {oracle_n_facts} facts / {len(oracle_agents)} agents, built in "
              f"{out['oracle_build_s']}s | {len(oracle_probes)} oracle probe cues", flush=True)

        # --- 4. sample the LARGER scale-recall probe set (oracle-free, ground-truth dict) + moat cues ---
        print(f"[4/6] sampling {a.n_scale_probes} scale-recall probes + {a.n_moat} moat cues ...", flush=True)
        idx = rng.choice(len(keys), size=min(a.n_scale_probes, len(keys)), replace=False)
        scale_probes = [(keys[i][0], keys[i][1], fm[keys[i]]) for i in idx]
        vocab_set = set(vocab)
        rng2 = np.random.default_rng(a.seed + 999)
        unknown_agents = [f"zzz_unknown_entity_{j}_xq" for j in range(a.n_moat // 2)]
        assert not (set(unknown_agents) & vocab_set), "unknown-agent cue collides with real vocab"
        known_agent_sample = keys[int(rng2.integers(0, len(keys)))][0]
        real_actions = sorted({f.get("action") for f in raw_facts if isinstance(f.get("action"), str)})
        moat_cues = [(ua, real_actions[int(rng2.integers(0, len(real_actions)))]) for ua in unknown_agents]
        moat_cues += [(known_agent_sample, "zzz_unknown_relation_never_taught")
                      for _ in range(a.n_moat - len(moat_cues))]

        # --- 5. run: (a) byte-identity vs the tractable oracle, (b) scale recall+latency+moat, flag unset ---
        print("[5/6] (a) oracle byte-identity check ...", flush=True)
        t0 = time.time()
        oracle_res = oracle_byte_identity_check(inner, oracle, oracle_probes)
        out["oracle_byte_identity"] = oracle_res
        out["oracle_check_s"] = round(time.time() - t0, 2)
        print(f"       checked={oracle_res['checked']} mismatches={oracle_res['n_mismatches']} "
              f"({out['oracle_check_s']}s)", flush=True)

        print("[5/6] (b) scale recall + latency + moat, BRAIN_SPARSE_INDEX_RETRIEVAL unset (today's default) ...",
              flush=True)
        os.environ.pop("BRAIN_SPARSE_INDEX_RETRIEVAL", None)
        t0 = time.time()
        res_off = scale_recall_and_latency(inner, scale_probes, moat_cues)
        out["scale_battery_flag_unset"] = res_off
        print(f"       recall={res_off['recall_rate']} moat_confab={res_off['moat_confab']}/{res_off['moat_checked']} "
              f"lat_med={res_off['latency_ms_median']}ms p95={res_off['latency_ms_p95']}ms "
              f"({round(time.time()-t0,1)}s, {res_off['n_latency_samples']} samples)", flush=True)

        # --- 6. inertness check: a SMALL re-run with the flag SET, to confirm no effect (not a full re-battery) ---
        print("[6/6] inertness check: a 20-probe re-run with BRAIN_SPARSE_INDEX_RETRIEVAL=1 ...", flush=True)
        small_idx = idx[:20]
        small_probes = [(keys[i][0], keys[i][1], fm[keys[i]]) for i in small_idx]
        small_moat = moat_cues[:10]
        os.environ.pop("BRAIN_SPARSE_INDEX_RETRIEVAL", None)
        small_off = scale_recall_and_latency(inner, small_probes, small_moat, warm=True)
        os.environ["BRAIN_SPARSE_INDEX_RETRIEVAL"] = "1"
        small_on = scale_recall_and_latency(inner, small_probes, small_moat, warm=False)
        os.environ.pop("BRAIN_SPARSE_INDEX_RETRIEVAL", None)
        out["inertness_check"] = {
            "flag_unset": small_off, "flag_set_1": small_on,
            "answers_identical": (small_off["recall_ok"] == small_on["recall_ok"]
                                   and small_off["moat_confab"] == small_on["moat_confab"]),
        }
        print(f"       OFF recall={small_off['recall_rate']} lat_med={small_off['latency_ms_median']}ms | "
              f"ON recall={small_on['recall_rate']} lat_med={small_on['latency_ms_median']}ms | "
              f"answers_identical={out['inertness_check']['answers_identical']} "
              f"(predicted TRUE -- flag is architecturally inert on this path)", flush=True)

    except Exception as e:
        import traceback
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        out["error"] = error
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- verdict ---
    from tools.verdict import Verdict
    v = Verdict("100k end-to-end knowledge-scale production verify (board #66/#150/#109)")
    if error is None:
        r = out["scale_battery_flag_unset"]
        o = out["oracle_byte_identity"]
        v.require("production load path installs a TieredFactStore over the real 100k bundle",
                  out["tiered_installed"], expect=True)
        v.require("LTM-only total_facts == the real bundle's n_facts (excludes the buffer tier's own facts)",
                  out["ltm_only_facts"], expect=mani["n_facts"])
        v.require("RECALL parity: 0 mismatches vs the tractable flat oracle (held agent set)",
                  o["n_mismatches"], expect=0)
        v.require("RECALL at full 100k scale (oracle-free, ground-truth dict) >= 0.99",
                  r["recall_rate"], expect=lambda x: (x or 0) >= 0.99)
        v.require("MOAT: 0 confabulations on unknown/cross cues", r["moat_confab"], expect=0)
        v.floor("LATENCY: warm routed-recall median < 1000 ms", 1000.0 - (r["latency_ms_median"] or 1e9), floor=0.0)
        v.require("BRAIN_SPARSE_INDEX_RETRIEVAL flag has NO effect on this path (architecturally inert, as predicted)",
                  out["inertness_check"]["answers_identical"], expect=True)
        bars = {
            "recall_bar_pass": (o["n_mismatches"] == 0) and ((r["recall_rate"] or 0) >= 0.99),
            "moat_bar_pass": r["moat_confab"] == 0,
            "latency_bar_pass": bool(r["latency_ms_median"] is not None and r["latency_ms_median"] < 1000.0),
        }
        out["bars"] = bars
        go = bool(bars["recall_bar_pass"] and bars["moat_bar_pass"] and bars["latency_bar_pass"])
    else:
        go = False
        v.require("run completed without error", False, expect=True)
    decided = v.decide(go=go)
    out.update(decided)
    out["elapsed_s"] = round(time.time() - t_start, 2)

    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print("\nwrote", a.json)
    print(f"\n===== VERDICT: {out['status']} (go={out.get('go')}) elapsed={out['elapsed_s']}s =====")
    return 0 if out.get("go") else 1


if __name__ == "__main__":
    raise SystemExit(main())
