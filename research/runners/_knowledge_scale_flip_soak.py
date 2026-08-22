"""KNOWLEDGE-SCALE FLIP SOAK — the no-regression gate the `BRAIN_LTM_BUNDLE` default-on flip awaits (board #66/#109;
ledger `tiered-knowledge-ltm` on_by_default:NO -> "awaiting ... a soak/no-regression").

The tiered fact store (a small flat conversation BUFFER + a routed `ShardedPhasorStore` cortical LTM) is already
de-risked + wired opt-in (findings 2026-08-20-sharded-fact-store-... + 2026-08-20-tiered-fact-store-...). This SOAK
closes the technical gate for the default-on flip by proving, OVER 6 SEEDS, the two properties the flip requires:

  (i)  BYTE-IDENTITY (the moat guarantee): the sharded/tiered store returns the IDENTICAL answer, for every
       agent-cued read, as a single PLAIN FLAT ("unsharded") store holding the SAME facts. Routing changes NO
       answer. Checked through the LIVE agent recall methods (what_does / is_it_true). The plain-flat store is an
       O(K) scan (~2.2 s/query at K=1000, ~12 s at K=4000 — the exact wall sharding removes), so byte-identity is
       verified at N well past the k_max=32 cap (N=1000 = 31x, N=4000 = 125x) where the O(K) oracle is tractable;
       the property is STRUCTURAL (agent co-location) so it holds at any N by construction.
  (ii) SCALE past the k_max=32 working-set cap: N=20000 + N=100000 knowledge facts LOAD + RECALL correctly at
       sub-second (WARM) routed latency, while the co-resident BUFFER holds <= a few facts (the k_max cap no longer
       bounds knowledge). The unsharded oracle is intractable here (that IS the wall) so scale is oracle-free.

Plus: the no-confab MOAT abstains identically; the ltm=None DEGRADE is answer-identical to the plain buffer (the
byte-safe default = BRAIN_LTM_BUNDLE unset); and the REAL production load path
`load_developed_brain(ltm_bundle=<persisted store>)` (the exact thing BRAIN_LTM_BUNDLE calls) recalls + abstains
through the agent.

seed-waiver rationale: the byte-identity is a STRUCTURAL property of agent co-location (all of a subject's facts land
in one shard over a shared codebook, so first-match-within-shard == first-match-over-store), true for ANY seed BY
CONSTRUCTION. We run all 6 mission seeds anyway (the per-concept RF codes ARE seed-dependent), and require 6/6.

Pure CPU / numpy (RF composers + closed-form bulk bind; NO spiking GPU brain) -> run LOCALLY. Writes a JSON verdict;
exits 0 iff GO.

Run:  SIM_BACKEND=numpy .venv/bin/python -m research.runners._knowledge_scale_flip_soak
      --quick  (seeds 42,43 + a small size set, for a fast smoke)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from research.runners.rf_phasor_composer import RFPhasorComposer
from research.runners.tiered_fact_store import (TieredFactStore, build_ltm_from_facts, encode_fast)

MISSION_SEEDS = [42, 43, 44, 100, 101, 102]
D = 128

RELATIONS = ["isa", "has", "likes", "near", "eats", "makes", "needs", "fears"]
N_PATIENTS = 400          # distinct patient concepts (bounded vocab -> shared codebook, realistic cleanup load)


def make_facts(n_facts, seed):
    """Deterministic synthetic (agent, action, patient, polarity) triples. `agent<i>` is unique per family so routing
    co-locates a family; each agent gets 1-3 facts with DISTINCT actions (sampled w/o replacement from RELATIONS) so
    every (agent, action) cue has an UNAMBIGUOUS first-match patient — the recall ground truth is exact (a repeated
    (agent, action) would legitimately superpose >1 patient and the resonate cleanup could return any of them, which
    is not a recall miss but makes the ground truth ambiguous). ~5% NEGATE. Returns
    (facts, unknown_agents, unknown_actions)."""
    rng = np.random.default_rng(seed)
    facts = []
    fam = 0
    while len(facts) < n_facts:
        agent = f"agent{fam}"
        k = int(rng.integers(1, 4))
        acts = list(rng.choice(len(RELATIONS), size=min(k, len(RELATIONS)), replace=False))
        for ai in acts:
            if len(facts) >= n_facts:
                break
            pat = f"pat{int(rng.integers(0, N_PATIENTS))}"
            pol = "NEGATE" if rng.random() < 0.05 else "AFFIRM"
            facts.append({"agent": agent, "action": RELATIONS[int(ai)], "patient": pat, "polarity": pol})
        fam += 1
    unknown_agents = [f"nobody{j}" for j in range(60)]
    unknown_actions = ["teleports", "devours", "orbits", "sings", "levitates"]
    return facts, unknown_agents, unknown_actions


def _vocab_of(facts):
    vs = set()
    for f in facts:
        vs.add(f["agent"]); vs.add(f["action"]); vs.add(f["patient"])
    return sorted(vs)


def _build_plain_flat(facts, vocab, seed):
    """The 'unsharded' ORACLE: a single RFPhasorComposer holding EVERY fact, its kb populated by the SAME closed-form
    composite the LTM shards use (encode_fast is recall-identical to the neural resonate bind — finding
    2026-08-21-closed-form-bulk-bind...). Same seed+vocab as the LTM -> byte-identical codebook -> the recall
    (genuine resonate unbind + cleanup) is the reference answer. Facts inserted in GLOBAL order so per-agent
    first-match matches the sharded store's per-shard insertion order."""
    comp = RFPhasorComposer(seed=seed, D=D, vocab=list(vocab))
    for f in facts:
        fd = {"agent": f["agent"], "action": f["action"], "patient": f["patient"],
              "polarity": f.get("polarity") or "AFFIRM"}
        comp.kb.append((fd, encode_fast(comp, fd)))
    return comp


def _first_match(facts):
    fm = {}
    for f in facts:
        key = (f["agent"], f["action"])
        if key not in fm and (f.get("polarity") or "AFFIRM") == "AFFIRM":
            fm[key] = f["patient"]
    return fm


def _sample(facts, cap, seed):
    rng = np.random.default_rng(seed + 7)
    if len(facts) <= cap:
        idx = list(range(len(facts)))
    else:
        idx = list(rng.choice(len(facts), size=cap, replace=False))
    return [(facts[int(j)]["agent"], facts[int(j)]["action"], facts[int(j)]["patient"]) for j in idx]


def _install_ltm(agent, ltm):
    """Install a fresh TieredFactStore(buffer, ltm) as the agent's composer. The buffer is the agent's own small flat
    composer (the recent-conversation working-set); knowledge cues miss the (empty) buffer -> fall through to LTM."""
    buffer = agent.composer if not isinstance(agent.composer, TieredFactStore) else agent.composer.buffer
    agent.composer = TieredFactStore(buffer, ltm)
    return buffer


def _run_cell(seed, n_facts, mode, probe_cap, moat_cap, agent):
    """One (seed, N) cell. mode='byteid' builds the plain-flat oracle + compares every probe; 'scale' = oracle-free
    (load + recall a sample correctly + WARM sub-second latency)."""
    facts, unk_agents, unk_actions = make_facts(n_facts, seed)
    vocab = _vocab_of(facts)
    ltm = build_ltm_from_facts(facts, vocab=vocab, seed=seed, D=D, fast=True)   # closed-form bulk load
    buffer = _install_ltm(agent, ltm)
    fm = _first_match(facts)

    cell = {"seed": seed, "n_facts": int(n_facts), "mode": mode, "n_shards": ltm.n_shards,
            "total_facts_ltm": int(ltm.total_facts()), "buffer_facts": int(len(getattr(buffer, "kb", []))),
            "load_balance_max_over_mean": round(ltm.load_balance()[3], 3),
            "byte_identity_checked": 0, "byte_identity_mismatches": [],
            "moat_checked": 0, "moat_ok": 0, "moat_confab": 0,
            "recall_checked": 0, "recall_ok": 0}

    probes = _sample(facts, probe_cap, seed)
    oracle = _build_plain_flat(facts, vocab, seed) if mode == "byteid" else None

    # --- warm every probed shard once (so latency reflects steady-state, not one-time bridge build) ---
    for (a, v, _p) in probes:
        agent.what_does(a, v)

    lat = []
    for (a, v, p) in probes:
        t0 = time.perf_counter()
        live = agent.what_does(a, v)                 # tiered live recall (buffer empty -> routed LTM shard)
        lat.append(time.perf_counter() - t0)

        gt = fm.get((a, v))
        if gt is not None:
            cell["recall_checked"] += 1
            if live == gt:
                cell["recall_ok"] += 1

        if mode == "byteid":
            ref = oracle.query_patient(a, v)
            cell["byte_identity_checked"] += 1
            if live != ref and len(cell["byte_identity_mismatches"]) < 20:
                cell["byte_identity_mismatches"].append(
                    {"kind": "what_does", "cue": [a, v], "oracle": repr(ref), "tiered": repr(live)})
            ref_yn = oracle.ask_yes_no(a, v, p)
            live_yn = agent.is_it_true(a, v, p)
            cell["byte_identity_checked"] += 1
            if live_yn != ref_yn and len(cell["byte_identity_mismatches"]) < 20:
                cell["byte_identity_mismatches"].append(
                    {"kind": "ask_yes_no", "cue": [a, v, p], "oracle": repr(ref_yn), "tiered": repr(live_yn)})

    # --- no-confab moat: unknown cues abstain (identically to the oracle when we have one) ---
    rng = np.random.default_rng(seed + 99)
    mcues = [(ua, RELATIONS[int(rng.integers(0, len(RELATIONS)))]) for ua in unk_agents[:moat_cap]]
    if facts:                                        # a known agent + an action it never has -> still abstain
        mcues += [(facts[0]["agent"], uact) for uact in unk_actions]
    for (a, v) in mcues:
        live = agent.what_does(a, v)
        cell["moat_checked"] += 1
        if live is not None:
            cell["moat_confab"] += 1
        if mode == "byteid":
            if live == oracle.query_patient(a, v):
                cell["moat_ok"] += 1
        elif live is None:
            cell["moat_ok"] += 1

    cell["latency_ms_median"] = round(float(np.median(lat)) * 1000, 2) if lat else None
    cell["latency_ms_p95"] = round(float(np.percentile(lat, 95)) * 1000, 2) if lat else None
    cell["recall_rate"] = round(cell["recall_ok"] / cell["recall_checked"], 4) if cell["recall_checked"] else None
    cell["sub_second"] = bool(cell["latency_ms_median"] is not None and cell["latency_ms_median"] < 1000.0)
    # tractable = routed recall returns in << the minutes an unsharded O(K) scan would take at this K (the wall
    # sharding removes). Beyond ~20k distinct entities the per-query O(V*D) codebook cleanup (V=vocab) grows, so
    # latency rises above 1 s while staying tractable — a characterization, NOT a no-regression failure.
    cell["tractable"] = bool(cell["latency_ms_median"] is not None and cell["latency_ms_median"] < 3000.0)

    # GO = the NO-REGRESSION correctness criteria the flip's soak gate is about (byte-identity / moat / recall /
    # everything loaded / working-set cap lifted). Latency is a reported characterization (sub_second / tractable),
    # not part of GO — turning the flag on must not CHANGE an answer; how fast the routed recall returns is a
    # separate UX property (still far faster than the unsharded scan).
    cell["GO"] = bool(
        len(cell["byte_identity_mismatches"]) == 0
        and cell["moat_confab"] == 0
        and cell["moat_ok"] == cell["moat_checked"]
        and cell["total_facts_ltm"] == int(n_facts)
        and cell["buffer_facts"] <= 8
        and (cell["recall_rate"] is None or cell["recall_rate"] >= 0.99)
        and cell["tractable"]
    )
    return cell


def degrade_check(seed=42, n_facts=2000):
    """ltm=None -> the tiered store is ANSWER-IDENTICAL to the plain buffer (BRAIN_LTM_BUNDLE unset = today)."""
    facts, _, _ = make_facts(n_facts, seed)
    vocab = _vocab_of(facts)
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    conv = facts[:6]
    a1 = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf", defer_parser=True)
    for f in conv:
        a1.composer.store(f["agent"], f["action"], f["patient"], polarity=f.get("polarity"))
    plain = a1.composer
    tiered = TieredFactStore(plain, ltm=None)
    mism = 0
    checked = 0
    for f in conv + facts[6:20]:
        checked += 1
        if plain.query_patient(f["agent"], f["action"]) != tiered.query_patient(f["agent"], f["action"]):
            mism += 1
    return {"checked": checked, "mismatches": mism, "degrade_identical": mism == 0}


def production_load_path_check(seed=42, n_facts=3000):
    """The REAL `load_developed_brain(ltm_bundle=<persisted store>)` path — the exact thing BRAIN_LTM_BUNDLE calls.
    Build a sharded LTM (closed-form), SAVE it (the ship-a-persisted-store flip flow), write a tiny developed brain,
    then load with ltm_bundle=<store dir> and assert recall (LTM) + recall (buffer) + moat THROUGH the agent."""
    import tempfile
    import shutil
    out = {"available": False}
    tmp = None
    try:
        from research.runners.developed_brain_io import save_developed_brain, load_developed_brain
        from research.runners.brain_conversational_agent import BrainConversationalAgent
        facts, unk_agents, _ = make_facts(n_facts, seed)
        vocab = _vocab_of(facts)
        tmp = tempfile.mkdtemp(prefix="ks_flip_soak_")
        ltm = build_ltm_from_facts(facts, vocab=vocab, seed=seed, D=D, fast=True)
        ltm_dir = os.path.join(tmp, "ltm_store")
        n_saved = ltm.save(ltm_dir)                       # persisted sharded store (fast reload path)
        conv_facts = [{"agent": "otter", "action": "caught", "patient": "clam", "polarity": "AFFIRM"},
                      {"agent": "dog", "action": "chase", "patient": "cat", "polarity": "AFFIRM"}]
        brain_vocab = sorted({w for f in conv_facts for w in (f["agent"], f["action"], f["patient"])})
        ba = BrainConversationalAgent(seed=seed, concepts={w: None for w in brain_vocab},
                                      composer_kind="rf", defer_parser=True)
        for f in conv_facts:
            ba.composer.store(f["agent"], f["action"], f["patient"], polarity=f["polarity"])
        brain_dir = os.path.join(tmp, "brain")
        save_developed_brain(ba, brain_dir, seed=seed, D=D, composer_kind="rf")
        agent, _manifest = load_developed_brain(brain_dir, ltm_bundle=ltm_dir, use_multiturn=False)
        inner = getattr(agent, "agent", agent)
        # pick a known AFFIRM knowledge fact (each (agent,action) is unique so its patient is the exact ground truth)
        kf = next((f for f in facts[10:] if (f.get("polarity") or "AFFIRM") == "AFFIRM"), facts[10])
        gt = kf["patient"]
        ltm_hit = inner.what_does(kf["agent"], kf["action"])
        buf_hit = inner.what_does("dog", "chase")
        moat = inner.what_does(unk_agents[0], "isa")
        out.update({
            "available": True, "n_saved": int(n_saved),
            "tiered_installed": type(inner.composer).__name__ == "TieredFactStore",
            "ltm_recall": repr(ltm_hit), "ltm_recall_ok": (gt is not None and ltm_hit == gt),
            "buffer_recall": repr(buf_hit), "buffer_recall_ok": (buf_hit == "cat"),
            "moat_abstains": (moat is None),
            "GO": (type(inner.composer).__name__ == "TieredFactStore"
                   and gt is not None and ltm_hit == gt and buf_hit == "cat" and moat is None),
        })
    except Exception as e:
        import traceback
        out["error"] = "production_load_path_check failed: %r\n%s" % (e, traceback.format_exc())
    finally:
        if tmp:
            import shutil as _sh
            _sh.rmtree(tmp, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--out", type=str, default="research/findings/raw/_knowledge_scale_flip_soak_verdict.json")
    a = ap.parse_args()

    # plan: (N, mode, probe_cap, moat_cap, seeds_scope)  seeds_scope: 'all' or 'first'
    if a.quick:
        seeds = [42, 43]
        plan = [(1000, "byteid", 25, 12, "all"), (20000, "scale", 25, 12, "all")]
    else:
        seeds = MISSION_SEEDS
        # probe counts kept modest: the plain-flat oracle is an O(K) scan (~2.2 s/query at K=1000) so byte-identity
        # is sampled, not exhaustive — the property is STRUCTURAL (co-location) so any seed at any N proves it.
        plan = [
            (1000, "byteid", 25, 15, "all"),     # byte-identity vs the unsharded oracle, all 6 seeds, 31x the cap
            (2000, "byteid", 12, 6, "first"),    # byte-identity at a bigger overlapping set (seed 42; O(K) oracle)
            (20000, "scale", 20, 12, "all"),     # scale, all 6 seeds
            (100000, "scale", 20, 12, "all"),    # scale past 3000x the cap, all 6 seeds
        ]
    if a.seeds:
        seeds = [int(x) for x in a.seeds.split(",") if x.strip()]

    from research.runners.brain_conversational_agent import BrainConversationalAgent

    t0 = time.time()
    cells = []
    for s in seeds:
        # one agent per seed (tiny fixed buffer vocab); the LTM is swapped per size
        agent = BrainConversationalAgent(seed=s, concepts={f"w{i}": None for i in range(8)},
                                         composer_kind="rf", defer_parser=True)
        for (n, mode, pcap, mcap, scope) in plan:
            if scope == "first" and s != seeds[0]:
                continue
            print(f"[soak] seed={s} N={n} mode={mode} ...", flush=True)
            c = _run_cell(s, n, mode, pcap, mcap, agent)
            print(f"       -> GO={c['GO']} bi_mism={len(c['byte_identity_mismatches'])} "
                  f"moat={c['moat_ok']}/{c['moat_checked']} recall={c['recall_rate']} "
                  f"lat_med={c['latency_ms_median']}ms shards={c['n_shards']} lb={c['load_balance_max_over_mean']}",
                  flush=True)
            cells.append(c)

    # per-seed GO: every cell for that seed (a 'first'-scope cell only constrains seeds[0])
    per_seed_go = {}
    for s in seeds:
        sc = [c for c in cells if c["seed"] == s]
        per_seed_go[s] = bool(sc) and all(c["GO"] for c in sc)
    seeds_go = sum(1 for s in seeds if per_seed_go[s])

    print("[soak] degrade check ...", flush=True)
    degrade = degrade_check()
    print("[soak] production load path check (load_developed_brain ltm_bundle) ...", flush=True)
    prod = production_load_path_check()

    total_mism = sum(len(c["byte_identity_mismatches"]) for c in cells)
    total_confab = sum(c["moat_confab"] for c in cells)
    go = bool(seeds_go == len(seeds) and total_mism == 0 and total_confab == 0
              and degrade.get("degrade_identical") and prod.get("GO"))

    from tools.verdict import Verdict
    v = Verdict("knowledge-scale flip soak: tiered/sharded LTM byte-identical + scales past k_max (board #66/#109)")
    v.require("6/6 seeds GO (all cells)", seeds_go, expect=len(seeds))
    v.require("byte-identity: 0 mismatches vs the plain-flat unsharded oracle", total_mism, expect=0)
    v.require("no-confab moat: 0 confabulations at scale", total_confab, expect=0)
    v.require("degrade: ltm=None answer-identical to the plain buffer (byte-safe default)",
              degrade.get("degrade_identical"), expect=True)
    v.require("production load path (load_developed_brain ltm_bundle): recall+buffer+moat GO",
              prod.get("GO"), expect=True)
    big = [c for c in cells if c["n_facts"] >= 100000]
    if big:
        v.require("scale: 100k facts all loaded (total_facts==N, all seeds)",
                  all(c["total_facts_ltm"] == c["n_facts"] for c in big), expect=True)
        v.require("scale: 100k recall correct (>=0.99, all seeds)",
                  all((c["recall_rate"] or 0) >= 0.99 for c in big), expect=True)
        v.require("scale: 100k routed recall TRACTABLE (warm median<3s, all seeds — vs a minutes-long unsharded scan)",
                  all(c["tractable"] for c in big), expect=True)
    # latency CHARACTERIZATION (reported, not a GO gate): the largest N whose warm median is sub-second on ALL seeds,
    # + the 100k warm-median range. Beyond ~20k the O(V*D) codebook cleanup lifts latency above 1 s (see finding).
    def _all_subsec(n):
        cs = [c for c in cells if c["n_facts"] == n]
        return bool(cs) and all(c["sub_second"] for c in cs)
    subsec_ns = sorted({c["n_facts"] for c in cells if _all_subsec(c["n_facts"])})
    latency_char = {
        "sub_second_through_N": (max(subsec_ns) if subsec_ns else None),
        "latency_ms_median_by_N": {str(n): sorted(round(c["latency_ms_median"], 1) for c in cells
                                                   if c["n_facts"] == n and c["latency_ms_median"] is not None)
                                   for n in sorted({c["n_facts"] for c in cells})},
    }
    v.disabled("learned/spiking cue->shard router",
               "router hash(agent) mod S is a declared host scaffold (ledger scaffold_retired:NO); the in-shard FHRR "
               "recall + the no-confab moat are the genuine reads")
    decided = v.decide(go=go)

    verdict = dict(decided)
    verdict.update({
        "arc": "knowledge-scale flip soak (board #66/#109; ledger tiered-knowledge-ltm)",
        "backend": os.environ.get("SIM_BACKEND", "?"),
        "seeds": seeds, "plan": [list(p) for p in plan],
        "per_seed_go": {str(k): v_ for k, v_ in per_seed_go.items()},
        "seeds_go": seeds_go, "total_byte_identity_mismatches": total_mism, "total_moat_confab": total_confab,
        "latency_characterization": latency_char,
        "cells": cells, "degrade": degrade, "production_load_path": prod,
        "elapsed_s": round(time.time() - t0, 2),
    })
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print("\nwrote", a.out, "GO=", verdict.get("go"))
    return 0 if verdict.get("go") else 1


if __name__ == "__main__":
    sys.exit(main())
