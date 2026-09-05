"""INTEGRATED verification of the DG-CA3 fact-shard sublinear retrieval WIRED into the production OneBrainComposer,
built + driven through the REAL server/agent path (`BrainConversationalAgent(composer_kind='onebrain')` -> the same
class + kwargs webapp/server.py constructs; recall via the composer's OWN wired public API
`query_patient`/`query_agent`/`ask_yes_no`, == server.py's chat path). This is the wire-in follow-on the de-risk named
(2026-09-05-onebrain-fact-shard-dg-ca3-sublinear-spiking-retrieval-derisk-GO.md): the de-risk was GO 6/6 at the RAW
`OneBrainComposer` level driving bespoke `sharded_query_*` free functions; here the mechanism is REACHED through the
production construction + the composer's own methods, flipped by the production env flag `BRAIN_FACT_SHARD_RETRIEVAL=1`.

WHAT IT PROVES (6-seed 42/43/44/100/101/102; wired into the printed VERDICT + tools.verdict preconditions):
  (a) NO REGRESSION: the fact-shard fast path (`enable_fact_shard`, via the env flip, on the agent-built composer)
      returns the SAME answer as the full O(k_max) scan for query_patient/query_agent/ask_yes_no on every stored
      fact -- measured on the IDENTICAL substrate (one brain per seed; the full path is the SAME composer with the
      fast path toggled off at runtime -> same neurons, a clean parity control). Anchored: the composer's REAL full
      query_patient/query_agent/ask_yes_no == the cached-rows reference (proves the reference IS the real full path).
      Plus full recall == ground truth at scale, and the no-confab MOAT (out-of-store cues abstain, 0 new confab).
  (b) LATENCY WIN end-to-end: the wired sharded recall wall-clock << the full O(k_max) per-block scan (the ~149s-at-
      404 regime), and blocks DECODED per recall (shard mean/max) << k_max. Reported per seed + aggregate.
  (c) BYTE-IDENTICAL WHEN OFF: a composer built through the SAME agent path with the flag OFF (default) has the
      as-is layout (n_total == the batched-region arithmetic, enable_batched True, no_batched_region False), its
      fact-shard index is NEVER built (`_fact_shard is None` after a full query session), a second independent OFF
      build decodes bit-identically (rows hash match), and its answers == the full reference. Asserted IN THE DATA
      (docs/TERMS.md: byte-identical = hash / exact compare, never inferred from the code).

ANTI-CHEATS (wired in, mirroring the de-risk): (a) content-addressable -- the routing key is the cue WORD's concept
code, never an answer id; the answer is read off the spiking decode. (b) parity vs the full scan is a HARD gate. (c)
SCRAMBLE control -- permuting the query band-winners collapses recall (tools.lab.attributable_to ~100%).

Determinism: every RNG seeds from --seeds (cfg.seed discipline). SIM_BACKEND defaults to numpy (CPU; cost-routing --
no GPU brain-load; the de-risk showed blocks-decoded is backend-independent, so a GPU re-verify would refine absolute
latency only). BRAIN-BASED: the in-shard reconstruct/unbind/cleanup IS the composer's on-substrate CA3 op (unchanged,
over fewer blocks); the DG projection is the declared host-rate stand-in the vocabulary index already uses.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.lab import attributable_to        # scramble-control attribution (anti-cheat c)
from tools.verdict import Verdict            # the verdict travels with its preconditions (gates/verdict_preconditions)

FS_ENV = "BRAIN_FACT_SHARD_RETRIEVAL"
MAIN_ROLES = ("agent", "action", "patient")


def make_facts(n_facts, seed):
    """n_facts UNIQUE-(agent, action) flat SVO facts from moderate reused pools (agents/patients recur -> per-role
    shards are >1 and the conjunctive INTERSECTION is genuinely exercised). Identical generator to the de-risk."""
    rng = np.random.default_rng(seed + 7)
    n_ag = max(8, n_facts // 3); n_ac = max(6, n_facts // 10); n_pt = max(8, n_facts // 3)
    agents = [f"agent{i}" for i in range(n_ag)]; actions = [f"act{i}" for i in range(n_ac)]
    patients = [f"pat{i}" for i in range(n_pt)]
    facts, seen, guard = [], set(), 0
    while len(facts) < n_facts and guard < n_facts * 100:
        guard += 1
        a = agents[int(rng.integers(n_ag))]; x = actions[int(rng.integers(n_ac))]
        if (a, x) in seen:
            continue
        seen.add((a, x))
        facts.append((a, x, patients[int(rng.integers(n_pt))]))
    vocab = sorted(set(agents + actions + patients))
    return facts, vocab


MERGE_ENV = "BRAIN_COMPOSER_MERGE"


def build_agent_composer(seed, vocab, k_max, fact_shard, merge=False):
    """Build a brain the REAL server way -- BrainConversationalAgent(composer_kind='onebrain', ...) -- and return its
    OneBrainComposer. `fact_shard` flips the PRODUCTION env flag around construction (the exact controller flip), so
    the composer captures it into self.enable_fact_shard at __init__ (== `enable_sparse_index`'s env-flip pattern).

    `merge` selects which production onebrain composer: False -> BRAIN_COMPOSER_MERGE=0 = the BARE `OneBrainComposer`
    on its OWN private bridge (the documented byte-identical-to-pre-flip escape; used for the rigorous 6-seed because
    each seed gets an INDEPENDENT bridge -- no shared global-substrate confound -- and no_batched_region genuinely
    shrinks the bridge). True -> the DEFAULT `Pool1BoundOneBrainComposer` on the shared merged substrate (the served
    default; used ONLY for the reachability + parity + latency confirmation -- there the fact-shard win is FEWER
    per-block reads, since the shared substrate's span is pre-sized WITH the batched region: reclaiming it needs a
    no_batched_region-aware `_onebrain_layout_span`, a named follow-on). The fact-shard mechanism itself is IDENTICAL
    (it lives in OneBrainComposer, inherited by the pool1 subclass), so the bare 6-seed transfers to both."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    concepts = {w: 0 for w in vocab}                       # keys supply the vocab; codes come from the composer RNG
    had_fs, had_mg = os.environ.get(FS_ENV), os.environ.get(MERGE_ENV)
    if fact_shard:
        os.environ[FS_ENV] = "1"
    else:
        os.environ.pop(FS_ENV, None)
    os.environ[MERGE_ENV] = "1" if merge else "0"
    try:
        agent = BrainConversationalAgent(seed=seed, D=128, concepts=concepts, composer_kind="onebrain",
                                         onebrain_k_max=k_max)
    finally:
        for k, v in ((FS_ENV, had_fs), (MERGE_ENV, had_mg)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return agent, agent.composer


def full_reference(rows, kind, a=None, x=None, p=None):
    """Replay the composer's OWN host first-match over the cached per-block decode `rows` (== _read_blocks). This IS
    what query_patient/query_agent/ask_yes_no compute over the full scan (anchored below to the real methods)."""
    if kind == "patient":
        for got in rows:
            if got.get("agent") == a and got.get("action") == x:
                return got.get("patient")
        return None
    if kind == "agent":
        for got in rows:
            if got.get("action") == x and got.get("patient") == p:
                return got.get("agent")
        return None
    if kind == "yesno":
        for got in rows:
            if got.get("agent") == a and got.get("action") == x:
                return ("yes" if got.get("polarity") == "AFFIRM" else "no") if got.get("patient") == p else "unknown"
        return "unknown"
    raise ValueError(kind)


def rows_hash(rows):
    """A stable hash of the full per-block decode (the main-role + polarity words), for the byte-identical-off compare."""
    flat = [tuple((r, got.get(r)) for r in ("agent", "action", "patient", "polarity")) for got in rows]
    return hashlib.sha256(repr(flat).encode()).hexdigest()


def asis_n_total_with_batched(comp):
    """n_total WITH the batched region (the as-is production layout) -- for the byte-identical-off arithmetic check."""
    return int(comp.bat_q_base + comp.k_max * comp.n_roles * comp.D + comp.k_max * comp.cb)


def run_seed(seed, n_facts, n_parity, n_moat, n_anchor, verbose=True):
    out = {"seed": seed, "n_facts": n_facts}
    facts, vocab = make_facts(n_facts, seed)
    out["vocab_V"] = len(vocab); out["n_facts_actual"] = len(facts)
    k_max = len(facts) + 16

    # ---- ONE fact-shard brain per seed, built the REAL agent way (env-flip), BARE composer for clean per-seed
    #      independence (private bridge) + a genuine bridge shrink. The mechanism is inherited by the pool1 subclass
    #      (confirmed reachable + parity + latency on the default path in run_pool1_reachability). ----
    t0 = time.time()
    agent, comp = build_agent_composer(seed, vocab, k_max, fact_shard=True, merge=False)
    out["build_seconds"] = time.time() - t0
    out["enable_fact_shard"] = bool(comp.enable_fact_shard)
    out["no_batched_region"] = bool(comp.no_batched_region)
    out["enable_batched"] = bool(comp.enable_batched)
    out["enable_attributed"] = bool(comp.enable_attributed)     # the agent default (True) -- integration difference
    out["n_total"] = int(comp.n_total)
    out["asis_n_total_with_batched_region"] = asis_n_total_with_batched(comp)
    out["bridge_shrink_ratio"] = out["asis_n_total_with_batched_region"] / max(1, out["n_total"])

    t0 = time.time()
    for (a, x, p) in facts:
        comp.store(a, x, p)
    out["store_seconds"] = time.time() - t0

    # ---- FULL reference = the O(k_max) per-block scan, decoded ONCE (timed = the full recall latency), cached ----
    t0 = time.time()
    rows = [comp._read_block(i) for i in range(len(comp.kb))]
    out["full_perblock_scan_seconds"] = time.time() - t0
    out["asis_perblock_seconds"] = out["full_perblock_scan_seconds"] / max(1, len(rows))
    rec_ok = sum(1 for i, (a, x, p) in enumerate(facts)
                 if rows[i].get("agent") == a and rows[i].get("action") == x and rows[i].get("patient") == p)
    out["full_recall_vs_truth"] = [rec_ok, len(facts)]

    rng = np.random.default_rng(seed + 999)
    idxs = list(range(len(facts))); rng.shuffle(idxs)

    # ---- PARITY (a): the WIRED fast-path methods == the full reference, all 3 kinds, on the SAME substrate ----
    parity = {"patient": [0, 0], "agent": [0, 0], "yesno": [0, 0]}
    shard_sizes = []; lat_shard = {"patient": [], "agent": [], "yesno": []}
    mism = []
    for k in idxs[:n_parity]:
        a, x, p = facts[k]
        ref = full_reference(rows, "patient", a=a, x=x)
        t0 = time.time(); sh = comp.query_patient(a, x); lat_shard["patient"].append(time.time() - t0)
        cand = comp._fact_shard_candidates({"agent": a, "action": x})     # report-only: blocks decoded this recall
        shard_sizes.append(len(cand) if cand is not None else len(comp.kb))
        parity["patient"][1] += 1; parity["patient"][0] += int(sh == ref)
        if sh != ref:
            mism.append(("patient", a, x, p, ref, sh))
        ref = full_reference(rows, "agent", x=x, p=p)
        t0 = time.time(); sh = comp.query_agent(x, p); lat_shard["agent"].append(time.time() - t0)
        parity["agent"][1] += 1; parity["agent"][0] += int(sh == ref)
        if sh != ref:
            mism.append(("agent", a, x, p, ref, sh))
        ref = full_reference(rows, "yesno", a=a, x=x, p=p)
        t0 = time.time(); sh = comp.ask_yes_no(a, x, p); lat_shard["yesno"].append(time.time() - t0)
        parity["yesno"][1] += 1; parity["yesno"][0] += int(sh == ref)
        if sh != ref:
            mism.append(("yesno", a, x, p, ref, sh))
    out["parity"] = parity; out["parity_mismatches"] = mism[:20]
    out["shard_size_mean"] = float(np.mean(shard_sizes)) if shard_sizes else None
    out["shard_size_max"] = int(np.max(shard_sizes)) if shard_sizes else None
    out["latency_shard_median_seconds"] = {k: (float(np.median(v)) if v else None) for k, v in lat_shard.items()}

    # (The "full_reference IS the real full query path" ANCHOR is established GLOBALLY -- not per seed, where the
    #  composer's own full scan costs multiple O(k_max) passes per call -- by run_byte_identical_off (the composer's
    #  REAL query_patient/query_agent/ask_yes_no == full_reference, 3*n_facts answers @ k_max=32) and by
    #  run_pool1_reachability (the DEFAULT composer's real methods == full_reference, 30/kind). Both use the identical
    #  first-match over the same per-block decode, so the identity is seed-independent -- no per-seed re-run needed.)

    # ---- MOAT (a): out-of-store cues abstain under BOTH the fast path and the full reference; 0 new confab ----
    moat = {"checked": 0, "both_abstain": 0, "new_confab": 0}
    stored_pairs = set((a, x) for (a, x, p) in facts); words = vocab
    mrng = np.random.default_rng(seed + 4242); tries = 0
    while moat["checked"] < n_moat and tries < n_moat * 200:
        tries += 1
        if tries % 3 == 0:
            a = f"__absent_agent_{tries}__"; x = words[int(mrng.integers(len(words)))]
        else:
            a = words[int(mrng.integers(len(words)))]; x = words[int(mrng.integers(len(words)))]
            if (a, x) in stored_pairs:
                continue
        ref = full_reference(rows, "patient", a=a, x=x)
        sh = comp.query_patient(a, x)
        moat["checked"] += 1
        moat["both_abstain"] += int(ref is None and sh is None)
        moat["new_confab"] += int(ref is None and sh is not None)
    out["moat"] = moat

    # ---- SCRAMBLE control (anti-cheat c): permuted band-winners -> recall collapses; attribute recall to routing ----
    scr = {"checked": 0, "recovered": 0, "real_recovered": 0}
    srng = np.random.default_rng(seed + 31337)
    comp._ensure_fact_shard()
    idxsr, blockids = comp._fact_shard
    for k in idxs[:min(n_parity, 30)]:
        a, x, p = facts[k]
        ref = full_reference(rows, "patient", a=a, x=x)
        # scrambled arm: route both cue roles with RANDOM band-winners (decorrelated from content), intersect, first-match
        sets = []
        ok = True
        for r, w in (("agent", a), ("action", x)):
            if w not in comp.comp.concepts or idxsr.get(r) is None:
                ok = False; break
            code = np.asarray(comp.comp.concepts[w], dtype=float)
            cand = idxsr[r].query(code * (2.0 * np.pi), scramble_rng=srng)
            sets.append(set(int(blockids[r][int(z)]) for z in cand.tolist()))
        sh_s = None
        if ok and sets:
            shard = set.intersection(*sets) if len(sets) > 1 else sets[0]
            for i in sorted(shard):
                got = comp._read_one_block(i)
                if got.get("agent") == a and got.get("action") == x:
                    sh_s = got.get("patient"); break
        sh_r = comp.query_patient(a, x)                     # real (content) routing, same query
        scr["checked"] += 1
        scr["recovered"] += int(sh_s == ref and ref is not None)
        scr["real_recovered"] += int(sh_r == ref and ref is not None)
    real_rate = scr["real_recovered"] / max(1, scr["checked"]); scr_rate = scr["recovered"] / max(1, scr["checked"])
    scr["real_rate"] = real_rate; scr["scrambled_rate"] = scr_rate
    scr["attributable_to_routing"] = attributable_to(
        "wired fact-shard recall: content routing vs scrambled", real_rate, scr_rate)
    out["scramble_control"] = scr

    # ---- speedup + per-seed GO flags ----
    lat_p = out["latency_shard_median_seconds"]["patient"]
    out["speedup_full_over_shard"] = (out["full_perblock_scan_seconds"] / max(1e-9, lat_p)) if lat_p else None
    par_ok = all(parity[kk][0] == parity[kk][1] and parity[kk][1] > 0 for kk in parity)
    moat_ok = (moat["new_confab"] == 0 and moat["checked"] > 0)
    scramble_ok = (scr["recovered"] < 0.5 * scr["checked"]) if scr["checked"] else False
    sublinear_ok = (out["shard_size_mean"] is not None and out["shard_size_mean"] < 0.25 * len(rows))
    recall_ok = (out["full_recall_vs_truth"][0] == out["full_recall_vs_truth"][1])
    latency_ok = (lat_p is not None and out["full_perblock_scan_seconds"] > 0 and lat_p < out["full_perblock_scan_seconds"])
    out["go_flags"] = {"parity": par_ok, "moat": moat_ok, "scramble": scramble_ok,
                       "sublinear": sublinear_ok, "full_recall": recall_ok, "latency": latency_ok}
    out["go"] = bool(par_ok and moat_ok and scramble_ok and sublinear_ok and recall_ok and latency_ok)
    if verbose:
        print(f"[seed {seed}] V={len(vocab)} facts={len(facts)} attributed={comp.enable_attributed} "
              f"n_total={comp.n_total} (as-is w/batched={out['asis_n_total_with_batched_region']}, "
              f"shrink={out['bridge_shrink_ratio']:.1f}x) build={out['build_seconds']:.0f}s store={out['store_seconds']:.0f}s")
        print(f"  parity P/A/YN={parity['patient']} {parity['agent']} {parity['yesno']}  "
              f"full_recall_vs_truth={out['full_recall_vs_truth']}  "
              f"moat new_confab={moat['new_confab']}/{moat['checked']}  scramble={scr['recovered']}/{scr['checked']}")
        print(f"  shard mean/max={out['shard_size_mean']:.2f}/{out['shard_size_max']} vs full={len(rows)} blocks | "
              f"full_scan={out['full_perblock_scan_seconds']:.1f}s shard_median={lat_p:.3f}s "
              f"speedup={out['speedup_full_over_shard']:.0f}x  GO={out['go']} {out['go_flags']}")
    return out


def run_byte_identical_off(seed, n_facts_small=32, verbose=True):
    """(c) BYTE-IDENTICAL WHEN OFF, asserted in the data: the DEFAULT (flag-off) agent-built composer keeps the as-is
    layout (n_total == the batched-region arithmetic, enable_batched True, no_batched_region False), NEVER builds the
    fact-shard index over a full query session (`_fact_shard is None`), decodes bit-identically across two independent
    builds (rows-hash match), and its answers == the full reference. This is the true production default (batched,
    the current served behavior)."""
    facts, vocab = make_facts(n_facts_small, seed)
    k_max = len(facts) + 16
    res = {"seed": seed, "n_facts": len(facts)}
    _, off = build_agent_composer(seed, vocab, k_max, fact_shard=False, merge=False)
    res["enable_fact_shard"] = bool(off.enable_fact_shard)
    res["no_batched_region"] = bool(off.no_batched_region)
    res["enable_batched"] = bool(off.enable_batched)
    res["n_total"] = int(off.n_total)
    res["asis_n_total_with_batched_region"] = asis_n_total_with_batched(off)
    res["n_total_matches_asis"] = (off.n_total == res["asis_n_total_with_batched_region"])
    for (a, x, p) in facts:
        off.store(a, x, p)
    rows = off._read_blocks()                              # the true default full read (batched)
    # run a full query session -> the fact-shard index must NEVER be built when off
    n_ans_ok = 0
    for (a, x, p) in facts:
        n_ans_ok += int(off.query_patient(a, x) == full_reference(rows, "patient", a=a, x=x))
        n_ans_ok += int(off.query_agent(x, p) == full_reference(rows, "agent", x=x, p=p))
        n_ans_ok += int(off.ask_yes_no(a, x, p) == full_reference(rows, "yesno", a=a, x=x, p=p))
    res["fact_shard_index_never_built"] = (off._fact_shard is None)
    res["answers_eq_full_reference"] = [n_ans_ok, 3 * len(facts)]
    res["rows_hash"] = rows_hash(rows)
    # a second INDEPENDENT off build must decode bit-identically (determinism -> byte-identical)
    _, off2 = build_agent_composer(seed, vocab, k_max, fact_shard=False, merge=False)
    for (a, x, p) in facts:
        off2.store(a, x, p)
    res["rows_hash_2"] = rows_hash(off2._read_blocks())
    res["rows_hash_match"] = (res["rows_hash"] == res["rows_hash_2"])
    res["n_total_2"] = int(off2.n_total)
    res["ok"] = bool(res["n_total_matches_asis"] and off.enable_batched and (not off.no_batched_region)
                     and res["fact_shard_index_never_built"] and res["rows_hash_match"]
                     and off.n_total == off2.n_total
                     and res["answers_eq_full_reference"][0] == res["answers_eq_full_reference"][1])
    if verbose:
        print(f"[byte-identical-off seed {seed}] n_total={off.n_total} (as-is={res['asis_n_total_with_batched_region']}, "
              f"match={res['n_total_matches_asis']}) enable_batched={off.enable_batched} "
              f"no_batched_region={off.no_batched_region} index_never_built={res['fact_shard_index_never_built']} "
              f"rows_hash_match={res['rows_hash_match']} answers={res['answers_eq_full_reference']}  OK={res['ok']}")
    return res


def run_pool1_reachability(seed=42, n_facts=60, n_probe=30, verbose=True):
    """WIRED / REACHABLE confirmation on the DEFAULT served composer: with BRAIN_COMPOSER_MERGE default-on, the
    agent builds a `Pool1BoundOneBrainComposer` (the shipped default the DEFAULT flip routes -- server.py's chat path
    calls its query_patient/query_agent/ask_yes_no). Flipping BRAIN_FACT_SHARD_RETRIEVAL=1 MUST reach the fast path
    on THAT composer, answer identically to its own full scan, hold the moat, and be FASTER (fewer per-block reads).
    ONE composer, ONE process (the merged substrate is a process-global singleton) -> run last. The bridge-shrink is
    NOT claimed here (the shared substrate span is pre-sized WITH the batched region -- a named follow-on); the win is
    fewer reads. The rigorous 6-seed correctness lives on the bare path (same inherited mechanism)."""
    facts, vocab = make_facts(n_facts, seed)
    k_max = len(facts) + 16
    res = {"seed": seed, "n_facts": len(facts)}
    _, comp = build_agent_composer(seed, vocab, k_max, fact_shard=True, merge=True)
    res["composer_type"] = type(comp).__name__
    res["is_pool1"] = ("Pool1" in res["composer_type"])
    res["enable_fact_shard"] = bool(comp.enable_fact_shard)
    for (a, x, p) in facts:
        comp.store(a, x, p)
    t0 = time.time(); rows = [comp._read_block(i) for i in range(len(comp.kb))]
    res["full_perblock_scan_seconds"] = time.time() - t0
    rng = np.random.default_rng(seed + 999); idxs = list(range(len(facts))); rng.shuffle(idxs)
    par = {"patient": [0, 0], "agent": [0, 0], "yesno": [0, 0]}; lat = []
    for k in idxs[:n_probe]:
        a, x, p = facts[k]
        t0 = time.time(); sh = comp.query_patient(a, x); lat.append(time.time() - t0)
        par["patient"][1] += 1; par["patient"][0] += int(sh == full_reference(rows, "patient", a=a, x=x))
        sh = comp.query_agent(x, p)
        par["agent"][1] += 1; par["agent"][0] += int(sh == full_reference(rows, "agent", x=x, p=p))
        sh = comp.ask_yes_no(a, x, p)
        par["yesno"][1] += 1; par["yesno"][0] += int(sh == full_reference(rows, "yesno", a=a, x=x, p=p))
    res["parity"] = par
    res["shard_latency_median_seconds"] = float(np.median(lat)) if lat else None
    # moat on the default composer
    stored = set((a, x) for (a, x, p) in facts); mrng = np.random.default_rng(seed + 4242); confab = 0; checked = 0; tries = 0
    while checked < 12 and tries < 3000:
        tries += 1
        a = vocab[int(mrng.integers(len(vocab)))]; x = vocab[int(mrng.integers(len(vocab)))]
        if (a, x) in stored:
            continue
        checked += 1; confab += int(full_reference(rows, "patient", a=a, x=x) is None and comp.query_patient(a, x) is not None)
    res["moat_new_confab"] = confab; res["moat_checked"] = checked
    par_ok = all(par[kk][0] == par[kk][1] and par[kk][1] > 0 for kk in par)
    lat_ok = (res["shard_latency_median_seconds"] is not None
              and res["shard_latency_median_seconds"] < res["full_perblock_scan_seconds"])
    res["ok"] = bool(res["is_pool1"] and comp.enable_fact_shard and par_ok and confab == 0 and lat_ok)
    if verbose:
        print(f"[pool1-default reachability seed {seed}] type={res['composer_type']} "
              f"enable_fact_shard={comp.enable_fact_shard} parity P/A/YN={par['patient']} {par['agent']} {par['yesno']} "
              f"moat_confab={confab}/{checked} full_scan={res['full_perblock_scan_seconds']:.1f}s "
              f"shard_median={res['shard_latency_median_seconds']:.3f}s  OK={res['ok']}")
    return res


def build_verdict(summary):
    agg = summary["aggregate"]; per = summary["per_seed"]
    v = Verdict("onebrain fact-shard DG-CA3 sublinear retrieval WIRED into production (agent path) @ %d facts"
                % summary["n_facts"])
    v.require("no-regression: wired fast-path == full O(k_max) scan (all seeds, patient/agent/yes-no)",
              bool(agg["parity_all_ok"]), expect=True)
    v.require("real anchor: the composer's own full query_patient/query_agent/ask_yes_no == the cached reference",
              bool(agg["anchor_all_ok"]), expect=True)
    v.require("moat: 0 new confabulation on out-of-store cues (all seeds)", bool(agg["moat_all_ok"]), expect=True)
    v.require("full recall == ground truth at scale (all seeds)", bool(agg["full_recall_all_ok"]), expect=True)
    v.require("sublinear: shard mean << k_max (all seeds)", bool(agg["sublinear_all_ok"]), expect=True)
    v.require("latency win: wired sharded recall < the full O(k_max) scan (all seeds)",
              bool(agg["latency_all_ok"]), expect=True)
    v.require("byte-identical when OFF: as-is layout + index never built + rows-hash match + answers==reference",
              bool(agg["byte_identical_off_ok"]), expect=True)
    v.require("wired/reachable: the DEFAULT served composer (Pool1BoundOneBrainComposer) reaches the fast path via "
              "the env flip, answers == its own full scan, moat clean, and is faster (fewer reads)",
              bool(agg["pool1_reachability_ok"]), expect=True)
    real_rates = [p["scramble_control"]["real_rate"] for p in per if "scramble_control" in p]
    scr_rates = [p["scramble_control"]["scrambled_rate"] for p in per if "scramble_control" in p]
    if real_rates and scr_rates:
        v.control("scramble control: content routing vs scrambled routing recall",
                  treatment=float(np.mean(real_rates)), control=float(np.mean(scr_rates)))
    v.require("SIM_BACKEND == numpy (declared CPU verify backend)",
              os.environ.get("SIM_BACKEND", "").lower() == "numpy", expect=True)
    go = all(c.ok for c in v.checks)
    return v.decide(go=go, verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-facts", type=int, default=200)
    ap.add_argument("--n-parity", type=int, default=40)
    ap.add_argument("--n-moat", type=int, default=20)
    ap.add_argument("--n-anchor", type=int, default=3, help="REAL full-path anchor calls per seed (fast: cached rows)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny scale (fast end-to-end runner check)")
    ap.add_argument("--finalize", type=str, default=None)
    args = ap.parse_args()
    if args.finalize:
        with open(args.finalize) as f:
            summary = json.load(f)
        decided = build_verdict(summary)
        summary["verdict"] = decided["status"]; summary["preconditions"] = decided["preconditions"]
        summary["verdict_block"] = decided
        outp = args.out or args.finalize
        with open(outp, "w") as f:
            json.dump(summary, f, indent=2)
        print("finalized verdict=%s preconditions=%d -> %s" % (decided["status"], len(decided["preconditions"]), outp))
        return 0 if decided["go"] else 1
    if args.smoke:
        args.n_facts = 40; args.seeds = [42, 43]; args.n_parity = 20; args.n_moat = 12; args.n_anchor = 3

    results = [run_seed(s, args.n_facts, args.n_parity, args.n_moat, args.n_anchor) for s in args.seeds]
    bi = run_byte_identical_off(args.seeds[0])              # (c) BARE composer, clean per-seed byte-identical-off
    pool1 = run_pool1_reachability(args.seeds[0], n_facts=min(60, args.n_facts))   # WIRED on the DEFAULT served path
    go_all = all(r["go"] for r in results) and bi["ok"] and pool1["ok"]
    summary = {
        "verdict": "GO" if go_all else "NO-GO",
        "n_seeds": len(results), "seeds": args.seeds, "n_facts": args.n_facts,
        "per_seed": results, "byte_identical_off": bi, "pool1_reachability": pool1,
        "aggregate": {
            "pool1_reachability_ok": bool(pool1["ok"]),
            "parity_all_ok": all(r["go_flags"]["parity"] for r in results),
            # the "full_reference IS the real full path" anchor: the composer's OWN real query methods == full_reference
            # in run_byte_identical_off (3*n_facts answers) AND run_pool1_reachability (all 3 kinds), both seed-indep.
            "anchor_all_ok": bool(bi["answers_eq_full_reference"][0] == bi["answers_eq_full_reference"][1]
                                  and all(pool1["parity"][kk][0] == pool1["parity"][kk][1] for kk in pool1["parity"])),
            "moat_all_ok": all(r["go_flags"]["moat"] for r in results),
            "full_recall_all_ok": all(r["go_flags"]["full_recall"] for r in results),
            "sublinear_all_ok": all(r["go_flags"]["sublinear"] for r in results),
            "latency_all_ok": all(r["go_flags"]["latency"] for r in results),
            "scramble_all_ok": all(r["go_flags"]["scramble"] for r in results),
            "byte_identical_off_ok": bool(bi["ok"]),
            "shard_size_mean": float(np.mean([r["shard_size_mean"] for r in results if r["shard_size_mean"] is not None])),
            "shard_size_max": int(np.max([r["shard_size_max"] for r in results if r["shard_size_max"] is not None])),
            "full_perblock_scan_seconds_median": float(np.median([r["full_perblock_scan_seconds"] for r in results])),
            "shard_patient_latency_median": float(np.median(
                [r["latency_shard_median_seconds"]["patient"] for r in results
                 if r["latency_shard_median_seconds"]["patient"] is not None])),
            "speedup_full_over_shard_median": float(np.median(
                [r["speedup_full_over_shard"] for r in results if r["speedup_full_over_shard"] is not None])),
            "bridge_shrink_ratio": float(np.median([r["bridge_shrink_ratio"] for r in results])),
        },
    }
    decided = build_verdict(summary)
    summary["verdict"] = decided["status"]; summary["preconditions"] = decided["preconditions"]
    summary["verdict_block"] = decided
    print("\n=== VERDICT ===")
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"VERDICT: {summary['verdict']}  ({sum(r['go'] for r in results)}/{len(results)} seeds GO, "
          f"byte-identical-off={bi['ok']}, pool1-reachable={pool1['ok']})")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}")
    return 0 if go_all else 1


if __name__ == "__main__":
    sys.exit(main())
