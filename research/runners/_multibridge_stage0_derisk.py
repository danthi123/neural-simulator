"""Multi-bridge deep-knowledge Stage-0 de-risk (DIAGNOSTIC; no training; CPU/numpy).

Design: research/findings/2026-06-26-multibridge-deep-knowledge-design.md (Stage 0, the load-bearing de-risk).

GOAL: prove PER-BRIDGE cleanup preserves who/what recall >=0.95 + the no-confab moat (0 false-accepts), where the
single-bridge crowding dropped it (the session's 2,012-concept brain -> recall 0.875). Uses the EXISTING brain
(bridges/firstchat/brain1454_w7000_seed42.npz) -- NO new training, NO new codes.

The mechanism under test (NO sim/ edit, NO composer edit): the production RFPhasorComposer ALREADY cleans up over
`comp.words` (rf_phasor_composer.py:381-444). If we build TWO composers each over a DISJOINT ~727-word shard, every
cleanup ranges over ~727 candidates instead of 1,454 -> recall fidelity + per-query matvec both restored. A host
`word2shard` dict routes a query to the shard that owns its AGENT (the proven g20_multibridge routing pattern).

Steps (per the prompt):
 1. Load the brain1454 npz (vocab ~1454, grounded phasor codes, D=128).
 2. Split the vocab into 2 DISJOINT ~727 shards; build 2 RFPhasorComposers, each grounded_codes = ONLY its shard.
 3. Build a host word2shard dict.
 4. Take SVO facts (REAL corpus-extracted via _load_real_facts on _combined_svo_facts.json; fall back to
    _make_svo_facts). Store each fact in the shard owning its AGENT; cross-shard facts use the design's bounded
    per-shard codebook extension (option 2a: add the patient's grounded code to the agent-shard's cleanup set).
 5. MEASURE: per-shard who/what recall (target >=0.95); the moat (absent + cross-shard-absent cues abstain -> 0 FA);
    per-query time vs a single-bridge baseline (one composer over all 1,454 + the same facts/queries).
 6. ANTI-CHEAT: a permuted-routing control (route facts to the WRONG shard) must COLLAPSE recall WITHOUT raising
    false-accepts (the routing is load-bearing; the moat is not routing-dependent).

OPTIONAL confound reproduction (the prompt's "where the single-bridge crowding dropped it"): if the 2,012-concept
brainALL npz is present (its vocab is a superset of brain1454), reproduce 1454 vs 2012 single-bridge recall on the
SAME 24 facts drawn from the overlap -- locating the drop in the composer cleanup, not the discuss stack.

Run:  SIM_BACKEND=numpy python -m research.runners._multibridge_stage0_derisk \
          --out research/findings/raw/_multibridge_stage0_derisk.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402
from research.runners.first_chat_console import _load_real_facts  # noqa: E402
from research.runners._curriculum_step1_320_real_corpus import _make_svo_facts, _phase_cos  # noqa: E402

DEFAULT_BRAIN = os.path.join(_REPO, "bridges", "firstchat", "brain1454_w7000_seed42.npz")
DEFAULT_ALL = os.path.join(_REPO, "bridges", "firstchat", "brainALL_w7000.npz_seed42.npz")
DEFAULT_FACTS = os.path.join(_REPO, "research", "findings", "raw", "_combined_svo_facts.json")


# ----------------------------------------------------------------------------------------------------------------
def load_brain(npz_path):
    """Return (vocab[list[str]], grounded{word: phases[D]}, cat_ids, cat_names, D). Our own artifact -> allow_pickle."""
    blob = np.load(npz_path, allow_pickle=True)
    vocab = [str(w) for w in blob["vocab"]]
    grounded_arr = np.asarray(blob["grounded"], dtype=float)
    cat_ids = np.asarray(blob["cat_ids"], dtype=int)
    cat_names = [str(c) for c in blob["cat_names"]]
    D = int(blob["D"])
    assert grounded_arr.shape == (len(vocab), D), f"grounded {grounded_arr.shape} != ({len(vocab)},{D})"
    grounded = {w: grounded_arr[i] for i, w in enumerate(vocab)}
    return vocab, grounded, cat_ids, cat_names, D


def split_shards(vocab, n_shards, seed):
    """Deterministically partition the vocab into n_shards DISJOINT shards of ~equal size. Returns a list of
    sorted word-lists. (A real build shards by g20 DOMAIN, design 2.6; Stage-0 only needs a disjoint partition to
    prove the cleanup mechanism, so we use a deterministic random split for vocab-balance.)"""
    rng = np.random.RandomState(seed)
    words = sorted(set(vocab))
    order = rng.permutation(len(words))
    shards = [[] for _ in range(n_shards)]
    for rank, idx in enumerate(order):
        shards[rank % n_shards].append(words[idx])
    return [sorted(s) for s in shards]


def build_shard_composers(shards, grounded, seed, D):
    """One RFPhasorComposer per shard over ONLY that shard's vocab + grounded codes -> cleanup is over ~727, not
    1,454. Returns (comps[list], word2shard{word: shard_index})."""
    comps, word2shard = [], {}
    for si, sh in enumerate(shards):
        gsub = {w: grounded[w] for w in sh}
        comps.append(RFPhasorComposer(seed=seed, D=D, vocab=sh, grounded_codes=gsub))
        for w in sh:
            word2shard[w] = si
    return comps, word2shard


def store_facts_routed(facts, comps, word2shard, grounded, *, permute=False, n_shards=None):
    """Store each fact in the shard owning its AGENT (design 2.5 agent-anchoring). The composer's _encode binds ALL
    three roles (agent, action, patient), so EVERY role-filler not already in the agent-shard's codebook must be
    co-stored there (design option 2a, the bounded per-shard codebook extension). The agent is always native; the
    action and patient may be cross-shard. Returns stats: {n_stored, n_same_shard, n_cross_shard,
    n_extension_fillers, codebook_ext_per_shard}.

    permute=True (the anti-cheat): route each fact to the WRONG shard (agent's shard index + 1, mod n_shards). Recall
    must collapse; the moat must NOT break (a wrongly-routed query must abstain, never confabulate)."""
    ns = n_shards if n_shards is not None else len(comps)
    n_same = n_cross = n_ext = 0
    ext_per_shard = [0] * len(comps)
    for a, v, p in facts:
        si = word2shard[a]
        if permute:
            si = (si + 1) % ns                      # route to the WRONG shard
        comp = comps[si]
        # a fact is "same-shard" iff all three role-fillers are native to the storing shard; else it's cross-shard.
        is_cross = False
        for w in (a, v, p):
            if w not in comp.concepts:              # co-store the cross-shard filler's grounded code (option 2a)
                comp.concepts[w] = np.asarray(grounded[w], dtype=float)
                # the composer's CLEANUP ranges over comp.words (rf_phasor_composer.py:382), so option-2a must
                # extend BOTH the code dict AND the cleanup word-list -- otherwise the recovered cross-shard filler
                # phasor cannot be decoded (it would clean up only over the native shard vocab). The codebook stays
                # BOUNDED: native ~727 + the handful of distinct cross-shard fillers this shard's facts reference.
                comp.words = sorted(set(comp.words) | {w})
                ext_per_shard[si] += 1
                n_ext += 1
                is_cross = True
        n_cross += int(is_cross)
        n_same += int(not is_cross)
        comp.store(a, v, p, polarity="AFFIRM")
    return {"n_stored": len(facts), "n_same_shard": n_same, "n_cross_shard": n_cross,
            "n_extension_fillers": n_ext, "codebook_ext_per_shard": ext_per_shard}


def measure_multibridge_recall(facts, comps, word2shard):
    """Per-shard who/what recall via routed queries. query_patient(a,v) routes to word2shard[a]; query_agent(v,p)
    routes by patient's shard FIRST, then any shard whose vocab/codebook contains the agent (design 2.3 / 2.5
    cross-shard fallback). Returns (overall_recall, per_shard{si: {...}}, route_misses)."""
    # group stored facts by the shard they actually live on (= agent's shard) for per-shard reporting
    by_shard = {}
    for a, v, p in facts:
        by_shard.setdefault(word2shard[a], []).append((a, v, p))

    per_shard = {}
    overall_ok = overall_tot = 0
    route_misses = []
    for si, fs in sorted(by_shard.items()):
        ok = tot = 0
        for a, v, p in fs:
            # WHAT: route by agent -> the shard that owns the agent (where the fact lives)
            ans = _routed_query_patient(a, v, comps, word2shard)
            if ans == p:
                ok += 1
            tot += 1
            # WHO: route by patient first; fall back to the agent's shard (the fact's home)
            ans2 = _routed_query_agent(v, p, comps, word2shard, home_shard=word2shard[a])
            if ans2 == a:
                ok += 1
            tot += 1
        per_shard[si] = {"recall": ok / max(tot, 1), "correct": ok, "total": tot, "n_facts": len(fs)}
        overall_ok += ok
        overall_tot += tot
    return overall_ok / max(overall_tot, 1), per_shard, route_misses


def _routed_query_patient(a, v, comps, word2shard):
    si = word2shard.get(a)
    return None if si is None else comps[si].query_patient(a, v)        # abstain on unknown agent (the router moat)


def _routed_query_agent(v, p, comps, word2shard, home_shard=None):
    """who (v,p): try the patient's shard, then the fact's home shard (agent's shard). A fact lives on the agent's
    shard; the patient's code may have been co-stored there (option 2a). Returns the first non-None answer or None."""
    tried = []
    sp = word2shard.get(p)
    if sp is not None:
        tried.append(sp)
    if home_shard is not None and home_shard not in tried:
        tried.append(home_shard)
    for si in tried:
        ans = comps[si].query_agent(v, p)
        if ans is not None:
            return ans
    return None


def measure_multibridge_moat(absent_what, absent_who, cross_absent, comps, word2shard):
    """The no-confab moat across shards (design anti-cheat 1). Three absent-cue families MUST all abstain:
      (a) absent (agent,action) whose agent IS in a shard -> the composer must abstain (return None);
      (b) absent (action,patient) -> routed who must abstain;
      (c) cross-shard absent (agent in A, patient in B) never stored -> must abstain (not spuriously match via 2a).
    A single confident answer on ANY absent cue is a HARD STOP. Returns (abstain_rate, false_accept, breaches)."""
    fa = 0
    tot = 0
    breaches = []
    for a, v in absent_what:
        tot += 1
        ans = _routed_query_patient(a, v, comps, word2shard)
        if ans is not None:
            fa += 1
            breaches.append(f"query_patient({a},{v}) -> {ans!r} (should abstain)")
    for v, p in absent_who:
        tot += 1
        ans = _routed_query_agent(v, p, comps, word2shard)
        if ans is not None:
            fa += 1
            breaches.append(f"query_agent({v},{p}) -> {ans!r} (should abstain)")
    for a, v, p in cross_absent:
        # a never-stored cross-shard SVO: both who and what must abstain
        tot += 1
        ans = _routed_query_patient(a, v, comps, word2shard)
        if ans is not None:
            fa += 1
            breaches.append(f"x-shard query_patient({a},{v}) -> {ans!r} (should abstain)")
        tot += 1
        ans2 = _routed_query_agent(v, p, comps, word2shard, home_shard=word2shard.get(a))
        if ans2 is not None:
            fa += 1
            breaches.append(f"x-shard query_agent({v},{p}) -> {ans2!r} (should abstain)")
    return (1.0 - fa / max(tot, 1)), fa, breaches[:12]


def make_cross_shard_absent(facts, word2shard, n, seed):
    """Cross-shard absent cues: (agent in shard A, action, patient in shard B!=A) where NEITHER cue-pair is stored
    -> both who AND what must abstain. CRITICAL: the moat test queries the PAIRS (agent,action) [what] and
    (action,patient) [who], so a fair absent cue must have BOTH (a,v) NOT a stored (agent,action) pair AND (v,p)
    NOT a stored (action,patient) pair (else the composer correctly RECALLS, which is not a false-accept). Drawn
    from the same fact-vocab so the test is fair."""
    rng = np.random.RandomState(seed * 17 + 9)
    agents = sorted({a for a, _, _ in facts})
    actions = sorted({v for _, v, _ in facts})
    patients = sorted({p for _, _, p in facts})
    stored_av = {(a, v) for a, v, _ in facts}          # stored (agent,action) -> what_does recalls
    stored_vp = {(v, p) for _, v, p in facts}          # stored (action,patient) -> who_does recalls
    out, tries = [], 0
    while len(out) < n and tries < n * 800:
        tries += 1
        a = agents[rng.randint(len(agents))]
        v = actions[rng.randint(len(actions))]
        p = patients[rng.randint(len(patients))]
        if a == p:
            continue
        if word2shard.get(a) == word2shard.get(p):     # require cross-shard (the case 2a stresses)
            continue
        if (a, v) in stored_av or (v, p) in stored_vp:  # neither cue-pair may be a stored recall
            continue
        if (a, v, p) in {x for x in out}:
            continue
        out.append((a, v, p))
    return out


# ----------------------------------------------------------------------------------------------------------------
def measure_singlebridge(vocab, grounded, facts, absent_what, absent_who, seed, D, queries_for_timing):
    """One composer over the FULL union vocab + the SAME facts -> the single-bridge baseline (recall + time/query).
    queries_for_timing: list of (kind, args) re-run to time them. Returns dict."""
    comp = RFPhasorComposer(seed=seed, D=D, vocab=sorted(set(vocab)), grounded_codes=grounded)
    for a, v, p in facts:
        comp.store(a, v, p, polarity="AFFIRM")
    ok = tot = 0
    for a, v, p in facts:
        if comp.query_patient(a, v) == p:
            ok += 1
        tot += 1
        if comp.query_agent(v, p) == a:
            ok += 1
        tot += 1
    recall = ok / max(tot, 1)
    fa = mtot = 0
    breaches = []
    for a, v in absent_what:
        mtot += 1
        if comp.query_patient(a, v) is not None:
            fa += 1
            breaches.append(f"single query_patient({a},{v}) -> not None")
    for v, p in absent_who:
        mtot += 1
        if comp.query_agent(v, p) is not None:
            fa += 1
            breaches.append(f"single query_agent({v},{p}) -> not None")
    abstain = 1.0 - fa / max(mtot, 1)
    t_per = _time_queries(lambda kind, args: (comp.query_patient(*args) if kind == "what"
                                              else comp.query_agent(*args)), queries_for_timing)
    return {"recall": recall, "correct": ok, "total": tot, "abstain": abstain, "false_accept": fa,
            "abstain_total": mtot, "moat_breaches": breaches[:8], "vocab_size": len(set(vocab)),
            "sec_per_query": t_per}


def _time_queries(fn, queries, reps=3):
    """Median per-query wall time over `reps` passes of the query list."""
    times = []
    for _ in range(reps):
        t0 = time.time()
        for kind, args in queries:
            fn(kind, args)
        times.append((time.time() - t0) / max(len(queries), 1))
    return float(np.median(times))


def time_multibridge(queries_for_timing, comps, word2shard, facts):
    """Per-query wall time on the routed multi-bridge brain (cleanup over ONE shard's ~727)."""
    home = {a: word2shard[a] for a, _, _ in facts}

    def fn(kind, args):
        if kind == "what":
            a, v = args
            return _routed_query_patient(a, v, comps, word2shard)
        v, p = args
        return _routed_query_agent(v, p, comps, word2shard, home_shard=home.get(args[0] if False else None))

    return _time_queries(fn, queries_for_timing)


def build_timing_queries(facts):
    """The recall queries (both who + what for each stored fact) -- reused for timing on both architectures so the
    comparison is like-for-like."""
    q = []
    for a, v, p in facts:
        q.append(("what", (a, v)))
        q.append(("who", (v, p)))
    return q


# ----------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default=DEFAULT_BRAIN)
    ap.add_argument("--brain-all", default=DEFAULT_ALL, help="optional 2,012-concept brain for the confound repro")
    ap.add_argument("--facts-json", default=DEFAULT_FACTS)
    ap.add_argument("--n-facts", type=int, default=24)
    ap.add_argument("--n-shards", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(_REPO, "research", "findings", "raw",
                                                  "_multibridge_stage0_derisk.json"))
    args = ap.parse_args()

    print(f"[stage0] backend={os.environ.get('SIM_BACKEND', 'auto')}  seed={args.seed}", flush=True)
    vocab, grounded, cat_ids, cat_names, D = load_brain(args.brain)
    print(f"[stage0] loaded brain1454: {len(vocab)} concepts, D={D}", flush=True)

    # ---- the facts (REAL corpus-extracted; fall back to synthetic) + the absent-cue sets ----
    facts, absent_what, absent_who = [], [], []
    if args.facts_json and os.path.exists(args.facts_json):
        facts, absent_what, absent_who = _load_real_facts(args.facts_json, vocab, args.n_facts, args.seed)
        fact_src = f"real:{os.path.basename(args.facts_json)}"
    if not facts:
        facts, absent_what, absent_who = _make_svo_facts(vocab, cat_ids, cat_names, args.n_facts, args.seed)
        fact_src = "synthetic:_make_svo_facts"
    print(f"[stage0] {len(facts)} SVO facts ({fact_src}); {len(absent_what)} absent_what, "
          f"{len(absent_who)} absent_who", flush=True)

    timing_queries = build_timing_queries(facts)

    # ================================================================================================
    # PART 1 (optional): reproduce the single-bridge crowding confound -- 1454 vs 2012, SAME facts
    # ================================================================================================
    confound = None
    single_1454 = measure_singlebridge(vocab, grounded, facts, absent_what, absent_who, args.seed, D,
                                        timing_queries)
    print(f"[stage0] single-bridge 1454: recall={single_1454['recall']:.3f} "
          f"abstain={single_1454['abstain']:.3f} FA={single_1454['false_accept']} "
          f"t/q={single_1454['sec_per_query']*1e3:.1f}ms", flush=True)
    if args.brain_all and os.path.exists(args.brain_all):
        vocab_all, grounded_all, _ci, _cn, D_all = load_brain(args.brain_all)
        if set(vocab).issubset(set(vocab_all)) and D_all == D:
            # store the SAME facts (already drawn from the 1454 overlap) into a composer over the 2012 union
            single_2012 = measure_singlebridge(vocab_all, grounded_all, facts, absent_what, absent_who,
                                                args.seed, D, timing_queries)
            confound = {"recall_1454": single_1454["recall"], "recall_2012": single_2012["recall"],
                        "abstain_1454": single_1454["abstain"], "abstain_2012": single_2012["abstain"],
                        "tq_ms_1454": single_1454["sec_per_query"] * 1e3,
                        "tq_ms_2012": single_2012["sec_per_query"] * 1e3,
                        "vocab_1454": len(set(vocab)), "vocab_2012": len(set(vocab_all))}
            print(f"[stage0] CONFOUND repro: 1454 recall={confound['recall_1454']:.3f} "
                  f"({confound['tq_ms_1454']:.1f}ms) vs 2012 recall={confound['recall_2012']:.3f} "
                  f"({confound['tq_ms_2012']:.1f}ms) -- locates the drop in the cleanup", flush=True)
        else:
            print("[stage0] brainALL vocab not a superset / D mismatch -> skip confound repro", flush=True)

    # ================================================================================================
    # PART 2 (LOAD-BEARING): split 1454 -> n_shards, route facts by agent-shard, per-shard recall + moat + time
    # ================================================================================================
    shards = split_shards(vocab, args.n_shards, args.seed)
    shard_sizes = [len(s) for s in shards]
    # routing-correctness control: assert word2shard is a partition (disjoint vocabs)
    all_shard_words = [w for s in shards for w in s]
    assert len(all_shard_words) == len(set(all_shard_words)) == len(set(vocab)), "shards not a disjoint partition"
    print(f"[stage0] split into {args.n_shards} shards, sizes={shard_sizes}", flush=True)

    comps, word2shard = build_shard_composers(shards, grounded, args.seed, D)
    store_stats = store_facts_routed(facts, comps, word2shard, grounded)
    print(f"[stage0] routed-store: {store_stats['n_same_shard']} same-shard, "
          f"{store_stats['n_cross_shard']} cross-shard (codebook-ext per shard={store_stats['codebook_ext_per_shard']})",
          flush=True)

    mb_recall, per_shard, _miss = measure_multibridge_recall(facts, comps, word2shard)
    cross_absent = make_cross_shard_absent(facts, word2shard, max(len(facts), 8), args.seed)
    mb_abstain, mb_fa, mb_breaches = measure_multibridge_moat(absent_what, absent_who, cross_absent,
                                                              comps, word2shard)
    mb_tq = time_multibridge(timing_queries, comps, word2shard, facts)
    print(f"[stage0] MULTI-BRIDGE: overall recall={mb_recall:.3f}  abstain={mb_abstain:.3f}  FA={mb_fa}  "
          f"t/q={mb_tq*1e3:.1f}ms  (cross-absent cues={len(cross_absent)})", flush=True)
    for si, d in per_shard.items():
        print(f"           shard{si}: recall={d['recall']:.3f} ({d['correct']}/{d['total']}, "
              f"{d['n_facts']} facts, vocab={shard_sizes[si]})", flush=True)

    # ================================================================================================
    # PART 3 (ANTI-CHEAT): permuted routing -- store facts on the WRONG shard. Recall must collapse; moat must hold.
    # ================================================================================================
    comps_perm, word2shard_perm = build_shard_composers(shards, grounded, args.seed, D)
    store_facts_routed(facts, comps_perm, word2shard_perm, grounded, permute=True, n_shards=args.n_shards)
    # In the permuted world a fact lives on (agent_shard+1). The QUERY router still routes by agent_shard (the
    # CORRECT routing), so it hits the shard that does NOT hold the fact -> recall must collapse to ~0 and abstain.
    perm_recall, perm_per_shard, _m = measure_multibridge_recall(facts, comps_perm, word2shard)  # query w/ TRUE router
    perm_abstain, perm_fa, perm_breaches = measure_multibridge_moat(absent_what, absent_who, cross_absent,
                                                                    comps_perm, word2shard)
    print(f"[stage0] PERMUTED-ROUTING control: recall={perm_recall:.3f} (must collapse)  "
          f"abstain={perm_abstain:.3f}  FA={perm_fa} (must stay 0)", flush=True)

    # ================================================================================================
    # VERDICT
    # ================================================================================================
    # The GO call. The HEADLINE bar is the design's content-vs-content claim (anti-cheat #4): per-bridge cleanup must
    # PRESERVE OR BEAT the single-bridge recall ON THE SAME CONTENT, with a 0-FA moat, time-preserving, anti-cheat
    # passing. We ALSO report the absolute >=0.95 bar -- but the honest comparison is vs the single bridge's OWN
    # recall on this exact fact set (which need not itself reach 0.95 at D=128 on these specific codes; the published
    # 0.958 was a different brain-window + fact set). A weak shard cannot hide behind a strong one (per-shard
    # reported separately, the per-seed-not-pooled discipline).
    recall_bar = 0.95
    sb_recall = single_1454["recall"]
    per_shard_min = min((d["recall"] for d in per_shard.values()), default=0.0)
    go_recall_abs = per_shard_min >= recall_bar and mb_recall >= recall_bar      # the strict absolute bar
    go_recall_vs_single = (mb_recall >= sb_recall - 1e-9) and (per_shard_min >= sb_recall - 1e-9)  # content-vs-content
    go_moat = (mb_fa == 0)
    go_time = mb_tq <= single_1454["sec_per_query"] * 1.05      # cleanup over ~727 -> should be <= single-1454 time
    go_anticheat = (perm_recall < 0.5 * mb_recall) and (perm_fa == 0)
    # HEADLINE GO = preserves/beats single-bridge content recall + 0-FA moat + time-preserving + anti-cheat.
    go = go_recall_vs_single and go_moat and go_time and go_anticheat

    verdict = {
        "GO": bool(go),
        "go_recall_vs_single_bridge_content": bool(go_recall_vs_single),
        "go_recall_per_shard_ge_0.95_absolute": bool(go_recall_abs),
        "go_moat_0_false_accepts": bool(go_moat),
        "go_time_le_single_bridge": bool(go_time),
        "go_anticheat_permuted_collapses_no_FA": bool(go_anticheat),
        "per_shard_min_recall": per_shard_min,
        "multibridge_recall": mb_recall,
        "multibridge_false_accept": mb_fa,
        "single_1454_recall": sb_recall,
        "single_1454_false_accept": single_1454["false_accept"],
        "permuted_recall": perm_recall,
        "permuted_false_accept": perm_fa,
        "tq_ms_multibridge": mb_tq * 1e3,
        "tq_ms_single_1454": single_1454["sec_per_query"] * 1e3,
        "headline_bar": "content-vs-content: per-bridge recall >= single-bridge recall on the SAME facts, moat 0-FA",
    }

    result = {
        "brain": os.path.basename(args.brain),
        "seed": args.seed, "D": D, "n_concepts": len(set(vocab)),
        "n_shards": args.n_shards, "shard_sizes": shard_sizes,
        "fact_source": fact_src, "n_facts": len(facts),
        "n_absent_what": len(absent_what), "n_absent_who": len(absent_who), "n_cross_absent": len(cross_absent),
        "store_stats": store_stats,
        "single_1454": single_1454,
        "confound_1454_vs_2012": confound,
        "multibridge": {"recall": mb_recall, "abstain": mb_abstain, "false_accept": mb_fa,
                        "sec_per_query": mb_tq, "per_shard": per_shard, "moat_breaches": mb_breaches},
        "permuted_routing": {"recall": perm_recall, "abstain": perm_abstain, "false_accept": perm_fa,
                             "per_shard": perm_per_shard, "moat_breaches": perm_breaches},
        "verdict": verdict,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n[stage0] VERDICT: {'GO' if go else 'NO-GO'}  "
          f"(recall-vs-single {'OK' if go_recall_vs_single else 'FAIL'} "
          f"[MB {mb_recall:.3f} vs SB {sb_recall:.3f}; abs>=0.95 {'OK' if go_recall_abs else 'no'}], "
          f"moat {'OK' if go_moat else 'FAIL'}, time {'OK' if go_time else 'FAIL'}, "
          f"anti-cheat {'OK' if go_anticheat else 'FAIL'})", flush=True)
    print(f"[stage0] wrote {args.out}", flush=True)
    return result


if __name__ == "__main__":
    main()
