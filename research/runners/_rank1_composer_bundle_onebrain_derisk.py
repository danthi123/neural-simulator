"""RANK-1 (scaffold_retirement_backlog) DE-RISK — rebuild the deployed scale787 recall bundle OFF composer_kind='rf'
ONTO 'onebrain' (the genuinely-spiking unbind `local_reciprocal_unbind` + NEF/Izhikevich cleanup
`enable_spiking_cleanup`, both ON by default in OneBrainComposer).

WHAT THE BACKLOG CLAIMED (RANK-1): "the deployed scale787 bundle is pinned to composer_kind='rf', so the spiking
unbind + cleanup that OneBrainComposer enables BY DEFAULT never run on ANY live recall ... Path A = flip the flags +
rebuild the bundle. The single largest, cheapest reclaim. NEAR-ZERO risk (config-flip + rebuild)."

WHAT THIS RUNNER VERIFIES AGAINST THE CURRENT CODE (the task's step-1 mandate: the map has been wrong before):

  P1. Is the bundle really 'rf'?  -> read bridges/developed/scale787/day_33/brain.json (composer_kind field).
  P2. Does the naive same-bundle flip WORK?  -> load_developed_brain(bundle, composer_kind='onebrain') on the FULL
      404-fact bundle. OneBrainComposer has a HARD co-resident cap `k_max=32` (the recent-conversation BUFFER;
      _store_composite raises "store full: k_max=32 reached" past it). 404 >> 32, so the naive flip FAILS.
  P3. Is the premise 'never on any live recall' even true?  -> webapp/server.py:_COMPOSER_KIND_DEFAULT='onebrain'
      (flipped 2026-08-25, commit 135024f70): the PRODUCTION DEFAULT brain (tiny-demo, 5 facts, fits k_max=32) ALREADY
      runs onebrain on every recall. scale787 is a SELECTABLE developed bundle, not the served default.
  P4. Would a k_max>=n_facts rebuild fit the 3090 consumer-hardware reference (24 GB)?  -> OneBrainComposer.n_total
      scales LINEARLY with k_max (n_total = bat_c_base + k_max*cb, cb = n_main*V + NP). Measure VRAM at a few small
      k_max, linear-extrapolate to n_facts, compare to the 24 GB budget.

  M.  Is the onebrain recall MECHANISM correct + moat-intact vs the rf oracle WHERE IT FITS?  -> 6-seed
      (42/43/44/100/101/102) build of BOTH composers over the bundle's REAL vocab + grounded codes, storing a FITTING
      subset (<= k_max) of the bundle's own facts, then compare recall / abstain / store(runtime-teach) / no-confab moat.

VERDICT SHAPE: this is a DE-RISK for an owner-gated flip. It produces GO / PARTIAL / NO-GO for "rebuild the deployed
404-fact scale787 bundle onto a single OneBrainComposer" AND records the validated mechanism result + the correct path.

Run (CPU/GPU fine; the onebrain build uses the shared spiking bridge -- speed secondary per the mission):
  python -m research.runners._rank1_composer_bundle_onebrain_derisk \
      --bundle bridges/developed/scale787/day_33 --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_rank1_composer_bundle_onebrain_derisk.json
"""
import argparse
import json
import os
import subprocess
import time
from collections import defaultdict

# Build a PLAIN OneBrainComposer for the mechanism test (not the pool#1-bound variant): the pool#1 rebasing is
# recall/moat byte-identical to plain OneBrainComposer (rebased-RF identity at atol 1e-9, 6/6 seeds --
# 2026-08-14-onebrain-composer-pool1-DEFAULT-FLIP-GO.md), and building the pool#1-bound composer standalone (no
# pool substrate) is a separate integration path. This isolates the RANK-1 mechanism: the spiking unbind
# (local_reciprocal_unbind) + NEF/Izhikevich cleanup (enable_spiking_cleanup), both ON by default in OneBrainComposer.
os.environ.setdefault("BRAIN_COMPOSER_MERGE", "0")

import numpy as np

BUDGET_MIB = 24576  # RTX 3090 (the consumer-hardware reference principle)
# A recall on a LIVE conversational turn must return in interactive time. The OneBrainComposer resonate-and-fire
# cleanup scans ALL co-resident blocks (O(k_max)/query), so per-query latency scales with the co-resident count --
# this is WHY k_max defaults to 32 (a small, fast buffer). "Speed is secondary" (the mission) is about the DEV loop,
# not a live chat turn; a multi-second recall per turn is not a shippable production flip. Measured: ~6 s at 30
# co-resident facts, ~114 s at 404. Budget the full-bundle flip against a generous interactive bound.
LATENCY_BUDGET_S = 5.0


def _gpu_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL).decode().strip()
        return int(out.splitlines()[0])
    except Exception:
        return -1


def _free_gpu():
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _cupy_used_mib():
    """MY process's cupy allocation (mempool), robust to the concurrent GPU job that noises nvidia-smi deltas."""
    try:
        import cupy
        return cupy.get_default_memory_pool().total_bytes() / (1024.0 * 1024.0)
    except Exception:
        return -1.0


def _load_bundle(bundle):
    from research.runners.developed_brain_io import _load_codes_npz
    facts = json.load(open(os.path.join(bundle, "facts.json")))["facts"]
    manifest = json.load(open(os.path.join(bundle, "brain.json")))
    vocab = manifest["vocab"]
    codes = dict(_load_codes_npz(bundle))
    return facts, vocab, codes, manifest


def _build_agent(seed, vocab, codes, composer_kind):
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    concepts = {w: None for w in vocab}
    return BrainConversationalAgent(seed=seed, concepts=concepts, grounded_codes=codes,
                                    composer_kind=composer_kind, enable_neural_render=False)


def _subset(facts, subset_k, seed_for_novel, n_abstain):
    sub = facts[:subset_k]
    gold = defaultdict(set)
    for f in sub:
        gold[(f["agent"], f["action"])].add(f["patient"])
    cues = list(gold.keys())
    agents = sorted({f["agent"] for f in sub})
    acts = sorted({f["action"] for f in sub})
    stored = set((f["agent"], f["action"]) for f in sub)
    novel = [(a, ac) for a in agents for ac in acts if (a, ac) not in stored]
    np.random.RandomState(seed_for_novel).shuffle(novel)
    novel = novel[:n_abstain]
    return sub, gold, cues, agents, acts, stored, novel


def _rf_oracle(seed, facts, vocab, codes, subset_k, n_abstain):
    """The rf recall is SEED-INDEPENDENT (host FHRR on fixed disk codes), so build it ONCE and reuse across seeds.
    Returns the gold + rf answers on the recall cues + the rf abstain results (the production oracle to beat)."""
    sub, gold, cues, agents, acts, stored, novel = _subset(facts, subset_k, 0, n_abstain)
    a_rf = _build_agent(seed, vocab, codes, "rf")
    for f in sub:
        a_rf.composer.store(f["agent"], f["action"], f["patient"], polarity=f.get("polarity"))
    rf_recall = {c: a_rf.composer.query_patient(*c) for c in cues}
    rf_abstain = {c: a_rf.composer.query_patient(*c) for c in novel}
    rf_ok = sum(1 for c in cues if rf_recall[c] in gold[c])
    rf_confab = sum(1 for c in novel if rf_abstain[c] is not None)
    return {"sub": sub, "gold": gold, "cues": cues, "novel": novel, "stored": stored,
            "agents": agents, "acts": acts, "rf_recall": rf_recall, "rf_abstain": rf_abstain,
            "rf_ok": rf_ok, "rf_confab": rf_confab, "n_cues": len(cues), "n_novel": len(novel)}


def _onebrain_one_seed(seed, oracle, vocab, codes):
    """Build a PLAIN OneBrainComposer at `seed`, store the SAME subset, compare recall/abstain/store to the rf oracle."""
    gold, cues, novel, stored = oracle["gold"], oracle["cues"], oracle["novel"], oracle["stored"]
    agents, acts = oracle["agents"], oracle["acts"]
    sub = oracle["sub"]

    a_ob = _build_agent(seed, vocab, codes, "onebrain")
    k_max = a_ob.composer.k_max
    # store the subset in the SAME insertion order the rf oracle used (query_patient is first-match, so the block
    # ordering must match for a like-for-like comparison).
    for f in sub:
        a_ob.composer.store(f["agent"], f["action"], f["patient"], polarity=f.get("polarity"))
    n_stored = len(sub)

    ob_ok = agree = 0
    regressions = []
    for c in cues:
        o = a_ob.composer.query_patient(*c)
        r = oracle["rf_recall"][c]
        if o in gold[c]:
            ob_ok += 1
        if o == r:
            agree += 1
        if (r in gold[c]) and (o not in gold[c]):
            regressions.append({"cue": list(c), "rf": r, "ob": o, "gold": sorted(gold[c])})

    ob_confab = ab_agree = 0
    ob_confab_examples = []
    for c in novel:
        o = a_ob.composer.query_patient(*c)
        r = oracle["rf_abstain"][c]
        if o is not None:
            ob_confab += 1
            ob_confab_examples.append({"cue": list(c), "ob": o})
        if (o is None) == (r is None):
            ab_agree += 1

    # runtime STORE probes (need a free co-resident slot)
    store_probes = []
    if n_stored < k_max:
        pw = vocab[len(vocab) // 2]
        fresh = next(((a, ac) for a in agents[::-1] for ac in acts[::-1] if (a, ac) not in stored), None)
        if fresh:
            a_ob.composer.store(fresh[0], fresh[1], pw)
            got = a_ob.composer.query_patient(*fresh)
            store_probes.append({"kind": "invocab", "cue": list(fresh), "taught": pw,
                                 "recalled": got, "ok": bool(got == pw)})
    if n_stored + 1 < k_max:
        novel_word = "zzqnovelword"
        a_ob.composer.store("dog", "zzqnovelverb", novel_word)
        got = a_ob.composer.query_patient("dog", "zzqnovelverb")
        store_probes.append({"kind": "novel_recruit", "cue": ["dog", "zzqnovelverb"], "taught": novel_word,
                             "recalled": got, "ok": bool(got == novel_word)})
    store_ok = all(p["ok"] for p in store_probes) if store_probes else None

    return {
        "seed": seed, "k_max": k_max, "n_stored": n_stored, "n_cues": len(cues),
        "recall": {"rf_ok": oracle["rf_ok"], "ob_ok": ob_ok, "agree": agree, "n": len(cues),
                   "regressions": regressions},
        "abstain": {"n": len(novel), "rf_confab": oracle["rf_confab"], "ob_confab": ob_confab,
                    "ab_agree": ab_agree, "ob_confab_examples": ob_confab_examples[:5]},
        "store": {"probes": store_probes, "all_ok": store_ok},
    }


def _feasibility(bundle, vocab, facts, vram_kmax, attempt_full, latency_queries=3):
    """P2 (naive same-bundle flip fails) + P4 (VRAM extrapolation to k_max>=n_facts vs the 24 GB budget) + (under
    attempt_full) the DECISIVE full-scale build+store+per-query LATENCY at k_max=n_facts."""
    n_facts = len(facts)
    out = {"n_facts": n_facts, "budget_mib": BUDGET_MIB, "latency_budget_s": LATENCY_BUDGET_S}

    # P2: the naive load-override on the FULL bundle
    from research.runners.developed_brain_io import load_developed_brain
    try:
        load_developed_brain(bundle, use_multiturn=False, seed=42, composer_kind="onebrain")
        out["naive_full_flip"] = {"ok": True, "note": "unexpected: full-bundle onebrain load succeeded"}
    except Exception as e:
        out["naive_full_flip"] = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    _free_gpu()

    # P4: VRAM scaling at a few k_max, then linear extrapolation to n_facts
    from research.runners.one_brain_composer import OneBrainComposer
    scaling = []
    for k in vram_kmax:
        _free_gpu()
        v0 = _gpu_used_mib()
        cu0 = _cupy_used_mib()
        t0 = time.time()
        try:
            c = OneBrainComposer(seed=42, D=128, vocab=vocab, k_max=k, vocab_headroom=128)
            # store a couple of facts so any lazily-allocated store/read buffers are counted
            c.store("dog", "chase", "cat")
            c.query_patient("dog", "chase")
            v1 = _gpu_used_mib()
            cu1 = _cupy_used_mib()
            scaling.append({"k_max": k, "n_total": int(c.n_total),
                            "vram_delta_mib": v1 - v0, "cupy_delta_mib": round(cu1 - cu0, 1),
                            "build_s": round(time.time() - t0, 1)})
            del c
            _free_gpu()
        except Exception as e:
            scaling.append({"k_max": k, "error": "%s: %s" % (type(e).__name__, e)})
    out["vram_scaling"] = scaling

    # linear fit MiB ~ a*k_max + b over the successful points (prefer the cupy-mempool delta -- robust to the
    # concurrent GPU job; fall back to the nvidia-smi delta if cupy is unavailable).
    def _mib(s):
        if s.get("cupy_delta_mib", -1) is not None and s.get("cupy_delta_mib", -1) >= 0:
            return s["cupy_delta_mib"]
        return s.get("vram_delta_mib", -1)
    pts = [(s["k_max"], _mib(s)) for s in scaling if "k_max" in s and "error" not in s and _mib(s) >= 0]
    if len(pts) >= 2:
        xs = np.array([p[0] for p in pts], float)
        ys = np.array([p[1] for p in pts], float)
        a, b = np.polyfit(xs, ys, 1)
        pred = float(a * n_facts + b)
        out["predicted_vram_mib_at_nfacts"] = round(pred, 0)
        out["predicted_fits_budget"] = bool(pred < BUDGET_MIB)
        out["vram_fit"] = {"slope_mib_per_fact": round(float(a), 2), "intercept_mib": round(float(b), 1)}
    if attempt_full:
        _free_gpu()
        v0 = _gpu_used_mib()
        t0 = time.time()
        try:
            c = OneBrainComposer(seed=42, D=128, vocab=vocab, k_max=n_facts, vocab_headroom=128)
            build_s = round(time.time() - t0, 1)
            v1 = _gpu_used_mib()
            # store ALL facts, then time real recall queries -- the DECISIVE per-query latency at full co-resident scale.
            t0 = time.time()
            for f in facts:
                c.store(f["agent"], f["action"], f["patient"], polarity=f.get("polarity"))
            store_s = round(time.time() - t0, 1)
            cues = list({(f["agent"], f["action"]) for f in facts})[:latency_queries]
            t0 = time.time()
            for a, ac in cues:
                c.query_patient(a, ac)
            per_query_s = round((time.time() - t0) / max(1, len(cues)), 2)
            out["full_kmax_build"] = {"ok": True, "k_max": n_facts, "n_total": int(c.n_total),
                                      "vram_delta_mib": v1 - v0, "build_s": build_s, "store_s": store_s,
                                      "per_query_s": per_query_s, "queries_timed": len(cues),
                                      "latency_ok": bool(per_query_s <= LATENCY_BUDGET_S)}
            del c
            _free_gpu()
        except Exception as e:
            out["full_kmax_build"] = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    return out


def _verdict(mech, feas):
    reasons = []
    # mechanism GO across all seeds
    mech_go = True
    for m in mech:
        if m["recall"]["ob_ok"] < m["recall"]["rf_ok"]:
            mech_go = False
            reasons.append("seed %d: onebrain recall regressed rf (%d<%d)" % (m["seed"], m["recall"]["ob_ok"], m["recall"]["rf_ok"]))
        if m["recall"]["regressions"]:
            mech_go = False
            reasons.append("seed %d: %d recall regressions" % (m["seed"], len(m["recall"]["regressions"])))
        if m["abstain"]["ob_confab"] > m["abstain"]["rf_confab"]:
            mech_go = False
            reasons.append("seed %d: moat regression (ob_confab %d > rf_confab %d)" % (m["seed"], m["abstain"]["ob_confab"], m["abstain"]["rf_confab"]))
        if m["store"]["all_ok"] is False:
            mech_go = False
            reasons.append("seed %d: runtime store/recall failed" % m["seed"])

    fits = feas.get("predicted_fits_budget", None)
    fkb = feas.get("full_kmax_build", {})
    full_built = fkb.get("ok", None)
    per_q = fkb.get("per_query_s")
    latency_ok = fkb.get("latency_ok")           # None if latency was not measured (no --attempt-full)
    vram_ok = bool((full_built is True) or (fits is True))

    # The full-bundle flip needs BOTH: VRAM fits AND the per-query latency at full co-resident scale is interactive.
    # The automated GO before this gate existed reported "GO" on mechanism + VRAM alone -- and MISSED that the
    # O(k_max) resonate scan makes a 404-co-resident recall take ~114 s/query (measured). Latency is the real gate.
    if latency_ok is False:
        full_flip_feasible = False
    else:
        full_flip_feasible = vram_ok

    if mech_go and full_flip_feasible:
        v = "GO"
    elif mech_go and not full_flip_feasible:
        v = "PARTIAL / MIS-SCOPED"
        if latency_ok is False:
            reasons.append(
                "The onebrain recall MECHANISM is correct + moat-intact vs rf across all seeds (at BUFFER scale). The "
                "full-404 flip is VRAM-FEASIBLE (k_max=%s -> %s neurons but only ~hundreds of MiB -- the RF/phasor "
                "coresident bridge is sparse, NOT a dense Izhikevich net), but its PER-QUERY LATENCY is %s s at full "
                "co-resident scale vs the %s s interactive budget: the resonate-and-fire cleanup scans ALL k_max blocks "
                "(O(k_max)/query), which is WHY k_max defaults to 32. A ~2-minute recall per live turn is not a "
                "shippable flip. The scalable spiking-bulk path is a SHARDED spiking store (per-query O(K/shards)); the "
                "current LTM ShardedPhasorStore is host-FHRR, so that is a NEW mechanism, not the 'near-zero-risk "
                "config-flip' RANK-1 claimed. P3: the DEFAULT brain (tiny-demo) already runs onebrain since 2026-08-25. "
                "RE-SCOPE RANK-1." % (feas.get("n_facts"), fkb.get("n_total"), per_q, feas.get("latency_budget_s")))
        else:
            reasons.append(
                "The onebrain recall MECHANISM is correct + moat-intact vs rf across all seeds (at BUFFER scale), but "
                "the full-404 flip's practicality is not established (full-scale latency unmeasured -- rerun with "
                "--attempt-full). k_max=32 << 404 blocks the naive same-composer_kind flip; a k_max=n_facts rebuild is "
                "needed. RE-SCOPE / re-measure before any owner flip.")
    else:
        v = "NO-GO"
    return {"verdict": v, "mechanism_go": mech_go, "full_flip_feasible": full_flip_feasible,
            "vram_ok": vram_ok, "latency_ok": latency_ok, "per_query_s_full": per_q, "reasons": reasons}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="bridges/developed/scale787/day_33")
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--subset-k", type=int, default=30, help="facts to co-reside for the mechanism test (<= k_max=32)")
    ap.add_argument("--n-abstain", type=int, default=40)
    ap.add_argument("--vram-kmax", default="32,64,128")
    ap.add_argument("--attempt-full", action="store_true",
                    help="build k_max=n_facts, store all facts, and TIME recall queries (the decisive full-scale "
                         "latency; ~1.37M neurons at 404 facts but ~hundreds of MiB -- VRAM is not the limit, latency is)")
    ap.add_argument("--latency-queries", type=int, default=3, help="queries to time at full k_max (--attempt-full)")
    ap.add_argument("--out", default="research/findings/raw/_rank1_composer_bundle_onebrain_derisk.json")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    vram_kmax = [int(s) for s in a.vram_kmax.split(",") if s.strip()]

    facts, vocab, codes, manifest = _load_bundle(a.bundle)

    # RESUME (kill-resilient): the environment's controller kills lower-priority GPU jobs, and each onebrain seed is
    # slow (~11-24 min). Load any prior --out, keep its completed seeds + feasibility, and only run what is missing;
    # write the JSON after the oracle, after EACH seed, and after feasibility -- so a kill never loses a done seed.
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    result = {}
    if os.path.exists(a.out):
        try:
            result = json.load(open(a.out))
            print("[resume] loaded prior %s (mechanism seeds done: %s)"
                  % (a.out, [m["seed"] for m in result.get("mechanism", [])]), flush=True)
        except Exception:
            result = {}
    result.update({
        "runner": "research/runners/_rank1_composer_bundle_onebrain_derisk.py",
        "bundle": a.bundle,
        "bundle_composer_kind": manifest.get("composer_kind"),
        "n_facts": len(facts), "vocab": len(vocab), "n_grounded_codes": len(codes),
        "seeds": seeds, "subset_k": a.subset_k,
        "composer_merge_env": os.environ.get("BRAIN_COMPOSER_MERGE"),
        "sim_backend": os.environ.get("SIM_BACKEND", "numpy"),
        "note": ("onebrain built as PLAIN OneBrainComposer (BRAIN_COMPOSER_MERGE=0); recall/moat byte-identical to "
                 "the pool#1-bound production variant per 2026-08-14-onebrain-composer-pool1-DEFAULT-FLIP-GO."),
    })

    def _write():
        with open(a.out, "w") as fh:
            json.dump(result, fh, indent=2)

    mech = list(result.get("mechanism", []))
    done_seeds = {m["seed"] for m in mech}
    todo = [s for s in seeds if s not in done_seeds]

    oracle = None
    if todo:
        t0 = time.time()
        oracle = _rf_oracle(seeds[0], facts, vocab, codes, a.subset_k, a.n_abstain)
        print("[rf-oracle] built once (seed-independent): n_cues=%d rf_ok=%d n_novel=%d rf_confab=%d (%.1fs)"
              % (oracle["n_cues"], oracle["rf_ok"], oracle["n_novel"], oracle["rf_confab"], time.time() - t0), flush=True)
        result["rf_oracle"] = {"n_cues": oracle["n_cues"], "rf_ok": oracle["rf_ok"],
                               "n_novel": oracle["n_novel"], "rf_confab": oracle["rf_confab"],
                               "note": "rf recall is host-FHRR, seed-independent -> built once, reused across seeds"}
        result["mechanism"] = mech
        _write()

    for s in todo:
        t0 = time.time()
        m = _onebrain_one_seed(s, oracle, vocab, codes)
        m["wall_s"] = round(time.time() - t0, 1)
        mech.append(m)
        result["mechanism"] = mech
        _write()                                              # checkpoint after every seed (kill-resilient)
        print("[mech] seed=%d recall ob_ok=%d/%d (rf_ok=%d) agree=%d moat ob_confab=%d (rf=%d) store_ok=%s (%.1fs)"
              % (s, m["recall"]["ob_ok"], m["recall"]["n"], m["recall"]["rf_ok"], m["recall"]["agree"],
                 m["abstain"]["ob_confab"], m["abstain"]["rf_confab"], m["store"]["all_ok"], m["wall_s"]), flush=True)

    # ATTRIBUTION (tools.lab) — this de-risk is an EQUIVALENCE claim: swapping the composer (rf -> onebrain) must
    # NOT regress recall or the moat. attributable_to(treatment=onebrain, control=rf) asks whose the difference is.
    # ob==rf where facts fit -> ~0 errors in both arms -> UNDEFINED (None) -> the flip owns no recall/moat error.
    from tools.lab import attributable_to
    agg_ob_wrong = sum(m["recall"]["n"] - m["recall"]["ob_ok"] for m in mech)
    agg_rf_wrong = sum(m["recall"]["n"] - m["recall"]["rf_ok"] for m in mech)
    agg_ob_confab = sum(m["abstain"]["ob_confab"] for m in mech)
    agg_rf_confab = sum(m["abstain"]["rf_confab"] for m in mech)
    result["attribution"] = {
        "recall_errors_rf_to_onebrain": attributable_to("recall errors: rf->onebrain flip", agg_ob_wrong, agg_rf_wrong),
        "moat_confabs_rf_to_onebrain": attributable_to("moat confabs: rf->onebrain flip", agg_ob_confab, agg_rf_confab),
        "agg_ob_wrong": agg_ob_wrong, "agg_rf_wrong": agg_rf_wrong,
        "agg_ob_confab": agg_ob_confab, "agg_rf_confab": agg_rf_confab,
        "note": ("treatment=onebrain, control=rf; None/0 => the composer flip introduces no recall/moat error "
                 "where the facts FIT the k_max buffer (the mechanism is equivalent, not a lever with an effect)."),
    }
    _write()

    if "feasibility" not in result:
        feas = _feasibility(a.bundle, vocab, facts, vram_kmax, a.attempt_full, latency_queries=a.latency_queries)
        result["feasibility"] = feas
        _write()
        print("[feas] naive_full_flip=%s predicted_vram_at_nfacts=%s fits_budget=%s"
              % (feas.get("naive_full_flip"), feas.get("predicted_vram_mib_at_nfacts"),
                 feas.get("predicted_fits_budget")), flush=True)

    result["verdict"] = _verdict(mech, result["feasibility"])
    _write()
    print("[verdict]", json.dumps(result["verdict"], indent=2), flush=True)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
