"""EXHAUSTIVE one-fact stub-mouth population: every AFFIRM fact of the production LTM store (wikidata_100k, 78,857 facts) as a one-fact
turn with the template StubRenderer, stock path vs BRAIN_MOAT_SLUG_NORM path, for the two decision functions RichAnswerComposer._verify_rendered
chooses between: stock  = ChatBrain._verify(surface, asserted, svo) (the stub's asserted SVO is canonical, so the vocabulary sets are not read);
flag on = ChatBrain._verify_claim_set(surface, [svo]) over the 1-fact set (if it returns None the composer falls back to the same _verify).
The composer-level sweep (composer_stub_sweep.py, production ChatBrain, _render_one_verified) is run on a seeded 3000-fact sample and the two
must agree fact by fact on that sample (checked in the output: composer_sample_agrees).  HURTS = stock answers, flag-on abstains.
ALLOWLIST ARM (2026-10-20, prereg 2026-10-20-moat-allowlist-claim-checker bar 5): with "allowlist" in --arms the same population is also
decided by ChatBrain._verify_claim_set with BRAIN_MOAT_ALLOWLIST=1 (BRAIN_MOAT_SLUG_NORM unset); flag on, the composer never falls back to
_verify, so a None verdict counts as a rejection.  HURTS_allowlist = stock answers, allowlist abstains.  --arms b2h,allowlist (default)
computes both flag arms; --arms allowlist skips the B2h arm (measured at c6bf28cb7: stub_population_exhaustive_after.json) and leaves its
keys null.  --limit N takes the first N facts (smoke only).
ROLE-REVERSED STOCK LEAKS (prereg AMENDMENT 1): a fact whose relation is ROLE-REVERSING in the reviewed table
(research/runners/_moat_allowlist_relations.py: author, child, father, parent_taxon, platform) has a stub sentence that reads reversed in
English ("The book_of_gen authors moshe_rabbenu."). When stock keeps it and the allowlist rejects it, it is counted under
allowlist_role_reversed_stock_leaks (by relation, with rows), NOT under HURTS_allowlist; the table file's sha256 is recorded.
MEANING-DRIFT STOCK LEAKS (build 2): the same for a MEANING-DRIFT relation (sport, conflict, instrument, spouse, eat: "The X sports Y." = X
wears Y): counted under allowlist_meaning_drift_stock_leaks, beside the role-reversed ones, not under HURTS_allowlist.
BUILD 6 (2026-10-21): the allowlist arm RE-RENDERS each fact with BRAIN_MOAT_ALLOWLIST=1 set, so it decides what production says under the
flag (the stub writes a reviewed shape-4 phrase for the nine no-reading relations under the flag only); the stock arm keeps the flag-off
surface. For builds 1-5 the two renders are identical, so earlier results are unchanged by this. Recorded: allowlist_surface_changed (facts
whose flag-on surface differs) and allowlist_kept_by_no_reading_relation (kept facts of the ROLE-REVERSING / MEANING-DRIFT relations).
usage (cwd = worktree, as a module; refuses to overwrite an existing <out.json>):
  python -m research.findings.raw._moat_slug_norm_heldout.stub_population_exhaustive <out.json> [parser_seed=1] [composer_sweep_sample.json] [--arms b2h,allowlist] [--limit N]
"""
import collections, hashlib, json, os, random, socket, sys, time

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.getcwd())
import research.runners  # noqa: F401,E402  (provenance door)
from research.runners.brain_chat_tui import ChatBrain, StubRenderer           # noqa: E402
from research.runners.brain_conversational_agent import BridgeParser          # noqa: E402
from research.runners._moat_allowlist_relations import MEANING_DRIFT, NO_READING_CLASSES, ROLE_REVERSING, relation_class   # noqa: E402

RELATION_TABLE = "research/runners/_moat_allowlist_relations.py"

FACTS = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"
FLAG = "BRAIN_MOAT_SLUG_NORM"
ALLOW = "BRAIN_MOAT_ALLOWLIST"
SAMPLE = "research/findings/raw/_moat_slug_norm_heldout/composer_stub_sweep_3000.json"


def main(out_path, seed=1, sample_path=SAMPLE, arms=("b2h", "allowlist"), limit=None):
    if os.path.exists(out_path):
        raise SystemExit("refusing to overwrite %s (committed results are never rewritten; pick a new name)" % out_path)
    os.environ.pop(FLAG, None); os.environ.pop(ALLOW, None)
    aff = [(r["fact"]["agent"], r["fact"]["action"], r["fact"]["patient"]) for r in json.load(open(FACTS))
           if r["fact"].get("polarity") == "AFFIRM"]
    n_population = len(aff)
    if limit:
        aff = aff[:limit]
    do_b2h = "b2h" in arms
    do_al = "allowlist" in arms
    chat = ChatBrain.__new__(ChatBrain)
    chat.inner = BridgeParser(seed=seed)
    stub = StubRenderer()
    cells = collections.Counter(); hurts = []; helps = []; why = collections.Counter(); by_rel = collections.Counter(); fb = 0
    acells = collections.Counter(); ahurts = []; ahelps = []; awhy = collections.Counter(); aby_rel = collections.Counter(); anone = 0
    arr = []; arr_rel = collections.Counter(); arr_stock_rejected = collections.Counter()
    amd = []; amd_rel = collections.Counter(); amd_stock_rejected = collections.Counter()
    asurf_changed = 0; ano_reading_kept = collections.Counter(); ano_reading_total = collections.Counter()
    t0 = time.time()
    for i, svo in enumerate(aff):
        surf, asserted = stub.render_svo(*svo)
        os.environ.pop(FLAG, None); os.environ.pop(ALLOW, None)
        off = bool(chat._verify(surf, asserted, list(svo)))
        if do_b2h:
            os.environ[FLAG] = "1"
            acc, res = chat._verify_claim_set(surf, [list(svo)])
            os.environ.pop(FLAG, None)
            if acc is None:
                fb += 1
                on = bool(chat._verify(surf, asserted, list(svo)))
            else:
                on = bool(acc)
            cells[(off, on)] += 1
            if off and not on:
                r = (res or {}).get("reject_reason") or "?"
                hurts.append({"svo": list(svo), "surface": surf, "reject": r})
                why[r[:70]] += 1; by_rel[svo[1]] += 1
            elif on and not off:
                helps.append({"svo": list(svo), "surface": surf})
        if do_al:
            os.environ[ALLOW] = "1"
            asurf, _aasserted = stub.render_svo(*svo)          # build 6: what the stub says under the flag
            aacc, ares = chat._verify_claim_set(asurf, [list(svo)])
            os.environ.pop(ALLOW, None)
            asurf_changed += asurf != surf
            if aacc is None:
                anone += 1                                 # flag on: no fallback, the composer rejects
            aon = bool(aacc)
            acells[(off, aon)] += 1
            cls = relation_class(svo[1])
            if cls in NO_READING_CLASSES:
                ano_reading_total[svo[1]] += 1
                ano_reading_kept[svo[1]] += aon
            if cls == ROLE_REVERSING and not off:
                arr_stock_rejected[svo[1]] += 1
            if cls == MEANING_DRIFT and not off:
                amd_stock_rejected[svo[1]] += 1
            if off and not aon:
                r = (ares or {}).get("reject_reason") or "?"
                if cls == ROLE_REVERSING:                  # a stock leak (reversed English reading), not an answer lost
                    arr.append({"svo": list(svo), "surface": asurf, "reject": r}); arr_rel[svo[1]] += 1
                elif cls == MEANING_DRIFT:                 # a stock leak (an English reading of another relation), not an answer lost
                    amd.append({"svo": list(svo), "surface": asurf, "reject": r}); amd_rel[svo[1]] += 1
                else:
                    ahurts.append({"svo": list(svo), "surface": asurf, "reject": r})
                    awhy[r[:70]] += 1; aby_rel[svo[1]] += 1
            elif aon and not off:
                ahelps.append({"svo": list(svo), "surface": asurf})
        if (i + 1) % 10000 == 0:
            print("[pop] %d/%d t=%.0fs cells %s allowlist cells %s" % (i + 1, len(aff), time.time() - t0, dict(cells), dict(acells)),
                  flush=True)
    agree = None
    if do_b2h and not limit and os.path.exists(sample_path):
        d = json.load(open(sample_path))
        rnd = random.Random()
        random.seed(d["sample_seed"]); sample = random.sample(aff, d["n_facts"])
        hurt_set = {tuple(h["svo"]) for h in hurts}
        comp_hurt = {tuple(h["svo"]) for h in d["hurts_rows"]}
        agree = {"sample_n": len(sample), "composer_hurts": len(comp_hurt),
                 "population_hurts_within_sample": len(hurt_set & set(sample)), "composer_sample_agrees": (hurt_set & set(sample)) == comp_hurt}
    out = {"_doc": "exhaustive one-fact stub-mouth population, stock _verify vs the BRAIN_MOAT_SLUG_NORM claim gate and the "
                   "BRAIN_MOAT_ALLOWLIST claim gate (module docstring)",
           "arms": sorted(arms), "limit": limit, "n_population": n_population, "elapsed_s": round(time.time() - t0, 1),
           "backend": os.environ.get("SIM_BACKEND"), "device": "cpu", "host": socket.gethostname(),
           "script": "research/findings/raw/_moat_slug_norm_heldout/stub_population_exhaustive.py", "argv": sys.argv,
           "parser_seed": seed, "n_facts": len(aff),
           "cells(stock_ok,flag_on_ok)": {str(k): v for k, v in cells.items()} if do_b2h else None,
           "stock_answer_rate": sum(v for k, v in (cells if do_b2h else acells).items() if k[0]) / len(aff),
           "flag_on_answer_rate": sum(v for k, v in cells.items() if k[1]) / len(aff) if do_b2h else None,
           "HURTS": len(hurts) if do_b2h else None, "HELPS": len(helps) if do_b2h else None,
           "flag_on_fallback_to_single_triple": fb if do_b2h else None,
           "hurts_reasons": dict(why), "hurts_by_relation": dict(by_rel.most_common()), "composer_sample_agreement": agree,
           "cells(stock_ok,allowlist_ok)": {str(k): v for k, v in acells.items()} if do_al else None,
           "allowlist_answer_rate": sum(v for k, v in acells.items() if k[1]) / len(aff) if do_al else None,
           "HURTS_allowlist": len(ahurts) if do_al else None, "HELPS_allowlist": len(ahelps) if do_al else None,
           "allowlist_unbuildable_rejected": anone if do_al else None,
           "allowlist_hurts_reasons": dict(awhy), "allowlist_hurts_by_relation": dict(aby_rel.most_common()),
           "allowlist_role_reversed_stock_leaks": len(arr) if do_al else None,
           "allowlist_role_reversed_by_relation": dict(arr_rel.most_common()),
           "allowlist_role_reversed_stock_rejected_by_relation": dict(arr_stock_rejected.most_common()),
           "allowlist_meaning_drift_stock_leaks": len(amd) if do_al else None,
           "allowlist_meaning_drift_by_relation": dict(amd_rel.most_common()),
           "allowlist_meaning_drift_stock_rejected_by_relation": dict(amd_stock_rejected.most_common()),
           "allowlist_surface_rendered_under_flag": True if do_al else None,
           "allowlist_surface_changed": asurf_changed if do_al else None,
           "allowlist_kept_by_no_reading_relation": {r: [ano_reading_kept[r], ano_reading_total[r]] for r in sorted(ano_reading_total)},
           "relation_table": RELATION_TABLE, "relation_table_sha256": hashlib.sha256(open(RELATION_TABLE, "rb").read()).hexdigest(),
           "allowlist_role_reversed_rows": arr, "allowlist_meaning_drift_rows": amd,
           "hurts_rows": hurts, "helps_rows": helps[:200], "allowlist_hurts_rows": ahurts, "allowlist_helps_rows": ahelps[:200]}
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("hurts_rows", "helps_rows", "allowlist_hurts_rows", "allowlist_helps_rows",
                                                                "allowlist_role_reversed_rows", "allowlist_meaning_drift_rows", "argv")}, indent=1)[:3000])


if __name__ == "__main__":
    _args, _kw = [], {}
    _it = iter(sys.argv[1:])
    for _a in _it:
        if _a == "--arms":
            _kw["arms"] = tuple(x for x in next(_it).split(",") if x)
        elif _a == "--limit":
            _kw["limit"] = int(next(_it))
        else:
            _args.append(_a)
    main(_args[0], int(_args[1]) if len(_args) > 1 else 1, _args[2] if len(_args) > 2 else SAMPLE, **_kw)
