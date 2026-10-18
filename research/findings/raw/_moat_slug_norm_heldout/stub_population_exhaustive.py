"""EXHAUSTIVE one-fact stub-mouth population: every AFFIRM fact of the production LTM store (wikidata_100k, 78,857 facts) as a one-fact
turn with the template StubRenderer, stock path vs BRAIN_MOAT_SLUG_NORM path, for the two decision functions RichAnswerComposer._verify_rendered
chooses between: stock  = ChatBrain._verify(surface, asserted, svo) (the stub's asserted SVO is canonical, so the vocabulary sets are not read);
flag on = ChatBrain._verify_claim_set(surface, [svo]) over the 1-fact set (if it returns None the composer falls back to the same _verify).
The composer-level sweep (composer_stub_sweep.py, production ChatBrain, _render_one_verified) is run on a seeded 3000-fact sample and the two
must agree fact by fact on that sample (checked in the output: composer_sample_agrees).  HURTS = stock answers, flag-on abstains.
usage (cwd = worktree, as a module): python -m research.findings.raw._moat_slug_norm_heldout.stub_population_exhaustive <out.json> [parser_seed=1]
"""
import collections, json, os, random, socket, sys, time

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.getcwd())
import research.runners  # noqa: F401,E402  (provenance door)
from research.runners.brain_chat_tui import ChatBrain, StubRenderer           # noqa: E402
from research.runners.brain_conversational_agent import BridgeParser          # noqa: E402

FACTS = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"
FLAG = "BRAIN_MOAT_SLUG_NORM"
SAMPLE = "research/findings/raw/_moat_slug_norm_heldout/composer_stub_sweep_3000.json"


def main(out_path, seed=1):
    aff = [(r["fact"]["agent"], r["fact"]["action"], r["fact"]["patient"]) for r in json.load(open(FACTS))
           if r["fact"].get("polarity") == "AFFIRM"]
    chat = ChatBrain.__new__(ChatBrain)
    chat.inner = BridgeParser(seed=seed)
    stub = StubRenderer()
    cells = collections.Counter(); hurts = []; helps = []; why = collections.Counter(); by_rel = collections.Counter(); fb = 0
    t0 = time.time()
    for i, svo in enumerate(aff):
        surf, asserted = stub.render_svo(*svo)
        os.environ.pop(FLAG, None)
        off = bool(chat._verify(surf, asserted, list(svo)))
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
        if (i + 1) % 10000 == 0:
            print("[pop] %d/%d t=%.0fs cells %s" % (i + 1, len(aff), time.time() - t0, dict(cells)), flush=True)
    agree = None
    if os.path.exists(SAMPLE):
        d = json.load(open(SAMPLE))
        rnd = random.Random()
        random.seed(d["sample_seed"]); sample = random.sample(aff, d["n_facts"])
        hurt_set = {tuple(h["svo"]) for h in hurts}
        comp_hurt = {tuple(h["svo"]) for h in d["hurts_rows"]}
        agree = {"sample_n": len(sample), "composer_hurts": len(comp_hurt),
                 "population_hurts_within_sample": len(hurt_set & set(sample)), "composer_sample_agrees": (hurt_set & set(sample)) == comp_hurt}
    out = {"_doc": "exhaustive one-fact stub-mouth population, stock _verify vs the BRAIN_MOAT_SLUG_NORM claim gate (module docstring)",
           "backend": os.environ.get("SIM_BACKEND"), "device": "cpu", "host": socket.gethostname(),
           "script": "research/findings/raw/_moat_slug_norm_heldout/stub_population_exhaustive.py", "argv": sys.argv,
           "parser_seed": seed, "n_facts": len(aff), "cells(stock_ok,flag_on_ok)": {str(k): v for k, v in cells.items()},
           "stock_answer_rate": sum(v for k, v in cells.items() if k[0]) / len(aff),
           "flag_on_answer_rate": sum(v for k, v in cells.items() if k[1]) / len(aff),
           "HURTS": len(hurts), "HELPS": len(helps), "flag_on_fallback_to_single_triple": fb,
           "hurts_reasons": dict(why), "hurts_by_relation": dict(by_rel.most_common()), "composer_sample_agreement": agree,
           "hurts_rows": hurts, "helps_rows": helps[:200]}
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("hurts_rows", "helps_rows", "argv")}, indent=1)[:3000])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
