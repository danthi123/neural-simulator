"""COMPOSER-PATH sweep, stub mouth: every AFFIRM fact of the production LTM store as a ONE-fact turn through the production
RichAnswerComposer._render_one_verified (-> _verify_rendered), flag BRAIN_MOAT_SLUG_NORM off vs on, on the REAL production ChatBrain
(webapp.server._build_chat_brain('tiny-demo','stub'): the brain id and GPU-free stub renderer a GPU-less production console runs;
the stub renderer returns the canonical asserted SVO, so the stock path is the single-triple chat._verify and the flag-on path is the
claim gate over the 1-fact set).  Per fact: the stock verdict, the flag-on verdict, and for flag-on which sub-verifier decided (claim
gate vs the chat._verify fallback when the claim verifier is unbuildable) and its reject reason.
HURTS = stock answers, flag-on abstains (the stub-inflection cases 'sports', 'countrys', 'shares_border_withs' land here).
usage (cwd = worktree, as a module):  python -m research.findings.raw._moat_slug_norm_heldout.composer_stub_sweep <out.json> [n=0 (all)] [seed=20261018]
"""
import collections, json, os, platform, random, socket, sys, time

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["BRAIN_CHAT_SEED"] = "1"
os.environ.pop("BRAIN_MOAT_SLUG_NORM", None)
sys.path.insert(0, os.getcwd())
import research.runners  # noqa: F401,E402  (provenance door)

FLAG = "BRAIN_MOAT_SLUG_NORM"
FACTS = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"


def main(out_path, n=0, seed=20261018):
    import webapp.server as S
    from research.runners.rich_answer_composer import RichAnswerComposer
    from research.runners.brain_chat_tui import ChatBrain
    t0 = time.time()
    chat, source = S._build_chat_brain("tiny-demo", "stub")
    rich = RichAnswerComposer(chat, max_sentences=4, neural_planner=False)
    build_s = time.time() - t0
    seen = {}
    orig_vc = ChatBrain._verify_claim_set

    def vc(self, surface, gated_facts):
        r = orig_vc(self, surface, gated_facts)
        seen["claim"] = (None if r[0] is None else bool(r[0]), (r[1] or {}).get("reject_reason"))
        return r
    ChatBrain._verify_claim_set = vc
    aff = [(r["fact"]["agent"], r["fact"]["action"], r["fact"]["patient"]) for r in json.load(open(FACTS))
           if r["fact"].get("polarity") == "AFFIRM"]
    if n:
        random.seed(seed)
        aff = random.sample(aff, n)
    cells = collections.Counter(); hurts = []; helps = []; reasons = collections.Counter(); fallback = 0
    for i, svo in enumerate(aff):
        g = [list(svo)]
        os.environ.pop(FLAG, None)
        seen.clear()
        off_sent, off_ok = rich._render_one_verified(list(svo), gated=g)
        os.environ[FLAG] = "1"
        seen.clear()
        on_sent, on_ok = rich._render_one_verified(list(svo), gated=g)
        claim = seen.get("claim")
        os.environ.pop(FLAG, None)
        cells[(bool(off_ok), bool(on_ok))] += 1
        if claim is not None and claim[0] is None:
            fallback += 1
        if off_ok and not on_ok:
            hurts.append({"svo": list(svo), "stock_surface": off_sent, "flag_on_claim": claim})
            reasons[(claim[1] if claim else "claim gate not called")[:70]] += 1
        elif on_ok and not off_ok:
            helps.append({"svo": list(svo), "surface": on_sent})
        if (i + 1) % 1000 == 0:
            print("[sweep] %d/%d t=%.0fs cells %s" % (i + 1, len(aff), time.time() - t0, dict(cells)), flush=True)
    out = {"_doc": "composer-path one-fact sweep, stub mouth, production tiny-demo+LTM brain; rule in the module docstring",
           "backend": os.environ.get("SIM_BACKEND"), "device": "cpu (%s)" % (platform.processor() or platform.machine()),
           "host": socket.gethostname(), "script": "research/findings/raw/_moat_slug_norm_heldout/composer_stub_sweep.py",
           "argv": sys.argv, "brain_source": source, "brain_build_s": round(build_s, 1),
           "vocab_sizes": [len(chat.agents_set), len(chat.actions_set), len(chat.patients_set)],
           "parser_seed": getattr(chat.inner, "seed", None), "n_facts": len(aff), "sample_seed": seed if n else None,
           "cells(stock_ok,flag_on_ok)": {str(k): v for k, v in cells.items()},
           "stock_answer_rate": sum(v for k, v in cells.items() if k[0]) / len(aff),
           "flag_on_answer_rate": sum(v for k, v in cells.items() if k[1]) / len(aff),
           "HURTS": len(hurts), "HELPS": len(helps), "flag_on_fallback_to_single_triple": fallback,
           "hurts_reasons": dict(reasons), "hurts_rows": hurts, "helps_rows": helps[:200]}
    json.dump(out, open(out_path, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k not in ("hurts_rows", "helps_rows")}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0, int(sys.argv[3]) if len(sys.argv) > 3 else 20261018)
