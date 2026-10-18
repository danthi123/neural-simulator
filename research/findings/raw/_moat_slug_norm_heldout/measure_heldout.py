"""Measure the claim verifier on the HELD-OUT corpus against the hand labels that were committed BEFORE this ran.
Verdicts come from the production wiring: ChatBrain._build_claim_verifier / _verify_claim_set with BRAIN_MOAT_SLUG_NORM unset
(stock) and =1 (B2h), over a real BridgeParser at each seed.  A sentence is "kept" iff the claim gate accepts it for its
turn's gathered set.  Bar (set before this ran): 0 sentences labelled U kept under B2h.  B kept is reported separately.
usage (cwd = worktree): measure_heldout.py <corpus.json> <labels.json> <out.json>"""
import collections, json, os, sys

os.environ["SIM_BACKEND"] = "numpy"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unique_sentences import unique                                              # noqa: E402
from research.runners.brain_chat_tui import ChatBrain                            # noqa: E402
from research.runners.brain_conversational_agent import BridgeParser             # noqa: E402

FLAG = "BRAIN_MOAT_SLUG_NORM"
SEEDS = [42, 43, 44, 100, 101, 102, 1]        # 1 = the seed the corpus-generation brain used (BRAIN_CHAT_SEED=1)
corpus_path, labels_path, out_path = sys.argv[1:4]
corpus = json.load(open(corpus_path))
lab_doc = json.load(open(labels_path))
uniq = unique(corpus)
lab = {}
for e in lab_doc["labels"]:
    lab[(e["turn"], tuple(e["svo"]), e["surface"])] = e["label"]
missing = [u for u in uniq if (u["turn"], tuple(u["svo"]), u["surface"]) not in lab]
extra = set(lab) - {(u["turn"], tuple(u["svo"]), u["surface"]) for u in uniq}
assert not missing and not extra, ("labels do not cover the corpus", len(missing), len(extra))
assert all(v in "FBUN" and len(v) == 1 for v in lab.values())
turns = [t for t in corpus["turns"] if t.get("calls")]


def run(seed, flag_on):
    if flag_on:
        os.environ[FLAG] = "1"
    else:
        os.environ.pop(FLAG, None)
    chat = ChatBrain.__new__(ChatBrain)
    chat.inner = BridgeParser(seed=seed)
    verdict = {}
    for t in turns:
        for c in t["calls"]:
            k = (t["label"], tuple(c["svo"]), c["surface"], c["method"])
            acc, res = chat._verify_claim_set(c["surface"], t["gated"])
            verdict[k] = (acc, (res or {}).get("reject_reason"))
    os.environ.pop(FLAG, None)
    return verdict


def summarise(verdict):
    raw = collections.Counter(); uni = collections.Counter(); seen = set(); kept_rows = []
    dist = collections.Counter(); seen2 = set()          # distinct (svo, surface) regardless of turn (shared facts repeat across turns)
    for (turn, svo, surface, method), (acc, why) in verdict.items():
        l = lab[(turn, svo, surface)]
        raw[(l, bool(acc))] += 1
        if (svo, surface, bool(acc)) not in seen2:
            seen2.add((svo, surface, bool(acc))); dist[(l, bool(acc))] += 1
        if (turn, svo, surface) not in seen:
            seen.add((turn, svo, surface)); uni[(l, bool(acc))] += 1
            if acc:
                kept_rows.append({"turn": turn, "svo": list(svo), "surface": surface, "label": l})
    tot_raw = collections.Counter(); tot_uni = collections.Counter()
    for (l, a), n in raw.items(): tot_raw[l] += n
    for (l, a), n in uni.items(): tot_uni[l] += n
    # composer semantics per turn: per fact, render_svo then render_svo_regen, first accepted is kept
    facts = kept_facts = 0; kept_fact_labels = collections.Counter(); turns_answered = 0
    for t in turns:
        by = collections.OrderedDict()
        for c in t["calls"]:
            by.setdefault(tuple(c["svo"]), []).append(c)
        got = 0
        for svo, cs in by.items():
            facts += 1
            for c in sorted(cs, key=lambda c: 0 if c["method"] == "render_svo" else 1):
                if verdict[(t["label"], tuple(c["svo"]), c["surface"], c["method"])][0]:
                    kept_facts += 1; got += 1
                    kept_fact_labels[lab[(t["label"], tuple(c["svo"]), c["surface"])]] += 1
                    break
        turns_answered += bool(got)
    return {"raw_rows": sum(tot_raw.values()), "unique_sentences": sum(tot_uni.values()),
            "label_totals_raw": dict(tot_raw), "label_totals_unique": dict(tot_uni),
            "kept_raw": {l: raw[(l, True)] for l in "FBUN"}, "kept_unique": {l: uni[(l, True)] for l in "FBUN"},
            "distinct_svo_surface_kept": {l: dist[(l, True)] for l in "FBUN"},
            "distinct_svo_surface_total": sum(n for (l, a), n in dist.items()),
            "kept_raw_total": sum(raw[(l, True)] for l in "FBUN"), "kept_unique_total": sum(uni[(l, True)] for l in "FBUN"),
            "facts": facts, "facts_kept": kept_facts, "facts_kept_by_label": dict(kept_fact_labels),
            "turns": len(turns), "turns_answered": turns_answered, "kept_sentences": kept_rows}


res = {"_doc": "B2h (BRAIN_MOAT_SLUG_NORM=1) vs stock on the held-out corpus; labels committed before this ran",
       "backend": "numpy", "device": "cpu", "script": "research/findings/raw/_moat_slug_norm_heldout/measure_heldout.py",
       "argv": sys.argv, "seeds": SEEDS, "corpus": corpus_path, "labels": labels_path, "per_seed": {}}
for seed in SEEDS:
    off = summarise(run(seed, False)); on = summarise(run(seed, True))
    res["per_seed"][str(seed)] = {"stock": {k: v for k, v in off.items() if k != "kept_sentences"}, "b2h": on}
    print("seed %3d | stock kept %d/%d | B2h kept raw %d (unique %d) | by label raw %s | unique %s | facts %d/%d | turns answered %d/%d" % (
        seed, off["kept_raw_total"], off["raw_rows"], on["kept_raw_total"], on["kept_unique_total"], on["kept_raw"],
        on["kept_unique"], on["facts_kept"], on["facts"], on["turns_answered"], on["turns"]), flush=True)
leaks = collections.OrderedDict()
for seed, r in res["per_seed"].items():
    for k in r["b2h"]["kept_sentences"]:
        if k["label"] in "UN":
            leaks.setdefault((k["turn"], tuple(k["svo"]), k["surface"]), []).append(int(seed))
res["unsupported_kept"] = [{"turn": a, "svo": list(b), "surface": c, "seeds": s} for (a, b, c), s in leaks.items()]
res["bar_zero_unsupported_kept"] = not leaks
json.dump(res, open(out_path, "w"), indent=1)
print("\nBAR (0 U/N kept under B2h at every seed): %s" % ("PASS" if not leaks else "FAIL"))
for (a, b, c), s in leaks.items():
    print("  LEAK turn %s svo %s seeds %s :: %s" % (a, list(b), s, c))
