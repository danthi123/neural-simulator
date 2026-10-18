"""Measure the claim verifier, stock vs BRAIN_MOAT_SLUG_NORM=1 ("b2h"), on every labelled held-out stratum, against labels that were
committed BEFORE this ran.  Verdicts come from the production wiring: ChatBrain._build_claim_verifier / _verify_claim_set over a real
BridgeParser at each parser seed.  A sentence is KEPT iff the claim gate accepts it for its turn's gathered set; REJECTED if it
refuses; UNBUILDABLE if the gathered set has a role-permutation collision (the verifier cannot be built, _verify_claim_set returns
(None, None), production falls back to the single-triple chat._verify -- flag-independent, not decided by this flag, counted apart).

Two readings of the bar (both reported):  lenient = a kept sentence labelled U or N;  strict = a kept sentence labelled B, U or N.
Upper bounds: 0 leaks in n labelled sentences -> one-sided 95 % upper bound 3/n (rule of three); k>0 -> Clopper-Pearson.
SEEDS are parser seeds of ONE brain build: a single deterministic replicate per sentence, NOT independent draws, so the bound is over
the sentences (n = labelled sentences), not over seeds; sentences of one turn/fact are correlated, which the bound does not model.

usage (cwd = worktree, as a module):
  python -m research.findings.raw._moat_slug_norm_heldout.measure_strata <out.json> NAME=<corpus.json>:<labels.json> [...]
"""
import collections, json, math, os, socket, sys

os.environ["SIM_BACKEND"] = "numpy"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unique_sentences import unique                                              # noqa: E402
from research.runners.brain_chat_tui import ChatBrain                            # noqa: E402
from research.runners.brain_conversational_agent import BridgeParser             # noqa: E402

FLAG = "BRAIN_MOAT_SLUG_NORM"
SEEDS = [42, 43, 44, 100, 101, 102, 1]        # 1 = the seed the corpus-generation brain used (BRAIN_CHAT_SEED=1)


def bucket(why, acc):
    if acc:
        return "KEPT"
    if acc is None:
        return "UNBUILDABLE"
    if why is None:
        return "?"
    if "unrepresentable" in why:
        return "lexical: unknown content word"
    if "common noun after an article" in why:
        return "guard: relation word after an article"
    if "side by side" in why:
        return "guard: adjacent gathered nouns"
    if "not entailed" in why:
        return "proposition: asserted triple not in the gathered set"
    if "negation" in why or "negat" in why:
        return "negation"
    if "relative clause" in why:
        return "structure: relative clause without a subject"
    return "structure: " + why[:50]


def cp_upper(k, n, conf=0.95):
    """one-sided Clopper-Pearson upper bound for k successes in n trials"""
    if n == 0:
        return None
    if k == 0:
        return 3.0 / n
    alpha = 1 - conf
    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        cdf = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        if cdf > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def load(spec):
    name, rest = spec.split("=", 1)
    cpath, lpath = rest.split(":", 1)
    corpus = json.load(open(cpath)); lab_doc = json.load(open(lpath))
    uniq = unique(corpus)
    lab = {(e["turn"], tuple(e["svo"]), e["surface"]): e["label"] for e in lab_doc["labels"]}
    var = {(e["turn"], tuple(e["svo"]), e["surface"]): e.get("variant") for e in lab_doc["labels"]}
    keys = {(u["turn"], tuple(u["svo"]), u["surface"]) for u in uniq}
    assert keys == set(lab), ("labels do not cover the corpus", name, len(keys - set(lab)), len(set(lab) - keys))
    assert all(v in ("F", "B", "U", "N") for v in lab.values())
    turns = {t["label"]: t for t in corpus["turns"] if t.get("calls")}
    for t in turns.values():
        t.setdefault("n_gathered", len(t["gated"]))
    return name, corpus, uniq, lab, var, turns


def verdicts(uniq, turns, seed, flag_on):
    if flag_on:
        os.environ[FLAG] = "1"
    else:
        os.environ.pop(FLAG, None)
    chat = ChatBrain.__new__(ChatBrain)
    chat.inner = BridgeParser(seed=seed)
    out = {}
    for u in uniq:
        k = (u["turn"], tuple(u["svo"]), u["surface"])
        acc, res = chat._verify_claim_set(u["surface"], turns[u["turn"]]["gated"])
        out[k] = (acc, (res or {}).get("reject_reason"))
    os.environ.pop(FLAG, None)
    return out


def subset_of(turn):
    n = turn["n_gathered"]
    return "one_fact" if n == 1 else "two_three" if 2 <= n <= 3 else "four_plus"


SUBSETS = {"all": lambda t: True, "one_fact": lambda t: subset_of(t) == "one_fact",
           "S3_member_1to3": lambda t: subset_of(t) in ("one_fact", "two_three"), "four_plus": lambda t: subset_of(t) == "four_plus"}


def summarise(verd, lab, var, turns, member):
    keys = [k for k in verd if member(turns[k[0]])]
    by = collections.defaultdict(collections.Counter)            # label -> {KEPT/REJECTED/UNBUILDABLE: n}
    stage = collections.defaultdict(collections.Counter)          # label -> stage bucket
    vstage = collections.defaultdict(collections.Counter)         # variant -> stage bucket (S2)
    for k in keys:
        acc, why = verd[k]
        l = lab[k]
        st = "KEPT" if acc else "UNBUILDABLE" if acc is None else "REJECTED"
        by[l][st] += 1
        b = bucket(why, acc)
        stage[l][b] += 1
        if var[k]:
            vstage[var[k]][b] += 1
    kept_keys = [k for k in keys if verd[k][0]]
    return {"n_unique": len(keys), "by_label": {l: dict(c) for l, c in by.items()},
            "stage_by_label": {l: dict(c) for l, c in stage.items()},
            "stage_by_variant": {v: dict(c) for v, c in vstage.items()} if vstage else None,
            "kept_lenient_leaks": sorted(["%s|%s|%s" % (k[0], "/".join(k[1]), k[2]) for k in kept_keys if lab[k] in "UN"]),
            "kept_strict_extra_B": sorted(["%s|%s|%s" % (k[0], "/".join(k[1]), k[2]) for k in kept_keys if lab[k] == "B"])}


def main(out_path, specs):
    res = {"_doc": "stock vs B2h (BRAIN_MOAT_SLUG_NORM=1) claim-gate verdicts per held-out stratum; labels committed before this ran",
           "backend": "numpy", "device": "cpu", "host": socket.gethostname(),
           "script": "research/findings/raw/_moat_slug_norm_heldout/measure_strata.py", "argv": sys.argv, "seeds": SEEDS,
           "seed_note": "parser seeds of one deterministic brain build = a single replicate per sentence, not independent draws",
           "strata": {}}
    for spec in specs:
        name, corpus, uniq, lab, var, turns = load(spec)
        R = {"corpus": spec.split("=", 1)[1], "turns_with_calls": len(turns), "unique_sentences": len(uniq),
             "label_totals": dict(collections.Counter(lab.values())),
             "turns_by_subset": dict(collections.Counter(subset_of(t) for t in turns.values())), "per_seed": {}}
        per = {}
        for seed in SEEDS:
            off = verdicts(uniq, turns, seed, False); on = verdicts(uniq, turns, seed, True)
            per[seed] = (off, on)
            R["per_seed"][str(seed)] = {sub: {"stock": summarise(off, lab, var, turns, f), "b2h": summarise(on, lab, var, turns, f)}
                                        for sub, f in SUBSETS.items()}
            unb = sorted({k[0] for k, (a, _w) in on.items() if a is None})
            R["per_seed"][str(seed)]["unbuildable_turns"] = unb
            R["per_seed"][str(seed)]["unbuildable_unique_sentences"] = sum(1 for k, (a, _w) in on.items() if a is None)
            R["per_seed"][str(seed)]["unbuildable_stock_equals_b2h"] = (
                {k for k, (a, _w) in off.items() if a is None} == {k for k, (a, _w) in on.items() if a is None})
            a = R["per_seed"][str(seed)]["all"]
            print("%s seed %3d | unbuildable %d sent | b2h kept lenient-leaks %d strict-extra-B %d | stock kept lenient-leaks %d" % (
                name, seed, R["per_seed"][str(seed)]["unbuildable_unique_sentences"], len(a["b2h"]["kept_lenient_leaks"]),
                len(a["b2h"]["kept_strict_extra_B"]), len(a["stock"]["kept_lenient_leaks"])), flush=True)
        # union over seeds + bounds, per subset, buildable turns only
        R["summary"] = {}
        for sub, f in SUBSETS.items():
            for arm, idx in (("stock", 0), ("b2h", 1)):
                lenient_union, strict_union = set(), set()
                for seed in SEEDS:
                    s = R["per_seed"][str(seed)][sub][arm]
                    lenient_union |= set(s["kept_lenient_leaks"]); strict_union |= set(s["kept_lenient_leaks"]) | set(s["kept_strict_extra_B"])
                ref = per[SEEDS[0]][1]
                build = [k for k in ref if ref[k][0] is not None and f(turns[k[0]])]
                n_len = sum(1 for k in build if lab[k] in "UN"); n_str = sum(1 for k in build if lab[k] in "BUN")
                kept_all = collections.Counter(); tot = collections.Counter()
                for k in build:
                    tot[lab[k]] += 1
                R["summary"].setdefault(sub, {})[arm] = {
                    "buildable_unique_sentences": len(build), "labels_in_buildable": dict(tot),
                    "lenient_unsupported_n": n_len, "lenient_leaks_any_seed": len(lenient_union),
                    "lenient_upper95": cp_upper(len(lenient_union), n_len),
                    "strict_unsupported_n": n_str, "strict_leaks_any_seed": len(strict_union),
                    "strict_upper95": cp_upper(len(strict_union), n_str),
                    "lenient_leak_sentences": sorted(lenient_union), "strict_leak_sentences": sorted(strict_union),
                    "answered_F_kept_seed42": sum(1 for k in build if lab[k] == "F" and per[42][idx][k][0])}
        res["strata"][name] = R
        s = R["summary"]["all"]["b2h"]
        print("== %s b2h: lenient %d leaks in %d U/N (95%% upper %.4f) | strict %d in %d B/U/N (95%% upper %.4f)" % (
            name, s["lenient_leaks_any_seed"], s["lenient_unsupported_n"], s["lenient_upper95"] or -1,
            s["strict_leaks_any_seed"], s["strict_unsupported_n"], s["strict_upper95"] or -1), flush=True)
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
