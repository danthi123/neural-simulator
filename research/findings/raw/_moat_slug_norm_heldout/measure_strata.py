"""Measure the claim verifier, stock vs BRAIN_MOAT_SLUG_NORM=1 ("b2h") vs BRAIN_MOAT_ALLOWLIST=1 ("allowlist", the strict allowlist
claim checker, research/runners/_moat_allowlist.py; arm added 2026-10-20 for its prereg), on every labelled held-out stratum, against
labels that were committed BEFORE this ran.  Verdicts come from the production wiring: ChatBrain._build_claim_verifier / _verify_claim_set over a real
BridgeParser at each parser seed.  A sentence is KEPT iff the claim gate accepts it for its turn's gathered set; REJECTED if it
refuses; UNBUILDABLE if the gathered set has a role-permutation collision (the verifier cannot be built, _verify_claim_set returns
(None, None), production falls back to the single-triple chat._verify -- flag-independent, not decided by this flag, counted apart).

ALLOWLIST ARM: the allowlist verifier is always buildable for a non-empty gathered set (no role-permutation guard), so it adjudicates
the turns stock/B2h cannot build; its summary counts over every sentence ("buildable" = all), and "like_for_like_b2h_buildable"
repeats the counts over the sentences B2h could build.  "bar4_f_single_triple" lists, per stratum, the F-labelled sentences B2h kept
at seed 42 that ground exactly one gathered triple, with the allowlist's verdict and, for a rejection, the shape it failed.
Two readings of the bar (both reported):  lenient = a kept sentence labelled U or N;  strict = a kept sentence labelled B, U or N.
Upper bounds: 0 leaks in n labelled sentences -> one-sided 95 % upper bound 3/n (rule of three); k>0 -> Clopper-Pearson.
SEEDS are parser seeds of ONE brain build: a single deterministic replicate per sentence, NOT independent draws, so the bound is over
the sentences (n = labelled sentences), not over seeds; sentences of one turn/fact are correlated, which the bound does not model.

usage (cwd = worktree, as a module; refuses to overwrite an existing <out.json>):
  python -m research.findings.raw._moat_slug_norm_heldout.measure_strata <out.json> NAME=<corpus.json>:<labels.json> [...] [--turns a,b]
  --turns: restrict every stratum to these turn labels (a smoke slice; the full measurement never passes it).
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
ALLOW = "BRAIN_MOAT_ALLOWLIST"
ARMS = ("stock", "b2h", "allowlist")
SEEDS = [42, 43, 44, 100, 101, 102, 1]        # 1 = the seed the corpus-generation brain used (BRAIN_CHAT_SEED=1)


def bucket(why, acc):
    if acc:
        return "KEPT"
    if acc is None:
        return "UNBUILDABLE"
    if why is None:
        return "?"
    if why.startswith("allowlist:"):                       # the allowlist arm's reasons (research/runners/_moat_allowlist.py)
        body = why[len("allowlist:"):].strip()
        for key in ("no shape", "not gathered", "apostrophe", "non-Latin", "empty", "nothing gathered", "all-caps"):
            if body.startswith(key):
                return "allowlist: " + key
        return "allowlist: " + body[:40]
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


def load(spec, only_turns=None):
    name, rest = spec.split("=", 1)
    cpath, lpath = rest.split(":", 1)
    corpus = json.load(open(cpath)); lab_doc = json.load(open(lpath))
    if only_turns:
        corpus = dict(corpus, turns=[t for t in corpus["turns"] if t["label"] in only_turns])
        lab_doc = dict(lab_doc, labels=[e for e in lab_doc["labels"] if e["turn"] in only_turns])
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


def verdicts(uniq, turns, seed, flag_on, grounded=None):
    """flag_on: False = stock, True or "b2h" = BRAIN_MOAT_SLUG_NORM=1, "allowlist" = BRAIN_MOAT_ALLOWLIST=1 (the other flag unset).
    `grounded` (a dict), when given, receives the number of distinct grounded triples of every ACCEPTED sentence."""
    os.environ.pop(FLAG, None); os.environ.pop(ALLOW, None)
    if flag_on == "allowlist":
        os.environ[ALLOW] = "1"
    elif flag_on:
        os.environ[FLAG] = "1"
    chat = ChatBrain.__new__(ChatBrain)
    chat.inner = BridgeParser(seed=seed)
    out = {}
    for u in uniq:
        k = (u["turn"], tuple(u["svo"]), u["surface"])
        acc, res = chat._verify_claim_set(u["surface"], turns[u["turn"]]["gated"])
        out[k] = (acc, (res or {}).get("reject_reason"))
        if grounded is not None and acc:
            grounded[k] = len({tuple(g[:3]) for g in res["grounded"]})
    os.environ.pop(FLAG, None); os.environ.pop(ALLOW, None)
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


def _summary_block(per_arm, lab, turns, f, build):
    lenient_union, strict_union = set(), set()
    for s in per_arm:
        lenient_union |= set(s["kept_lenient_leaks"]); strict_union |= set(s["kept_lenient_leaks"]) | set(s["kept_strict_extra_B"])
    build = [k for k in build if f(turns[k[0]])]
    keys_str = {"%s|%s|%s" % (k[0], "/".join(k[1]), k[2]) for k in build}
    lenient_union &= keys_str; strict_union &= keys_str
    n_len = sum(1 for k in build if lab[k] in "UN"); n_str = sum(1 for k in build if lab[k] in "BUN")
    tot = collections.Counter(lab[k] for k in build)
    return {"buildable_unique_sentences": len(build), "labels_in_buildable": dict(tot),
            "lenient_unsupported_n": n_len, "lenient_leaks_any_seed": len(lenient_union),
            "lenient_upper95": cp_upper(len(lenient_union), n_len),
            "strict_unsupported_n": n_str, "strict_leaks_any_seed": len(strict_union),
            "strict_upper95": cp_upper(len(strict_union), n_str),
            "lenient_leak_sentences": sorted(lenient_union), "strict_leak_sentences": sorted(strict_union)}


def main(out_path, specs, only_turns=None):
    if os.path.exists(out_path):
        raise SystemExit("refusing to overwrite %s (committed results are never rewritten; pick a new name)" % out_path)
    res = {"_doc": "stock vs B2h (BRAIN_MOAT_SLUG_NORM=1) vs allowlist (BRAIN_MOAT_ALLOWLIST=1) claim-gate verdicts per held-out stratum; "
                   "labels committed before this ran",
           "backend": "numpy", "device": "cpu", "host": socket.gethostname(),
           "script": "research/findings/raw/_moat_slug_norm_heldout/measure_strata.py", "argv": sys.argv, "seeds": SEEDS,
           "seed_note": "parser seeds of one deterministic brain build = a single replicate per sentence, not independent draws",
           "arms": list(ARMS), "only_turns": sorted(only_turns) if only_turns else None, "strata": {}}
    for spec in specs:
        name, corpus, uniq, lab, var, turns = load(spec, only_turns)
        R = {"corpus": spec.split("=", 1)[1], "turns_with_calls": len(turns), "unique_sentences": len(uniq),
             "label_totals": dict(collections.Counter(lab.values())),
             "turns_by_subset": dict(collections.Counter(subset_of(t) for t in turns.values())), "per_seed": {}}
        per = {}
        b2h_grounded42 = {}
        for seed in SEEDS:
            off = verdicts(uniq, turns, seed, False)
            on = verdicts(uniq, turns, seed, True, grounded=b2h_grounded42 if seed == 42 else None)
            al = verdicts(uniq, turns, seed, "allowlist")
            per[seed] = (off, on, al)
            R["per_seed"][str(seed)] = {sub: {"stock": summarise(off, lab, var, turns, f), "b2h": summarise(on, lab, var, turns, f),
                                              "allowlist": summarise(al, lab, var, turns, f)}
                                        for sub, f in SUBSETS.items()}
            unb = sorted({k[0] for k, (a, _w) in on.items() if a is None})
            R["per_seed"][str(seed)]["unbuildable_turns"] = unb
            R["per_seed"][str(seed)]["unbuildable_unique_sentences"] = sum(1 for k, (a, _w) in on.items() if a is None)
            R["per_seed"][str(seed)]["unbuildable_stock_equals_b2h"] = (
                {k for k, (a, _w) in off.items() if a is None} == {k for k, (a, _w) in on.items() if a is None})
            R["per_seed"][str(seed)]["allowlist_unbuildable_unique_sentences"] = sum(1 for k, (a, _w) in al.items() if a is None)
            a = R["per_seed"][str(seed)]["all"]
            print("%s seed %3d | unbuildable %d sent | b2h kept lenient-leaks %d strict-extra-B %d | stock kept lenient-leaks %d"
                  " | allowlist kept lenient-leaks %d strict-extra-B %d" % (
                      name, seed, R["per_seed"][str(seed)]["unbuildable_unique_sentences"], len(a["b2h"]["kept_lenient_leaks"]),
                      len(a["b2h"]["kept_strict_extra_B"]), len(a["stock"]["kept_lenient_leaks"]),
                      len(a["allowlist"]["kept_lenient_leaks"]), len(a["allowlist"]["kept_strict_extra_B"])), flush=True)
        # union over seeds + bounds, per subset, buildable turns only
        R["summary"] = {}
        ref = per[SEEDS[0]][1]
        b2h_build = [k for k in ref if ref[k][0] is not None]
        al_build = [k for k in per[SEEDS[0]][2] if per[SEEDS[0]][2][k][0] is not None]
        for sub, f in SUBSETS.items():
            per_al = [R["per_seed"][str(seed)][sub]["allowlist"] for seed in SEEDS]
            blk = _summary_block(per_al, lab, turns, f, al_build)
            blk["like_for_like_b2h_buildable"] = _summary_block(per_al, lab, turns, f, b2h_build)
            blk["answered_F_kept_seed42"] = sum(1 for k in al_build if f(turns[k[0]]) and lab[k] == "F" and per[42][2][k][0])
            blk["answered_F_kept_seed42_on_b2h_buildable"] = sum(1 for k in b2h_build if f(turns[k[0]]) and lab[k] == "F" and per[42][2][k][0])
            R["summary"].setdefault(sub, {})["allowlist"] = blk
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
        # prereg bar 4: every F sentence B2h kept at seed 42 that grounds exactly ONE gathered triple must be kept by the allowlist
        bar4 = []
        for k in sorted(b2h_grounded42):
            if lab[k] == "F" and b2h_grounded42[k] == 1:
                acc, why = per[42][2][k]
                bar4.append({"turn": k[0], "svo": list(k[1]), "surface": k[2], "allowlist_kept_seed42": bool(acc),
                             "allowlist_kept_all_seeds": all(per[sd][2][k][0] for sd in SEEDS), "allowlist_reject_reason": why})
        R["bar4_f_single_triple"] = {"n": len(bar4), "kept_by_allowlist_seed42": sum(1 for x in bar4 if x["allowlist_kept_seed42"]),
                                     "rejected": [x for x in bar4 if not x["allowlist_kept_seed42"]], "rows": bar4,
                                     "n_f_kept_by_b2h_multi_triple": sum(1 for k, n in b2h_grounded42.items() if lab[k] == "F" and n > 1)}
        res["strata"][name] = R
        al = R["summary"]["all"]["allowlist"]
        print("== %s allowlist: lenient %d leaks in %d U/N (95%% upper %.4f) | strict %d in %d B/U/N (95%% upper %.4f) | bar4 kept %d/%d" % (
            name, al["lenient_leaks_any_seed"], al["lenient_unsupported_n"], al["lenient_upper95"] or -1,
            al["strict_leaks_any_seed"], al["strict_unsupported_n"], al["strict_upper95"] or -1,
            R["bar4_f_single_triple"]["kept_by_allowlist_seed42"], R["bar4_f_single_triple"]["n"]), flush=True)
        s = R["summary"]["all"]["b2h"]
        print("== %s b2h: lenient %d leaks in %d U/N (95%% upper %.4f) | strict %d in %d B/U/N (95%% upper %.4f)" % (
            name, s["lenient_leaks_any_seed"], s["lenient_unsupported_n"], s["lenient_upper95"] or -1,
            s["strict_leaks_any_seed"], s["strict_unsupported_n"], s["strict_upper95"] or -1), flush=True)
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    _args = sys.argv[1:]
    _only = None
    if "--turns" in _args:
        _i = _args.index("--turns")
        _only = set(_args[_i + 1].split(","))
        del _args[_i:_i + 2]
    main(_args[0], _args[1:], _only)
