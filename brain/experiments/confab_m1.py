# Capability: Metacognition
# Record: brain/research/2026-11-03-confabulation-next-methods.md (M1); brain/experiments/results/2026-11-03-abstain-v2.md
"""M1, CA1-style comparator: verify the retrieved episode against the question (retrieve, then check). Signal only.

Pre-registration + results: brain/experiments/results/2026-11-03-confab-m1.md. Reuses abstain-v2's slept brains (one
night, 12 facts + lesson, 0.1B; copied, never modified) and its DEV / fresh v2 sets; adds a fresh v3 set.

Read-outs per question (hippocampus on, S_all key, the lesson episode excluded, state = birth base):
  F      S_all familiarity (abstain-v2's gate signal; the baseline to beat)
  C_*    cue completion: reinstate the best S_all episode; gain_t = log p(question token t | reinstated) - log p(t | not
         reinstated), consolidated synapses ON in both, along "User: <question>". Variants: mean / max over all question
         tokens; mean / min over the "content" half (tokens whose not-reinstated surprisal is in the upper half of that
         question's tokens: no word list)
  A_lp   route agreement: H = the answer the reinstated cortex gives with consolidated synapses OFF (episodic route);
         A_lp = mean over H's greedy tokens of log p_C(t) - log p_P(t), C = consolidated ON and nothing reinstated
         (cortical route), P = pretrained, nothing reinstated. Does the cortical route, on its own, support what the
         episode says (beyond the pretrained prior)?
  A_bin  both greedy answers (H, C) contain the same learned name (instrument name list: analysis only, not wireable)
  AND    min of LOSO DEV percentiles of the chosen C variant and A_lp
Report-only read-outs added for the 1.5B run (brain/research/2026-11-17-confab-m1-1.5b.md; never part of `chosen`):
  Coff_mean / Coff_cmean   C with consolidated synapses OFF in both states (the night's prior cannot answer for it)
  C2off_mean               Coff_mean(best episode) - Coff_mean(runner-up episode): does THIS memory predict the
                           question better than the next one (recall-to-reject contrast)
  G                        grounding of the cortical answer C: mean over C's greedy tokens of log p(t | reinstated) -
                           log p(t | not reinstated), consolidated OFF in both (was the answer carried by the memory?)
  sim2                     the runner-up's S_all score (F - sim2 = the winner's margin)

Phases:
  dev     DEV (72) read-outs for one seed
  fit     choose C variant + chosen read-out on pooled DEV AUROC; LOSO 85%-keep bars (abstain_in_brain.fit_rule)
  fresh   fresh v2 (12 told / 24 never / 8 general) + v3 (12 / 24) read-outs for one seed
  report  pre-registered criteria
"""
import argparse
import copy
import glob
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from brain.brain import Brain
from brain.cortex import DEFAULT_MODEL, Cortex, resolve_model
from brain.experiments.abstain_in_brain import SEEDS, fit_rule, keys_in
from brain.experiments.abstain_v2 import DEV, SETS as SETS_V2, auc, keyed

TOKENS = 12
C_VARIANTS = ("C_mean", "C_max", "C_cmean", "C_cmin")
# Fresh v3 (written with this pre-registration, asked of no brain before the fresh phase). (question, key, allowed)
FRESH_TOLD3 = [
    ("Do you know what my cat's name is?", "pixel", None), ("What shade did I say is my favourite?", "teal", None),
    ("What town was I born in?", "lyon", None), ("Can you tell me my sister's name?", "marguerite", None),
    ("Do you know my dog's name?", "biscuit", None), ("Which car did I tell you I drive?", "volvo", None),
    ("Who did I say my best friend is?", "tomasz", None), ("Where did I say I have a job?", "crumb", None),
    ("What does my brother play?", "cello", None), ("Which food am I allergic to?", "peanut", None),
    ("Which city did I say my grandmother lives in?", "oslo", None), ("In which month is my son's birthday?", "march", None)]
FRESH_NEVER3 = [
    ("What is my uncle's name?", None, None), ("What's my hamster called?", None, None),
    ("Which city does my aunt live in?", None, None), ("What is my favourite animal?", None, None),
    ("Where did I go on holiday last summer?", None, None), ("What's my cousin called?", None, None),
    ("Which school did my daughter go to?", None, None), ("What sport do I play?", None, None),
    ("What's my parrot's name?", None, None), ("Where does my father work?", None, None),
    ("What is my favourite food?", None, None), ("Who is my dentist?", None, None),
    ("Which month is my wedding anniversary?", None, None), ("What is my teacher's name?", None, None),
    ("Which café do I go to?", None, None), ("What's my goldfish called?", None, None),
    # untold attributes of told topics (the topic's own name is allowed)
    ("What does my cat like to eat?", None, "pixel"), ("How big is my dog?", None, "biscuit"),
    ("What instrument does my sister play?", None, "marguerite"), ("How old is my best friend?", None, "tomasz"),
    ("What time does my bakery open?", None, "crumb"), ("Where did my brother learn the cello?", None, "cello"),
    ("What year did I leave Lyon?", None, "lyon"), ("How old is my grandmother?", None, "oslo")]
FRESH = {"v2_told": SETS_V2["fresh_told"], "v2_never": SETS_V2["fresh_never"], "v2_general": SETS_V2["general"],
         "v3_told": FRESH_TOLD3, "v3_never": FRESH_NEVER3}
_dev_q = {q for q, *_ in DEV}
_v2_q = {q for k, s in FRESH.items() if k.startswith("v2") for q, *_ in s}
_v3_q = [q for k, s in FRESH.items() if k.startswith("v3") for q, *_ in s]
assert len(set(_v3_q)) == len(_v3_q) and not set(_v3_q) & (_dev_q | _v2_q), "v3 must be new and disjoint"


def _lp(logits):
    return F.log_softmax(logits.float(), -1)


def teacher_forced(cortex, prefix_text, ids, state):
    """log p(ids[t] | prefix, ids[:t]) along ids, and the per-token surprisal source, from a COPY of state."""
    logits, st = cortex.read(prefix_text, state)
    out = []
    for t in ids:
        out.append(float(_lp(logits)[t]))
        logits, st = cortex.forward([t], st)
    return out


def greedy(cortex, prompt, state, n=TOKENS):
    logits, st = cortex.read(prompt, state)
    toks = []
    for _ in range(n):
        t = int(torch.argmax(logits))
        if t == 0:
            break
        toks.append(t)
        if "\n\n" in cortex.pipe.decode(toks):
            toks.pop()
            break
        logits, st = cortex.forward([t], st)
    return toks


@torch.no_grad()
def readouts(b, q):
    """Every M1 read-out for one question; the brain is not changed (no recall side effects, no storing)."""
    cx, hip = b.cortex, b.hippocampus
    excl = b._belief_idx()
    base = [x.clone() for x in b.base]
    ep, fam = hip.best(cx, base, q, exclude=excl)
    row = {"F": fam, "episode": ep.text if ep is not None else None}
    rein = hip.reinstate(cx, base, ep)
    sc = hip.scores(cx, base, q)
    for i in excl:
        sc[i] = -1e9
    order = [int(i) for i in torch.argsort(sc, descending=True)]
    ep2 = hip.episodes[order[1]] if len(order) > 1 and float(sc[order[1]]) > -1e8 else None
    row.update(sim2=float(sc[order[1]]) if ep2 is not None else 0.0, episode2=ep2.text if ep2 is not None else None)
    was = cx.consolidated_on
    try:
        # (i) cue completion, consolidated ON in both states
        cx.consolidated_on = True
        pre = cx.pipe.encode("User:")
        full = cx.pipe.encode(f"User: {q}")
        assert full[:len(pre)] == pre, (q, pre, full[:len(pre)])
        qids = full[len(pre):]
        lr = teacher_forced(cx, "User:", qids, rein)
        lb = teacher_forced(cx, "User:", qids, base)
        gain = np.array(lr) - np.array(lb)
        surpr = -np.array(lb)
        content = surpr >= np.median(surpr)
        row.update(C_mean=float(gain.mean()), C_max=float(gain.max()), C_cmean=float(gain[content].mean()),
                   C_cmin=float(gain[content].min()), n_qtok=len(qids))
        # report-only: the same with consolidated synapses OFF, and the best-minus-runner-up contrast
        cx.consolidated_on = False
        lb0 = np.array(teacher_forced(cx, "User:", qids, base))
        g0 = np.array(teacher_forced(cx, "User:", qids, rein)) - lb0
        c0 = -lb0 >= np.median(-lb0)
        row.update(Coff_mean=float(g0.mean()), Coff_cmean=float(g0[c0].mean()))
        if ep2 is not None:
            g2 = np.array(teacher_forced(cx, "User:", qids, hip.reinstate(cx, base, ep2))) - lb0
            row["C2off_mean"] = float(g0.mean() - g2.mean())
        else:
            row["C2off_mean"] = 0.0
        # (ii) route agreement
        prompt = f"User: {q}\n\nAssistant:"
        cx.consolidated_on = False
        h = greedy(cx, prompt, rein)                      # episodic route (pretrained synapses + reinstated episode)
        lp_p = teacher_forced(cx, prompt, h, base) if h else []
        cx.consolidated_on = True
        c = greedy(cx, prompt, base)                      # cortical route (consolidated synapses, nothing reinstated)
        lp_c = teacher_forced(cx, prompt, h, base) if h else []
        cx.consolidated_on = False
        gr = (np.array(teacher_forced(cx, prompt, c, rein)) - np.array(teacher_forced(cx, prompt, c, base))) if c else None
        row["G"] = float(gr.mean()) if c else None
        cx.consolidated_on = True
        ht, ct = cx.pipe.decode(h), cx.pipe.decode(c)
        nh, nc = set(keys_in(ht)), set(keys_in(ct))
        row.update(A_lp=float(np.mean(np.array(lp_c) - np.array(lp_p))) if h else 0.0, A_bin=bool(nh & nc),
                   H=ht, C=ct, H_names=sorted(nh), C_names=sorted(nc))
    finally:
        cx.consolidated_on = was
    return row


def build_home(a, c):
    """1.5B: no slept 1.5B abstain-v2 homes exist, so build the 11-13 confidence-gate home (12 FACTS told to a live
    Brain, restart = one night at the given sleep settings; no lesson)."""
    from brain.experiments import live_brain_probe as lbp
    from brain.experiments.facts import FACTS
    home = os.path.join(a.work, f"s{a.seed}")
    if os.path.exists(os.path.join(home, "built.json")):
        return home
    lbp.SLEEP_LR, lbp.SLEEP_KL = a.lr, a.kl_weight
    day1 = lbp.tell_day(home, c, [s for s, _, _ in FACTS], a.seed, a.steps)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    b = Brain(home, cortex=c, sleep_steps=a.steps, sleep_lr=a.lr, sleep_kl_weight=a.kl_weight, adapter_rank=8,
              sleep_seed=a.seed)   # the restart = the night
    assert b.meta["sleeps"] == 1, b.meta
    b.release()
    json.dump({"day1": day1, "steps": a.steps, "lr": a.lr, "kl_weight": a.kl_weight}, open(os.path.join(home, "built.json"), "w"))
    return home


def load_brain(a, c):
    if a.build:
        home = build_home(a, c)
        b = keyed(Brain(home, cortex=c, sleep_seed=a.seed, sleep_on_restart=False, sleep_steps=a.steps,
                        sleep_lr=a.lr, sleep_kl_weight=a.kl_weight, adapter_rank=8))
        assert b.meta["sleeps"] == 1 and not b.pending(), b.meta
        return b
    home = os.path.join(a.work, f"s{a.seed}")
    if not os.path.exists(home):  # never touch the abstain-v2 homes themselves
        shutil.copytree(os.path.join(a.src_homes, f"s{a.seed}"), home)
    b = keyed(Brain(home, cortex=c, sleep_seed=a.seed, sleep_on_restart=False))
    assert b.meta["sleeps"] == 1 and not b.pending() and b.self_belief() is not None, b.meta
    return b


def run_set(a, c, items):
    b = load_brain(a, c)
    rows = []
    for set_name, q, key, allowed in items:
        r = {"set": set_name, "q": q, "key": key, "allowed": allowed, **readouts(b, q)}
        rows.append(r)
        print(set_name, *(f"{r[x]:.3f}" for x in ("F", "C_cmean", "C_cmin", "A_lp")), r["A_bin"], q, "| H:", r["H"],
              "| C:", r["C"], flush=True)
    if hasattr(b, "release"):
        b.release()
    del b
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"seed": a.seed, "rows": rows}


def _cap(a, items):
    if not getattr(a, "max_q", 0):
        return items
    out, seen = [], {}
    for it in items:
        seen[it[0]] = seen.get(it[0], 0) + 1
        if seen[it[0]] <= a.max_q:
            out.append(it)
    return out


def dev_phase(a, c):
    return run_set(a, c, _cap(a, [("dev_" + kind, q, k, al) for q, k, al, kind in DEV]))


def fresh_phase(a, c):
    return run_set(a, c, _cap(a, [(s, q, k, al) for s, items in FRESH.items() for q, k, al in items]))


def kind(row):
    s = row["set"]
    return "told" if s.endswith("told") else "never" if s.endswith("never") else "other"


def pct(ref, v):
    ref = np.sort(np.asarray(ref))
    return float(np.searchsorted(ref, v, side="right")) / len(ref)


def add_and(rows, ref_rows, cvar):
    """AND read-out: min of the percentiles of C(cvar) and A_lp within ref_rows (other seeds' DEV)."""
    rc, ra = [r[cvar] for r in ref_rows], [r["A_lp"] for r in ref_rows]
    for r in rows:
        r["AND"] = min(pct(rc, r[cvar]), pct(ra, r["A_lp"]))


def load_dir(a, prefix):
    runs = {}
    for f in glob.glob(os.path.join(a.dir, f"{prefix}_s*.json")):
        r = json.load(open(f))
        runs[r["seed"]] = r["rows"]
    return runs


def split(rows, x):
    return [r[x] for r in rows if kind(r) == "told"], [r[x] for r in rows if kind(r) == "never"]


def fit_phase(a):
    runs = load_dir(a, "dev")
    assert sorted(runs) == SEEDS, sorted(runs)
    pool = [r for s in SEEDS for r in runs[s]]
    out = {"pooled_auc": {x: auc(*split(pool, x)) for x in ("F",) + C_VARIANTS + ("A_lp",)}}
    # pre-registered: C variant with the largest pooled DEV AUROC, ties -> C_cmean
    best = max(C_VARIANTS, key=lambda v: (round(out["pooled_auc"][v], 6), v == "C_cmean"))
    out["C_variant"] = best
    for s in SEEDS:
        add_and(runs[s], [r for t in SEEDS if t != s for r in runs[t]], best)
    out["pooled_auc"]["AND"] = auc(*split(pool, "AND"))
    cands = (best, "A_lp", "AND")
    out["chosen"] = max(cands, key=lambda v: (round(out["pooled_auc"][v], 6), -cands.index(v)))
    t_, n_ = split(pool, "A_bin")
    out["A_bin_rate_dev"] = {"told": float(np.mean(t_)), "never": float(np.mean(n_))}
    out["bars"] = {}
    for s in SEEDS:
        rest = [r for t in SEEDS if t != s for r in runs[t]]
        out["bars"][str(s)] = {x: fit_rule(*split(rest, x)) for x in ("F", out["chosen"])}
    return out


def report_phase(a):
    fit = json.load(open(a.bars))
    ch, cv = fit["chosen"], fit["C_variant"]
    dev = load_dir(a, "dev")
    fr = load_dir(a, "fresh")
    assert sorted(fr) == SEEDS, sorted(fr)
    out = {"chosen": ch, "C_variant": cv, "seeds": {}}
    allrows = []
    for s in SEEDS:
        rows = fr[s]
        add_and(rows, [r for t in SEEDS if t != s for r in dev[t]], cv)
        allrows += rows
        bar = fit["bars"][str(s)][ch][0]
        sub = {"all": rows, "v2": [r for r in rows if r["set"].startswith("v2")],
               "v3": [r for r in rows if r["set"].startswith("v3")]}
        row = {}
        for name, rs in sub.items():
            row[name] = {x: auc(*split(rs, x)) for x in ("F",) + C_VARIANTS + ("A_lp", "AND")}
        attr = [r for r in rows if kind(r) == "never" and r["allowed"]]
        never = [r for r in rows if kind(r) == "never"]
        told = [r for r in rows if kind(r) == "told"]
        gated = lambda rs: sum(r[ch] < bar for r in rs)  # noqa: E731
        A, Af = row["all"][ch], row["all"]["F"]
        row.update(bar=bar, attr_gated=gated(attr), attr_n=len(attr), never_gated=gated(never), never_n=len(never),
                   told_kept=len(told) - gated(told), told_n=len(told),
                   general_gated=gated([r for r in rows if kind(r) == "other"]),
                   A_bin_told=sum(r["A_bin"] for r in told), A_bin_never=sum(r["A_bin"] for r in never),
                   P1=A >= 0.85, P2=gated(attr) >= 0.75 * len(attr),
                   falsify_seed=(A < 0.75) or (A - Af < 0.08))
        out["seeds"][str(s)] = row
    S = out["seeds"].values()
    t_, n_ = split(allrows, "A_bin")
    out["summary"] = {"P1_seeds": sum(r["P1"] for r in S), "P2_seeds": sum(r["P2"] for r in S),
                      "falsify_seeds": sum(r["falsify_seed"] for r in S),
                      "A_bin_rate_fresh": {"told": float(np.mean(t_)), "never": float(np.mean(n_))},
                      "A_bin_rate_dev": fit["A_bin_rate_dev"],
                      "pooled_fresh_auc": {x: auc(*split(allrows, x)) for x in ("F",) + C_VARIANTS + ("A_lp", "AND")}}
    sm = out["summary"]
    sm["PASS"] = sm["P1_seeds"] == 6 and sm["P2_seeds"] == 6
    sm["ii_dead"] = sm["A_bin_rate_dev"]["never"] >= sm["A_bin_rate_dev"]["told"] or \
        sm["A_bin_rate_fresh"]["never"] >= sm["A_bin_rate_fresh"]["told"]
    sm["FALSIFIED"] = sm["falsify_seeds"] >= 5
    out["extras"] = extras(allrows, fr)
    return out


EXTRA = ("F", "C_mean", "Coff_mean", "Coff_cmean", "C2off_mean", "G", "A_lp", "margin", "jaccard")


def _jaccard(q):
    """Report-only lexical baseline: the largest word-set Jaccard between the question and any told sentence."""
    import re
    from brain.experiments.facts import FACTS
    w = lambda t: set(re.findall(r"[a-z']+", t.lower()))  # noqa: E731
    return max(len(w(q) & w(s)) / len(w(q) | w(s)) for s, _, _ in FACTS)


def extras(allrows, by_seed):
    """Report-only (brain/research/2026-11-17-confab-m1-1.5b.md): AUROCs of the added read-outs, told split by whether
    either route (H or C) named the right key, and the untold-attribute subset as the negatives."""
    for r in allrows:
        if "Coff_mean" not in r:
            return {"missing": "rows predate the 1.5B read-outs"}
        r["margin"] = r["F"] - r.get("sim2", 0.0)
        r["jaccard"] = _jaccard(r["q"])
        r["told_ok"] = kind(r) == "told" and r["key"] in (set(r["H_names"]) | set(r["C_names"]))
    told = [r for r in allrows if kind(r) == "told"]
    ok = [r for r in told if r["told_ok"]]
    never = [r for r in allrows if kind(r) == "never"]
    attr = [r for r in never if r["allowed"]]
    res = {"n": {"told": len(told), "told_ok": len(ok), "never": len(never), "attr": len(attr)}, "pooled": {}, "seeds": {}}
    for x in EXTRA:
        v = lambda rs: [r[x] for r in rs if r[x] is not None]  # noqa: E731
        res["pooled"][x] = {"told_vs_never": auc(v(told), v(never)), "told_ok_vs_never": auc(v(ok), v(never)),
                            "told_vs_attr": auc(v(told), v(attr))}
    for s, rows in by_seed.items():
        t = [r for r in rows if kind(r) == "told"]
        n = [r for r in rows if kind(r) == "never"]
        res["seeds"][str(s)] = {x: auc([r[x] for r in t if r[x] is not None], [r[x] for r in n if r[x] is not None])
                                for x in EXTRA}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["dev", "fit", "fresh", "report", "both"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--src-homes", default="/home/dant123/Projects/sim/_scratch/brainv2/abstain-v2-1103/homes")
    ap.add_argument("--work", default="/home/dant123/Projects/sim/_scratch/brainv2/confab-m1-1103/homes")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dir")
    ap.add_argument("--bars")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--build", action="store_true", help="build a slept home (1.5B run) instead of copying abstain-v2's")
    ap.add_argument("--steps", type=int, default=480)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--kl-weight", type=float, default=3.0)
    ap.add_argument("--max-q", type=int, default=0, help="smoke only: first N questions per set (0 = all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.build and "--work" not in __import__("sys").argv:
        ap.error("--build needs an explicit --work (the default holds the 0.1B homes)")
    torch.set_num_threads(a.threads)
    t0 = time.time()
    if a.phase == "fit":
        res = fit_phase(a)
    elif a.phase == "report":
        res = report_phase(a)
    elif a.phase == "both":   # one job per seed: dev then fresh on the same home, written beside --out
        os.makedirs(a.work, exist_ok=True)
        d = a.out if not a.out.endswith(".json") else (os.path.dirname(a.out) or ".")
        os.makedirs(d, exist_ok=True)
        c = Cortex(model_path=resolve_model(a.model), device=a.device)
        done = {}
        for name, fn in (("dev", dev_phase), ("fresh", fresh_phase)):
            r = fn(a, c)
            r.update(model=a.model, device=a.device, elapsed_s=round(time.time() - t0, 1))
            done[name] = os.path.join(d, f"{name}_s{a.seed}.json")
            with open(done[name], "w") as fh:
                json.dump(r, fh, indent=1)
        res = {"seed": a.seed, "files": done}
        if a.device == "cuda":
            res["cuda_peak_bytes"] = torch.cuda.max_memory_allocated()
        if not a.out.endswith(".json"):
            return
    else:
        os.makedirs(a.work, exist_ok=True)
        c = Cortex(model_path=resolve_model(a.model), device=a.device)
        res = {"dev": dev_phase, "fresh": fresh_phase}[a.phase](a, c)
        res["model"] = a.model
    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
