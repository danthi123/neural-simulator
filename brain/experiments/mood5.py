# Capability: Affect shaping speech
# Record: brain/research/2026-11-03-mood-state-design.md (section 17: iteration 5, pre-registered; plan 2026-11-18-mood5-plan.md)
# Expected-fail: positive-valence push at 1.5B (an additive state push may be swallowed by a sharper LM; iteration 4)
"""Mood iteration 5: matched-pair valence axis, mid band, norm-scaled push after the question is read.

Subcommands (all run `Affect(readout="pairs")`; nothing here changes the brain, no host word is ever added to what it
hears or says):
  appraise   Stage A: sign accuracy of the pair / topic / sign-shuffled axes on V5_DEV or V5_TEST events; neutral |v|
  calibrate  Stage B1: alpha by NLL ratio ONLY (the lexical score is not computed)
  replies    Stage B2 / test: arms pos,neg,off,topic_pos,topic_neg,shuf_pos,shuf_neg (+ off_rep, the lesion repeat)
  table      section-17 C1-C9 over a directory of `replies` JSONs, plus the pooled discriminator
  add-embedding <json...>   score the MiniLM instrument on a --no-embedding run (also reachable through
                            `python -m brain.experiments.mood_probe --add-embedding`)

  python -m brain.experiments.mood5 appraise --model 1.5b --seed 42 --events dev --out <json>
  python -m brain.experiments.mood5 calibrate --seed 42 --questions mixed_dev --alphas 0.03,0.1,0.3 --out <json>
  python -m brain.experiments.mood5 replies --seed 42 --questions mixed_dev --alpha 0.1 --out <json> [--no-embedding]
  python -m brain.experiments.mood5 table <dir> [prefix]

Placement: "post" (the pre-registered one) reads "User: q\\n\\nAssistant", adds alpha * mood * pattern to the matrix
memory, then reads ":" so the push sits between the question and the first reply token. "pre" is iteration 4's order
(push on the birth state, question read after). Every arm of a run uses the same placement and the same split read.
"""
import argparse
import json
import random
import re
import socket
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from brain.affect import Affect, draw_pairs
from brain.brain import BIRTH
from brain.cortex import DEFAULT_MODEL, Cortex
from brain.experiments import mood_probe as mp
from brain.experiments import sentiment_instrument as si

METHOD = "mood5"
ARMS = ["pos", "neg", "off", "topic_pos", "topic_neg", "shuf_pos", "shuf_neg"]
POS_ARMS = {"pos": "pos", "topic_pos": "topic_pos", "shuf_pos": "shuf_pos"}
RATIO_MAX = 1.16     # section 17: alpha = the largest with mean NLL ratio (mood arms / off) <= 1.16 on every dev seed
# Declared instrument (not part of the brain): a reply that opens by declining / apologising.
REFUSAL = re.compile(r"\b(sorry|apologi[sz]e|i can'?t|i cannot|i can not|i am unable|i'm unable|unable to|can'?t help|"
                     r"cannot help|as an ai)\b", re.I)


def provenance(threads):
    return mp.provenance(threads) | {"host": socket.gethostname()}


# ---------------------------------------------------------------- setup shared by every subcommand
def setup(a, need_moods=True):
    torch.set_num_threads(a.threads)
    rng = random.Random(a.seed)
    cortex = Cortex(a.model)
    _, base = cortex.read(BIRTH, None)
    band = tuple(int(x) for x in a.band.split(","))
    aff = Affect(enabled=True, readout="pairs", mod_band=band, alpha=getattr(a, "alpha", 0.0), placement=a.placement)
    pairs = draw_pairs(rng)
    aff.grow(cortex, base, pairs=pairs)
    shuf_delta, shuf_axis = aff.shuffled(random.Random(f"{a.seed}|shuf"))
    ctx = {"cortex": cortex, "base": base, "aff": aff, "pairs": pairs, "rng": rng, "shuf_delta": shuf_delta,
           "shuf_axis": shuf_axis}
    if need_moods:   # induced moods: the section-15 integrator over 6 of the induction pool (iteration 4's events)
        ctx["m_pos"], ctx["v_pos"] = mp.induce(aff, cortex, base, rng.sample(mp.POS_EVENTS, 6), rng)
        ctx["m_neg"], ctx["v_neg"] = mp.induce(aff, cortex, base, rng.sample(mp.NEG_EVENTS, 6), rng)
    return ctx


def arm_cfg(ctx):
    """arm -> (modulation on, mood, pattern). off / off_rep: lesion (modulation off)."""
    aff, mpos, mneg = ctx["aff"], ctx["m_pos"], ctx["m_neg"]
    return {"pos": (True, mpos, aff.axis_delta), "neg": (True, mneg, aff.axis_delta),
            "off": (False, mpos, aff.axis_delta), "off_rep": (False, mpos, aff.axis_delta),
            "topic_pos": (True, mpos, aff.topic_delta), "topic_neg": (True, mneg, aff.topic_delta),
            "shuf_pos": (True, mpos, ctx["shuf_delta"]), "shuf_neg": (True, mneg, ctx["shuf_delta"])}


def generate(ctx, q, cfg, sseed, max_tokens):
    """One reply under one arm. Common random numbers: the caller passes the same sseed to every arm."""
    cortex, base, aff = ctx["cortex"], ctx["base"], ctx["aff"]
    on, mood, pattern = cfg
    aff.enabled, aff.mood, aff.axis_delta = on, mood, pattern
    np.random.seed(sseed)
    torch.manual_seed(sseed)
    if aff.placement == "pre":
        st = aff.modulate(cortex, base)
        logits, s2 = cortex.read(f"User: {q}\n\nAssistant:", st)
    else:
        _, s1 = cortex.read(f"User: {q}\n\nAssistant", base)
        s1 = aff.modulate(cortex, s1)
        logits, s2 = cortex.read(":", s1)
    text, _ = cortex.speak(logits, s2, max_tokens=max_tokens)
    return text


def reply_nll(ctx, q, reply):
    """Mean per-token NLL of the reply under the UNmodulated cortex, read the same way the reply was generated."""
    cortex, base = ctx["cortex"], ctx["base"]
    if ctx["aff"].placement == "pre":
        return mp.reply_nll(cortex, base, q, reply)
    toks = cortex.pipe.encode(" " + reply) if reply else []
    if not toks:
        return None
    _, s1 = cortex.read(f"User: {q}\n\nAssistant", base)
    logits, st = cortex.read(":", s1)
    nll = [float(-F.log_softmax(logits.float(), -1)[toks[0]])]
    if len(toks) > 1:
        out, _ = cortex.model.forward(toks[:-1], [x.clone() for x in st], full_output=True)
        lp = F.log_softmax(out.float(), -1)
        nll += [float(-lp[i, toks[i + 1]]) for i in range(len(toks) - 1)]
    return sum(nll) / len(nll)


def make_row(ctx, qi, k, q, text, lexical=True):
    w = si.words(text)
    row = {"q": qi, "k": k, "reply": text, "echo": si.primary_echo_v5(text), "echo_reg": si.primary_echo_registered(text),
           "nll": reply_nll(ctx, q, text), "words": len(w), "refusal": bool(REFUSAL.search(text))}
    if lexical:
        row["lex"] = si.lexical(text)
    return row


# ---------------------------------------------------------------- Stage A
def appraisal_block(ctx, events):
    """Per axis (pair, topic, shuf): the appraisal v (tanh of the centred projection / scale) of every event, one read
    each, no mood carried (project_pure touches no state)."""
    aff, cortex, base = ctx["aff"], ctx["cortex"], ctx["base"]
    axes = {"pair": None, "topic": aff.topic_axis, "shuf": ctx["shuf_axis"]}
    pos, neg, neu = events
    out = {}
    for name, ax in axes.items():
        vals = {pole: [round(aff.project_pure(cortex, base, e, "User", ax)[1], 4) for e in evs]
                for pole, evs in (("pos", pos), ("neg", neg), ("neutral", neu))}
        out[name] = {"values": vals,
                     "sign_acc": {"pos": float(np.mean([v > 0 for v in vals["pos"]])),
                                  "neg": float(np.mean([v < 0 for v in vals["neg"]]))},
                     "neutral_abs": float(np.mean([abs(v) for v in vals["neutral"]]))}
    return out


def gate_a(block):
    p, t = block["pair"], block["topic"]
    bal = lambda b: (b["sign_acc"]["pos"] + b["sign_acc"]["neg"]) / 2  # noqa: E731
    g = {"pair_both_poles_ge_0.75": min(p["sign_acc"].values()) >= 0.75,
         "pair_beats_topic_by_0.25": bal(p) - bal(t) >= 0.25,
         "neutral_abs_le_0.15": p["neutral_abs"] <= 0.15}
    g["pass"] = all(g.values())
    g["margin_over_topic"] = round(bal(p) - bal(t), 4)
    return g


def run_appraise(a):
    t0 = time.time()
    ctx = setup(a, need_moods=False)
    block = appraisal_block(ctx, mp.EVENTS_V5[a.events])
    aff = ctx["aff"]
    out = {"method": METHOD, "stage": "A", "seed": a.seed, "model": a.model, "events": a.events,
           "band_used": list(aff.band_used), "layer_norm": {str(k): v for k, v in aff.layer_norm.items()},
           "pairs": ctx["pairs"], "appraisal": block, "gate": gate_a(block), "provenance": provenance(a.threads),
           "seconds": round(time.time() - t0, 1)}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"seed": a.seed, "band_used": out["band_used"], "gate": out["gate"]} |
                     {k: {"sign_acc": v["sign_acc"], "neutral_abs": v["neutral_abs"]} for k, v in block.items()}, indent=1))


# ---------------------------------------------------------------- Stage B1
def run_calibrate(a):
    """Alpha by fluency only: per alpha, mean NLL of the pos and neg replies / mean NLL of the off replies. The lexical
    score is deliberately not computed here."""
    t0 = time.time()
    ctx = setup(a)
    cfgs = arm_cfg(ctx)
    aff = ctx["aff"]
    qs = mp.QUESTION_SETS[a.questions][:a.n_questions]
    alphas = [float(x) for x in a.alphas.split(",")]
    per = {}
    off_rows = []
    for qi, q in enumerate(qs):
        sseed = zlib.crc32(f"{a.seed}|{qi}|0".encode())
        aff.alpha = 0.0
        off_rows.append(make_row(ctx, qi, 0, q, generate(ctx, q, cfgs["off"], sseed, a.max_tokens), lexical=False))
    off_nll = mp.mean([r["nll"] for r in off_rows])
    for al in alphas:
        aff.alpha = al
        rows = {"pos": [], "neg": []}
        for qi, q in enumerate(qs):
            sseed = zlib.crc32(f"{a.seed}|{qi}|0".encode())
            for arm in rows:
                rows[arm].append(make_row(ctx, qi, 0, q, generate(ctx, q, cfgs[arm], sseed, a.max_tokens), lexical=False))
        nll = {arm: mp.mean([r["nll"] for r in rs]) for arm, rs in rows.items()}
        per[str(al)] = {"nll": nll, "nll_ratio": (nll["pos"] + nll["neg"]) / 2 / off_nll,
                        "identical_to_off": mp.mean([float(r["reply"] == o["reply"]) for arm in rows
                                                     for r, o in zip(rows[arm], off_rows)]),
                        "words": {arm: mp.mean([r["words"] for r in rs]) for arm, rs in rows.items()},
                        "rows": rows}
        print(f"[s{a.seed}] alpha {al}: NLL ratio {per[str(al)]['nll_ratio']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    ok = [al for al in alphas if per[str(al)]["nll_ratio"] <= a.ratio_max]
    out = {"method": METHOD, "stage": "B1", "seed": a.seed, "model": a.model, "questions": a.questions,
           "placement": a.placement, "band_used": list(aff.band_used), "max_tokens": a.max_tokens,
           "moods": {"m_pos": ctx["m_pos"], "m_neg": ctx["m_neg"]}, "off_nll": off_nll, "off_rows": off_rows,
           "alphas": per, "ratio_max": a.ratio_max, "largest_alpha_ok": max(ok) if ok else None,
           "provenance": provenance(a.threads), "seconds": round(time.time() - t0, 1)}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"seed": a.seed, "m_pos": ctx["m_pos"], "m_neg": ctx["m_neg"], "off_nll": off_nll,
                      "nll_ratio": {k: v["nll_ratio"] for k, v in per.items()}, "largest_alpha_ok": out["largest_alpha_ok"]},
                     indent=1))


# ---------------------------------------------------------------- Stage B2 / test
def paired(rows, p, n, key):
    return [x[key] - y[key] for x, y in zip(rows[p], rows[n])]


def contrasts5(rows, seed):
    """Section-17 paired contrasts (per (question, sample) row) with bootstrap CIs, for lex and emb where present."""
    nrng = np.random.default_rng(seed + 5)
    c = {}
    keys = [k for k in ("lex", "emb") if k in rows["pos"][0]]

    def put(name, diffs_by_key):
        c[name] = {}
        for key, d in diffs_by_key.items():
            c[name][key + "_diff"] = float(np.mean(d))
            c[name][key + "_ci95"] = mp.boot_ci(d, nrng)

    if all(x in rows for x in ("topic_pos", "topic_neg")):
        put("disc_topic", {k: [(p[k] - n[k]) - (tp[k] - tn[k]) for p, n, tp, tn in
                              zip(rows["pos"], rows["neg"], rows["topic_pos"], rows["topic_neg"])] for k in keys})
    if all(x in rows for x in ("shuf_pos", "shuf_neg")):
        put("disc_shuf", {k: [(p[k] - n[k]) - (sp[k] - sn[k]) for p, n, sp, sn in
                             zip(rows["pos"], rows["neg"], rows["shuf_pos"], rows["shuf_neg"])] for k in keys})
    if all(x in rows for x in ("topic_pos", "shuf_pos")):
        put("pos_vs_ctrl", {k: [p[k] - (t[k] + s[k]) / 2 for p, t, s in
                               zip(rows["pos"], rows["topic_pos"], rows["shuf_pos"])] for k in keys})
    if all(x in rows for x in ("topic_neg", "shuf_neg")):
        put("neg_vs_ctrl", {k: [n[k] - (t[k] + s[k]) / 2 for n, t, s in
                               zip(rows["neg"], rows["topic_neg"], rows["shuf_neg"])] for k in keys})
    if "off_rep" in rows and "off" in rows:
        c["lesion_repeat"] = {"identical_replies": mp.mean([float(x["reply"] == y["reply"])
                                                            for x, y in zip(rows["off"], rows["off_rep"])])}
    return c


def score(rows, seed, embed):
    summary, contrasts = mp.score_rows(rows, {"pos": [], "neg": []}, seed, embed=embed)
    for arm in rows:   # mp's summary has no refusal count
        summary[arm]["refusals"] = int(sum(r["refusal"] for r in rows[arm]))
    return summary, contrasts | contrasts5(rows, seed)


def run_replies(a):
    t0 = time.time()
    ctx = setup(a)
    aff, cfgs = ctx["aff"], arm_cfg(ctx)
    appraisal = None
    if a.events != "none":
        appraisal = appraisal_block(ctx, mp.EVENTS_V5[a.events])
    arms = a.arms.split(",")
    if "off" in arms and "off_rep" not in arms and not a.no_repeat:
        arms.append("off_rep")
    qs = mp.QUESTION_SETS[a.questions][:a.n_questions]
    rows = {arm: [] for arm in arms}
    for qi, q in enumerate(qs):
        for k in range(a.samples):
            sseed = zlib.crc32(f"{a.seed}|{qi}|{k}".encode())
            for arm in arms:
                rows[arm].append(make_row(ctx, qi, k, q, generate(ctx, q, cfgs[arm], sseed, a.max_tokens)))
        print(f"[s{a.seed}] q{qi} done {time.time() - t0:.0f}s", flush=True)
    summary, contrasts = score(rows, a.seed, embed=not a.no_embedding)
    out = {"method": METHOD, "seed": a.seed, "model": a.model, "readout": "pairs", "alpha": a.alpha,
           "placement": a.placement, "band_used": list(aff.band_used), "questions": a.questions, "samples": a.samples,
           "max_tokens": a.max_tokens, "n_rand": 0, "pairs": ctx["pairs"], "events": a.events,
           "appraisal": appraisal,
           "appraisal_sign_acc": None if appraisal is None else {k: v["sign_acc"] | {"neutral_abs": v["neutral_abs"]}
                                                                for k, v in appraisal.items()},
           "induction": {"m_pos": ctx["m_pos"], "m_neg": ctx["m_neg"], "v_pos": ctx["v_pos"], "v_neg": ctx["v_neg"],
                         "alpha_x_m_pos": a.alpha * ctx["m_pos"], "alpha_x_m_neg": a.alpha * ctx["m_neg"]},
           "provenance": provenance(a.threads), "summary": summary, "contrasts": contrasts, "rows": rows,
           "seconds": round(time.time() - t0, 1)}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("seed", "alpha", "induction", "contrasts", "appraisal_sign_acc")} |
                     {"summary": summary}, indent=1))


def add_embedding(path):
    """Score the saved replies of a --no-embedding run with the frozen MiniLM instrument and recompute summary and
    contrasts (nodes without transformers). Same rng order as an on-node run."""
    j = json.load(open(path))
    assert not any("emb" in r for arm in j["rows"].values() for r in arm), "already has embedding"
    j["summary"], j["contrasts"] = score(j["rows"], j["seed"], embed=True)
    j["provenance"]["embedding_added_on"] = socket.gethostname()
    with open(path, "w") as fh:
        json.dump(j, fh, indent=1)
    c = j["contrasts"]
    print(path, "pos_vs_ctrl emb", c.get("pos_vs_ctrl", {}).get("emb_diff"), "neg_vs_ctrl emb",
          c.get("neg_vs_ctrl", {}).get("emb_diff"))


# ---------------------------------------------------------------- table: section 17 C1-C9
def criteria(j):
    c, s, ind, rows = j["contrasts"], j["summary"], j["induction"], j["rows"]
    acc = (j.get("appraisal_sign_acc") or {}).get("pair")
    echo_d = (s["pos"]["echo_rate"] + s["neg"]["echo_rate"]) / 2 - s["off"]["echo_rate"]
    nll_ratio = (s["pos"]["nll"] + s["neg"]["nll"]) / 2 / s["off"]["nll"]
    len_ok = all(abs(s[x]["words"] / s["off"]["words"] - 1) <= 0.25 for x in ("pos", "neg"))
    has_emb = "emb_diff" in c["pos_vs_ctrl"]
    cr = {
        "C1_appraisal": None if acc is None else
        bool(acc["pos"] >= 0.75 and acc["neg"] >= 0.75 and acc["neutral_abs"] <= 0.15),
        "C2_mood_signs": bool(ind["m_pos"] > 0 > ind["m_neg"]),
        "C3_pos_vs_ctrl_ci_above_0": bool(c["pos_vs_ctrl"]["lex_ci95"][0] > 0),
        "C4_neg_vs_ctrl_ci_below_0": bool(c["neg_vs_ctrl"]["lex_ci95"][1] < 0),
        "C5_embedding_sign": bool(c["pos_vs_ctrl"]["emb_diff"] > 0 > c["neg_vs_ctrl"]["emb_diff"]) if has_emb else None,
        "C6_echo_le_0.10": bool(echo_d <= 0.10),
        "C7_nll_le_1.20_len_pm25": bool(nll_ratio <= 1.20 and len_ok),
        "C8_neg_refusals_le_off_plus_3": bool(s["neg"]["refusals"] <= s["off"]["refusals"] + 3),
        "C9_lesion_repeat_and_pos_changes": bool(c["lesion_repeat"]["identical_replies"] == 1.0 and
                                                 1.0 - c["pos_off"]["identical_replies"] >= 0.5),
    }
    return cr, {"echo_d": echo_d, "nll_ratio": nll_ratio, "len_ok": len_ok,
                "pos_changed": 1.0 - c["pos_off"]["identical_replies"]}


NEED = {"C2_mood_signs": 6, "C6_echo_le_0.10": 6}


def table(d, prefix="rep"):
    import glob
    files = sorted(glob.glob(f"{d}/{prefix}_s*.json"), key=lambda x: int(re.findall(r"_s(\d+)\.json", x)[0]))
    if not files:
        print(f"no {prefix}_s*.json in {d}")
        return None
    print("| seed | alpha | m_pos | m_neg | alpha*m +/- | acc pair +/- (neut) | lex pos-ctrl [CI] | lex neg-ctrl [CI] | "
          "emb pos/neg-ctrl | disc topic [CI] | disc shuf [CI] | pos-off | neg-off | echo | NLL ratio | refusals n/o | "
          "changed | met |")
    print("|" + "---|" * 18)
    tally, n = {}, len(files)
    pooled = {k: [] for k in ("disc_topic", "disc_shuf")}
    for f in files:
        j = json.load(open(f))
        cr, info = criteria(j)
        c, s, ind = j["contrasts"], j["summary"], j["induction"]
        for k, v in cr.items():
            tally[k] = tally.get(k, 0) + int(bool(v))
        r = j["rows"]
        for key, (p, q, tp, tq) in {"disc_topic": ("pos", "neg", "topic_pos", "topic_neg"),
                                    "disc_shuf": ("pos", "neg", "shuf_pos", "shuf_neg")}.items():
            pooled[key] += [(x["lex"] - y["lex"]) - (u["lex"] - v["lex"]) for x, y, u, v in zip(r[p], r[q], r[tp], r[tq])]
        ci = lambda x, k="lex": f"{x[k + '_diff']:+.2f} [{x[k + '_ci95'][0]:+.2f},{x[k + '_ci95'][1]:+.2f}]"  # noqa: E731
        acc = (j.get("appraisal_sign_acc") or {}).get("pair")
        accs = "n/a" if acc is None else f"{acc['pos']:.2f}/{acc['neg']:.2f} ({acc['neutral_abs']:.2f})"
        emb = (f"{c['pos_vs_ctrl']['emb_diff']:+.3f}/{c['neg_vs_ctrl']['emb_diff']:+.3f}"
               if "emb_diff" in c["pos_vs_ctrl"] else "n/a")
        print(f"| {j['seed']} | {j['alpha']} | {ind['m_pos']:+.3f} | {ind['m_neg']:+.3f} | "
              f"{ind['alpha_x_m_pos']:+.3f}/{ind['alpha_x_m_neg']:+.3f} | {accs} | {ci(c['pos_vs_ctrl'])} | "
              f"{ci(c['neg_vs_ctrl'])} | {emb} | {ci(c['disc_topic'])} | {ci(c['disc_shuf'])} | "
              f"{c['pos_off']['lex_diff']:+.2f} | {c['neg_off']['lex_diff']:+.2f} | {info['echo_d']:+.3f} | "
              f"{info['nll_ratio']:.2f} | {s['neg']['refusals']}/{s['off']['refusals']} | {info['pos_changed']:.2f} | "
              f"{sum(bool(v) for v in cr.values())}/9 (unmet: {','.join(k[:2] for k, v in cr.items() if not v) or '-'}) |")
    print(f"\nCriteria over {n} seeds (bar 5/6; C2 and C6 6/6; a criterion that could not be evaluated is 'n/a'):")
    allpass = n == 6
    for k in cr:
        if k not in tally:
            continue
        ok = tally[k] >= NEED.get(k, 5)
        na = all(criteria(json.load(open(f)))[0][k] is None for f in files)
        allpass &= ok and not na
        print(f"- {k}: {tally[k]}/{n} {'n/a (not evaluated)' if na else 'PASS' if ok else 'FAIL'}")
    nrng = np.random.default_rng(0)
    disc = {k: mp.boot_ci(v, nrng) for k, v in pooled.items() if v}
    print("Pooled discriminator (lex, paired bootstrap over all rows; must exclude 0): " +
          ", ".join(f"{k} mean {np.mean(pooled[k]):+.2f} CI [{ci[0]:+.2f},{ci[1]:+.2f}] "
                    f"{'excludes 0' if ci[0] > 0 or ci[1] < 0 else 'includes 0'}" for k, ci in disc.items()))
    print(f"\nOVERALL: {'PASS (a lead only)' if allpass else 'NOT PASS (or not complete: need 6 seeds, all criteria evaluated)'}")
    return {"tally": tally, "n": n, "pooled": disc, "pass": bool(allpass)}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--band", default="8,16", help="modulation / read band [lo,hi); scaled to the middle third "
                                                      "when the model has fewer than 16 layers")
        p.add_argument("--placement", choices=["pre", "post"], default="post")
        p.add_argument("--out", required=True)

    p = sub.add_parser("appraise")
    common(p)
    p.add_argument("--events", choices=sorted(mp.EVENTS_V5), default="dev")
    p.set_defaults(fn=run_appraise)
    p = sub.add_parser("calibrate")
    common(p)
    p.add_argument("--questions", choices=sorted(mp.QUESTION_SETS), default="mixed_dev")
    p.add_argument("--alphas", default="0.03,0.1,0.3")
    p.add_argument("--max-tokens", type=int, default=40)
    p.add_argument("--n-questions", type=int, default=None, help="first N questions only (smoke runs)")
    p.add_argument("--ratio-max", type=float, default=RATIO_MAX)
    p.set_defaults(fn=run_calibrate)
    p = sub.add_parser("replies")
    common(p)
    p.add_argument("--questions", choices=sorted(mp.QUESTION_SETS), default="mixed_test")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--max-tokens", type=int, default=40)
    p.add_argument("--n-questions", type=int, default=None, help="first N questions only (smoke runs)")
    p.add_argument("--events", choices=sorted(mp.EVENTS_V5) + ["none"], default="test",
                   help="V5 events for the appraisal check recorded in the JSON (C1 is read from the test split)")
    p.add_argument("--no-embedding", action="store_true")
    p.add_argument("--no-repeat", action="store_true", help="skip the off_rep arm (the C9 lesion repeat)")
    p.set_defaults(fn=run_replies)
    p = sub.add_parser("table")
    p.add_argument("dir")
    p.add_argument("prefix", nargs="?", default="rep")
    p.set_defaults(fn=lambda a: table(a.dir, a.prefix))
    p = sub.add_parser("add-embedding")
    p.add_argument("paths", nargs="+")
    p.set_defaults(fn=lambda a: [add_embedding(x) for x in a.paths])
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
