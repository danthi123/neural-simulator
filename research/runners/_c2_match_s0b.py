#!/usr/bin/env python
"""_c2_match_s0b.py — S0b: separability of `acc` at the drive level (c2 match-unit step 1).

Read-only analysis of landed c2_hops_step0 shards (research/findings/raw/<wave>/s<seed>/*.json).
Writes nothing unless --out is given; prints all numbers to stdout.

Unit of analysis: the (cue, block) pair — each shard runs CUES_A cues against its K kb blocks,
so every (cue, block) pair is classified by the cue words lit in that block's drive.

Classification (per (cue, block); cue = (a, x), block drive lit words = dlw[b] = [dA-lit, dX-lit]):
  true        : block in cue['true_matches']
  both-lit    : false AND a in dA-lit AND x in dX-lit   (per-role reading)
  one-lit     : false AND exactly one of (a in dA-lit, x in dX-lit)
  neither     : false AND neither
The draft's 98/86/31/1045 fingerprint further gates the one-lit class on m{b} FIRING at the
current cue (m > match_thresh): 31 one-lit blocks fired (27 of them also fired at the previous
cue), and the 'neither-lit' bucket (1045) = strict-neither + one-lit-but-silent, all with 0 fire.

S0b numbers (per the draft, on the reset-ON / gatereset wave), on the 86 both-lit FALSE pairs:
  acc_true    = pool_spikes.get(block.decoded_patient, 0.0)
  acc_spurious= see --help / report: cue's true filler if != block filler, else the alternative
                candidate patient implied by the cue, else the other lit pool of the block
                (0.0 if only the true filler is lit).
  cross-block AUC  = P(acc_true_i > acc_spurious_j) over all cross pairs (i, j) of the n pairs,
                0.5 for ties (rank AUC between the n true-filler accs and the n spurious accs).
  paired AUC      = P(acc_true > acc_spurious) within a pair + 0.5 ties (also printed).
  within-block margin = acc_true - acc_spurious.

Usage:
  .venv/bin/python research/runners/_c2_match_s0b.py <ROOT> [--out X.json] [--cross-check-root R]
ROOT is e.g. research/findings/raw/_c2_step0_gatereset (any wave root works; the fingerprint
reproduces on _c2_step0_relay).
"""
import argparse
import glob
import json
import os
import statistics
from collections import Counter

CUE_ORDER = ["heron,hunt", "heron,eat", "frog,eat", "frog,chase"]


def shards(root):
    """All landed shard JSONs that carry c2_hops.reads, as (seed, arm, reads, c-blocks)."""
    out = []
    for sd in sorted(glob.glob(os.path.join(root, "s*"))):
        seed = int(os.path.basename(sd)[1:])
        for f in sorted(glob.glob(os.path.join(sd, "*.json"))):
            if f.endswith(".prov.json"):
                continue
            d = json.load(open(f))
            r = (d.get("c2_hops") or {}).get("reads")
            if r is None:
                continue
            arm = os.path.basename(f).split("__")[-1].replace(".json", "")
            out.append((seed, arm, r))
    return out


def classify(r):
    """Per (cue, block) classification + m-rate facts, for one shard's reads."""
    K = r["K"]
    thr = float(r["b"]["match_thresh"])
    dlw = r["b"]["drives_lit_words"]
    cues = r["b"]["cues"]
    pairs = []  # dicts per (cue, block)
    for ck in CUE_ORDER:
        c = cues[ck]
        a, x = c["cue"]
        tm = set(c["true_matches"])
        m = c.get("m_rates")
        prev_ck = CUE_ORDER[CUE_ORDER.index(ck) - 1]
        m_prev = cues[prev_ck].get("m_rates")
        for b in range(K):
            dA, dX = set(dlw[b][0]), set(dlw[b][1])
            n_lit = (a in dA) + (x in dX)
            if b in tm:
                cls = "true"
            else:
                cls = {2: "both", 1: "one", 0: "neither"}[n_lit]
            mb = None if m is None else float(m[b])
            mp = None if m_prev is None else float(m_prev[b])
            pairs.append(dict(cue=ck, a=a, x=x, b=b, cls=cls, m=mb, m_prev=mp,
                              fired=(mb is not None and mb > thr),
                              prev_fired=(mp is not None and mp > thr),
                              n_lit=n_lit, other_absent=(n_lit == 1 and
                                  (x if (a in dA) else a) not in (dA | dX))))
    return pairs, thr


def fingerprint(pairs, thr):
    """Draft fingerprint: 98 / 86 / 31 / 1045 (+ 51 fired / 35 zero among both-lit)."""
    cnt = Counter(p["cls"] for p in pairs)
    both = [p for p in pairs if p["cls"] == "both"]
    one = [p for p in pairs if p["cls"] == "one"]
    neither = [p for p in pairs if p["cls"] == "neither"]
    both_fired = [p for p in both if p["fired"]]
    both_zero = [p for p in both if p["m"] == 0.0]
    one_fired = [p for p in one if p["fired"]]
    one_fired_prev = sum(1 for p in one_fired if p["prev_fired"])
    # strict previous cue only (skip the first cue, which has no previous)
    one_fired_prev_strict = sum(1 for p in one_fired if p["cue"] != CUE_ORDER[0] and p["prev_fired"])
    rest_silent = [p for p in one + neither if not p["fired"]]
    one_strict = sum(1 for p in one if p["other_absent"])
    def rng(ps):
        vs = [p["m"] for p in ps if p["m"] is not None and p["m"] > 0]
        return (min(vs), max(vs)) if vs else None
    return {
        "true": cnt["true"], "both": cnt["both"], "one": cnt["one"], "neither": cnt["neither"],
        "both_fired_m_gt0": len([p for p in both if p["m"] and p["m"] > 0]),
        "both_fired_m_gt_thr": len(both_fired),
        "both_m_eq_0": len(both_zero),
        "both_fired_m_range": rng(both_fired),
        "one_fired_m_gt_thr": len(one_fired),
        "one_fired_m_range": rng(one_fired),
        "one_fired_prev_cue_fired_wrap": one_fired_prev,
        "one_fired_prev_cue_fired_strict": one_fired_prev_strict,
        "one_strict_other_word_absent": one_strict,
        "rest_bucket_neither_plus_one_silent": len(rest_silent),
        "rest_bucket_any_fire": sum(1 for p in rest_silent if p["fired"]),
        "match_thresh": thr,
    }


def spurious_word(cue_true_patients, p_b, pool_spikes):
    """The spurious word for a false pair, and which branch was used.

    cue_true_patients : set of decoded_patient over the cue's true_matches blocks (this shard)
    p_b               : the false block's decoded_patient (true filler)
    pool_spikes       : the false block's pool_spikes {word: acc}
    Rule (conservative): candidates = cue_true_patients - {p_b}; if non-empty take the candidate
    with the HIGHEST acc in this block (hardest for the true filler to beat; ties -> lex smallest).
    If empty (cue has no true match, or its only filler == p_b): take the OTHER lit pool of the
    block (pool_spikes keys - {p_b}) with the highest acc; if only the true filler is lit -> 0.0.
    """
    pool = dict(pool_spikes or {})
    cand = sorted(set(cue_true_patients) - {p_b})
    if cand:
        w = max(cand, key=lambda t: (pool.get(t, 0.0), t))
        return w, pool.get(w, 0.0), "cue_filler"
    others = sorted(set(pool) - {p_b})
    if others:
        w = max(others, key=lambda t: (pool.get(t, 0.0), t))
        return w, pool.get(w, 0.0), "block_lit_other"
    return None, 0.0, "only_true_filler_lit"


def s0b(root):
    """S0b numbers on the both-lit FALSE pairs of this wave root."""
    per_seed = {}
    rows = []  # per pair: seed, arm, cue, b, acc_true, acc_spurious, margin, branch
    for seed, arm, r in shards(root):
        pairs, thr = classify(r)
        cblocks = {blk["block"]: blk for blk in r["c"]["blocks"]}
        for p in pairs:
            if p["cls"] != "both":
                continue
            blk = cblocks[p["b"]]
            p_b = blk["decoded_patient"]
            cue_tm = set(r["b"]["cues"][p["cue"]]["true_matches"])
            cue_pats = {cblocks[t]["decoded_patient"] for t in cue_tm}
            acc_true = float((blk.get("pool_spikes") or {}).get(p_b, 0.0))
            w, acc_spur, branch = spurious_word(cue_pats, p_b, blk.get("pool_spikes") or {})
            rows.append(dict(seed=seed, arm=arm, cue=p["cue"], b=p["b"], p_b=p_b,
                             spurious=w, acc_true=acc_true, acc_spurious=float(acc_spur),
                             margin=acc_true - float(acc_spur), branch=branch,
                             n_pool_lits=len(blk.get("pool_spikes") or {})))
    return rows


def rank_auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie) over all cross pairs (Mann-Whitney AUC)."""
    if not pos or not neg:
        return None
    wins = ties = 0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def paired_auc(rows):
    wins = ties = 0
    for row in rows:
        if row["acc_true"] > row["acc_spurious"]:
            wins += 1
        elif row["acc_true"] == row["acc_spurious"]:
            ties += 1
    return (wins + 0.5 * ties) / len(rows) if rows else None


def fmt(x):
    return "n/a" if x is None else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser(description="S0b acc-separability at the drive level")
    ap.add_argument("root", help="wave root, e.g. research/findings/raw/_c2_step0_gatereset")
    ap.add_argument("--out", help="optional JSON dump path (the only file ever written)")
    ap.add_argument("--cross-check-root", help="second wave root: verify identical drive lit-sets")
    args = ap.parse_args()

    all_pairs = []
    thr = None
    for seed, arm, r in shards(args.root):
        pairs, thr = classify(r)
        for p in pairs:
            p["seed"], p["arm"] = seed, arm
        all_pairs.extend(pairs)
    fp = fingerprint(all_pairs, float(thr or 0.06))
    print(f"== classification on {args.root} ({len(all_pairs)} (cue,block) pairs) ==")
    for k, v in fp.items():
        print(f"  {k}: {v}")
    print("  draft fingerprint target: true=98 both=86 one_fired=31 rest=1045 "
          "both: 51 fired / 35 zero ; 27/31 prev-cue")

    if args.cross_check_root:
        def drives(root):
            return {(seed, arm): [list(map(list, blk)) for blk in r["b"]["drives_lit_words"]]
                    for seed, arm, r in shards(root)}
        d1, d2 = drives(args.root), drives(args.cross_check_root)
        same_keys = set(d1) == set(d2)
        same = same_keys and all(d1[k] == d2[k] for k in d1)
        n_diff = 0 if same else sum(1 for k in set(d1) & set(d2) if d1[k] != d2[k])
        print(f"== drive lit-sets {args.root} vs {args.cross_check_root}: "
              f"same shard keys={same_keys} identical={same} differing shards={n_diff}")

    rows = s0b(args.root)
    n = len(rows)
    print(f"\n== S0b on {args.root} (unit = (cue, block) pair) ==")
    print(f"  n both-lit false pairs: {n}")
    print(f"  n distinct blocks: {len({(r['seed'], r['arm'], r['b']) for r in rows})}")
    branches = Counter(r["branch"] for r in rows)
    print(f"  spurious-word branches: {dict(branches)}")
    print(f"  acc_spurious == 0 count: {sum(1 for r in rows if r['acc_spurious'] == 0.0)}")
    print(f"  acc_true == 0 count:     {sum(1 for r in rows if r['acc_true'] == 0.0)}")

    pos = [r["acc_true"] for r in rows]
    neg = [r["acc_spurious"] for r in rows]
    auc_pooled = rank_auc(pos, neg)
    print(f"  cross-block AUC pooled: {fmt(auc_pooled)}   (paired AUC: {fmt(paired_auc(rows))})")

    per_seed = {}
    for s in sorted({r["seed"] for r in rows}):
        rs = [r for r in rows if r["seed"] == s]
        a = rank_auc([r["acc_true"] for r in rs], [r["acc_spurious"] for r in rs])
        per_seed[s] = a
        print(f"  seed {s}: n={len(rs)} cross-block AUC={fmt(a)} paired AUC={fmt(paired_auc(rs))}")

    margins = [r["margin"] for r in rows]
    gt0 = sum(1 for m in margins if m > 0)
    eq0 = sum(1 for m in margins if m == 0)
    print(f"  within-block margin > 0: {gt0}/{n} = {fmt(gt0 / n if n else None)}")
    print(f"  within-block margin = 0: {eq0}/{n}   margin >= 0 fraction: "
          f"{fmt((gt0 + eq0) / n if n else None)}")
    if margins:
        ms = sorted(margins)
        print(f"  margin min/median/max: {ms[0]:.4f} / {statistics.median(ms):.4f} / {ms[-1]:.4f}")
        buckets = Counter()
        for m in margins:
            if m <= -0.5:
                buckets["<=-0.5"] += 1
            elif m < 0:
                buckets["(-0.5,0)"] += 1
            elif m == 0:
                buckets["0"] += 1
            elif m <= 0.5:
                buckets["(0,0.5]"] += 1
            elif m <= 1.0:
                buckets["(0.5,1]"] += 1
            else:
                buckets[">1"] += 1
        print(f"  margin histogram: {dict(buckets)}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"root": args.root, "fingerprint": fp,
                       "n_both_lit_false_pairs": n,
                       "auc_per_seed": per_seed, "auc_pooled": auc_pooled,
                       "auc_paired": paired_auc(rows),
                       "margin_gt0_fraction": gt0 / n if n else None,
                       "margin_ge0_fraction": (gt0 + eq0) / n if n else None,
                       "margins": margins, "rows": rows}, fh, indent=1)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
