"""LEXICON v2 de-risk — is the D6 referent (noun-category) DECISION made by a coupled spiking WTA whose graded drive
comes through Hebbian-learned frame->category synapses, is that learned edge load-bearing, and does it let the
multi-referent WM faculty be EXERCISED on "the wolf watches the owl"? (language lane E, 2026-09-23)

Mechanism under test: `research/runners/lexicon_spiking_frame_category.py` (see its docstring). v1 (host
label-spreading, spike-relayed) is the baseline of `_lexicon_learned_referent_derisk.py`; its 6-seed verdict is banked
separately (NO-GO on G7 recover, amendment A1).

Ground truth: the INDEPENDENT spaCy dominant-POS fixture (`research/fixtures/lexicon_referent_pos_gt.json`). HELD-OUT
= every fixture word not in any hand list (D6 table, HAND_NOUN_SEEDS, NONNOUN_SEEDS). Cross-validated curriculum:
k=12 seeds per class drawn per seed (rng = seed) from the hand lists. Evaluation words: a fixed sample of N_EVAL=400
held-out words (rng seed+5); abstain counts WRONG. Balanced accuracy = mean(TPR on nouns, TNR on non-nouns).

Parameters were calibrated on DEV seed 7 only (not an evaluation seed). Dev numbers SEEN before this
pre-registration (seed 7, k=12, 300 eval words): intact balanced acc 0.820 with 5% abstain at K_OCC=32/EPOCHS=8
(0.798 at K_OCC=24/EPOCHS=4); learned-edge lesion 0.138-0.156 (68-70% abstain); competition lesion 0.774 with
loser/winner rate ratio 0.17 -> 0.61; an unsupervised competitive self-training phase added nothing (0.794 vs 0.798)
and is NOT part of the mechanism; full 38+37 curriculum 0.808. The full `--part main` on dev seed 7 (final config,
open-vocabulary environment): bacc 0.798, abstain 0.06, learned-edge lesion 0.174 (decided-only 0.503, abstain
0.68), competition lesion 0.792 with loser/winner 0.150 -> 0.463, Spearman 0.968, full rebuild deterministic,
organ recover 0.67 / lesion 0.00 / false-positive 0.00 / hand 0.00, capability intact ["wolf","owl"] (watches ->
non-referent), capability under learned-edge lesion ["wolf","watches","owl"]. The S7 recover threshold (0.60) and
the S8 lesion arm's report-only status were set from these dev numbers, before any evaluation seed ran. The null
(S2) was timed on dev seed 7 with one 40-permutation batch; its value is not used for any threshold.

PRE-REGISTERED GO GATE (seeds 42 43 44 100 101 102). EVIDENCE gates (each can fail):
  S1 LEARNING: spiking held-out balanced accuracy mean >= 0.75 AND every seed >= 0.70.
  S2 PERMUTATION NULL (part=null): replica-0 (true labels) balanced accuracy on the 200-word null subset lies at or
     above the 99th percentile of >= 1000 seed-label permutations (same 24 curriculum words, labels permuted, each
     permutation retrained from W_INIT on its own replica circuit) on >= 5 of 6 seeds.
  S3 LEARNED-EDGE LESION (FR->CN/CX restored to W_INIT; the circuit and drive stay): (a) lesioned balanced acc mean
     <= 0.55 and intact - lesioned >= 0.20 on every seed; (b) balanced acc over the words the lesioned circuit still
     DECIDES is <= 0.60 (mean over seeds where it is defined) — the residual decisions carry no category.
  S4 COMPETITION (IN->CX and IX->CN zeroed): mean loser/winner rate ratio rises by >= 0.25 (absolute) on every seed.
  S5 GRADED DRIVE: Spearman(synaptic drive margin through the installed weights, spike-rate margin CN-CX) >= 0.50
     mean over seeds — the WTA outcome tracks graded synaptic input, not a binary host sign.
  S7 ORGAN (reply level, real MultiReferentWMOrgan, "the A and the B walked in" -> "who are we talking about?",
     A,B held-out GT nouns off the hand table, 12 trials): learned recovers BOTH mean >= 0.60; learned-edge lesion
     recover mean <= 0.20; false-positive arm ("we V1 and V2 all day", held-out GT non-nouns) in-scope mean <= 0.20.
  S8 CAPABILITY (deployment curriculum 38+37, per seed): with the learned lexicon, "the wolf watches the owl"
     extracts exactly ["wolf","owl"] and the organ's `held` turn (after "the fox and the wolf walked in") is
     IN-SCOPE with n_referents == 2, on >= 5 of 6 seeds. The same turn under the learned-edge lesion is REPORTED,
     not gated: on dev seed 7 the lesioned circuit still decided "owl" (and "watches") as referents — the
     pre-learning circuit makes ~30% residual decisions with no category information (S3b), so a single-word lesion
     outcome is a coin flip; the gated lesion evidence is the 12-trial population arm S7.
INTEGRITY SMOKES (must pass; pass by construction, NOT evidence): S6 determinism (a full rebuild from the corpus —
tokens, frame environment, bridge, training — reproduces the learned-weight hash and the decisions of 100 eval words);
afferent-zero lesion abstain rate == 1.0; hand-table organ baseline in-scope <= 0.10; the weight hash holds across
every read (a lesion is verified to hold at measurement; `decide()` raises otherwise).

Run:
  SIM_BACKEND=numpy python -u -m research.runners._lexicon_spiking_referent_derisk --part main --seeds 42 \
      --json research/findings/raw/_lexicon_spiking_referent/main_s42.json
  SIM_BACKEND=numpy python -u -m research.runners._lexicon_spiking_referent_derisk --part null --seeds 42 \
      --json research/findings/raw/_lexicon_spiking_referent/null_s42.json
  python -m research.runners._lexicon_spiking_referent_derisk --score research/findings/raw/_lexicon_spiking_referent
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._comprehension_learned_animacy_cue_derisk import load_tokens, build_vocab  # noqa: E402
from research.runners import lexicon_spiking_frame_category as L  # noqa: E402
from research.runners._lexicon_learned_referent_derisk import _held_out, _bacc, FIXTURE  # noqa: E402
from tools.lab import lever, undefined_if_empty  # noqa: E402

SEEDS6 = [42, 43, 44, 100, 101, 102]
K_SEED = 12
N_EVAL = 400
N_NULL_EVAL = 200
GATE = dict(s1_mean=0.75, s1_min=0.70, s2_pct=99.0, s2_min_seeds=5, s3_les_mean=0.55, s3_drop=0.20,
            s3_decided_max=0.60, s4_ratio_rise=0.25, s5_spearman=0.50, s7_recover=0.60, s7_lesion_max=0.20,
            s7_fp_max=0.20, s7_hand_max=0.10, s8_min_seeds=5)
HOLD, HELD = "the fox and the wolf walked in", "the wolf watches the owl"


def _env(corpus, max_chars=8_000_000, top_v=2000):
    tokens = load_tokens(corpus, max_chars)
    vocab, freq = build_vocab(tokens, top_v)
    env = L.FrameEnvironment(tokens, vocab + [w for w in L.HAND_NOUN_SEEDS + L.NONNOUN_SEEDS if w not in vocab])
    return tokens, vocab, env


def _eval_words(seed, held):
    return sorted(np.random.default_rng(seed + 5).choice(sorted(held), N_EVAL, replace=False).tolist())


def _decided_bacc(pred, gt):
    """Balanced accuracy over the words that received a decision; None (UNDEFINED) if a class has none."""
    d = {w: p for w, p in pred.items() if p is not None}
    pos = [w for w in d if gt[w]]
    neg = [w for w in d if not gt[w]]
    if not pos or not neg:
        return None
    return 0.5 * (sum(d[w] is True for w in pos) / len(pos) + sum(d[w] is False for w in neg) / len(neg))


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else None


def _read_all(lex, words):
    pred, rn, rx = {}, {}, {}
    for w in words:
        d, a, b = lex.decide(w)
        pred[w] = d[0]
        if a is not None:
            rn[w], rx[w] = float(a[0]), float(b[0])
    return pred, rn, rx


def _lose_win(rn, rx):
    r = [min(rn[w], rx[w]) / max(rn[w], rx[w]) for w in rn if max(rn[w], rx[w]) > 0]
    return float(np.mean(r)) if r else None


class _HandOnly:
    def is_referent(self, word):
        return False


def _organ_arm(seed, lex, noun_pool, nonnoun_pool, n_pairs):
    from research.runners.d6_multiref_wm_production_organ import MultiReferentWMOrgan
    rng = np.random.default_rng(seed + 991)
    q = "who are we talking about?"
    trials, fps = [], []
    for _ in range(n_pairs):
        a, b = [str(x) for x in rng.choice(noun_pool, 2, replace=False)]
        row = {"a": a, "b": b}
        for cond in ("hand", "learned", "learned_edge"):
            org = MultiReferentWMOrgan(seed=seed)
            if cond == "hand":
                org.referent_lexicon = _HandOnly()
            else:
                lex.set_lesion(None if cond == "learned" else "learned_edge")
                org.referent_lexicon = lex
            j1 = org.judge(f"the {a} and the {b} walked in")
            j2 = org.judge(q)
            rec = set(((j2 or {}).get("recovered") or {}).values())
            row[cond] = {"in_scope": j1 is not None, "readout": (j2 or {}).get("readout"),
                         "recovered_both": bool(j2 is not None and {a, b} <= rec)}
        lex.set_lesion(None)
        trials.append(row)
    for _ in range(n_pairs):
        v1, v2 = [str(x) for x in rng.choice(nonnoun_pool, 2, replace=False)]
        org = MultiReferentWMOrgan(seed=seed)
        org.referent_lexicon = lex
        j1 = org.judge(f"we {v1} and {v2} all day")
        fps.append({"v1": v1, "v2": v2, "in_scope": j1 is not None, "refs": (j1 or {}).get("input_order")})
    n = len(trials)
    return {"trials": trials, "fp_trials": fps,
            "learned_recover_rate": sum(t["learned"]["recovered_both"] for t in trials) / n,
            "lesion_recover_rate": sum(t["learned_edge"]["recovered_both"] for t in trials) / n,
            "hand_in_scope_rate": sum(t["hand"]["in_scope"] for t in trials) / n,
            "reply_differs_learned_vs_lesion_rate":
                sum(t["learned"]["readout"] != t["learned_edge"]["readout"] for t in trials) / n,
            "fp_in_scope_rate": sum(f["in_scope"] for f in fps) / len(fps)}


def capability_arm(seed, env):
    """S8: the deployment curriculum (all 38+37 seeds) at this seed; the battery's hold -> held pair on a real organ."""
    from research.runners.d6_multiref_wm_production_organ import MultiReferentWMOrgan, extract_referents
    lex = L.SpikingFrameCategoryLexicon(seed, env, 1)
    words, labels = L.seed_curriculum(env)
    lex.train(words, labels[:, None])
    out = {"n_curriculum": len(words)}
    for cond in (None, "learned_edge"):
        lex.set_lesion(cond)
        org = MultiReferentWMOrgan(seed=seed)
        org.referent_lexicon = lex
        j1 = org.judge(HOLD)
        j2 = org.judge(HELD)
        out[cond or "intact"] = {
            "held_refs": extract_referents(HELD, referent_lexicon=lex),
            "owl": lex.classify("owl"), "watches": lex.classify("watches"),
            "hold_in_scope": j1 is not None, "held_in_scope": j2 is not None,
            "held_n_referents": (j2 or {}).get("n_referents"), "held_input_order": (j2 or {}).get("input_order")}
    lex.set_lesion(None)
    return out


def run_main(seed, corpus, pos_gt, n_pairs):
    t0 = time.time()
    tokens, vocab, env = _env(corpus)
    held = _held_out(vocab, pos_gt)
    words = _eval_words(seed, held)
    gt = {w: held[w] for w in words}
    lex = L.SpikingFrameCategoryLexicon(seed, env, 1)
    sw, sl = L.seed_curriculum(env, k_seed=K_SEED, cv_seed=seed)
    lex.train(sw, sl[:, None])
    res = {"seed": seed, "n_held": len(held), "n_eval": len(words), "n_eval_noun": sum(gt.values()),
           "curriculum": sw, "w_hash": hashlib.sha256(lex.W.tobytes()).hexdigest()}

    pred, rn, rx = _read_all(lex, words)
    res["bacc_spiking"] = _bacc(pred, gt)
    res["abstain"] = sum(p is None for p in pred.values()) / len(words)
    res["frac_referent"] = sum(p is True for p in pred.values()) / len(words)
    res["lose_win_intact"] = _lose_win(rn, rx)
    gd = {w: float(lex.graded_drive(w)[0]) for w in rn}
    res["spearman_drive_vs_rate"] = _spearman([gd[w] for w in rn], [rn[w] - rx[w] for w in rn])

    for kind in ("learned_edge", "competition", "afferent_zero"):
        lex.set_lesion(kind)
        p2, rn2, rx2 = _read_all(lex, words)
        res[f"bacc_{kind}"] = _bacc(p2, gt)
        res[f"abstain_{kind}"] = sum(p is None for p in p2.values()) / len(words)
        res[f"decided_bacc_{kind}"] = _decided_bacc(p2, gt)
        res[f"lose_win_{kind}"] = _lose_win(rn2, rx2)
        lex.set_lesion(None)
    lever("learned-edge lesion (FR->category restored to W_INIT)", res["bacc_spiking"], res["bacc_learned_edge"],
          required=True)
    lever("competition lesion (FSI cross-inhibition zeroed)", res["lose_win_intact"] or 0.0,
          res["lose_win_competition"] or 0.0, required=True)

    # S6 integrity: a FULL rebuild from the corpus
    tokens2, vocab2, env2 = _env(corpus)
    lex2 = L.SpikingFrameCategoryLexicon(seed, env2, 1)
    sw2, sl2 = L.seed_curriculum(env2, k_seed=K_SEED, cv_seed=seed)
    lex2.train(sw2, sl2[:, None])
    dw = words[:100]
    h1 = hashlib.sha256(json.dumps([[w, lex.classify(w)] for w in dw]).encode() + lex.W.tobytes()).hexdigest()
    h2 = hashlib.sha256(json.dumps([[w, lex2.classify(w)] for w in dw]).encode() + lex2.W.tobytes()).hexdigest()
    res["deterministic_full_rebuild"] = h1 == h2
    del lex2, env2, tokens2

    noun_pool = sorted(w for w in held if held[w] and w in env.pos)
    nonnoun_pool = sorted(w for w in held if not held[w] and w in env.pos)
    res["organ"] = _organ_arm(seed, lex, noun_pool, nonnoun_pool, n_pairs)
    res["capability"] = capability_arm(seed, env)
    res["elapsed_s"] = round(time.time() - t0, 1)
    return res


def run_null(seed, corpus, pos_gt, n_perm, replicas):
    """S2: replica 0 = true curriculum labels; replicas 1..R-1 = independent label permutations. Every batch retrains
    all replicas from W_INIT. Replica 0 recurs in every batch (its bacc must be identical across batches)."""
    t0 = time.time()
    tokens, vocab, env = _env(corpus)
    held = _held_out(vocab, pos_gt)
    words = sorted(np.random.default_rng(seed + 6).choice(_eval_words(seed, held), N_NULL_EVAL,
                                                          replace=False).tolist())
    gt = {w: held[w] for w in words}
    sw, sl = L.seed_curriculum(env, k_seed=K_SEED, cv_seed=seed)
    lex = L.SpikingFrameCategoryLexicon(seed, env, replicas)
    prng = np.random.default_rng(seed * 7919 + 23)
    null, true_b = [], []
    while len(null) < n_perm:
        labs = [sl] + [prng.permutation(sl) for _ in range(replicas - 1)]
        lex.reset_learning()
        lex.train(sw, np.stack(labs, axis=1))
        preds = [dict() for _ in range(replicas)]
        for w in words:
            d = lex.decide(w)[0]
            for r in range(replicas):
                preds[r][w] = d[r]
        true_b.append(_bacc(preds[0], gt))
        null.extend(_bacc(preds[r], gt) for r in range(1, replicas))
        print(f"[null seed {seed}] {len(null)}/{n_perm} true={true_b[-1]:.3f} ({time.time() - t0:.0f}s)", flush=True)
    null = np.array(null[:n_perm])
    tb = true_b[0]
    pct = 100.0 * (np.sum(null < tb) + 0.5 * np.sum(null == tb)) / len(null)
    return {"seed": seed, "n_perm": int(len(null)), "replicas": replicas, "n_null_eval": len(words),
            "true_bacc": tb, "true_bacc_all_batches_identical": bool(len(set(true_b)) == 1),
            "percentile": float(pct), "null_q50": float(np.quantile(null, 0.5)),
            "null_q95": float(np.quantile(null, 0.95)), "null_q99": float(np.quantile(null, 0.99)),
            "null_max": float(null.max()), "null": null.tolist(), "elapsed_s": round(time.time() - t0, 1)}


def score(src):
    mains, nulls = {}, {}
    report_s8_lesion = None
    for p in sorted(glob.glob(os.path.join(src, "*.json"))):
        if p.endswith(".prov.json") or os.path.basename(p) == "verdict.json":
            continue
        d = json.load(open(p))
        for r in d.get("per_seed", []):
            (nulls if d.get("part") == "null" else mains)[r["seed"]] = r
    g = GATE
    M = [mains[s] for s in SEEDS6 if s in mains]
    N = [nulls[s] for s in SEEDS6 if s in nulls]
    complete = len(M) == 6 and len(N) == 6
    ev, integ = {}, {}
    if M:
        b = [r["bacc_spiking"] for r in M]
        ev["S1_bacc_mean"] = (float(np.mean(b)), np.mean(b) >= g["s1_mean"])
        ev["S1_bacc_min"] = (float(min(b)), min(b) >= g["s1_min"])
        le = [r["bacc_learned_edge"] for r in M]
        ev["S3a_lesion_bacc_mean"] = (float(np.mean(le)), np.mean(le) <= g["s3_les_mean"])
        drops = [r["bacc_spiking"] - r["bacc_learned_edge"] for r in M]
        ev["S3a_drop_min"] = (float(min(drops)), min(drops) >= g["s3_drop"])
        dd = [r["decided_bacc_learned_edge"] for r in M if r["decided_bacc_learned_edge"] is not None]
        ev["S3b_lesion_decided_bacc_mean"] = (float(np.mean(dd)) if dd else None,
                                              (np.mean(dd) <= g["s3_decided_max"]) if dd else True)
        rise = [(r["lose_win_competition"] or 0) - (r["lose_win_intact"] or 0) for r in M]
        ev["S4_loser_ratio_rise_min"] = (float(min(rise)), min(rise) >= g["s4_ratio_rise"])
        sp = [r["spearman_drive_vs_rate"] for r in M if r["spearman_drive_vs_rate"] is not None]
        ev["S5_spearman_mean"] = (float(np.mean(sp)) if sp else None, bool(sp) and np.mean(sp) >= g["s5_spearman"])
        o = [r["organ"] for r in M]
        ev["S7_recover_mean"] = (float(np.mean([x["learned_recover_rate"] for x in o])),
                                 np.mean([x["learned_recover_rate"] for x in o]) >= g["s7_recover"])
        ev["S7_lesion_recover_mean"] = (float(np.mean([x["lesion_recover_rate"] for x in o])),
                                        np.mean([x["lesion_recover_rate"] for x in o]) <= g["s7_lesion_max"])
        ev["S7_fp_in_scope_mean"] = (float(np.mean([x["fp_in_scope_rate"] for x in o])),
                                     np.mean([x["fp_in_scope_rate"] for x in o]) <= g["s7_fp_max"])
        cap_ok = [r["capability"]["intact"]["held_refs"] == ["wolf", "owl"] and r["capability"]["intact"]["held_in_scope"]
                  and r["capability"]["intact"]["held_n_referents"] == 2 for r in M]
        cap_les = [not r["capability"]["learned_edge"]["held_in_scope"] for r in M]
        ev["S8_exercised_seeds"] = (int(sum(cap_ok)), sum(cap_ok) >= g["s8_min_seeds"])
        report_s8_lesion = {"lesion_out_of_scope_seeds": int(sum(cap_les)),
                            "lesion_held_refs": {r["seed"]: r["capability"]["learned_edge"]["held_refs"] for r in M},
                            "intact_held_refs": {r["seed"]: r["capability"]["intact"]["held_refs"] for r in M}}
        integ["S6_full_rebuild_deterministic"] = (all(r["deterministic_full_rebuild"] for r in M),
                                                  all(r["deterministic_full_rebuild"] for r in M))
        az = [r["abstain_afferent_zero"] for r in M]
        integ["afferent_zero_abstain_min"] = (float(min(az)), min(az) >= 1.0)
        hs = [x["hand_in_scope_rate"] for x in o]
        integ["hand_baseline_in_scope_mean"] = (float(np.mean(hs)), np.mean(hs) <= g["s7_hand_max"])
    if N:
        ok = [r["percentile"] >= g["s2_pct"] for r in N]
        ev["S2_null_pct_seeds"] = (int(sum(ok)), sum(ok) >= g["s2_min_seeds"])
        integ["null_true_replica_stable"] = (all(r["true_bacc_all_batches_identical"] for r in N),
                                             all(r["true_bacc_all_batches_identical"] for r in N))
    passed = all(bool(v[1]) for v in ev.values()) and all(bool(v[1]) for v in integ.values())
    verdict = "INCOMPLETE" if not complete else ("GO" if passed else "NO-GO")
    return {"verdict": verdict, "complete_6seed": complete, "gate": GATE,
            "mechanism": "v2: coupled spiking WTA (reciprocal FSI inhibition), graded drive through Hebbian (Oja) "
                         "frame->category synapses, teacher-supervised seed curriculum, host read-out of the winner",
            "seeds_main": sorted(mains), "seeds_null": sorted(nulls),
            "evidence": {k: {"value": v, "pass": bool(p)} for k, (v, p) in ev.items()},
            "integrity_smokes": {k: {"value": v, "pass": bool(p)} for k, (v, p) in integ.items()},
            "per_seed": {s: {k: mains[s].get(k) for k in ("bacc_spiking", "abstain", "bacc_learned_edge",
                                                          "bacc_competition", "lose_win_intact",
                                                          "lose_win_competition", "spearman_drive_vs_rate")}
                         for s in sorted(mains)},
            "null_percentiles": {s: nulls[s]["percentile"] for s in sorted(nulls)},
            "report_only_s8_lesion": report_s8_lesion}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("main", "null"), default="main")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS6)
    ap.add_argument("--corpus", default=os.path.join(_REPO, "data", "corpus", "tinystories.txt"))
    ap.add_argument("--n-pairs", type=int, default=12)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--replicas", type=int, default=41)
    ap.add_argument("--json", default="")
    ap.add_argument("--score", default="")
    a = ap.parse_args()
    if a.score:
        out = score(a.score)
        print(json.dumps(out, indent=1, default=str))
        json.dump(out, open(os.path.join(a.score, "verdict.json"), "w"), indent=1, default=str)
        return
    corpus = a.corpus if os.path.isabs(a.corpus) else os.path.join(_REPO, a.corpus)
    fx = json.load(open(FIXTURE))
    pos_gt = fx["pos"]
    per = []
    for s in a.seeds:
        if a.part == "main":
            r = run_main(s, corpus, pos_gt, a.n_pairs)
            o = r["organ"]
            print(f"[main seed {s}] bacc={r['bacc_spiking']:.3f} abst={r['abstain']:.2f} "
                  f"edge-lesion={r['bacc_learned_edge']:.3f} (decided {r['decided_bacc_learned_edge']}) "
                  f"comp-lesion={r['bacc_competition']:.3f} lose/win {r['lose_win_intact']:.3f}->"
                  f"{r['lose_win_competition']:.3f} spearman={r['spearman_drive_vs_rate']:.3f} "
                  f"det={r['deterministic_full_rebuild']} organ recover={o['learned_recover_rate']:.2f} "
                  f"lesion={o['lesion_recover_rate']:.2f} fp={o['fp_in_scope_rate']:.2f} "
                  f"cap={r['capability']['intact']['held_refs']}/{r['capability']['learned_edge']['held_refs']} "
                  f"({r['elapsed_s']}s)", flush=True)
        else:
            r = run_null(s, corpus, pos_gt, a.n_perm, a.replicas)
            print(f"[null seed {s}] true={r['true_bacc']:.3f} pct={r['percentile']:.1f} q99={r['null_q99']:.3f}",
                  flush=True)
        per.append(r)
    out = {"runner": "_lexicon_spiking_referent_derisk", "part": a.part, "gate": GATE, "per_seed": per}
    if a.json:
        dst = a.json if os.path.isabs(a.json) else os.path.join(_REPO, a.json)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(out, open(dst, "w"), indent=1, default=str)
        print("wrote", dst)


if __name__ == "__main__":
    main()
