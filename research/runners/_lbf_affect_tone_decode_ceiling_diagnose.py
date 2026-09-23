"""PHASE 1 DIAGNOSIS (2026-09-22, affect->tone next-method sub-arc): is the affect->tone open-output
NEGATIVE-ASYMMETRY (research/findings/2026-09-22-affect-tone-open-output-directional-6seed-positive-asymmetric-
NOGO.md) a DECODE-CEILING (negative-valence words carry usable-but-sub-margin probability mass -> a
distribution-shifting coupling could plausibly surface them) or a TRAINING-DATA LIMIT (the linattn mouth's
free-gen distribution places essentially ZERO mass on negative-valence vocabulary on these prompts, so no decode
coupling -- of any shape -- could move it without destroying fluency)?

WHAT THIS MEASURES, on the REAL linattn mouth (no host build, no server, no spiking read-out -- the raw
`LinAttnReadout.logits` forward pass the mouth's own free-gen loop already uses), UNBIASED (no
`_apply_affect_bias`, no fact boost -- exactly the natural next-token distribution the affect coupling would
have to act on):

For each seed in {42,43,44,100,101,102} and each of the SAME 10 free-talk TONE prompts the NO-GO's own harness
used (`research.runners._lbf_affect_tone_open_output_derisk.TONE_PROMPTS`, kept identical so this diagnosis
characterizes the SAME operating point the NO-GO measured, not a different one), a GREEDY (argmax, deterministic,
reproducible -- no RNG draw needed) 110-token continuation is decoded through the production repetition controls
(repetition_penalty=1.3, no_repeat_ngram_size=3, matching `webapp/open_ended_chat.py`'s live call) but WITHOUT
affect bias or fact boost. At every step, the FULL-VOCAB natural softmax `pfull=softmax(lg)` (temperature 1.0,
no top-K cut) is read (never sampled from -- a pure read of what the model itself believes) and:
  - `pos_mass`/`neg_mass`: total probability mass on every in-vocab token matching a positive/negative word in
    the SAME independent, WARRINER-disjoint lexicon the NO-GO's own tone scorer used
    (`_lbf_affect_tone_open_output_derisk.load_indep_lexicon`) -- so this diagnosis and that finding characterize
    the identical vocabulary the NO-GO's ruler already certified as non-circular.
  - whether the single closest negative/positive-lexicon token (by raw logit) falls inside the top-64 candidate
    window the real decode loop actually samples from (`topk=64`, the production default) -- the window ANY
    decode-time coupling (host-logit OR neural-pool) must place a word inside of to have any chance of it being
    selected at all.
  - `margin-to-top1` for that closest token -- how large a gap ANY coupling would need to close for full parity
    with the model's own top pick (mirrors `_apply_affect_bias`'s own `margin(best)` quantity exactly, but over
    the RAW model, not the affect-biased one).

PREREGISTERED DECISION RULE (fixed BEFORE running, so the read is not tuned to a preferred answer -- HARD RULE 2):
  DATA-LIMIT  if  mean(pos_mass)/max(mean(neg_mass), 1e-12) >= 50  AND  frac_steps_neg_in_top64 < 0.10
  DECODE-CEILING  otherwise (negative-valence mass is within ~50x of positive AND/OR regularly reaches the
  top-64 window -- a coupling has plausible material to act on).
These thresholds are round, conservative numbers chosen to separate "structurally near-absent" (worse than 50x
+ essentially never a candidate) from "present but sub-margin" (the DECODE-CEILING's own definition) -- not
fit to this run's own numbers.

NO sim/ edit. Reuses `webapp.wkv_mouth_generator`'s own checkpoint loader / BPE tokenizer / repetition-control
function and `research.runners._wkv_fewspike_read_derisk.LinAttnReadout` (both read-only imports, nothing
touched) plus the NO-GO harness's own lexicon loader and prompt list (so this diagnosis is anchored to the
SAME instrument, not a fresh one). CPU/numpy only; no brain build, no webapp server, no GPU.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

SEEDS = [42, 43, 44, 100, 101, 102]
MAX_NEW_TOKENS = 110          # matches webapp/open_ended_chat.py's live default
TOPK = 64                     # matches generate()'s default candidate window
REPETITION_PENALTY = 1.3      # matches the live wkv_mouth_generator.generate() call in open_ended_chat.py
NO_REPEAT_NGRAM = 3           # matches the same live call

# preregistered decision thresholds (see module docstring) -- fixed before running
_RATIO_THRESHOLD = 50.0
_TOPK_FRAC_THRESHOLD = 0.10


def _strip_bpe_marker(raw: str) -> str:
    return raw[:-4] if raw.endswith("</w>") else raw


def build_lexicon_ids(ro, lex):
    """{token_id: signed_lexicon_value} for every checkpoint vocabulary id whose (BPE-marker-stripped,
    lowercased) word appears in the independent lexicon -- the identical normalization
    `webapp/wkv_mouth_generator.py::_affect_bias_ids` uses for the WARRINER lexicon, applied here to the
    NO-GO's independent lexicon so the diagnosis measures exactly the vocabulary a coupling could ever match
    against (a content word split across multiple BPE subwords is out of scope here exactly as it is for the
    live mechanism -- an honest shared limitation, not something this diagnosis papers over)."""
    pos_ids, neg_ids, val = [], [], {}
    for tid, w in enumerate(ro.words):
        lookup = _strip_bpe_marker(str(w).lower())
        if not lookup or lookup not in lex:
            continue
        v = float(lex[lookup])
        val[tid] = v
        if v > 0:
            pos_ids.append(tid)
        elif v < 0:
            neg_ids.append(tid)
    return np.array(pos_ids, dtype=np.intp), np.array(neg_ids, dtype=np.intp), val


def run_seed(seed, prompts, max_new_tokens=MAX_NEW_TOKENS):
    from webapp import wkv_mouth_generator as _WKVG
    from research.runners._wkv_fewspike_read_derisk import LinAttnReadout
    from research.runners._lbf_affect_tone_open_output_derisk import load_indep_lexicon

    lex, overlap = load_indep_lexicon()
    assert overlap == 0, "independent lexicon must stay disjoint from WARRINER (anti-circularity anti-cheat)"

    ckpt = _WKVG._ckpt_path(seed)
    ro = LinAttnReadout(ckpt)
    pos_ids, neg_ids, _val = build_lexicon_ids(ro, lex)
    bpe = _WKVG._get_bpe_tokenizer()

    per_prompt = []
    for prompt in prompts:
        pid = _WKVG._bpe_encode_prompt(bpe, prompt)
        if not pid:
            pid = [0]
        state = ro.init_state()
        for t in pid:
            state = ro.advance(state, t)
        gen = list(pid)
        step_rows = []
        for _step in range(max_new_tokens):
            lg = ro.logits(state, gen[-1])
            if ro.unk_idx >= 0:
                lg = lg.copy()
                lg[ro.unk_idx] = -1e30
            lg = _WKVG._apply_repetition_controls(lg, gen, REPETITION_PENALTY, NO_REPEAT_NGRAM)
            pfull = _WKVG._softmax(lg)
            pos_mass = float(pfull[pos_ids].sum()) if pos_ids.size else 0.0
            neg_mass = float(pfull[neg_ids].sum()) if neg_ids.size else 0.0
            top1 = float(lg.max())
            cand = np.argpartition(-lg, TOPK - 1)[:TOPK]
            cand_set = set(cand.tolist())
            neg_in_topk = bool(neg_ids.size) and any(int(i) in cand_set for i in neg_ids)
            pos_in_topk = bool(pos_ids.size) and any(int(i) in cand_set for i in pos_ids)
            neg_margin = float(top1 - lg[neg_ids].max()) if neg_ids.size else None
            pos_margin = float(top1 - lg[pos_ids].max()) if pos_ids.size else None
            step_rows.append({
                "pos_mass": pos_mass, "neg_mass": neg_mass,
                "neg_in_top64": neg_in_topk, "pos_in_top64": pos_in_topk,
                "neg_margin_to_top1": neg_margin, "pos_margin_to_top1": pos_margin,
            })
            nxt = int(np.argmax(lg))          # greedy, deterministic advance (no RNG draw)
            gen.append(nxt)
            state = ro.advance(state, nxt)
            if ro.words[nxt] == "endoftext":
                break
        text = bpe.decode([i for i in gen if 0 <= i < len(ro.words) and ro.words[i] != "endoftext"])
        per_prompt.append({"prompt": prompt, "n_steps": len(step_rows), "raw": text, "steps": step_rows})

    return {
        "seed": seed, "ckpt": os.path.relpath(ckpt, _REPO),
        "n_pos_vocab_ids": int(pos_ids.size), "n_neg_vocab_ids": int(neg_ids.size),
        "vocab_size": len(ro.words),
        "per_prompt": per_prompt,
    }


def aggregate(seed_results):
    all_steps = []
    for sr in seed_results:
        for pp in sr["per_prompt"]:
            all_steps.extend(pp["steps"])
    n = len(all_steps)
    mean_pos = sum(s["pos_mass"] for s in all_steps) / n
    mean_neg = sum(s["neg_mass"] for s in all_steps) / n
    median_pos = float(np.median([s["pos_mass"] for s in all_steps]))
    median_neg = float(np.median([s["neg_mass"] for s in all_steps]))
    frac_neg_topk = sum(1 for s in all_steps if s["neg_in_top64"]) / n
    frac_pos_topk = sum(1 for s in all_steps if s["pos_in_top64"]) / n
    neg_margins = [s["neg_margin_to_top1"] for s in all_steps if s["neg_margin_to_top1"] is not None]
    pos_margins = [s["pos_margin_to_top1"] for s in all_steps if s["pos_margin_to_top1"] is not None]
    mean_neg_margin = sum(neg_margins) / len(neg_margins) if neg_margins else None
    mean_pos_margin = sum(pos_margins) / len(pos_margins) if pos_margins else None
    ratio = mean_pos / max(mean_neg, 1e-12)

    diagnosis = "DATA-LIMIT" if (ratio >= _RATIO_THRESHOLD and frac_neg_topk < _TOPK_FRAC_THRESHOLD) else "DECODE-CEILING"

    # per-seed breakdown (for the finding's table)
    per_seed = []
    for sr in seed_results:
        steps = [s for pp in sr["per_prompt"] for s in pp["steps"]]
        n_s = len(steps)
        per_seed.append({
            "seed": sr["seed"],
            "mean_pos_mass": sum(s["pos_mass"] for s in steps) / n_s,
            "mean_neg_mass": sum(s["neg_mass"] for s in steps) / n_s,
            "frac_neg_in_top64": sum(1 for s in steps if s["neg_in_top64"]) / n_s,
            "frac_pos_in_top64": sum(1 for s in steps if s["pos_in_top64"]) / n_s,
            "n_pos_vocab_ids": sr["n_pos_vocab_ids"], "n_neg_vocab_ids": sr["n_neg_vocab_ids"],
        })

    return {
        "n_steps_total": n,
        "mean_pos_mass": mean_pos, "mean_neg_mass": mean_neg,
        "median_pos_mass": median_pos, "median_neg_mass": median_neg,
        "pos_over_neg_ratio": ratio,
        "frac_steps_neg_in_top64": frac_neg_topk, "frac_steps_pos_in_top64": frac_pos_topk,
        "mean_neg_margin_to_top1": mean_neg_margin, "mean_pos_margin_to_top1": mean_pos_margin,
        "decision_rule": "DATA-LIMIT iff pos/neg-mass-ratio>=%.0f AND frac_steps_neg_in_top64<%.2f; else DECODE-CEILING"
                          % (_RATIO_THRESHOLD, _TOPK_FRAC_THRESHOLD),
        "diagnosis": diagnosis,
        "per_seed": per_seed,
    }


def selftest():
    """Pure-logic selftest: the decision rule and lexicon-id builder must be able to FAIL in their failing
    direction (no brain build, no checkpoint load)."""
    ok = True

    def check(name, cond):
        nonlocal ok
        print("  [%s] %s" % ("PASS" if cond else "FAIL", name))
        ok = ok and cond

    class _FakeRO:
        words = ["cat", "happy</w>", "sad</w>", "the</w>", "<UNK>"]

    lex = {"happy": 1.0, "sad": -1.0}
    pos_ids, neg_ids, val = build_lexicon_ids(_FakeRO(), lex)
    check("pos id resolved for 'happy</w>'", list(pos_ids) == [1])
    check("neg id resolved for 'sad</w>'", list(neg_ids) == [2])
    check("val map carries signed value", val[1] == 1.0 and val[2] == -1.0)

    # decision rule: extreme asymmetry + rare top-64 presence -> DATA-LIMIT
    fake_steps_data_limit = [{"pos_mass": 0.01, "neg_mass": 0.0001, "neg_in_top64": False, "pos_in_top64": True,
                              "neg_margin_to_top1": 40.0, "pos_margin_to_top1": 2.0}] * 100
    fake_seed_dl = [{"seed": 42, "n_pos_vocab_ids": 10, "n_neg_vocab_ids": 10,
                     "per_prompt": [{"steps": fake_steps_data_limit}]}]
    agg_dl = aggregate(fake_seed_dl)
    check("extreme asymmetry + rare top64 -> DATA-LIMIT", agg_dl["diagnosis"] == "DATA-LIMIT")

    fake_steps_ceiling = [{"pos_mass": 0.01, "neg_mass": 0.004, "neg_in_top64": True, "pos_in_top64": True,
                           "neg_margin_to_top1": 3.0, "pos_margin_to_top1": 2.0}] * 100
    fake_seed_c = [{"seed": 42, "n_pos_vocab_ids": 10, "n_neg_vocab_ids": 10,
                    "per_prompt": [{"steps": fake_steps_ceiling}]}]
    agg_c = aggregate(fake_seed_c)
    check("comparable mass + frequent top64 -> DECODE-CEILING (must NOT report DATA-LIMIT)",
          agg_c["diagnosis"] == "DECODE-CEILING")

    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=str, default=None, help="comma-separated subset, default all 6")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--out", type=str, default="research/findings/raw/_affect_tone_decode_ceiling_diagnose.json")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    from research.runners._lbf_affect_tone_open_output_derisk import TONE_PROMPTS

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    t0 = time.time()
    seed_results = []
    for s in seeds:
        ts = time.time()
        sr = run_seed(s, TONE_PROMPTS, max_new_tokens=args.max_new_tokens)
        seed_results.append(sr)
        print("[diagnose] seed=%d done (%.1fs) n_pos_vocab=%d n_neg_vocab=%d"
              % (s, time.time() - ts, sr["n_pos_vocab_ids"], sr["n_neg_vocab_ids"]), flush=True)
    agg = aggregate(seed_results)
    out = {
        "runner": "_lbf_affect_tone_decode_ceiling_diagnose",
        "what": "PHASE 1 diagnosis: decode-ceiling vs data-limit for affect->tone negative-asymmetry "
                "(unbiased full-vocab probability mass on independent-lexicon pos/neg vocabulary, real "
                "linattn mouth, greedy decode, production repetition controls, no affect bias/fact boost).",
        "seeds": seeds, "max_new_tokens": args.max_new_tokens, "topk": TOPK,
        "repetition_penalty": REPETITION_PENALTY, "no_repeat_ngram_size": NO_REPEAT_NGRAM,
        "backend": "numpy", "cuda_visible_devices": "",
        "tone_prompts": TONE_PROMPTS,
        "aggregate": agg,
        "per_seed_full": seed_results,
        "wall_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("[diagnose] wrote %s (%.1fs total)" % (args.out, out["wall_seconds"]), flush=True)
    print("[diagnose] DIAGNOSIS = %s  ratio(pos/neg)=%.3g  frac_neg_top64=%.3f  mean_neg_margin=%s mean_pos_margin=%s"
          % (agg["diagnosis"], agg["pos_over_neg_ratio"], agg["frac_steps_neg_in_top64"],
             agg["mean_neg_margin_to_top1"], agg["mean_pos_margin_to_top1"]), flush=True)
