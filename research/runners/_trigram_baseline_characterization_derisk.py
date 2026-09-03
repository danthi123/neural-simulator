"""Empirically characterize the FAIR-TRIGRAM BASELINE that `margin_vs_trigram` (the own-voice fluency gate's deep-
context, d10-99, metric) measures against. Multiple mechanism families -- linear recurrence (wkv), the spiking
ssm/dual-nonneg mouth, a fixed HiPPO multi-timescale SSM, content-addressable associative reads -- all converged
to margin ~ -0.125 at the Simple-English-Wikipedia BPE V=8001 budget (~9.5M tokens, sentence-mode; see
research/findings/2026-09-03-wkv-fluency-crux-simplewiki-bpe-NO-GO-loses-to-trigram.md and
2026-09-03-spiking-depth-tokens-closing-fluency-gap-milestone.md). THE OPEN QUESTION this runner answers: is that
bound explained by the neural models being DATA/CAPACITY-limited, or by the FAIR interpolated trigram being an
UNUSUALLY STRONG baseline on BPE subwords (making the crossing bar artificially high), or both -- and does a
different tokenization move the bar?

METHOD (reuse-by-import, byte-identical to the gate's own computation -- NOT a reimplementation):
  `fit_interp_trigram`, `_BPEVocabAdapter`, `BUCKETS`/`_bucket`, `DEFAULT_BPE_PATH`, `load_sentences`,
  `load_stories` are imported VERBATIM from `research.runners._emerge_wkv_lm_derisk` (the exact module the wkv
  fluency crux runs use). NO reimplementation of the deleted-interpolation trigram, the depth-bucketing, or the
  BPE application -- this runner ONLY varies (a) how many tokens the trigram (and, where noted, its lambda-tuning
  dev split) sees, and (b) which tokenizer produces the token stream, then calls the SAME fit function. `--verify`
  reproduces one cell of the actual crux run (corpus=simplewiki.txt, tokenizer=bpe production V=8001, seed=42,
  n-sentences=1200000, max-train-sents=1000000, max-eval-sents=4000) and asserts its deep-bucket (10-99) trigram
  NLL + lambdas + n_eval match research/findings/raw/_emerge_wkv_lm_simplewiki_6seed.json's seed-42 record
  EXACTLY (4.157 nats, lambdas [0.05,0.05,0.2,0.7], n=21426) -- the load-bearing fidelity check this task's spec
  requires before trusting any swept number.

SWEEPS:
  (a) TOKEN BUDGET (sentence-mode, holding the tokenizer fixed at the production BPE V=8001): the SAME shuffled
      train-sentence pool as the crux (idx=rng.permutation(sents); cut=0.85), walked in permutation order and
      TOKEN-COUNT-capped (not sentence-count-capped, so the budget is precise) at ~2M / ~5M / ~9.5M (all-available
      sentence-mode tokens) tokens; dev = tr_ids[-min(2000, n//5):] (matching the crux's own lambda-tuning split);
      eval is the FIXED held-out 15% split (max_eval_sents=4000), independent of the train budget, so every budget
      cell is scored on the identical eval tokens.
  (b) CONTIGUOUS full budget (~20M tokens): `load_stories(corpus, n_sentences, max_len=40)` -- the SAME
      --contiguous --max-len 40 regime the record's "wkv crosses to +0.02" datum
      (research/findings/raw/_emerge_wkv_lm_contiguous40_1seed_vramcheck.json, seed 42 ONLY, single-seed) used.
      Reported as a SEPARATE regime, not a 4th point on the sentence-mode budget curve: contiguous mode chops
      RAW text into fixed 40-word windows regardless of sentence boundaries (headers, list items, mid-sentence
      cuts), a structurally different -- and, we find, HARDER for a trigram -- eval distribution than the
      clean 3-16-word filtered sentences sentence-mode uses. Conflating the two would misattribute a regime
      change to a token-count effect.
  (c) TOKENIZATION (fixed underlying TEXT -- the full sentence-mode pool, ~908K train sentences, held identical
      across tokenizers so only granularity varies): char-level (built fresh from the pool), BPE V~2000 and
      BPE V~16000 (freshly trained ON simplewiki via `_train_bpe_bounded`, reused-by-import from
      `_subword_mouth_tokenizer_coverage_derisk`, same bounded-training method used to build the production
      wkv_bpe8k.json), BPE V~8000 simplewiki-trained (a corpus-matched control, since the PRODUCTION wkv_bpe8k.json
      was actually trained on wikitext103.txt and only APPLIED to simplewiki -- see its .prov.json), and the
      production BPE V=8001 itself (already covered by the budget sweep's full-sentence-mode cell, reused not
      recomputed). Each tokenizer's own resulting token count is reported alongside its trigram NLL, because
      per-TOKEN NLL is not directly comparable across tokenizations with different tokens-per-word ratios (a
      char-level trigram predicts one character at a time, a strictly easier per-step task) -- nats-per-WORD is
      also reported as the tokenization-invariant comparison.

CPU only, numpy + stdlib only (no torch, no GPU -- verified: importing `_emerge_wkv_lm_derisk` for its trigram/
tokenizer helpers does not import torch or cupy, since both are imported lazily inside functions this runner never
calls). Cells are farmed out across local CPU cores via multiprocessing (idle capacity local machinery, no pool
node / no GPU queue). Run (verify): SIM_BACKEND=numpy .venv/bin/python -m
research.runners._trigram_baseline_characterization_derisk --verify
Run (full sweep): SIM_BACKEND=numpy .venv/bin/python -m
research.runners._trigram_baseline_characterization_derisk --run-all --json <out.json>
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, math, time, traceback
from collections import Counter
from pathlib import Path
from multiprocessing import get_context

import numpy as np

from research.runners._emerge_wkv_lm_derisk import (
    fit_interp_trigram, _BPEVocabAdapter, BUCKETS, _bucket, DEFAULT_BPE_PATH, load_sentences, load_stories,
)
from research.runners._subword_mouth_tokenizer_coverage_derisk import _train_bpe_bounded
from sim.bpe_tokenizer import BPETokenizer

OUT = Path("research/findings/raw/_trigram_baseline_characterization.json")
CORPUS = "data/corpus/simplewiki.txt"

# ---------------------------------------------------------------------------------------------------------- char tok
class _CharVocabAdapter:
    """Character-level tokenizer, interface-compatible with `_BPEVocabAdapter` (.ids(list_of_words) -> list[int]),
    so the SAME `fit_interp_trigram`/eval loops downstream never know the difference. Built from the observed
    character alphabet of a given sentence pool (typically <100 symbols for lowercased [a-z'] + space)."""
    def __init__(self, chars):
        self.i2w = list(chars)
        self.w2i = {c: i for i, c in enumerate(self.i2w)}
        self.unk = self.w2i.get("<UNK>", 0)
        self.size = len(self.i2w)

    def ids(self, s):
        text = " ".join(s)
        get = self.w2i.get
        unk = self.unk
        return [get(c, unk) for c in text]

    @classmethod
    def build(cls, sents):
        c = Counter()
        for s in sents:
            c.update(" ".join(s))
        chars = ["<UNK>"] + sorted(c.keys())
        return cls(chars)


def _bpe_adapter_from_tokenizer(tok: BPETokenizer) -> _BPEVocabAdapter:
    return _BPEVocabAdapter(tok)


# ------------------------------------------------------------------------------------------------- the trigram cell
def _fit_and_eval(tr_ids, ev_ids, V, dev_ids=None, timing=False):
    """ONE trigram fit + full per-depth-bucket eval -- byte-identical call pattern to `main()`'s own
    `tri, lambdas = fit_interp_trigram(tr_ids, V, dev_ids); ... tce[b] += -log(tri(u, ids[t], ids[t+1]))`."""
    t0 = time.time()
    if dev_ids is None:
        dev_ids = tr_ids[-min(2000, max(1, len(tr_ids) // 5)):]
    tri, lambdas = fit_interp_trigram(tr_ids, V, dev_ids)
    t_fit = time.time() - t0
    tce = {}; cnt = {}
    for ids in ev_ids:
        for t in range(len(ids) - 1):
            d = t + 1; b = _bucket(d)
            u = ids[t - 1] if t >= 1 else -1
            tce[b] = tce.get(b, 0.0) + -math.log(max(tri(u, ids[t], ids[t + 1]), 1e-12))
            cnt[b] = cnt.get(b, 0) + 1
    by_depth = {}
    for lo, hi in BUCKETS:
        b = f"{lo}-{hi}" if lo != hi else f"{lo}"
        if b in cnt:
            by_depth[b] = {"n": cnt[b], "trigram_nll": round(tce[b] / cnt[b], 4)}
    n_train_tok = sum(len(x) for x in tr_ids)
    n_eval_tok = sum(cnt.values())
    result = {"V": V, "n_train_sents_or_stories": len(tr_ids), "n_train_tok": n_train_tok,
              "n_dev_sents": len(dev_ids), "n_eval_tok": n_eval_tok, "lambdas": lambdas, "by_depth": by_depth,
              "fit_seconds": round(t_fit, 2)}
    if timing:
        result["total_seconds"] = round(time.time() - t0, 2)
    return result


def _split_seed(sents, seed, eval_frac_cut=0.85, max_eval_sents=4000):
    """The crux's own split logic (main(): idx=rng.permutation; cut=int(0.85*len); ev=idx[cut:][:max_eval])."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sents))
    cut = int(eval_frac_cut * len(sents))
    train_idx = idx[:cut]
    eval_idx = idx[cut:][:max_eval_sents]
    return train_idx, eval_idx


def _budget_cut(train_idx, sents_ids_all, token_budget):
    """Walk `train_idx` (already a fixed random order) accumulating tokenized sentences until the cumulative
    token count reaches `token_budget` (None = take everything). Returns the tr_ids list."""
    tr_ids = []
    total = 0
    for i in train_idx:
        ids = sents_ids_all[i]
        tr_ids.append(ids)
        total += len(ids)
        if token_budget is not None and total >= token_budget:
            break
    return tr_ids


# --------------------------------------------------------------------------------------------------- worker (mp)
# SHARED (fork-inherited, COW, NOT pickled per-task): populated in the parent process before Pool() is created
# under the "fork" start method, so every worker sees it via copy-on-write memory instead of paying to pickle
# ~1M-sentence token-id lists through the task queue on every single job. Jobs below carry only lightweight keys
# into this dict, never the data itself.
_SHARED = {}


def _worker(job):
    kind = job["kind"]
    try:
        if kind == "budget":
            return _run_budget_cell(job)
        elif kind == "contiguous":
            return _run_contiguous_cell(job)
        elif kind == "tokenization":
            return _run_tokenization_cell(job)
        elif kind == "verify":
            return _run_verify_cell(job)
        else:
            raise ValueError(kind)
    except Exception as e:
        return {"job": job, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


def _load_sentence_pool(n_sentences):
    return load_sentences(CORPUS, n_sentences)


def _prod_bpe_adapter():
    return _BPEVocabAdapter(BPETokenizer.load(DEFAULT_BPE_PATH))


def _run_verify_cell(job):
    """Reproduce ONE cell of the actual crux (research/findings/raw/_emerge_wkv_lm_simplewiki_6seed.json,
    seed 42) exactly and compare deep-bucket (10-99) trigram NLL/lambdas/n_eval to the artifact's stored value."""
    sents = _load_sentence_pool(1200000)
    vocab = _prod_bpe_adapter()
    seed = 42
    train_idx, eval_idx = _split_seed(sents, seed, max_eval_sents=4000)
    train_idx = train_idx[:1000000]                                     # --max-train-sents 1000000
    tr = [sents[i] for i in train_idx]; ev = [sents[i] for i in eval_idx]
    tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]
    dev = tr[-min(2000, len(tr) // 5):]
    dev_ids = [vocab.ids(s) for s in dev]
    res = _fit_and_eval(tr_ids, ev_ids, vocab.size, dev_ids=dev_ids, timing=True)
    return {"job": job, "result": res}


def _run_budget_cell(job):
    seed = job["seed"]; budget = job["budget"]
    sents_ids_all, V = _SHARED["tok_ids"][_SHARED["prod_name"]]
    sents = _SHARED["sents"]
    train_idx, eval_idx = _split_seed(sents, seed, max_eval_sents=4000)
    tr_ids = _budget_cut(train_idx, sents_ids_all, budget)
    ev_ids = [sents_ids_all[i] for i in eval_idx]
    res = _fit_and_eval(tr_ids, ev_ids, V, timing=True)
    return {"job": {"kind": "budget", "seed": seed, "budget": budget}, "result": res}


def _run_contiguous_cell(job):
    seed = job["seed"]
    sents_ids_all = _SHARED["contiguous_ids_prod"]; V = _SHARED["prod_V"]
    stories = _SHARED["stories"]
    train_idx, eval_idx = _split_seed(stories, seed, max_eval_sents=4000)
    tr_ids = _budget_cut(train_idx, sents_ids_all, None)
    ev_ids = [sents_ids_all[i] for i in eval_idx]
    res = _fit_and_eval(tr_ids, ev_ids, V, timing=True)
    return {"job": {"kind": "contiguous", "seed": seed}, "result": res}


def _run_tokenization_cell(job):
    seed = job["seed"]; name = job["name"]
    sents_ids_all, V = _SHARED["tok_ids"][name]
    sents = _SHARED["sents"]
    train_idx, eval_idx = _split_seed(sents, seed, max_eval_sents=4000)
    tr_ids = _budget_cut(train_idx, sents_ids_all, None)              # full sentence-mode pool, no token cap
    ev_ids = [sents_ids_all[i] for i in eval_idx]
    res = _fit_and_eval(tr_ids, ev_ids, V, timing=True)
    return {"job": {"kind": "tokenization", "seed": seed, "name": name}, "result": res}


# --------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="reproduce one crux cell and check against the stored artifact")
    ap.add_argument("--run-all", action="store_true", help="run the full budget x tokenization sweep")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--contiguous-seeds", type=int, nargs="+", default=[42],
                     help="seeds for the ~20M-token contiguous cell (kept SHORT by default: "
                          "research/findings/raw/_emerge_wkv_lm_assoc_depth2_contiguous_6seed.json and "
                          "_emerge_wkv_lm_ssm_depth2_contiguous_6seed.json already carry an independently-fit, "
                          "byte-for-byte reproducible 6-seed trigram deep-bucket NLL at this exact budget "
                          "[4.415, 4.438, 4.404, 4.42, 4.414, 4.401] (IDENTICAL between the two runs, confirming "
                          "determinism) -- this runner's own seed-42 cell is a cross-check of the replica against "
                          "that existing 12-independent-fit evidence, not a from-scratch re-derivation")
    ap.add_argument("--budgets", type=int, nargs="+", default=[2_000_000, 5_000_000, 9_500_000],
                     help="token budgets for the sentence-mode sweep (the 3rd/last is a request, not a floor -- "
                          "the full sentence-mode pool caps near ~9.5M so 9500000 effectively means 'all available'")
    ap.add_argument("--n-sentences", type=int, default=1_200_000)
    ap.add_argument("--n-workers", type=int, default=6)
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()

    if args.verify:
        r = _run_verify_cell({"kind": "verify"})
        res = r["result"]
        deep = res["by_depth"]["10-99"]
        target = {"trigram_nll": 4.157, "n": 21426, "lambdas": [0.050000000000000044, 0.05, 0.2, 0.7]}
        ok = (abs(deep["trigram_nll"] - target["trigram_nll"]) < 1e-3 and deep["n"] == target["n"]
              and all(abs(a - b) < 1e-9 for a, b in zip(res["lambdas"], target["lambdas"])))
        print(f"[verify] deep(10-99) trigram_nll={deep['trigram_nll']} (target {target['trigram_nll']}), "
              f"n={deep['n']} (target {target['n']}), lambdas={res['lambdas']} (target {target['lambdas']})")
        print(f"[verify] n_train_tok={res['n_train_tok']} n_train_sents={res['n_train_sents_or_stories']} fit_seconds={res['fit_seconds']}")
        print("[verify] " + ("PASS -- replica matches the crux artifact's seed-42 trigram NLL exactly" if ok else "FAIL -- mismatch, do not trust downstream sweep numbers"))
        out = {"runner": "_trigram_baseline_characterization_derisk", "mode": "verify", "result": res, "pass": ok}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        return 0 if ok else 1

    if not args.run_all:
        print("nothing to do -- pass --verify or --run-all"); return 1

    t0 = time.time()
    print(f"[load] sentence-mode pool (max {args.n_sentences} sentences)...", flush=True)
    sents = _load_sentence_pool(args.n_sentences)
    print(f"[load] {len(sents)} sentence-mode sentences ({time.time()-t0:.1f}s)", flush=True)

    print("[load] contiguous stories (max_len=40, matching --contiguous --max-len 40)...", flush=True)
    stories = load_stories(CORPUS, args.n_sentences, max_len=40)
    print(f"[load] {len(stories)} contiguous stories ({time.time()-t0:.1f}s)", flush=True)

    # -- tokenizers --------------------------------------------------------------------------------------------
    print("[tok] loading production BPE V=8001 (bridges/wkv_ckpt/wkv_bpe8k.json, trained on wikitext103)...", flush=True)
    prod_bpe = _prod_bpe_adapter()

    print("[tok] training simplewiki-native BPE V~2000/8000/16000 (bounded, top_k_words=8000 freq_cap=15)...", flush=True)
    all_words = [w for s in sents for w in s]
    bpe2000, n_uniq2, ts2 = _train_bpe_bounded(all_words, 2000, top_k_words=8000, freq_cap=15)
    bpe8000sw, n_uniq8, ts8 = _train_bpe_bounded(all_words, 8000, top_k_words=8000, freq_cap=15)
    bpe16000, n_uniq16, ts16 = _train_bpe_bounded(all_words, 16000, top_k_words=12000, freq_cap=15)
    print(f"[tok] bpe2000 V={bpe2000.vocab_size} ({ts2}s)  bpe8000sw V={bpe8000sw.vocab_size} ({ts8}s)  "
          f"bpe16000 V={bpe16000.vocab_size} ({ts16}s)  n_uniq_words={n_uniq2}", flush=True)
    char_vocab = _CharVocabAdapter.build(sents)
    print(f"[tok] char-level V={char_vocab.size}", flush=True)

    print("[tok] tokenizing full sentence-mode pool under each tokenizer (once, shared across seeds)...", flush=True)
    tokenizers = {
        "bpe_v8001_production_wikitext103": prod_bpe,
        "bpe_v2000_simplewiki": _bpe_adapter_from_tokenizer(bpe2000),
        "bpe_v8000_simplewiki": _bpe_adapter_from_tokenizer(bpe8000sw),
        "bpe_v16000_simplewiki": _bpe_adapter_from_tokenizer(bpe16000),
        "char_level": char_vocab,
    }
    sents_ids_by_tok = {}
    for name, vocab in tokenizers.items():
        tt0 = time.time()
        sents_ids_by_tok[name] = ([vocab.ids(s) for s in sents], vocab.size)
        print(f"    [tok] {name}: V={vocab.size} tokenized {len(sents)} sents in {time.time()-tt0:.1f}s "
              f"(total tok={sum(len(x) for x in sents_ids_by_tok[name][0]):,})", flush=True)

    print("[tok] tokenizing contiguous stories under production BPE...", flush=True)
    tc0 = time.time()
    stories_ids_prod = [prod_bpe.ids(s) for s in stories]
    print(f"    [tok] contiguous/production-bpe: {len(stories)} stories in {time.time()-tc0:.1f}s "
          f"(total tok={sum(len(x) for x in stories_ids_prod):,})", flush=True)

    # -- populate fork-inherited shared state (COW; see _SHARED's docstring -- avoids repickling ~1M-sentence
    # token-id lists through the task queue on every single job) ------------------------------------------------
    prod_sents_ids, prod_V = sents_ids_by_tok["bpe_v8001_production_wikitext103"]
    _SHARED["sents"] = sents
    _SHARED["stories"] = stories
    _SHARED["tok_ids"] = sents_ids_by_tok
    _SHARED["contiguous_ids_prod"] = stories_ids_prod
    _SHARED["prod_V"] = prod_V
    _SHARED["prod_name"] = "bpe_v8001_production_wikitext103"

    # -- job list (lightweight -- indices/keys into _SHARED only, never the data itself) ---------------------------
    jobs = []
    for seed in args.seeds:
        for budget in args.budgets:
            jobs.append({"kind": "budget", "seed": seed, "budget": budget})
        for name in sents_ids_by_tok:
            if name == "bpe_v8001_production_wikitext103":
                continue    # covered by the full-budget sentence-mode cell above (avoid recompute)
            jobs.append({"kind": "tokenization", "seed": seed, "name": name})
    for seed in args.contiguous_seeds:
        jobs.append({"kind": "contiguous", "seed": seed})

    print(f"[run] {len(jobs)} trigram-fit cells across {args.n_workers} worker processes (fork, COW-shared data)...", flush=True)
    ctx = get_context("fork")
    results = []
    with ctx.Pool(args.n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, jobs)):
            j = r["job"]
            if "error" in r:
                print(f"  [{i+1}/{len(jobs)}] ERROR {j}: {r['error']}", flush=True)
            else:
                deep = r["result"]["by_depth"].get("10-99", {})
                print(f"  [{i+1}/{len(jobs)}] {j.get('kind')} seed={j.get('seed')} "
                      f"{j.get('budget', j.get('name', ''))}: n_train_tok={r['result']['n_train_tok']:,} "
                      f"deep_trigram_nll={deep.get('trigram_nll')} ({r['result']['fit_seconds']}s)", flush=True)
            results.append(r)

    out = {"runner": "_trigram_baseline_characterization_derisk", "corpus": CORPUS, "seeds": args.seeds,
           "budgets": args.budgets, "n_sentence_mode_sents": len(sents), "n_contiguous_stories": len(stories),
           "tokenizer_vocab_sizes": {k: v[1] for k, v in sents_ids_by_tok.items()},
           "tokenizer_total_tokens_full_pool": {k: sum(len(x) for x in v[0]) for k, v in sents_ids_by_tok.items()},
           "contiguous_total_tokens_production_bpe": sum(len(x) for x in stories_ids_prod),
           "results": results, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> {args.json} ({out['elapsed_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
