"""One-brain-wiring de-risk #1 (research/findings/2026-09-03-one-brain-mouth-integration-ROADMAP.md).

THE QUESTION. `webapp/wkv_mouth_generator.py::scope_mode()` (`BRAIN_WKV_MOUTH_SCOPE=broad`, the mode the live
6/6-trigram-crossing verification actually ran under -- see `research/findings/2026-09-03-OPEN-FLUENCY-
BREAKTHROUGH-linattn-deployable-spiking-mouth-beats-trigram-6of6.md`) currently makes `in_vocab_scope` ADMIT
EVERY PROMPT unconditionally -- its own docstring names this a placeholder ("ADMITS every prompt to this gate
... rather than fabricating an ungrounded number") and the module's block comment says the real threshold
should be "set from the 6-seed's own held-out coverage, NOT guessed here." This runner is that measurement:
does the deployed `--recurrence linattn` BPE checkpoint (`bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_
seed{seed}.npz` x6 seeds + `bridges/wkv_ckpt/wkv_bpe8k.json`) actually cover realistic CONVERSATIONAL prompts,
and if not fully, what coverage cutoff should gate SCOPE=broad instead of admitting everything?

THE PROBE CORPUS -- REUSED VERBATIM, nothing invented. `_wkv_mouth_chat_topic_vocab_coverage_derisk.
_build_probe_corpus` (the SAME 124-utterance realistic-chat probe an earlier rung already built + used for the
CLOSED-VOCAB V=1000 checkpoint: 14 Turing-test conversational-register turns + 10 "Tell me about <famous
everyday topic>" queries + 100 seeded "Tell me about <live wikidata_core_15k agent>" queries). Reusing it here
(against the DIFFERENT, BPE, linattn checkpoint) makes this measurement directly comparable to that earlier
finding's numbers, not a fresh ad hoc sample.

TWO DISTINCT TOKENIZATIONS, MEASURED SEPARATELY (the reason this runner exists rather than reusing
`_subword_mouth_tokenizer_coverage_derisk` verbatim: that runner's own `_bpe_row` LOWERCASES + WORD_RE-extracts
before calling `tok.encode(w)` -- a cleaner input than what production code actually feeds the tokenizer):
  1. `asfed` -- EXACTLY what `webapp.wkv_mouth_generator._free_gen`/`_free_gen_linattn` do to a raw prompt:
     `bpe.encode(prompt)` with NO lowercasing (`sim.bpe_tokenizer.BPETokenizer.encode` splits on `text.split()`,
     case and punctuation both flow through unmodified). This is the PRODUCTION-FAITHFUL number.
  2. `content_word` -- the earlier findings' methodology (lowercased, `WKV._WORD_RE`-extracted, minus
     `WKV._FUNCTION_WORDS`), each content word BPE-encoded independently. This isolates genuine TOPIC/
     VOCABULARY coverage from the case/punctuation artifact `asfed` also carries.

HONEST HEADLINE FINDING (verified by hand before writing this runner, see the finding doc): because
`BPETokenizer`'s merge table was trained EXCLUSIVELY on lowercase `[a-z']+` words (`_train_bpe_bounded`'s
`raw.lower()` + `WORD_RE.findall`), an uppercase letter is a CHARACTER OUTSIDE THE TRAINED ALPHABET -> it maps
to the tokenizer's own `<UNK>` id 0, one per capital letter. Real conversational English capitalizes every
sentence-initial word and every proper noun, so the `asfed` numbers below are dominated by this MECHANICAL
case-folding gap, not by genuine topic mismatch -- e.g. "Tell me about Ac Le Havre." asfed-encodes with 5/13
UNK tokens (one per capitalized letter), while the SAME words lowercased are fully clean. This is a SEPARATE,
narrow, cheaply-fixable bug (`bpe.encode(prompt.lower())` at the two `_free_gen*` call sites -- not applied
here, this is a measurement-only runner) from the genuine vocabulary/topic coverage question the SCOPE=broad
threshold is actually meant to answer -- which is why both numbers are reported, not conflated.

TEACHER-FORCED PERPLEXITY (a genuine held-out CONFIDENCE signal, not a coverage proxy): for each probe
utterance's `asfed` token ids, score `-log P(tok_t | tok_<t>)` under the ACTUAL checkpoint's own next-token
distribution via `LinAttnReadout.advance`/`.logits` (pure matrix ops -- NOT the few-spike WTA sampler; this
measures the MODEL's own confidence in the prompt text, independent of the read-out mechanism under test
elsewhere). Computed per SEED (all 6 non-negotiable seeds: 42,43,44,100,101,102) since it depends on the
trained weights; the coverage/OOV numbers above are tokenizer-only and seed-INVARIANT by construction (verified
below: every seed's checkpoint reports the identical V=8001 vocab).

THE RECOMMENDATION this runner's output is FOR: a per-prompt coverage score (`content_word.whole_word_frac` --
fraction of content words BPE-encoding to exactly one known piece, i.e. words the tokenizer's OWN frequency-
based merge table judged common enough to deserve a dedicated symbol) with a swept cutoff, cross-referenced
against teacher-forced perplexity at each cutoff, to name a principled `BRAIN_WKV_MOUTH_SCOPE=broad` threshold
in place of today's "admit everything" placeholder.

MEMORY / COST DISCIPLINE. `LinAttnReadout.__init__` is pure `np.load` (~16 MB/seed) + numpy arrays -- no
`SimulationBridge`, no GPU, no torch, no RNG draws (matches this file's whole read-only-measurement class).
CPU/numpy only, well under a 4 GB RSS budget for 6 x ~16 MB checkpoints + a 124-utterance probe.

CHECKOUT-LOCATION NOTE (matches `_subword_mouth_tokenizer_coverage_derisk._resolve_corpus`'s own precedent):
the `bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed*.npz` files are gitignored multi-MB binaries produced
by a GPU training run and live only in the primary checkout's working tree -- a git worktree does not carry
them (verified 2026-09-03: present under `/home/dant123/Projects/sim/bridges/wkv_ckpt/`, absent from this
runner's own worktree checkout). `_resolve_ckpt` below applies the SAME worktree-first-then-shared-checkout
fallback that runner already established, so this module runs unmodified from either location.

Run (the real 6-seed measurement):
    SIM_BACKEND=numpy PYTHONPATH=. .venv/bin/python -m research.runners._wkv_mouth_linattn_broad_scope_coverage_derisk \\
        --out research/findings/raw/_wkv_mouth_linattn_broad_scope_coverage.json
Run (fast single-seed smoke):
    SIM_BACKEND=numpy PYTHONPATH=. .venv/bin/python -m research.runners._wkv_mouth_linattn_broad_scope_coverage_derisk \\
        --seeds 42 --out /tmp/smoke.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from webapp import wkv_mouth_generator as WKV  # noqa: E402
from research.runners._wkv_fewspike_read_derisk import LinAttnReadout  # noqa: E402
from research.runners._wkv_mouth_chat_topic_vocab_coverage_derisk import _build_probe_corpus  # noqa: E402

WORD_RE = WKV._WORD_RE
FUNC = WKV._FUNCTION_WORDS

# See the module docstring's CHECKOUT-LOCATION NOTE. Same precedent as
# `_subword_mouth_tokenizer_coverage_derisk._resolve_corpus`.
_SHARED_CHECKOUT = Path("/home/dant123/Projects/sim")

DEFAULT_SEEDS = (42, 43, 44, 100, 101, 102)
DEFAULT_CUTOFFS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0)


def _resolve_ckpt(rel_path: str) -> str:
    """`rel_path` MUST be relative (to REPO_ROOT) -- an absolute worktree path would make the shared-checkout
    fallback below a no-op (`Path(shared) / <absolute>` returns the absolute operand unchanged, pathlib's own
    join semantics), silently defeating the fallback this function exists to provide."""
    here = REPO_ROOT / rel_path
    if here.exists():
        return str(here)
    alt = _SHARED_CHECKOUT / rel_path
    if alt.exists():
        return str(alt)
    raise FileNotFoundError(
        f"linattn checkpoint not found in this checkout ({here}) or the shared checkout ({alt}) -- "
        "see the module docstring's CHECKOUT-LOCATION NOTE."
    )


def _load_indomain_reference_sentences(max_bytes: int = 3_000_000, n_sentences: int = 20) -> list:
    """A fixed, small, genuine-simplewiki sentence sample (the checkpoint's OWN training corpus, already
    lowercased in the raw file -- see the module docstring's finding that the corpus itself is pre-lowercased,
    which is WHY the checkpoint never learned capital letters at all) -- a baseline reference so a raw
    perplexity number on the probe corpus has something to be compared against. NOT a claim about the exact
    train/held-out split the training run used (`_emerge_wkv_lm_derisk --max-train-sents/--max-eval-sents`
    controls that internally); this is illustrative context for how confident this small model is on ITS OWN
    register, not a precise held-out-loss reproduction."""
    rel = "data/corpus/simplewiki.txt"
    here = REPO_ROOT / rel
    path = here if here.exists() else (_SHARED_CHECKOUT / rel)
    if not path.exists():
        return []
    import re as _re
    raw = open(path, encoding="utf-8", errors="ignore").read(max_bytes)
    lines = raw.split("\n")
    sents = []
    for ln in lines:
        for s in _re.split(r"(?<=[.!?]) ", ln):
            s = s.strip()
            if 40 < len(s) < 120 and " " in s:
                sents.append(s)
    # skip the first slice (title-adjacent short lines cluster early) and take a representative run
    return sents[200:200 + n_sentences]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ── tokenization (two views, see module docstring) ──────────────────────────────────────────────────────────
def _asfed_ids(bpe, text: str) -> list:
    """EXACTLY `webapp.wkv_mouth_generator._free_gen`/`_free_gen_linattn`'s prompt encode: `bpe.encode(prompt)`,
    no lowercasing, no punctuation stripping."""
    return bpe.encode(text or "")


def _lowercased_ids(bpe, text: str) -> list:
    """A cheap COUNTERFACTUAL: what `asfed` would look like if the one-line case-fold fix named in the module
    docstring were applied (`bpe.encode(prompt.lower())`) -- isolates the case-folding bug's OWN contribution
    to UNK/perplexity from genuine vocabulary/topic coverage (punctuation still flows through unmodified, so a
    trailing '.'/'!'/'?' still produces one UNK here -- this view removes ONLY the capitalization artifact)."""
    return bpe.encode((text or "").lower())


def _content_words(text: str) -> list:
    toks = [w.lower() for w in WORD_RE.findall(text)]
    return [t for t in toks if t not in FUNC]


def _content_word_pieces(bpe, text: str) -> dict:
    cw = _content_words(text)
    per_word = []
    n_pieces = 0
    n_hard_oov = 0
    n_whole = 0
    for w in cw:
        ids = bpe.encode(w)
        pieces = len(ids)
        is_unk = any(i == 0 for i in ids)
        per_word.append({"word": w, "pieces": pieces, "hard_oov": is_unk})
        n_pieces += pieces
        n_hard_oov += int(is_unk)
        n_whole += int(pieces == 1 and not is_unk)
    n_cw = len(cw)
    return {
        "n_content_words": n_cw,
        "mean_pieces_per_content_word": round(n_pieces / n_cw, 4) if n_cw else None,
        "hard_oov_content_words": n_hard_oov,
        "hard_oov_rate": round(n_hard_oov / n_cw, 4) if n_cw else None,
        "whole_word_content_words": n_whole,
        "whole_word_frac": round(n_whole / n_cw, 4) if n_cw else None,  # PRIMARY per-prompt coverage score
        "per_word": per_word,
    }


# ── teacher-forced confidence (pure matrix ops, no spiking WTA -- see module docstring) ─────────────────────
def _teacher_forced_nll_linattn(ro: LinAttnReadout, ids: list) -> dict:
    if len(ids) < 2:
        return {"n_scored": 0, "mean_nll": None, "ppl": None}
    state = ro.init_state()
    prev = ids[0]
    state = ro.advance(state, prev)
    total_nll = 0.0
    n = 0
    for tid in ids[1:]:
        lg = ro.logits(state, prev)   # `tid` arg is accepted-and-ignored by LinAttnReadout.logits (see its own
        p = _softmax(lg)              # docstring) -- kept for call-shape parity only; `state["hh"]` is what's read.
        total_nll += -math.log(max(float(p[tid]), 1e-12))
        n += 1
        state = ro.advance(state, tid)
        prev = tid
    mean_nll = total_nll / n
    return {"n_scored": n, "mean_nll": round(mean_nll, 4), "ppl": round(math.exp(min(mean_nll, 30.0)), 3)}


def _analyze_utterance(bpe, text: str) -> dict:
    ids = _asfed_ids(bpe, text)
    n = len(ids)
    n_unk = sum(1 for i in ids if i == 0)
    lc_ids = _lowercased_ids(bpe, text)
    n_lc = len(lc_ids)
    n_lc_unk = sum(1 for i in lc_ids if i == 0)
    return {
        "text": text,
        "asfed_ids": ids,
        "asfed_n_tokens": n,
        "asfed_n_unk": n_unk,
        "asfed_unk_rate": round(n_unk / n, 4) if n else None,
        "lowercased_ids": lc_ids,
        "lowercased_n_tokens": n_lc,
        "lowercased_n_unk": n_lc_unk,
        "lowercased_unk_rate": round(n_lc_unk / n_lc, 4) if n_lc else None,
        "content_word": _content_word_pieces(bpe, text),
    }


def _aggregate(rows: list, cutoffs) -> dict:
    n = len(rows)
    if n == 0:
        return {"n_utterances": 0, "note": "UNDEFINED (empty group) -- not a 0% score"}
    asfed_tokens = sum(r["asfed_n_tokens"] for r in rows)
    asfed_unk = sum(r["asfed_n_unk"] for r in rows)
    lc_tokens = sum(r["lowercased_n_tokens"] for r in rows)
    lc_unk = sum(r["lowercased_n_unk"] for r in rows)
    cw = [r["content_word"] for r in rows]
    tot_cw = sum(c["n_content_words"] for c in cw)
    tot_pieces = sum(round(c["mean_pieces_per_content_word"] * c["n_content_words"]) for c in cw if c["n_content_words"])
    tot_hard_oov = sum(c["hard_oov_content_words"] for c in cw)
    tot_whole = sum(c["whole_word_content_words"] for c in cw)
    whole_fracs = [c["whole_word_frac"] for c in cw if c["whole_word_frac"] is not None]
    serve_at_cutoff = {}
    for tau in cutoffs:
        n_serve = sum(1 for c in cw if (c["whole_word_frac"] is not None and c["whole_word_frac"] >= tau))
        serve_at_cutoff[f"whole_word_frac>={tau}"] = {"n_serve": n_serve, "frac_serve": round(n_serve / n, 4)}
    return {
        "n_utterances": n,
        "asfed_token_unk_rate": round(asfed_unk / asfed_tokens, 4) if asfed_tokens else None,
        "asfed_token_coverage_pct": round(100.0 * (1 - asfed_unk / asfed_tokens), 2) if asfed_tokens else None,
        "lowercased_token_unk_rate": round(lc_unk / lc_tokens, 4) if lc_tokens else None,
        "lowercased_token_coverage_pct": round(100.0 * (1 - lc_unk / lc_tokens), 2) if lc_tokens else None,
        "content_word_hard_oov_rate": round(tot_hard_oov / tot_cw, 4) if tot_cw else None,
        "content_word_coverage_pct": round(100.0 * (1 - tot_hard_oov / tot_cw), 2) if tot_cw else None,
        "content_word_whole_word_pct": round(100.0 * tot_whole / tot_cw, 2) if tot_cw else None,
        "mean_pieces_per_content_word": round(tot_pieces / tot_cw, 3) if tot_cw else None,
        "mean_whole_word_frac_per_prompt": round(sum(whole_fracs) / len(whole_fracs), 4) if whole_fracs else None,
        "serve_fraction_at_cutoff": serve_at_cutoff,
    }


def main(seeds=DEFAULT_SEEDS, n_wikidata: int = 100, out: str | None = None,
         ckpt_template: str | None = None, bpe_path: str | None = None, cutoffs=DEFAULT_CUTOFFS) -> dict:
    ckpt_tmpl = ckpt_template or "bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{seed}.npz"

    bpe = WKV._get_bpe_tokenizer(bpe_path)
    out_data: dict = {
        "runner": "_wkv_mouth_linattn_broad_scope_coverage_derisk",
        "seeds": list(seeds),
        "bpe_path": bpe_path or WKV._DEFAULT_BPE_PATH,
        "bpe_vocab_size": bpe.vocab_size,
        "scope_mode_today": "broad admits every prompt unconditionally (webapp.wkv_mouth_generator.scope_mode "
                             "placeholder) -- this measurement is what that placeholder's own docstring named "
                             "as the missing input.",
    }

    # ── coverage/OOV: TOKENIZER-ONLY, seed-invariant by construction -- computed ONCE ────────────────────────
    groups = _build_probe_corpus(42, n_wikidata)   # seed=42 only controls WHICH 100 wikidata agents are sampled
    out_data["n_probe_utterances"] = sum(len(v) for v in groups.values())
    rows_by_group = {}
    for gname, items in groups.items():
        rows_by_group[gname] = [_analyze_utterance(bpe, it["text"]) for it in items]
    all_rows = [r for rows in rows_by_group.values() for r in rows]

    out_data["coverage"] = {
        "per_group": {g: _aggregate(rows, cutoffs) for g, rows in rows_by_group.items()},
        "overall": _aggregate(all_rows, cutoffs),
    }
    out_data["cutoffs_tested"] = list(cutoffs)

    # ── in-domain reference (a fixed genuine-simplewiki sample, lowercase, matching training preprocessing) --
    # a BASELINE this tiny (d_model=192, depth-2) model's OWN confidence ceiling, so a raw ppl number on the
    # probe corpus has something to be compared against (an 8001-way softmax's "uniform guessing" ceiling is
    # ppl==8001; this reference is what the model actually achieves on text drawn from its OWN training
    # distribution) ───────────────────────────────────────────────────────────────────────────────────────────
    indomain_sents = _load_indomain_reference_sentences()
    out_data["indomain_reference"] = {"n_sentences": len(indomain_sents), "sentences_sample": indomain_sents[:3]}

    # ── teacher-forced perplexity: PER SEED (depends on trained weights). Scored on THREE token views per
    # prompt: asfed (production-real), lowercased (the case-fold-fix counterfactual), and the fixed in-domain
    # reference sample (the baseline) -- see module docstring ───────────────────────────────────────────────
    out_data["teacher_forced_by_seed"] = {}
    vocab_sizes_seen = set()
    per_prompt_ppl_by_seed = {}   # seed -> list of per-prompt dicts (asfed_ppl, lowercased_ppl, whole_word_frac)
    indomain_ppl_by_seed = {}
    for seed in seeds:
        ckpt_path = _resolve_ckpt(ckpt_tmpl.format(seed=seed))
        ro = LinAttnReadout(ckpt_path)
        vocab_sizes_seen.add(ro.V)
        seed_rows = []
        for gname, rows in rows_by_group.items():
            for r in rows:
                tf_asfed = _teacher_forced_nll_linattn(ro, r["asfed_ids"])
                tf_lc = _teacher_forced_nll_linattn(ro, r["lowercased_ids"])
                seed_rows.append({
                    "group": gname, "text": r["text"],
                    "asfed_ppl": tf_asfed["ppl"], "lowercased_ppl": tf_lc["ppl"],
                    "whole_word_frac": r["content_word"]["whole_word_frac"],
                })
        indomain_ppls = [ppl for s in indomain_sents
                         if (ppl := _teacher_forced_nll_linattn(ro, _asfed_ids(bpe, s))["ppl"]) is not None]
        asfed_ppls = [r["asfed_ppl"] for r in seed_rows if r["asfed_ppl"] is not None]
        lc_ppls = [r["lowercased_ppl"] for r in seed_rows if r["lowercased_ppl"] is not None]
        out_data["teacher_forced_by_seed"][str(seed)] = {
            "checkpoint_path": ckpt_path, "checkpoint_V": ro.V, "checkpoint_d_model": ro.D,
            "checkpoint_unk_idx_detected": ro.unk_idx,
            "mean_asfed_ppl": round(sum(asfed_ppls) / len(asfed_ppls), 2) if asfed_ppls else None,
            "median_asfed_ppl": round(sorted(asfed_ppls)[len(asfed_ppls) // 2], 2) if asfed_ppls else None,
            "mean_lowercased_ppl": round(sum(lc_ppls) / len(lc_ppls), 2) if lc_ppls else None,
            "median_lowercased_ppl": round(sorted(lc_ppls)[len(lc_ppls) // 2], 2) if lc_ppls else None,
            "mean_indomain_reference_ppl": round(sum(indomain_ppls) / len(indomain_ppls), 2) if indomain_ppls else None,
        }
        per_prompt_ppl_by_seed[seed] = seed_rows
        indomain_ppl_by_seed[seed] = indomain_ppls

    out_data["vocab_size_seed_invariant"] = (len(vocab_sizes_seen) == 1)
    out_data["vocab_sizes_seen"] = sorted(vocab_sizes_seen)

    # ── cross-seed case-fold impact summary: matched-pair asfed vs lowercased ppl, mean across all 6 seeds ────
    n_prompts = len(all_rows)

    def _cross_seed_mean(key):
        return [sum(per_prompt_ppl_by_seed[seed][i][key] for seed in seeds) / len(seeds) for i in range(n_prompts)]

    mean_asfed_ppl_per_prompt = _cross_seed_mean("asfed_ppl")
    mean_lc_ppl_per_prompt = _cross_seed_mean("lowercased_ppl")
    whole_word_frac_per_prompt = [r["content_word"]["whole_word_frac"] for r in all_rows]
    indomain_all = [p for seed in seeds for p in indomain_ppl_by_seed[seed]]

    out_data["case_fold_impact"] = {
        "mean_asfed_ppl_across_probe_and_seeds": round(sum(mean_asfed_ppl_per_prompt) / n_prompts, 2),
        "mean_lowercased_ppl_across_probe_and_seeds": round(sum(mean_lc_ppl_per_prompt) / n_prompts, 2),
        "mean_indomain_reference_ppl_across_seeds": round(sum(indomain_all) / len(indomain_all), 2) if indomain_all else None,
        "interpretation": "asfed >> lowercased implies the capital-letter/punctuation UNK artifact (module "
                           "docstring) -- not genuine topic mismatch -- is the dominant driver of asfed's poor "
                           "score; lowercased vs indomain_reference isolates the REMAINING register/vocabulary "
                           "gap once that artifact is removed.",
    }

    # ── coverage (whole_word_frac) vs LOWERCASED ppl (the case-artifact-free correlation -- the evidence base
    # for the recommended cutoff; bucketing against asfed ppl would conflate the case bug with genuine
    # vocabulary coverage) ──────────────────────────────────────────────────────────────────────────────────
    buckets = [(1.0, 1.0), (0.9, 0.999), (0.7, 0.9), (0.5, 0.7), (0.0, 0.5)]
    bucket_stats = []
    for lo, hi in buckets:
        idxs = [i for i, f in enumerate(whole_word_frac_per_prompt)
                if f is not None and (lo <= f <= hi if hi == 1.0 else lo <= f < hi)]
        if not idxs:
            bucket_stats.append({"whole_word_frac_range": f"[{lo},{hi}]", "n": 0, "mean_lowercased_ppl": None})
            continue
        vals = sorted(mean_lc_ppl_per_prompt[i] for i in idxs)
        bucket_stats.append({
            "whole_word_frac_range": f"[{lo},{hi}]",
            "n": len(idxs),
            "mean_lowercased_ppl_across_6seed": round(sum(vals) / len(vals), 2),
            "median_lowercased_ppl_across_6seed": round(vals[len(vals) // 2], 2),
            "min_ppl": round(vals[0], 2), "max_ppl": round(vals[-1], 2),
        })
    out_data["coverage_vs_confidence_buckets"] = bucket_stats

    # ── per-group cross-seed lowercased ppl (group membership is a cleaner explanatory variable than the
    # coarse whole_word_frac bucket for SHORT prompts, whose coverage score is heavily quantized by small
    # denominators -- e.g. a 2-content-word prompt can only score {0, 0.5, 1.0}) ────────────────────────────
    group_of_row = [gname for gname, rows in rows_by_group.items() for _ in rows]
    per_group_ppl = {}
    for gname in rows_by_group:
        vals = sorted(mean_lc_ppl_per_prompt[i] for i, g in enumerate(group_of_row) if g == gname)
        per_group_ppl[gname] = {
            "n": len(vals),
            "mean_lowercased_ppl_across_6seed": round(sum(vals) / len(vals), 2) if vals else None,
            "median_lowercased_ppl_across_6seed": round(vals[len(vals) // 2], 2) if vals else None,
        }
    out_data["per_group_lowercased_ppl"] = per_group_ppl

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(json.dumps({"coverage": out_data["coverage"]["overall"],
                          "teacher_forced_by_seed": out_data["teacher_forced_by_seed"],
                          "case_fold_impact": out_data["case_fold_impact"],
                          "coverage_vs_confidence_buckets": out_data["coverage_vs_confidence_buckets"]}, indent=2))
    return out_data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--n-wikidata", type=int, default=100)
    ap.add_argument("--ckpt-template", type=str, default=None)
    ap.add_argument("--bpe-path", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    main(seeds=tuple(args.seeds), n_wikidata=args.n_wikidata, out=args.out,
         ckpt_template=args.ckpt_template, bpe_path=args.bpe_path)
