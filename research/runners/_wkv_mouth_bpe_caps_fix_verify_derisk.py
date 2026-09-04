"""Verify the 2026-09-04 BPE-caps fix (webapp/wkv_mouth_generator.py) against the DOMINANT broad-scope coverage
blocker research/findings/2026-09-03-linattn-mouth-broad-scope-coverage-threshold.md's Result 2 measured: the
deployed `--recurrence linattn` BPE checkpoint's tokenizer was trained EXCLUSIVELY on lowercase text (the merge
table's own training regex plus the fact `data/corpus/simplewiki.txt` is pre-lowercased on disk), so a raw-case
chat prompt's every capital letter fell outside the trained alphabet and BPE-encoded to `<UNK>` -- measured at a
~5.6x teacher-forced-perplexity cost (12827.32 asfed -> 2283.64 lowercased, cited artifact
`research/findings/raw/_wkv_mouth_linattn_broad_scope_coverage.json`) independent of genuine topic coverage.

THE FIX under test (webapp/wkv_mouth_generator.py, 2026-09-04): two independent, independently-guarded, default-ON
pieces --
  INPUT  -- `_bpe_encode_prompt`/`bpe_lowercase_enabled()`: lowercases the prompt before BPE-encoding it, at the
            ONLY two `bpe.encode` call sites in `_free_gen`/`_free_gen_linattn`. `BRAIN_WKV_MOUTH_BPE_LOWERCASE=0`
            reverts to the pre-fix raw-case encode.
  OUTPUT -- `_truecase`/`truecase_enabled()`: a lightweight sentence-initial + pronoun-"I" + small known-name-
            allowlist heuristic applied to `generate()`'s final return text (both checkpoint families ship an
            entirely lowercase vocabulary and so cannot themselves emit a capital letter). `BRAIN_WKV_MOUTH_
            TRUECASE=0` reverts to the raw all-lowercase text.

THIS RUNNER answers three questions, each against the ACTUAL production code path (not a hand-copied
reimplementation):
  A. INPUT RECOVERY -- does `WKV._bpe_encode_prompt` (fix ON, the default) produce EXACTLY the same token ids as
     the cited finding's own `lowercased` counterfactual (`bpe.encode(text.lower())`) for every one of the same
     124 probe utterances, across all 6 non-negotiable seeds -- and does the resulting mean teacher-forced
     perplexity land at the SAME ~2283 the finding measured (recovering the ~5.6x from the ~12827 raw-case
     number), NOT a reimplementation's approximation of it?
  B. BYTE-IDENTICAL OFF -- with `BRAIN_WKV_MOUTH_BPE_LOWERCASE=0` / `BRAIN_WKV_MOUTH_TRUECASE=0`, is EVERY
     observable (token ids, teacher-forced perplexity, `generate()`'s returned text) EXACTLY what it was before
     this fix existed (raw-case ids == the finding's own `asfed` counterfactual; generated text carries zero
     uppercase letters, since neither checkpoint vocabulary contains one)?
  C. OUTPUT + MOAT -- does a REAL `generate()` call (genuine few-spike Izhikevich spiking read, not a stub) with
     both fixes ON produce readable, properly-capitalized text, and do `in_vocab_scope`/`fact_grounding_ids` (the
     scope gate + fact-grounding lever the caller `webapp/open_ended_chat.py` also uses) return IDENTICAL results
     regardless of the new flags -- i.e. this fix cannot silently perturb scope-routing or fact-grounding?

MEMORY / COST DISCIPLINE. Part A is pure `LinAttnReadout` matrix ops (np.load + numpy), same class as the cited
runner -- no SimulationBridge, no GPU, no torch. Part C builds a handful of TINY `FewSpikeWordRead` spiking banks
(`num_neurons = topk*pop = 64*8 = 512`, `connections_per_neuron=0`, no plasticity) for a SMALL prompt sample, not
the full 124-utterance probe -- CPU/numpy only, well under a 4 GB RSS budget (reported in the output artifact's
own `peak_rss_mb`).

CHECKOUT-LOCATION NOTE (same precedent as `_wkv_mouth_linattn_broad_scope_coverage_derisk._resolve_ckpt`): the
`wkv_linattn_depth2_contiguous_seed*.npz` files are gitignored multi-MB binaries that a git worktree does not
carry -- `_resolve_ckpt_template` below applies the identical local-checkout-first-then-shared-checkout fallback.

Run:
    CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy PYTHONPATH=. .venv/bin/python \\
        -m research.runners._wkv_mouth_bpe_caps_fix_verify_derisk \\
        --out research/findings/raw/_wkv_mouth_bpe_caps_fix_verify.json
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# See the module docstring's CHECKOUT-LOCATION NOTE.
_SHARED_CHECKOUT = Path("/home/dant123/Projects/sim")
_LINATTN_REL = "bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{seed}.npz"


def _resolve_ckpt_template() -> str:
    if (REPO_ROOT / _LINATTN_REL.format(seed=42)).exists():
        return str(REPO_ROOT / _LINATTN_REL)
    if (_SHARED_CHECKOUT / _LINATTN_REL.format(seed=42)).exists():
        return str(_SHARED_CHECKOUT / _LINATTN_REL)
    raise FileNotFoundError(
        f"linattn checkpoint not found in this checkout ({REPO_ROOT}) or the shared checkout "
        f"({_SHARED_CHECKOUT}) -- see the module docstring's CHECKOUT-LOCATION NOTE."
    )


# `wkv_mouth_generator._CKPT_TEMPLATE` is a MODULE-LEVEL constant read from `BRAIN_WKV_MOUTH_CKPT` once at import
# time -- MUST be set before the `from webapp import wkv_mouth_generator` import below, or `_get_readout` would
# default to the word-level `wkv_ssmU6_*` template instead of the linattn/BPE checkpoint this verification is
# actually about.
os.environ.setdefault("BRAIN_WKV_MOUTH_CKPT", _resolve_ckpt_template())

import numpy as np  # noqa: E402

from webapp import wkv_mouth_generator as WKV  # noqa: E402
from research.runners._wkv_fewspike_read_derisk import LinAttnReadout, FewSpikeWordRead  # noqa: E402
from research.runners._wkv_mouth_chat_topic_vocab_coverage_derisk import _build_probe_corpus  # noqa: E402
from research.runners._wkv_mouth_linattn_broad_scope_coverage_derisk import (  # noqa: E402
    _resolve_ckpt, _asfed_ids, _lowercased_ids, _teacher_forced_nll_linattn,
)

DEFAULT_SEEDS = (42, 43, 44, 100, 101, 102)
# Cited artifact this rung cross-checks against (research/findings/2026-09-03-linattn-mouth-broad-scope-
# coverage-threshold.md Result 2 / case_fold_impact).
CITED_ARTIFACT = "research/findings/raw/_wkv_mouth_linattn_broad_scope_coverage.json"
CITED_MEAN_ASFED_PPL = 12827.32
CITED_MEAN_LOWERCASED_PPL = 2283.64


def _peak_rss_mb() -> float:
    # ru_maxrss is KB on Linux, bytes on macOS -- this project runs Linux (env), KB assumed.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


# ── Part A + B: INPUT fix recovery + byte-identical-off, via the REAL production helper ──────────────────────
def _part_ab(seeds, n_wikidata: int) -> dict:
    bpe = WKV._get_bpe_tokenizer()
    groups = _build_probe_corpus(42, n_wikidata)
    texts = [it["text"] for items in groups.values() for it in items]
    n = len(texts)

    # A1: fix ON (default) ids MUST equal the finding's own lowercased counterfactual, for every utterance.
    assert WKV.bpe_lowercase_enabled(), "expected default ON at start of _part_ab"
    ids_fix_on = [WKV._bpe_encode_prompt(bpe, t) for t in texts]
    ids_lowercased_counterfactual = [_lowercased_ids(bpe, t) for t in texts]
    a1_id_match = sum(1 for a, b in zip(ids_fix_on, ids_lowercased_counterfactual) if a == b)

    # B1: fix OFF ids MUST equal the finding's own raw asfed (pre-fix) ids, for every utterance.
    os.environ["BRAIN_WKV_MOUTH_BPE_LOWERCASE"] = "0"
    assert not WKV.bpe_lowercase_enabled()
    ids_fix_off = [WKV._bpe_encode_prompt(bpe, t) for t in texts]
    ids_raw_asfed = [_asfed_ids(bpe, t) for t in texts]
    b1_id_match = sum(1 for a, b in zip(ids_fix_off, ids_raw_asfed) if a == b)
    del os.environ["BRAIN_WKV_MOUTH_BPE_LOWERCASE"]
    assert WKV.bpe_lowercase_enabled(), "flag must resume default ON after del"

    # A2/B2: teacher-forced perplexity across all 6 seeds, on `ids_fix_on` (A) and `ids_fix_off` (B) -- the SAME
    # cross-seed-mean-per-prompt-then-mean-over-probe methodology the cited finding used for `case_fold_impact`.
    per_prompt_fixon_ppl = [[] for _ in range(n)]
    per_prompt_fixoff_ppl = [[] for _ in range(n)]
    per_seed = {}
    for seed in seeds:
        ckpt_path = _resolve_ckpt(_LINATTN_REL.format(seed=seed))
        ro = LinAttnReadout(ckpt_path)
        fixon_ppls, fixoff_ppls = [], []
        for i in range(n):
            p_on = _teacher_forced_nll_linattn(ro, ids_fix_on[i])["ppl"]
            p_off = _teacher_forced_nll_linattn(ro, ids_fix_off[i])["ppl"]
            if p_on is not None:
                per_prompt_fixon_ppl[i].append(p_on)
                fixon_ppls.append(p_on)
            if p_off is not None:
                per_prompt_fixoff_ppl[i].append(p_off)
                fixoff_ppls.append(p_off)
        per_seed[str(seed)] = {
            "checkpoint_path": ckpt_path,
            "mean_fix_on_ppl": round(sum(fixon_ppls) / len(fixon_ppls), 2) if fixon_ppls else None,
            "mean_fix_off_ppl": round(sum(fixoff_ppls) / len(fixoff_ppls), 2) if fixoff_ppls else None,
        }

    mean_per_prompt_fixon = [sum(v) / len(v) for v in per_prompt_fixon_ppl if v]
    mean_per_prompt_fixoff = [sum(v) / len(v) for v in per_prompt_fixoff_ppl if v]
    mean_fixon = sum(mean_per_prompt_fixon) / len(mean_per_prompt_fixon)
    mean_fixoff = sum(mean_per_prompt_fixoff) / len(mean_per_prompt_fixoff)

    return {
        "n_probe_utterances": n,
        "seeds": list(seeds),
        "A_input_fix_on_ids_match_lowercased_counterfactual": {"n_match": a1_id_match, "n_total": n,
                                                                 "all_match": a1_id_match == n},
        "B_input_fix_off_ids_match_raw_asfed_pre_fix": {"n_match": b1_id_match, "n_total": n,
                                                          "all_match": b1_id_match == n},
        "per_seed_ppl": per_seed,
        "cross_seed_mean_ppl": {
            "fix_on__production_asfed_now": round(mean_fixon, 2),
            "fix_off__pre_fix_raw_asfed": round(mean_fixoff, 2),
            "recovery_ratio": round(mean_fixoff / mean_fixon, 3) if mean_fixon else None,
        },
        "cited_artifact_cross_check": {
            "artifact": CITED_ARTIFACT,
            "cited_mean_asfed_ppl_pre_fix": CITED_MEAN_ASFED_PPL,
            "cited_mean_lowercased_ppl": CITED_MEAN_LOWERCASED_PPL,
            "this_run_fix_off_vs_cited_asfed_close": abs(mean_fixoff - CITED_MEAN_ASFED_PPL) < 50.0,
            "this_run_fix_on_vs_cited_lowercased_close": abs(mean_fixon - CITED_MEAN_LOWERCASED_PPL) < 50.0,
            "note": "small residual vs the cited numbers is expected: that finding round-trips through JSON-"
                    "rounded per-seed/per-prompt intermediate ppl before its own cross-seed mean, this run "
                    "recomputes from scratch against the same checkpoints/probe -- an exact match is not implied.",
        },
    }


# ── Part C: real generate() calls (genuine spiking read) -- output truecasing + moat/fact-routing regression ──
_DEMO_PROMPTS = [
    "Hi there! How are you doing today?",
    "Tell me about Paris.",
    "What do you know about the United Kingdom?",
    "Tell me about music.",
]


def _has_uppercase(s: str) -> bool:
    return any(c.isupper() for c in s)


def _part_c(seed: int, max_new_tokens: int) -> dict:
    rows = []
    for prompt in _DEMO_PROMPTS:
        # both fixes ON (default)
        os.environ.pop("BRAIN_WKV_MOUTH_BPE_LOWERCASE", None)
        os.environ.pop("BRAIN_WKV_MOUTH_TRUECASE", None)
        os.environ["BRAIN_WKV_MOUTH_RECURRENCE"] = "linattn"
        os.environ["BRAIN_WKV_MOUTH_TOKENIZER"] = "bpe"
        text_on, secs_on = WKV.generate(prompt, seed=seed, max_new_tokens=max_new_tokens)

        # both fixes OFF -- must carry zero uppercase (neither checkpoint vocab has one, and truecase is skipped)
        os.environ["BRAIN_WKV_MOUTH_BPE_LOWERCASE"] = "0"
        os.environ["BRAIN_WKV_MOUTH_TRUECASE"] = "0"
        text_off, secs_off = WKV.generate(prompt, seed=seed, max_new_tokens=max_new_tokens)
        os.environ.pop("BRAIN_WKV_MOUTH_BPE_LOWERCASE", None)
        os.environ.pop("BRAIN_WKV_MOUTH_TRUECASE", None)

        rows.append({
            "prompt": prompt,
            "fix_on_text": text_on, "fix_on_has_uppercase": _has_uppercase(text_on),
            "fix_off_text": text_off, "fix_off_has_uppercase": _has_uppercase(text_off),
            "fix_off_nonempty": len(text_off.strip()) > 0,
        })
    os.environ.pop("BRAIN_WKV_MOUTH_RECURRENCE", None)
    os.environ.pop("BRAIN_WKV_MOUTH_TOKENIZER", None)

    n_on_capitalized = sum(1 for r in rows if r["fix_on_has_uppercase"])
    n_off_capitalized = sum(1 for r in rows if r["fix_off_has_uppercase"])

    # moat / fact-routing regression: in_vocab_scope + fact_grounding_ids must be IDENTICAL regardless of the
    # new flags -- neither function references BRAIN_WKV_MOUTH_BPE_LOWERCASE/BRAIN_WKV_MOUTH_TRUECASE at all, so
    # this is a direct empirical check, not an inference from reading the code.
    probe_msg = "Tell me about Paris and music."
    sample_facts = [("paris", "capital_of", "france"), ("music", "genre", "jazz")]
    moat_rows = []
    for bpe_lc, tc in ((None, None), ("0", "0"), (None, "0"), ("0", None)):
        if bpe_lc is None:
            os.environ.pop("BRAIN_WKV_MOUTH_BPE_LOWERCASE", None)
        else:
            os.environ["BRAIN_WKV_MOUTH_BPE_LOWERCASE"] = bpe_lc
        if tc is None:
            os.environ.pop("BRAIN_WKV_MOUTH_TRUECASE", None)
        else:
            os.environ["BRAIN_WKV_MOUTH_TRUECASE"] = tc
        scope = WKV.in_vocab_scope(probe_msg, seed=seed)
        fg_ids = WKV.fact_grounding_ids(sample_facts, seed=seed)
        moat_rows.append({"bpe_lowercase_flag": bpe_lc, "truecase_flag": tc,
                           "in_vocab_scope": scope, "fact_grounding_ids": fg_ids})
    os.environ.pop("BRAIN_WKV_MOUTH_BPE_LOWERCASE", None)
    os.environ.pop("BRAIN_WKV_MOUTH_TRUECASE", None)
    moat_identical = all(r["in_vocab_scope"] == moat_rows[0]["in_vocab_scope"]
                          and r["fact_grounding_ids"] == moat_rows[0]["fact_grounding_ids"] for r in moat_rows)

    return {
        "seed": seed, "max_new_tokens": max_new_tokens,
        "rows": rows,
        "n_prompts": len(rows),
        "n_fix_on_produced_a_capital": n_on_capitalized,
        "n_fix_off_produced_a_capital": n_off_capitalized,  # MUST be 0 -- structural, see module docstring
        "moat_fact_routing_regression": {"rows": moat_rows, "identical_regardless_of_new_flags": moat_identical},
    }


def main(seeds=DEFAULT_SEEDS, n_wikidata: int = 100, demo_seed: int = 42, max_new_tokens: int = 20,
         out: str | None = None) -> dict:
    out_data = {
        "runner": "_wkv_mouth_bpe_caps_fix_verify_derisk",
        "fix_module": "webapp/wkv_mouth_generator.py",
        "part_A_B_input_recovery_and_byte_identical_off": _part_ab(seeds, n_wikidata),
        "part_C_output_truecase_and_moat_regression": _part_c(demo_seed, max_new_tokens),
        "peak_rss_mb": _peak_rss_mb(),
    }

    ab = out_data["part_A_B_input_recovery_and_byte_identical_off"]
    c = out_data["part_C_output_truecase_and_moat_regression"]
    out_data["verdict"] = {
        "input_fix_recovers_lowercased_ids_on_every_probe_utterance":
            ab["A_input_fix_on_ids_match_lowercased_counterfactual"]["all_match"],
        "input_fix_off_is_byte_identical_to_pre_fix_on_every_probe_utterance":
            ab["B_input_fix_off_ids_match_raw_asfed_pre_fix"]["all_match"],
        "input_fix_perplexity_recovery_ratio": ab["cross_seed_mean_ppl"]["recovery_ratio"],
        "output_fix_on_produced_capitals": c["n_fix_on_produced_a_capital"] > 0,
        "output_fix_off_produced_zero_capitals_structural_check": c["n_fix_off_produced_a_capital"] == 0,
        "moat_and_fact_routing_unaffected_by_new_flags": c["moat_fact_routing_regression"]["identical_regardless_of_new_flags"],
    }
    out_data["verdict"]["GO"] = all([
        out_data["verdict"]["input_fix_recovers_lowercased_ids_on_every_probe_utterance"],
        out_data["verdict"]["input_fix_off_is_byte_identical_to_pre_fix_on_every_probe_utterance"],
        out_data["verdict"]["output_fix_off_produced_zero_capitals_structural_check"],
        out_data["verdict"]["moat_and_fact_routing_unaffected_by_new_flags"],
    ])

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    print(json.dumps(out_data["verdict"], indent=2))
    return out_data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--n-wikidata", type=int, default=100)
    ap.add_argument("--demo-seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    main(seeds=tuple(args.seeds), n_wikidata=args.n_wikidata, demo_seed=args.demo_seed,
         max_new_tokens=args.max_new_tokens, out=args.out)
